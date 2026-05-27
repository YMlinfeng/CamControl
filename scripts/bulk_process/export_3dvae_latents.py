import csv
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional, Literal
from concurrent.futures import ThreadPoolExecutor

from rich import print
import pandas as pd
from tqdm import tqdm
import tempfile

import torch
import deepspeed
from einops import rearrange

from stable_diffusion_application.kaimm.accelerate_utils import prepare
from stable_diffusion_application.kaimm.state import PartialState


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.base_data import DataConfig
from src.models.autoencoders.visual_tokenizer import VisualTokenizerConfig
from src.pipelines import T2VPixArtAlphaPipelineConfig
from src.utils import measure_time, log_to_rank0

def save_async(para, save_path):
    torch.save(para.cpu(), save_path)

@dataclass
class VAEExportConfig:
    csv_path: str = "/ytech_m2v_hdd/m2v_data/m2v-image-s2-v0.1_t5_2.csv"
    ckpt_path: str = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.2.ckpt"
    save_dir: str = "/ytech_m2v_hdd/vae_latents_image/m2v-image-s2-v0.1"
    output_csv: str = '/ytech_m2v_hdd/zhengmingwu/m2v-image-s2-v0.1_t5_vae256x256.csv'
    tmp_csv_dir: str = "/video/zhengmingwu/tmp_vae"

    caption_column: str = "cogvlm_caption"
    image_column: Optional[str] = None
    video_column: Optional[str] = None
    index_column: str = "t5_embed_path"
    height: int = 256
    width: int = 256
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"

    remove_temp: bool = True
    batch_size: int = 128
    num_processes: int = 8



def main(config: VAEExportConfig):

    torch.set_grad_enabled(False)
    deepspeed.init_distributed()
    state = PartialState() # set device
    executor = ThreadPoolExecutor(72)

    if config.dtype == "bf16":
        dtype = torch.bfloat16
    elif config.dtype == "fp32":
        dtype = torch.float32
    elif config.dtype == "fp16":
        dtype = torch.float16 

    rank = torch.distributed.get_rank()
    world_size = torch.distributed.get_world_size()

    csv_dir = config.tmp_csv_dir
    save_dir = os.path.join(config.save_dir, os.path.basename(config.csv_path).split(".csv")[0])
    # if rank == 0:
    #     if os.path.exists(csv_dir):
    #         shutil.rmtree(csv_dir)
    #     if os.path.exists(save_dir):
    #         shutil.rmtree(save_dir)
    if rank == 0:
        os.makedirs(csv_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)

    torch.cuda.synchronize()
    torch.distributed.barrier()

    vaeConfig = VisualTokenizerConfig(output_conv_kernel_size = (3, 3, 3), vae_ckpt_path=config.ckpt_path)
    vae = vaeConfig.from_pretrained()
    data_config = DataConfig(
        height=config.height,
        width=config.width,
        path=config.csv_path,
        index_column=config.index_column,
        caption_column=config.caption_column,
        video_path_column=config.video_column,
        image_path_column=config.image_column,
        batch_size=config.batch_size,
        num_processes=config.num_processes,
        num_frames=1,
        measure_time=True,
        num_samples=8*384*2*2,
    )
    data_column = data_config.video_path_column if data_config.video_path_column is not None else data_config.image_path_column
    data = data_config.setup()
    vae = vae.cuda()
    vae = vae.to(dtype)
    vae, data.dataloader = prepare(
        vae, data.dataloader
    )
    vae.disable_slicing()
    log_to_rank0('Start inference...')
    with open(os.path.join(csv_dir, f"rank{rank}.csv"), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([data_column, data_config.caption_column, config.index_column, '3dvae_latents_path'])
        for batch in tqdm(data.dataloader):
            with measure_time('Inferece', run=True, synchronize=True, rank0_only=True):
                for param in vae.parameters():
                    param.data = param.data.clone().contiguous()

                samples = batch["data"]
                prompts = batch["prompts"]
                data_paths = batch["data_paths"]
                t5_paths = batch["index"]
                # batch_size, height, width = samples.shape[0], samples.shape[-2], samples.shape[-1]

                samples = samples.to(dtype).contiguous()
                samples = rearrange(samples, "b 1 c h w -> b c 1 h w")
                parameters = vae.encode(samples).latent_dist.parameters
                parameters  = rearrange(parameters, "b c 1 h w -> b c h w")

            with measure_time('Save and write', run=True, synchronize=False, rank0_only=True):
                for data_path, prompt, t5_path, para in zip(data_paths, prompts, t5_paths, parameters):

                    # img_dir = os.path.dirname(img_path).replace('/ytech_m2v_hdd/vcg_raw_image_info_v3/', '')
                    data_dir = os.path.dirname(data_path)[1:]
                    file_name = os.path.basename(data_path).split('.')[0]
                    target_dir = os.path.join(save_dir, data_dir)
                    os.makedirs(target_dir, exist_ok=True)
                    save_path = os.path.join(target_dir, f'{file_name}.pt')
                    with measure_time('save', run=False, synchronize=False, rank0_only=True):
                        executor.submit(save_async, para, save_path)
                        # torch.save(para.cpu(), save_path)
                    with measure_time('write', run=False, synchronize=False, rank0_only=True):
                        writer.writerow([data_path, prompt, t5_path, save_path])

            torch.cuda.empty_cache()

    torch.cuda.synchronize()
    torch.distributed.barrier()

    if rank == 0:
        all_files = []
        for i in range(world_size):
            all_files.append(os.path.join(csv_dir, f"rank{i}.csv"))
        dfs = [pd.read_csv(file) for file in all_files]
        df = pd.concat(dfs, ignore_index=True)
        unique_df = df.drop_duplicates()
        unique_df.to_csv(config.output_csv, index=False)
        print(f"Save to {config.output_csv}")
        if config.remove_temp:
            shutil.rmtree(csv_dir)


if __name__ == "__main__":

    os.environ["http_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["no_proxy"] = "localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
    os.environ["TORCH_HOME"] = "/group/ckpt/torchhub"
    os.environ["HF_DATASETS_CACHE"] = "/video/cache/huggingface"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    config = VAEExportConfig(
        remove_temp=False,
        csv_path="/ytech_m2v_hdd/zhengmingwu/m2v-image-s2-v0.1_t5_2.csv",
        ckpt_path="/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.2.ckpt",

        # save_dir="/ytech_m2v_hdd/3dvae_latents_image_256x256/",
        # output_csv="/ytech_m2v_hdd/zhengmingwu/m2v-image-s2-v0.1_t5_3dvae256.csv",
        # tmp_csv_dir="/video/zhengmingwu/tmp_3dvae_256",

        save_dir="/video/zhengmingwu/m2v-diffusers/outputs/test_3dvae_precompute",
        output_csv="/video/zhengmingwu/m2v-diffusers/outputs/test_3dvae_precompute/out.csv",
        tmp_csv_dir="/video/zhengmingwu/tmp_3dvae_256_temp",


        batch_size=384,
        width=256,
        height=256,
        image_column='img_path',
        caption_column='cogvlm_caption',
        index_column='t5_embed_path',
        dtype='bf16',
        num_processes=32,
    )
    main(config)
    
