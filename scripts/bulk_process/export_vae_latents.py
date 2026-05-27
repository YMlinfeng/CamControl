import csv
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional

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


@dataclass
class VAEExportConfig:
    csv_path: str = "/ytech_m2v_hdd/m2v_data/m2v-image-s2-v0.1_t5_2.csv"
    ckpt_path: str = "/group/ckpt/diffusers/PixArt-XL-2-512x512"
    save_dir: str = "/ytech_m2v_hdd/vae_latents_image/m2v-image-s2-v0.1"
    output_csv: str = '/ytech_m2v_hdd/zhengmingwu/m2v-image-s2-v0.1_t5_vae256x256.csv'
    tmp_csv_dir: str = "/video/zhengmingwu/tmp_vae"

    caption_column: str = "cogvlm_caption"
    image_column: Optional[str] = None
    video_column: Optional[str] = None
    index_column: str = "t5_embed_path"
    height: int = 256
    width: int = 256
    dtype: str = "bf16"

    remove_temp: bool = True
    batch_size: int = 128
    num_processes: int = 8



def main(config: VAEExportConfig):

    torch.set_grad_enabled(False)
    deepspeed.init_distributed()
    state = PartialState() # set device

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

    pipeline = T2VPixArtAlphaPipelineConfig(ckpt_path=config.ckpt_path).from_pretrained(low_cpu_mem_usage=False, device_map=None)
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
    )
    data_column = data_config.video_path_column if data_config.video_path_column is not None else data_config.image_path_column
    data = data_config.setup()
    pipeline.vae.cuda()
    pipeline.vae.to(dtype)
    pipeline.vae, data.dataloader = prepare(
        pipeline.vae, data.dataloader
    )
    pipeline.vae.disable_slicing()
    with open(os.path.join(csv_dir, f"rank{rank}.csv"), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([data_column, data_config.caption_column, config.index_column, 'vae_latents_path'])
        for batch in tqdm(data.dataloader):
            samples = batch["data"]
            prompts = batch["prompts"]
            data_paths = batch["data_paths"]
            t5_paths = batch["index"]
            # batch_size, height, width = samples.shape[0], samples.shape[-2], samples.shape[-1]

            samples = samples.to(dtype)
            samples = rearrange(samples, "b 1 c h w -> b c h w")
            parameters = pipeline.vae.encode(samples).latent_dist.parameters

            for data_path, prompt, t5_path, para in zip(data_paths, prompts, t5_paths, parameters):

                # img_dir = os.path.dirname(img_path).replace('/ytech_m2v_hdd/vcg_raw_image_info_v3/', '')
                data_dir = os.path.dirname(data_path)[1:]
                file_name = os.path.basename(data_path).split('.')[0]
                target_dir = os.path.join(save_dir, data_dir)
                os.makedirs(target_dir, exist_ok=True)
                save_path = os.path.join(target_dir, f'{file_name}.pt')
                torch.save(para.cpu(), save_path)
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
        ckpt_path="/group/ckpt/diffusers/PixArt-XL-2-512x512",
        # save_dir="/video/zhengmingwu/m2v-diffusers/outputs/test_vae_precompute",
        save_dir="/ytech_m2v_hdd/vae_latents_image_512x512/",
        output_csv="/ytech_m2v_hdd/zhengmingwu/m2v-image-s2-v0.1_t5_vae512.csv",
        tmp_csv_dir="/video/zhengmingwu/tmp_vae_512",
        batch_size=64,
        width=512,
        height=512,
        image_column='img_path',
        caption_column='cogvlm_caption',
        index_column='t5_embed_path',
    )
    main(config)
    
