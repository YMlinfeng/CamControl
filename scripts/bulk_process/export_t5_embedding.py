import csv
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass

import deepspeed
import pandas as pd
import torch
from rich import print
from stable_diffusion_application.kaimm.accelerate_utils import prepare
from stable_diffusion_application.kaimm.state import PartialState
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.data.base_data import DataConfig
from src.models.autoencoders.visual_tokenizer import VisualTokenizerConfig
from src.pipelines import T2VPixArtAlphaPipelineConfig


def mask_embeds(embeds, mask):
    """
    根据mask裁剪embeds。

    参数:
    - embeds: Tensor, 形状为 (b, l, d)，表示嵌入向量。
    - mask: Tensor, 形状为 (b, l)，其中1表示保留对应位置的embeds，0表示不保留。

    返回:
    - masked_embeds_list: list of Tensor, 长度为b，每个元素的形状为 (l', d)，
                           其中l'为mask之后的长度。
    """
    # 确保mask是布尔类型
    mask = mask.bool()
    masked_embeds_list = [embeds[i][mask[i]] for i in range(embeds.size(0))]

    return masked_embeds_list


def unmask_embeds(original_shape, masked_embeds_list, mask):
    """
    将裁剪后的embeds恢复到原始形状。

    参数:
    - original_shape: tuple, 原始embeds的形状，即(b, l, d)。
    - masked_embeds_list: list of Tensor, 裁剪后的embeds列表，每个元素的形状为(l', d)。
    - mask: Tensor, 原始的掩码，形状为(b, l)，其中1表示保留的位置。

    返回:
    - restored_embeds: Tensor, 恢复后的embeds，形状为(b, l, d)。
    """
    b, l, d = original_shape
    restored_embeds = torch.zeros(b, l, d)

    for i, (masked_embed, mask_row) in enumerate(zip(masked_embeds_list, mask)):
        # 获取当前行应插入数据的索引
        indices = mask_row.nonzero(as_tuple=True)[0]
        if len(indices) > 0:
            restored_embeds[i].index_put_((indices,), masked_embed)

    return restored_embeds


@dataclass
class T5ExportConfig:
    csv_path: str = "/ytech_m2v_hdd/m2v_data/m2v-image-s2-v0.1.csv"
    ckpt_path: str = "/group/ckpt/diffusers/PixArt-XL-2-512x512"
    save_dir: str = "/ytech_m2v_hdd/t5_embeddings_image/m2v-image-s2-v0.1"
    output_csv: str = '/ytech_m2v_hdd/zhengmingwu/m2v-image-s2-v0.1_t5_2.csv'
    tmp_csv_dir: str = "/video/zhengmingwu/tmp"

    caption_column: str = "cogvlm_caption"
    index_column: str = "video_ceph_path"

    remove_temp: bool = True
    batch_size: int = 128
    num_processes: int = 8
    max_sequence_length: int = 500



def main(config: T5ExportConfig):

    torch.set_grad_enabled(False)
    deepspeed.init_distributed()
    state = PartialState() # set device

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
        path=config.csv_path,
        index_column=config.index_column,
        caption_column=config.caption_column,
        video_path_column=None,
        image_path_column=None,
        batch_size=config.batch_size,
        num_processes=config.num_processes,
    )
    data = data_config.setup()
    pipeline.text_encoder.cuda()
    pipeline.text_encoder.to(torch.bfloat16)
    pipeline.text_encoder, data.dataloader = prepare(
        pipeline.text_encoder, data.dataloader
    )
    with open(os.path.join(csv_dir, f"rank{rank}.csv"), 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([data_config.index_column, data_config.caption_column, "t5_embed_path"])
        for batch in tqdm(data.dataloader):
            prompts = batch["prompts"]
            data_paths = batch["index"]
            prompt_embeds, prompt_attention_mask, _, _ = pipeline.encode_prompt(
                prompts,
                do_classifier_free_guidance=False,
                device=pipeline.text_encoder.device,
                clean_caption=True,
                max_length=config.max_sequence_length,
            )
            embedding_list = mask_embeds(prompt_embeds, prompt_attention_mask)

            for data_path, prompt, embeds in zip(data_paths, prompts, embedding_list):

                # img_dir = os.path.dirname(img_path).replace('/ytech_m2v_hdd/vcg_raw_image_info_v3/', '')
                data_dir = os.path.dirname(data_path)[1:]

                file_name = os.path.basename(data_path).split('.')[0]
                target_dir = os.path.join(save_dir, data_dir)
                os.makedirs(target_dir, exist_ok=True)
                save_path = os.path.join(target_dir, f'{file_name}.pt')

                torch.save(embeds.cpu(), save_path)
                writer.writerow([data_path, prompt, save_path])

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

    config = T5ExportConfig(
        remove_temp=False,
        caption_column='caption',
        index_column='video_ceph_path',
        csv_path="/ytech_m2v_hdd/m2v_data/m2v-video-s2-v0.3.csv",
        save_dir="/ytech_m2v_hdd/t5_embeddings_video/m2v-video-s2-v0.3",
        tmp_csv_dir="/video/zhengmingwu/tmp_video",
        output_csv="/ytech_m2v_hdd/zhengmingwu/m2v-video-s2-v0.3_t5.csv",
        batch_size=128,
    )
    main(config)
