import csv
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Optional, Literal
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from glob import glob

from rich import print
import pandas as pd
from tqdm import tqdm
import tempfile
import numpy as np

import torch
import deepspeed
from datasets import Dataset
from einops import rearrange

import decord

from stable_diffusion_application.kaimm.accelerate_utils import prepare
from stable_diffusion_application.kaimm.state import PartialState


sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from src.data.base_data import DataConfig
from src.models.autoencoders.visual_tokenizer import VisualTokenizerConfig
from src.pipelines import T2VPixArtAlphaPipelineConfig
from src.utils import measure_time, log_to_rank0, Timer, eval_setup

from src.configs.config_utils import save_config

@dataclass
class VAEExportConfig:
    csv_path: str = "/video/zhengmingwu/m2v-diffusers/inputs/video2latents_240.csv"
    ckpt_path: str = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.2.ckpt"
    save_dir: str = "/video/zhengmingwu/m2v-diffusers/outputs/test_video2latents"
    output_csv: str = '/video/zhengmingwu/m2v-diffusers/outputs/test_video2latents/test_video2latents.csv'
    tmp_csv_dir: str = "/video/zhengmingwu/tmp_vae"

    caption_column: Optional[str] = None
    image_column: Optional[str] = None
    video_column: Optional[str] = None
    index_column: str = "video_path"
    height: int = 256
    width: int = 256
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    mode: Literal["video", "image"] = "video"

    remove_temp: bool = False
    batch_size: int = 1
    height: int = 256
    width: int = 256
    num_processes: int = 8
    segment_size: int = 81
    chunk_size: int = 3  
    max_duration: int = 30
    '''seconds'''
    gpu_decode: bool = False
    measure_time: bool = False
    resume_path: Optional[str] = None
    trans_dir: Optional[str] = None
    frame_chunk_size: int = 1


class Precomputer:
    def __init__(self, config) -> None:
        self.config = config
        if config.dtype == "bf16":
            dtype = torch.bfloat16
        elif config.dtype == "fp32":
            dtype = torch.float32
        elif config.dtype == "fp16":
            dtype = torch.float16 

        self.timer = Timer()
        self.dtype = dtype
        self.rank = torch.distributed.get_rank()
        self.world_size = torch.distributed.get_world_size()

        self.csv_dir = config.tmp_csv_dir
        self.save_dir = os.path.join(config.save_dir, os.path.basename(config.csv_path).split(".csv")[0])
        self.tmp_csv_path = os.path.join(self.csv_dir, f"rank{self.rank}.csv")


        if self.rank == 0:
            shutil.rmtree(self.csv_dir, ignore_errors=True)
            # shutil.rmtree(self.save_dir, ignore_errors=True)
            os.makedirs(self.csv_dir, exist_ok=True)
            os.makedirs(self.save_dir, exist_ok=True)

        vaeConfig = VisualTokenizerConfig(output_conv_kernel_size = (3, 3, 3), vae_ckpt_path=config.ckpt_path)
        vae = vaeConfig.from_pretrained()
        if config.mode == "video":
            self.data_config = DataConfig(
                path=config.csv_path,
                index_column=config.index_column,
                caption_column=None,
                video_path_column=None,
                image_path_column=None,
                batch_size=config.batch_size,
                num_processes=8,
                num_frames=1,
                measure_time=False,
                sample_fps=15,
                height=config.height,
                width=config.width,
                crop_type='center',
                random_flip=False,
            )
        elif config.mode == "image":
            self.data_config = DataConfig(
                path=config.csv_path,
                index_column=None,
                caption_column=config.caption_column,
                video_path_column=None,
                image_path_column=config.index_column,
                batch_size=config.batch_size,
                num_processes=0,
                num_frames=1,
                measure_time=False,
                height=config.height,
                width=config.width,
                crop_type='center',
                random_flip=False,
            )
        self.data = self.data_config.setup()
        vae = torch.compile(vae)
        vae = vae.cuda()
        vae = vae.to(dtype)
        vae.disable_slicing()
        self.vae, self.data.dataloader = prepare(
            vae, self.data.dataloader
        )

    def process_batch(self, batch, writer):
        if self.config.mode == "video":
            self.process_video(batch, writer)
        elif self.config.mode == "image":
            self.process_image(batch, writer)

    def read_and_process_videos(self, batch):
        decord.bridge.set_bridge("torch")
        config = self.config
        segment_size = config.segment_size
        with measure_time('Initial decord', run=config.measure_time, synchronize=False, rank0_only=True, timer=self.timer):
            data_paths = batch["index"]
            assert len(data_paths) == 1
            video_path = data_paths[0]
            local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
            ctx = decord.gpu(local_rank) if config.gpu_decode else decord.cpu(0)
            reader = decord.VideoReader(video_path, ctx=ctx)
            length = len(reader)
            fps = reader.get_avg_fps()

            if length < segment_size:
                return None
            assert segment_size != 1
            frame_stride_full = max(1, (length - 1) // (segment_size - 1))
            frame_stride = min(max(1, int(fps // self.data_config.sample_fps)), frame_stride_full) # Ensure that the frame_stride is not too large
            frame_indexes = range(0, length, frame_stride)
            frame_indexes = [min(frame_index, length - 1) for frame_index in frame_indexes]
            frame_indexes = frame_indexes[:(segment_size * (len(frame_indexes) // segment_size))]
        
        frame_chunk_size = config.frame_chunk_size
        frame_tensors_all = []
        for i in range(0, len(frame_indexes), frame_chunk_size):
            frame_indexes_chunk = frame_indexes[i:i+frame_chunk_size]
            with measure_time('Video read', run=config.measure_time, synchronize=False, rank0_only=True, timer=self.timer):
                try:
                    all_frames = reader.get_batch(frame_indexes_chunk) # F H W C
                except Exception as e:
                    print(f"Error in {video_path}: {e}. Trying to read with CPU...")
                    try:
                        cpu_reader = decord.VideoReader(video_path, ctx=decord.cpu(0))
                        all_frames = cpu_reader.get_batch(frame_indexes_chunk).cuda()
                    except Exception as e:
                        print(f"Error in {video_path}: {e}. Skip this video.")
                        return None
                
            with measure_time('Image Process', run=config.measure_time, synchronize=False, rank0_only=True, timer=self.timer):
                frame_tensors, original_size, crop_top_left = self.data.preprocess_image(all_frames) # F C H W

            del all_frames
            frame_tensors_all.append(frame_tensors)

        frame_tensors = torch.cat(frame_tensors_all, dim=0)
        assert frame_tensors.shape[0] % segment_size == 0
        frame_tensors = rearrange(frame_tensors, "(b s) c h w -> b s c h w", s=segment_size)
        if not config.gpu_decode:
            frame_tensors = frame_tensors.cuda()
        del frame_tensors_all
        return frame_tensors, original_size, crop_top_left, fps, frame_stride, video_path

    def inference_video(self, frame_tensors):
        with measure_time('Inference', run=config.measure_time, synchronize=True, rank0_only=True, timer=self.timer):
            for param in self.vae.parameters():
                param.data = param.data.clone().contiguous()
            output = []
            chunk_size = config.chunk_size
            for i in range(0, frame_tensors.shape[0], chunk_size):
                chunk = frame_tensors[i:i+chunk_size]
                chunk = rearrange(chunk, "b f c h w -> b c f h w")
                latents = self.vae.encode(chunk).latent_dist
                parameters = latents.parameters # B C F H W
                output.append(parameters)
            output = torch.cat(output, dim=0).cpu() # N C F H W
        del frame_tensors, latents, parameters
        return output
    
    def save_and_write_video(self, output, original_size, crop_top_left, fps, frame_stride, video_path, writer):
        with measure_time('Save and write', run=config.measure_time, synchronize=False, rank0_only=True, timer=self.timer):
            file_name = os.path.basename(video_path).split('.')[0]
            target_dir = os.path.join(self.save_dir, os.path.dirname(video_path)[1:])
            os.makedirs(target_dir, exist_ok=True)
            save_path = os.path.join(target_dir, f'{file_name}.pt')

            final_dict = {
                "data": output, # N C F H W
                "original_size": original_size,
                "crop_top_left": crop_top_left,
                "fps": fps,
                "frame_stride": frame_stride,
            }

            torch.save(final_dict, save_path)
            writer.writerow([video_path, save_path])

        del output
        torch.cuda.empty_cache()

    def process_video(self, batch, writer):
        packs = self.read_and_process_videos(batch)
        if packs is None:
            print(f"Skip {batch['index'][0]}")
            return
        
        frame_tensors, original_size, crop_top_left, fps, frame_stride, video_path = packs
        output = self.inference_video(frame_tensors)
        self.save_and_write_video(output, original_size, crop_top_left, fps, frame_stride, video_path, writer)


    def process_image(self, batch, writer):
        config = self.config
        with measure_time('Inference', run=config.measure_time, synchronize=False, rank0_only=True, timer=self.timer):
            data_paths = batch["data_paths"]
            samples = batch["data"]
            samples = samples.to(self.dtype)
            samples = rearrange(samples, "b 1 c h w -> b c 1 h w")
            parameters = self.vae.encode(samples).latent_dist.parameters
        with measure_time('Save and write', run=config.measure_time, synchronize=False, rank0_only=True, timer=self.timer):
            for data_path, para in zip(data_paths, parameters):
                data_dir = os.path.dirname(data_path)[1:]
                file_name = os.path.basename(data_path).split('.')[0]
                target_dir = os.path.join(self.save_dir, data_dir)
                os.makedirs(target_dir, exist_ok=True)
                save_path = os.path.join(target_dir, f'{file_name}.pt')
                torch.save(para.cpu(), save_path)
                writer.writerow([data_path, save_path])
        del samples, parameters
        torch.cuda.empty_cache()

    def merge_result(self):
        if self.rank != 0:
            return
        all_files = []
        for i in range(self.world_size):
            all_files.append(os.path.join(self.csv_dir, f"rank{i}.csv"))
        for i in range(self.world_size):
            all_files.append(os.path.join(self.csv_dir, f"rank{i}.csv"))
        dfs = [pd.read_csv(file) for file in all_files]
        df = pd.concat(dfs, ignore_index=True)
        unique_df = df.drop_duplicates()
        unique_df.to_csv(config.output_csv, index=False)
        print(f"Save to {config.output_csv}")
        if self.config.remove_temp:
            shutil.rmtree(self.csv_dir)

def main(config: VAEExportConfig):

    torch.set_grad_enabled(False)
    deepspeed.init_distributed()
    state = PartialState() # set device

    if torch.distributed.get_rank() == 0:
        process_csvs(config)
        configs = [config for _ in range(torch.distributed.get_world_size())]
    else:
        configs = [None for _ in range(torch.distributed.get_world_size())]
    torch.distributed.broadcast_object_list(configs, src=0)
    config = configs[0]

    torch.distributed.barrier()
    precomputer = Precomputer(config)
    torch.cuda.synchronize()
    torch.distributed.barrier()

    log_to_rank0('Start inference...')
    with open(precomputer.tmp_csv_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([config.index_column, '3dvae_latents_path'])
        
        for batch in tqdm(precomputer.data.dataloader, desc=f"{config.csv_path}"):
            try:
                precomputer.process_batch(batch, writer)
            except Exception as e:
                print(f"Error in batch: {e}")
                del batch
            
    torch.cuda.synchronize()
    torch.distributed.barrier()
    print(precomputer.timer)
    precomputer.merge_result()


def process_csvs(current_config):
    trans_dir = current_config.trans_dir
    os.makedirs(trans_dir, exist_ok=True)
    resume_path = current_config.resume_path
    if resume_path is None:
        csv_name = os.path.basename(current_config.csv_path).split(".csv")[0]
        current_config.target = csv_name
        if os.path.exists(os.path.join(trans_dir, csv_name)):
            shutil.rmtree(os.path.join(trans_dir, csv_name))
        os.makedirs(os.path.join(trans_dir, csv_name), exist_ok=False)
        save_config(current_config, os.path.join(trans_dir, csv_name, csv_name + '.yaml'))
        return
    print('Begin to process resume path:', resume_path)
    resume_config = eval_setup(resume_path)
    target_name = resume_config.target
    target_dir = os.path.join(trans_dir, target_name)
    assert os.path.exists(target_dir), f"{target_dir} does not exist."

    orginal_csv = resume_config.csv_path
    tmp_csv_dir = resume_config.tmp_csv_dir
    csvs = glob(os.path.join(tmp_csv_dir, "*.csv"))

    dfs = []
    for csv in csvs:
        try:
            dfs.append(pd.read_csv(csv).iloc[:-1])
        except Exception as e:
            print(f"Error in {csv}: {e}")

    assert len(dfs) > 0, "No csv files found."
    done_df = pd.concat(dfs, ignore_index=True)
    done_paths = set(done_df[resume_config.index_column])
    
    timestamp = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")

    done_dir = os.path.join(target_dir, 'done')
    os.makedirs(done_dir, exist_ok=True)
    done_csv = os.path.join(done_dir, timestamp + '.csv')
    print(f"Done num: {len(done_paths)}, going to save to {done_csv}")
    done_df.to_csv(done_csv, index=False)

    todo_dir = os.path.join(target_dir, 'todo')
    os.makedirs(todo_dir, exist_ok=True)
    orginal_df = pd.read_csv(orginal_csv)
    print('Total num:', len(orginal_df))
    todo_df = orginal_df[~orginal_df[resume_config.index_column].isin(done_paths)]
    todo_csv = os.path.join(todo_dir, timestamp + '.csv')
    print(f"Todo num: {len(todo_df)}, going to save to {todo_csv}")
    todo_df.to_csv(todo_csv, index=False)

    current_config.csv_path = todo_csv
    current_config.output_csv = os.path.join(done_dir, target_name + '.csv')
    current_config.target= target_name
    print(f"Save to {os.path.join(target_dir, timestamp + '.yaml')}")
    save_config(current_config, os.path.join(target_dir, timestamp + '.yaml'))
    quit()
    # reference_csv = '/ytech_m2v_hdd/zhengmingwu/m2v-video-s2-v0.3_t5.csv'  # 这个CSV文件包含已出现过的img_path值
    # reference_df = pd.read_csv(reference_csv)
    # print(len(reference_df))
    # existing_img_paths = set(reference_df['video_ceph_path'])
    # print(len(existing_img_paths))

if __name__ == "__main__":

    os.environ["http_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["https_proxy"] = "http://oversea-squid1.jp.txyun:11080"
    os.environ["no_proxy"] = "localhost,127.0.0.1,localaddress,localdomain.com,internal,corp.kuaishou.com,test.gifshow.com,staging.kuaishou.com"
    os.environ["TORCH_HOME"] = "/group/ckpt/torchhub"
    os.environ["HF_DATASETS_CACHE"] = "/video/cache/huggingface"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    config = VAEExportConfig()

    video_config = VAEExportConfig(
        csv_path = "/video/zhengmingwu/m2v-diffusers/outputs/m2v-video-s1-v0.1_t5_clip_3dvae_0411_s2000.csv",
        ckpt_path = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.3.ckpt",
        save_dir = "/video/zhengmingwu/m2v-diffusers/outputs/test_video2latents",
        output_csv = '/video/zhengmingwu/m2v-diffusers/outputs/test_video2latents/test_video2latents.csv',
        tmp_csv_dir = "/video/zhengmingwu/tmp_vae",
        batch_size=1,
        mode="video",
        trans_dir='/video/zhengmingwu/m2v-diffusers/outputs/test_video2latents',
        gpu_decode=True,
        segment_size=81,
        chunk_size=3,
        frame_chunk_size=81,
        dtype='bf16',
        measure_time=True,
    )

    video_config_h800 = VAEExportConfig(
        csv_path = "/video/zhengmingwu/m2v-diffusers/middle/m2v-video-s1-v0.1-h800.csv",
        ckpt_path = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.3.ckpt",
        save_dir = "/video/zhengmingwu/latents_3dvae_h800",
        output_csv = '/video/zhengmingwu/latents_3dvae_h800/m2v-video-s1-v0.1-h800-3dvae.csv',
        tmp_csv_dir = "/video/zhengmingwu/tmp_vae_h800",
        index_column="video_ceph_path",
        batch_size=1,
        mode="video",
        gpu_decode=True,
        chunk_size=16,
        segment_size=81,
        trans_dir='/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_h800',
        measure_time=False,
        frame_chunk_size=81,
    )

    video_config_4090_1 = VAEExportConfig(
        csv_path = "/video/zhengmingwu/m2v-diffusers/middle/m2v-video-s1-v0.1-4090-part1.csv",
        ckpt_path = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.3.ckpt",
        save_dir = "/video/zhengmingwu/latents_3dvae_4090",
        output_csv = '/video/zhengmingwu/latents_3dvae_4090/m2v-video-s1-v0.1-4090-3dvae-part1.csv',
        tmp_csv_dir = "/video/zhengmingwu/tmp_vae_4090_part1",
        index_column="video_ceph_path",
        batch_size=1,
        mode="video",
        gpu_decode=False,
        chunk_size=1,
        segment_size=81,
        trans_dir='/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090',
        measure_time=False,
        frame_chunk_size=10,
    )

    video_config_4090_2 = VAEExportConfig(
        csv_path = "/video/zhengmingwu/m2v-diffusers/middle/m2v-video-s1-v0.1-4090-part2.csv",
        ckpt_path = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.3.ckpt",
        save_dir = "/video/zhengmingwu/latents_3dvae_4090",
        output_csv = '/video/zhengmingwu/latents_3dvae_4090/m2v-video-s1-v0.1-4090-3dvae-part2.csv',
        tmp_csv_dir = "/video/zhengmingwu/tmp_vae_4090_part2",
        index_column="video_ceph_path",
        batch_size=1,
        mode="video",
        gpu_decode=False,
        chunk_size=1,
        segment_size=81,
        trans_dir='/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090',
        measure_time=False,
        frame_chunk_size=10,
    )

    video_config_4090_3 = VAEExportConfig(
        csv_path = "/video/zhengmingwu/m2v-diffusers/middle/m2v-video-s1-v0.1-4090-part3.csv",
        ckpt_path = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.3.ckpt",
        save_dir = "/video/zhengmingwu/latents_3dvae_4090",
        output_csv = '/video/zhengmingwu/latents_3dvae_4090/m2v-video-s1-v0.1-4090-3dvae-part3.csv',
        tmp_csv_dir = "/video/zhengmingwu/tmp_vae_4090_part3",
        index_column="video_ceph_path",
        batch_size=1,
        mode="video",
        gpu_decode=False,
        chunk_size=1,
        segment_size=81,
        trans_dir='/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090',
        measure_time=False,
        frame_chunk_size=10,
    )

    image_config = VAEExportConfig(
        csv_path="/video/zhengmingwu/m2v-diffusers/inputs/image2latents_240.csv",
        ckpt_path = "/video/zhengmingwu/ckpts/3dvae_wogan_v0.0.3.ckpt",
        save_dir = "/video/zhengmingwu/m2v-diffusers/outputs/test_image2latents",
        output_csv = '/video/zhengmingwu/m2v-diffusers/outputs/test_image2latents/test_image2latents.csv',
        tmp_csv_dir = "/video/zhengmingwu/tmp_vae",
        index_column = "img_path",
        mode="image",
        batch_size=30,
    )
    # dones = [
    #     '/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_h800/m2v-video-s1-v0/done/2024.04.04_17:55:08.csv',
    #     '/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090/m2v-video-s1-v0.1-4090-part2/done/2024.04.08_17:17:56.csv',
    #     '/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090/m2v-video-s1-v0.1-4090-part1/done/2024.04.08_17:14:28.csv',
    #     '/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090/m2v-video-s1-v0.1-4090-part3/done/2024.04.08_17:20:40.csv'
    # ]
    # dfs = []
    # for done in dones:
    #     try:
    #         dfs.append(pd.read_csv(done))
    #     except Exception as e:
    #         print(f"Error in {done}: {e}")

    # done_df = pd.concat(dfs, ignore_index=True)
    # print('Total done csv:', len(done_df))
    # done_paths = set(done_df['video_ceph_path'])
    # print('Total unique done paths:', len(done_paths))
    
    # done_csv = os.path.join('/video/zhengmingwu/m2v-diffusers/middle', 'm2v-video-s1-v0.1-done-0408.csv')
    # done_df = done_df.drop_duplicates(subset=['video_ceph_path'])
    # done_df.to_csv(done_csv, index=False)
    # print(f"Save to {done_csv}")

    # orginal_df = pd.read_csv('/ytech_m2v_hdd/m2v_data/m2v-video-s1-v0.1.csv')
    # print('Total ori num:', len(orginal_df))
    # todo_df = orginal_df[~orginal_df['video_ceph_path'].isin(done_paths)]
    # print('Total todo num:', len(todo_df))
    # # remove duplicated
    # todo_df = todo_df.drop_duplicates(subset=['video_ceph_path'])
    # print('Total todo num after drop duplicates:', len(todo_df))
    # todo_csv = os.path.join('/video/zhengmingwu/m2v-diffusers/middle', 'm2v-video-s1-v0.1-todo-0408.csv')
    
    # todo_df.to_csv(todo_csv, index=False)
    # print(f"Save to {todo_csv}")
    # quit()
    
    # video_config_h800.resume_path = '/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_h800/m2v-video-s1-v0/m2v-video-s1-v0.yaml'
    # video_config_4090_3.resume_path = '/video/zhengmingwu/m2v-diffusers/middle/transportation_3dvae_4090/m2v-video-s1-v0.1-4090-part3/m2v-video-s1-v0.1-4090-part3.yaml'
    main(video_config)
    
