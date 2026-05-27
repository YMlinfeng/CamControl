import os
import io
import sys
import random
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union


import tyro
from rich import print
from rich.progress import track
from tqdm import tqdm

import numpy as np
import torch
from einops import rearrange
import imageio.v2 as imageio
from PIL import Image
import pandas as pd


from stable_diffusion_application.kaimm.accelerate_utils import prepare
from stable_diffusion_application.kaimm.state import PartialState
import deepspeed
import deepspeed.utils.zero_to_fp32


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.configs.base_config import PrintableConfig
from src.data.base_data import DataConfig
from src.pipelines import T2VCPFlowPipelineConfig
from src.utils import eval_setup, hashize, set_environments
from scripts.m2v_dist_infer import TestConfig, M2V

USE_DIST = True  # for debug


@dataclass
class TestUpConfig(TestConfig):
    data: DataConfig = DataConfig()
    pipeline: T2VCPFlowPipelineConfig = T2VCPFlowPipelineConfig()
    test_dir: str = "outputs"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    transformer_ckpt_path: Optional[str] = None
    random_every_time: bool = False

    upscale_factor: int = 2
    width: int = 672 * 2
    height: int = 384 * 2
    fps: float = 15
    num_frames: int = 77
    seed: Optional[int] = 42
    num_inference_steps: int = 50
    val_gt: bool = False

    total_num_frames: Optional[int] = None

    cat_result: bool = True


class M2VUpScale(M2V):

    def update_config(self):
        self.total_num_frames = self.config.total_num_frames or self.config.num_frames
        self.config.data.height = self.config.height
        self.config.data.width = self.config.width
        self.config.data.sample_fps = self.config.fps
        self.config.data.num_frames = self.total_num_frames
        self.config.data.sample_position = "first"
        self.config.data.batch_size = 1

        fps = self.config.fps
        num_frames = int(self.config.num_frames)

        if self.is_video:
            fps = min(max(fps, 11), 22)
            num_frames = min(max(num_frames, 17), 77)
        else:
            fps = self.config.fps = 0
            num_frames = self.config.num_frames = 1

        steps = min(max(self.config.num_inference_steps, 20), 100)

        call_params = {
            "num_frames": num_frames,
            "num_inference_steps": steps,
            "fps": fps,
            "upscale_factor": self.config.upscale_factor,
        }
        self.config.pipeline.call.update(call_params)

    def run(self):
        cnt = 0
        result_csv = []
        for batch in track(self.data.dataloader, total=len(self.data.dataloader)):
            video = self.inference(batch)
            for i in range(self.config.data.batch_size):
                target_path, metadata = self.get_save_name(batch, i, cnt)
                if self.config.cat_result:
                    self.write_data(video[i], target_path)
                else:
                    video_ori, video_downup, video_out = video[0][i], video[1][i], video[2][i]
                    video_ori_path = target_path.replace(".mp4", "_ori.mp4")
                    video_downup_path = target_path.replace(".mp4", "_downup.mp4")
                    self.write_data(video_ori, video_ori_path)
                    self.write_data(video_downup, video_downup_path)
                    self.write_data(video_out, target_path)
                print(f"Saved to {target_path}")
                cnt += 1
                result_csv.append(metadata)

        if USE_DIST:
            all_csv = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_csv, result_csv)
        else:
            all_csv = [result_csv]

        if self.rank == 0:
            all_results_csv = [item for sublist in all_csv for item in sublist]
            csv_path = os.path.join(self.config.test_dir, "results.csv")
            df = pd.DataFrame(all_results_csv, columns=["data_path", "output_path"])
            df.drop_duplicates(subset=["data_path"], inplace=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")

    def get_save_name(self, batch, batch_index, cnt):
        data_path = (os.path.splitext(os.path.basename(batch["data_paths"][batch_index]))[0] + ".") if "data_paths" in batch else ""

        end_fix = f"seed{self.config.seed}_{self.config.height}x{self.config.width}x{self.config.total_num_frames}"
        name_endfix = f"{hashize(str(self.config))}.{end_fix}"

        if self.is_video:
            file_type = "mp4"
        else:
            file_type = "png"

        if "index" in batch:
            video_name = batch["index"][batch_index] + f".{file_type}"
        else:
            video_name = f"R{self.rank}L{cnt}_{data_path}{name_endfix}.{file_type}"
        output_path = os.path.join(self.config.test_dir, video_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path, (data_path, output_path)

    @torch.no_grad()
    def inference(self, batch):
        self.pipeline.set_progress_bar_config(disable=True)
        self.pipeline.vae.enable_slicing()
        video_ori = batch["data"]  # b f c h w
        b = video_ori.shape[0]

        segment_size = self.config.num_frames
        frame_indexes = list(range(0, video_ori.shape[1]))
        # frame_indexes = frame_indexes[: (segment_size * (len(frame_indexes) // segment_size))]
        
        frame_tensors_all = []
        for i in tqdm(range(0, len(frame_indexes), segment_size), desc="Inference"):
            frame_indexes_chunk = frame_indexes[i : i + segment_size]
            # Ensure that the last segment has the right length for vae (4n+1)
            num_frames_chunk = (len(frame_indexes_chunk) - 1) // 4 * 4 + 1
            frame_indexes_chunk = frame_indexes_chunk[:num_frames_chunk]
            
            video_ori_chunk = video_ori[:, frame_indexes_chunk] 
            video_4d = rearrange(video_ori_chunk, "b f c h w -> (b f) c h w")
            if self.config.val_gt:
                video_4d_down = torch.nn.functional.interpolate(video_4d, scale_factor=1 / self.config.upscale_factor, mode="bilinear", align_corners=False)
                video_down = rearrange(video_4d_down, "(b f) c h w -> b f c h w", b=b)
                video_4d_downup = torch.nn.functional.interpolate(video_4d_down, scale_factor=self.config.upscale_factor, mode="bilinear", align_corners=False)
            else:
                video_down = video_ori_chunk
                video_4d_downup = torch.nn.functional.interpolate(video_4d, scale_factor=self.config.upscale_factor, mode="bilinear", align_corners=False)

            video_downup = rearrange(video_4d_downup, "(b f) c h w -> b f c h w", b=b)

            if self.config.random_every_time:
                generator = None
            else:
                generator = torch.Generator(device=self.pipeline.device).manual_seed(self.seed)
            # call
            video_out = self.pipeline(videos=video_down, generator=generator, **self.config.pipeline.call)["images"]

            video_out = rearrange(video_out, "(b f) c h w -> b f h w c", b=self.config.data.batch_size).cpu().to(torch.float32)
            video_downup = rearrange((video_downup / 2) + 0.5, "b f c h w -> b f h w c").cpu().to(torch.float32)
            video_ori_chunk = rearrange((video_ori_chunk / 2) + 0.5, "b f c h w -> b f h w c").cpu().to(torch.float32)
            # cat on w
            if self.config.val_gt:
                cat_video = [video_ori_chunk, video_downup, video_out]
            else:
                cat_video = [video_downup, video_downup, video_out]

            if self.config.cat_result:
                video_out = torch.cat(cat_video, dim=-2)
            else:
                video_out = cat_video

            frame_tensors_all.append(video_out)

        if self.config.cat_result:
            frames = (torch.cat(frame_tensors_all, dim=1).numpy() * 255).astype(np.uint8)
        else:
            frames = [torch.cat(tensors, dim=1) for tensors in zip(*frame_tensors_all)]
            frames = [(frame.numpy() * 255).astype(np.uint8) for frame in frames]
        return frames


def main(config: TestUpConfig):
    torch.autograd.set_grad_enabled(False)
    if config.transformer_ckpt_path:
        config.pipeline.transformer_config.transformer_ckpt_path = config.transformer_ckpt_path
    m2v = M2VUpScale(config)
    m2v.run()


if __name__ == "__main__":
    """
    使用例，这里是开发机，使用dist_run
    bash scripts/dist_run.sh \
        python scripts/m2v_dist.py \
        /ytech_m2v2_hdd/houliang/m2v-diffusers/exps/102_mvb_10b_77x256x256_800gpus/config.yml \
        --transformer_ckpt_path /ytech_m2v2_hdd/houliang/m2v-diffusers/exps/102_mvb_10b_77x256x256_800gpus/checkpoints/checkpoint-8000000/ema/ema.ckpt \
        --data.cache_dir None \
        --test_dir /video/zhengmingwu/m2v-diffusers/outputs/m2v_output/102_mvb_10b_8mstep_video \
        --num_frames 77 \
    """

    set_environments()
    yaml_path = deepcopy(sys.argv[1])
    del sys.argv[1]
    assert os.path.exists(yaml_path), f"Trainer config's YAML file not found: {yaml_path}. Note that YAML path must be the first argument."
    trainer_config = eval_setup(yaml_path)
    test_config = TestUpConfig(data=trainer_config.val_data, pipeline=trainer_config.pipeline)
    main(tyro.cli(TestUpConfig, default=test_config))
