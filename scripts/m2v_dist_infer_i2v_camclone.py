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
from src.pipelines.t2v_pipeline_flow import T2VFlowPipelineConfig
from src.utils import eval_setup, hashize, set_environments

USE_DIST = True  # for debug


@dataclass
class TestConfig(PrintableConfig):
    data: DataConfig = DataConfig()
    pipeline: T2VFlowPipelineConfig = T2VFlowPipelineConfig()
    test_dir: str = "outputs"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    transformer_ckpt_path: Optional[str] = None
    random_every_time: bool = False

    negative_prompt: str = "animation, 2d animation, 3d animation, Anime, Cartoon"
    width: int = 320
    height: int = 192
    fps: float = 15
    num_frames: int = 77
    guidance_scale: float = 12.5
    seed: Optional[int] = 42
    num_inference_steps: int = 50
    timestep_shift: float = 1.0

class M2V:
    def __init__(self, config: TestConfig):
        self.config = config
        self.update_config()

        if self.config.seed is None:
            self.seed = random.randint(0, sys.maxsize)
        else:
            self.seed = self.config.seed

        self.data = config.data.setup()
        if USE_DIST:
            deepspeed.init_distributed()
            self.state = PartialState()  # set device
            self.data.dataloader = prepare(self.data.dataloader)
            self.rank = torch.distributed.get_rank()
        else:
            self.rank = 0

        if self.rank == 0:
            print(config)

        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.dtype = dtypes[config.dtype]

        # must be initialized after PartialState()
        self.pipeline = config.pipeline.from_pretrained().to(device="cuda", dtype=self.dtype)

    def update_config(self):
        fps = self.config.fps
        total_frames = int(self.config.num_frames)

        if self.is_video:
            fps = min(max(fps, 11), 22)
            total_frames = min(max(total_frames, 1), 77)
        else:
            fps = self.config.fps = 0
            total_frames = self.config.num_frames = 1

        cfg = min(max(self.config.guidance_scale, 1), 50)
        steps = min(max(self.config.num_inference_steps, 1), 1000)
        negative_prompt = self.config.negative_prompt

        call_params = {
            "num_frames": total_frames,
            "width": self.config.width,
            "height": self.config.height,
            "num_inference_steps": steps,
            "negative_prompt": negative_prompt,
            "guidance_scale": cfg,
            "fps": fps,
            "timestep_shift": self.config.timestep_shift,
        }
        self.config.pipeline.call.update(call_params)

    @property
    def is_video(self):
        return self.config.num_frames != 1

    def images2video_buffer(self, images, kwargs):
        fps = kwargs.get("fps", 30)
        format = kwargs.get("format", "mp4")  # 默认为 mp4 格式
        codec = kwargs.get("codec", "libx264")  # 默认为 libx264 编码器
        ffmpeg_params = ["-crf", str(kwargs.get("crf", 12))]
        pixelformat = kwargs.get("pixelformat", "yuv420p")  # 视频像素格式

        # 创建一个 BytesIO 对象作为视频数据的内存存储
        video_stream = io.BytesIO()

        with imageio.get_writer(video_stream, fps=fps, format=format, codec=codec, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat) as writer:
            for idx in range(len(images)):
                writer.append_data(images[idx])

        return video_stream.getvalue()

    def write_data(self, data, output_path):
        if self.is_video:
            video_data = self.images2video_buffer(data, {"fps": max(1, self.config.pipeline.call["fps"])})
            with open(output_path, "wb") as f:
                f.write(video_data)
        else:
            img_pil = Image.fromarray(data[0])  # h w c
            img_pil.save(output_path)

    def run(self):
        cnt = 0
        result_csv = []
        for batch in track(self.data.dataloader, total=len(self.data.dataloader)):
            video = self.inference(batch)
            for i in range(self.config.data.batch_size):
                target_path, metadata = self.get_save_name(batch, i, cnt)
                
                import decord
                f, h, w, c = video[i].shape
                ctx = decord.cpu(0)
                rgb_reader = decord.VideoReader(batch['ref_data_paths'][0], ctx=ctx, height=h, width=w)
                length = len(rgb_reader)
                frame_indexes = list(range(batch['start_frames'][0], batch['num_frames'][0] * batch['frame_strides'][0], batch['frame_strides'][0]))
                frame_indexes = [min(frame_index, length - 1) for frame_index in frame_indexes]
                rgb_video = rgb_reader.get_batch(frame_indexes).numpy()
                concat_video = np.concatenate((rgb_video, video[i]), axis=2)
                self.write_data(concat_video, target_path)
                print(f"Saved to {target_path}")
                cnt += 1
                result_csv.append(metadata)

        # if USE_DIST:
        #     all_csv = [None for _ in range(torch.distributed.get_world_size())]
        #     torch.distributed.all_gather_object(all_csv, result_csv)
        # else:
        #     all_csv = [result_csv]

        # if self.rank == 0:
        #     all_results_csv = [item for sublist in all_csv for item in sublist]
        #     csv_path = os.path.join(self.config.test_dir, "results.csv")
        #     df = pd.DataFrame(all_results_csv, columns=["prompt", "data_path", "output_path"])
        #     df.drop_duplicates(subset=["prompt"], inplace=True)
        #     df.to_csv(csv_path, index=False)
        #     print(f"Saved to {csv_path}")

    @torch.no_grad()
    def inference(self, batch):
        prompt = batch["prompts"]
        if self.config.random_every_time:
            generator = None
        else:
            generator = torch.Generator(device=self.pipeline.device).manual_seed(self.seed)
        video_data = (self.pipeline(prompt, generator=generator, batch=batch, **self.config.pipeline.call, task="camclone")["images"].cpu().to(torch.float32).numpy() * 255).astype(
            np.uint8
        )  # (B F) C H W
        video_data = rearrange(video_data, "(b f) c h w -> b f h w c", b=self.config.data.batch_size)
        return video_data

    def get_save_name(self, batch, batch_index, cnt):
        data_path = (os.path.splitext(os.path.basename(batch["data_paths"][batch_index]))[0] + ".") if "data_paths" in batch else ""
        prompt = batch["prompts"][batch_index] if "prompts" in batch else ""

        end_fix = f"seed{self.config.seed}_{self.config.height}x{self.config.width}_cfg{self.config.guidance_scale}_shift{self.config.timestep_shift}"
        if prompt:
            words = "".join(char for char in prompt if char.isalnum() or char.isspace()).split()
            linked_words = "_".join(words[:10])
            name_endfix = f"{linked_words}.{hashize(prompt)}.{end_fix}"
        else:
            name_endfix = f"{hashize(str(self.config))}.{end_fix}"

        if self.is_video:
            file_type = "mp4"
        else:
            file_type = "png"

        if "index" in batch:
            video_name = batch["index"][batch_index] + f".{file_type}"
        else:
            video_name = f"CamCloneMaster_{(8 * cnt + self.rank):04d}.{file_type}"
            # video_name = f"R{self.rank}L{cnt}_{data_path}{name_endfix}.{file_type}"
        output_path = os.path.join(self.config.test_dir, video_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path, (prompt, data_path, output_path)


def main(config: TestConfig):
    torch.autograd.set_grad_enabled(False)
    if config.transformer_ckpt_path:
        config.pipeline.transformer_config.transformer_ckpt_path = config.transformer_ckpt_path
    m2v = M2V(config)
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
    # import debugpy
    # import os
    # if int(os.environ.get("RANK", 0)) == 0:
    #     debugpy.listen(("0.0.0.0", 5678))
    #     print("⏳ Waiting for debugger attach on port 5678...")
    #     debugpy.wait_for_client()
    #     print("✅ Debugger attached!")
    set_environments()
    yaml_path = deepcopy(sys.argv[1])
    del sys.argv[1]
    assert os.path.exists(yaml_path), f"Trainer config's YAML file not found: {yaml_path}. Note that YAML path must be the first argument."
    trainer_config = eval_setup(yaml_path)
    test_config = TestConfig(data=trainer_config.val_data, pipeline=trainer_config.pipeline)
    main(tyro.cli(TestConfig, default=test_config))
