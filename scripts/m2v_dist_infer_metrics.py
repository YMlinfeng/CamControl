import os
import io
import sys
import random
import time
import traceback
from datetime import datetime
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
import matplotlib.pyplot as plt

from stable_diffusion_application.kaimm.accelerate_utils import prepare
from stable_diffusion_application.kaimm.state import PartialState
import deepspeed
import deepspeed.utils.zero_to_fp32

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.configs.base_config import PrintableConfig
from src.data.base_data import DataConfig
from src.pipelines.t2v_pipeline_flow import T2VFlowPipelineConfig
from src.utils import load_model, log_to_rank0, eval_setup, hashize, set_environments

from assets.models_eval import IQAIGCModelV2, MotionSmoothModel, UMTScoreModel, VQAIGCModelV2, get_overall_score_v2, get_overall_score_v3

USE_DIST = True  # for debug


@dataclass
class TestConfig(PrintableConfig):
    data: DataConfig = DataConfig()
    pipeline: T2VFlowPipelineConfig = T2VFlowPipelineConfig()
    test_dir: str = "outputs"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    transformer_ckpt_path: Optional[str] = None

    negative_prompt: str = "animation, 2d animation, 3d animation, Anime, Cartoon" + ", blurry, deformed, disfigured, low quality, text, collage, grainy, logo, no visual content, blurred effect, striped background, abstract, illustration, computer generated, distorted"
    width: int = 320
    height: int = 192
    fps: float = 15
    num_frames: int = 77
    guidance_scale: float = 12.5
    seed: Optional[int] = 42
    num_inference_steps: int = 50
    offload: bool = False
    timestep_shift: float = 5.0


class M2VMetrics:
    def __init__(self, config: TestConfig):
        self.config = config

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

        self.config = config

        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        self.dtype = dtypes[config.dtype]

        # must be initialized after PartialState()
        #self.pipeline = config.pipeline.from_pretrained().to(device="cuda", dtype=self.dtype)
        self.pipeline = None
        self.update_config()

        self.model_iq = IQAIGCModelV2().to(device="cuda", dtype=self.dtype)
        self.model_vq = VQAIGCModelV2().to(device="cuda", dtype=self.dtype)
        self.model_umt = UMTScoreModel().to(device="cuda", dtype=self.dtype)
        self.model_motionsmooth = MotionSmoothModel().to(device="cuda", dtype=self.dtype)
        self.model_iq.eval()
        self.model_vq.eval()
        self.model_umt.eval()
        self.model_motionsmooth.eval()

    def update_ckpt(self, transformer_ckpt_path, test_dir):
        def rename_func(state_dict):
            new_dict = {}
            for k in state_dict.keys():
                ori_k = k
                if "transformer_blocks" not in k:  # global
                    if "scale_table" in k:
                        if "global_scale_table" not in k:
                            k = k.replace("scale_table", "global_scale_table")  # rename
                        if "weight" not in k:
                            k += ".weight"  # nn.Parameter -> nn.Embedding
                else:  # layer
                    if "scale_table" in k and "scale_table.weight" not in k:
                        k += ".weight"
                if k.startswith("transformer."):
                    k = k[12:]
                new_dict[k] = state_dict[ori_k]
            return new_dict

        self.config.test_dir = test_dir
        if self.pipeline is None:
            self.config.pipeline.transformer_config.transformer_ckpt_path = transformer_ckpt_path
            self.pipeline = self.config.pipeline.from_pretrained().to(device="cuda", dtype=self.dtype)
        self.pipeline.transformer = load_model(self.pipeline.transformer, transformer_ckpt_path, rename_func=rename_func)
        self.pipeline.pipeline_config.offload = self.config.offload

    def update_config(self):
        fps = self.config.fps
        total_frames = int(self.config.num_frames)

        if self.is_video:
            fps = min(max(fps, 11), 22)
            total_frames = min(max(total_frames, 1), 77)
        else:
            fps = 0
            total_frames = 1
        cfg = min(max(self.config.guidance_scale, 1), 50)
        steps = min(max(self.config.num_inference_steps, 1), 1000)
        negative_prompt = self.config.negative_prompt

        call_params = {
            "num_frames": total_frames,
            "width": self.config.width,
            "height": self.config.height,
            "num_inference_steps": steps,
            "negative_prompt": negative_prompt if negative_prompt != "" else "animation, 2d animation, 3d animation, Anime, Cartoon",
            "guidance_scale": cfg,
            "fps": fps,
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
            video, fitted_score = self.inference(batch)
            for i in range(self.config.data.batch_size):
                target_path, metadata = self.get_save_name(batch, i, cnt)
                self.write_data(video[i], target_path)
                print(f"Saved to {target_path}")
                cnt += 1
                expanded_metadata = (metadata[0], metadata[1], metadata[2],
                                     fitted_score["overall"][i].item(),
                                     fitted_score["vq"][i].item(),
                                     fitted_score["t2valign"][i].item(),
                                     fitted_score["dq"][i].item(),
                )
                result_csv.append(expanded_metadata)

        if USE_DIST:
            all_csv = [None for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather_object(all_csv, result_csv)
        else:
            all_csv = [result_csv]

        if self.rank == 0:
            all_results_csv = [item for sublist in all_csv for item in sublist]
            csv_path = os.path.join(self.config.test_dir, "results.csv")
            df = pd.DataFrame(all_results_csv, columns=["prompt", "data_path", "output_path",
                                                        "overall", "vq", "t2valign", "dq"])
            df.drop_duplicates(subset=["prompt"], inplace=True)
            df.to_csv(csv_path, index=False)
            print(f"Saved to {csv_path}")

    @torch.no_grad()
    def inference(self, batch):
        prompt = batch["prompts"]
        generator = torch.Generator(device=self.pipeline.device).manual_seed(self.seed)
        video_data = (self.pipeline(prompt, generator=generator, timestep_shift=self.config.timestep_shift, **self.config.pipeline.call)["images"].cpu().to(torch.float32).numpy() * 255).astype(np.uint8)
        print(f"Generated video data shape: {video_data.shape}")

        video_data = rearrange(video_data, "(b f) c h w -> b f h w c", b=self.config.data.batch_size)
        iq_score_a, iq_score_q = self.model_iq(video_data, prompt)
        iq_score_q = rearrange(iq_score_q, "(b f) -> b f", b=self.config.data.batch_size).mean(1)
        vq_score_vq, vq_score_dq, vq_score_a, vq_score_overall = self.model_vq(video_data, prompt)
        umt_score = self.model_umt(video_data, prompt)
        motionsmooth_score = self.model_motionsmooth(video_data)

        fitted_overall_score, fitted_vq_score, fitted_t2valign_score, fitted_dq_score = get_overall_score_v3(
            iq_score_a=iq_score_a, iq_score_q=iq_score_q,
            vq_score_vq=vq_score_vq, vq_score_dq=vq_score_dq, vq_score_a=vq_score_a, vq_score_overall=vq_score_overall,
            umt_score=umt_score, motionsmooth_score=motionsmooth_score
        )

        fitted_scores = {
            "overall": fitted_overall_score,
            "vq": fitted_vq_score,
            "t2valign": fitted_t2valign_score,
            "dq": fitted_dq_score
        }

        return video_data, fitted_scores

    def get_save_name(self, batch, batch_index, cnt):
        data_path = (os.path.splitext(os.path.basename(batch["data_paths"][batch_index]))[0] + ".") if "data_paths" in batch else ""
        prompt = batch["prompts"][batch_index] if "prompts" in batch else ""

        end_fix = f"seed{self.config.seed}_{self.config.height}x{self.config.width}_cfg{self.config.guidance_scale}"
        if prompt != "":
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
            video_name = f"R{self.rank}L{cnt}_{data_path}{name_endfix}_S{self.config.timestep_shift}.{file_type}"
        output_path = os.path.join(self.config.test_dir, "video", video_name)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        return output_path, (prompt, data_path, output_path)


def get_new_ckpt(ckpt_list, valmetric_result_dir):
    new_ckpt_list = []
    for ckpt_name in ckpt_list:
        if "checkpoint" not in ckpt_name:
            continue
        ckpt_valmetric_result_dir = os.path.join(valmetric_result_dir, ckpt_name)
        if not os.path.exists(os.path.join(ckpt_valmetric_result_dir, "done.txt")):
            new_ckpt_list.append(ckpt_name)
    return new_ckpt_list


def get_metrics_plot(valmetric_result_dir, model_type="ema"):
    result_list = []
    for ckpt_name in os.listdir(valmetric_result_dir):
        ckpt_valmetric_result_dir = os.path.join(valmetric_result_dir, ckpt_name)
        csv_pth = os.path.join(ckpt_valmetric_result_dir, model_type, "results.csv")
        if not os.path.exists(csv_pth):
            continue
        overall, vq, t2valign, dq = [], [], [], []
        step_num = ckpt_name.replace("checkpoint-", "")
        df = pd.read_csv(csv_pth)
        for i, line in df.iterrows():
            overall.append(line["overall"])
            vq.append(line["vq"])
            t2valign.append(line["t2valign"])
            dq.append(line["dq"])
        result_list.append((int(step_num), np.mean(overall), np.mean(vq), np.mean(t2valign), np.mean(dq)))

    if result_list:
        result_list.sort(key=lambda p: p[0])
        x = [item[0] for item in result_list]
        overall = [item[1] for item in result_list]
        vq = [item[2] for item in result_list]
        t2valign = [item[3] for item in result_list]
        dq = [item[4] for item in result_list]

        fig, axs = plt.subplots(2, 2, figsize=(10, 8))

        axs[0, 0].plot(x, overall)
        axs[0, 0].set_title('Fitted overall score')
        axs[0, 0].grid(True)
        for i in range(len(x)):
            axs[0, 0].text(x[i], overall[i], f'{overall[i]:.2f}', fontsize=8, verticalalignment='bottom')

        axs[0, 1].plot(x, vq)
        axs[0, 1].set_title('Fitted visual quality score')
        axs[0, 1].grid(True)
        for i in range(len(x)):
            axs[0, 1].text(x[i], vq[i], f'{vq[i]:.2f}', fontsize=8, verticalalignment='bottom')

        axs[1, 0].plot(x, t2valign)
        axs[1, 0].set_title('Fitted t2v alignment score')
        axs[1, 0].grid(True)
        for i in range(len(x)):
            axs[1, 0].text(x[i], t2valign[i], f'{t2valign[i]:.2f}', fontsize=8, verticalalignment='bottom')

        axs[1, 1].plot(x, dq)
        axs[1, 1].set_title('Fitted dynamic quality score')
        axs[1, 1].grid(True)
        for i in range(len(x)):
            axs[1, 1].text(x[i], dq[i], f'{dq[i]:.2f}', fontsize=8, verticalalignment='bottom')

        plt.tight_layout()
        plt.savefig(valmetric_result_dir + '/{}-{}-plots.png'.format(model_type, x[-1]))


if __name__ == "__main__":
    """
    使用例，这里是开发机，使用dist_run
    bash scripts/dist_run.sh \
        python scripts/m2v_dist_infer_metrics.py \
        /ytech_m2v2_hdd/houliang/m2v-diffusers/exps/102_mvb_10b_77x256x256_800gpus/config.yml \
        --width 320 \
        --height 192 \
        --fps 15 \
        --num_frames 77
    """

    set_environments()
    yaml_path = deepcopy(sys.argv[1])
    del sys.argv[1]
    assert os.path.exists(yaml_path), f"Trainer config's YAML file not found: {yaml_path}. Note that YAML path must be the first argument."

    trainer_config = eval_setup(yaml_path)
    trainer_config.valmetrics_data.batch_size = 1
    test_config = TestConfig(data=trainer_config.valmetrics_data, pipeline=trainer_config.pipeline)
    test_config = tyro.cli(TestConfig, default=test_config)

    torch.autograd.set_grad_enabled(False)
    m2v = M2VMetrics(test_config)

    exp_dir = os.path.dirname(yaml_path)
    model_dir = exp_dir + "/checkpoints"
    valmetric_result_dir = exp_dir + "/valmetrics_result"
    os.makedirs(valmetric_result_dir, exist_ok=True)

    if torch.distributed.get_rank() == 0:
        get_metrics_plot(valmetric_result_dir)

    while(1):
        if not os.path.exists(model_dir):
            continue

        ckpt_list = os.listdir(model_dir)
        new_ckpt_list = get_new_ckpt(ckpt_list, valmetric_result_dir)

        for ckpt_name in new_ckpt_list:
            log_to_rank0("evaluating metrics on ckpt: {}".format(ckpt_name))

            ckpt_dir = os.path.join(model_dir, ckpt_name)
            transformer_ckpt_path = ckpt_dir + "/ema/ema.ckpt"
            if not os.path.exists(transformer_ckpt_path):
                log_to_rank0("No valid ckpt found for: {}".format(ckpt_name))
                transformer_ckpt_path = ckpt_dir + "/ema/ema_merged.ckpt"
                if not os.path.exists(transformer_ckpt_path):
                    log_to_rank0("No valid merged ckpt found for: {}".format(ckpt_name))
                    continue

            ckpt_valmetric_result_dir = os.path.join(valmetric_result_dir, ckpt_name)
            os.makedirs(ckpt_valmetric_result_dir, exist_ok=True)

            # ema
            test_dir = ckpt_valmetric_result_dir + "/ema"
            os.makedirs(test_dir, exist_ok=True)

            try:
                m2v.update_ckpt(transformer_ckpt_path, test_dir)
                m2v.run()
                if torch.distributed.get_rank() == 0:
                    f = open(os.path.join(ckpt_valmetric_result_dir, "done.txt"), "w")
                    f.writelines("done")
                    f.close()
            except:
                traceback.print_exc()

        if not new_ckpt_list and torch.distributed.get_rank() == 0:
            get_metrics_plot(valmetric_result_dir)

        current_date_and_time = datetime.now()
        log_to_rank0("Time: ", current_date_and_time)
        time.sleep(10)
