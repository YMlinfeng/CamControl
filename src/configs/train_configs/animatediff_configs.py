import copy
from typing import Dict

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig
from ...models import *
from ...pipelines import *

animatediff_configs: Dict[str, TrainerConfig] = {}


animatediff_configs["animatediff"] = TrainerConfig(
    experiment_name="animatediff",
    train_data=(
        train_data := DataConfig(
            path="/video/vcg/final_list/get_all/vcg_all_llama2.csv",
            caption_column="llama2_caption",
            batch_size=16,
            height=256,
            width=256,
        )
    ),
    trainable_modules=[{"unet": ["motion_modules"]}],
    optimizer=AdamWOptimizerConfig(),
    scheduler=SchedulerConfig(),
    pipeline=T2VPipelineConfig(
        unet_config=AnimateDiffUNetConfig(
            num_frames=train_data.num_frames,
        ),
        scheduler_kwargs={
            "_class_name": "DDIMScheduler",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "steps_offset": 1,
            "clip_sample": False,
        },
        call={
            "height": train_data.height,
            "width": train_data.width,
            "num_inference_steps": 25,
        },
        proportion_empty_prompts=0.1,
    ),
    val_data=DataConfig(
        path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
        height=train_data.height,
        width=train_data.width,
        num_frames=train_data.num_frames,
        num_samples=6,
        batch_size=6,
    ),
)

animatediff_configs["animatediff_v1"] = copy.deepcopy(animatediff_configs["animatediff"])
animatediff_configs["animatediff_v1"].pipeline.unet_config.update(
    ckpt_path="/group/gaoyuan/huggingface/animatediff/Motion_Module/mm_sd_v14.ckpt",
    max_len=24,
    mid_block_type="UNetMidBlock2DCrossAttn",
    replace_groupnorm_forward=True,
)

animatediff_configs["codebasev2_animatediff"] = copy.deepcopy(animatediff_configs["animatediff"])
animatediff_configs["codebasev2_animatediff"].seed = 666
animatediff_configs["codebasev2_animatediff"].pipeline.negative_prompt = (
    "prompt poorly rendered face, poorly drawn face, poor facial details, poorly drawn hands, poorly rendered hands, low resolution, images cut out at the top, left, right, bottom. bad composition, mutated body parts, blurry image, disfigured, oversaturated, bad anatomy, deformed body features"
)


animatediff_configs["animatediff_xl"] = TrainerConfig(
    experiment_name="animatediff_xl",
    train_data=(
        train_data := DataConfig(
            path="/video/vcg/final_list/get_all/vcg_all_llama2.csv",
            caption_column="llama2_caption",
            batch_size=1,
            height=320,
            width=512,
        )
    ),
    trainable_modules=[{"unet": ["motion_modules"]}],
    optimizer=AdamWOptimizerConfig(),
    scheduler=SchedulerConfig(),
    pipeline=T2VXLPipelineConfig(
        unet_config=AnimateDiffXLUNetConfig(
            num_frames=train_data.num_frames,
        ),
        scheduler_kwargs={
            "_class_name": "EulerDiscreteScheduler",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.02,
            "beta_schedule": "scaled_linear",
            "steps_offset": 1,
            "timestep_spacing": "leading",
        },
        ckpt_path="/group/ckpt/diffusers/stable-diffusion-xl-base-1.0",
        diffusers_vae_ckpt_path="/group/ckpt/diffusers/sdxl-vae-fp16-fix",
        call={
            "height": train_data.height,
            "width": train_data.width,
            "num_inference_steps": 100,
        },
        noise_offset=0.05,
        proportion_empty_prompts=0.1,
    ),
    val_data=DataConfig(
        path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
        height=train_data.height,
        width=train_data.width,
        num_frames=train_data.num_frames,
        num_samples=1,
        batch_size=1,
    ),
)

animatediff_configs["rpe_animatediff"] = TrainerConfig(
    experiment_name="rpe_animatediff",
    train_data=(
        train_data := DataConfig(
            path="/video/vcg/final_list/get_all/vcg_all_llama2.csv",
            caption_column="llama2_caption",
            batch_size=16,
            height=360,
            width=640,
        )
    ),
    trainable_modules=[{"unet": ["motion_modules"]}],
    optimizer=AdamWOptimizerConfig(),
    scheduler=SchedulerConfig(),
    pipeline=T2VPipelineConfig(
        unet_config=AnimateDiffUNetConfig(
            pe_type="rand",
            pe_max_len=240,
            num_frames=train_data.num_frames,
            zero_init=True,
        ),
        scheduler_kwargs={
            "_class_name": "DDIMScheduler",
            "num_train_timesteps": 1000,
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "linear",
            "steps_offset": 1,
            "clip_sample": False,
        },
        call={
            "height": train_data.height,
            "width": train_data.width,
            "num_inference_steps": 25,
        },
        proportion_empty_prompts=0.1,
    ),
    val_data=DataConfig(
        path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
        height=train_data.height,
        width=train_data.width,
        num_frames=train_data.num_frames,
        num_samples=6,
        batch_size=6,
    ),
)
