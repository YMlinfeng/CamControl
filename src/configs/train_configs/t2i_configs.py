from typing import Dict

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig
from ...models import *
from ...pipelines import *

svd_train_scheduler = {
    "_class_name": "EulerDiscreteScheduler",
    "beta_end": 0.012,
    "beta_schedule": "scaled_linear",
    "beta_start": 0.00085,
    "clip_sample": False,
    "interpolation_type": "linear",
    "num_train_timesteps": 1000,
    "prediction_type": "v_prediction",
    "set_alpha_to_one": False,
    "sigma_max": 80.0,
    "sigma_min": 0.002,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "timestep_type": "continuous",
    "trained_betas": None,
    "use_karras_sigmas": True,
}


t2i_configs: Dict[str, TrainerConfig] = {}
t2i_configs["sdxl"] = TrainerConfig(
    experiment_name="sdxl",
    data_per_save=500000,
    data_per_val=500000,
    train_data=(
        train_data := DataConfig(
            path="/video/vcg/final_list/get_all/vcg_all_llama2.csv",
            caption_column="llama2_caption",
            batch_size=12,
            height=256,
            width=448,
            sample_position="random",
            sample_type="random",
            num_frames=16,
        )
    ),
    trainable_modules=["unet"],
    optimizer=AdamWOptimizerConfig(lr=1e-6),
    scheduler=SchedulerConfig(),
    pipeline=T2VXLPipelineConfig(
        edm_config=EDMTrainConfig(
            P_mean=-1.2,
            P_std=1.2,
        ),
        unet_config=UNet2DConfig(num_frames=16),
        scheduler_kwargs=svd_train_scheduler,
        ckpt_path="/group/ckpt/diffusers/stable-diffusion-xl-base-1.0",
        diffusers_vae_ckpt_path="/group/ckpt/diffusers/sdxl-vae-fp16-fix",
        call={
            "height": train_data.height,
            "width": train_data.width,
            "num_inference_steps": 50,
        },
        proportion_empty_prompts=0.1,
    ),
    val_data=DataConfig(
        path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
        height=train_data.height,
        width=train_data.width,
        num_frames=train_data.num_frames,
        num_samples=8,
        batch_size=1,
    ),
    val_types=["gt", "t2v"],
)
