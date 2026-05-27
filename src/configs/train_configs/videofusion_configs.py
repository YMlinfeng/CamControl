import copy
from typing import Dict

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig
from ...models import *
from ...pipelines import *

videofusion_configs: Dict[str, TrainerConfig] = {}


videofusion_configs["videofusion"] = TrainerConfig(
    experiment_name="videofusion",
    train_data=(
        train_data := DataConfig(
            path="/video/vcg/final_list/get_all/vcg_all_llama2.csv",
            caption_column="llama2_caption",
            batch_size=16,
            height=320,
            width=576,
        )
    ),
    trainable_modules=["unet"],
    optimizer=AdamWOptimizerConfig(),
    pipeline=T2VPipelineConfig(
        unet_config=UNet3DConfig(num_frames=16),
        scheduler_kwargs={
            "_class_name": "DDIMScheduler",
        },
        ckpt_path="/group/ckpt/diffusers/zeroscope_v2_576w_m2v",
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
        num_samples=6,
        batch_size=6,
    ),
)


H_VF = 768  # 836
W_VF = 512  # 512
videofusion_configs["videofusion_hair"] = TrainerConfig(
    num_epochs=200,
    experiment_name="videofusion_hair",
    trainable_modules=["unet"],
    optimizer=AdamWOptimizerConfig(
        lr=1e-5,
    ),
    pipeline=T2VPipelineConfig(
        unet_config=UNet3DConfig(
            # unet_ckpt_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/exps/videofusion-hair-09x3-2s-448x256/checkpoints/checkpoint-448000/unet/pytorch_model.ckpt",
            # unet_ckpt_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/exps/videofusion-hair-0920-2s/checkpoints/checkpoint-1296000/unet/pytorch_model.ckpt",
            unet_ckpt_path="/group/houliang/video/m2v-diffusers/exps/base_v1_24gpus/ckpt/checkpoint-23808000",
            num_frames=16,
        ),
        # noise_scheduler_kwargs={},
        ckpt_path="/group/ckpt/diffusers/zeroscope_v2_576w",
        call={
            "height": H_VF,
            "width": W_VF,
            "num_inference_steps": 50,
        },
        proportion_empty_prompts=0.1,
    ),
    train_data=DataConfig(
        path="/group/dengyufan/kvq/final_09x3-1030-1101-2s_kvq.csv",
        caption_column="caption",
        height=H_VF,
        width=W_VF,
        batch_size=4,
    ),
    val_data=DataConfig(
        path="/group/dengyufan/kvq/final_09x3-1030-1101-2s_kvq.csv",
        height=H_VF,
        width=W_VF,
        num_frames=16,
        num_samples=6,
        batch_size=6,
    ),
)
