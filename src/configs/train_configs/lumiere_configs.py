from copy import deepcopy
from typing import Any, List, Literal, Optional, Tuple, Type, Union

from diffusers.models.transformer_temporal import TransformerTemporalModel

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig
from ...models import *
from ...models.blocks import *
from ...models.resnet import *
from ...pipelines import *

# 必须是64的倍数, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216]
HEIGHT, WIDTH = 576, 1024
# HEIGHT, WIDTH = 320, 512
# HEIGHT, WIDTH = 256, 448
# HEIGHT, WIDTH = 128, 256
# HEIGHT, WIDTH = 768, 1344
# HEIGHT, WIDTH = 768, 1344

SAMPLE_FPS = 16
NUM_FRAMES = 16


vcg_train_data = DataConfig(
    path="/video/vcg/final_list/get_all/vcg_all_llama2.csv",
    caption_column="llama2_caption",
    batch_size=32,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    sample_type="fps",
    sample_position="random",
    sample_fps=SAMPLE_FPS,
)
vcg_val_data = DataConfig(
    path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    num_samples=8,
    batch_size=1,
    shuffle=False,
)

lumiere_configs = {}
lumiere_configs["lumiere_xl"] = TrainerConfig(
    experiment_name="lumiere_xl",
    train_data=vcg_train_data,
    trainable_modules=[{"unet": ["motion_modules", "downsamplers1d", "upsamplers1d"]}],
    optimizer=AdamWOptimizerConfig(
        lr=1e-4,
    ),
    scheduler=SchedulerConfig(),
    pipeline=T2VXLPipelineConfig(
        unet_config=AnimateDiffXLUNetConfig(
            mm_ckpt_path=None,
            down_block_configs=[
                DownBlock3DConfig(
                    motion_module_cls=LumiereTemporalConvLayer_Warped,
                    add_downsample_1d=True,
                ),
                CrossAttnDownBlock3DConfig(
                    motion_module_cls=LumiereTemporalConvLayer_Warped,
                    add_downsample_1d=True,
                ),
                CrossAttnDownBlock3DConfig(
                    motion_module_cls=LumiereTemporalConvLayer_Warped,
                    add_downsample_1d=True,
                ),
            ],
            mid_block_config=UNetMidBlock3DCrossAttnConfig(
                motion_module_cls=TransformerTemporalModel,
                num_temporal_layers=4,
            ),
            up_block_configs=[
                CrossAttnUpBlock3DConfig(
                    motion_module_cls=LumiereTemporalConvLayer_Warped,
                    add_upsample_1d=True,
                ),
                CrossAttnUpBlock3DConfig(
                    motion_module_cls=LumiereTemporalConvLayer_Warped,
                    add_upsample_1d=True,
                ),
                UpBlock3DConfig(
                    motion_module_cls=LumiereTemporalConvLayer_Warped,
                    add_upsample_1d=True,
                ),
            ],
            pe_max_len=max(32, NUM_FRAMES),
            num_frames=NUM_FRAMES,
            zero_init=True,
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
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": 100,
        },
        noise_offset=0.05,
        proportion_empty_prompts=0.15,
    ),
    val_data=vcg_val_data,
    val_types=["gt", "t2v"],
)


lumiere_configs["lumiere_xxl"] = deepcopy(lumiere_configs["lumiere_xl"])
lumiere_configs["lumiere_xxl"].trainable_modules = [{"unet": ["motion_modules", "downsamplers1d", "upsamplers1d", "temp_convs"]}]
lumiere_configs["lumiere_xxl"].pipeline.unet_config = AnimateDiffXLUNetConfig(
    mm_ckpt_path=None,
    down_block_configs=[
        DownBlock3DConfig(
            temp_conv_cls=LumiereTemporalConvLayer,
            add_downsample_1d=True,
        ),
        CrossAttnDownBlock3DConfig(
            temp_conv_cls=LumiereTemporalConvLayer,
            motion_module_cls=TransformerTemporalModel,
            add_downsample_1d=True,
        ),
        CrossAttnDownBlock3DConfig(
            temp_conv_cls=LumiereTemporalConvLayer,
            motion_module_cls=TransformerTemporalModel,
        ),
    ],
    mid_block_config=UNetMidBlock3DCrossAttnConfig(
        temp_conv_cls=LumiereTemporalConvLayer,
        motion_module_cls=TransformerTemporalModel,
    ),
    up_block_configs=[
        CrossAttnUpBlock3DConfig(
            temp_conv_cls=LumiereTemporalConvLayer,
            motion_module_cls=TransformerTemporalModel,
            add_upsample_1d=True,
        ),
        CrossAttnUpBlock3DConfig(
            temp_conv_cls=LumiereTemporalConvLayer,
            motion_module_cls=TransformerTemporalModel,
            add_upsample_1d=True,
        ),
        UpBlock3DConfig(
            temp_conv_cls=LumiereTemporalConvLayer,
        ),
    ],
    num_frames=NUM_FRAMES,
    zero_init=False,
)


lumiere_configs["lumiere_xxl_debug"] = deepcopy(lumiere_configs["lumiere_xl"])
lumiere_configs["lumiere_xxl_debug"].pipeline.call = {
    "height": HEIGHT,
    "width": WIDTH,
    "num_inference_steps": 5,
}
lumiere_configs["lumiere_xxl_debug"].train_data = vcg_val_data
