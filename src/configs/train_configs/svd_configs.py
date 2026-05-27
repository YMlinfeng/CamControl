from copy import deepcopy
from dataclasses import replace
from typing import Dict

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig
from ...models import *
from ...pipelines import *

# 必须是64的倍数, [64, 128, 192, 256, 320, 384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216]
# HEIGHT, WIDTH = 576, 1024
# HEIGHT, WIDTH = 320, 512
HEIGHT, WIDTH = 256, 448
# HEIGHT, WIDTH = 320, 576
# HEIGHT, WIDTH = 128, 256


SAMPLE_FPS = 7
NUM_FRAMES = 14


vcg_traindata = DataConfig(
    path="/video/vcg/final_list/get_all/vcg_motion_bucket_top75.csv",  # "/video/vcg/final_list/get_all/vcg_all_llama2.csv",
    caption_column="llama2_caption",
    batch_size=10,
    height=HEIGHT,
    width=WIDTH,
    num_frames=NUM_FRAMES,
    sample_fps=SAMPLE_FPS,
    motion_bucket_id_column="motion_bucket",
    sample_position="random",
    sample_type="fps",
)
vcg_valdata = DataConfig(
    path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
    height=HEIGHT,
    width=WIDTH,
    batch_size=2,
    num_samples=8,
    num_frames=NUM_FRAMES,
    sample_fps=SAMPLE_FPS,
    shuffle=False,
)
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
    "sigma_max": 700.0,
    "sigma_min": 0.002,
    "skip_prk_steps": True,
    "steps_offset": 1,
    "timestep_spacing": "leading",
    "timestep_type": "continuous",
    "trained_betas": None,
    "use_karras_sigmas": True,
}

svd_configs: Dict[str, TrainerConfig] = {}


svd_configs["i2v_svd_xt"] = TrainerConfig(
    experiment_name="i2v_svd",
    train_data=vcg_traindata,
    optimizer=AdamWOptimizerConfig(lr=1e-5),
    trainable_modules=["unet"],
    pipeline=I2VSVDPipelineConfig(
        ckpt_path="/group/ckpt/diffusers/stable-video-diffusion-img2vid-xt",
        unet_config=SVDUNetConfig(
            num_frames=NUM_FRAMES,
        ),
        scheduler_kwargs=svd_train_scheduler,
        call={
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": 25,
        },
    ),
    val_data=vcg_valdata,
    val_types=["gt", "i2v"],
)

svd_configs["i2v_svd"] = deepcopy(svd_configs["i2v_svd_xt"])
svd_configs["i2v_svd"].pipeline.ckpt_path = "/group/ckpt/diffusers/stable-video-diffusion-img2vid"

svd_configs["i2v_svd_scale"] = deepcopy(svd_configs["i2v_svd"])
svd_configs["i2v_svd_scale"].pipeline.unet_config.use_scalelong = True


svd_configs["t2v_svd"] = TrainerConfig(
    experiment_name="t2v_svd",
    train_data=replace(vcg_traindata, batch_size=8, shuffle=True),
    optimizer=AdamWOptimizerConfig(lr=1e-5),
    num_epochs=10,
    trainable_modules=["unet"],
    pipeline=T2VSVDPipelineConfig(
        ckpt_path="/group/ckpt/diffusers/stable-video-diffusion-img2vid",
        text_encoder_config=ClipTextEncoderConfig(
            tokenizer_ckpt_path="/group/ckpt/diffusers/stable-diffusion-2-1/tokenizer",
            text_encoder_ckpt_path="/group/ckpt/diffusers/stable-diffusion-2-1/text_encoder",
        ),
        unet_config=SVDUNetConfig(
            num_frames=NUM_FRAMES,
            in_channels=4,
            projection_class_embeddings_input_dim=512,
            ignore_mismatched_sizes=True,
            cross_attention_dim=1024,
        ),
        scheduler_kwargs=svd_train_scheduler,
        call={
            "height": HEIGHT,
            "width": WIDTH,
            "num_frames": NUM_FRAMES,
            "num_inference_steps": 25,
            "guidance_scale": 12.5,
        },
    ),
    val_data=replace(vcg_valdata, caption_column="caption"),
    val_types=["gt", "t2v"],
)

svd_configs["t2v_svd_uae"] = deepcopy(svd_configs["t2v_svd"])
svd_configs["t2v_svd_uae"].experiment_name = "t2v_svd_uae"
svd_configs["t2v_svd_uae"].pipeline.text_encoder_config = UAETextEncoderConfig(
    tokenizer_ckpt_path="/group/ckpt/assets/UAE-Large-V1",
    text_encoder_ckpt_path="/group/ckpt/assets/UAE-Large-V1",
)
