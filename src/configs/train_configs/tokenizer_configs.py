from copy import deepcopy
from typing import Dict

from ...data import DataConfig
from ...engine import AdamOptimizerConfig, AdamWOptimizerConfig, SchedulerConfig, StabilityConfig, TrainerConfig
from ...engine.trainer_gan import TrainerGANConfig
from ...models import *
from ...pipelines import *

HEIGHT, WIDTH = 128, 128

NUM_FRAMES = 17

image_traindata = DataConfig(
    path="/group/dengyufan/repo/magvit2-pytorch/img_align_celeba.csv",
    height=HEIGHT,
    width=WIDTH,
    image_path_column="image_path",
    caption_column=None,
    video_path_column=None,
    batch_size=32,
    num_frames=1,
    random_flip=True,
    crop_type="random",
)

image_valdata = DataConfig(
    path="/group/dengyufan/repo/magvit2-pytorch/val.csv",
    height=HEIGHT,
    width=WIDTH,
    image_path_column="image_path",
    caption_column=None,
    video_path_column=None,
    batch_size=1,
    num_samples=8,
    num_frames=1,
    shuffle=False,
)

video_traindata = DataConfig(
    path="/video/yht/data_0204_sel_ms_top75.csv",
    height=HEIGHT,
    width=WIDTH,
    caption_column=None,
    batch_size=1,
    random_flip=True,
    sample_position="random",
    crop_type="random",
    sample_type="fps",
    sample_fps=30,
    num_frames=17,
)

video_valdata = DataConfig(
    path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
    video_path_column="video_path",
    height=HEIGHT,
    width=WIDTH,
    caption_column=None,
    batch_size=1,
    num_samples=8,
    sample_type="fps",
    shuffle=False,
    sample_fps=30,
    num_frames=17,
)


vtokenizer_configs: Dict[str, TrainerConfig] = {}

vtokenizer_configs["visual_tokenizer_wo_gan"] = TrainerConfig(
    stability=StabilityConfig(stability_protection=False),
    experiment_name="visual_tokenizer",
    train_data=image_traindata,
    scheduler=SchedulerConfig(name="cosine", num_warmup_steps_rate=0.05),
    optimizer=AdamWOptimizerConfig(lr=1e-4, betas=(0, 0.99)),
    ema_decay=0.999,
    trainable_modules=["visual_tokenizer"],
    pipeline=VisualTokenizerPipelineConfig(
        visual_tokenizer=VisualTokenizerConfig(
            output_conv_kernel_size=(1, 3, 3),
            gradient_checkpointing=True,
        ),
        discriminator=DiscriminatorConfig(),
        loss_weights={"l1_loss": 1.0, "perceptual_loss": 1.0, "kl_loss": 0.000001},
    ),
    data_per_val=10240,
    data_per_save=102400,
    val_data=image_valdata,
    val_types=["gt", "vtoken"],
)

vtokenizer_configs["visual_tokenizer"] = TrainerConfig(
    stability=StabilityConfig(stability_protection=False),
    experiment_name="visual_tokenizer",
    train_data=image_traindata,
    scheduler=SchedulerConfig(name="cosine", num_warmup_steps_rate=0.05),
    optimizer=AdamOptimizerConfig(lr=1e-4, betas=(0, 0.99)),
    ema_decay=0.999,
    trainable_modules=["visual_tokenizer", "discriminator"],
    pipeline=VisualTokenizerPipelineConfig(
        visual_tokenizer=VisualTokenizerConfig(
            output_conv_kernel_size=(1, 3, 3),
        ),
        discriminator=DiscriminatorConfig(),
        loss_weights={"l1_loss": 1.0, "perceptual_loss": 1.0, "kl_loss": 0.000001, "gan_loss": 0.5},
    ),
    data_per_val=400,
    data_per_save=102400,
    val_data=image_valdata,
    val_types=["gt", "vtoken"],
)

vtokenizer_configs["visual_tokenizer_video"] = deepcopy(vtokenizer_configs["visual_tokenizer"])
vtokenizer_configs["visual_tokenizer_video"].pipeline.visual_tokenizer.output_conv_kernel_size = (3, 3, 3)
vtokenizer_configs["visual_tokenizer_video"].train_data = video_traindata
vtokenizer_configs["visual_tokenizer_video"].val_data = video_valdata


vtokenizer_configs["visual_tokenizer_video_wogan"] = deepcopy(vtokenizer_configs["visual_tokenizer_wo_gan"])
vtokenizer_configs["visual_tokenizer_video_wogan"].pipeline.visual_tokenizer.output_conv_kernel_size = (3, 3, 3)
vtokenizer_configs["visual_tokenizer_video_wogan"].train_data = video_traindata
vtokenizer_configs["visual_tokenizer_video_wogan"].val_data = video_valdata


vtokenizer_configs["visual_tokenizer_gan"] = TrainerGANConfig(
    experiment_name="visual_tokenizer",
    step_per_ema=100,
    ema_start_step=1000,
    train_data=image_traindata,
    scheduler=SchedulerConfig(name="cosine", num_warmup_steps_rate=0.05),
    optimizer=AdamWOptimizerConfig(lr=1e-4),
    ema_decay=0.993,
    trainable_modules=[{"visual_tokenizer": ["decoder"]}, "discriminator"],
    # trainable_modules=["visual_tokenizer", "discriminator"],
    pipeline=VisualTokenizerPipelineConfig(
        visual_tokenizer=VisualTokenizerConfig(
            output_conv_kernel_size=(1, 3, 3),
        ),
        discriminator=DiscriminatorConfig(
            use_pixel_discriminator=True,
        ),
        loss_weights={
            "l1_loss": 1.0,
            "perceptual_loss": 10.0,
            "kl_loss": 0.000001,
            "gan_loss": 1.0,
            "feature_matching_loss": 1.0,
        },
    ),
    data_per_val=400,
    data_per_save=102400,
    val_data=image_valdata,
    val_types=["gt", "vtoken"],
)

vtokenizer_configs["visual_tokenizer_gan_video"] = deepcopy(vtokenizer_configs["visual_tokenizer_gan"])
vtokenizer_configs["visual_tokenizer_gan_video"].pipeline.visual_tokenizer.output_conv_kernel_size = (3, 3, 3)
vtokenizer_configs["visual_tokenizer_gan_video"].train_data = video_traindata
vtokenizer_configs["visual_tokenizer_gan_video"].val_data = video_valdata


vtokenizer_configs["visual_tokenizer_2p1d_gan_video"] = deepcopy(vtokenizer_configs["visual_tokenizer_gan_video"])
vtokenizer_configs["visual_tokenizer_2p1d_gan_video"].pipeline.visual_tokenizer.output_conv_kernel_size = (1, 3, 3)
vtokenizer_configs["visual_tokenizer_2p1d_gan_video"].pipeline.visual_tokenizer.use_2plus1d = True

vtokenizer_configs["visual_tokenizer_video_2p1d_wogan"] = deepcopy(vtokenizer_configs["visual_tokenizer_video_wogan"])
vtokenizer_configs["visual_tokenizer_video_2p1d_wogan"].pipeline.visual_tokenizer.output_conv_kernel_size = (1, 3, 3)
vtokenizer_configs["visual_tokenizer_video_2p1d_wogan"].pipeline.visual_tokenizer.use_2plus1d = True
