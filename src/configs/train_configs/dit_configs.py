from copy import deepcopy as copy
from dataclasses import replace
from typing import Dict, Optional
import sys

from ...data import DataConfig
from ...engine import AdamWOptimizerConfig, SchedulerConfig, TrainerConfig, StabilityConfig
from ...models import *
from ...pipelines import *


# diffusers_ckpt_path = "/group/ckpt/diffusers/PixArt-XL-2-512x512"
diffusers_ckpt_path = "/m2v_intern/luoyawen/cache/PixArt-XL-2-512x512"

# just for not use init_noise_sigma, any scheduler but Euler is ok
scheduler_kwargs = {
    "_class_name": "PNDMScheduler",
    "beta_end": 0.02,
    "beta_schedule": "linear",
    "beta_start": 0.0001,
    "variance_type": "learned_range",
}

# clip_ckpt_path = {
#     "clip_tokenizer_path": "/group/ckpt/diffusers/clip-vit-large-patch14",
#     "clip_l_path": "/group/ckpt/diffusers/clip-vit-large-patch14",
#     "clip_g_path": "/group/tianye/models/models--laion--CLIP-ViT-bigG-14-laion2B-39B-b160k/open_clip_pytorch_model.bin",
# }

clip_ckpt_path = {
    "clip_tokenizer_path": "m2v_intern/luoyawen/cache/clip-vit-large-patch14",
    "clip_l_path": "/group/ckpt/diffusers/clip-vit-large-patch14",
    "clip_g_path": "/m2v_intern/luoyawen/cache/open_clip_pytorch_model.bin",
}

# tiny_vae_config = VisualTokenizerConfig(
#     vae_ckpt_path="/video/pansiyuan/model/tiny_vae/tinyvae.ckpt",
#     scaling_factor=0.3414321773546459,
#     encoder_init_dim=32,
#     split_conv_3d=True,
# )

tiny_vae_config = VisualTokenizerConfig(
    vae_ckpt_path="/m2v_intern/luoyawen/cache/tiny_vae/tinyvae.ckpt",
    scaling_factor=0.3414321773546459,
    encoder_init_dim=32,
    split_conv_3d=True,
)


def config(
    fps: float = 15,
    num_frames: int = 77,
    height: int = 256,
    width: int = 256,
    use_navit: bool = True,
    patch_size: int = 2,
    stfit_size: int = 2,
    num_layers: int = 40,
    num_attention_heads: int = 40,
    embed_dim: int = 72,
    use_ema: bool = True,
    use_clip: bool = True,
    qk_norm: bool = True,
    theta: float = 10000.0,
    theta_3d: Optional[float] = None,
    batch_size: int = 1,
    lr: float = 1e-4,
    random_num_frames: bool = False,
    max_grad_norm: float = 1.0,
    training_precision: str = "bf16",
    num_inference_steps: int = 50,
    guidance_scale: float = 12.5,
    negative_prompt: str = "animation, 2d animation, 3d animation, Anime, Cartoon",
    transformer_ckpt_path: Optional[str] = None,
    use_resolution_condition: bool = True,
    use_aspect_ratio_condition: bool = True,
    use_frames_condition: bool = True,
    use_fps_condition: bool = True,
    use_text_condition: bool = True,
    split_conditions: bool = False,
    timestep_shift: float = 1.0,
    cache_dir: Optional[str] = None,
):
    return TrainerConfig(
        num_epochs=1000,
        training_precision=training_precision,
        step_per_save=2000,
        step_per_val=sys.maxsize,   # 不做validation
        val_at_begin=False,
        val_types=["t2v", "t2v_16_9", "t2v_9_16"] if use_navit else ["t2v"],
        use_ema=use_ema,
        step_per_ema=1,
        ema_decay=0.9999,
        ema_start_step=0,
        seed=None,
        delete_deepspeed_weights=False, # 不删除DS模型，防止CUDA OOM
        train_data=DataConfig(
            # path="/video/oujiarong/csv/video_mq_v2_0430.csv",
            # video_path_column="video_ceph_path",
            # path="/ytech_m2v2_hdd/m2v_data/data_version/video_mq_v2_0504_merge.csv" if use_navit else "/video/linke/utils/m2v-video-s1-v0.1_3dvae_0416_5000w.csv",
            # path="/ytech_m2v2_hdd/m2v_data/data_version/video_mq_v2_0504_merge.csv" if use_navit else "/video/linke/utils/m2v-video-s1-v0.1_3dvae_0416_5000w.csv",
            path="/m2v_intern/luoyawen/Dataset/video_mq_v2_0504_merge.csv" if use_navit else "/video/linke/utils/m2v-video-s1-v0.1_3dvae_0416_5000w.csv",
            latent_path_column="vae_path" if use_navit else "3dvae_latents_path",
            caption_column="caption" if use_navit else None,
            t5_prompt_embed_column=None if use_navit else "t5_embed_path",
            clip_prompt_embed_column=None if use_navit else "clip_embed_path" if use_clip else None,
            batch_size=batch_size,
            sample_fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            num_processes=4,
            sample_position="random",
            crop_type=None if use_navit else "center",
            shuffle=False,
            spatial_token_merge_size=stfit_size,
            random_num_frames=random_num_frames,
            cache_dir=cache_dir,
        ),
        aux_data=DataConfig(
            # path="/video/oujiarong/csv/image_hq_v2_0430.csv",
            # image_path_column="new_path",
            # caption_column="caption",
            # path="/video/oujiarong/csv/image_hq_v2_0503.csv" if use_navit else "/ytech_m2v_hdd/yanghaotian/m2v-image-s2-v0.1_t5_clip_vae256_3dvae256.csv",
            path="/m2v_intern/luoyawen/Dataset/image_hq_v2_0503.csv" if use_navit else "/ytech_m2v_hdd/yanghaotian/m2v-image-s2-v0.1_t5_clip_vae256_3dvae256.csv",
            latent_path_column="vae_ceph_path" if use_navit else None,
            image_latent_path_column=None if use_navit else "3dvae_latents_path",
            t5_prompt_embed_column="t5_ceph_path" if use_navit else "t5_embed_path",
            clip_prompt_embed_column="clip_ceph_path" if use_navit else "clip_embed_path" if use_clip else None,
            batch_size=batch_size * ((num_frames - 1) // 4 + 1),
            num_frames=1,
            height=height,
            width=width,
            num_processes=4,
            crop_type=None if use_navit else "center",
            shuffle=False,
            spatial_token_merge_size=stfit_size,
            cache_dir=cache_dir,
        ),
        joint_train_prob=1 / (1 + ((num_frames - 1) // 4 + 1)),
        val_data=DataConfig(
            # path="/ytech_m2v2_hdd/houliang/m2v-diffusers/data/sora_prompt.csv",
            # caption_column="prompt",
            # path="/video/houliang/m2v-diffusers/data/val_data.csv",
            path="/m2v_intern/luoyawen/Dataset/val_data.csv",
            caption_column="llama_caption",
            num_samples=16,
            sample_fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            shuffle=False,
            spatial_token_merge_size=stfit_size,
            cache_dir=cache_dir,
        ),
        valmetrics_data=DataConfig(
            # path="/video/houliang/m2v-diffusers/data/new_qa_test.csv",
            # caption_column="response",
            # path="/video/yht/qa_recaption_llama8b_wo_sft_raw_0509_v1.csv",
            # path="/video/yht/qa150_recaption_llama8b_wo_sft_raw_0509_v1.csv",
            path="/m2v_intern/luoyawen/Dataset/qa150_recaption_llama8b_wo_sft_raw_0509_v1.csv",
            caption_column="llama_caption",
            sample_fps=fps,
            num_frames=num_frames,
            height=height,
            width=width,
            shuffle=False,
            spatial_token_merge_size=stfit_size,
            cache_dir=cache_dir,
        ),
        trainable_modules=["transformer"],
        optimizer=AdamWOptimizerConfig(fused=True, lr=lr, max_grad_norm=max_grad_norm),
        scheduler=SchedulerConfig(),
        stability=StabilityConfig(stability_protection=True),
        pipeline=T2VFlowPipelineConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=clip_ckpt_path if use_clip else None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLModelConfig(
                num_layers=num_layers,
                num_attention_heads=num_attention_heads,
                cross_attention_dim=num_attention_heads * embed_dim,
                num_frames=num_frames,
                height=int(height * 1.3125) if use_navit else height,
                width=int(width * 1.3125) if use_navit else width,
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=patch_size,
                in_channels=8,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=theta,
                use_1d_rope=True,
                theta_1d_rope=theta,
                use_temp_attn=True,
                image_temp_attn=True,
                temporal_attention_config=TemporalAttentionConfig(
                    attn_type="stfit",
                    theta_3d_rope=theta if theta_3d is None else theta_3d,
                    stfit_patch_size=(1, stfit_size, stfit_size),
                ) if stfit_size is not None else None,
                qk_norm=qk_norm,
                transformer_ckpt_path=transformer_ckpt_path,
                # NOTE: 拆分spatial和temporal conditions
                use_resolution_condition=use_resolution_condition,
                use_aspect_ratio_condition=use_aspect_ratio_condition,
                use_frames_condition=use_frames_condition,
                use_fps_condition=use_fps_condition,
                use_text_condition=use_text_condition,
                split_conditions=split_conditions,
            ),
            call={
                "num_frames": num_frames,
                "height": height,
                "width": width,
                "num_inference_steps": num_inference_steps,
                "negative_prompt": negative_prompt,
                "guidance_scale": guidance_scale,
                "fps": fps,
            },
            timestep_shift=timestep_shift,
        ),
    )


dit_configs: Dict[str, TrainerConfig] = {}


'''
    1B model Config 
'''

dit_configs["1b_512"] = config(
    fps=15,
    num_frames=77,
    height=512,
    width=512,
    use_navit=True, # 可变分辨率
    stfit_size=2,   # 使用TokenMerge 2x2
    num_layers=28,  # 1B模型
    num_attention_heads=16, # 1B模型
    embed_dim=72,   # 1B模型
    use_ema=True,   # 使用EMA
    use_clip=False, # 不使用CLIP
    qk_norm=False,   # 使用QKNorm
    theta=100.0,    # ViT RoPE theta (https://arxiv.org/abs/2403.13298)
    batch_size=8,
    lr=5e-5,
    max_grad_norm=0.1,
    random_num_frames=False,    # 视频随机帧数
    timestep_shift=1.0,
    cache_dir="/m2v_intern/luoyawen/cache/csv",
)

##  ========= data ========= 
dit_configs["1b_512"].train_data = replace(
    dit_configs["1b_512"].train_data,
    path="/ytech_m2v2_hdd/m2v_data/data_version/video_hq_v3_0527.csv",
    video_path_column="video_path",
    caption_column="caption",
    latent_path_column=None,
    t5_prompt_embed_column=None,
    clip_prompt_embed_column=None,
    # shuffle=True,
)
dit_configs["1b_512"].aux_data = replace(
    dit_configs["1b_512"].aux_data,
    path="/ytech_m2v2_hdd/m2v_data/data_version/image_hq_v2_0510.csv",
    image_path_column=None,
    caption_column=None,
    latent_path_column=None,
    t5_prompt_embed_column=None,
    clip_prompt_embed_column=None,
)

##  ========= model ========= 

dit_configs["1b_512"].pipeline = T2VFlowPipelineConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLModelConfig(
                num_layers=28,
                num_attention_heads=16,
                cross_attention_dim=16 * 72,
                num_frames=77,
                height=int(512 * 1.3125),       # use navit will need to * 1.3125
                width=int(512 * 1.3125),        # use navit will need to * 1.3125
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                in_channels=8,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100.0,
                use_1d_rope=True,
                theta_1d_rope=100.0,
                use_temp_attn=True,
                image_temp_attn=True,
                transformer_ckpt_path=[
                    "/m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/mvb_1b_f77_distill_ema_merged.ckpt",
                    ],
                temporal_attention_config=TemporalAttentionConfig(
                attn_type="stfit",
                use_3d_rope=True,
                theta_3d_rope=10000.0,
                stfit_latent_dims_scale=1,
                stfit_patch_size=(2, 2),
                ), 
                qk_norm=False,
                # NOTE: 拆分spatial和temporal conditions
                use_additional_conditions=False,
                use_aspect_ratio_condition=False,
                use_fps_condition=False,
                use_frames_condition=True,
                use_resolution_condition=True,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
                "guidance_scale": 12.5,
                "fps": 15,
            },
)

##  ========= Hyper Parameters ========= 
dit_configs["1b_512"].step_per_save = 1000
dit_configs["1b_512"].step_per_val = sys.maxsize
## add by lyw
dit_configs["1b_512"].pipeline.max_sequence_length = 512
dit_configs["1b_512"].joint_train_prob = 0
dit_configs["1b_512"].save_ema = False  # 防止超时

'''
    1b base model Config 
'''

dit_configs["1b_base"] = config(
    fps=15,
    num_frames=77,
    height=512,
    width=512,
    use_navit=True, # 可变分辨率
    stfit_size=2,   # 使用TokenMerge 2x2
    num_layers=40,  # 10B模型
    num_attention_heads=40, # 10B模型
    embed_dim=72,   # 10B模型
    use_ema=True,   # 使用EMA
    use_clip=False, # 不使用CLIP
    qk_norm=True,   # 使用QKNorm
    theta=100.0,    # ViT RoPE theta (https://arxiv.org/abs/2403.13298)
    batch_size=1,
    lr=5e-5,
    max_grad_norm=0.1,
    random_num_frames=False,    # 视频随机帧数
    timestep_shift=5.0,
    cache_dir="/m2v_intern/luoyawen/cache/csv",
)

##  ========= data ========= 
dit_configs["1b_base"].train_data = replace(
    dit_configs["1b_base"].train_data,
    # path="/ytech_m2v2_hdd/m2v_data/data_version/video_hq_v3_0527.csv",
    path="/m2v_intern/public_datasets/Camera_Dataset/Csv/0413_recam_all_filtered.csv",
    video_path_column="video_path",
    caption_column="caption",
    ref_path_column="ref_video_path",
    content_ref_path_column="content_video_path",
    latent_path_column=None,
    t5_prompt_embed_column=None,
    clip_prompt_embed_column=None,
    crop_type=None
    # shuffle=True,
)

##  ========= model ========= 

dit_configs["1b_base"].pipeline = T2VFlowPipelineConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLModelConfig(
                # num_layers=1,
                num_layers=40,
                num_attention_heads=40,
                cross_attention_dim=40 * 72,
                num_frames=77,
                height=int(512 * 1.3125),       # use navit will need to * 1.3125
                width=int(512 * 1.3125),        # use navit will need to * 1.3125
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                in_channels=8,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100.0,
                use_1d_rope=True,
                theta_1d_rope=100.0,
                use_temp_attn=True,
                image_temp_attn=True,
                transformer_ckpt_path=[
                    "/m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/mvb_1b_f77_distill_ema_merged.ckpt",
                    ],
                temporal_attention_config=TemporalAttentionConfig(
                attn_type="stfit",
                use_3d_rope=True,
                theta_3d_rope=10000.0,
                stfit_latent_dims_scale=1,
                stfit_patch_size=(2, 2),
                ), 
                qk_norm=True,
                # NOTE: 拆分spatial和temporal conditions
                use_additional_conditions=False,
                use_aspect_ratio_condition=False,
                use_fps_condition=False,
                use_frames_condition=True,
                use_resolution_condition=True,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
                "guidance_scale": 12.5,
                "fps": 15,
            },
)

dit_configs["1b_base"].step_per_save = 1000
dit_configs["1b_base"].step_per_val = sys.maxsize
dit_configs["1b_base"].pipeline.transformer_config.transformer_ckpt_path = "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt"
dit_configs["1b_base"].joint_train_prob = 0
dit_configs["1b_base"].pipeline.timestep_shift = 5.0
dit_configs["1b_base"].pipeline.t2v_ratio = 0.5 # half i2v, half t2v
dit_configs["1b_base"].gradient_accumulation_steps = 4 # half i2v, half t2v
# max sequence length


## ============== camclone and recamclone ============= ##
dit_configs["1b_camclonemaster"] = copy(dit_configs["1b_512"])
dit_configs["1b_camclonemaster"].train_data = replace(   
    dit_configs["1b_camclonemaster"].train_data,
    path="/m2v_intern/public_datasets/Camera_Dataset/Csv/0413_recam_all_filtered.csv", 
    video_path_column="video_path",
    caption_column="caption",
    ref_path_column="ref_video_path",
    content_ref_path_column="content_video_path",
    latent_path_column=None,
    t5_prompt_embed_column=None,
    clip_prompt_embed_column=None,
    batch_size=2,
    num_processes=1,
)
dit_configs["1b_camclonemaster"].pipeline = T2VFlowPipelineCamCloneMasterConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLModelConfig(
                num_layers=1,
                # num_layers=40,
                num_attention_heads=40,
                cross_attention_dim=40 * 72,
                num_frames=77,
                height=int(512 * 1.3125),       # use navit will need to * 1.3125
                width=int(512 * 1.3125),        # use navit will need to * 1.3125
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                in_channels=8,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100.0,
                use_1d_rope=True,
                theta_1d_rope=100.0,
                use_temp_attn=True,
                image_temp_attn=True,
                transformer_ckpt_path=[
                    "/m2v_intern/yuanziyang/share/mvb_t2v_1b_distill_ckpt/mvb_1b_f77_distill_ema_merged.ckpt",
                    ],
                temporal_attention_config=TemporalAttentionConfig(
                attn_type="stfit",
                use_3d_rope=True,
                theta_3d_rope=10000.0,
                stfit_latent_dims_scale=1,
                stfit_patch_size=(2, 2),
                ), 
                qk_norm=True,
                # NOTE: 拆分spatial和temporal conditions
                use_additional_conditions=False,
                use_aspect_ratio_condition=False,
                use_fps_condition=False,
                use_frames_condition=True,
                use_resolution_condition=True,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
                "guidance_scale": 12.5,
                "fps": 15,
            },
)


dit_configs["1b_camclonemaster"].pipeline.transformer_config.transformer_ckpt_path = "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt"
dit_configs["1b_camclonemaster"].gradient_accumulation_steps = 1
dit_configs["1b_camclonemaster"].trainable_modules=["transformer"]
dit_configs["1b_camclonemaster"].trainable_only_layers = "attnt"
dit_configs["1b_camclonemaster"].pipeline.max_sequence_length = 512
dit_configs["1b_camclonemaster"].step_per_save = 1000

dit_configs["1b_camclonemaster_bs_16"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_bs_16"].step_per_save = 1000

dit_configs["1b_camclonemaster_bs_32"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_bs_32"].gradient_accumulation_steps = 2

dit_configs["1b_camclonemaster_bs_64"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_bs_64"].gradient_accumulation_steps = 4



dit_configs["1b_camclonemaster_bs_node_4"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_bs_node_4"].train_data.batch_size = 2

dit_configs["1b_camclonemaster_bs_96"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_bs_96"].gradient_accumulation_steps = 6

dit_configs["1b_camclonemaster_node_12"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_node_12"].step_per_save = 250
dit_configs["1b_camclonemaster_node_12"].train_data.batch_size = 1

dit_configs["1b_camclonemaster_bs_node_3"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_bs_node_3"].train_data.batch_size = 2
dit_configs["1b_camclonemaster_bs_node_3"].gradient_accumulation_steps = 2

dit_configs["1b_camclonemaster_node_12_ablation_finetune_transformer"] = copy(dit_configs["1b_camclonemaster_node_12"])
dit_configs["1b_camclonemaster_node_12_ablation_finetune_transformer"].trainable_only_layers = None

## =========== Rebuttal: Ablation in Dataset ============ ##
dit_configs["1b_camclonemaster_rebuttal_dataset_half"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_rebuttal_dataset_half"].step_per_save = 500
dit_configs["1b_camclonemaster_rebuttal_dataset_half"].train_data.path = "/m2v_intern/public_datasets/Camera_Dataset/Csv/0715_rebuttal_half.csv"
dit_configs["1b_camclonemaster_rebuttal_dataset_half"].train_data.batch_size = 2

dit_configs["1b_camclonemaster_rebuttal_dataset_delete_complex"] = copy(dit_configs["1b_camclonemaster_rebuttal_dataset_half"])
dit_configs["1b_camclonemaster_rebuttal_dataset_delete_complex"].train_data.path = "/m2v_intern/public_datasets/Camera_Dataset/Csv/0715_rebuttal_delete_complex.csv"

## =========== Rebuttal: Ablation for RT ==========##
dit_configs["1b_camclonemaster_rebuttal_RT_control"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_rebuttal_RT_control"].pipeline = T2VFlowCamCloneRTPipelineConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLRtModelConfig(
                # num_layers=1,
                num_layers=40,
                num_attention_heads=40,
                cross_attention_dim=40 * 72,
                num_frames=77,
                height=int(512 * 1.3125),       # use navit will need to * 1.3125
                width=int(512 * 1.3125),        # use navit will need to * 1.3125
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                in_channels=8,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100.0,
                use_1d_rope=True,
                theta_1d_rope=100.0,
                use_temp_attn=True,
                image_temp_attn=True,
                transformer_ckpt_path=[
                    "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt",
                    ],
                temporal_attention_config=TemporalAttentionConfig(
                attn_type="stfit",
                use_3d_rope=True,
                theta_3d_rope=10000.0,
                stfit_latent_dims_scale=1,
                stfit_patch_size=(2, 2),
                ), 
                qk_norm=True,
                # NOTE: 拆分spatial和temporal conditions
                use_additional_conditions=False,
                use_aspect_ratio_condition=False,
                use_fps_condition=False,
                use_frames_condition=True,
                use_resolution_condition=True,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
                "guidance_scale": 12.5,
                "fps": 15,
            })
dit_configs["1b_camclonemaster_rebuttal_RT_control"].train_data.path = "/m2v_intern/public_datasets/Camera_Dataset/Csv/0715_rebuttal_RT_with_cam_path_filtered.csv"
dit_configs["1b_camclonemaster_rebuttal_RT_control"].train_data.cam_rt_path_column = "cam_path"
dit_configs["1b_camclonemaster_rebuttal_RT_control"].train_data.batch_size = 3
dit_configs["1b_camclonemaster_rebuttal_RT_control"].trainable_only_layers = "attnt_projector"
dit_configs["1b_camclonemaster_rebuttal_RT_control"].pipeline.max_sequence_length = 512

dit_configs["1b_camclonemaster_debug"] = copy(dit_configs["1b_camclonemaster_rebuttal_RT_control"])
dit_configs["1b_camclonemaster_debug"].step_per_save = 100
dit_configs["1b_camclonemaster_debug"].pipeline.transformer_config.num_layers =1
dit_configs["1b_camclonemaster_debug"].train_data.path = "/m2v_intern/public_datasets/Camera_Dataset/Csv/0715_rebuttal_RT_with_cam_path_filtered.csv"
dit_configs["1b_camclonemaster_debug"].train_data.cam_rt_path_column = "cam_path"

## =========== for channel concat ============= ##
dit_configs["1b_camclonemaster_node_12_ablation_channel_concat"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_node_12_ablation_channel_concat"].pipeline = T2VFlowPipelineChannelConcatConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLModelConfig(
                # num_layers=1,
                num_layers=40,
                num_attention_heads=40,
                cross_attention_dim=40 * 72,
                num_frames=77,
                height=int(512 * 1.3125),       # use navit will need to * 1.3125
                width=int(512 * 1.3125),        # use navit will need to * 1.3125
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                in_channels=24,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100.0,
                use_1d_rope=True,
                theta_1d_rope=100.0,
                use_temp_attn=True,
                image_temp_attn=True,
                transformer_ckpt_path=[
                    "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt",
                    ],
                temporal_attention_config=TemporalAttentionConfig(
                attn_type="stfit",
                use_3d_rope=True,
                theta_3d_rope=10000.0,
                stfit_latent_dims_scale=1,
                stfit_patch_size=(2, 2),
                ), 
                qk_norm=True,
                # NOTE: 拆分spatial和temporal conditions
                use_additional_conditions=False,
                use_aspect_ratio_condition=False,
                use_fps_condition=False,
                use_frames_condition=True,
                use_resolution_condition=True,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
                "guidance_scale": 12.5,
                "fps": 15,
            })
dit_configs["1b_camclonemaster_node_12_ablation_channel_concat"].pipeline.max_sequence_length = 512
dit_configs["1b_camclonemaster_node_12_ablation_channel_concat"].train_data.batch_size = 2

## ========== only concat at temporal layer ========== ##
dit_configs["1b_camclonemaster_node_12_ablation_only_temporal"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_node_12_ablation_only_temporal"].pipeline = T2VFlowPipelineOnlyTemporalConfig(
            ckpt_path=diffusers_ckpt_path,
            vae_config=tiny_vae_config,
            clip_ckpt_path=None,
            proportion_empty_prompts=0.1,
            scheduler_kwargs=scheduler_kwargs,
            transformer_config=TransformerXLModelOnlyTemporalConfig(
                # num_layers=1,
                num_layers=40,
                num_attention_heads=40,
                cross_attention_dim=40 * 72,
                num_frames=77,
                height=int(512 * 1.3125),       # use navit will need to * 1.3125
                width=int(512 * 1.3125),        # use navit will need to * 1.3125
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                in_channels=8,
                out_channels=8,
                from_scratch=True,
                gradient_checkpointing=True,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100.0,
                use_1d_rope=True,
                theta_1d_rope=100.0,
                use_temp_attn=True,
                image_temp_attn=True,
                transformer_ckpt_path=[
                    "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt",
                    ],
                temporal_attention_config=TemporalAttentionConfig(
                attn_type="stfit",
                use_3d_rope=True,
                theta_3d_rope=10000.0,
                stfit_latent_dims_scale=1,
                stfit_patch_size=(2, 2),
                ), 
                qk_norm=True,
                # NOTE: 拆分spatial和temporal conditions
                use_additional_conditions=False,
                use_aspect_ratio_condition=False,
                use_fps_condition=False,
                use_frames_condition=True,
                use_resolution_condition=True,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
                "guidance_scale": 12.5,
                "fps": 15,
            })
dit_configs["1b_camclonemaster_node_12_ablation_only_temporal"].pipeline.max_sequence_length = 512
dit_configs["1b_camclonemaster_node_12_ablation_only_temporal"].train_data.batch_size = 1

## =============== ablation with controlnet =================== ##
dit_configs["1b_camclonemaster_node_12_ablation_controlnet"] = copy(dit_configs["1b_camclonemaster"])
dit_configs["1b_camclonemaster_node_12_ablation_controlnet"].pipeline = T2VFlowPipelineControlNetConfig(
    ckpt_path=diffusers_ckpt_path,
    vae_config=tiny_vae_config,
    clip_ckpt_path=None,
    proportion_empty_prompts=0.1,
    scheduler_kwargs=scheduler_kwargs,
    transformer_config=TransformerXLModelControlNetConfig(
        # num_layers=1,
        num_layers=40,
        num_attention_heads=40,
        cross_attention_dim=40 * 72,
        num_frames=77,
        height=int(512 * 1.3125),
        width=int(512 * 1.3125),
        vae_temporal_scale_factor=4,
        vae_scale_factor=8,
        patch_size=2,
        in_channels=8,
        out_channels=8,
        from_scratch=True,
        gradient_checkpointing=True,
        use_flash_attn=True,
        use_2d_rope=True,
        theta_2d_rope=100.0,
        use_1d_rope=True,
        theta_1d_rope=100.0,
        use_temp_attn=True,
        image_temp_attn=True,
        transformer_ckpt_path=[
            "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt",
        ],
        temporal_attention_config=TemporalAttentionConfig(
            attn_type="stfit",
            use_3d_rope=True,
            theta_3d_rope=10000.0,
            stfit_latent_dims_scale=1,
            stfit_patch_size=(2, 2),
        ),
        qk_norm=True,
        use_additional_conditions=False,
        use_aspect_ratio_condition=False,
        use_fps_condition=False,
        use_frames_condition=True,
        use_resolution_condition=True,
        use_text_condition=False,
        split_conditions=True,
    ),
    control_transformer_config=TransformerXLModelControlNet_ControlConfig(
        # num_layers=1,
        num_layers=10,
        num_attention_heads=40,
        cross_attention_dim=40 * 72,
        num_frames=77,
        height=int(512 * 1.3125),
        width=int(512 * 1.3125),
        vae_temporal_scale_factor=4,
        vae_scale_factor=8,
        patch_size=2,
        in_channels=8,
        out_channels=8,
        from_scratch=True,
        gradient_checkpointing=True,
        use_flash_attn=True,
        use_2d_rope=True,
        theta_2d_rope=100.0,
        use_1d_rope=True,
        theta_1d_rope=100.0,
        use_temp_attn=True,
        image_temp_attn=True,
        transformer_ckpt_path=[
            "/m2v_intern/luoyawen/Coding/Kelin/m2v_camclone_v2/exps/0000-1b-camclone-base/1b-camclone-base.ckpt",
        ],
        temporal_attention_config=TemporalAttentionConfig(
            attn_type="stfit",
            use_3d_rope=True,
            theta_3d_rope=10000.0,
            stfit_latent_dims_scale=1,
            stfit_patch_size=(2, 2),
        ),
        qk_norm=True,
        use_additional_conditions=False,
        use_aspect_ratio_condition=False,
        use_fps_condition=False,
        use_frames_condition=True,
        use_resolution_condition=True,
        use_text_condition=False,
        split_conditions=True,
    ),
    call={
        "num_frames": 77,
        "height": 512,
        "width": 512,
        "num_inference_steps": 50,
        "negative_prompt": "animation, 2d animation, 3d animation, Anime, Cartoon",
        "guidance_scale": 12.5,
        "fps": 15,
    },
)
dit_configs["1b_camclonemaster_node_12_ablation_controlnet"].train_data.batch_size = 1
dit_configs["1b_camclonemaster_node_12_ablation_controlnet"].trainable_modules = ["conditions_transformer"]
