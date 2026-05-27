import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Type, TypeVar, Union

from ..utils import NO_UPDATE, Updatable, eval_setup, parse_config_string
from .base_config import PrintableConfig, dataclass
from .train_configs import train_configs


@dataclass
class TestConfig(PrintableConfig):
    """Configuration for Test instantiation
    NO_UPDATE 指代 不做更新，具体见PrintableConfig实现
    """

    train_configs = train_configs

    train_name: Updatable[str] = NO_UPDATE
    yml_path: Updatable[str] = NO_UPDATE

    output_dir: Updatable[str] = "test"
    experiment_name: Updatable[str] = NO_UPDATE

    num_frames: Updatable[int] = NO_UPDATE
    sample_fps: Updatable[int] = NO_UPDATE
    train_len: Updatable[int] = NO_UPDATE
    height: Updatable[int] = NO_UPDATE
    width: Updatable[int] = NO_UPDATE
    crop_type: Updatable[str] = NO_UPDATE

    ckpt_path: Updatable[str] = NO_UPDATE
    unet_ckpt_path: Updatable[str] = NO_UPDATE
    transformer_ckpt_path: Updatable[str] = NO_UPDATE
    use_flash_attn: Updatable[bool] = NO_UPDATE
    diffusers_vae_ckpt_path: Updatable[str] = NO_UPDATE
    temporal_vae_ckpt_path: Updatable[str] = NO_UPDATE
    mm_ckpt_path: Updatable[str] = NO_UPDATE

    target: Updatable[str] = NO_UPDATE
    endfix_prompt: Updatable[str] = NO_UPDATE
    negative_prompt: Updatable[str] = NO_UPDATE
    seed: Updatable[int] = NO_UPDATE
    call: Updatable[str] = NO_UPDATE  # "key1=value1,key2=value2"

    data_mode: Literal["video", "image"] = "video"
    csv_path: Updatable[str] = NO_UPDATE
    index_column: Updatable[str] = NO_UPDATE
    video_path_column: Updatable[str] = NO_UPDATE
    image_path_column: Updatable[str] = NO_UPDATE
    vae_latent_column: Updatable[str] = NO_UPDATE
    caption_column: Updatable[str] = NO_UPDATE
    sample_type: Updatable[Literal["fix_fps", "fix_stride", "full_video", "random"]] = NO_UPDATE
    batch_size: Updatable[int] = NO_UPDATE
    num_samples: Updatable[int] = -1
    control_columns: Updatable[List[str]] = NO_UPDATE
    training_precision: str = NO_UPDATE

    motion_strength: Optional[float] = None
    """a user level motion_strength config, only for I2V"""

    def update_config(self):
        assert self.train_name != NO_UPDATE or self.yml_path != NO_UPDATE
        if self.yml_path != NO_UPDATE and os.path.exists(self.yml_path):
            config = eval_setup(self.yml_path)
        else:
            config = train_configs[self.train_name]

        config.pipeline.update(
            target=self.target,
            endfix_prompt=self.endfix_prompt,
            negative_prompt=self.negative_prompt,
            seed=self.seed,
        )  # TODO: update call, some settings are not in pipeline.call

        call_dict = parse_config_string(self.call)
        if self.height is not NO_UPDATE:
            call_dict['height'] = self.height
        if self.width is not NO_UPDATE:
            call_dict['width'] = self.width

        if self.motion_strength is not None:
            assert self.motion_strength >= 0.0 and self.motion_strength <= 1.0, "motion_strength should be in [0.0, 1.0]"
            STRENGTH_MIN = 0.6
            STRENGTH_MAX = 1.0
            DOWNGRADE_SCALE = 0.0
            if self.motion_strength == 1.0:
                call_dict.pop("strength", None)
                call_dict.pop("downgrade_scale", None)
            else:
                call_dict["strength"] = STRENGTH_MIN + (STRENGTH_MAX - STRENGTH_MIN) * self.motion_strength
                call_dict["strength"] = round(call_dict["strength"], 1)
                call_dict["downgrade_scale"] = DOWNGRADE_SCALE
        config.pipeline.call.update(**call_dict)

        config.update(
            output_dir=self.output_dir,
            experiment_name=self.experiment_name,
            training_precision=self.training_precision,
        )
        config.pipeline.update(
            ckpt_path=self.ckpt_path,
            diffusers_vae_ckpt_path=self.diffusers_vae_ckpt_path,
            temporal_vae_ckpt_path=self.temporal_vae_ckpt_path,
        )
        if config.pipeline.unet_config is not None:
            config.pipeline.unet_config.update(
                num_frames=self.num_frames,
                train_len=self.train_len,
                unet_ckpt_path=self.unet_ckpt_path,
                mm_ckpt_path=self.mm_ckpt_path,
                zero_init=False,
            )
        if config.pipeline.transformer_config is not None:
            config.pipeline.transformer_config.update(
                num_frames=self.num_frames,
                transformer_ckpt_path=self.transformer_ckpt_path,
                use_flash_attn=self.use_flash_attn,
            )

        config.val_data.update(
            mode=self.data_mode,
            path=self.csv_path,
            video_path_column=self.video_path_column,
            image_path_column=self.image_path_column,
            caption_column=self.caption_column,
            vae_latent_column=self.vae_latent_column,
            index_column=self.index_column,
            num_frames=self.num_frames,
            sample_type=self.sample_type,
            sample_fps=self.sample_fps,
            batch_size=self.batch_size,
            num_samples=self.num_samples,
            height=self.height,
            width=self.width,
            crop_type=self.crop_type,
            control_columns=self.control_columns,
        )
        return config


#### base ####
test_configs = {"test": TestConfig()}

#### t2v ####
test_configs["test_t2v"] = TestConfig(
    train_name="animatediff",
    output_dir="test/t2v/openpose",
    data_mode="video",
    csv_path="./tmp/v2v.csv",
    batch_size=6,
    height=640,
    width=360,
    target="StableDiffusionControlNetPipeline",
    mm_ckpt_path="/group/ckpt/diffusers/animatediff/mm_sd_v15.ckpt",
    negative_prompt="worst quality, low quality, normal quality, lowres, bad anatomy, bad hands, watermark, moles",
    seed=42,
    call="""
        height=640
        width=360
        guidance_scale=12.5
        num_inference_steps=25
        downgrade_scale=0
        controlnet_conditioning_scale=[1.0]
        control_guidance_start=[0]
        control_guidance_end=[1.0]
    """,
    control_columns=["openpose_path"],
)

test_configs["test_t2v_svd"] = TestConfig(
    train_name="t2v_svd",
    output_dir="outputs/t2v_svd_0219/svd_t2v_0226_vcg_new_caption_amd_20cluster_iter5760000_recaption/",
    data_mode="video",
    csv_path="/video/yht/data/test_prompts_llama2_7b.csv",
    batch_size=1,
    height=256,
    width=448,
    unet_ckpt_path="/video/yht/exps/svd_t2v_0226_vcg_new_caption_amd_20cluster/checkpoints/checkpoint-5760000/unet/pytorch_model.ckpt",
    seed=42,
    call="""
        height=256
        width=448
        num_inference_steps=25
    """,
)


test_configs["test_i2v_svd"] = TestConfig(
    train_name="i2v_svd",
    output_dir="outputs/i2v_svd",
    data_mode="video",
    csv_path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
    batch_size=2,
    height=576,
    width=1024,
    unet_ckpt_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/exps/i2v_svd_vcg_edm/checkpoints/checkpoint-24000/unet/pytorch_model_trans.ckpt",
    seed=42,
    call="""
        height=576
        width=1024
        num_inference_steps=25
    """,
)


test_configs["test_t2v_lumeire_xl"] = TestConfig(
    train_name="lumiere_xl",
    output_dir="outputs/lumiere_xl",
    data_mode="video",
    csv_path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",  # "/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv"
    video_path_column=None,
    batch_size=2,
    height=320,
    width=512,
    sample_fps=8,
    num_frames=16,
    unet_ckpt_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/exps/lumiere_xl_4layer/checkpoints/checkpoint-1000704/unet/pytorch_model.ckpt",
    seed=42,
    call="""
        height=320
        width=512
        num_inference_steps=25
    """,
)


test_configs["test_t2v_lumeire_xxl"] = TestConfig(
    train_name="lumiere_xxl",
    output_dir="outputs/lumiere_xxl",
    data_mode="video",
    csv_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv",  # "/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv", # "/group/houliang/video/m2v-diffusers/data/vcg_val.csv"
    video_path_column=None,
    batch_size=20,
    height=256,
    width=448,
    sample_fps=16,
    num_frames=16,
    unet_ckpt_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/exps/lumiere_xxl_53M_8/checkpoints/checkpoint-23000064/unet/pytorch_model.ckpt",
    seed=42,
    call="""
        height=256
        width=448
        num_inference_steps=100
    """,
)


test_configs["test_animatediff_xl"] = TestConfig(
    train_name="animatediff_xl",
    output_dir="outputs/animatediff_xl",
    data_mode="video",
    csv_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv",
    video_path_column=None,
    batch_size=4,
    height=1024,
    width=1024,
    sample_fps=8,
    num_frames=16,
    seed=42,
    call="""
        height=1024
        width=1024
        num_inference_steps=100
        guidance_scale=12.5
    """,
)

test_configs["test_latte"] = TestConfig(
    train_name="latte-official",
    output_dir="outputs/latte-official/latte-official-0318-ckpt0",
    data_mode="video",
    csv_path="/video/yht/testset/m2v_test_prompts_1k_0319.csv",
    transformer_ckpt_path=[
        "/k4d/yanghaotian03/m2v-diffusers/exps/latte_official_joint_finetune_4nv_bs1_res512_fp32_vcg_0319/checkpoints/checkpoint-160000/transformer/pytorch_model.ckpt",
    ],
    batch_size=1,
    height=512,
    width=512,
    num_frames=16,
    seed=42,
    num_samples=1063,
    call="""
        num_frames = 16
        height = 512
        width = 512
        num_inference_steps = 50
        guidance_scale = 7.5
        enable_temporal_attentions = True
        num_images_per_prompt = 1
        mask_feature = True
        enable_vae_temporal_decoder = True
    """,
)

test_configs["test_pixart_alpha_edm"] = TestConfig(
    train_name="pixart-alpha-edm-baseline",
    output_dir="outputs/pixart-alpha-edm-baseline",
    data_mode="image",
    csv_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv",
    transformer_ckpt_path=[
        "/video/zhengmingwu/m2v-diffusers/exps/pixart-alpha-edm-baseline_256x256_178bs_h800x8/checkpoints/checkpoint-2272704000/ema/ema.ckpt",
    ],
    video_path_column=None,
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=1,
    num_samples=1063,
    seed=0,
    call="""
        num_frames=1
        height=256
        width=256
        num_inference_steps=20
        guidance_scale=4.5
        negative_prompts=""
    """,
)

test_configs["test_pixart_alpha"] = TestConfig(
    train_name="pixart-alpha-iddpm",
    output_dir="outputs/pixart-alpha-offical_2_SAM-256x256",
    data_mode="image",
    csv_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv",
    video_path_column=None,
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=1,
    num_samples=1063,
    seed=0,
    call="""
        num_frames=1
        height=256
        width=256
        num_inference_steps=20
        guidance_scale=4.5
        negative_prompts=""
    """,
)

test_configs["test_dit_imagenet"] = TestConfig(
    train_name="dit-t2i",
    output_dir="outputs/dit_t2i",
    data_mode="image",
    csv_path="/video/pansiyuan/data/test_ImageNet_100000.csv",
    transformer_ckpt_path=[
        "/home/pansiyuan/.jupyter/m2v-diffusers/exps/dit_t2i/checkpoints/checkpoint-12800000/transformer/pytorch_model.ckpt",
    ],
    video_path_column=None,
    batch_size=9,
    height=256,
    width=256,
    num_frames=1,
    num_samples=1000,
    caption_column=None,
    seed=66,
    call="""
        num_inference_steps=25
        guidance_scale=4.5
        output_type="pt"
    """,
)

test_configs["test-transformerxl-flow-baseline"] = TestConfig(
    train_name="transformerxl-flow-baseline",
    output_dir="outputs/transformerxl-flow-baseline",
    data_mode="image",
    csv_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv",
    transformer_ckpt_path=[
        "/video/zhengmingwu/m2v-diffusers/exps/transformerxl-flow-baseline/checkpoints/checkpoint-2358144000/ema/ema.ckpt",
    ],
    video_path_column=None,
    vae_latent_column=None,
    caption_column="caption",
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=1,
    num_samples=1063,
    seed=0,
    call="""
        num_frames=1
        height=256
        width=256
        num_inference_steps=20
        guidance_scale=4.5
        negative_prompts=""
    """,
)


test_configs["test-t2v-transformerxl-flow"] = TestConfig(
    train_name="t2v-transformerxl-flow-baseline",
    output_dir="outputs/t2v-transformerxl-flow-baseline",
    data_mode="video",
    csv_path="/video/houliang/m2v-diffusers/data/new_qa_test.csv",
    transformer_ckpt_path=[
        # "/video/houliang/m2v-diffusers/exps/t2v-transformerxl-flow-17x256x256_60m_59gpus/checkpoints/checkpoint-475776000/ema/ema.ckpt",
        "/video/houliang/m2v-diffusers/exps/t2v-transformerxl-flow-17x256x256/checkpoints/checkpoint-11136000/ema/ema.ckpt",
    ],
    video_path_column=None,
    vae_latent_column=None,
    caption_column="response",
    index_column="index",
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=17,
    num_samples=225,
    seed=0,
    training_precision="fp32",
    call="""
        num_frames=17
        height=256
        width=256
        num_inference_steps=50
        guidance_scale=7.5
        negative_prompts=""
        fps=15
    """,
)



test_configs["test-transformerxl-flow-baseline"] = TestConfig(
    train_name="transformerxl-flow-baseline",
    output_dir="outputs/transformerxl-flow-baseline",
    data_mode="image",
    csv_path="/group/zhengmingwu/m2v-diffusers-v3/m2v-diffusers/inputs/m2v_test_prompts_1k.csv",
    transformer_ckpt_path=[
        "/video/zhengmingwu/m2v-diffusers/exps/transformerxl-flow-baseline/checkpoints/checkpoint-2358144000/ema/ema.ckpt",
    ],
    video_path_column=None,
    vae_latent_column=None,
    caption_column='caption',
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=1,
    num_samples=1063,
    seed=0,
    call="""
        num_frames=1
        height=256
        width=256
        num_inference_steps=20
        guidance_scale=4.5
        negative_prompts=""
    """,
)


test_configs["test-t2v-transformerxl-flow-baseline-navit"] = TestConfig(
    train_name="t2v-transformerxl-flow-baseline-navit",
    output_dir="outputs/t2v-transformerxl-flow-baseline-navit",
    data_mode="video",
    #    csv_path="/video/houliang/m2v-diffusers/data/new_qa_test.csv",
    csv_path="/group/houliang/video/m2v-diffusers/data/vcg_val.csv",
    transformer_ckpt_path=[
        "/group/gaoyuan/code/vit/join_train/m2v-diffusers/exps/xl-flow-varl/checkpoints/checkpoint-4608000/ema/ema.ckpt",
    ],
    video_path_column=None,
    #caption_column='response',
    batch_size=1,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=17,
    num_samples=8,
    call="""
        num_inference_steps=50
        guidance_scale=7.5
        negative_prompts=""
    """,
)


test_configs["test-t2v-transformerxl-flow"] = TestConfig(
    train_name="t2v-transformerxl-flow-baseline",
    output_dir="outputs/t2v-transformerxl-flow-baseline",
    data_mode="video",
    csv_path="/video/houliang/m2v-diffusers/data/new_qa_test.csv",
    transformer_ckpt_path=[
        # "/video/houliang/m2v-diffusers/exps/t2v-transformerxl-flow-17x256x256_60m_59gpus/checkpoints/checkpoint-475776000/ema/ema.ckpt",
        "/video/houliang/m2v-diffusers/exps/t2v-transformerxl-flow-17x256x256/checkpoints/checkpoint-11136000/ema/ema.ckpt",
    ],
    video_path_column=None,
    vae_latent_column=None,
    caption_column="response",
    index_column="index",
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=17,
    num_samples=225,
    seed=0,
    training_precision="fp32",
    call="""
        num_frames=17
        height=256
        width=256
        num_inference_steps=50
        guidance_scale=7.5
        negative_prompts=""
        fps=15
    """,
)



test_configs["test-transformerxl-flow-baseline-yht"] = TestConfig(
    train_name="t2v-transformerxl-flow-stfit",
    output_dir="outputs/t2v-transformerxl-flow-stfit-1d-hl-baseline",
    data_mode="video",
    csv_path="/video/yht/exps/test_yht.csv",
    transformer_ckpt_path=[
        #"/video/yht/exps/t2v-transformerxl-flow-baseline-0416-speed-test/checkpoints/checkpoint-7680000/ema/ema.ckpt",
        #"/video/yht/exps/t2v-transformerxl-flow-baseline-24amd-0424/checkpoints/checkpoint-18432000/ema/ema.ckpt",
        "/video/houliang/m2v-diffusers/exps/t2v-transformerxl-cdflow-81x256x256-bs32-12gpus/checkpoints/checkpoint-24576000/ema/ema.ckpt"
    ],
    video_path_column=None,
    vae_latent_column=None,
    caption_column='caption',
    index_column="index",
    batch_size=4,
    height=256,
    width=256,
    sample_fps=15,
    num_frames=81,
    num_samples=16,
    seed=0,
    call="""
        height=256
        width=256
        num_inference_steps=50
        guidance_scale=7.5
        negative_prompts=""
    """,
)