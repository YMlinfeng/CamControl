import torch
import imageio, io
import numpy as np
import copy
import os
import json, random, sys
from src.pipelines.t2v_pipeline_flow import T2VFlowPipelineConfig
from src.models.autoencoders.visual_tokenizer import VisualTokenizerConfig
from src.models.transformers.transformer_xl import TransformerXLModelConfig, TemporalAttentionConfig
from src.utils import get_current_git_commit
from datetime import datetime

class TaskRunningExecutor(object):
    def __init__(self, model_path):
        self.video_size = 512
        pipeline=T2VFlowPipelineConfig(
            ckpt_path=os.path.join(model_path, "PixArt-XL-2-512x512"),
            vae_config=VisualTokenizerConfig(
                vae_ckpt_path=os.path.join(model_path, "tinyvae.ckpt"),
                scaling_factor=0.3414321773546459,
                encoder_init_dim=32,
                split_conv_3d=True,
            ),
            clip_ckpt_path= None,
            scheduler_kwargs={
                "_class_name": "PNDMScheduler",
                "beta_end": 0.02,
                "beta_schedule": "linear",
                "beta_start": 0.0001,
                "variance_type": "learned_range",
            },
            transformer_config=TransformerXLModelConfig(
                num_layers=40,
                num_attention_heads=40,
                cross_attention_dim=2880,
                num_frames=77,
                height=int(self.video_size * 1.3125),
                width=int(self.video_size * 1.3125),
                vae_temporal_scale_factor=4,
                vae_scale_factor=8,
                patch_size=2,
                out_channels=8,
                in_channels=8,
                from_scratch=True,
                gradient_checkpointing=False,
                use_flash_attn=True,
                use_2d_rope=True,
                theta_2d_rope=100,
                use_1d_rope=True,
                theta_1d_rope=100,
                use_temp_attn=True,
                image_temp_attn=True,
                temporal_attention_config=TemporalAttentionConfig(
                    attn_type="stfit",
                    theta_3d_rope=100,
                    stfit_patch_size=(1,2,2),
                ),
                transformer_ckpt_path=[
                    os.path.join(model_path, "m2v_ema.ckpt"),
                ],
                qk_norm=True,
                use_resolution_condition=True,
                use_aspect_ratio_condition=False,
                use_frames_condition=True,
                use_fps_condition=False,
                use_text_condition=False,
                split_conditions=True,
            ),
            call={
                "num_frames": 77,
                "height": 512,
                "width": 512,
                "num_inference_steps": 50,
                "negative_prompt": "",
                "guidance_scale": 7.5,
                "fps": 15,
                "output_type": "pt",
                "timestep_shift": 5,
            },
            max_sequence_length=256,
            proportion_empty_prompts=0.1,
            offload=False,
        )
        self.pipeline = copy.deepcopy(pipeline)
        self.pipe_run = self.pipeline.from_pretrained().to(device='cuda', dtype=torch.bfloat16)
        # self.save_params_path = os.path.join(os.getenv('LOG_INPUT_DIR'), 'm2v_cases')
        # os.makedirs(self.save_params_path, exist_ok=True)
        self.aspect_size = {
            "16:9": {512: [672, 384], 720: [1280, 720], 1080: [1920, 1080]},
            "1:1": {512: [512, 512], 720: [960, 960], 1080: [1440, 1440]},
            "9:16": {512: [384, 672], 720: [720, 1280], 1080: [1080, 1920]},
        }

    def images2video_buffer(self, images, kwargs):
        fps = kwargs.get('fps', 30)
        format = kwargs.get('format', 'mp4')  # 默认为 mp4 格式
        codec = kwargs.get('codec', 'libx264')  # 默认为 libx264 编码器
        ffmpeg_params = ['-crf', str(kwargs.get('crf', 12))]
        pixelformat = kwargs.get('pixelformat', 'yuv420p')  # 视频像素格式

        # 创建一个 BytesIO 对象作为视频数据的内存存储
        video_stream = io.BytesIO()

        with imageio.get_writer(video_stream, fps=fps, format=format, codec=codec, ffmpeg_params=ffmpeg_params, pixelformat=pixelformat) as writer:
            for idx in range(len(images)):
                writer.append_data(images[idx])

        return video_stream.getvalue()

    def execute(self, prompt, negative_prompt="", width="0", height="0", aspect_ratio="", fps="15", total_frames="77", cfg="12.5", steps="50", seed="", process_call_back=None):
        width = int(width)
        height = int(height)
        fps = int(fps)
        total_frames = int(total_frames)
        cfg = float(cfg)
        steps = int(steps)
        try:
            if ((128 <= width <= 1920 and width % 32 == 0) and (128 <=height <= 1920 and height % 32 == 0)) or aspect_ratio in self.aspect_size.keys():
                if seed == "":
                    seed = random.randint(-pow(2, 31), pow(2, 31)-1)
                else:
                    seed = int(seed)
                generator = torch.Generator(device=self.pipe_run.device).manual_seed(seed)
                save_fps = 15
                if fps == 0 and total_frames == 1:
                    save_fps = 1
                else:
                    fps = min(max(fps, 11), 22)
                    total_frames = min(max(total_frames, 17), 77)
                cfg = min(max(cfg, 1), 50)
                steps = min(max(steps, 20), 100)
                call_params = {
                    "num_frames": total_frames,
                    "width": self.aspect_size[aspect_ratio][int(self.video_size)][0] if aspect_ratio in self.aspect_size.keys() else width,
                    "height": self.aspect_size[aspect_ratio][int(self.video_size)][1] if aspect_ratio in self.aspect_size.keys() else height,
                    "num_inference_steps": steps,
                    "negative_prompt": negative_prompt,
                    "guidance_scale": cfg,
                    "fps": fps,
                    "process_call_back": process_call_back,
                }
                self.pipeline.call.update(call_params)
                video_data = (self.pipe_run(prompt, generator=generator, **self.pipeline.call)['images'].cpu().to(torch.float32).numpy()*255).astype(np.uint8).transpose(0,2,3,1)
                video_data = self.images2video_buffer(video_data, {'fps': save_fps})

                call_params['seed'] = seed
                call_params['prompt'] = prompt
                call_params.pop('process_call_back')
                call_params['commitID'] = get_current_git_commit()
                call_params['aspect_ratio'] = aspect_ratio
                # current_timestamp = datetime.now()
                # file_name = '-'.join(['_'.join(prompt[:80].split(' ')), current_timestamp.strftime("%Y_%m_%d_%H_%M_%S_%f"), str(seed), str(cfg)])[:128]
                # with open(os.path.join(self.save_params_path, file_name+'.txt'), 'w') as f:
                #     json.dump(call_params, f, indent=4)
                return True, (True, video_data, json.dumps(call_params),)
            else:
                error_message = "NOT SUPPORT THIS RESOLUTION(width、height or ratio)"
                return False, (False, error_message, "", )
        except:
            error_message = "M2V MODEL ERROR BUT MESSAGE IS NULL"
            return False, (False, error_message, "", )
