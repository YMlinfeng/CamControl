import json
import math
import random
import resource
import sys
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Type

import cv2
from cv2 import repeat
import decord
import numpy as np
import torch
from datasets import Dataset
from stable_diffusion_application.kaimm.accelerate_utils import set_seed
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.transforms.functional import crop
from einops import rearrange

try:
    import pyvips

    pyvips_installed = True
except:
    pyvips_installed = False

from ..configs.base_config import InstantiateConfig
from ..utils import log_to_rank0, measure_time

class Camera(object):
    def __init__(self, c2w):
        c2w_mat = np.array(c2w).reshape(4, 4)
        self.c2w_mat = c2w_mat
        self.w2c_mat = np.linalg.inv(c2w_mat)


def get_relative_pose(cam_params):
    abs_w2cs = [cam_param.w2c_mat for cam_param in cam_params]
    abs_c2ws = [cam_param.c2w_mat for cam_param in cam_params]

    cam_to_origin = 0
    target_cam_c2w = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, -cam_to_origin],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    abs2rel = target_cam_c2w @ abs_w2cs[0]
    ret_poses = [target_cam_c2w, ] + [abs2rel @ abs_c2w for abs_c2w in abs_c2ws[1:]]
    ret_poses = np.array(ret_poses, dtype=np.float32)
    return ret_poses

def custom_meshgrid(*args):
    from packaging import version as pver
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')

def ray_condition(K, c2w, H=384//16, W=672//16, device='cpu', flip_flag=None):
    # c2w: B, V, 4, 4
    # K: B, V, 4

    B, V = K.shape[:2]

    j, i = custom_meshgrid(
        torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
        torch.linspace(0, W - 1, W, device=device, dtype=c2w.dtype),
    )
    i = i.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]
    j = j.reshape([1, 1, H * W]).expand([B, V, H * W]) + 0.5          # [B, V, HxW]

    n_flip = torch.sum(flip_flag).item() if flip_flag is not None else 0
    if n_flip > 0:
        j_flip, i_flip = custom_meshgrid(
            torch.linspace(0, H - 1, H, device=device, dtype=c2w.dtype),
            torch.linspace(W - 1, 0, W, device=device, dtype=c2w.dtype)
        )
        i_flip = i_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        j_flip = j_flip.reshape([1, 1, H * W]).expand(B, 1, H * W) + 0.5
        i[:, flip_flag, ...] = i_flip
        j[:, flip_flag, ...] = j_flip

    fx, fy, cx, cy = K.chunk(4, dim=-1)     # B,V, 1

    zs = torch.ones_like(i)                 # [B, V, HxW]
    xs = (i - cx) / fx * zs
    ys = (j - cy) / fy * zs
    zs = zs.expand_as(ys)

    directions = torch.stack((xs, ys, zs), dim=-1)              # B, V, HW, 3
    directions = directions / directions.norm(dim=-1, keepdim=True)             # B, V, HW, 3

    rays_d = directions @ c2w[..., :3, :3].transpose(-1, -2)        # B, V, HW, 3
    rays_o = c2w[..., :3, 3]                                        # B, V, 3
    rays_o = rays_o[:, :, None].expand_as(rays_d)                   # B, V, HW, 3
    # c2w @ dirctions
    rays_dxo = torch.cross(rays_o, rays_d)                          # B, V, HW, 3
    plucker = torch.cat([rays_dxo, rays_d], dim=-1)
    plucker = plucker.reshape(B, c2w.shape[1], H, W, 6)             # B, V, H, W, 6
    # plucker = plucker.permute(0, 1, 4, 2, 3)
    return plucker

@dataclass
class DataConfig(InstantiateConfig):
    _target: Type = field(default_factory=lambda: Data)
    """target class to instantiate"""

    path: Optional[str] = None
    """Path to dataset."""
    index_column: Optional[str] = None
    """Column name for index."""
    video_path_column: Optional[str] = None
    """Column name for image path."""
    image_path_column: Optional[str] = None
    """Column name for image path."""
    latent_path_column: Optional[str] = None
    """Column name for latent path."""
    condition_image_path_column: Optional[str] = None  # add for condition image
    """Column name for condition image path."""
    ref_path_column: Optional[str] = None  # add for ref video
    """Column name for ref video maps."""
    content_ref_path_column: Optional[str] = None  # add for content ref video
    """Recam Column name for content video path."""
    cam_rt_path_column: Optional[str] = None  # add for camera RT
    """Column name for Camera RT."""
    image_latent_path_column: Optional[str] = None
    """Column name for image latent path."""
    caption_column: Optional[str] = None
    """Column name for caption."""
    class_labels_column: Optional[str] = None
    """Class labels"""
    class_id_to_num: Optional[str] = "/video/pansiyuan/data/ImageNet_class_to_id.json"
    """Path of the class to number of imagenet if we use class_num_column"""
    t5_prompt_embed_column: Optional[str] = None
    """Column name for t5 prompt embed."""
    clip_prompt_embed_column: Optional[str] = None
    """Column name for clip prompt embed."""
    motion_bucket_id_column: Optional[str] = None
    """Column name for motion bucket id"""
    control_columns: Optional[List[str]] = None
    """Column name for controls."""
    height: int = 512
    """Height of video."""
    width: int = 512
    """Width of video."""
    crop_type: Optional[Literal["center", "random"]] = "center"
    # crop_type: Optional[Literal["center", "random"]] = None
    """Whether to center crop the video."""
    resize_video: bool = True
    """whether to resize to target size at video read"""
    random_flip: bool = False
    """Whether to randomly flip the video."""
    num_frames: Optional[int] = 16
    """Number of frames to sample"""
    max_duration: Optional[float] = None
    """Maximum duration (s)"""
    max_fps: float = 60
    """Maximum fps"""
    max_aspect_ratio: float = 16 / 9
    """Maximum aspect ratio"""
    sample_fps: int = 8
    """FPS of video to sample."""
    sample_stride: int = 4
    """Stride of video to sample."""
    sample_type: Literal["fps", "stride", "full", "random"] = "fps"
    """Type of sampling."""
    sample_position: Literal["first", "center", "last", "random"] = "first"
    """Position of sampling."""
    max_frame_stride: int = sys.maxsize
    """Maximal sample stride when randomly sample"""
    num_samples: Optional[int] = None
    """Maximal number of samples"""
    batch_size: int = 1
    """Batch size to use."""
    num_processes: int = 1
    """Number of processes to use."""
    shuffle: bool = True
    """Whether to shuffle the dataloader"""
    cache_dir: Optional[str] = "/group/gaoyuan/huggingface/datasets"
    """Cache directory for huggingface dataset"""
    use_determinstic_dataset: bool = False
    """Whether to use fully determinstic dataset"""
    measure_time: bool = False
    """Whether to measure time"""
    gpu_decode: bool = False
    """Whether to use GPU to decode video"""
    gpu_read_chunk_size: Optional[int] = None
    """Chunk size for GPU read"""
    use_pyvips: bool = False
    """Whether to use pyvips for reading images"""

    vae_spatial_scale_factor: int = 8
    """VAE spatial scale factor"""
    vae_temporal_scale_factor: int = 4
    """VAE temporal scale factor"""
    spatial_patch_size: int = 2
    """Spatial patch_size"""
    temporal_patch_size: int = 1
    """Temporal patch_size"""
    spatial_token_merge_size: Optional[int] = None
    """Spatial token merge size."""
    temporal_token_merge_size: Optional[int] = None
    """Temporal token merge size."""

    random_num_frames: bool = False
    """Whether to sample random num_frames for videos."""


class DeterminsticDataset:
    def __init__(self, data: Dataset):
        self.data = data

    def __getitem__(self, key):
        assert isinstance(key, int)
        set_seed(key + 1)
        return self.data.__getitem__(key)

    def __len__(self):
        return len(self.data)


def set_soft_limit(resource_type, soft_limit):
    try:
        current_limits = resource.getrlimit(resource_type)
        new_limits = (soft_limit, current_limits[1])
        resource.setrlimit(resource_type, new_limits)
        log_to_rank0(f"Soft limit for {resource_type} set to {soft_limit}")
    except Exception as e:
        print(f"An error occurred: {e}")


class Data:
    def __init__(self, config: DataConfig, timer=None):
        # NOTE: 更改系统最大打开文件数量为1048576
        set_soft_limit(resource.RLIMIT_NOFILE, 1048576)

        self.vae_spatial_scale_factor = config.vae_spatial_scale_factor
        self.vae_temporal_scale_factor = config.vae_temporal_scale_factor
        self.spatial_patch_size = config.spatial_patch_size
        self.temporal_patch_size = config.temporal_patch_size
        self.spatial_unit_size = config.vae_spatial_scale_factor * config.spatial_patch_size
        self.temporal_unit_size = config.vae_temporal_scale_factor * config.temporal_patch_size

        # NOTE: FIT Token Merge
        self.spatial_token_merge_size = config.spatial_token_merge_size
        self.temporal_token_merge_size = config.temporal_token_merge_size
        if config.spatial_token_merge_size is not None:
            self.spatial_unit_size = self.spatial_unit_size * config.spatial_token_merge_size
        if config.temporal_token_merge_size is not None:
            self.temporal_unit_size = self.temporal_unit_size * config.temporal_token_merge_size

        self.index_column = config.index_column
        self.video_path_column = config.video_path_column
        self.image_path_column = config.image_path_column
        self.latent_path_column = config.latent_path_column
        self.condition_image_path_column = config.condition_image_path_column  # add for condition image
        self.ref_path_column = config.ref_path_column  # add for ref video
        self.content_ref_path_column = config.content_ref_path_column  # add for content ref video
        self.cam_rt_path_column = config.cam_rt_path_column  
        self.image_latent_path_column = config.image_latent_path_column
        self.caption_column = config.caption_column
        self.t5_prompt_embed_column = config.t5_prompt_embed_column
        self.clip_prompt_embed_column = config.clip_prompt_embed_column
        self.motion_bucket_id_column = config.motion_bucket_id_column
        self.control_columns = config.control_columns
        self.class_labels_column = config.class_labels_column
        if self.class_labels_column is not None:
            with open(config.class_id_to_num, "r") as f:
                self.class_id_to_num = json.load(f)

        self.height = config.height
        self.width = config.width
        self.crop_type = config.crop_type
        self.random_flip = config.random_flip
        self.max_fps = config.max_fps  # 用于去除异常数据
        self.max_aspect_ratio = config.max_aspect_ratio

        self.num_frames = config.num_frames
        self.sample_fps = config.sample_fps
        self.sample_stride = config.sample_stride
        self.sample_type = config.sample_type
        self.sample_position = config.sample_position

        self.max_num_frames = config.num_frames
        # 可变时长
        self.max_duration = config.max_duration
        self.is_variable_duration = False
        if self.max_duration is not None:
            self.is_variable_duration = True

        # 可变长宽比
        self.is_variable_aspect_ratio = False
        # self.is_variable_aspect_ratio = True
        # self.crop_type = None  # TODO 注意这里可能会bug
        if self.crop_type == None:
            self.is_variable_aspect_ratio = True
            self.max_area = self.height * self.width
        #         max_num_frames = 1
        #         if self.is_variable_duration:
        #             if self.sample_type == "fps":
        #                 max_num_frames = self.sample_fps * self.max_duration
        #             elif self.sample_type == "stride":
        #                 frame_stride = max(1, self.sample_stride)
        #                 max_num_frames = (self.max_duration * self.max_fps - 1) // frame_stride + 1
        #             elif self.sample_type == "full":
        #                 max_num_frames = int(self.max_duration * self.max_fps)
        #             elif self.sample_type == "random":
        #                 max_num_frames = int(self.max_duration * self.max_fps)
        #         self.max_num_frames = int(max_num_frames)
        # self.max_volume = self.max_num_frames * self.max_area

        self.config = config

        # for dataset
        self.dataset = Dataset.from_csv(config.path, cache_dir=config.cache_dir)
        self.dataset.set_transform(self.process_items)
        if config.num_samples and config.num_samples > 0:
            self.dataset = self.dataset.select(range(config.num_samples))

        self.resolution = (config.height, config.width)
        if self.crop_type == "center":
            self.train_crop = torch.jit.script(transforms.CenterCrop(self.resolution))
        elif self.crop_type == "random":
            self.train_crop = transforms.RandomCrop(self.resolution)
        self.train_flip = transforms.RandomHorizontalFlip(p=1.0)
        self.train_transforms = torch.jit.script(transforms.Normalize([127.5], [127.5]))

        # for dataloader
        if config.use_determinstic_dataset:
            # NOTE: for perfect resume, default False
            self.dataloader = DataLoader(
                DeterminsticDataset(self.dataset),
                batch_size=config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=config.num_processes,
                pin_memory=True,
                shuffle=config.shuffle,
                generator=torch.Generator().manual_seed(1),
            )
        else:
            self.dataloader = DataLoader(
                self.dataset,
                batch_size=config.batch_size,
                collate_fn=self.collate_fn,
                num_workers=config.num_processes,
                pin_memory=True,
                shuffle=config.shuffle,
            )

        self.timer = timer

        decord.bridge.set_bridge("torch")

    def process_items(self, items):
        return_dict = {}

        if self.caption_column:
            captions = items[self.caption_column]
            prompts = []
            for caption in captions:
                try:
                    prompt = random.choice(eval(caption))
                except:
                    prompt = caption
                prompts.append(prompt)
            return_dict.update(
                {
                    "prompts": prompts,
                }
            )

        if self.class_labels_column is not None:
            all_the_classes = items[self.class_labels_column]
            class_labels = []
            for each_class in all_the_classes:
                class_labels.append(self.class_id_to_num[each_class])
            return_dict.update(
                {
                    "class_labels": class_labels,
                }
            )

        if self.t5_prompt_embed_column:
            with measure_time("DATA_T5", self.config.measure_time, timer=self.timer, verbose=False):
                prompt_embed_paths = items[self.t5_prompt_embed_column]
                prompt_embeds = []
                for prompt_embed_path in prompt_embed_paths:
                    try:
                        prompt_embed = torch.load(prompt_embed_path, map_location="cpu")
                    except Exception as e:
                        print(f"Failed to load t5 prompt embed {prompt_embed_path}. {e}")
                        return {"failed": [True]}

                    prompt_embeds.append(prompt_embed)
                return_dict.update(
                    {
                        "t5_prompt_embeds": prompt_embeds,
                    }
                )

        if self.clip_prompt_embed_column:
            with measure_time("DATA_CLIP", self.config.measure_time, timer=self.timer, verbose=False):
                prompt_embed_paths = items[self.clip_prompt_embed_column]
                prompt_embeds = []
                for prompt_embed_path in prompt_embed_paths:
                    try:
                        prompt_embed = torch.load(prompt_embed_path, map_location="cpu")
                    except Exception as e:
                        print(f"Failed to load clip prompt embed {prompt_embed_path}. {e}")
                        return {"failed": [True]}

                    prompt_embeds.append(prompt_embed)

                return_dict.update(
                    {
                        "clip_prompt_embeds": prompt_embeds,
                    }
                )

        if self.index_column:
            index = items[self.index_column]
            return_dict.update(
                {
                    "index": index,
                }
            )

        # if self.video_path_column:
        if self.video_path_column and not self.ref_path_column and not self.content_ref_path_column:
            with measure_time("DATA_Video", self.config.measure_time, timer=self.timer, verbose=False):
                video_paths = items[self.video_path_column]
                videos = []
                start_frames = []
                frame_strides = []
                fps = []
                sample_fps = []
                durations = []
                num_frames = []

                target_sizes = []
                original_sizes = []
                crop_top_lefts = []

                for video_path in video_paths:
                    try:
                        frames, start_frame, frame_stride, _fps, duration, original_size = self.read_video(video_path)
                        print(f"frames shape is {frames.shape}")
                    except Exception as e:
                        print(f"Failed to read video {video_path}. {e}")
                        return {"failed": [True]}

                    start_frames.append(start_frame)
                    frame_strides.append(frame_stride)
                    fps.append(_fps)
                    sample_fps.append(_fps / frame_stride)
                    durations.append(duration)

                    frames, _, crop_top_left = self.preprocess_image(frames, resize=False)

                    num_frames.append(frames.shape[-4]) # (f, c, h, w)
                    target_sizes.append(frames.shape[-2:])
                    original_sizes.append(original_size)
                    crop_top_lefts.append(crop_top_left)

                    videos.append(frames.unsqueeze(0))

                return_dict.update(
                    {
                        "data_paths": video_paths,
                        "data": videos,
                        "start_frames": start_frames,
                        "frame_strides": frame_strides,
                        "fps": fps,
                        "sample_fps": sample_fps,
                        "durations": durations,
                        "num_frames": num_frames,
                        "target_sizes": target_sizes,
                        "original_sizes": original_sizes,
                        "crop_top_lefts": crop_top_lefts,
                    }
                )
        elif self.image_path_column:
            with measure_time("DATA_Image", self.config.measure_time, timer=self.timer, verbose=False):
                image_paths = items[self.image_path_column]
                images = []
                start_frames = []
                frame_strides = []
                fps = []
                sample_fps = []
                durations = []
                num_frames = []

                target_sizes = []
                original_sizes = []
                crop_top_lefts = []

                for image_path in image_paths:
                    try:
                        image = self.read_image(image_path)
                    except Exception as e:
                        print(f"Failed to read image {image_path}. {e}")
                        return {"failed": [True]}

                    image, original_size, crop_top_left = self.preprocess_image(image)
                    images.append(image.unsqueeze(1))  # b c h w -> b f c h w

                    num_frames.append(1)
                    target_sizes.append(image.shape[-2:])
                    original_sizes.append(original_size)
                    crop_top_lefts.append(crop_top_left)

                    start_frames.append(0)
                    frame_strides.append(1)
                    fps.append(0)
                    sample_fps.append(0)
                    durations.append(0)

                return_dict.update(
                    {
                        "data_paths": image_paths,
                        "data": images,
                        "start_frames": start_frames,
                        "frame_strides": frame_strides,
                        "fps": fps,
                        "sample_fps": sample_fps,
                        "durations": durations,
                        "num_frames": num_frames,
                        "target_sizes": target_sizes,
                        "original_sizes": original_sizes,
                        "crop_top_lefts": crop_top_lefts,
                    }
                )
        elif self.latent_path_column:
            with measure_time("DATA_Latent", self.config.measure_time, timer=self.timer, verbose=False):
                latent_paths = items[self.latent_path_column]
                latents = []
                start_frames = []
                frame_strides = []
                fps = []
                sample_fps = []
                durations = []
                num_frames = []

                target_sizes = []
                original_sizes = []
                crop_top_lefts = []

                for latent_path in latent_paths:
                    try:
                        latent, start_frame, frame_stride, _fps, duration, target_size, original_size, crop_top_left, _num_frames = self.read_latent(
                            latent_path.strip()
                        )
                    except Exception as e:
                        print(f"Failed to read latent {latent_path}. {e}")
                        return {"failed": [True]}

                    latents.append(latent.unsqueeze(0))
                    start_frames.append(start_frame)
                    frame_strides.append(frame_stride)
                    fps.append(_fps)
                    sample_fps.append(_fps / frame_stride)
                    durations.append(duration)
                    num_frames.append(_num_frames)
                    target_sizes.append(target_size)
                    original_sizes.append(original_size)
                    crop_top_lefts.append(crop_top_left)

                return_dict.update(
                    {
                        "data_paths": latent_paths,
                        "data": latents,
                        "start_frames": start_frames,
                        "frame_strides": frame_strides,
                        "fps": fps,
                        "sample_fps": sample_fps,
                        "durations": durations,
                        "num_frames": num_frames,
                        "target_sizes": target_sizes,
                        "original_sizes": original_sizes,
                        "crop_top_lefts": crop_top_lefts,
                    }
                )
        elif self.image_latent_path_column:
            with measure_time("DATA_ImageLatent", self.config.measure_time, timer=self.timer, verbose=False):
                vae_latent_paths = items[self.image_latent_path_column]
                vae_latents = []

                frame_strides = []
                fps = []
                sample_fps = []
                num_frames = []
                target_sizes = []

                for vae_latent_path in vae_latent_paths:
                    try:
                        vae_latent = torch.load(vae_latent_path, map_location="cpu")
                        if isinstance(vae_latent, dict):
                            vae_latents.append(vae_latent["vae_latents"])
                        else:
                            vae_latents.append(vae_latent)
                    except Exception as e:
                        print(f"Failed to load vae latent embed {vae_latent_path}. {e}")
                        return {"failed": [True]}

                    # TODO: 图片和视频 vae latent 更优雅的读取方式
                    if isinstance(vae_latent, dict):
                        frame_strides.append(vae_latent["frame_strides"])
                        fps.append(vae_latent["fps"])
                        sample_fps.append(vae_latent["fps"] / vae_latent["frame_strides"])
                        num_frames.append(vae_latent["num_frames"])
                        target_sizes.append(vae_latent["target_sizes"])
                    else:
                        frame_strides.append(1)
                        fps.append(0)
                        sample_fps.append(0)
                        num_frames.append(1)
                        target_sizes.append((vae_latent.shape[-2], vae_latent.shape[-1]))

                return_dict.update(
                    {
                        "vae_latents": vae_latents,
                        "frame_strides": frame_strides,
                        "fps": fps,
                        "sample_fps": sample_fps,
                        "num_frames": num_frames,
                        "target_sizes": target_sizes,
                    }
                )
        elif self.video_path_column and self.ref_path_column and self.content_ref_path_column:  # rgb video & ref video & content ref video
            with measure_time("DATA_Video_Ref_Video_Content_Video", self.config.measure_time, timer=self.timer, verbose=False):
                
                def _parse_path(p):
                    if isinstance(p, str) and p.strip().startswith('[') and p.strip().endswith(']'):
                        try:
                            import json
                            parsed = json.loads(p)
                            if isinstance(parsed, list) and len(parsed) > 0 and isinstance(parsed[0], dict) and "value" in parsed[0]:
                                return parsed[0]["value"]
                        except:
                            pass
                    return p
                
                video_paths = [_parse_path(p) for p in items[self.video_path_column]]
                ref_video_paths = [_parse_path(p) for p in items[self.ref_path_column]]
                content_ref_video_paths = [_parse_path(p) for p in items[self.content_ref_path_column]]
                videos = []
                ref_videos = []
                content_ref_videos = []
                start_frames = []
                frame_strides = []
                fps = []
                sample_fps = []
                durations = []
                num_frames = []

                target_sizes = []
                original_sizes = []
                crop_top_lefts = []

                for video_path, ref_video_path, content_ref_video_path in zip(video_paths, ref_video_paths, content_ref_video_paths):
                    try:
                        frames, ref_frames, content_ref_frames, start_frame, frame_stride, _fps, duration, original_size = self.read_video(video_path, ref_video_path, content_ref_video_path)
                    # 分别为读出来的rgb帧, 开始的index, 帧之间的stride，原视频的fps，原视频的时长，原视频的大小height, width
                    except Exception as e:
                        print(f"Failed to read video {video_path}, ref video {ref_video_path}. {e}")
                        return {"failed": [True]}

                    start_frames.append(start_frame)
                    frame_strides.append(frame_stride)
                    fps.append(_fps)
                    sample_fps.append(_fps / frame_stride)
                    durations.append(duration)
                    frames, _, crop_top_left = self.preprocess_image(frames, resize=False)
                    ref_frames, _, crop_top_left = self.preprocess_image(ref_frames, resize=False)
                    content_ref_frames, _, crop_top_left = self.preprocess_image(content_ref_frames, resize=False)

                    num_frames.append(frames.shape[-4])  # TODO: 之前有问题是-3 需要注意
                    target_sizes.append(frames.shape[-2:])
                    original_sizes.append(original_size)
                    crop_top_lefts.append(crop_top_left)

                    videos.append(frames.unsqueeze(0))
                    ref_videos.append(ref_frames.unsqueeze(0))
                    content_ref_videos.append(content_ref_frames.unsqueeze(0))

                return_dict.update(
                    {
                        "data_paths": video_paths,
                        "ref_data_paths": ref_video_paths,
                        "content_ref_data_paths": content_ref_video_paths,
                        "data": videos,
                        "ref_data": ref_videos,
                        "content_ref_data": content_ref_videos,
                        "start_frames": start_frames,
                        "frame_strides": frame_strides,
                        "fps": fps,
                        "sample_fps": sample_fps,
                        "durations": durations,
                        "num_frames": num_frames,
                        "target_sizes": target_sizes,
                        "original_sizes": original_sizes,
                        "crop_top_lefts": crop_top_lefts,
                    }
                )

        # add for 
        if self.cam_rt_path_column:
            camera_rt_paths = items[self.cam_rt_path_column]
            pose_embeddings = []
            plucker_embeddings = []
            c2w_all = []
            # for camera_rt_path in camera_rt_paths:
            #     camera_rt = np.load(camera_rt_path)
            #     traj = camera_rt.transpose(0, 2, 1)
            #     c2ws = []
            #     for c2w in traj:
            #         c2w = c2w[:, [1, 2, 0, 3]]
            #         c2w[:3, 1] *= -1.
            #         c2w[:3, 3] /= 100
            #         c2ws.append(c2w)
            #     c2w_numpy = np.stack(c2ws, axis=0)[None, ...]
            for camera_rt_path in camera_rt_paths:
                camera_rt = np.load(camera_rt_path)
                # traj = camera_rt.transpose(0, 2, 1)
                traj = camera_rt
                c2ws = []
                for c2w in traj:
                    # c2w = c2w[:, [1, 2, 0, 3]]
                    # c2w[:3, 1] *= -1.
                    # c2w[:3, 3] /= 100
                    c2ws.append(np.linalg.inv(c2w))
                c2w_numpy = np.stack(c2ws, axis=0)[None, ...]
                if "0401" in camera_rt_path or "0404" in camera_rt_path:
                    k = np.array([387.10, 1188.48, 192, 336]).reshape(1, 1, 4)
                else:   
                    k = np.array([565.65, 1732.72, 192, 336]).reshape(1, 1, 4)

                k = np.repeat(k, 77, axis=1)
                # print(f"k shape is {k.shape}")

                # print(f"camera path is {camera_rt_path}")
                # print(f"c2w_numpy shape is {c2w_numpy.shape}")
                cond_cam_params = [Camera(cam_param) for cam_param in c2ws]
                relative_poses = []
                for i in range(len(cond_cam_params)):
                    relative_pose = get_relative_pose([cond_cam_params[0], cond_cam_params[i]])
                    relative_poses.append(torch.as_tensor(relative_pose)[:,:3,:][1])
                pose_embedding = torch.stack(relative_poses, dim=0)  
                
                pose_embeddings.append(pose_embedding.unsqueeze(0))
                plucker_embedding = ray_condition(torch.tensor(k), torch.tensor(c2w_numpy))
                # print(f"plucker shape is {plucker_embedding.shape}")
                plucker_embeddings.append(plucker_embedding)  # each shape is [1, f, h, w, 6]

            return_dict.update(
                {
                    "pose_embeddings": pose_embeddings,
                    "plucker_embeddings": plucker_embeddings,
                }
            )

        if self.condition_image_path_column:
            with measure_time("Conditional Image", self.config.measure_time, timer=self.timer, verbose=False):  
                first_image_paths = items[self.condition_image_path_column]  # first frame
                first_images = []
                start_frames = []
                frame_strides = []
                fps = []
                sample_fps = []
                durations = []
                num_frames = []

                target_sizes = []
                original_sizes = []
                crop_top_lefts = []

                for first_image_path in first_image_paths:  # first frame & ref video
                    try:
                        first_image = self.read_image(first_image_path)
                    except Exception as e:
                        print(f"Failed to read image {first_image_path}. {e}")
                        return {"failed": [True]}

                    first_image, original_size, crop_top_left = self.preprocess_image(first_image, resize=True)  # [b, c, h, w]
                    first_images.append(first_image.unsqueeze(1))  # b c h w -> b f c h w
                    
                    num_frames.append(1)  # TODO: 之前有问题是-3 需要注意
                    target_sizes.append(first_image.shape[-2:])
                    original_sizes.append(original_size)
                    crop_top_lefts.append(crop_top_left)

                    start_frames.append(0)
                    frame_strides.append(1)
                    fps.append(0)
                    sample_fps.append(0)
                    durations.append(0)
            
                return_dict.update(
                    {
                        "first_image_paths": first_image_paths,
                        "condition_image": first_images,
                        "start_frames": start_frames,
                        "frame_strides": frame_strides,
                        "fps": fps,
                        "sample_fps": sample_fps,
                        "durations": durations,
                        "num_frames": num_frames,
                        "target_sizes": target_sizes,
                        "original_sizes": original_sizes,
                        "crop_top_lefts": crop_top_lefts,
                    }
                )

        if self.motion_bucket_id_column:
            return_dict.update(
                {
                    "motion_bucket_ids": items[self.motion_bucket_id_column],
                }
            )

        if self.control_columns:
            control_frames = []
            multi_control_paths = []
            multi_control_frames = []

            for control_column in self.control_columns:
                control_paths = items[control_column]
                control_frames = []
                for control_path in control_paths:
                    frames, *_ = self.read_video(control_path)
                    frames = self.preprocess_image(frames)
                    frames = frames.mul(0.5).add(0.5)  # [-1, 1] -> [0, 1]
                    control_frames.append(frames)

                multi_control_paths.append(control_paths)
                multi_control_frames.append(control_frames)

            multi_control_paths = [list(row) for row in zip(*multi_control_paths)]  # list转置
            multi_control_frames = [list(row) for row in zip(*multi_control_frames)]  # list转置

            return_dict.update(
                {
                    "control_paths": multi_control_paths,
                    "control_frames": multi_control_frames,
                }
            )

        return return_dict

    def read_image(self, path):
        # BUG for pyvips, set use_pyvips = False by default
        if self.config.use_pyvips and pyvips_installed:
            img_rgb = pyvips.Image.new_from_file(path, access="sequential")
        else:
            img_rgb = cv2.imread(path)[..., ::-1]
        frames = torch.from_numpy(np.ascontiguousarray(img_rgb))[None]
        assert frames.ndim == 4, f"{path} ndim={frames.ndim}"
        return frames

    def read_video(self, path, ref_path=None, content_ref_path=None):
        cap = cv2.VideoCapture(path)
        assert cap.isOpened(), "无法打开视频文件或文件路径错误，请检查路径是否正确并确保文件格式被支持。"
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)

        if self.config.resize_video:  # Training, we can use Navit
            resize_width, resize_height = self.get_target_size(width, height)  # 得到对长宽比进行约束之后的视频， 使用NAViT即为可变长宽比
        else:
            resize_width, resize_height = int(width), int(height)
        
        if self.config.gpu_decode:  # 使用GPU对视频进行解码 这里不使用GPU
            # get local rank
            local_rank = torch.distributed.get_rank() % torch.cuda.device_count()
            ctx = decord.gpu(local_rank)
        else:
            ctx = decord.cpu(0)

        reader = decord.VideoReader(path, ctx=ctx, height=resize_height, width=resize_width)
        length = len(reader)

        if ref_path and ".mp4" in ref_path:
            ref_cap = cv2.VideoCapture(ref_path)
            assert ref_cap.isOpened(), f"无法打开ref video文件或文件路径错误，请检查路径是否正确并确保文件格式被支持。path is {ref_path}"
            ref_reader = decord.VideoReader(ref_path, ctx=ctx, height=resize_height, width=resize_width)
            ref_length = len(ref_reader)
            length = min(length, ref_length)  # Add for when length of ref shorter than rgb

        if content_ref_path and ".mp4" in content_ref_path:
            content_ref_cap = cv2.VideoCapture(content_ref_path)
            assert content_ref_cap.isOpened(), f"无法打开content ref video文件或文件路径错误，请检查路径是否正确并确保文件格式被支持。path is {content_ref_path}"
            content_ref_reader = decord.VideoReader(content_ref_path, ctx=ctx, height=resize_height, width=resize_width)
            content_ref_length = len(content_ref_reader)
            length = min(length, content_ref_length)  # Add for when length of ref shorter than rgb

        if self.config.gpu_decode:
            length = length - 1  # Avoid hang at last frame  使用GPU解码视频要注意最后一帧被挂起的情况

        # NOTE 不够合理，应先换算stride，预处理视频，但改动太大，暂时搁置，可在数据侧剔除异常fps数据
        fps = min(reader.get_avg_fps(), self.max_fps)  # 取平均帧率和最大帧率之间的小值
        duration = length / fps
        num_frames = (self.num_frames - 1) // self.temporal_unit_size * self.temporal_unit_size + 1  # 变成4的倍数 + 1

        if self.is_variable_duration:
            # 触发可变时常时，无视self.num_frames，根据self.max_duration重新计算num_frames
            duration = min(duration, self.max_duration)
            length = int(duration * fps)
            duration = length / fps  # 根据length更新duration
            num_frames = length

            if self.sample_type == "fps":
                frame_stride = max(1, round(fps / self.sample_fps))
            elif self.sample_type == "stride":
                frame_stride = max(1, self.sample_stride)
            elif self.sample_type == "full":
                frame_stride = 1
            elif self.sample_type == "random":
                max_frame_stride = length - 1
                frame_stride = random.randint(1, max_frame_stride)
            else:
                raise NotImplementedError
            num_frames = (num_frames - 1) // frame_stride // self.temporal_unit_size * self.temporal_unit_size + 1
        else:
            if num_frames == 1:
                frame_stride_full = sys.maxsize  # python解释器可以处理的最大的数字
            else:
                frame_stride_full = max(1, round((length - 1) / (num_frames - 1)))  # 整个视频的长度 // 

            if self.sample_type == "fps":
                frame_stride = max(1, round(fps / self.sample_fps))  # fps是视频自身的fps, sample_fps is 15
                frame_stride = min(frame_stride, frame_stride_full)  
            elif self.sample_type == "stride":
                frame_stride = max(1, self.sample_stride)
                frame_stride = min(frame_stride, frame_stride_full)
            elif self.sample_type == "full":
                frame_stride = frame_stride_full
            elif self.sample_type == "random":
                max_frame_stride = min(self.config.max_frame_stride, frame_stride_full)
                frame_stride = random.randint(1, max_frame_stride)
            else:
                raise NotImplementedError
            # frame_stride=1
            # print(f"frame stride is {frame_stride}, type is {self.sample_type}")
        num_frames = min(num_frames, self.max_num_frames)
        max_start_frame = max(0, length - ((num_frames - 1) * frame_stride + 1))
        if self.sample_position == "center":
            start_frame = max_start_frame // 2
        elif self.sample_position == "first":
            start_frame = 0
        elif self.sample_position == "last":
            start_frame = max_start_frame
        elif self.sample_position == "random":
            start_frame = random.randint(0, max_start_frame)
        else:
            raise NotImplementedError

        if num_frames == 1:
            frame_stride = 1
            frame_indexes = [start_frame]
        else:
            frame_indexes = range(start_frame, start_frame + num_frames * frame_stride, frame_stride)
            frame_indexes = [min(frame_index, length - 1) for frame_index in frame_indexes]

        if self.config.gpu_decode and self.config.gpu_read_chunk_size is not None:
            frame_tensors_all = []
            for i in range(0, len(frame_indexes), self.config.gpu_read_chunk_size):
                frame_indexes_chunk = frame_indexes[i : i + self.config.gpu_read_chunk_size]
                frames_chunk = reader.get_batch(frame_indexes_chunk)
                frame_tensors_all.append(frames_chunk)
            frames = torch.cat(frame_tensors_all, dim=0)

            if ref_path:
                if ".mp4" in ref_path:
                    ref_frame_tensors_all = []
                    for i in range(0, len(frame_indexes), self.config.gpu_read_chunk_size):
                        frame_indexes_chunk = frame_indexes[i : i + self.config.gpu_read_chunk_size]
                        frames_chunk = ref_reader.get_batch(frame_indexes_chunk)
                        ref_frame_tensors_all.append(frames_chunk)
                    ref_frames = torch.cat(ref_frame_tensors_all, dim=0)
                else:
                    ref_frames = self.read_image(ref_path)

            if content_ref_path:
                if ".mp4" in content_ref_path:
                    content_ref_frame_tensors_all = []
                    for i in range(0, len(frame_indexes), self.config.gpu_read_chunk_size):
                        frame_indexes_chunk = frame_indexes[i : i + self.config.gpu_read_chunk_size]
                        frames_chunk = content_ref_reader.get_batch(frame_indexes_chunk)
                        content_ref_frame_tensors_all.append(frames_chunk)
                    content_ref_frames = torch.cat(content_ref_frame_tensors_all, dim=0)
                else:
                    content_ref_frames = self.read_image(content_ref_path)

        else:
            frames = reader.get_batch(frame_indexes)
            if ref_path:
                if ".mp4" in ref_path:
                    ref_frames = ref_reader.get_batch(frame_indexes)
                else:
                    ref_frames = self.read_image(ref_path)
            if content_ref_path:
                if ".mp4" in content_ref_path:
                    content_ref_frames = content_ref_reader.get_batch(frame_indexes)
                else:
                    content_ref_frames = self.read_image(content_ref_path)
        
        if ref_path and not content_ref_path:
            return frames, ref_frames, start_frame, frame_stride, fps, duration, (height, width)
        elif ref_path and content_ref_path:
            return frames, ref_frames, content_ref_frames, start_frame, frame_stride, fps, duration, (height, width)
        else:     
            return frames, start_frame, frame_stride, fps, duration, (height, width)


    @torch.no_grad()
    def read_latent(self, path):
        # 读取离线3dvae
        latent_dict = torch.load(path, map_location="cpu")
        latent_parts = latent_dict["data"].detach()
        num_part_frames = (latent_parts.shape[2] - 1) * self.vae_temporal_scale_factor + 1
        num_select_parts = (self.num_frames - 1) // num_part_frames + 1

        # NOTE: FIT Token Merge
        if self.spatial_token_merge_size is not None:
            height, width = latent_parts.shape[-2], latent_parts.shape[-1]
            crop_height = height // (self.spatial_token_merge_size * self.spatial_patch_size) * (self.spatial_token_merge_size * self.spatial_patch_size)
            crop_width = width // (self.spatial_token_merge_size * self.spatial_patch_size) * (self.spatial_token_merge_size * self.spatial_patch_size)
            latent_parts = latent_parts[:, :, :, :crop_height, :crop_width]

        num, _, f, h, w = latent_parts.shape
        assert num >= num_select_parts, f"num_select_parts({num_select_parts})> num({num})"
        target_size = (h * self.vae_spatial_scale_factor, w * self.vae_spatial_scale_factor)
        num_part_frames = (f - 1) * self.vae_temporal_scale_factor + 1

        frame_stride = latent_dict["frame_stride"]
        num_sample_frames = num_part_frames * num
        num_original_frames = 1 + (num_sample_frames - 1) * frame_stride
        fps = latent_dict["fps"]  # original fps
        if fps > 0:
            duration = num_original_frames / fps
        else:
            duration = 0

        if self.sample_position == "center":
            start_idx = num // 2
        elif self.sample_position == "first":
            start_idx = 0
        elif self.sample_position == "last":
            start_idx = num - num_select_parts
        elif self.sample_position == "random":
            start_idx = random.randint(0, num - num_select_parts)
        else:
            raise NotImplementedError

        start_frame = num_part_frames * start_idx
        num_last_part_latent_frames = int((self.num_frames - (num_part_frames * (num_select_parts - 1)) - 1) / self.vae_temporal_scale_factor + 1)
        if self.num_frames == 1:
            last_part_latent_frame_indexes = [0]
        else:
            last_part_latent_frame_indexes = range(0, num_last_part_latent_frames)

        latent = latent_parts[start_idx + num_select_parts - 1, :, last_part_latent_frame_indexes]  # c f h w
        latent = rearrange(latent, "c f h w -> f c h w")
        if num_select_parts > 1:
            select_part = latent_parts[start_idx : start_idx + num_select_parts - 1]
            select_part = rearrange(select_part, "n c f h w -> (n f) c h w")
            latent = torch.cat((select_part, latent), dim=0)

        if self.temporal_token_merge_size is not None:
            num_frames = latent.shape[0]
            num_crop_frames = (
                num_frames // (self.temporal_token_merge_size * self.temporal_patch_size) * (self.temporal_token_merge_size * self.temporal_patch_size)
            )
            latent = latent[:num_crop_frames]

        return latent, start_frame, frame_stride, fps, duration, target_size, latent_dict["original_size"], latent_dict["crop_top_left"], self.num_frames

    def get_target_size(self, width, height):
        aspect_ratio_image = width / height
        aspect_ratio_target = self.width / self.height
        if self.is_variable_aspect_ratio:
            area = self.max_area
            resize_height = math.sqrt(area / aspect_ratio_image)
            resize_width = round(resize_height * aspect_ratio_image)
            resize_height = round(resize_height)
        else:
            # 短边resize
            if aspect_ratio_image > aspect_ratio_target:
                # 更宽
                resize_height = self.height
                resize_width = round(self.height * aspect_ratio_image)
            else:
                resize_width = self.width
                resize_height = round(self.width / aspect_ratio_image)
        return resize_width, resize_height

    def preprocess_image(self, frames, resize=True):
        frames = frames.float().permute(0, 3, 1, 2)  # f h w c -> f c h w
        _, _, height, width = frames.shape
        original_size = (height, width)

        if resize:
            resize_width, resize_height = self.get_target_size(width, height)
            train_resize = transforms.Resize((resize_height, resize_width), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
            frames = train_resize(frames)
        else:
            resize_height, resize_width = height, width
            original_size = None

        # 为保证是vae、patchify down scale size的整数倍，对图片进行crop
        # 关于为什么选择crop，而不是resize，参考：SDXL image bucket的构建方式
        # 详见 https://github.com/NovelAI/novelai-aspect-ratio-bucketing/tree/main，Image Loading部分
        crop_height = resize_height // self.spatial_unit_size * self.spatial_unit_size
        crop_width = resize_width // self.spatial_unit_size * self.spatial_unit_size
        frames = frames[:, :, :crop_height, :crop_width]

        _, _, height, width = frames.shape
        if self.crop_type == "center":
            y1 = max(0, int(round((height - self.height) / 2.0)))
            x1 = max(0, int(round((width - self.width) / 2.0)))
            frames = self.train_crop(frames)
        elif self.crop_type == "random":
            y1, x1, h, w = self.train_crop.get_params(frames, self.resolution)
            frames = crop(frames, y1, x1, h, w)
        else:  # no crop
            y1 = x1 = 0
        if self.random_flip and random.random() < 0.5:
            x1 = width - x1
            frames = self.train_flip(frames)
        crop_top_left = (y1, x1)
        frames = self.train_transforms(frames)
        frames = frames.clip(-1, 1)
        return frames, original_size, crop_top_left

    def collate_fn(self, examples):
        len_batch = len(examples)  # original batch length
        examples = list(filter(lambda x: "failed" not in x, examples))  # filter out all the Nones
        if len_batch > len(examples):  # source all the required samples from the original dataset at random
            diff = len_batch - len(examples)
            count = 0
            while diff != 0:
                if count >= 10:
                    raise Exception("Encounter 10 bad samples continuously! Exit!")
                sample = self.dataset[np.random.randint(0, len(self.dataset))]
                if "failed" in sample:
                    count += 1
                    continue
                examples.append(sample)
                diff -= 1

        return_dict = {}

        if self.video_path_column or self.image_path_column or self.latent_path_column:
            data_paths = [example["data_paths"] for example in examples]
            data = [example["data"] for example in examples]
            if (not self.is_variable_duration and not self.is_variable_aspect_ratio) or self.config.batch_size == 1:
                # 非可变长宽比/可变时长, 或 batch_size为1
                data = torch.vstack(data).contiguous()
            start_frames = [example["start_frames"] for example in examples]
            frame_strides = [example["frame_strides"] for example in examples]
            fps = [example["fps"] for example in examples]
            sample_fps = [example["sample_fps"] for example in examples]
            durations = [example["durations"] for example in examples]
            num_frames = [example["num_frames"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]

            data_key = "vae_latents" if self.latent_path_column else "data"
            return_dict.update(
                {
                    "data_paths": data_paths,
                    data_key: data,
                    "start_frames": start_frames,
                    "frame_strides": frame_strides,
                    "fps": fps,
                    "sample_fps": sample_fps,
                    "durations": durations,
                    "num_frames": num_frames,
                    "target_sizes": target_sizes,
                    "original_sizes": original_sizes,
                    "crop_top_lefts": crop_top_lefts,
                }
            )
        elif self.image_latent_path_column:
            vae_latents = [example["vae_latents"] for example in examples]
            vae_latents = torch.stack(vae_latents).contiguous()
            if vae_latents.ndim == 4:  # B C H W -> B 1 C H W
                vae_latents = vae_latents.unsqueeze(1)
            frame_strides = [example["frame_strides"] for example in examples]
            fps = [example["fps"] for example in examples]
            sample_fps = [example["sample_fps"] for example in examples]
            num_frames = [example["num_frames"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            return_dict.update(
                {
                    "vae_latents": vae_latents,
                    "frame_strides": frame_strides,
                    "fps": fps,
                    "sample_fps": sample_fps,
                    "num_frames": num_frames,
                    "target_sizes": target_sizes,
                }
            )
        
        if self.condition_image_path_column:
            first_image_paths = [example["first_image_paths"] for example in examples]
            condition_image = [example["condition_image"] for example in examples]
            # start_frames = [example["start_frames"] for example in examples]
            # frame_strides = [example["frame_strides"] for example in examples]
            # fps = [example["fps"] for example in examples]
            # sample_fps = [example["sample_fps"] for example in examples]
            # durations = [example["durations"] for example in examples]
            # num_frames = [example["num_frames"] for example in examples]
            # target_sizes = [example["target_sizes"] for example in examples]
            # original_sizes = [example["original_sizes"] for example in examples]
            # crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            return_dict.update(
                {
                    "first_image_paths": first_image_paths,
                    "condition_image": condition_image,
                    # "start_frames": start_frames,
                    # "frame_strides": frame_strides,
                    # "fps": fps,
                    # "sample_fps": sample_fps,
                    # "durations": durations,
                    # "num_frames": num_frames,
                    # "target_sizes": target_sizes,
                    # "original_sizes": original_sizes,
                    # "crop_top_lefts": crop_top_lefts,
                }
            )


        # add for ref video
        if self.ref_path_column:
            ref_data_paths = [example["ref_data_paths"] for example in examples]
            ref_data = [example["ref_data"] for example in examples]
            start_frames = [example["start_frames"] for example in examples]
            frame_strides = [example["frame_strides"] for example in examples]
            fps = [example["fps"] for example in examples]
            sample_fps = [example["sample_fps"] for example in examples]
            durations = [example["durations"] for example in examples]
            num_frames = [example["num_frames"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            return_dict.update(
                {
                    "ref_data_paths": ref_data_paths,
                    "ref_data": ref_data,
                    "start_frames": start_frames,
                    "frame_strides": frame_strides,
                    "fps": fps,
                    "sample_fps": sample_fps,
                    "durations": durations,
                    "num_frames": num_frames,
                    "target_sizes": target_sizes,
                    "original_sizes": original_sizes,
                    "crop_top_lefts": crop_top_lefts,
                }
            )
        
        # add for content ref video
        if self.content_ref_path_column:
            content_ref_data_paths = [example["content_ref_data_paths"] for example in examples]
            content_ref_data = [example["content_ref_data"] for example in examples]
            start_frames = [example["start_frames"] for example in examples]
            frame_strides = [example["frame_strides"] for example in examples]
            fps = [example["fps"] for example in examples]
            sample_fps = [example["sample_fps"] for example in examples]
            durations = [example["durations"] for example in examples]
            num_frames = [example["num_frames"] for example in examples]
            target_sizes = [example["target_sizes"] for example in examples]
            original_sizes = [example["original_sizes"] for example in examples]
            crop_top_lefts = [example["crop_top_lefts"] for example in examples]
            return_dict.update(
                {
                    "content_ref_data_paths": content_ref_data_paths,
                    "content_ref_data": content_ref_data,
                    "start_frames": start_frames,
                    "frame_strides": frame_strides,
                    "fps": fps,
                    "sample_fps": sample_fps,
                    "durations": durations,
                    "num_frames": num_frames,
                    "target_sizes": target_sizes,
                    "original_sizes": original_sizes,
                    "crop_top_lefts": crop_top_lefts,
                }
            )

        if self.cam_rt_path_column:
            pose_embeddings = [example["pose_embeddings"] for example in examples]
            plucker_embeddings = [example["plucker_embeddings"] for example in examples]
            return_dict.update(
                {
                    "pose_embeddings": pose_embeddings,
                    "plucker_embeddings": plucker_embeddings,
                }
            )

        if self.caption_column:
            prompts = [example["prompts"] for example in examples]
            return_dict.update(
                {
                    "prompts": prompts,
                }
            )

        if self.t5_prompt_embed_column:
            prompt_embeds = [example["t5_prompt_embeds"] for example in examples]
            return_dict.update(
                {
                    "t5_prompt_embeds": prompt_embeds,
                }
            )

        if self.clip_prompt_embed_column:
            prompt_embeds = [example["clip_prompt_embeds"] for example in examples]
            return_dict.update(
                {
                    "clip_prompt_embeds": prompt_embeds,
                }
            )

        if self.index_column:
            index = [example["index"] for example in examples]
            return_dict.update(
                {
                    "index": index,
                }
            )

        if self.motion_bucket_id_column:
            motion_bucket_ids = [example["motion_bucket_ids"] for example in examples]
            return_dict.update(
                {
                    "motion_bucket_ids": motion_bucket_ids,
                }
            )

        if self.control_columns:
            control_paths = [example["control_paths"] for example in examples]
            control_frames = [example["control_frames"] for example in examples]
            control_frames = [torch.vstack(row) for row in zip(*control_frames)]  # list转置
            return_dict.update(
                {
                    "control_paths": control_paths,
                    "controls": control_frames,
                }
            )

        if self.class_labels_column is not None:
            class_labels = [example["class_labels"] for example in examples]
            return_dict.update(
                {
                    "class_labels": class_labels,
                }
            )

        return return_dict
