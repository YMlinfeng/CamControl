import random
from dataclasses import dataclass, field
from typing import Type

import torch
from diffusers.pipelines import StableDiffusionXLPipeline

from .base_pipeline import PipelineMixin
from .t2v_pipeline import T2VPipelineConfig


@dataclass
class T2VXLPipelineConfig(T2VPipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: T2VXLPipeline)
    """target class to instantiate"""


class T2VXLPipeline(StableDiffusionXLPipeline, PipelineMixin):
    def forward(self, batch):
        videos, prompts = batch["videos"], batch["prompts"]
        batch_size = videos.shape[0]
        self.unet.set_num_videos(batch_size)
        latents = self.video2latents(videos, out_pattern="(b f) c h w")

        original_sizes, crop_top_lefts, target_sizes = batch["original_sizes"], batch["crop_top_lefts"], batch["target_sizes"]

        time_ids = self.get_time_ids(original_sizes, crop_top_lefts, target_sizes, videos.device, videos.dtype)
        time_ids = time_ids.repeat_interleave(self.unet.num_frames, dim=0)

        prompt_masks = [random.random() > self.pipeline_config.proportion_empty_prompts for i in range(batch_size)]
        prompts = [prompt if mask else "" for mask, prompt in zip(prompt_masks, prompts)]

        prompt_embeds, _, pooled_prompt_embeds, _ = self.encode_prompt(prompts, do_classifier_free_guidance=False)
        prompt_embeds = prompt_embeds.repeat_interleave(self.unet.num_frames, dim=0)
        pooled_prompt_embeds = pooled_prompt_embeds.repeat_interleave(self.unet.num_frames, dim=0)
        unet_added_conditions = {"time_ids": time_ids, "text_embeds": pooled_prompt_embeds}

        scaled_input, scaled_sigmas, target, loss_weight = self.edm_step(batch_size, latents)
        model_output = self.unet(scaled_input, scaled_sigmas, prompt_embeds, added_cond_kwargs=unet_added_conditions).sample
        return self.compute_loss({"pred": model_output, "target": target, "loss_weight": loss_weight})

    def get_time_ids(self, original_sizes, crop_top_lefts, target_sizes, device, dtype):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        add_time_ids_list = []
        for original_size, crops_coords_top_left, target_size in zip(original_sizes, crop_top_lefts, target_sizes):
            target_size = [self.pipeline_config.call["height"], self.pipeline_config.call["width"]]
            add_time_ids = list(original_size + crops_coords_top_left + target_size)
            add_time_ids = torch.tensor([add_time_ids])
            add_time_ids = add_time_ids.to(device, dtype=dtype)
            add_time_ids_list.append(add_time_ids)
        add_time_ids = torch.cat(add_time_ids_list)
        return add_time_ids

    def val_t2v(self, batch):
        kwargs = self.prepare_call_kwargs(self, batch)
        return self(**kwargs).images.mul(2).sub(1)
