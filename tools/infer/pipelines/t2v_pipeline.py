import random
from dataclasses import dataclass, field
from typing import Any, Dict, Type

from diffusers.pipelines import StableDiffusionPipeline

from ..configs.config_utils import to_immutable_dict
from .base_pipeline import BasePipelineConfig, PipelineMixin


@dataclass
class T2VPipelineConfig(BasePipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: T2VPipeline)
    """target class to instantiate"""

    proportion_empty_prompts: float = 0.1
    """Proportion of empty prompts to use."""

    call: Dict[str, Any] = to_immutable_dict(
        {
            "height": 360,
            "width": 640,
            "num_inference_steps": 25,
        }
    )
    """The inference call arguments for the pipeline."""


class T2VPipeline(StableDiffusionPipeline, PipelineMixin):
    def forward(self, batch):
        videos, prompts = batch["videos"], batch["prompts"]
        batch_size = videos.shape[0]
        self.unet.set_num_videos(batch_size)
        latents = self.video2latents(videos, out_pattern="(b f) c h w")
        prompt_masks = [random.random() > self.pipeline_config.proportion_empty_prompts for i in range(batch_size)]
        prompts = [prompt if mask else "" for mask, prompt in zip(prompt_masks, prompts)]

        prompt_embeds, _ = self.encode_prompt(prompts, device=latents.device, num_images_per_prompt=1, do_classifier_free_guidance=False)
        prompt_embeds = prompt_embeds.repeat_interleave(self.unet.num_frames, dim=0)

        scaled_input, scaled_sigmas, target, loss_weight = self.edm_step(batch_size, latents)
        model_output = self.unet(scaled_input, scaled_sigmas, prompt_embeds).sample
        return self.compute_loss({"pred": model_output, "target": target, "loss_weight": loss_weight})

    def val_t2v(self, batch):
        kwargs = self.prepare_call_kwargs(self, batch)
        return self(**kwargs).images.mul(2).sub(1)
