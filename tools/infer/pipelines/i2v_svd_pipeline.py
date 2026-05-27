from dataclasses import dataclass, field
from typing import Type

import torch
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange, repeat

from ..utils import prepare_image_for_clip
from .base_pipeline import BasePipelineConfig, EDMTrainConfig, PipelineMixin


@dataclass
class I2VSVDPipelineConfig(BasePipelineConfig):
    """Configuration for Pipeline instantiation"""

    proportion_empty_images: float = 0.1
    """Proportion of empty images to use."""

    _target: Type = field(default_factory=lambda: I2VSVDPipeline)
    """target class to instantiate"""

    edm_config: EDMTrainConfig = EDMTrainConfig()
    """EDM config"""


class I2VSVDPipeline(StableVideoDiffusionPipeline, PipelineMixin):
    def _encode_image(self, image, device, num_videos_per_prompt, do_classifier_free_guidance):
        dtype = next(self.image_encoder.parameters()).dtype

        if not isinstance(image, torch.Tensor):
            image = self.image_processor.pil_to_numpy(image)
            image = self.image_processor.numpy_to_pt(image)

        # NOTE: here is modified from the original SVD pipeline to properly handle image tensor
        # We normalize the image before resizing to match with the original implementation.
        # Then we unnormalize it after resizing.
        # image = image * 2.0 - 1.0 # NOTE: we assume image is in [-1, 1]
        image = prepare_image_for_clip(image)

        image = image.to(device=device, dtype=dtype)
        image_embeddings = self.image_encoder(image).image_embeds
        image_embeddings = image_embeddings.unsqueeze(1)

        # duplicate image embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = image_embeddings.shape
        image_embeddings = image_embeddings.repeat(1, num_videos_per_prompt, 1)
        image_embeddings = image_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])

        return image_embeddings

    @torch.no_grad()
    def _encode_image_batch(self, image):
        image = prepare_image_for_clip(image)
        image_embeddings = self.image_encoder(image).image_embeds
        return image_embeddings

    def forward(self, batch):
        videos = batch["videos"]
        batch_size = videos.shape[0]
        self.unet.set_num_videos(batch_size)

        latents = self.video2latents(videos, out_pattern="b f c h w")

        noise_scale_condition = torch.exp(torch.normal(-3, 0.5, size=(batch_size,), device=videos.device))  # See svd paper page 20

        first_frames = videos[:, 0]
        encoder_hidden_states = self._encode_image_batch(first_frames).unsqueeze(1)  # sequence length is 1

        noise = randn_tensor(first_frames.shape, device=first_frames.device, dtype=first_frames.dtype)
        noised_first_frames = first_frames + noise * noise_scale_condition[:, None, None, None]
        condition_latents = self.vae.encode(noised_first_frames.to(torch.float16)).latent_dist.sample()

        conditioning_dropout_prob = self.pipeline_config.proportion_empty_images
        # Conditioning dropout to support classifier-free guidance during inference. For more details check out the section 3.2.1 of the paper https://arxiv.org/abs/2211.09800.
        random_p = torch.rand(batch_size, device=condition_latents.device)
        # Sample masks for the condition 1 (usually prompts).
        prompt_mask = random_p < 2 * conditioning_dropout_prob
        prompt_mask = prompt_mask.reshape(batch_size, 1, 1)
        # Final prompt conditioning.
        null_conditioning = torch.zeros_like(encoder_hidden_states)
        encoder_hidden_states = torch.where(prompt_mask, null_conditioning, encoder_hidden_states)
        # Sample masks for the condition 2 (usually images).
        image_mask_dtype = condition_latents.dtype
        image_mask = 1 - ((random_p >= conditioning_dropout_prob).to(image_mask_dtype) * (random_p < 3 * conditioning_dropout_prob).to(image_mask_dtype))
        image_mask = image_mask.reshape(batch_size, 1, 1, 1)
        # Final image conditioning.
        condition_latents = image_mask * condition_latents

        condition_latents = repeat(condition_latents, "b c h w -> b f c h w", f=self.unet.num_frames)

        added_time_ids = torch.tensor(
            [[batch["fps"][b] / batch["frame_strides"][b], batch["motion_bucket_ids"][b], noise_scale_condition[b]] for b in range(batch_size)],
            dtype=condition_latents.dtype,
            device=condition_latents.device,
        )

        scaled_input, scaled_sigmas, target, loss_weight = self.edm_step(batch_size, latents)
        noisy_latents_with_condition = torch.cat([scaled_input, condition_latents], dim=2)
        model_output = self.unet(noisy_latents_with_condition, scaled_sigmas, encoder_hidden_states, added_time_ids=added_time_ids, return_dict=True).sample
        return self.compute_loss({"pred": model_output, "target": target, "loss_weight": loss_weight})

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        pipeline.vae.to(torch.float16)
        call_kwargs = super().prepare_call_kwargs(pipeline, batch, **kwargs)
        call_kwargs.pop("prompt", None)
        call_kwargs.pop("negative_prompt", None)
        call_kwargs.pop("guidance_scale", None)
        call_kwargs["image"] = batch["videos"][:, 0].to(torch.float32)
        return call_kwargs

    def val_i2v(self, batch):
        kwargs = self.prepare_call_kwargs(self, batch)
        res = self(**kwargs).frames
        res = torch.cat(res, dim=0).mul(2).sub(1)  # (b f) c h w  \in [-1, 1]
        return res
