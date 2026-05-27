from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Type, Union

import torch
from diffusers.models import AutoencoderKLTemporalDecoder, UNetSpatioTemporalConditionModel
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import (
    StableVideoDiffusionPipeline,
    StableVideoDiffusionPipelineOutput,
    tensor2vid,
)
from diffusers.schedulers import EulerDiscreteScheduler
from diffusers.utils import deprecate
from transformers import AutoModel, AutoTokenizer, CLIPImageProcessor, CLIPVisionModelWithProjection

from .base_pipeline import BasePipelineConfig, EDMTrainConfig, PipelineMixin


@dataclass
class T2VSVDPipelineConfig(BasePipelineConfig):
    """Configuration for Pipeline instantiation"""

    proportion_empty_prompts: float = 0.1
    """Proportion of empty prompts to use."""

    _target: Type = field(default_factory=lambda: T2VSVDPipeline)
    """target class to instantiate"""

    edm_config: EDMTrainConfig = EDMTrainConfig()
    """EDM config"""


class T2VSVDPipeline(StableVideoDiffusionPipeline, PipelineMixin):
    def __init__(
        self,
        vae: AutoencoderKLTemporalDecoder,
        image_encoder: CLIPVisionModelWithProjection,
        unet: UNetSpatioTemporalConditionModel,
        scheduler: EulerDiscreteScheduler,
        feature_extractor: CLIPImageProcessor,
        text_encoder: AutoModel,
        tokenizer: AutoTokenizer,
    ):
        unet.replace_transformer_spatio_temporal_forward()

        super().__init__(vae, image_encoder, unet, scheduler, feature_extractor)

        self.register_modules(text_encoder=text_encoder, tokenizer=tokenizer)

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.encode_prompt with num_images_per_prompt -> num_videos_per_prompt
    @torch.no_grad()
    def _encode_prompt(self, prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt):
        batch_size = len(prompt) if isinstance(prompt, list) else 1

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])

            deprecation_message = (
                "The following part of your input was truncated because CLIP can only handle sequences up to"
                f" {self.tokenizer.model_max_length} tokens: {removed_text}"
            )
            deprecate("Prompt was truncated!", "1.0.0", deprecation_message, standard_warn=False)

        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.to(device)
        else:
            attention_mask = None

        text_embeddings = self.text_encoder(text_inputs, attention_mask=attention_mask, device=device)
        text_embeddings = text_embeddings[0]

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        bs_embed, seq_len, _ = text_embeddings.shape
        text_embeddings = text_embeddings.repeat(1, num_videos_per_prompt, 1)
        text_embeddings = text_embeddings.view(bs_embed * num_videos_per_prompt, seq_len, -1)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            uncond_tokens: List[str]
            if negative_prompt is None:
                uncond_tokens = [""] * batch_size
            elif type(prompt) is not type(negative_prompt):
                raise TypeError(f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !=" f" {type(prompt)}.")
            elif isinstance(negative_prompt, str):
                uncond_tokens = [negative_prompt]
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )
            else:
                uncond_tokens = negative_prompt

            max_length = text_input_ids.shape[-1]
            uncond_input = self.tokenizer(
                uncond_tokens,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )

            if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
                attention_mask = uncond_input.attention_mask.to(device)
            else:
                attention_mask = None

            uncond_embeddings = self.text_encoder(uncond_input, attention_mask=attention_mask, device=device)
            uncond_embeddings = uncond_embeddings[0]

            # duplicate unconditional embeddings for each generation per prompt, using mps friendly method
            seq_len = uncond_embeddings.shape[1]
            uncond_embeddings = uncond_embeddings.repeat(1, num_videos_per_prompt, 1)
            uncond_embeddings = uncond_embeddings.view(batch_size * num_videos_per_prompt, seq_len, -1)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

        return text_embeddings

    # Modified from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion.StableVideoDiffusionPipeline._get_add_time_ids, removed noise_aug_strength in I2V
    def _get_add_time_ids(
        self,
        fps,
        motion_bucket_id,
        dtype,
        batch_size,
        num_videos_per_prompt,
        do_classifier_free_guidance,
    ):
        # NOTE: removed noise_aug_strength
        add_time_ids = [fps, motion_bucket_id]

        passed_add_embed_dim = self.unet.config.addition_time_embed_dim * len(add_time_ids)
        expected_add_embed_dim = self.unet.add_embedding.linear_1.in_features

        if expected_add_embed_dim != passed_add_embed_dim:
            raise ValueError(
                f"Model expects an added time embedding vector of length {expected_add_embed_dim}, but a vector of {passed_add_embed_dim} was created. The model has an incorrect config. Please check `unet.config.time_embedding_type` and `text_encoder_2.config.projection_dim`."
            )

        add_time_ids = torch.tensor([add_time_ids], dtype=dtype)
        add_time_ids = add_time_ids.repeat(batch_size * num_videos_per_prompt, 1)

        if do_classifier_free_guidance:
            add_time_ids = torch.cat([add_time_ids, add_time_ids])

        return add_time_ids

    # Modified from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion.StableVideoDiffusionPipeline.call, removed image embedding and first frame condition, add text embedding
    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        guidance_scale: float = 12.5,
        fps: int = 7,
        motion_bucket_id: int = 127,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
    ):
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames

        # 2. Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        else:
            batch_size = len(prompt)

        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # NOTE: image embedding -> text embedding
        text_embeddings = self._encode_prompt(prompt, device, num_videos_per_prompt, do_classifier_free_guidance, negative_prompt=negative_prompt)

        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        fps = fps - 1

        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            text_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance,
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels

        # NOTE:removed noised first frame input
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents * 2,
            height,
            width,
            text_embeddings.dtype,
            device,
            generator,
            latents,
        )

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                # predict the noise residual
                noise_pred = self.unet(
                    latent_model_input,
                    t,
                    encoder_hidden_states=text_embeddings,
                    added_time_ids=added_time_ids,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                    noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents).prev_sample

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]
                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = tensor2vid(frames, self.image_processor, output_type=output_type)
        else:
            frames = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames

        return StableVideoDiffusionPipelineOutput(frames=frames)

    def forward(self, batch):
        videos, prompts = batch["videos"], batch["prompts"]
        batch_size = videos.shape[0]
        self.unet.set_num_videos(batch_size)
        first_frames = videos[:, 0]

        conditioning_dropout_prob = self.pipeline_config.proportion_empty_prompts
        random_p = torch.rand(batch_size)
        for i in range(batch_size):
            if random_p[i] < conditioning_dropout_prob:
                prompts[i] = ""

        encoder_hidden_states = self._encode_prompt(
            prompts, first_frames.device, num_videos_per_prompt=1, do_classifier_free_guidance=False, negative_prompt=None
        )

        added_time_ids = torch.tensor(
            [[batch["fps"][b] / batch["frame_strides"][b], batch["motion_bucket_ids"][b]] for b in range(batch_size)],
            device=encoder_hidden_states.device,
            dtype=encoder_hidden_states.dtype,
        )

        latents = self.video2latents(videos, out_pattern="b f c h w")

        scaled_input, scaled_sigmas, target, loss_weight = self.edm_step(batch_size, latents)
        model_output = self.unet(scaled_input, scaled_sigmas, encoder_hidden_states, added_time_ids=added_time_ids, return_dict=True).sample

        return self.compute_loss({"pred": model_output, "target": target, "loss_weight": loss_weight})

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        pipeline.vae.to(torch.float16)
        kwargs["prompt"] = batch["prompts"]  # DO NOT REPEAT
        call_kwargs = super().prepare_call_kwargs(pipeline, batch, **kwargs)
        return call_kwargs

    def val_t2v(self, batch):
        kwargs = self.prepare_call_kwargs(self, batch)
        res = self(**kwargs).frames
        res = torch.cat(res, dim=0).mul(2).sub(1)  # (b f) c h w  \in [-1, 1]
        return res
