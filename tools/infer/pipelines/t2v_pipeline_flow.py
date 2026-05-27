import inspect
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import deprecate
from einops import rearrange

from ..configs.config_utils import to_immutable_dict
from ..models import VisualTokenizer, VisualTokenizerConfig
from ..models.transformers import Transformer2DModelConfig
from ..utils import measure_time
from .base_pipeline import BasePipelineConfig, EDMTrainConfig, PipelineMixin
from .t2v_pipeline_pixart_alpha import T2VPixArtAlphaPipeline, T2VPixArtAlphaPipelineConfig


@dataclass
class T2VFlowPipelineConfig(T2VPixArtAlphaPipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: T2VFlowPipeline)
    """target class to instantiate"""

    ckpt_path: str = "/group/ckpt/diffusers/PixArt-XL-2-512x512"

    edm_config: Optional[EDMTrainConfig] = None
    """EDM config, default to EDM"""

    epsilon: float = 1e-3

    transformer_config: Transformer2DModelConfig = Transformer2DModelConfig()
    """The transformer config for the pipeline."""
    vae_config: VisualTokenizerConfig = VisualTokenizerConfig()
    """The VAE config for the pipeline."""

    proportion_empty_prompts: float = 0.1
    """Proportion of empty prompts to use."""

    max_sequence_length: int = 256
    """The maximum number of tokens for text encoding"""
    logit_normal: bool = True
    timestep_shift: float = 1.0
    match_snr: bool = False

    call: Dict[str, Any] = to_immutable_dict(
        {
            "num_frames": 16,
            "height": 256,
            "width": 384,
            "num_inference_steps": 50,
        }
    )
    """The inference call arguments for the pipeline."""
    measure_time: bool = False

    t2v_ratio: float = 1.0


class T2VFlowPipeline(T2VPixArtAlphaPipeline, PipelineMixin):

    def forward(self, batch):

        with measure_time("VAE", self.pipeline_config.measure_time):
            latents, attention_mask = self.get_latents(batch)
        batch_size, num_frames = latents.shape[0], latents.shape[2]

        added_cond_kwargs = self.prepare_added_cond_kwargs(
            batch_size,
            fps=batch["sample_fps"],
            num_frames=batch["num_frames"],
            height=[height for height, _ in batch["target_sizes"]],
            width=[width for _, width in batch["target_sizes"]],
            dtype=latents.dtype,
            device=latents.device,
        )
        with measure_time("Text", self.pipeline_config.measure_time):
            dtype = self.transformer.pos_embed.proj.weight.dtype
            prompt_embeds, prompt_attention_mask, prompt_masks = self.get_t5_prompt_embeddings(
                prompts=batch.get("prompts", None),
                t5_prompt_embeds_list=batch.get("t5_prompt_embeds", None),
                device=latents.device,
                dtype=dtype,
            )

            prompt_embeds_pooled = None
            if self.pipeline_config.clip_ckpt_path is not None:
                prompt_embeds_clip, prompt_attention_mask_clip, prompt_embeds_pooled = self.get_clip_prompt_embeddings(
                    prompts=batch.get("prompts", None),
                    clip_prompt_embeds_list=batch.get("clip_prompt_embeds", None),
                    device=latents.device,
                    dtype=dtype,
                    prompt_masks=prompt_masks,
                    target_size=prompt_embeds.shape[2],
                )

                prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_clip.to(prompt_embeds.dtype)), dim=1)  # bs * 120+77 * 4096
                prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask_clip.to(prompt_attention_mask.dtype)], dim=1)

            prompt_embeds = prompt_embeds.to(dtype)
            prompt_attention_mask = prompt_attention_mask.to(dtype)
            if prompt_embeds_pooled is not None:
                prompt_embeds_pooled = prompt_embeds_pooled.to(dtype)

        added_cond_kwargs["prompt_embeds_pooled"] = prompt_embeds_pooled

        z_1 = self.generate_noise(latents)
        eps = self.pipeline_config.epsilon

        if "t_step" in batch:
            t = batch["t_step"].repeat_interleave(batch_size).to(device=latents.device, dtype=latents.dtype)
        else:
            if self.pipeline_config.logit_normal:
                t_logit = torch.exp(torch.randn(batch_size, device=latents.device))
                t = t_logit / (t_logit + 1)
            else:
                t = torch.rand(batch_size, device=latents.device)

            t = self.pipeline_config.timestep_shift * t / (1 - t + self.pipeline_config.timestep_shift * t)

            # timestep = t
            if self.pipeline_config.match_snr:
                scale_factor = latents.shape[2] ** 0.5
                t = scale_factor * t / (1 - t + scale_factor * t)

        t_expand = t[:, None, None, None, None]

        # for mix training
        prob = torch.rand(1).to(t_expand.device)
        torch.distributed.broadcast(prob, src=0)
        if prob <= self.pipeline_config.t2v_ratio:
            # t2v
            condition_index = []
        else:
            # i2v
            condition_index = [0]
        t_expand = self.zero_condition_frame_timestep(t_expand, num_frames, condition_index)

        # 1 is noise, 0 is real data
        z_t = (1 - t_expand) * latents + (eps + (1 - eps) * t_expand) * z_1
        u = (1 - eps) * z_1 - latents  # TODO: need check with xiaoyu 

        z_t = z_t.to(latents.dtype)
        u = u.to(latents.dtype)

        with measure_time("Transformer", self.pipeline_config.measure_time):
            v = self.transformer(
                z_t,
                timestep=t_expand * 999,
                # timestep=timestep * 999,
                encoder_hidden_states=prompt_embeds,
                attention_mask=attention_mask,
                encoder_attention_mask=prompt_attention_mask,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
        return self.compute_loss({"pred": v, "target": u}, attention_mask)

    def zero_condition_frame_timestep(self, timesteps, num_frames, condition_index):
        timesteps = timesteps.repeat_interleave(num_frames, dim=2)  # b 1 1 1 1 -> b 1 f 1 1
        timesteps[:, :, condition_index] = 0
        return timesteps


    def get_val_loss(self, batch, num_valloss_timesteps: int = 20):

        def forward(batch):

            with measure_time("VAE", self.pipeline_config.measure_time):
                latents, attention_mask = self.get_latents(batch)
            batch_size = latents.shape[0]

            added_cond_kwargs = self.prepare_added_cond_kwargs(
                batch_size,
                fps=batch["sample_fps"],
                num_frames=batch["num_frames"],
                height=[height for height, _ in batch["target_sizes"]],
                width=[width for _, width in batch["target_sizes"]],
                dtype=latents.dtype,
                device=latents.device,
            )
            with measure_time("Text", self.pipeline_config.measure_time):
                dtype = self.transformer.pos_embed.proj.weight.dtype
                prompt_embeds, prompt_attention_mask, prompt_masks = self.get_t5_prompt_embeddings(
                    prompts=batch.get("prompts", None),
                    t5_prompt_embeds_list=batch.get("t5_prompt_embeds", None),
                    device=latents.device,
                    dtype=dtype,
                )

                prompt_embeds_pooled = None
                if self.pipeline_config.clip_ckpt_path is not None:
                    prompt_embeds_clip, prompt_attention_mask_clip, prompt_embeds_pooled = self.get_clip_prompt_embeddings(
                        prompts=batch.get("prompts", None),
                        clip_prompt_embeds_list=batch.get("clip_prompt_embeds", None),
                        device=latents.device,
                        dtype=dtype,
                        prompt_masks=prompt_masks,
                        target_size=prompt_embeds.shape[2],
                    )

                    prompt_embeds = torch.cat((prompt_embeds, prompt_embeds_clip.to(prompt_embeds.dtype)), dim=1)  # bs * 120+77 * 4096
                    prompt_attention_mask = torch.cat([prompt_attention_mask, prompt_attention_mask_clip.to(prompt_attention_mask.dtype)], dim=1)

                prompt_embeds = prompt_embeds.to(dtype)
                prompt_attention_mask = prompt_attention_mask.to(dtype)
                if prompt_embeds_pooled is not None:
                    prompt_embeds_pooled = prompt_embeds_pooled.to(dtype)

            added_cond_kwargs["prompt_embeds_pooled"] = prompt_embeds_pooled

            with measure_time("Transformer", self.pipeline_config.measure_time):
                z_1 = self.generate_noise(latents)
                eps = self.pipeline_config.epsilon

                t = batch["t_step"].repeat_interleave(batch_size).to(device=latents.device)

                t_expand = t[:, None, None, None, None]
                # 1 is noise, 0 is real data
                z_t = (1 - t_expand) * latents + (eps + (1 - eps) * t_expand) * z_1
                u = (1 - eps) * z_1 - latents

                v = self.transformer(
                    z_t,
                    timestep=t * 999,
                    encoder_hidden_states=prompt_embeds,
                    attention_mask=attention_mask,
                    encoder_attention_mask=prompt_attention_mask,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]
            return self.compute_loss({"pred": v, "target": u}, attention_mask)

        val_loss = {}
        for t_step in torch.linspace(1 / num_valloss_timesteps, 1 - 1 / num_valloss_timesteps, num_valloss_timesteps - 1):
            batch["t_step"] = torch.Tensor([t_step])
            for k, v in self.forward(batch).items():
                if k in val_loss:
                    val_loss[k].append(v.detach()[None])
                else:
                    val_loss[k] = [v.detach()[None]]
        return val_loss

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        call_kwargs = {}
        if "prompt" not in kwargs:
            call_kwargs["prompt"] = batch["prompts"]
        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)
        return call_kwargs

    def val_t2v(self, batch):
        call_kwargs = self.prepare_call_kwargs(self, batch)
        return self(**call_kwargs).images.mul(2).sub(1)

    def val_t2v_16_9(self, batch):
        return self.t2v_by_ratio(batch, 16 / 9)

    def val_t2v_9_16(self, batch):
        return self.t2v_by_ratio(batch, 9 / 16)

    def t2v_by_ratio(self, batch, ratio):
        call_kwargs = self.prepare_call_kwargs(self, batch)
        spatial_unit_size = (
            self.pipeline_config.transformer_config.vae_scale_factor
            * self.pipeline_config.transformer_config.patch_size
            * (
                self.pipeline_config.transformer_config.temporal_attention_config.stfit_patch_size[-1]
                if hasattr(self.pipeline_config.transformer_config, "temporal_attention_config")
                and self.pipeline_config.transformer_config.temporal_attention_config is not None
                else 1
            )
        )

        area = call_kwargs["height"] * call_kwargs["width"]
        width = math.sqrt(area / ratio) * ratio
        height = round(width / ratio) // spatial_unit_size * spatial_unit_size
        width = round(width) // spatial_unit_size * spatial_unit_size
        call_kwargs.update(height=height, width=width)
        return self(**call_kwargs).images.mul(2).sub(1)

    def add_input_condition(self, latents, input_conditions, input_latent_index, input_condition_index, timestep=None,
                            ref_video_latent=None):
        batch_size, latent_num_frames = latents.shape[0], latents.shape[2]
        if input_condition_index is not None:
            latents[:, :, input_latent_index] = input_conditions[:, :, input_condition_index]
        if ref_video_latent is not None:
            ref_video_index = latent_num_frames - ref_video_latent.shape[2]
            latents[:, :, ref_video_index:] = ref_video_latent

        if timestep is not None:
            timestep = timestep.view(-1, 1, 1, 1, 1).repeat_interleave(latent_num_frames, dim=2)  # b -> b 1 f 1 1
            if input_latent_index is not None:
                timestep[:, :, input_latent_index] = 0
            if ref_video_latent is not None:
                timestep[:, :, ref_video_index:] = 0
        return latents, timestep

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]] = None,
        negative_prompt: str = "",
        num_inference_steps: int = 20,
        timesteps: List[int] = None,
        guidance_scale: float = 4.5,
        num_images_per_prompt: Optional[int] = 1,
        num_frames: int = 1,
        cond_num_frames: Optional[int] = None,
        fps: float = 15.0,
        cond_fps: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        cond_height: Optional[int] = None,
        cond_width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        timestep_shift: Optional[float] = None,
        batch: Optional[Dict] = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        if "process_call_back" in kwargs:
            process_call_back = kwargs["process_call_back"]
        else:
            process_call_back = None
        if "mask_feature" in kwargs:
            deprecation_message = "The use of `mask_feature` is deprecated. It is no longer used in any computation and that doesn't affect the end results. It will be removed in a future version."
            deprecate("mask_feature", "1.0.0", deprecation_message, standard_warn=False)

        # i2v
        if "condition_image" in batch:
            task = "i2v"
            input_condition_latent, _ = self.get_first_frame_latent(batch["condition_image"])
            input_latent_index = [0]
            input_condition_index = [0]
            height, width = batch["condition_image"][0].shape[-2:] # assume condition image is resized in base_data
        else:
            task = "t2v"
            input_latent_index, input_conditon = None, None

        # 1. Check inputs. Raise error if not correct
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        cond_height = cond_height or height
        cond_width = cond_width or width
        cond_num_frames = cond_num_frames or num_frames
        cond_fps = cond_fps or fps
        timestep_shift = timestep_shift or self.pipeline_config.timestep_shift

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_steps,
            prompt_embeds,
            negative_prompt_embeds,
            prompt_attention_mask,
            negative_prompt_attention_mask,
        )

        # 2. Default height and width to transformer
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        self.text_encoder.to(torch.bfloat16)  # NOTE(m2v): nan when float16
        (prompt_embeds, prompt_attention_mask, negative_prompt_embeds, negative_prompt_attention_mask, prompt_embeds_pooled, negative_prompt_embeds_pooled) = (
            self.encode_prompt(
                prompt,
                do_classifier_free_guidance,
                negative_prompt=negative_prompt,
                num_images_per_prompt=num_images_per_prompt,
                device=device,
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=negative_prompt_embeds,
                prompt_attention_mask=prompt_attention_mask,
                negative_prompt_attention_mask=negative_prompt_attention_mask,
                clean_caption=clean_caption,
                max_sequence_length=self.pipeline_config.max_sequence_length,
            )
        )

        if do_classifier_free_guidance:
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            prompt_attention_mask = torch.cat([negative_prompt_attention_mask, prompt_attention_mask], dim=0)
            if self.pipeline_config.clip_ckpt_path is not None:
                prompt_embeds_pooled = torch.cat([negative_prompt_embeds_pooled, prompt_embeds_pooled], dim=0)

        prompt_embeds = prompt_embeds.to(self.transformer.dtype)
        prompt_attention_mask = prompt_attention_mask.to(self.transformer.dtype)
        if self.pipeline_config.clip_ckpt_path is not None:
            prompt_embeds_pooled = prompt_embeds_pooled.to(self.transformer.dtype)

        # 5. Prepare latents.
        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            num_frames + (num_frames - 1) // self.vae.config.segment_size * (self.pipeline_config.transformer_config.vae_temporal_scale_factor - 1),
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )


        # 6.1 Prepare micro-conditions.
        added_cond_kwargs = self.prepare_added_cond_kwargs(
            batch_size,
            cond_fps,
            cond_num_frames,
            cond_height,
            cond_width,
            prompt_embeds.dtype,
            device,
            num_images_per_prompt,
            do_classifier_free_guidance,
        )
        added_cond_kwargs["prompt_embeds_pooled"] = prompt_embeds_pooled

        if process_call_back:
            process_call_back(0.001, 300.0)
        # 7. Denoising loop
        timesteps_all = torch.linspace(1.0, 0, num_inference_steps + 1, device=latents.device)
        timesteps_all = timestep_shift * timesteps_all / (1 - timesteps_all + timestep_shift * timesteps_all)
        dts = timesteps_all[:-1] - timesteps_all[1:]
        timesteps = timesteps_all[:-1]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            t0 = time.time()
            for i, t in enumerate(timesteps):
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents

                if self.pipeline_config.match_snr:
                    scale_factor = latents.shape[2] ** 0.5
                    current_timestep = t / (scale_factor - scale_factor * t + t)
                else:
                    current_timestep = t

                if not torch.is_tensor(current_timestep):
                    # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                    # This would be a good case for the `match` statement (Python 3.10+)
                    is_mps = latent_model_input.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latent_model_input.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latent_model_input.device)
                # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
                current_timestep = current_timestep.expand(latent_model_input.shape[0])
                # predict noise model_output

                if task in ["i2v"]:
                    latent_model_input, current_timestep = self.add_input_condition(
                            latent_model_input, input_condition_latent, input_latent_index, input_condition_index, current_timestep
                    )

                v_pred = self.transformer(
                    latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    encoder_attention_mask=prompt_attention_mask,
                    timestep=current_timestep * 999,
                    added_cond_kwargs=added_cond_kwargs,
                    return_dict=False,
                )[0]

                # perform guidance
                if do_classifier_free_guidance:
                    v_pred_uncond, v_pred_text = v_pred.chunk(2)
                    v_pred = v_pred_uncond + guidance_scale * (v_pred_text - v_pred_uncond)

                # compute previous image: x_t -> x_t-1
                latents = latents - dts[i] * v_pred

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents)

                if process_call_back:
                    process_call_back((i + 1) / len(timesteps), (time.time() - t0) / (i + 1) * (len(timesteps) - i - 1))

        if task in ["i2v"]:
            latents, _ = self.add_input_condition(latents, input_condition_latent, input_latent_index, input_condition_index)

        if not output_type == "latent":
            if isinstance(self.vae, VisualTokenizer):
                if self.pipeline_config.offload:
                    self.to("cpu")
                    self.vae.to("cuda")
                    torch.cuda.empty_cache()
                # NOTE: our 3d vae always expects latents in the format b c f h w
                image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
                image = rearrange(image, "b c f h w -> (b f) c h w")
                if self.pipeline_config.offload:
                    self.to("cuda")
            else:
                latents = rearrange(latents, "b c f h w -> (b f) c h w")
                enable_vae_temporal_decoder = "num_frames" in set(inspect.signature(self.vae.forward).parameters.keys())
                if enable_vae_temporal_decoder:
                    image = self.decode_latents_with_temporal_decoder(latents / self.vae.config.scaling_factor)
                else:
                    image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
        else:
            image = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (image,)

        return ImagePipelineOutput(images=image)
