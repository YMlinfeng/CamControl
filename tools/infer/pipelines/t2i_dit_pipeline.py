import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union

import torch
import torch.nn.functional as F
from diffusers.image_processor import VaeImageProcessor
from diffusers.models import AutoencoderKL
from diffusers.pipelines.dit import DiTPipeline
from diffusers.pipelines.pipeline_utils import DiffusionPipeline, ImagePipelineOutput
from diffusers.schedulers import DPMSolverMultistepScheduler, KarrasDiffusionSchedulers
from diffusers.utils import deprecate
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange
from transformers import T5EncoderModel, T5Tokenizer

from ..configs.config_utils import to_immutable_dict
from ..models import VisualTokenizer
from ..models.transformers import Transformer2DModel, Transformer2DModelConfig
from ..utils import log_to_rank0, measure_time
from .base_pipeline import BasePipelineConfig, EDMTrainConfig, PipelineMixin


@dataclass
class T2IDitPipelineConfig(BasePipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: T2IDitPipeline)
    """target class to instantiate"""

    ckpt_path: str = "/group/ckpt/diffusers/PixArt-XL-2-1024-MS-T2V"

    edm_config: EDMTrainConfig = EDMTrainConfig(
        c_in_type="EDM",
        c_noise_type="EDM",
        c_skip_type="EDM",
        c_out_type="VPred",
        loss_weight_type="EDM",
        noise_dist_type="EDM",
    )

    transformer_config: Optional[Transformer2DModelConfig] = None

    vae_scale_factor: int = 1
    """VAE temporal scale factor"""

    proportion_empty_prompts: float = 0.0
    """Proportion of empty prompts to use."""

    call: Dict[str, Any] = to_immutable_dict(
        {
            "num_inference_steps": 30,
        }
    )
    measure_time: bool = False
    """The inference call arguments for the pipeline."""


class T2IDitPipeline(DiTPipeline, PipelineMixin):
    def __init__(
        self,
        vae: AutoencoderKL,
        transformer: Transformer2DModel,
        scheduler: KarrasDiffusionSchedulers,
        id2label: Optional[Dict[int, str]] = None,
    ):
        super().__init__(
            vae=vae,
            transformer=transformer,
            scheduler=scheduler,
            id2label=id2label,
        )

    def forward(self, batch):
        # videos, prompts = batch["videos"], batch["prompts"]
        samples = batch["data"]
        class_labels = batch["class_labels"]
        batch_size, height, width = samples.shape[0], samples.shape[-2], samples.shape[-1]

        samples = samples.to(self.transformer.pos_embed.proj.weight.dtype)
        if samples.ndim == 4:
            samples = rearrange(samples, "b c h w -> b 1 c h w")
        with measure_time("VAE", self.pipeline_config.measure_time):
            latents = self.video2latents(samples, out_pattern="b c f h w")

        added_cond_kwargs = {"resolution": None, "aspect_ratio": None}
        if self.transformer.config.sample_size == 128:
            resolution = torch.tensor([height, width]).repeat(batch_size, 1)
            aspect_ratio = torch.tensor([float(height / width)]).repeat(batch_size, 1)
            resolution = resolution.to(dtype=latents.dtype, device=latents.device)
            aspect_ratio = aspect_ratio.to(dtype=latents.dtype, device=latents.device)

            added_cond_kwargs = {"resolution": resolution, "aspect_ratio": aspect_ratio}

        """prompt_masks = [random.random() > proportion_empty_prompts for i in range(batch_size)]
        prompts = [prompt if mask else "" for mask, prompt in zip(prompt_masks, [" "]*batch_size)]
        if prompt_embeds is None:
            self.text_encoder.to(torch.float32) # NOTE: nan when float16
            with torch.no_grad():
                prompt_embeds, prompt_attention_mask, _, _ = self.encode_prompt(
                    prompts,
                    do_classifier_free_guidance=False,
                    device=latents.device,
                )
            prompt_embeds = prompt_embeds.to(torch.float16)"""
        # class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        # class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels

        condition_masks = [random.random() > self.pipeline_config.proportion_empty_prompts for i in range(batch_size)]
        class_labels = [each_class if mask else 1000 for mask, each_class in zip(condition_masks, class_labels)]

        scaled_input, scaled_sigmas, target, loss_weight = self.edm_step(batch_size, latents)
        class_labels = torch.tensor(class_labels).cuda()

        model_output = self.transformer(
            scaled_input, timestep=scaled_sigmas, added_cond_kwargs=added_cond_kwargs, return_dict=False, class_labels=class_labels
        )[0]
        if self.transformer.config.out_channels // 2 == target.shape[1]:
            model_output = model_output.chunk(2, dim=1)[0]
        return self.compute_loss({"pred": model_output, "target": target, "loss_weight": loss_weight})

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        call_kwargs = {}
        if "class_labels" not in kwargs:
            call_kwargs["class_labels"] = batch["class_labels"]
        if "generator" not in kwargs:
            call_kwargs["generator"] = torch.Generator(device=pipeline.device).manual_seed(pipeline.pipeline_config.seed)
        call_kwargs.update(pipeline.pipeline_config.call)
        call_kwargs.update(kwargs)
        call_kwargs["output_type"] = "pt"  # NOTE: return pytorch tensor

        return call_kwargs

    def val_t2v(self, batch):
        call_kwargs = self.prepare_call_kwargs(self, batch)
        return self(**call_kwargs).images.mul(2).sub(1)

    @torch.no_grad()
    def __call__(
        self,
        class_labels: List[int],
        guidance_scale: float = 4.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 50,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
    ) -> Union[ImagePipelineOutput, Tuple]:
        r"""
        The call function to the pipeline for generation.

        Args:
            class_labels (List[int]):
                List of ImageNet class labels for the images to be generated.
            guidance_scale (`float`, *optional*, defaults to 4.0):
                A higher guidance scale value encourages the model to generate images closely linked to the text
                `prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 250):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`ImagePipelineOutput`] instead of a plain tuple.

        Examples:

        ```py
        >>> from diffusers import DiTPipeline, DPMSolverMultistepScheduler
        >>> import torch

        >>> pipe = DiTPipeline.from_pretrained("facebook/DiT-XL-2-256", torch_dtype=torch.float16)
        >>> pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        >>> pipe = pipe.to("cuda")

        >>> # pick words from Imagenet class labels
        >>> pipe.labels  # to print all available words

        >>> # pick words that exist in ImageNet
        >>> words = ["white shark", "umbrella"]

        >>> class_ids = pipe.get_label_ids(words)

        >>> generator = torch.manual_seed(33)
        >>> output = pipe(class_labels=class_ids, num_inference_steps=25, generator=generator)

        >>> image = output.images[0]  # label 'white shark'
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """

        batch_size = len(class_labels)
        latent_size = self.transformer.config.sample_size
        latent_channels = self.transformer.config.in_channels

        latents = randn_tensor(
            shape=(batch_size, latent_channels, latent_size, latent_size),
            generator=generator,
            device=self._execution_device,
            dtype=self.transformer.dtype,
        )
        latent_model_input = torch.cat([latents] * 2) if guidance_scale > 1 else latents

        class_labels = torch.tensor(class_labels, device=self._execution_device).reshape(-1)
        class_null = torch.tensor([1000] * batch_size, device=self._execution_device)
        class_labels_input = torch.cat([class_labels, class_null], 0) if guidance_scale > 1 else class_labels
        latent_model_input = latent_model_input.unsqueeze(2)

        # set step values
        self.scheduler.set_timesteps(num_inference_steps)
        for t in self.progress_bar(self.scheduler.timesteps):
            if guidance_scale > 1:
                half = latent_model_input[: len(latent_model_input) // 2]
                latent_model_input = torch.cat([half, half], dim=0)
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            timesteps = t
            if not torch.is_tensor(timesteps):
                # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
                # This would be a good case for the `match` statement (Python 3.10+)
                is_mps = latent_model_input.device.type == "mps"
                if isinstance(timesteps, float):
                    dtype = torch.float32 if is_mps else torch.float64
                else:
                    dtype = torch.int32 if is_mps else torch.int64
                timesteps = torch.tensor([timesteps], dtype=dtype, device=latent_model_input.device)
            elif len(timesteps.shape) == 0:
                timesteps = timesteps[None].to(latent_model_input.device)
            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            timesteps = timesteps.expand(latent_model_input.shape[0])
            # predict noise model_output
            noise_pred = self.transformer(latent_model_input, timestep=timesteps, class_labels=class_labels_input).sample

            # perform guidance
            if guidance_scale > 1:
                eps, rest = noise_pred[:, :latent_channels], noise_pred[:, latent_channels:]
                cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)

                half_eps = uncond_eps + guidance_scale * (cond_eps - uncond_eps)
                eps = torch.cat([half_eps, half_eps], dim=0)

                noise_pred = torch.cat([eps, rest], dim=1)

            # learned sigma
            if self.transformer.config.out_channels // 2 == latent_channels:
                model_output, _ = torch.split(noise_pred, latent_channels, dim=1)
            else:
                model_output = noise_pred

            # compute previous image: x_t -> x_t-1
            latent_model_input = self.scheduler.step(model_output, t, latent_model_input).prev_sample

        if guidance_scale > 1:
            latents, _ = latent_model_input.chunk(2, dim=0)
        else:
            latents = latent_model_input

        latents = 1 / self.vae.config.scaling_factor * latents
        latents = latents.squeeze(2)

        samples = self.vae.decode(latents).sample

        samples = (samples / 2 + 0.5).clamp(0, 1)

        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16

        if output_type == "pil":
            samples = samples.cpu().permute(0, 2, 3, 1).float().numpy()
            samples = self.numpy_to_pil(samples)
            samples[0].save("./fuck.png")
        elif output_type == "np":
            samples = samples.cpu().float().numpy()

        if not return_dict:
            return (samples,)

        return ImagePipelineOutput(images=samples)
