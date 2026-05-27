from dataclasses import dataclass, field
from typing import Any, Optional, Type

import lpips
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..models.autoencoders.components import GANLoss, cal_gradient_penalty
from ..models.autoencoders.discriminator import DiscriminatorConfig
from ..models.autoencoders.visual_tokenizer import VisualTokenizerConfig
from .base_pipeline import BasePipelineConfig, PipelineMixin


@dataclass
class VisualTokenizerPipelineConfig(BasePipelineConfig):
    """Configuration for Pipeline instantiation"""

    _target: Type = field(default_factory=lambda: VisualTokenizerPipeline)
    """target class to instantiate"""

    visual_tokenizer: VisualTokenizerConfig = VisualTokenizerConfig()
    """visual tokenizer config"""

    discriminator: Optional[DiscriminatorConfig] = None
    """discriminator config"""

    loss_weights: dict = field(default_factory=lambda: {"l1_loss": 5.0, "perceptual_loss": 0.1, "kl_loss": 0.1})
    """The loss weights for the pipeline."""

    visual_tokenizer_ckpt_path: Optional[str] = None
    """The path to the visual tokenizer checkpoint."""

    discriminator_ckpt_path: Optional[str] = None
    """The path to the discriminator checkpoint."""

    gan_mode: str = "lsgan"

    def from_pretrained(self, **kwargs) -> Any:
        if self.visual_tokenizer_ckpt_path is not None:
            self.visual_tokenizer.vae_ckpt_path = self.visual_tokenizer_ckpt_path
        visual_tokenizer = self.visual_tokenizer.from_pretrained()
        discriminator = self.discriminator.from_pretrained(self.discriminator_ckpt_path) if self.discriminator is not None else None

        return self._target(visual_tokenizer, discriminator, self)


class VisualTokenizerPipeline(PipelineMixin):
    def __init__(
        self,
        visual_tokenizer,
        discriminator,
        config: VisualTokenizerPipelineConfig,
    ) -> None:
        self.pipeline_config = config
        self.visual_tokenizer = visual_tokenizer
        self.discriminator = discriminator
        self.perceptual_loss = lpips.LPIPS(net="vgg").eval().to("cuda")
        self.gan_loss = GANLoss(gan_mode=self.pipeline_config.gan_mode)
        visual_tokenizer.enable_gradient_checkpointing()

    def loss_existed(self, loss_name):
        return loss_name in self.pipeline_config.loss_weights and self.pipeline_config.loss_weights[loss_name] > 0.0

    def forward(self, batch):
        video = rearrange(batch["data"], "b f c h w -> b c f h w")
        video = video.to(dtype=self.visual_tokenizer.logvar.dtype)

        posterior = self.visual_tokenizer.encode(video).latent_dist
        dec = self.visual_tokenizer.decode(posterior.sample()).sample

        return self.compute_loss(dec, video, posterior, self.visual_tokenizer.logvar)

    def compute_loss(self, pred, target, posteriors, logvar=0.0):
        # Without GAN

        loss_dict = {}

        # Reconstruction loss
        l1_loss = torch.abs(pred.float().contiguous() - target.float().contiguous())
        loss_dict["l1_loss"] = torch.mean(l1_loss)
        rec_loss = l1_loss * self.pipeline_config.loss_weights["l1_loss"]
        # perceptual_loss
        if self.loss_existed("perceptual_loss"):
            p_pred = rearrange(pred, "b c f h w -> (b f) c h w").float().contiguous()
            p_target = rearrange(target, "b c f h w -> (b f) c h w").float().contiguous()
            p_loss = torch.utils.checkpoint.checkpoint(self.perceptual_loss, p_pred, p_target, use_reentrant=False)
            p_loss = rearrange(p_loss, "(b f) 1 1 1 -> b 1 f 1 1", f=pred.shape[2])
            loss_dict["perceptual_loss"] = torch.mean(p_loss)
            rec_loss += p_loss * self.pipeline_config.loss_weights["perceptual_loss"]

        nll_loss = rec_loss / torch.exp(logvar) + logvar
        nll_loss = torch.mean(nll_loss)
        loss_dict["var"] = torch.exp(logvar)

        # KL loss
        kl_loss = posteriors.kl()
        kl_loss = torch.mean(kl_loss)
        loss_dict["kl_loss"] = kl_loss
        loss = nll_loss + kl_loss * self.pipeline_config.loss_weights["kl_loss"]

        return {"total_loss": loss, **loss_dict}

    def forward_gan(self, batch):
        video = rearrange(batch["data"], "b f c h w -> b c f h w")
        video = video.to(dtype=self.visual_tokenizer.logvar.dtype)
        posterior = self.visual_tokenizer.encode(video).latent_dist
        dec = self.visual_tokenizer.decode(posterior.sample()).sample
        return dec, video, posterior, self.visual_tokenizer.logvar

    def compute_loss_g(self, pred, target, posteriors, logvar=0.0):
        loss_dict_G = {}

        # Reconstruction loss
        l1_loss = torch.abs(pred.float().contiguous() - target.float().contiguous())
        loss_dict_G["l1_loss"] = torch.mean(l1_loss)
        rec_loss = l1_loss * self.pipeline_config.loss_weights["l1_loss"]
        # perceptual_loss
        if self.loss_existed("perceptual_loss"):
            p_pred = rearrange(pred, "b c f h w -> (b f) c h w").float().contiguous()
            p_target = rearrange(target, "b c f h w -> (b f) c h w").float().contiguous()
            p_loss = self.perceptual_loss(p_pred, p_target)
            p_loss = rearrange(p_loss, "(b f) 1 1 1 -> b 1 f 1 1", f=pred.shape[2])
            loss_dict_G["perceptual_loss"] = torch.mean(p_loss)
            rec_loss += p_loss * self.pipeline_config.loss_weights["perceptual_loss"]

        nll_loss = rec_loss / torch.exp(logvar) + logvar
        nll_loss = torch.mean(nll_loss)
        loss_dict_G["var"] = torch.exp(logvar)

        # KL loss
        kl_loss = posteriors.kl()
        kl_loss = torch.mean(kl_loss)
        loss_dict_G["kl_loss"] = kl_loss
        loss_g = nll_loss + kl_loss * self.pipeline_config.loss_weights["kl_loss"]

        # G loss
        disc_out_fake, disc_out_fake_features = self.discriminator(pred, True)
        loss_gan_g = self.gan_loss(disc_out_fake, True)
        loss_dict_G["gan_loss_g"] = loss_gan_g.mean()
        loss_g += loss_gan_g * self.pipeline_config.loss_weights["gan_loss"]

        if self.loss_existed("feature_matching_loss"):
            _, disc_out_real_features = self.discriminator(target, True)
            feature_loss = 0.0
            for fake_feature, real_feature in zip(disc_out_fake_features, disc_out_real_features):
                feature_loss += torch.mean(torch.abs(fake_feature - real_feature.detach()))
            feature_loss /= len(disc_out_fake_features)
            loss_dict_G["feature_matching_loss"] = feature_loss
            loss_g += feature_loss * self.pipeline_config.loss_weights["feature_matching_loss"]

        return {"total_loss": loss_g, **loss_dict_G}

    def get_discriminator_loss(self, pred, target, discriminator, loss_dict_D, name=""):
        disc_input = torch.cat([target, pred.detach()], dim=0)
        disc_out_real, disc_out_fake = discriminator(disc_input).chunk(2)

        loss_dict_D[f"disc_out_real{name}"] = disc_out_real.mean()
        loss_dict_D[f"disc_out_fake{name}"] = disc_out_fake.mean()

        loss_d_real = self.gan_loss(disc_out_real, True)
        loss_d_fake = self.gan_loss(disc_out_fake, False)
        loss_d = 0.5 * (loss_d_real + loss_d_fake)

        return loss_d

    def compute_loss_d(self, pred, target, posteriors, logvar=0.0):
        loss_dict_D = {}

        # GAN loss D
        loss_d_ori = self.get_discriminator_loss(pred, target, self.discriminator, loss_dict_D)
        if self.pipeline_config.discriminator.use_pixel_discriminator:
            loss_d_pixel = self.get_discriminator_loss(pred, target, self.discriminator.pixel_discriminator, loss_dict_D, "_pixel")
            loss_d_ori += loss_d_pixel
        loss_dict_D["gan_loss_d"] = loss_d_ori
        loss_d = loss_d_ori * self.pipeline_config.loss_weights["gan_loss"]

        # Discriminator gradient penalty
        if self.loss_existed("r1_gradient_penalty_cost"):
            gradients_penalty, _ = cal_gradient_penalty(
                self.discriminator,
                target,
                pred.detach(),
                device=pred.device,
                lambda_gp=self.pipeline_config.loss_weights["r1_gradient_penalty_cost"],
            )
            loss_dict_D["r1_gradient_penalty"] = gradients_penalty
            loss_d += gradients_penalty

        return {"total_loss": loss_d, **loss_dict_D}

    def _get_signature_keys(self, cls):
        # get all trainable modules
        module_list = ["visual_tokenizer"]
        if self.discriminator is not None:
            module_list.append("discriminator")
        return (module_list, None)

    def to(self, device):
        self.visual_tokenizer.to(device)
        if self.discriminator is not None:
            self.discriminator.to(device)
        self.perceptual_loss.to(device)

    @torch.no_grad()
    def __call__(self, video):
        is_image = video.ndim == 4
        if is_image:
            video = rearrange(video, "b c ... -> b c 1 ...")
        else:
            video = rearrange(video, "b f c h w -> b c f h w")
        video = video.to(dtype=self.visual_tokenizer.logvar.dtype)

        return self.visual_tokenizer(video, sample_posterior=False).sample

    @staticmethod
    def prepare_call_kwargs(pipeline, batch, **kwargs):
        call_kwargs = {"video": batch["data"]}
        return call_kwargs

    @torch.no_grad()
    def val_gt(self, batch):
        frames = rearrange(batch["data"], "b f c h w -> (b f) c h w")
        return frames

    @torch.no_grad()
    def val_vtoken(self, batch):
        kwargs = self.prepare_call_kwargs(self, batch)
        recon_video = self(**kwargs)
        recon_video = rearrange(recon_video, "b c f h w -> (b f) c h w")
        return recon_video  # (b f) c h w  \in [-1, 1]
