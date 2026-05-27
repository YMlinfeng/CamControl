from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
try:
    from diffusers.models.unets.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, UNetSpatioTemporalConditionOutput
except ImportError:
    from diffusers.models.unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel, UNetSpatioTemporalConditionOutput
try:
    from diffusers.models.transformers.transformer_temporal import TransformerSpatioTemporalModel, TransformerTemporalModelOutput
except:
    from diffusers.models.transformer_temporal import TransformerSpatioTemporalModel, TransformerTemporalModelOutput
from einops import rearrange

from .base_unet import UNet2DConfig, UNet2DMixin


@dataclass
class SVDUNetConfig(UNet2DConfig):
    _target: Type = field(default_factory=lambda: SVDUNet)

    in_channels: int = 8
    """number of input channels."""

    projection_class_embeddings_input_dim: int = 768
    """dim of projection class embeddings."""

    cross_attention_dim: int = 1024
    """input dim of text/image embedding."""

    ignore_mismatched_sizes: bool = False
    """ignore mismatched sizes when loading ckpt."""

    use_scalelong: bool = False
    """ScaleLong: Towards More Stable Training of Diffusion Model via Scaling Network Long Skip Connection https://arxiv.org/pdf/2310.13545.pdf"""

    def from_pretrained(self, ckpt_path, **kwargs) -> Any:
        if hasattr(self, "in_channels"):
            kwargs["in_channels"] = self.in_channels
        if hasattr(self, "projection_class_embeddings_input_dim"):
            kwargs["projection_class_embeddings_input_dim"] = self.projection_class_embeddings_input_dim
        if hasattr(self, "ignore_mismatched_sizes"):
            kwargs["ignore_mismatched_sizes"] = self.ignore_mismatched_sizes
        if hasattr(self, "cross_attention_dim"):
            kwargs["cross_attention_dim"] = self.cross_attention_dim
        return super().from_pretrained(ckpt_path, **kwargs)


class SElayer(nn.Module):
    def __init__(self, dim, r=16):
        super(SElayer, self).__init__()
        self.layer1 = nn.Linear(dim, int(dim // r))
        self.layer2 = nn.Linear(int(dim // r), dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inp):
        return self.sigmoid(self.layer2(self.relu(self.layer1(torch.mean(inp, dim=-1))))).unsqueeze(-1)


class SVDUNet(UNetSpatioTemporalConditionModel, UNet2DMixin):
    def __init__(
        self,
        sample_size: Optional[int] = None,
        in_channels: int = 8,
        out_channels: int = 4,
        down_block_types: Tuple[str, ...] = (),
        up_block_types: Tuple[str, ...] = (),
        block_out_channels: Tuple[int, ...] = (),
        addition_time_embed_dim: int = 256,
        projection_class_embeddings_input_dim: int = 768,
        layers_per_block: Union[int, Tuple[int]] = 2,
        cross_attention_dim: Union[int, Tuple[int]] = 1024,
        transformer_layers_per_block: Union[int, Tuple[int], Tuple[Tuple]] = 1,
        num_attention_heads: Union[int, Tuple[int]] = (),
        num_frames: int = 25,
        unet_config: Optional[UNet2DConfig] = None,
    ):
        super().__init__(
            sample_size,
            in_channels,
            out_channels,
            down_block_types,
            up_block_types,
            block_out_channels,
            addition_time_embed_dim,
            projection_class_embeddings_input_dim,
            layers_per_block,
            cross_attention_dim,
            transformer_layers_per_block,
            num_attention_heads,
            num_frames,
        )
        self.unet_config = unet_config
        self.setup()
        if unet_config.use_scalelong:
            scale_model = []
            scale_model.append(SElayer(dim=block_out_channels[0]))
            for i, downsample_block in enumerate(self.down_blocks):
                if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                    num_res_sample = 3
                else:
                    num_res_sample = 2
                scale_model.extend([SElayer(dim=block_out_channels[i]) for _ in range(num_res_sample)])
            self.scale_model = nn.ModuleList(scale_model)

    # Modified from diffusers.models.transformer_temporal.TransformerSpatioTemporalModel
    def replace_transformer_spatio_temporal_forward(self):
        import types

        def transformer_spatio_temporal_forward(
            self,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            image_only_indicator: Optional[torch.Tensor] = None,
            return_dict: bool = True,
        ):
            """
            Args:
                hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                    Input hidden_states.
                num_frames (`int`):
                    The number of frames to be processed per batch. This is used to reshape the hidden states.
                encoder_hidden_states ( `torch.LongTensor` of shape `(batch size, encoder_hidden_states dim)`, *optional*):
                    Conditional embeddings for cross attention layer. If not given, cross-attention defaults to
                    self-attention.
                image_only_indicator (`torch.LongTensor` of shape `(batch size, num_frames)`, *optional*):
                    A tensor indicating whether the input contains only images. 1 indicates that the input contains only
                    images, 0 indicates that the input contains video frames.
                return_dict (`bool`, *optional*, defaults to `True`):
                    Whether or not to return a [`~models.transformer_temporal.TransformerTemporalModelOutput`] instead of a plain
                    tuple.

            Returns:
                [`~models.transformer_temporal.TransformerTemporalModelOutput`] or `tuple`:
                    If `return_dict` is True, an [`~models.transformer_temporal.TransformerTemporalModelOutput`] is
                    returned, otherwise a `tuple` where the first element is the sample tensor.
            """
            # 1. Input
            batch_frames, _, height, width = hidden_states.shape
            num_frames = image_only_indicator.shape[-1]
            batch_size = batch_frames // num_frames

            # NOTE: Fixed hard coded dim of encoder_hidden_states to support text embeddings
            time_context = encoder_hidden_states
            time_context_first_timestep = time_context[None, :].reshape(batch_size, num_frames, -1, time_context.shape[-1])[:, 0]
            time_context = time_context_first_timestep[None, :].broadcast_to(height * width, batch_size, encoder_hidden_states.shape[1], time_context.shape[-1])
            time_context = time_context.reshape(height * width * batch_size, encoder_hidden_states.shape[1], time_context.shape[-1])

            residual = hidden_states

            hidden_states = self.norm(hidden_states)
            inner_dim = hidden_states.shape[1]
            hidden_states = hidden_states.permute(0, 2, 3, 1).reshape(batch_frames, height * width, inner_dim)
            hidden_states = self.proj_in(hidden_states)

            num_frames_emb = torch.arange(num_frames, device=hidden_states.device)
            num_frames_emb = num_frames_emb.repeat(batch_size, 1)
            num_frames_emb = num_frames_emb.reshape(-1)
            t_emb = self.time_proj(num_frames_emb)

            # `Timesteps` does not contain any weights and will always return f32 tensors
            # but time_embedding might actually be running in fp16. so we need to cast here.
            # there might be better ways to encapsulate this.
            t_emb = t_emb.to(dtype=hidden_states.dtype)

            emb = self.time_pos_embed(t_emb)
            emb = emb[:, None, :]

            # 2. Blocks
            for block, temporal_block in zip(self.transformer_blocks, self.temporal_transformer_blocks):
                if self.training and self.gradient_checkpointing:
                    hidden_states = torch.utils.checkpoint.checkpoint(
                        block,
                        hidden_states,
                        None,
                        encoder_hidden_states,
                        None,
                        use_reentrant=False,
                    )
                else:
                    hidden_states = block(
                        hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                    )

                hidden_states_mix = hidden_states
                hidden_states_mix = hidden_states_mix + emb

                hidden_states_mix = temporal_block(
                    hidden_states_mix,
                    num_frames=num_frames,
                    encoder_hidden_states=time_context,
                )
                hidden_states = self.time_mixer(
                    x_spatial=hidden_states,
                    x_temporal=hidden_states_mix,
                    image_only_indicator=image_only_indicator,
                )

            # 3. Output
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.reshape(batch_frames, height, width, inner_dim).permute(0, 3, 1, 2).contiguous()

            output = hidden_states + residual

            if not return_dict:
                return (output,)

            return TransformerTemporalModelOutput(sample=output)

        for module in self.modules():
            if isinstance(module, TransformerSpatioTemporalModel):
                module.forward = types.MethodType(transformer_spatio_temporal_forward, module)

    def enable_gradient_checkpointing(self) -> None:
        super().enable_gradient_checkpointing()
        from functools import partial

        from diffusers.models.attention import TemporalBasicTransformerBlock
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, apply_activation_checkpointing, checkpoint_wrapper

        non_reentrant_wrapper = partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        check_fn = lambda submodule: isinstance(submodule, TemporalBasicTransformerBlock)
        apply_activation_checkpointing(self, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn)

    def save_ckpt(self, ds_state_dict, ckpt_path):
        new_ckpt = {}
        for key, _ in ds_state_dict.items():
            new_key = key
            if "_checkpoint_wrapped_module" in key:
                new_key = new_key.replace("_checkpoint_wrapped_module.", "")
            new_ckpt[new_key] = ds_state_dict[key]
        return super().save_ckpt(new_ckpt, ckpt_path)

    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        added_time_ids: torch.Tensor,
        return_dict: bool = True,
    ) -> Union[UNetSpatioTemporalConditionOutput, Tuple]:
        # 1. time
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            # TODO: this requires sync between CPU and GPU. So try to pass timesteps as tensors if you can
            # This would be a good case for the `match` statement (Python 3.10+)
            is_mps = sample.device.type == "mps"
            if isinstance(timestep, float):
                dtype = torch.float32 if is_mps else torch.float64
            else:
                dtype = torch.int32 if is_mps else torch.int64
            timesteps = torch.tensor([timesteps], dtype=dtype, device=sample.device)
        elif len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(sample.device)

        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        batch_size, num_frames = sample.shape[:2]
        timesteps = timesteps.expand(batch_size)

        t_emb = self.time_proj(timesteps)

        # `Timesteps` does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=sample.dtype)

        emb = self.time_embedding(t_emb)

        time_embeds = self.add_time_proj(added_time_ids.flatten())
        time_embeds = time_embeds.reshape((batch_size, -1))
        time_embeds = time_embeds.to(emb.dtype)
        aug_emb = self.add_embedding(time_embeds)
        emb = emb + aug_emb

        # Flatten the batch and frames dimensions
        # sample: [batch, frames, channels, height, width] -> [batch * frames, channels, height, width]
        sample = sample.flatten(0, 1)
        # Repeat the embeddings num_video_frames times
        # emb: [batch, channels] -> [batch * frames, channels]
        emb = emb.repeat_interleave(num_frames, dim=0)
        # encoder_hidden_states: [batch, 1, channels] -> [batch * frames, 1, channels]
        encoder_hidden_states = encoder_hidden_states.repeat_interleave(num_frames, dim=0)

        # 2. pre-process
        sample = self.conv_in(sample)

        image_only_indicator = torch.zeros(batch_size, num_frames, dtype=sample.dtype, device=sample.device)

        down_block_res_samples = (sample,)
        for downsample_block in self.down_blocks:
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    image_only_indicator=image_only_indicator,
                )

            down_block_res_samples += res_samples

        # 4. mid
        sample = self.mid_block(
            hidden_states=sample,
            temb=emb,
            encoder_hidden_states=encoder_hidden_states,
            image_only_indicator=image_only_indicator,
        )

        # NOTE: this is different from original SVD forward
        if self.unet_config.use_scalelong:
            scaled_down_block_res_samples = ()
            for i, sample in enumerate(down_block_res_samples):
                scaled_down_block_res_samples += (down_block_res_samples[i] * self.scale_model[i](rearrange(sample, "b c h w -> b c (h w)"))[..., None],)
            down_block_res_samples = scaled_down_block_res_samples

        # 5. up
        for i, upsample_block in enumerate(self.up_blocks):
            res_samples = down_block_res_samples[-len(upsample_block.resnets) :]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    image_only_indicator=image_only_indicator,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    image_only_indicator=image_only_indicator,
                )

        # 6. post-process
        sample = self.conv_norm_out(sample)
        sample = self.conv_act(sample)
        sample = self.conv_out(sample)

        # 7. Reshape back to original shape
        sample = sample.reshape(batch_size, num_frames, *sample.shape[1:])

        if not return_dict:
            return (sample,)

        return UNetSpatioTemporalConditionOutput(sample=sample)
