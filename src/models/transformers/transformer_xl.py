from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple, Type, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import register_to_config
from diffusers.models.attention_processor import Attention
from diffusers.utils import is_torch_version
from einops import rearrange, repeat

from ...configs.base_config import PrintableConfig
from ...utils import load_model, log_to_rank0
from ..attention_processor import MaskedAttnProcessor2_0, Neighbour3dAttnProcessor
from ..embeddings import NoPEPatchEmbed, prepare_mask, RotaryEmbeddingFast
from ..normalization import AdaRMSNormSingle, RMSNorm
from .transformer_2d import Transformer2DModel, Transformer2DModelConfig, Transformer2DModelOutput


@torch.compile
def multiply_addition(a, b):
    return a * (b + 1)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, inner_dim, mult=4, scale=1.0, dropout=0.0, bias=False):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult * scale)
        self.linear1 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear2 = nn.Linear(dim, inner_dim, bias=bias)
        self.linear3 = nn.Linear(inner_dim, dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    @torch.compile
    def silu_multiply(self, a, b):
        return F.silu(a) * b

    def forward(self, hidden_states):
        hidden_states_1 = self.linear1(hidden_states)
        hidden_states_2 = self.linear2(hidden_states)
        hidden_states = self.silu_multiply(hidden_states_1, hidden_states_2)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.linear3(hidden_states)
        return hidden_states


@dataclass
class TemporalAttentionConfig(PrintableConfig):
    attn_type: Literal["1d", "stfit", "neighbourhood_attn"] = "1d"
    """temporal attn type."""
    neighbourhood_attn_window_size: Union[int, Tuple[int, int, int]] = (5, 5, 5)
    """neighbourhood_attn window size."""
    neighbourhood_attn_dilation_size: Union[int, Tuple[int, int, int]] = (1, 1, 1)
    """neighbourhood_attn dilation size."""
    use_3d_rope: bool = True
    """Whether to use 3d rope in neighbourhood_attn and stfit."""
    theta_3d_rope: float = 10000.0
    """theta of 3d rope in neighbourhood_attn and stfit."""
    stfit_latent_dims_scale: int = 1
    """times of the latent dim number in stfit."""
    stfit_patch_size: Tuple[int, int] = (1, 2, 2)
    """merged token size in stfit."""


@dataclass
class TransformerXLModelConfig(Transformer2DModelConfig):
    _target: Type = field(default_factory=lambda: TransformerXLModel)

    num_frames: int = 1
    vae_temporal_scale_factor: int = 4
    use_2d_rope: bool = True
    theta_2d_rope: float = 10000.0
    use_temp_attn: bool = False
    use_1d_rope: bool = True
    theta_1d_rope: float = 10000.0
    num_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    attention_head_dim: Optional[int] = None
    cross_attention_dim: Optional[int] = None
    temporal_attention_config: Optional[TemporalAttentionConfig] = None
    resolution: Tuple[int, int, int] = (16, 32, 32)
    image_temp_attn: bool = False
    use_additional_conditions: bool = False
    use_text_condition: bool = False
    use_resolution_condition: bool = False
    use_aspect_ratio_condition: bool = False
    use_frames_condition: bool = False
    use_fps_condition: bool = False
    split_conditions: bool = False
    ffn_scale: float = 1.0

    def from_pretrained(self, ckpt_path, **kwargs):
        if self.in_channels is not None:
            kwargs["in_channels"] = self.in_channels
        if self.out_channels is not None:
            kwargs["out_channels"] = self.out_channels

        if self.from_scratch:
            config = self._target.load_config(ckpt_path, subfolder="transformer")
            transformer = self._target.from_config(config, low_cpu_mem_usage=False, device_map=None, transformer_config=self, **kwargs)
        else:
            transformer = self._target.from_pretrained(
                ckpt_path, subfolder="transformer", ignore_mismatched_sizes=True, low_cpu_mem_usage=False, device_map=None, transformer_config=self, **kwargs
            )

        transformer.set_selfattn_processor(use_flash_attn=self.use_flash_attn, use_rope=self.use_2d_rope, qk_norm=self.qk_norm)
        transformer.set_crossattn_processor(use_flash_attn=self.use_flash_attn, qk_norm=self.qk_norm)
        # for temporal attn processor
        if self.temporal_attention_config is not None and self.temporal_attention_config.attn_type != "1d":
            if self.temporal_attention_config.attn_type == "neighbourhood_attn":
                if isinstance(self.temporal_attention_config.neighbourhood_attn_window_size, int):
                    self.temporal_attention_config.neighbourhood_attn_window_size = (self.temporal_attention_config.neighbourhood_attn_window_size,) * 3
                if isinstance(self.temporal_attention_config.neighbourhood_attn_dilation_size, int):
                    self.temporal_attention_config.neighbourhood_attn_dilation_size = (self.temporal_attention_config.neighbourhood_attn_dilation_size,) * 3
                transformer.set_neighbourhood_attn_processor(
                    self.resolution,
                    self.temporal_attention_config.neighbourhood_attn_window_size,
                    self.temporal_attention_config.neighbourhood_attn_dilation_size,
                    self.temporal_attention_config.use_3d_rope,
                    self.temporal_attention_config.theta_3d_rope,
                )
            elif self.temporal_attention_config.attn_type == "stfit":
                transformer.set_stfit_attn_processor(
                    use_flash_attn=self.use_flash_attn, use_rope=self.temporal_attention_config.use_3d_rope, qk_norm=self.qk_norm
                )
            else:
                raise NotImplementedError
        else:
            transformer.set_tempattn_processor(use_flash_attn=self.use_flash_attn, use_rope=self.use_1d_rope, qk_norm=self.qk_norm)

        def rename_func(state_dict):
            new_dict = {}
            for k in state_dict.keys():
                ori_k = k
                if "transformer_blocks" not in k:  # global
                    if "scale_table" in k:
                        if "global_scale_table" not in k:
                            k = k.replace("scale_table", "global_scale_table")  # rename
                        if "weight" not in k:
                            k += ".weight"  # nn.Parameter -> nn.Embedding
                else:  # layer
                    if "scale_table" in k and "scale_table.weight" not in k:
                        k += ".weight"
                if k.startswith("transformer."):
                    k = k[12:]
                k = k.replace("ada_norm_single", "adaln_single")
                new_dict[k] = state_dict[ori_k]
            return new_dict

        if isinstance(self.transformer_ckpt_path, str):
            transformer = load_model(transformer, self.transformer_ckpt_path, rename_func=rename_func)
        elif isinstance(self.transformer_ckpt_path, list):
            for transformer_ckpt_path in self.transformer_ckpt_path:
                transformer = load_model(transformer, transformer_ckpt_path, rename_func=rename_func)

        return transformer


from diffusers.models.attention import FeedForward, _chunked_feed_forward
from diffusers.utils.torch_utils import maybe_allow_in_graph


@maybe_allow_in_graph
class BasicTransformerXLBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        dropout=0.0,
        cross_attention_dim: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        attention_bias: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_elementwise_affine: bool = True,
        norm_type: str = "layer_norm",  # 'layer_norm', 'ada_norm', 'ada_norm_zero', 'ada_norm_single', 'ada_norm_continuous', 'layer_norm_i2vgen'
        norm_eps: float = 1e-5,
        final_dropout: bool = False,
        attention_type: str = "default",
        positional_embeddings: Optional[str] = None,
        num_positional_embeddings: Optional[int] = None,
        ada_norm_continous_conditioning_embedding_dim: Optional[int] = None,
        ada_norm_bias: Optional[int] = None,
        ff_inner_dim: Optional[int] = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
        use_temp_attn: bool = False,
        image_temp_attn: bool = False,
        ffn_scale: float = 1.0,
    ):
        super().__init__()

        # Define 4 blocks. Each block has its own normalization layer.

        # 1. Self-Attn
        self.norm1 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.attn1 = Attention(
            query_dim=dim,
            heads=num_attention_heads,
            dim_head=attention_head_dim,
            dropout=dropout,
            bias=attention_bias,
            cross_attention_dim=cross_attention_dim if only_cross_attention else None,
            upcast_attention=upcast_attention,
            out_bias=attention_out_bias,
        )

        # 2. Temp-Attn
        self.image_temp_attn = image_temp_attn
        self.use_temp_attn = use_temp_attn
        if use_temp_attn:
            self.normt = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attnt = Attention(
                query_dim=dim,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                cross_attention_dim=cross_attention_dim if only_cross_attention else None,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )

            self.attnt.to_out[0].weight.data.zero_()
            self.attnt.to_out[0].bias.data.zero_()

        # 3. Cross-Attn
        if cross_attention_dim is not None or double_self_attention:
            self.norm2 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

            self.attn2 = Attention(
                query_dim=dim,
                cross_attention_dim=cross_attention_dim if not double_self_attention else None,
                heads=num_attention_heads,
                dim_head=attention_head_dim,
                dropout=dropout,
                bias=attention_bias,
                upcast_attention=upcast_attention,
                out_bias=attention_out_bias,
            )
        else:
            self.norm2 = None
            self.attn2 = None

        # 4. Feed-forward
        self.norm3 = RMSNorm(dim, elementwise_affine=norm_elementwise_affine, eps=norm_eps)

        self.ff = SwiGLUFeedForward(
            dim,
            dropout=dropout,
            inner_dim=ff_inner_dim,
            scale=ffn_scale,
            bias=ff_bias,
        )

        # 5. Scale
        self.scale_table = nn.Embedding.from_pretrained(torch.randn(1, 4*dim) / dim**0.5)

        # let chunk size default to None
        self._chunk_size = None
        self._chunk_dim = 0

        self.register_buffer("tensor_0", torch.tensor([[[0]]]))
        # self.register_buffer("tensor_1", torch.tensor([[1]]))
        # self.register_buffer("tensor_2", torch.tensor([[2]]))
        # self.register_buffer("tensor_3", torch.tensor([[3]]))

    def set_chunk_feed_forward(self, chunk_size: Optional[int], dim: int = 0):
        # Sets chunk feed-forward
        self._chunk_size = chunk_size
        self._chunk_dim = dim

    def forward(
        self,
        hidden_states: torch.FloatTensor,
        spatial_attention_mask: Optional[torch.FloatTensor] = None,
        temporal_attention_mask: Optional[torch.FloatTensor] = None,
        spatial_temporal_attention_mask: Optional[torch.FloatTensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.FloatTensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        patch_resolution: Optional[Tuple[int, int, int]] = None,
        spatial_attn_mask_kwargs: Dict[str, Any] = None,
        temporal_attn_mask_kwargs: Dict[str, Any] = None,
        spatial_temporal_attn_mask_kwargs: Dict[str, Any] = None,
        cross_attn_mask_kwargs: Dict[str, Any] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        class_labels: Optional[torch.LongTensor] = None,
        num_frames: int = 1,
        added_cond_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.FloatTensor:
        # copied from diffusers/models/attention.py BasicTransformerBlock.forward
        cross_attention_kwargs = cross_attention_kwargs.copy() if cross_attention_kwargs is not None else {}

        # Notice that normalization is always applied before the real computation in the following blocks.
        # 1. Self-Attention
        batch_size, num_patches = hidden_states.shape[0], hidden_states.shape[1]
        assert batch_size % num_frames == 0
        real_batch_size = batch_size // num_frames

        temporal_patch_resolution = patch_resolution[0] if patch_resolution is not None else None
        spatial_patch_resolution = patch_resolution[1:] if patch_resolution is not None else None

        scale = self.scale_table(self.tensor_0) + timestep  # 1 1 1 d + b f 1 d
        scale_msa, scale_mta, scale_mca, scale_mlp = scale.view(real_batch_size, num_frames, 1, 4, -1).unbind(dim=-2)

        scale_msa = rearrange(scale_msa, "b f 1 d -> (b f) 1 d", f=num_frames)
        # scale_mta = rearrange(scale_mta, "b f 1 d -> (b f) 1 d", f=num_frames)
        scale_mca = rearrange(scale_mca, "b f 1 d -> (b f) 1 d", f=num_frames)
        scale_mlp = rearrange(scale_mlp, "b f 1 d -> (b f) 1 d", f=num_frames)

        norm_hidden_states = multiply_addition(self.norm1(hidden_states), scale_msa)
        attn_output = self.attn1(
            norm_hidden_states,
            attention_mask=spatial_attention_mask,
            patch_resolution=spatial_patch_resolution,
            selfattn_mask_kwargs=spatial_attn_mask_kwargs,
        )
        hidden_states = hidden_states + attn_output

        # 2. Temp-Attention
        if self.use_temp_attn and (num_frames > 1 or self.image_temp_attn):
            if self.attnt.processor.token_merge is not None:
                scale_mta = repeat(scale_mta, "b f 1 d -> b f p d", p=hidden_states.shape[1])
                scale_mta = rearrange(scale_mta, "b f p d -> b (f p) d")
                hidden_states = rearrange(hidden_states, "(b f) p d -> b (f p) d", f=num_frames)
                norm_hidden_states = multiply_addition(self.normt(hidden_states), scale_mta)
                attn_output = self.attnt(
                    norm_hidden_states,
                    attention_mask=spatial_temporal_attention_mask,
                    patch_resolution=patch_resolution,
                    selfattn_mask_kwargs=spatial_temporal_attn_mask_kwargs,
                )
                hidden_states = hidden_states + attn_output
                hidden_states = rearrange(hidden_states, "b (f p) d -> (b f) p d", f=num_frames)
            else:
                scale_mta = repeat(scale_mta, "b 1 d -> (b p) 1 d", p=num_patches)
                hidden_states = rearrange(hidden_states, "(b f) p d -> (b p) f d", b=real_batch_size)
                norm_hidden_states = multiply_addition(self.normt(hidden_states), scale_mta)
                attn_output = self.attnt(
                    norm_hidden_states,
                    attention_mask=temporal_attention_mask if num_frames > 1 else None,
                    patch_resolution=temporal_patch_resolution,
                    selfattn_mask_kwargs=temporal_attn_mask_kwargs if num_frames > 1 else None,
                )
                hidden_states = hidden_states + attn_output
                hidden_states = rearrange(hidden_states, "(b p) f d -> (b f) p d", b=real_batch_size)

        # 3. Cross-Attention
        if self.attn2 is not None:
            norm_hidden_states = multiply_addition(self.norm2(hidden_states), scale_mca)
            attn_output = self.attn2(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                patch_resolution=spatial_patch_resolution,
                selfattn_mask_kwargs=spatial_attn_mask_kwargs,
                crossattn_mask_kwargs=cross_attn_mask_kwargs,
            )
            hidden_states = attn_output + hidden_states

        # 4. Feed-forward
        norm_hidden_states = multiply_addition(self.norm3(hidden_states), scale_mlp)

        if self._chunk_size is not None:
            # "feed_forward_chunk_size" can be used to save memory
            ff_output = _chunked_feed_forward(self.ff, norm_hidden_states, self._chunk_dim, self._chunk_size)
        else:
            ff_output = self.ff(norm_hidden_states)

        hidden_states = ff_output + hidden_states

        return hidden_states


class TransformerXLModel(Transformer2DModel):
    _supports_gradient_checkpointing = True

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 16,
        attention_head_dim: int = 88,
        in_channels: Optional[int] = None,
        out_channels: Optional[int] = None,
        num_layers: int = 1,
        dropout: float = 0.0,
        norm_num_groups: int = 32,
        cross_attention_dim: Optional[int] = None,
        attention_bias: bool = False,
        sample_size: Optional[int] = None,
        num_vector_embeds: Optional[int] = None,
        patch_size: Optional[int] = None,
        activation_fn: str = "geglu",
        num_embeds_ada_norm: Optional[int] = None,
        use_linear_projection: bool = False,
        only_cross_attention: bool = False,
        double_self_attention: bool = False,
        upcast_attention: bool = False,
        norm_type: str = "layer_norm",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        attention_type: str = "default",
        caption_channels: int = None,
        transformer_config: TransformerXLModelConfig = None,
    ):

        num_layers = transformer_config.num_layers or num_layers
        num_attention_heads = transformer_config.num_attention_heads or num_attention_heads
        attention_head_dim = transformer_config.attention_head_dim or attention_head_dim
        cross_attention_dim = transformer_config.cross_attention_dim or cross_attention_dim

        super().__init__(
            num_attention_heads=num_attention_heads,
            attention_head_dim=attention_head_dim,
            in_channels=in_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            norm_num_groups=norm_num_groups,
            cross_attention_dim=cross_attention_dim,
            attention_bias=attention_bias,
            sample_size=sample_size,
            num_vector_embeds=num_vector_embeds,
            patch_size=patch_size,
            activation_fn=activation_fn,
            num_embeds_ada_norm=num_embeds_ada_norm,
            use_linear_projection=use_linear_projection,
            only_cross_attention=only_cross_attention,
            double_self_attention=double_self_attention,
            upcast_attention=upcast_attention,
            norm_type=norm_type,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            attention_type=attention_type,
            caption_channels=caption_channels,
            transformer_config=transformer_config,
        )

        inner_dim = num_attention_heads * attention_head_dim
        patch_size = transformer_config.patch_size or patch_size

        if self.is_input_patches:
            self.patch_size = patch_size
            self.pos_embed = NoPEPatchEmbed(
                height=sample_size,
                width=sample_size,
                patch_size=patch_size,
                in_channels=in_channels,
                embed_dim=inner_dim,
            )

        self.transformer_blocks = nn.ModuleList(
            [
                BasicTransformerXLBlock(
                    inner_dim,
                    num_attention_heads,
                    attention_head_dim,
                    dropout=dropout,
                    cross_attention_dim=cross_attention_dim,
                    activation_fn=activation_fn,
                    num_embeds_ada_norm=num_embeds_ada_norm,
                    attention_bias=attention_bias,
                    only_cross_attention=only_cross_attention,
                    double_self_attention=double_self_attention,
                    upcast_attention=upcast_attention,
                    norm_type=norm_type,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                    attention_type=attention_type,
                    use_temp_attn=transformer_config.use_temp_attn,
                    image_temp_attn=transformer_config.image_temp_attn,
                    ffn_scale=transformer_config.ffn_scale,
                )
                for d in range(num_layers)
            ]
        )

        del self.scale_shift_table
        self.global_scale_table = nn.Embedding.from_pretrained(torch.randn(1, inner_dim) / inner_dim**0.5)
        self.norm_out = RMSNorm(inner_dim, elementwise_affine=False, eps=1e-6)

        if transformer_config.use_additional_conditions:
            transformer_config.use_text_condition = True
            transformer_config.use_resolution_condition = True
            transformer_config.use_aspect_ratio_condition = True
            transformer_config.use_frames_condition = True
            transformer_config.use_fps_condition = True

        self.adaln_single = AdaRMSNormSingle(
            inner_dim,
            num_scales=4,
            use_text_condition=transformer_config.use_text_condition,  # false
            use_resolution_condition=transformer_config.use_resolution_condition,  # true
            use_unpadded_resolution_condition=False,
            use_frames_condition=transformer_config.use_frames_condition,  # true
            split_conditions=transformer_config.split_conditions,  # true
        )
        self.use_additional_conditions = self.adaln_single.emb.use_additional_conditions
        self.split_conditions = transformer_config.split_conditions
        self.transformer_config = transformer_config

    def set_tempattn_processor(self, use_flash_attn=False, use_rope=False, qk_norm=False):
        log_to_rank0(f"set temporal attention processor for {type(self)}, use_rope={use_rope}, qk_norm={qk_norm}, use_flash_attn={use_flash_attn}")

        if use_rope:
            rope = RotaryEmbeddingFast(
                embed_dim=self.attention_head_dim,
                patch_resolution=self.transformer_config.resolution[0],
                theta=self.transformer_config.theta_1d_rope,
            )
        else:
            rope = None

        for name, module in self.transformer_blocks.named_modules():
            if isinstance(module, Attention) and name.endswith("attnt"):
                processor = MaskedAttnProcessor2_0(
                    use_flash_attn=use_flash_attn,
                    rope=rope,
                    qk_norm=qk_norm,
                    embed_dim=self.attention_head_dim,
                )
                module.set_processor(processor)

    def set_neighbourhood_attn_processor(self, resolution, window_size_, dilation_, use_3d_rope, rope_theta):
        log_to_rank0(f"set neighbourhood attention processor for {type(self)}")
        for name, module in self.transformer_blocks.named_modules():
            if isinstance(module, Attention) and name.endswith("attnt"):
                processor = Neighbour3dAttnProcessor(
                    resolution=resolution,
                    window_size_s=window_size_[0],
                    window_dilation_s=dilation_[0],
                    window_size_t=window_size_[-1],
                    window_dilation_t=dilation_[-1],
                    use_3d_rope=use_3d_rope,
                    rope_theta=rope_theta,
                    embed_dim=self.attention_head_dim,
                )
                module.set_processor(processor)

    def set_stfit_attn_processor(self, use_flash_attn=False, use_rope=False, qk_norm=False):
        log_to_rank0(f"set stfit attention processor for {type(self)}, use_rope={use_rope}, qk_norm={qk_norm}, use_flash_attn={use_flash_attn}")

        # NOTE: 兼容之前的config
        if len(self.transformer_config.temporal_attention_config.stfit_patch_size) == 2:
            self.transformer_config.temporal_attention_config.stfit_patch_size = (1,) + self.transformer_config.temporal_attention_config.stfit_patch_size

        if use_rope:
            rope = RotaryEmbeddingFast(
                embed_dim=self.attention_head_dim,
                patch_resolution=(
                    ((self.transformer_config.num_frames - 1) // self.transformer_config.vae_temporal_scale_factor + 1)
                    // self.transformer_config.temporal_attention_config.stfit_patch_size[0],
                    self.transformer_config.height
                    // self.transformer_config.vae_scale_factor
                    // self.patch_size
                    // self.transformer_config.temporal_attention_config.stfit_patch_size[1],
                    self.transformer_config.width
                    // self.transformer_config.vae_scale_factor
                    // self.patch_size
                    // self.transformer_config.temporal_attention_config.stfit_patch_size[2],
                ),
                theta=self.transformer_config.temporal_attention_config.theta_3d_rope,
            )
        else:
            rope = None

        for name, module in self.transformer_blocks.named_modules():
            if isinstance(module, Attention) and name.endswith("attnt"):
                processor = MaskedAttnProcessor2_0(
                    use_flash_attn=use_flash_attn,
                    rope=rope,
                    qk_norm=qk_norm,
                    embed_dim=self.attention_head_dim,
                    token_merge_size=self.transformer_config.temporal_attention_config.stfit_patch_size,
                    hidden_dim=self.num_attention_heads * self.attention_head_dim,
                    latent_dim=self.num_attention_heads * self.attention_head_dim * self.transformer_config.temporal_attention_config.stfit_latent_dims_scale,
                )
                module.set_processor(processor)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        timestep: Optional[torch.LongTensor] = None,
        added_cond_kwargs: Dict[str, torch.Tensor] = None,
        class_labels: Optional[torch.LongTensor] = None,
        cross_attention_kwargs: Dict[str, Any] = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ):
        # ensure attention_mask is a bias, and give it a singleton query_tokens dimension.
        #   we may have done this conversion already, e.g. if we came here via UNet3DConditionModel#forward.
        #   we can tell by counting dims; if ndim == 2: it's a mask rather than a bias.
        # expects mask of shape:
        #   [batch, key_tokens]
        # adds singleton query_tokens dimension:
        #   [batch,                    1, key_tokens]
        # this helps to broadcast it as a bias over attention scores, which will be in one of the following shapes:
        #   [batch,  heads, query_tokens, key_tokens] (e.g. torch sdp attn)
        #   [batch * heads, query_tokens, key_tokens] (e.g. xformers or classic attn)

        real_batch_size, num_frames, height, width = (
            hidden_states.shape[0],
            hidden_states.shape[-3],
            hidden_states.shape[-2] // self.patch_size,
            hidden_states.shape[-1] // self.patch_size,
        )
        hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")

        # Retrieve lora scale.
        lora_scale = cross_attention_kwargs.get("scale", 1.0) if cross_attention_kwargs is not None else 1.0

        patch_size = (1, self.patch_size, self.patch_size)  # f h w
        encoder_attention_mask, cross_attn_mask_kwargs = prepare_mask(encoder_attention_mask, num_frames, patch_size, mask_type="cross")
        spatial_attention_mask, spatial_attn_mask_kwargs = prepare_mask(attention_mask, num_frames, patch_size, mask_type="spatial")
        temporal_attention_mask, temporal_attn_mask_kwargs = prepare_mask(attention_mask, num_frames, patch_size, mask_type="temporal")
        spatial_temporal_attention_mask, spatial_temporal_attn_mask_kwargs = prepare_mask(
            attention_mask,
            num_frames,
            patch_size,
            token_merge_size=(
                self.transformer_config.temporal_attention_config.stfit_patch_size if self.transformer_config.temporal_attention_config is not None else 1
            ),
            mask_type="3d",
        )

        # 1. Input
        if self.is_input_patches:
            hidden_states = self.pos_embed(hidden_states)

            # if self.adaln_single is not None:
            #     if self.use_additional_conditions and added_cond_kwargs is None:
            #         raise ValueError("`added_cond_kwargs` cannot be None when using additional conditions for `adaln_single`.")
            #     if self.split_conditions:
            #         timestep, timestep_temporal, timestep_out = self.adaln_single(
            #             timestep, added_cond_kwargs, batch_size=real_batch_size, hidden_dtype=hidden_states.dtype
            #         )
            #         embed_dim = timestep_out.shape[-1]
            #         timestep[:, embed_dim : 2 * embed_dim] = timestep_temporal[:, embed_dim : 2 * embed_dim]
            #         timestep_out = repeat(timestep_out, "b d -> (b f) d", f=num_frames)
            #     else:
            #         timestep, timestep_out = self.adaln_single(timestep, added_cond_kwargs, batch_size=real_batch_size, hidden_dtype=hidden_states.dtype)
            #         timestep_out = repeat(timestep_out, "b d -> (b f) d", f=num_frames)
            timestep, timestep_out = self.adaln_single(timestep.contiguous(), added_cond_kwargs, batch_size=real_batch_size, hidden_dtype=hidden_states.dtype)

            # timestep shape: (b f) d or b d
            if timestep.shape[0] == real_batch_size * num_frames:
                timestep = timestep.view(real_batch_size, num_frames, 1, -1)
                timestep_out = timestep_out.view(real_batch_size, num_frames, 1, -1)
            elif timestep.shape[0] == real_batch_size:
                timestep = timestep.view(real_batch_size, 1, 1, -1).repeat(1, num_frames, 1, 1)
                timestep_out = timestep_out.view(real_batch_size, 1, 1, -1).repeat(1, num_frames, 1, 1)
            else:
                raise ValueError(f"Invalid timestep shape: {timestep.shape} for batch_size: {real_batch_size} and num_frames: {num_frames}.")


        # 2. Blocks
        if self.caption_projection is not None and encoder_hidden_states is not None:
            encoder_hidden_states = self.caption_projection(encoder_hidden_states)
            encoder_hidden_states = encoder_hidden_states.view(real_batch_size, -1, hidden_states.shape[-1])
            encoder_hidden_states = repeat(encoder_hidden_states, "b t d -> (b f) t d", f=num_frames)

        for block in self.transformer_blocks:
            if self.training and self.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": True} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    spatial_attention_mask,
                    temporal_attention_mask,
                    spatial_temporal_attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    timestep,
                    (num_frames, height, width),
                    spatial_attn_mask_kwargs,
                    temporal_attn_mask_kwargs,
                    spatial_temporal_attn_mask_kwargs,
                    cross_attn_mask_kwargs,
                    cross_attention_kwargs,
                    class_labels,
                    num_frames,
                    **ckpt_kwargs,
                )
            else:
                hidden_states = block(
                    hidden_states,
                    spatial_attention_mask=spatial_attention_mask,
                    temporal_attention_mask=temporal_attention_mask,
                    spatial_temporal_attention_mask=spatial_temporal_attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    timestep=timestep,
                    patch_resolution=(num_frames, height, width),
                    spatial_attn_mask_kwargs=spatial_attn_mask_kwargs,
                    temporal_attn_mask_kwargs=temporal_attn_mask_kwargs,
                    spatial_temporal_attn_mask_kwargs=spatial_temporal_attn_mask_kwargs,
                    cross_attn_mask_kwargs=cross_attn_mask_kwargs,
                    cross_attention_kwargs=cross_attention_kwargs,
                    class_labels=class_labels,
                    num_frames=num_frames,
                )
        # 3. Output
        if self.is_input_patches:
            scale = self.global_scale_table(torch.tensor([[[0]]], device=hidden_states.device)) + timestep_out
            scale = rearrange(scale, "b f 1 d -> (b f) 1 d")
            hidden_states = self.norm_out(hidden_states)
            # Modulation
            hidden_states = multiply_addition(hidden_states, scale)
            hidden_states = self.proj_out(hidden_states)
            hidden_states = hidden_states.squeeze(1)

            # unpatchify
            if self.adaln_single is None:
                height = width = int(hidden_states.shape[1] ** 0.5)
            hidden_states = hidden_states.reshape(shape=(-1, height, width, self.patch_size, self.patch_size, self.out_channels))
            hidden_states = torch.einsum("nhwpqc->nchpwq", hidden_states)
            output = hidden_states.reshape(shape=(-1, self.out_channels, height * self.patch_size, width * self.patch_size))
            output = rearrange(output, "(b f) c h w -> b c f h w", f=num_frames)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
