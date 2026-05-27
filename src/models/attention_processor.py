import math
from typing import Optional, Tuple, Union

try:
    import natten
except:
    pass
import flash_attn
import torch
import torch.nn as nn
import torch.nn.functional as F


from diffusers.models.attention_processor import Attention
from diffusers.models.embeddings import SinusoidalPositionalEmbedding
from diffusers.utils import USE_PEFT_BACKEND
from einops import rearrange, reduce, repeat
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input  # noqa

from .blocks import TokenMerge, TokenSplit, TokenSplitWithoutSkip
from .embeddings import RandomSinusoidalPositionalEmbedding, RotaryEmbeddingFast
from .normalization import RMSNorm


class Neighbour3dAttnProcessor(nn.Module):
    def __init__(
        self,
        resolution=(16, 32, 32),
        window_size_s=7,
        window_dilation_s=1,
        window_size_t=7,
        window_dilation_t=1,
        use_3d_rope=False,
        rope_theta=None,
        embed_dim=72,
    ):
        super().__init__()
        self.window_size_s = window_size_s
        self.window_dilation_s = window_dilation_s
        self.window_size_t = window_size_t
        self.window_dilation_t = window_dilation_t
        self.T, self.H, self.W = resolution
        self.use_3d_rope = use_3d_rope
        if self.use_3d_rope:
            self.rope = RotaryEmbeddingFast(embed_dim=embed_dim, patch_resolution=resolution, theta=rope_theta)

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        seqlens_in_batch_kv=None,
        indices_kv=None,
        max_seqlen_in_batch_kv=None,
        cu_seqlens_kv=None,
        patch_resolution=None,
        selfattn_mask_kwargs=None,
        **kwargs,
    ) -> torch.FloatTensor:

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim)
        key = key.view(batch_size, -1, attn.heads, head_dim)
        value = value.view(batch_size, -1, attn.heads, head_dim)
        value = rearrange(value, "(b h w) t nh hd -> b nh (t h w) hd", t=self.T, h=self.H, w=self.W)
        key = rearrange(key, "(b h w) t nh hd -> b nh (t h w) hd", t=self.T, h=self.H, w=self.W)
        query = rearrange(query, "(b h w) t nh hd -> b nh (t h w) hd", t=self.T, h=self.H, w=self.W)
        if self.use_3d_rope:
            query = self.rope(query)
            key = self.rope(key)

        value = rearrange(value, "b nh (t h w) hd -> b nh t h w hd", t=self.T, h=self.H, w=self.W)
        key = rearrange(key, "b nh (t h w) hd -> b nh t h w hd", t=self.T, h=self.H, w=self.W)
        query = rearrange(query, "b nh (t h w) hd -> b nh t h w hd", t=self.T, h=self.H, w=self.W)

        dtpye_backup = query.dtype
        query = query.to(torch.float32)
        key = key.to(torch.float32)
        value = value.to(torch.float32)

        qk = natten.functional.natten3dqk(
            query, key, kernel_size=self.window_size_s, kernel_size_d=self.window_size_t, dilation=self.window_dilation_s, dilation_d=self.window_dilation_t
        )
        ats = torch.softmax(qk, dim=-1).to(torch.float32)
        hidden_states = natten.functional.natten3dav(
            ats, value, kernel_size=self.window_size_s, kernel_size_d=self.window_size_t, dilation=self.window_dilation_s, dilation_d=self.window_dilation_t
        )

        hidden_states = hidden_states.to(dtpye_backup)
        query = query.to(dtpye_backup)
        key = key.to(dtpye_backup)
        value = value.to(dtpye_backup)

        hidden_states = rearrange(hidden_states, "b nh t h w hd -> (b h w) t nh hd")

        hidden_states = hidden_states.reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class AnimateDiffAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, dim, max_len, pe_type: str = "abs", train_len: Optional[int] = None):
        super().__init__()
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")

        if pe_type == "abs":
            self.pos_encoder = SinusoidalPositionalEmbedding(dim, max_len)
        elif pe_type == "rand":
            self.pos_encoder = RandomSinusoidalPositionalEmbedding(dim, max_len)
        else:
            raise NotImplementedError

        self.train_len = train_len

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        **kwargs,
    ) -> torch.FloatTensor:
        hidden_states = self.pos_encoder(hidden_states)

        residual = hidden_states

        args = () if USE_PEFT_BACKEND else (scale,)

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        args = () if USE_PEFT_BACKEND else (scale,)
        query = attn.to_q(hidden_states, *args)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states, *args)
        value = attn.to_v(encoder_hidden_states, *args)

        if self.train_len is not None:
            query = query * math.log(sequence_length, self.train_len)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states, *args)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class MaskedAttnProcessor2_0(nn.Module):
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, use_flash_attn=False, rope=None, qk_norm: bool = False, embed_dim=72, eps: float = 1e-6, token_merge_size = None, hidden_dim: int = 1152, latent_dim: int = 1152):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        super().__init__()

        if token_merge_size is not None:
            self.token_merge = TokenMerge(in_features=hidden_dim, out_features=latent_dim, patch_size=token_merge_size)
            self.token_split = TokenSplitWithoutSkip(in_features=latent_dim, out_features=hidden_dim, patch_size=token_merge_size)
        else:
            self.token_merge = None
            self.token_split = None

        self.rope = rope

        if qk_norm:
            self.q_norm = RMSNorm(embed_dim, eps=eps)
            self.k_norm = RMSNorm(embed_dim, eps=eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.use_flash_attn = use_flash_attn
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        if torch.cuda.is_available() and torch.version.hip:
            self.flash_attn_max_head_dim = 128
        elif torch.cuda.is_available() and torch.version.cuda:
            self.flash_attn_max_head_dim = 256
        else:
            self.flash_attn_max_head_dim = None

    def _attn_varlen(self, query, key, value, crossattn_mask_kwargs=None, selfattn_mask_kwargs=None):
        assert crossattn_mask_kwargs != None or selfattn_mask_kwargs != None, "crossattn_mask_kwargs 和 selfattn_mask_kwargs不可同时为None"

        batch, seqlen = query.shape[:2]

        # for q
        if selfattn_mask_kwargs is None:
            max_seqlen_in_batch_q = query.shape[1]
            cu_seqlens_q = torch.arange(0, query.shape[0] * query.shape[1] + 1, query.shape[1], dtype=torch.int32, device="cuda")
            indices_q = torch.arange(0, query.shape[0] * query.shape[1], device="cuda")
            query = rearrange(query, "b s ... -> (b s) ...")
        else:
            max_seqlen_in_batch_q = selfattn_mask_kwargs["max_seqlen_in_batch"]
            cu_seqlens_q = selfattn_mask_kwargs["cu_seqlens"]
            indices_q = selfattn_mask_kwargs["indices"]
            query = index_first_axis(rearrange(query, "b s ... -> (b s) ..."), indices_q)

        # for k & v
        if crossattn_mask_kwargs != None:
            cu_seqlens_kv = crossattn_mask_kwargs["cu_seqlens"]
            max_seqlen_in_batch_kv = crossattn_mask_kwargs["max_seqlen_in_batch"]
            indices_kv = crossattn_mask_kwargs["indices"]
        else:
            cu_seqlens_kv = selfattn_mask_kwargs["cu_seqlens"]
            max_seqlen_in_batch_kv = selfattn_mask_kwargs["max_seqlen_in_batch"]
            indices_kv = selfattn_mask_kwargs["indices"]

        # TODO: index_first_axis is not efficient.
        key = index_first_axis(rearrange(key, "b s ... -> (b s) ..."), indices_kv)
        value = index_first_axis(rearrange(value, "b s ... -> (b s) ..."), indices_kv)
        attn_output_unpad = flash_attn_varlen_func(
            query,
            key,
            value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_kv,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_kv,
            dropout_p=0.0,
            softmax_scale=None,
            causal=False,
        )

        hidden_states = pad_input(attn_output_unpad, indices_q, batch, seqlen)
        return hidden_states

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        scale: float = 1.0,
        patch_resolution: Optional[Tuple[int, int, int]] = None,
        crossattn_mask_kwargs: Optional[dict] = None,
        selfattn_mask_kwargs: Optional[dict] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        if self.token_merge is not None:
            hidden_states = rearrange(hidden_states, "b (t h w) d -> b t h w d", t=patch_resolution[0], h=patch_resolution[1], w=patch_resolution[2])
            hidden_states = self.token_merge(hidden_states)
            merge_b, merge_t, merge_h, merge_w, merge_d = hidden_states.shape
            patch_resolution = (merge_t, merge_h, merge_w)
            hidden_states = rearrange(hidden_states, "b t h w d -> b (t h w) d")

        batch_size, sequence_length, _ = hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        if self.rope is not None:
            query = self.rope(query, patch_resolution)
            key = self.rope(key, patch_resolution)

        if self.q_norm is not None:
            query = self.q_norm(query)
        if self.k_norm is not None:
            key = self.k_norm(key)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        if self.use_flash_attn and query.dtype is not torch.float32 and query.shape[-1] <= self.flash_attn_max_head_dim:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            if selfattn_mask_kwargs is None and crossattn_mask_kwargs is None:
                hidden_states = flash_attn_func(query, key, value, dropout_p=0.0, softmax_scale=None, causal=False)
            else:
                hidden_states = self._attn_varlen(query, key, value, crossattn_mask_kwargs=crossattn_mask_kwargs, selfattn_mask_kwargs=selfattn_mask_kwargs)
            hidden_states = hidden_states.transpose(1, 2)
        else:
            if attention_mask is not None:
                attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                # scaled_dot_product_attention expects attention_mask shape to be
                # (batch, heads, source_length, target_length)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

            hidden_states = F.scaled_dot_product_attention(query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False)

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if self.token_split is not None:
            hidden_states = rearrange(hidden_states, "b (t h w) d -> b t h w d", t=merge_t, h=merge_h, w=merge_w)
            hidden_states = self.token_split(hidden_states)
            hidden_states = rearrange(hidden_states, "b t h w d -> b (t h w) d")

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
