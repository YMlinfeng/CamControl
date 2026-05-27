from dataclasses import dataclass, field
from typing import Any, List, Literal, Optional, Tuple, Type

try:
    from diffusers.models.transformers.transformer_temporal import TransformerTemporalModel
except:
    from diffusers.models.transformer_temporal import TransformerTemporalModel
from diffusers.models.resnet import ResnetBlock2D
from einops import rearrange

from ...utils import load_state_dict, log_to_rank0
from ..blocks import *
from .base_unet import UNet2D, UNet2DConfig


@dataclass
class AnimateDiffUNetConfig(UNet2DConfig):
    _target: Type = field(default_factory=lambda: AnimateDiffUNet)

    mm_ckpt_path: Optional[str] = "/group/ckpt/diffusers/animatediff/mm_sd_v15_v2.ckpt"
    """ckpt path for unet"""

    down_block_configs: Optional[List[Any]] = field(
        default_factory=lambda: [
            CrossAttnDownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnDownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnDownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            DownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
        ]
    )
    mid_block_config: Optional[Any] = UNetMidBlock3DCrossAttnConfig(motion_module_cls=TransformerTemporalModel)
    up_block_configs: Optional[List[Any]] = field(
        default_factory=lambda: [
            UpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnUpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnUpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnUpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
        ]
    )

    pe_type: Literal["abs", "rand"] = "abs"
    """type of positional encoding"""
    pe_max_len: Optional[int] = 32
    """max length of temporal positional encoding"""
    num_frames: int = 16
    """Number of frames for training"""
    train_len: Optional[int] = None
    """Number of frames during training, used for logn scale"""
    zero_init: bool = False
    """Whether to zero out temp attn"""
    replace_groupnorm_forward: bool = False
    """Whether to replace group norm forward"""


@dataclass
class AnimateDiffXLUNetConfig(AnimateDiffUNetConfig):
    _target: Type = field(default_factory=lambda: AnimateDiffUNet)

    mm_ckpt_path: Optional[str] = "/group/ckpt/diffusers/animatediff/mm_sdxl_v10_beta.ckpt"
    """ckpt path for unet"""

    down_block_configs: Optional[List[Any]] = field(
        default_factory=lambda: [
            DownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnDownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnDownBlock3DConfig(motion_module_cls=TransformerTemporalModel),
        ]
    )
    mid_block_config: Optional[Any] = UNetMidBlock3DCrossAttnConfig()
    up_block_configs: Optional[List[Any]] = field(
        default_factory=lambda: [
            CrossAttnUpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            CrossAttnUpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
            UpBlock3DConfig(motion_module_cls=TransformerTemporalModel),
        ]
    )

    pe_type: Literal["abs", "rand"] = "abs"
    """type of positional encoding"""
    pe_max_len: Optional[int] = 32
    """max length of temporal positional encoding"""
    num_frames: int = 16
    """Number of frames for training"""
    train_len: Optional[int] = None
    """Number of frames during training, used for logn scale"""
    zero_init: bool = False
    """Whether to zero out temp attn"""
    replace_groupnorm_forward: bool = False
    """Whether to replace group norm forward"""


class AnimateDiffUNet(UNet2D):
    def setup(self):
        super().setup()
        if self.unet_config.replace_groupnorm_forward:
            self.replace_groupnorm_forward()

    def load_ckpts(self):
        if self.unet_config.mm_ckpt_path is not None:
            self.load_mm_ckpt(self.unet_config.mm_ckpt_path)
        super().load_ckpts()
        if self.unet_config.zero_init:
            self.zero_init()

    def load_mm_ckpt(self, ckpt_path):
        log_to_rank0(f"Loading model from {ckpt_path}...")
        state_dict = load_state_dict(ckpt_path)
        state_dict = self.rename_to_m2v(state_dict)
        missing, unexpected = self.load_state_dict(state_dict, strict=False)
        assert not unexpected

    def replace_attn_processor(self):
        AnimateDiffUNet.replace_tempattn_processor(self, self.unet_config)

    @staticmethod
    def replace_tempattn_processor(model, config):
        from diffusers.models.attention_processor import Attention

        from ..attention_processor import AnimateDiffAttnProcessor2_0

        log_to_rank0(f"Replacing temporal attention processor in {type(model)}'s motion modules.")
        for name, module in model.named_modules():
            if "motion_modules" in name and isinstance(module, Attention):
                processor = AnimateDiffAttnProcessor2_0(
                    dim=module.to_q.in_features, max_len=config.pe_max_len, pe_type=config.pe_type, train_len=config.train_len
                )
                module.set_processor(processor)

    def replace_groupnorm_forward(self):
        def reshape_to_5d(func):
            def wrapper(hidden_states, **kwargs):
                hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=self.num_frames)
                hidden_states = func(hidden_states, **kwargs)
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                return hidden_states

            return wrapper

        def reshape_to_4d(func):
            def wrapper(hidden_states, **kwargs):
                hidden_states = rearrange(hidden_states, "b c f h w -> (b f) c h w")
                hidden_states = func(hidden_states, **kwargs)
                hidden_states = rearrange(hidden_states, "(b f) c h w -> b c f h w", f=self.num_frames)
                return hidden_states

            return wrapper

        for module in self.modules():
            if isinstance(module, TransformerTemporalModel):
                module.norm.forward = reshape_to_4d(module.norm.forward)
            if isinstance(module, ResnetBlock2D):
                module.norm1.forward = reshape_to_5d(module.norm1.forward)
                module.norm2.forward = reshape_to_5d(module.norm2.forward)
        self.conv_norm_out.forward = reshape_to_5d(self.conv_norm_out.forward)

    def zero_init(self):
        for module in self.modules():
            if isinstance(module, TransformerTemporalModel):
                module.proj_out.weight.data.zero_()
                if hasattr(module, "bias"):
                    module.proj_out.bias.data.zero_()

    @staticmethod
    def rename_to_m2v(state_dict):
        rules = {
            "temporal_transformer.": "",
            "attention_blocks.0": "attn1",
            "attention_blocks.1": "attn2",
            "norms.0": "norm1",
            "norms.1": "norm2",
            "ff_norm": "norm3",
            "pos_encoder.pe": "processor.pos_encoder.pe",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            if "motion_modules" in key:
                if key.endswith("pos_encoder.pe"):
                    continue
                for name in rules:
                    if name in key:
                        key = key.replace(name, rules[name])
            new_state_dict[key] = value
        return new_state_dict

    @staticmethod
    def rename_from_m2v(state_dict):
        rules = {
            "attn1": "attention_blocks.0",
            "attn2": "attention_blocks.1",
            "norm1": "norms.0",
            "norm2": "norms.1",
            "norm3": "ff_norm",
            "processor.pos_encoder.pe": "pos_encoder.pe",
        }
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            if "motion_modules" in key:
                prefix, endfix = key.split("motion_modules")
                endfix = endfix.split(".")
                endfix.insert(2, "temporal_transformer")
                endfix = ".".join(endfix)
                new_key = "motion_modules".join([prefix, endfix])
                for name in rules:
                    if name in new_key:
                        new_key = new_key.replace(name, rules[name])
            new_state_dict[new_key] = value
        return new_state_dict
