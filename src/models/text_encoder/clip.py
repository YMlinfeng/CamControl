from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type, Union

from transformers import CLIPTextModel, CLIPTokenizer

from ...models.text_encoder.base_text_encoder import TextEncoder, TextEncoderConfig, TextEncoderWrapper


@dataclass
class ClipTextEncoderConfig(TextEncoderConfig):
    _target: Type = field(default_factory=lambda: ClipTextEncoder)

    tokenizer_ckpt_path: Optional[str] = None
    """ckpt path for tokenizer"""

    text_encoder_ckpt_path: Optional[str] = None
    """ckpt path for text encoder"""


class ClipTextEncoder(TextEncoder):
    def __init__(
        self,
        text_encoder_config: Optional[TextEncoderConfig] = None,
    ):
        super().__init__(text_encoder_config)
        self.tokenizer = CLIPTokenizer.from_pretrained(text_encoder_config.tokenizer_ckpt_path)
        self.text_encoder = TextEncoderWrapper(CLIPTextModel.from_pretrained(text_encoder_config.text_encoder_ckpt_path))
        self.tokenizer.model_max_length = text_encoder_config.max_length
