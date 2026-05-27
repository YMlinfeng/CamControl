from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple, Type, Union

from transformers import AutoModel, AutoTokenizer

from ...models.text_encoder.base_text_encoder import TextEncoder, TextEncoderConfig, TextEncoderWrapper


@dataclass
class UAETextEncoderConfig(TextEncoderConfig):
    _target: Type = field(default_factory=lambda: UAETextEncoder)

    tokenizer_ckpt_path: Optional[str] = None
    """ckpt path for tokenizer"""

    text_encoder_ckpt_path: Optional[str] = None
    """ckpt path for text encoder"""


class UAETextEncoderWrapper(TextEncoderWrapper):
    def __call__(self, text_inputs, attention_mask, device):
        return [self._instance(**text_inputs.to(device)).last_hidden_state]


class UAETextEncoder(TextEncoder):
    def __init__(
        self,
        text_encoder_config: Optional[TextEncoderConfig] = None,
    ):
        super().__init__(text_encoder_config)
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_config.tokenizer_ckpt_path, trust_remote_code=True)
        self.text_encoder = UAETextEncoderWrapper(
            AutoModel.from_pretrained(text_encoder_config.text_encoder_ckpt_path, trust_remote_code=True, torch_dtype="auto")
        )
        self.tokenizer.model_max_length = text_encoder_config.max_length
        self.text_encoder.config.use_attention_mask = None
