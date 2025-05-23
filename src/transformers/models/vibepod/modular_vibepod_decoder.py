import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn

from ...activations import ACT2FN
from ...cache_utils import Cache
from ...modeling_flash_attention_utils import flash_attn_supports_top_left_mask, is_flash_attn_available

from ...modeling_outputs import BaseModelOutput, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...utils import auto_docstring, logging
from ...utils.deprecation import deprecate_kwarg
from ...processing_utils import Unpack
from ..auto import AutoModel, AutoModelForCausalLM

from ..qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention, Qwen2DecoderLayer, Qwen2Model

from .configuration_vibepod import VibePodDecoderConfig

logger = logging.get_logger(__name__)


# class VibePodDecoderMLP(Qwen2MLP):
#     def __init__(self, config: VibePodDecoderConfig):
#         super().__init__(config)
        
# class VibePodAttention(Qwen2Attention):
#     def __init__(self, config: VibePodDecoderConfig, layer_idx: int):
#         super().__init__(config, layer_idx)

# class VibePodDecoderLayer(Qwen2DecoderLayer):
#     def __init__(self, config: VibePodDecoderConfig, layer_idx: int):
#         super().__init__(config, layer_idx)

class VibePodDecoder(Qwen2Model):
    def __init__(self, config: VibePodDecoderConfig):
        super().__init__(config)
        