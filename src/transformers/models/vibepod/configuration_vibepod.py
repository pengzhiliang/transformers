# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" VibePod_AcousticTokenizer model configuration"""

from typing import Dict, List, Optional, Tuple

from ...configuration_utils import PretrainedConfig 
from ...utils import logging


logger = logging.get_logger(__name__)


class VibePodAcousticTokenizerConfig(PretrainedConfig):
    model_type = "vibepod"
    base_config_key = "acoustic_tokenizer"

    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0.5,
        std_dist_type: str = 'gaussian',
        # common 
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        # decoder specific
        decoder_n_filters: int = 32,
        decoder_ratios: Optional[List[int]] = None, # if None, same as encoder
        decoder_depths: Optional[str] = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        
        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        
        # decoder specific parameters
        self.decoder_ratios = decoder_ratios if decoder_ratios is not None else encoder_ratios
        self.decoder_n_filters = decoder_n_filters
        self.decoder_depths = decoder_depths


class VibePodSemanticTokenizerConfig(PretrainedConfig):
    model_type = "vibepod"
    base_config_key = "semantic_tokenizer"
    
    def __init__(
        self,
        channels: int = 1,
        corpus_normalize: float = 0.0,
        causal: bool = True,
        vae_dim: int = 64,
        fix_std: float = 0,
        std_dist_type: str = 'none',
        # common 
        mixer_layer: str = 'depthwise_conv',
        conv_norm: str = 'none',
        pad_mode: str = 'constant',
        disable_last_norm: bool = True,
        layernorm: str = 'RMSNorm',
        layernorm_eps: float = 1e-5,
        layernorm_elementwise_affine: bool = True,
        conv_bias: bool = True,
        layer_scale_init_value: float = 1e-6,
        weight_init_value: float = 1e-2,
        # encoder specific
        encoder_n_filters: int = 32,
        encoder_ratios: Optional[List[int]] = [8,5,5,4,2,2],
        encoder_depths: str = "3-3-3-3-3-3-8",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.channels = channels
        self.corpus_normalize = corpus_normalize
        self.causal = causal
        self.vae_dim = vae_dim
        self.fix_std = fix_std
        self.std_dist_type = std_dist_type
        
        # common parameters
        self.conv_norm = conv_norm
        self.pad_mode = pad_mode
        self.layernorm_eps = layernorm_eps
        self.disable_last_norm = disable_last_norm
        self.layernorm = layernorm
        self.layernorm_elementwise_affine = layernorm_elementwise_affine
        self.conv_bias = conv_bias
        self.layer_scale_init_value = layer_scale_init_value
        self.weight_init_value = weight_init_value
        self.mixer_layer = mixer_layer

        # encoder specific parameters
        self.encoder_n_filters = encoder_n_filters
        self.encoder_ratios = encoder_ratios
        self.encoder_depths = encoder_depths
        

class VibePodDiffusionHeadConfig(PretrainedConfig):
    model_type = "vibepod"
    base_config_key = "diffusion_head"

    def __init__(
        self,
        hidden_size=768,
        head_layers=4,
        head_ffn_ratio=3.0,
        rms_norm_eps=1e-5,
        latent_size=16,
        speech_vae_dim=None,
        prediction_type="v_prediction",
        diffusion_type="ddpm",
        ddpm_num_steps=1000,
        ddpm_num_inference_steps=20,
        ddpm_beta_schedule="cosine",
        ddpm_batch_mul=4,
        **kwargs
    ):
        self.hidden_size = hidden_size
        self.head_layers = head_layers
        self.head_ffn_ratio = head_ffn_ratio
        self.rms_norm_eps = rms_norm_eps
        self.latent_size = latent_size
        self.speech_vae_dim = speech_vae_dim
        self.prediction_type = prediction_type
        self.diffusion_type = diffusion_type
        self.ddpm_num_steps = ddpm_num_steps
        self.ddpm_num_inference_steps = ddpm_num_inference_steps
        self.ddpm_beta_schedule = ddpm_beta_schedule
        self.ddpm_batch_mul = ddpm_batch_mul
        
        super().__init__(**kwargs)

class VibePodDecoderConfig(PretrainedConfig):
    model_type = "vibepod"
    base_config_key = 'decoder'

    # Default tensor parallel plan for base model `Qwen2`
    base_model_tp_plan = {
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.up_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
    }
    base_model_pp_plan = {
        "embed_tokens": (["input_ids"], ["inputs_embeds"]),
        "layers": (["hidden_states", "attention_mask"], ["hidden_states"]),
        "norm": (["hidden_states"], ["hidden_states"]),
    }

    def __init__(
        self,
        vocab_size=151936,
        hidden_size=4096,
        intermediate_size=22016,
        num_hidden_layers=32,
        num_attention_heads=32,
        num_key_value_heads=32,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=10000.0,
        rope_scaling=None,
        use_sliding_window=False,
        sliding_window=4096,
        max_window_layers=28,
        attention_dropout=0.0,
        **kwargs,
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window  # we check `use_sliding_window` in the modeling code
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

class VibePodConfig(PretrainedConfig):
    model_type = "vibepod"
    sub_configs = {
        "acoustic_tokenizer": VibePodAcousticTokenizerConfig, 
        "semantic_tokenizer": VibePodSemanticTokenizerConfig,
        "decoder": VibePodDecoderConfig,
        "diffusion_head": VibePodDiffusionHeadConfig,
    }
    # keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        acoustic_tokenizer_config=None,
        semantic_tokenizer_config=None,
        decoder_config=None,
        diffusion_head_config=None,
          
        # Multi-modal parameters
        batch_size: Optional[int] = None,
        tokens_per_sample: Optional[int] = None,
        model_parallel_size: int = 1,
        
        # Head configuration
        head_layers: int = 4,
        head_ffn_ratio: float = 3.0,
        
        # VAE parameters
        vae_model: Optional[str] = None,
        use_vae_mode: bool = False,
        
        # Connector and image parameters
        connector: str = "simple",
        image_size: Optional[int] = None,
        latent_query_num: Optional[int] = None,
        input_size: int = 16,
        latent_size: int = 16,
        
        # Speech parameters
        speech_tokenizer: Optional[str] = None,
        speech_vae_dim: Optional[int] = None,
        speech_semantic_tokenizer: Optional[str] = None,
        speech_semantic_vae_dim: Optional[int] = None,
        
        # Additional config parameters from PretrainedConfig
        pad_token_id: Optional[int] = None,
        bos_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs
    ):
        if isinstance(acoustic_tokenizer_config, dict):
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer"](**acoustic_tokenizer_config)
        elif acoustic_tokenizer_config is None:
            self.acoustic_tokenizer_config = self.sub_configs["acoustic_tokenizer"]()

        if isinstance(semantic_tokenizer_config, dict):
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer"](**semantic_tokenizer_config)
        elif semantic_tokenizer_config is None:
            self.semantic_tokenizer_config = self.sub_configs["semantic_tokenizer"](**kwargs)
        
        if isinstance(decoder_config, dict):
            self.decoder_config = self.sub_configs["decoder"](**decoder_config)
        elif decoder_config is None:
            self.decoder_config = self.sub_configs["decoder"](**kwargs)

        if isinstance(diffusion_head_config, dict):
            self.diffusion_head_config = self.sub_configs["diffusion_head"](**diffusion_head_config)
        elif diffusion_head_config is None:
            self.diffusion_head_config = self.sub_configs["diffusion_head"](**kwargs)

        # LLM architecture parameters
        self.dim = dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.vocab_size = vocab_size
        self.max_batch_size = max_batch_size
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.norm_eps = norm_eps
        
        # VAE parameters
        self.vae_model = vae_model
        self.use_vae_mode = use_vae_mode
        
        # Connector and image parameters
        self.connector = connector
        self.image_size = image_size
        self.latent_query_num = latent_query_num
        self.input_size = input_size
        self.latent_size = latent_size
        
        # Speech parameters
        self.speech_tokenizer = speech_tokenizer
        self.speech_vae_dim = speech_vae_dim
        self.speech_semantic_tokenizer = speech_semantic_tokenizer
        self.speech_semantic_vae_dim = speech_semantic_vae_dim

        super().__init__(**kwargs)

__all__ = [
    "VibePodAcousticTokenizerConfig", 
    "VibePodSemanticTokenizerConfig", 
    "VibePodDiffusionHeadConfig", 
    "VibePodDecoderConfig",
    "VibePodConfig"
]