# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import copy
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ...activations import ACT2FN
from ...generation import GenerationMixin
from ...modeling_outputs import CausalLMOutput, BaseModelOutputWithPast
from ...modeling_utils import PreTrainedModel
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging
from ..auto import AutoModel
from ..llama.modeling_llama import LlamaRMSNorm
from .modular_vibepod_tokenizer import VibePodModel, VibePodTokenizer
from .modular_vibepod_diffusion_head import VibePodPredictionHead, VibePodDiffusionHeadModel
from .schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibepod import VibePodConfig

from .modular_vibepod_decoder import VibePodDecoder
from .modular_vibepod_tokenizer import VibePodTokenizer
from .modular_vibepod_diffusion_head import VibePodDiffusionHeadModel

logger = logging.get_logger(__name__)


class SpeechConnector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, output_dim)
        self.norm = LlamaRMSNorm(output_dim, eps=1e-6)
        self.fc2 = nn.Linear(output_dim, output_dim)

    def forward(self, features, **kwargs):    
        x = self.fc1(features)
        x = self.norm(x)
        x = self.fc2(x)
        return x

@auto_docstring
class VibePodPreTrainedModel(PreTrainedModel):
    config_class = VibePodConfig
    base_model_prefix = ""
    supports_gradient_checkpointing = True
    _skip_keys_device_placement = "past_key_values"
    _supports_cache_class = True
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_quantized_cache = True
    _supports_static_cache = True
    _supports_attention_backend = True

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.weight.data.fill_(1.0)
            module.bias.data.zero_()

@auto_docstring
class VibePodModel(VibePodPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize Qwen2 model for language modeling
        self.language_model = VibePodDecoder(config.language_model_config)
        
        # Initialize speech components if needed
        self.speech_encoder = VibePodTokenizer(config=config.acoustic_tokenizer_config)
        self.speech_connector = SpeechConnector(config.speech_vae_dim, config.hidden_size)
        self.speech_semantic_encoder = VibePodTokenizer(config=config.semantic_tokenizer_config)
        self.speech_semantic_connector = SpeechConnector(config.speech_vae_dim, config.hidden_size)
        
        # Register scaling factors as buffers
        self.register_buffer('speech_scaling_factor', torch.tensor(1.0))  
        self.register_buffer('speech_bias_factor', torch.tensor(0.0))
        
        # Initialize prediction head for speech generation
        self.prediction_head = VibePodPredictionHead(
            config, 
            latent_size=config.speech_vae_dim
        )
        
        # Initialize noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.ddpm_num_steps,
            beta_schedule=config.ddpm_beta_schedule,
            prediction_type=config.prediction_type
        )
        
        # LM head for text generation
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.embed_tokens
    
    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def set_speech_tokenizers(self, speech_tokenizer=None, speech_semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.speech_encoder = speech_tokenizer
        self.speech_semantic_encoder = speech_semantic_tokenizer
        
        # Reset the encoder to evaluation mode
        if self.speech_encoder is not None:
            self.speech_encoder.eval()
            
        if self.speech_semantic_encoder is not None:
            self.speech_semantic_encoder.eval()
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speech_input_ids: Optional[torch.FloatTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        """
        Forward pass for the VibePod model.
        
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of input sequence tokens in the vocabulary.
            attention_mask (`torch.FloatTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices.
            position_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Indices of positions of each input sequence tokens in the position embeddings.
            past_key_values (`tuple(tuple(torch.FloatTensor))`, *optional*):
                Cached past key and value projection states.
            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Embedded inputs to the model.
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for language modeling loss computation.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned.
            output_attentions (`bool`, *optional*):
                Whether to return the attentions tensors of all attention layers.
            output_hidden_states (`bool`, *optional*):
                Whether to return the hidden states of all layers.
            return_dict (`bool`, *optional*):
                Whether to return a `BaseModelOutputWithPastAndCrossAttentions` instead of a tuple.
            speech_input_ids (`torch.FloatTensor`, *optional*):
                Speech inputs to be processed.
            speech_input_mask (`torch.BoolTensor`, *optional*):
                Mask to indicate speech token positions in the sequence.
            
        Returns:
            `CausalLMOutput` or tuple: The language model outputs.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Handle speech inputs if provided
        if speech_input_ids is not None and speech_input_mask is not None:
            # Convert speech inputs to embeddings
            speech_embeds = self._process_speech_inputs(speech_input_ids, speech_input_mask)
            
            # If inputs_embeds is not provided, create it from input_ids
            if inputs_embeds is None and input_ids is not None:
                inputs_embeds = self.language_model.embed_tokens(input_ids)
                
            # Replace positions indicated by speech_input_mask with speech embeddings
            if inputs_embeds is not None:
                flat_speech_embeds = speech_embeds.reshape(-1, self.config.hidden_size)
                inputs_embeds[speech_input_mask] = flat_speech_embeds
        
        # Forward pass through the language model
        outputs = self.language_model(
            input_ids=None if inputs_embeds is not None else input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output
            
        return CausalLMOutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def _process_speech_inputs(self, speech_inputs, speech_masks):
        """Process speech inputs through speech encoder and connector."""
        with torch.no_grad():
            # Process through speech encoder
            if isinstance(self.speech_encoder, VibePodModel):
                speech_features = self.speech_encoder.encode(speech_inputs)[speech_masks]
                
                # Apply scaling and bias
                if not torch.isnan(self.speech_scaling_factor) and not torch.isnan(self.speech_bias_factor):
                    speech_features = (speech_features + self.speech_bias_factor) * self.speech_scaling_factor
            else:
                # For non-VibePod speech encoders
                speech_features = speech_inputs
            
            # Connect to language model's embedding space
            speech_embeds = self.speech_connector(speech_features)
            
        return speech_embeds
    
class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring(
    custom_intro="""
    VibePod.
    """
)
class VibePodForConditionalGeneration(VibePodPreTrainedModel, GenerationMixin):
    pass

