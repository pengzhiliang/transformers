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
from ..qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention, Qwen2DecoderLayer, Qwen2Model
from .modular_vibepod_tokenizer import VibePodTokenizerStreamingCache, VibePodAcousticTokenizerModel, VibePodSemanticTokenizerModel
from .modular_vibepod_diffusion_head import VibePodPredictionHead, VibePodDiffusionHeadModel
from .schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibepod import VibePodConfig

from .modular_vibepod_text_tokenizer import VibePodTextTokenizer, VibePodTextTokenizerFast

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
        # Use the language model's initializer_range if available
        if hasattr(self.config, 'language_model_config') and hasattr(self.config.language_model_config, 'initializer_range'):
            std = self.config.language_model_config.initializer_range
        elif hasattr(self.config, 'decoder_config') and hasattr(self.config.decoder_config, 'initializer_range'):
            std = self.config.decoder_config.initializer_range
        else:
            std = 0.02  # Default value
            
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
        # Use decoder_config if available, otherwise use language_model_config
        lm_config = config.decoder_config 
        self.language_model = Qwen2Model(lm_config)
        
        # Initialize speech components if needed
        self.acoustic_tokenizer = VibePodAcousticTokenizerModel(config=config.acoustic_tokenizer_config)
        self.semantic_tokenizer = VibePodSemanticTokenizerModel(config=config.semantic_tokenizer_config)

        self.acoustic_connector = SpeechConnector(config.acostic_vae_dim, lm_config.hidden_size)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size)
        
        # Register scaling factors as buffers
        self.register_buffer('speech_scaling_factor', torch.tensor(1.0))  
        self.register_buffer('speech_bias_factor', torch.tensor(0.0))
        
        # Initialize prediction head for speech generation
        self.prediction_head = VibePodPredictionHead(
            config.diffusion_head_config, 
            latent_size=config.acostic_vae_dim
        )
        
        # Initialize noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )
        
        # LM head for text generation
        self.lm_head = nn.Linear(lm_config.hidden_size, lm_config.vocab_size, bias=False)
        
        # Initialize weights
        self.post_init()
    
    def get_input_embeddings(self):
        return self.language_model.embed_tokens

    def set_input_embeddings(self, value):
        self.language_model.embed_tokens = value
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.acoustic_tokenizer = acoustic_tokenizer
        self.semantic_tokenizer = semantic_tokenizer
        
        # Reset the encoder to evaluation mode
        if self.acoustic_tokenizer is not None:
            self.acoustic_tokenizer.eval()
            
        if self.semantic_tokenizer is not None:
            self.semantic_tokenizer.eval()
    
class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring(
    custom_intro="""
    VibePod: A multimodal TTS model combining Qwen2.5 LLM with diffusion for speech generation.
    """
)
class VibePodForConditionalGeneration(VibePodPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["model.lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the base model
        self.model = VibePodModel(config)
        
        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        # Tie lm_head.weight to language_model.embed_tokens.weight
        if hasattr(self.model, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.model.lm_head.weight = self.model.language_model.embed_tokens.weight
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.model.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.model.lm_head = new_embeddings
    
    def set_speech_tokenizers(self, acoustic_tokenizer=None, semantic_tokenizer=None):
        """Set the speech tokenizers used for encoding and decoding speech."""
        self.model.set_speech_tokenizers(acoustic_tokenizer, semantic_tokenizer)
    
    def set_ddpm_inference_steps(self, num_steps=None):
        self.ddpm_inference_steps = num_steps or self.config.diffusion_head_config.ddpm_num_inference_steps

    def _process_speech_inputs(self, speech_tensors, speech_masks, speech_type="audio"):
        """Process speech inputs through tokenizers and connectors."""
        with torch.no_grad():
            if speech_type == "audio":
                # Encode audio to acoustic latents
                encoder_output = self.model.acoustic_tokenizer.encode(speech_tensors.unsqueeze(1))
                acoustic_latents = encoder_output.sample(dist_type=self.model.acoustic_tokenizer.std_dist_type)[0]
                
                # Apply scaling and bias
                acoustic_features = (acoustic_latents + self.model.speech_bias_factor) * self.model.speech_scaling_factor
                
                # Connect to language model space
                acoustic_connected = self.model.acoustic_connector(acoustic_features)[speech_masks]
                
                return acoustic_features, acoustic_connected
            else:
                raise NotImplementedError(f"Speech type {speech_type} not implemented")
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        acoustic_loss_mask: Optional[torch.BoolTensor] = None,
        **kwargs,
    ):
        raise NotImplementedError("VibePodForConditionalGeneration does not support forward method directly. Use generate method for generation tasks.")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Process speech inputs if provided
        if speech_tensors is not None and speech_masks is not None:
            _, speech_embeds = self._process_speech_inputs(speech_tensors, speech_masks)
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds[speech_masks]
        
        # Forward through language model
        outputs = self.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        
        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        
        loss = None
        speech_diff_loss = None
        
        if labels is not None:
            # Compute language modeling loss
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Compute speech diffusion loss if needed
            if acoustic_loss_mask is not None and acoustic_loss_mask.sum() > 0:
                # Extract condition features for diffusion
                condition_features = hidden_states[acoustic_loss_mask]
                
                # Here you would compute the diffusion loss
                # This is a placeholder - you'll need to adapt based on your diffusion head
                speech_diff_loss = torch.tensor(0.0, device=hidden_states.device)
        
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
    
    @torch.no_grad()
    def sample_speech_tokens(self, condition, neg_condition, cfg_scale=3.0):
        self.model.noise_scheduler.set_timesteps(self.ddpm_inference_steps)
        condition = torch.cat([condition, neg_condition], dim=0)
        speech = torch.randn(condition.shape[0], self.config.acostic_vae_dim).to(condition)
        for t in self.model.noise_scheduler.timesteps:
            half = speech[: len(speech) // 2]
            combined = torch.cat([half, half], dim=0)
            eps = self.model.prediction_head(combined, t.repeat(combined.shape[0]).to(combined), condition=condition)
            cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
            half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            eps = torch.cat([half_eps, half_eps], dim=0)
            speech = self.model.noise_scheduler.step(eps, t, speech).prev_sample
        return speech[: len(speech) // 2]

    @torch.no_grad()
    def generate_multiple_speeches_semantic_acoustic(
        self,
        prompt_ids: torch.LongTensor,
        last_ids: torch.LongTensor,
        tokenizer,
        neg_prompt_ids: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        speech_type: str = "audio",
        max_new_tokens: int = 260,
        cfg_scale: float = 3.0,
        temperature: float = 0.0,
        top_p: float = 0.9,
        return_latent: bool = False,
        verbose: bool = False,
        **kwargs
    ):
        """
        Generate speech with text using CFG sampling.
        
        Args:
            prompt_ids: Input prompt token IDs
            last_ids: Last token ID (not included in prompt_ids)
            tokenizer: Tokenizer instance
            neg_prompt_ids: Negative prompt for CFG
            speech_tensors: Input speech waveforms for voice cloning
            speech_masks: Masks for speech tensors
            speech_input_mask: Positions to insert speech embeddings
            speech_type: Type of speech input ("audio" or "vae")
            max_new_tokens: Maximum tokens to generate
            cfg_scale: CFG scale factor
            temperature: Sampling temperature
            top_p: Top-p sampling threshold
            return_latent: Whether to return latent representations
            verbose: Whether to show progress bar
        
        Returns:
            output_tokens: Generated token IDs
            output_speech: List of generated speech waveforms or latents
        """
        device = prompt_ids.device
        batch_size = 1  # Currently only support batch_size=1
        
        # Get special tokens
        speech_begin_id = tokenizer.speech_start_id
        speech_end_id = tokenizer.speech_end_id
        pad_id = tokenizer.pad_id if hasattr(tokenizer, 'pad_id') else tokenizer.pad_token_id
        eos_id = tokenizer.eos_id if hasattr(tokenizer, 'eos_id') else tokenizer.eos_token_id
        
        # Prepare negative prompt
        if neg_prompt_ids is None:
            neg_prompt_ids = torch.tensor([speech_begin_id], dtype=torch.long, device=device)
        
        # Get embeddings
        embed_layer = self.model.get_input_embeddings()
        prompt_embeds = embed_layer(prompt_ids)
        neg_prompt_embeds = embed_layer(neg_prompt_ids)
        
        # Process speech inputs if provided
        last_embed = None
        if speech_tensors is not None:
            _, speech_embeds = self._process_speech_inputs(speech_tensors, speech_masks, speech_type)
            
            # Insert speech embeddings into prompt
            prompt_embeds[speech_input_mask] = speech_embeds
        
        # Initialize past key values for positive and negative prompts
        past_key_values = None
        neg_past_key_values = None
        
        # Prefill positive prompt
        with torch.no_grad():
            outputs = self.model.language_model(
                inputs_embeds=prompt_embeds.unsqueeze(0),
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,  # We need hidden states for diffusion conditioning
            )
            past_key_values = outputs.past_key_values
        
        # Prefill negative prompt
        with torch.no_grad():
            neg_outputs = self.model.language_model(
                inputs_embeds=neg_prompt_embeds.unsqueeze(0),
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            neg_past_key_values = neg_outputs.past_key_values
        
        # Initialize generation variables
        if last_embed is None:
            current_embeds = embed_layer(last_ids)
        else:
            current_embeds = last_embed
        
        # Track whether we're generating speech or text
        is_speech_mode = (last_ids == speech_begin_id).any() or (last_ids == pad_id).any()
        speech_token_count = 0 if is_speech_mode else 1
        
        # Output containers
        output_tokens = torch.full((1, max_new_tokens), pad_id, dtype=torch.long, device=device)
        speech_latents = [[]]  # Store latents for each speech segment
        speech_segment_index = 0 if is_speech_mode else -1
        
        # Initialize separate streaming caches for acoustic and semantic tokenizers
        acoustic_cache = VibePodTokenizerStreamingCache()
        semantic_cache = VibePodTokenizerStreamingCache()
        sample_indices = torch.tensor([0], device=device)  # Single sample for now
        
        # Generation loop
        for idx in tqdm(range(max_new_tokens), disable=not verbose):
            # Forward pass for positive prompt
            # Make sure current_embeds has the right shape [batch_size, seq_len, hidden_size]
            if current_embeds.dim() == 2:
                current_embeds = current_embeds.unsqueeze(0)
            
            outputs = self.model.language_model(
                inputs_embeds=current_embeds,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
                output_hidden_states=True,
            )
            hidden_states = outputs.hidden_states[-1] if outputs.hidden_states else outputs[0]
            past_key_values = outputs.past_key_values
            
            # Get logits and sample next token
            logits = self.model.lm_head(hidden_states)
            
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs.squeeze(0), top_p)
            else:
                next_token = torch.argmax(logits, dim=-1).squeeze()
            
            # Check if we should switch modes
            if next_token == speech_end_id:
                is_speech_mode = False
                speech_token_count = 0
                # Clear both streaming caches when speech ends
                acoustic_cache = VibePodTokenizerStreamingCache()
                semantic_cache = VibePodTokenizerStreamingCache()
            elif next_token == speech_begin_id:
                is_speech_mode = True
                speech_token_count = 0
                speech_segment_index += 1
                if len(speech_latents) <= speech_segment_index:
                    speech_latents.append([])
                
                # Reset negative cache for new speech segment
                neg_past_key_values = None
                with torch.no_grad():
                    neg_outputs = self.model.language_model(
                        inputs_embeds=neg_prompt_embeds.unsqueeze(0),
                        use_cache=True,
                        return_dict=True,
                        output_hidden_states=True,
                    )
                    neg_past_key_values = neg_outputs.past_key_values
                
                # Clear both streaming caches for new speech segment
                acoustic_cache = VibePodTokenizerStreamingCache()
                semantic_cache = VibePodTokenizerStreamingCache()
            
            # Handle text generation
            if not is_speech_mode or next_token == speech_begin_id or next_token == speech_end_id:
                output_tokens[0, idx] = next_token
                current_embeds = embed_layer(next_token.unsqueeze(0))
                
                if next_token == eos_id:
                    break
            
            # Handle speech generation
            else:
                # Forward pass for negative prompt
                if current_embeds.dim() == 2:
                    current_embeds_neg = current_embeds.unsqueeze(0)
                else:
                    current_embeds_neg = current_embeds
                    
                neg_outputs = self.model.language_model(
                    inputs_embeds=current_embeds_neg,
                    past_key_values=neg_past_key_values,
                    use_cache=True,
                    return_dict=True,
                    output_hidden_states=True,
                )
                neg_hidden_states = neg_outputs.hidden_states[-1] if neg_outputs.hidden_states else neg_outputs[0]
                neg_past_key_values = neg_outputs.past_key_values
                
                # Sample speech token using diffusion with CFG
                condition = hidden_states.squeeze(0)
                neg_condition = neg_hidden_states.squeeze(0)
                
                speech_latent = self.sample_speech_tokens(
                    condition,
                    neg_condition,
                    cfg_scale=cfg_scale,
                )
                
                speech_latents[speech_segment_index].append(speech_latent)
                
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent.unsqueeze(0) / self.model.speech_scaling_factor - self.model.speech_bias_factor
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent,
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=sample_indices,
                    use_cache=True,
                    debug=False
                ).squeeze(0)
                
                # Encode audio to semantic features using semantic streaming cache
                semantic_output = self.model.semantic_tokenizer.encode(
                    audio_chunk.unsqueeze(0),
                    cache=semantic_cache,  # Use semantic-specific cache
                    sample_indices=sample_indices,
                    use_cache=True,
                    debug=False
                )
                semantic_features = semantic_output.mean.squeeze(0)
                
                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                current_embeds = acoustic_embed + semantic_embed
                
                output_tokens[0, idx] = next_token
                speech_token_count += 1
        
        # Process collected latents
        output_speech = []
        for latent_list in speech_latents:
            if latent_list:
                speech_latent = torch.cat(latent_list, dim=0)
                if not return_latent:
                    # Decode latents to audio (non-streaming for final output)
                    scaled_latents = speech_latent.unsqueeze(0) / self.model.speech_scaling_factor - self.model.speech_bias_factor
                    audio = self.model.acoustic_tokenizer.decode(scaled_latents)
                    output_speech.append(audio.squeeze(0))
                else:
                    output_speech.append(speech_latent)
        
        # Replace padding with EOS
        output_tokens[output_tokens == pad_id] = eos_id
        
        return output_tokens, output_speech


def sample_top_p(probs, p):
    """Sample from top-p (nucleus) distribution."""
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token.squeeze(-1)

__all__ = [
    "VibePodForConditionalGeneration",
]