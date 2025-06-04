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
from typing import Dict, List, Optional, Tuple, Union, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

from ..auto import AutoModel, AutoModelForCausalLM

from ...activations import ACT2FN
from ...generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from ...modeling_outputs import CausalLMOutput, BaseModelOutputWithPast, ModelOutput
from ...modeling_utils import PreTrainedModel
from ...modeling_flash_attention_utils import FlashAttentionKwargs
from ...utils import LossKwargs, auto_docstring, can_return_tuple, is_torchdynamo_compiling, logging

from ..llama.modeling_llama import LlamaRMSNorm
from ..qwen2.modeling_qwen2 import Qwen2MLP, Qwen2Attention, Qwen2DecoderLayer, Qwen2Model
from .modular_vibepod_tokenizer import VibePodTokenizerStreamingCache, VibePodAcousticTokenizerModel, VibePodSemanticTokenizerModel
from .modular_vibepod_diffusion_head import VibePodDiffusionHead
from .schedule.dpm_solver import DPMSolverMultistepScheduler

from .configuration_vibepod import VibePodConfig

from .modular_vibepod_text_tokenizer import VibePodTextTokenizer, VibePodTextTokenizerFast

logger = logging.get_logger(__name__)

import pdb

@dataclass
class VibePodCausalLMOutputWithPast(BaseModelOutputWithPast):
    logits: Optional[torch.FloatTensor] = None


@dataclass
class VibePodGenerationOutput(ModelOutput):
    """
    Output type for VibePod generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


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
    base_model_prefix = "model"
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
        lm_config = config.decoder_config 
        self.language_model = AutoModel.from_config(lm_config)
        
        # Initialize speech components if needed
        self.acoustic_tokenizer = AutoModel.from_config(config.acoustic_tokenizer_config).to(self.dtype)
        self.semantic_tokenizer = AutoModel.from_config(config.semantic_tokenizer_config).to(self.dtype)

        self.acoustic_connector = SpeechConnector(config.acostic_vae_dim, lm_config.hidden_size).to(self.dtype)
        self.semantic_connector = SpeechConnector(config.semantic_vae_dim, lm_config.hidden_size).to(self.dtype)
        
        # Register scaling factors as buffers
        self.register_buffer('speech_scaling_factor', torch.tensor(1.0, dtype=self.dtype))  
        self.register_buffer('speech_bias_factor', torch.tensor(0.0, dtype=self.dtype))
        
        # Initialize prediction head for speech generation
        self.prediction_head = AutoModel.from_config(config.diffusion_head_config).to(self.dtype)

        # Initialize noise scheduler
        self.noise_scheduler = DPMSolverMultistepScheduler(
            num_train_timesteps=config.diffusion_head_config.ddpm_num_steps,
            beta_schedule=config.diffusion_head_config.ddpm_beta_schedule,
            prediction_type=config.diffusion_head_config.prediction_type
        )
        
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
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Forward through language model
        outputs = self.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        if not return_dict:
            return outputs
            
        return BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


class KwargsForCausalLM(FlashAttentionKwargs, LossKwargs): ...


@auto_docstring(
    custom_intro="""
    VibePod: A multimodal TTS model combining Qwen2.5 LLM with diffusion for speech generation.
    """
)
class VibePodForConditionalGeneration(VibePodPreTrainedModel, GenerationMixin):
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config):
        super().__init__(config)
        
        # Initialize the base model
        self.model = VibePodModel(config)
        
        # LM head for text generation
        self.lm_head = nn.Linear(config.decoder_config.hidden_size, config.decoder_config.vocab_size, bias=False)
        
        # inference configuration
        self.ddpm_inference_steps = config.diffusion_head_config.ddpm_num_inference_steps

        # Initialize weights and apply final processing
        self.post_init()
    
    def tie_weights(self):
        """
        Tie the weights between the input embeddings and the output embeddings.
        """
        # Tie lm_head.weight to language_model.embed_tokens.weight
        if hasattr(self, 'lm_head') and hasattr(self.model.language_model, 'embed_tokens'):
            self.lm_head.weight = self.model.language_model.embed_tokens.weight
        
    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        self.model.set_input_embeddings(value)
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
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
    
    @can_return_tuple
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
        cache_position: Optional[torch.LongTensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        logits_to_keep: Union[int, slice] = 0,
        **kwargs: KwargsForCausalLM,
    ) -> Union[Tuple, VibePodCausalLMOutputWithPast]:
        """
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.
            speech_tensors (`torch.FloatTensor`, *optional*):
                Input speech waveforms for voice cloning or speech understanding.
            speech_masks (`torch.BoolTensor`, *optional*):
                Masks indicating valid speech frames.
            speech_input_mask (`torch.BoolTensor`, *optional*):
                Positions in the input sequence where speech embeddings should be inserted.
        
        Returns:
            `VibePodCausalLMOutputWithPast` or tuple
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # Get embeddings
        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
        
        # Process speech inputs if provided
        if speech_tensors is not None and speech_masks is not None:
            acoustic_features, speech_embeds = self._process_speech_inputs(speech_tensors.to(self.dtype), speech_masks)
            if speech_input_mask is not None:
                inputs_embeds[speech_input_mask] = speech_embeds
        
        # Forward through language model
        outputs = self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )
        
        hidden_states = outputs[0] if not return_dict else outputs.last_hidden_state
        # logits = self.lm_head(hidden_states)
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
                
        if labels is not None:
            raise NotImplementedError("Loss computation is not implemented in this version.")

        return VibePodCausalLMOutputWithPast(
            logits=logits,
            past_key_values=outputs.past_key_values,
            last_hidden_state=hidden_states,
            attentions=outputs.attentions,
        )

    def _build_generate_config_model_kwargs(self, inputs, tokenizer, **kwargs):
        generation_config = GenerationConfig()
        generation_config.bos_token_id = tokenizer.bos_token_id 
        generation_config.eos_token_id = tokenizer.eos_token_id
        generation_config.pad_token_id = tokenizer.pad_token_id
        generation_config, model_kwargs = self._prepare_generation_config(generation_config, True, speech_start_id = tokenizer.speech_start_id, speech_end_id = tokenizer.speech_end_id, speech_diffusion_id = tokenizer.speech_diffusion_id, **kwargs)
        generation_config.speech_start_id = tokenizer.speech_start_id
        generation_config.speech_end_id = tokenizer.speech_end_id
        generation_config.speech_diffusion_id = tokenizer.speech_diffusion_id

        inputs_tensor, model_input_name, model_kwargs = self._prepare_model_inputs(inputs, generation_config.bos_token_id, model_kwargs)
        batch_size = inputs_tensor.shape[0]
        device = self.device
        
        self._prepare_special_tokens(generation_config, True, device=device)
        generation_config.use_cache = True
        model_kwargs["use_cache"] = generation_config.use_cache
        input_ids = inputs_tensor.to(self.device)

        input_ids_length = input_ids.shape[1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        has_default_min_length = kwargs.get("min_length") is None and generation_config.min_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            has_default_min_length=has_default_min_length,
            model_input_name=model_input_name,
            inputs_tensor=inputs_tensor,
            input_ids_length=input_ids_length,
        )

        max_cache_length = generation_config.max_length - 1
        self._prepare_cache_for_generation(generation_config, model_kwargs, None, batch_size, max_cache_length, device)
        model_kwargs['cache_position'] = torch.arange(input_ids_length, device=device, dtype=torch.long)
        for k, v in model_kwargs.items():
            if isinstance(v, torch.Tensor):
                model_kwargs[k] = v.to(device=device)
        
        return generation_config, model_kwargs, input_ids

    @torch.no_grad()
    def generate_negative_with_start_end_token(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> Union[torch.LongTensor, VibePodGenerationOutput]:
        """
        Generates sequences of token ids and optionally speech outputs.
        
        Args:
            All standard generation arguments from GenerationMixin
            negative_prompt_ids: Negative prompt for CFG in speech generation
            negative_prompt_attention_mask: Attention mask for negative prompt
            speech_tensors: Input speech for voice cloning
            speech_masks: Masks for speech tensors  
            speech_input_mask: Positions to insert speech embeddings
            return_speech: Whether to decode and return speech outputs
            cfg_scale: CFG scale for speech generation
 
        Returns:
            Generated token sequences and optionally speech outputs
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

        generation_config, model_kwargs, input_ids = self._build_generate_config_model_kwargs(
            inputs, tokenizer, **kwargs
        )
        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, tokenizer, **negative_kwargs
        )

        logits_processor = LogitsProcessorList()

        acoustic_cache = VibePodTokenizerStreamingCache()
        semantic_cache = VibePodTokenizerStreamingCache()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]

        while (~finished_tags).any():
            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopping early.")
                break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                # we process the speech inputs only during the first generation step
                prefill_inputs = {
                    "speech_tensors": speech_tensors.to(device=device),
                    "speech_masks": speech_masks.to(device),
                    "speech_input_mask": speech_input_mask.to(device),
                }
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}
            # Forward pass through the model
            outputs = self(
                **model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            # Get logits and apply logits processor
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # reached end of generation
            if generation_config.eos_token_id is not None and (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                # Only print for samples that are newly finished (not already marked as finished)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    print(f"Reached EOS at indices: {new_eos_indices.tolist()}")
                    finished_tags[new_eos_indices] = True
                    
            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                # Clear tokenizer caches for samples that reached speech end
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)
            
            # speech_begin
            diffusion_start_indices = (next_tokens == generation_config.speech_start_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_start_indices.numel() > 0:
                # No special handling needed for start tokens
                # pdb.set_trace()
                # maybe clear the negative cache here? 
                pass
            
            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]
            

            # forward diffusion
            diffusion_indices = (next_tokens == generation_config.speech_diffusion_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_indices.numel() > 0:
                # pdb.set_trace()
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                negative_outputs = self(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)
                                
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor - self.model.speech_bias_factor
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent,
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                )
                
                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                # Encode audio to semantic features using semantic streaming cache
                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,  # Use semantic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                ).mean
                
                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds
            
            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds
            
        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)

        # pdb.set_trace()
        return VibePodGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None
        )
    
    @torch.no_grad()
    def generate_negative_without_start_end_token(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> Union[torch.LongTensor, VibePodGenerationOutput]:
        """
        Generates sequences of token ids and optionally speech outputs.
        
        Args:
            All standard generation arguments from GenerationMixin
            negative_prompt_ids: Negative prompt for CFG in speech generation
            negative_prompt_attention_mask: Attention mask for negative prompt
            speech_tensors: Input speech for voice cloning
            speech_masks: Masks for speech tensors  
            speech_input_mask: Positions to insert speech embeddings
            return_speech: Whether to decode and return speech outputs
            cfg_scale: CFG scale for speech generation
 
        Returns:
            Generated token sequences and optionally speech outputs
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

        generation_config, model_kwargs, input_ids = self._build_generate_config_model_kwargs(
            inputs, tokenizer, **kwargs
        )
        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, tokenizer, **negative_kwargs
        )

        logits_processor = LogitsProcessorList()

        acoustic_cache = VibePodTokenizerStreamingCache()
        semantic_cache = VibePodTokenizerStreamingCache()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]

        while (~finished_tags).any():
            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopping early.")
                break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                # we process the speech inputs only during the first generation step
                prefill_inputs = {
                    "speech_tensors": speech_tensors.to(device=device),
                    "speech_masks": speech_masks.to(device),
                    "speech_input_mask": speech_input_mask.to(device),
                }
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}
            # Forward pass through the model
            outputs = self(
                **model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            # Get logits and apply logits processor
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # reached end of generation
            if generation_config.eos_token_id is not None and (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                # Only print for samples that are newly finished (not already marked as finished)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    print(f"Reached EOS at indices: {new_eos_indices.tolist()}")
                    finished_tags[new_eos_indices] = True
                    
            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                # Clear tokenizer caches for samples that reached speech end
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)
            
            # speech_begin
            diffusion_start_indices = (next_tokens == generation_config.speech_start_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_start_indices.numel() > 0:
                # No special handling needed for start tokens
                # pdb.set_trace()
                # maybe clear the negative cache here? 
                pass
            
            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # forward diffusion
            diffusion_indices = (next_tokens == generation_config.speech_diffusion_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_indices.numel() > 0:
                # pdb.set_trace()
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                negative_outputs = self(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # correct the non-diffusion indices
                # we forward add samples' negative outputs even if 
                #   they are not in diffusion mode to keep the cache consistent
                # So we need to correct the kv cache of non-diffusion samples
                if diffusion_indices.numel() < batch_size - finished_tags.sum():
                    # not all samples are in diffusion mode, but we calculate all negative outputs
                    non_diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens != generation_config.speech_diffusion_id)]
                    start_indices = correct_cnt[non_diffusion_indices]
                    print(f"Non-diffusion indices: {non_diffusion_indices.tolist()}, Start indices: {start_indices.tolist()}")

                    # 1. Update attention_mask - need to handle each sample separately
                    seq_len = negative_model_kwargs['attention_mask'].shape[1]
                    for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                        # Shift the attention mask for this sample
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = \
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                    # 2. Update past_key_values
                    for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                        negative_model_kwargs['past_key_values'].value_cache)):
                        # Process each non-diffusion sample
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < k_cache.shape[2] - 1:
                                # Shift cache for this sample
                                k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                    
                    # 3. Update negative_input_ids
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx+1:] = \
                                negative_input_ids[sample_idx, start_idx:-1].clone()
                                
                    correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)
                                
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor - self.model.speech_bias_factor
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent,
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                )
                
                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                # Encode audio to semantic features using semantic streaming cache
                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,  # Use semantic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                ).mean
                
                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds
            
            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds
            
        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)

        # pdb.set_trace()
        return VibePodGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None
        )
    
    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        logits_processor: Optional[LogitsProcessorList] = None,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        prefix_allowed_tokens_fn: Optional[Callable[[int, torch.Tensor], List[int]]] = None,
        synced_gpus: Optional[bool] = None,
        assistant_model: Optional["PreTrainedModel"] = None,
        streamer: Optional["BaseStreamer"] = None,
        negative_prompt_ids: Optional[torch.Tensor] = None,
        negative_prompt_attention_mask: Optional[torch.Tensor] = None,
        speech_tensors: Optional[torch.FloatTensor] = None,
        speech_masks: Optional[torch.BoolTensor] = None,
        speech_input_mask: Optional[torch.BoolTensor] = None,
        return_speech: bool = True,
        cfg_scale: float = 1.0,
        do_sample: bool = False,
        **kwargs,
    ) -> Union[torch.LongTensor, VibePodGenerationOutput]:
        """
        Generates sequences of token ids and optionally speech outputs.
        
        Args:
            All standard generation arguments from GenerationMixin
            negative_prompt_ids: Negative prompt for CFG in speech generation
            negative_prompt_attention_mask: Attention mask for negative prompt
            speech_tensors: Input speech for voice cloning
            speech_masks: Masks for speech tensors  
            speech_input_mask: Positions to insert speech embeddings
            return_speech: Whether to decode and return speech outputs
            cfg_scale: CFG scale for speech generation
 
        Returns:
            Generated token sequences and optionally speech outputs
        """
        # 1. Handle `generation_config` and kwargs that might update it, and validate the `.generate()` call
        tokenizer = kwargs.pop("tokenizer", None)  # Pull this out first, we only use it for stopping criteria
        parsed_scripts = kwargs.pop("parsed_scripts", None)
        all_speakers_list = kwargs.pop("all_speakers_list", None)
        if kwargs.get('max_new_tokens', None) is None:
            kwargs['max_new_tokens'] = self.config.decoder_config.max_position_embeddings - kwargs['input_ids'].shape[-1]

        generation_config, model_kwargs, input_ids = self._build_generate_config_model_kwargs(
            inputs, tokenizer, **kwargs
        )
        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self._build_generate_config_model_kwargs(
            None, tokenizer, **negative_kwargs
        )

        logits_processor = LogitsProcessorList()

        acoustic_cache = VibePodTokenizerStreamingCache()
        semantic_cache = VibePodTokenizerStreamingCache()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]

        while (~finished_tags).any():
            if input_ids.shape[-1] >= generation_config.max_length:
                print(f"Reached maximum generation length {generation_config.max_length}, stopping early.")
                break
            model_inputs = self.prepare_inputs_for_generation(input_ids, **model_kwargs)
            if is_prefill:
                # we process the speech inputs only during the first generation step
                prefill_inputs = {
                    "speech_tensors": speech_tensors.to(device=device),
                    "speech_masks": speech_masks.to(device),
                    "speech_input_mask": speech_input_mask.to(device),
                }
                is_prefill = False
            else:
                _ = model_inputs.pop('inputs_embeds', None)
                prefill_inputs = {'inputs_embeds': inputs_embeds}
            # Forward pass through the model
            outputs = self(
                **model_inputs, **prefill_inputs, logits_to_keep=1, return_dict=True, output_attentions=False, output_hidden_states=False,
            )
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=False,
            )

            # Get logits and apply logits processor
            next_token_logits = outputs.logits[:, -1, :].to(copy=True, dtype=torch.float32, device=input_ids.device)
            next_token_scores = logits_processor(input_ids, next_token_logits)
            # token selection
            if do_sample:
                probs = nn.functional.softmax(next_token_scores, dim=-1)
                # TODO (joao): this OP throws "skipping cudagraphs due to ['incompatible ops']", find solution
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(next_token_scores, dim=-1)

            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            # reached end of generation
            if generation_config.eos_token_id is not None and (next_tokens == generation_config.eos_token_id).any():
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                # Only print for samples that are newly finished (not already marked as finished)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    print(f"Reached EOS at indices: {new_eos_indices.tolist()}")
                    finished_tags[new_eos_indices] = True
                    
            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                # Clear tokenizer caches for samples that reached speech end
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)
            
            # speech_begin
            diffusion_start_indices = (next_tokens == generation_config.speech_start_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_start_indices.numel() > 0:
                # No special handling needed for start tokens
                # pdb.set_trace()
                # maybe clear the negative cache here? 
                pass
            
            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.model.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            # # pdb.set_trace()
            # negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
            # # Forward negative pass through the model
            # if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
            #     negative_model_inputs['inputs_embeds'] = inputs_embeds
            #     negative_model_inputs['input_ids'] = None

            # negative_outputs = self(
            #     **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
            # )
            # negative_model_kwargs = self._update_model_kwargs_for_generation(
            #     negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
            # )
            # negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)
            
            # forward diffusion
            diffusion_indices = (next_tokens == generation_config.speech_diffusion_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_indices.numel() > 0:
                # pdb.set_trace()
                negative_model_inputs = self.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                negative_outputs = self(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)

                # correct the non-diffusion indices
                # we forward add samples' negative outputs even if 
                #   they are not in diffusion mode to keep the cache consistent
                # So we need to correct the kv cache of non-diffusion samples
                # if diffusion_indices.numel() < batch_size - finished_tags.sum():
                #     pdb.set_trace()
                #     # not all samples are in diffusion mode, but we calculate all negative outputs
                #     non_diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens != generation_config.speech_diffusion_id)]
                #     start_indices = correct_cnt[non_diffusion_indices]
                #     print(f"Non-diffusion indices: {non_diffusion_indices.tolist()}, Start indices: {start_indices.tolist()}")

                #     # 1. Update attention_mask - need to handle each sample separately
                #     seq_len = negative_model_kwargs['attention_mask'].shape[1]
                #     for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                #         # Shift the attention mask for this sample
                #         if start_idx + 1 < seq_len - 1:
                #             negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = \
                #                 negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                #         negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                #     # 2. Update past_key_values
                #     for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                #                                                         negative_model_kwargs['past_key_values'].value_cache)):
                #         # Process each non-diffusion sample
                #         for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                #             if start_idx + 1 < k_cache.shape[2] - 1:
                #                 # Shift cache for this sample
                #                 k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                #                 v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                    
                #     # 3. Update negative_input_ids
                #     for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                #         if start_idx + 1 < negative_input_ids.shape[1] - 1:
                #             negative_input_ids[sample_idx, start_idx+1:] = \
                #                 negative_input_ids[sample_idx, start_idx:-1].clone()
                                
                #     correct_cnt[non_diffusion_indices] += 1

                positive_condition = outputs.last_hidden_state[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                
                speech_latent = self.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)
                                
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model.speech_scaling_factor - self.model.speech_bias_factor
                audio_chunk = self.model.acoustic_tokenizer.decode(
                    scaled_latent,
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                )
                
                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                # Encode audio to semantic features using semantic streaming cache
                semantic_features = self.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,  # Use semantic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                ).mean
                
                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model.acoustic_connector(speech_latent)
                semantic_embed = self.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds
            
            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds
            
        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)

        # pdb.set_trace()
        return VibePodGenerationOutput(
            sequences=input_ids,
            speech_outputs=final_audio_outputs if return_speech else None
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
    
    
    def _extract_and_decode_speech(self, token_ids):
        """Extract speech segments from generated tokens and decode them."""
        # This would parse token_ids for speech segments and decode them
        # Placeholder implementation
        return []

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
            logits = self.lm_head(hidden_states)
            
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

AutoModel.register(VibePodConfig, VibePodModel)
AutoModelForCausalLM.register(VibePodConfig, VibePodForConditionalGeneration)

__all__ = [
    "VibePodModel",
    "VibePodForConditionalGeneration",
    # "VibePodCausalLMOutputWithPast",
    "VibePodGenerationOutput",
]