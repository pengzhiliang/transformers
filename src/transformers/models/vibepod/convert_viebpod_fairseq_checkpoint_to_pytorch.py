#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
from pathlib import Path
import re
import torch
from typing import Dict, List, Tuple
from safetensors.torch import save_file

from transformers.models.vibepod.configuration_vibepod import (
    VibePodConfig, 
    VibePodAcousticTokenizerConfig, 
    VibePodSemanticTokenizerConfig,
    VibePodDiffusionHeadConfig
)
from transformers.models.vibepod.modeling_vibepod import VibePodForConditionalGeneration
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.utils import logging

logging.set_verbosity_info()
logger = logging.get_logger(__name__)

def get_vibepod_acoustic_tokenizer_config(fairseq_dict: Dict) -> VibePodAcousticTokenizerConfig:
    """
    Extract config parameters from fairseq checkpoint and create a VibePodAcousticTokenizerConfig
    """
    # Extract config from checkpoint if possible
    if 'cfg' in fairseq_dict and 'model' in fairseq_dict['cfg']:
        cfg = fairseq_dict['cfg']['model']
    else:
        logger.warning("No configuration found in checkpoint, using default values")
        cfg = {}
    
    # Create config with parameters from checkpoint
    config = VibePodAcousticTokenizerConfig(
        channels=cfg.get('channels', 1),
        vae_dim=cfg.get('vae_dim', 64),  
        fix_std=cfg.get('fix_std', 0.5),  
        causal=cfg.get('causal', True),
        encoder_n_filters=cfg.get('encoder_n_filters', 32),
        encoder_ratios=cfg.get('encoder_ratios', [8, 5, 5, 4, 2, 2]), 
        encoder_depths=cfg.get('depths', "3-3-3-3-3-3-8"),  
        decoder_n_filters=cfg.get('decoder_n_filters', 32),
        decoder_ratios=cfg.get('decoder_ratios', None),
        layernorm=cfg.get('layernorm', 'RMSNorm'), 
        conv_norm=cfg.get('conv_norm', 'none'),
        mixer_layer=cfg.get('mixer_layer', 'depthwise_conv'),
        disable_last_norm=cfg.get('disable_last_norm', True),
        pad_mode=cfg.get('pad_mode', 'constant'),
        weight_init_value=1e-2,
    )
    
    return config

def get_vibepod_semantic_tokenizer_config(fairseq_dict: Dict) -> VibePodSemanticTokenizerConfig:
    """
    Extract config parameters from fairseq checkpoint and create a VibePodSemanticTokenizerConfig
    """
    # Extract config from checkpoint if possible
    if 'cfg' in fairseq_dict and 'model' in fairseq_dict['cfg']:
        cfg = fairseq_dict['cfg']['model']
    else:
        logger.warning("No configuration found in checkpoint, using default values")
        cfg = {}
    
    # Create config with parameters from checkpoint
    config = VibePodSemanticTokenizerConfig(
        channels=cfg.get('channels', 1),
        vae_dim=cfg.get('vae_dim', 128),  
        causal=cfg.get('causal', True),
        encoder_n_filters=cfg.get('encoder_n_filters', 32),
        encoder_ratios=cfg.get('encoder_ratios', [8, 5, 5, 4, 2, 2]),
        encoder_depths=cfg.get('encoder_depths', "3-3-3-3-3-3-8"),
        layernorm=cfg.get('layernorm', 'RMSNorm'), 
        conv_norm=cfg.get('conv_norm', 'none'),
        mixer_layer=cfg.get('mixer_layer', 'depthwise_conv'),
        disable_last_norm=cfg.get('disable_last_norm', True),
        pad_mode=cfg.get('pad_mode', 'constant'),
        weight_init_value=1e-2,
    )
    
    return config

def get_qwen2_config_from_checkpoint(fairseq_dict: Dict) -> Qwen2Config:
    """
    Extract Qwen2 config from fairseq checkpoint
    """
    
    # Count number of layers
    layer_keys = [k for k in fairseq_dict.keys() if re.match(r'backbone\.layers\.\d+\.', k)]
    num_layers = max([int(re.search(r'backbone\.layers\.(\d+)\.', k).group(1)) for k in layer_keys]) + 1
    
    # Get dimensions from weight shapes
    hidden_size = fairseq_dict['backbone.norm.weight'].shape[0]
    
    # load predifined config
    if num_layers == 28 and hidden_size == 1536: # 1.5b model
        predefined_config = {
            "hidden_act": "silu",
            "hidden_size": 1536,
            "initializer_range": 0.02,
            "intermediate_size": 8960,
            "max_position_embeddings": 32768,
            "max_window_layers": 28,
            "model_type": "qwen2",
            "num_attention_heads": 12,
            "num_hidden_layers": 28,
            "num_key_value_heads": 2,
            "rms_norm_eps": 1e-06,
            "rope_theta": 1000000.0,
            "sliding_window": None,
            "tie_word_embeddings": True,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "use_sliding_window": False,
            "vocab_size": 151936,
            "attention_dropout": 0.0,
        }
        config = Qwen2Config.from_dict(predefined_config)
        pretrained_name = "Qwen/Qwen2.5-1.5B"
    else:
        raise NotImplementedError(
            f"Unsupported Qwen2 configuration: {num_layers} layers with hidden size {hidden_size}. "
            "Please provide a valid checkpoint."
        )
                         
    return config, pretrained_name


def get_diffusion_head_config_from_checkpoint(fairseq_dict: Dict, hidden_size: int) -> VibePodDiffusionHeadConfig:
    """
    Extract diffusion head config from checkpoint
    """
    # Count diffusion head layers
    diffusion_layer_keys = [k for k in fairseq_dict.keys() if re.match(r'speech_prediction_head\.layers\.\d+\.', k)]
    num_layers = max([int(re.search(r'speech_prediction_head\.layers\.(\d+)\.', k).group(1)) for k in diffusion_layer_keys]) + 1 if diffusion_layer_keys else 4
    
    # Get latent size from noisy_images_proj
    if 'speech_prediction_head.noisy_images_proj.weight' in fairseq_dict:
        latent_size = fairseq_dict['speech_prediction_head.noisy_images_proj.weight'].shape[1]
    else:
        latent_size = 64  # default
    
    config = VibePodDiffusionHeadConfig(
        hidden_size=hidden_size,
        head_layers=num_layers,
        latent_size=latent_size,
        speech_vae_dim=latent_size,
    )
    
    return config

def rename_fairseq_keys(state_dict: Dict) -> Dict:
    """
    Rename keys from fairseq format to HuggingFace format
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        # Handle speech scaling factors
        if key in ['speech_scaling_factor', 'speech_bias_factor']:
            new_key = f'model.{key}'
        
        # Handle embeddings
        elif key == 'embed_tokens.weight':
            new_key = 'model.language_model.embed_tokens.weight'
        
        # Handle output projection (lm_head)
        elif key == 'output_projection.weight':
            new_key = 'model.lm_head.weight'
        
        # Handle backbone (language model) layers
        elif key.startswith('backbone.'):
            # Replace backbone with model.language_model
            new_key = key.replace('backbone.', 'model.language_model.')
            # DON'T rename layer normalizations - Qwen2 uses the same names as fairseq
            # new_key = new_key.replace('input_layernorm', 'self_attn_layer_norm')
            # new_key = new_key.replace('post_attention_layernorm', 'mlp_layer_norm')
        
        # Handle speech encoder (acoustic tokenizer)
        elif key.startswith('speech_encoder.'):
            if key == 'speech_encoder.fix_std':
                new_key = 'model.acoustic_tokenizer.fix_std'
            else:
                # Remove speech_encoder prefix and add model.acoustic_tokenizer
                new_key = key.replace('speech_encoder.', 'model.acoustic_tokenizer.')
        
        # Handle speech semantic encoder (semantic tokenizer)
        elif key.startswith('speech_semantic_encoder.'):
            new_key = key.replace('speech_semantic_encoder.', 'model.semantic_tokenizer.')
        
        # Handle connectors
        elif key.startswith('speech_connector.'):
            new_key = key.replace('speech_connector.', 'model.acoustic_connector.')
        elif key.startswith('speech_semantic_connector.'):
            new_key = key.replace('speech_semantic_connector.', 'model.semantic_connector.')
        
        # Handle prediction head
        elif key.startswith('speech_prediction_head.'):
            new_key = key.replace('speech_prediction_head.', 'model.prediction_head.')
            # DON'T rename ffn to mlp - keep it as ffn
            # new_key = new_key.replace('.ffn.', '.mlp.')
            # Fix adaLN_modulation path (remove .linear.)
            # new_key = new_key.replace('.adaLN_modulation.', '.adaLN_modulation.linear.')
        
        else:
            logger.warning(f"Unhandled key: {key}")
            continue
        
        new_state_dict[new_key] = value
    
    return new_state_dict

def convert_vibepod_fairseq_checkpoint_to_hf(
    checkpoint_path: str,
    pytorch_dump_folder_path: str,
    config_path: str = None,
):
    """
    Convert a fairseq VibePod checkpoint to HuggingFace format.
    """
    # Load the fairseq checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract the model state dict
    model_state_dict = checkpoint["model"]
    # remove prefix 'decoder.model.' in each key
    model_state_dict = {k.replace('decoder.model.', ''): v for k, v in model_state_dict.items() if k.startswith('decoder.model.')}

    # Get configurations from checkpoint
    logger.info("Extracting configurations from checkpoint")
    
    # Get language model config
    qwen2_config, pretrained_name = get_qwen2_config_from_checkpoint(model_state_dict)
    
    # Get acoustic tokenizer config
    acoustic_config = get_vibepod_acoustic_tokenizer_config(checkpoint)
    
    # Get semantic tokenizer config  
    semantic_config = get_vibepod_semantic_tokenizer_config(checkpoint)
    
    # Get diffusion head config
    diffusion_config = get_diffusion_head_config_from_checkpoint(model_state_dict, qwen2_config.hidden_size)
    
    # Create VibePod config
    config = VibePodConfig(
        acoustic_tokenizer_config=acoustic_config,
        semantic_tokenizer_config=semantic_config,
        decoder_config=qwen2_config,  # Changed from language_model_config to decoder_config
        diffusion_head_config=diffusion_config,
        # Set VAE dimensions based on tokenizer configs
        acostic_vae_dim=acoustic_config.vae_dim,
        semantic_vae_dim=semantic_config.vae_dim,
    )
    
    # Override with provided config if available
    if config_path:
        logger.info(f"Loading config from {config_path}")
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = VibePodConfig.from_dict(config_dict)
    
    # breakpoint()
    # Create the HuggingFace model
    logger.info("Creating HuggingFace VibePodForConditionalGeneration model")
    model = VibePodForConditionalGeneration(config)
    
    # Rename keys to match HF format
    logger.info("Renaming checkpoint keys")
    hf_state_dict = rename_fairseq_keys(model_state_dict)
    
    # Load the state dict
    logger.info("Loading weights into model")
    missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)
    
    if missing_keys:
        logger.warning(f"Missing keys: {missing_keys}")
    if unexpected_keys:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    # Create output directory
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    
    # Save the model and config
    logger.info(f"Saving model to {pytorch_dump_folder_path}")
    
    # Save config
    config.save_pretrained(pytorch_dump_folder_path)
    
    # Save VibePodProcessor configuration
    logger.info("Saving VibePodProcessor configuration")
    processor_config = {
        "processor_class": "VibePodProcessor",
        "speech_tok_compress_ratio": 3200,
        "db_normalize": True,
        # Audio processor configuration
        "audio_processor": {
            "feature_extractor_type": "VibePodTokenizerProcessor",
            "sampling_rate": 24000,
            "normalize_audio": True,
            "target_dB_FS": -25,
            "eps": 1e-6,
        },
        "language_model_pretrained_name": pretrained_name,
    }
    
    processor_config_path = os.path.join(pytorch_dump_folder_path, "preprocessor_config.json")
    with open(processor_config_path, 'w') as f:
        json.dump(processor_config, f, indent=2)
    logger.info(f"Saved processor config to {processor_config_path}")
    
    # Handle tied weights when saving
    # Get state dict and handle tied weights
    state_dict = model.state_dict()
    
    # Check if weights are tied (they should point to the same tensor)
    if 'model.lm_head.weight' in state_dict and 'model.language_model.embed_tokens.weight' in state_dict:
        if state_dict['model.lm_head.weight'].data_ptr() == state_dict['model.language_model.embed_tokens.weight'].data_ptr():
            logger.info("Detected tied weights between lm_head and embed_tokens")
            # Remove the tied weight to avoid duplication
            del state_dict['model.lm_head.weight']
    
    # Save model state dict in safetensors format without metadata
    model_path = os.path.join(pytorch_dump_folder_path, "model.safetensors")
    save_file(state_dict, model_path, metadata={"format": "pt"})
    logger.info(f"Model weights saved to {model_path}")
    
    logger.info("Conversion complete!")
    
    # Verify the saved model can be loaded
    logger.info("Verifying saved model...")
    loaded_model = VibePodForConditionalGeneration.from_pretrained(pytorch_dump_folder_path)
    logger.info("Model successfully loaded from saved checkpoint!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fairseq_checkpoint_path",
        type=str,
        required=True,
        help="Path to the fairseq checkpoint (.pt file)",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path", 
        type=str,
        required=True,
        help="Path to the output PyTorch model directory",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Optional path to a config JSON file to override extracted config",
    )
    
    args = parser.parse_args()
    
    convert_vibepod_fairseq_checkpoint_to_hf(
        args.fairseq_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_path,
    )


if __name__ == "__main__":
    main()

# Example usage:
# python src/transformers/models/vibepod/convert_vibepod_fairseq_checkpoint_to_pytorch.py \
#     --fairseq_checkpoint_path /mnt/conversationhub/zhiliang/exp/sp_mllm/qwen_1.5b_stream_podcast_v2_4096_text-1_ddpm-diff-5_acous-seman-tok3200x64_lr1e-4_bsz4m_8n_100k/ \
#     --pytorch_dump_folder_path /tmp/vibepod-model