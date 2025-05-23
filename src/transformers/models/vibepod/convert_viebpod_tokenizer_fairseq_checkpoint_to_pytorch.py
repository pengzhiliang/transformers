#!/usr/bin/env python
# coding=utf-8

import argparse
import json
import os
from pathlib import Path
import re
import torch
from typing import Dict, List, Tuple

from transformers.models.vibepod.configuration_vibepod import VibePodAcousticTokenizerConfig, VibePodSemanticTokenizerConfig
from transformers.models.vibepod.modular_vibepod_tokenizer import VibePodAcousticTokenizerModel, VibePodSemanticTokenizerModel
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

def rename_keys(state_dict: Dict) -> Dict:
    """
    Rename keys in the state dict from fairseq format to HF format
    """
    new_state_dict = {}
    
    for key, value in state_dict.items():
        if key.startswith('generator.'):
            # Remove 'generator.' prefix
            if key == 'generator.fix_std':
                new_key = 'fix_std'  # Keep fix_std at top level
            else:
                new_key = key[len('generator.'):]
            
            new_state_dict[new_key] = value
        else:
            # Keep keys that don't start with 'generator.'
            new_state_dict[key] = value
    
    return new_state_dict

def convert_tokenizer_fairseq_checkpoint_to_hf(
    checkpoint_path: str, 
    pytorch_dump_folder_path: str,
    config_name: str = None,
    config_type: str = 'acoustic',
):
    """
    Convert a fairseq VibePod tokenizer checkpoint to a HuggingFace checkpoint.
    """
    assert config_type in ['acoustic', 'semantic'], "config_type must be either 'acoustic' or 'semantic'"
    # Load the fairseq checkpoint
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Extract the model state dict
    if "model" not in checkpoint:
        raise ValueError("No 'model' key found in checkpoint")
    
    model_state_dict = checkpoint["model"]
    
    # Get the model configuration
    if config_type == 'acoustic':
        logger.info("Using VibePodAcousticTokenizerConfig")
        config = get_vibepod_acoustic_tokenizer_config(checkpoint)

        # Create a new VibePod model
        logger.info("Creating new VibePodAcousticTokenizerModel")
        model = VibePodAcousticTokenizerModel(config)
    elif config_type == 'semantic':
        logger.info("Using VibePodSemanticTokenizerConfig")
        config = get_vibepod_semantic_tokenizer_config(checkpoint)

        # Create a new VibePod model
        logger.info("Creating new VibePodAcousticTokenizerModel")
        model = VibePodSemanticTokenizerModel(config)
    else:
        raise ValueError(f"Unknown config type: {config_type}. Use 'acoustic' or 'semantic'.")
    
    # Rename the keys to match the HF model
    hf_state_dict = rename_keys(model_state_dict)
  
    # Try to load the weights
    missing_keys, unexpected_keys = model.load_state_dict(hf_state_dict, strict=False)
    
    if len(missing_keys) > 0:
        logger.warning(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warning(f"Unexpected keys: {unexpected_keys}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    
    # Save the model state dict and config manually instead of using save_pretrained
    logger.info(f"Saving model to {pytorch_dump_folder_path}")
    
    # Save in safetensors format (preferred)
    try:
        from safetensors.torch import save_file as safe_save
        
        # Convert state dict to CPU if it's not already
        cpu_state_dict = {k: v.cpu() for k, v in model.state_dict().items()}
        
        # Save using safetensors
        safe_save(
            cpu_state_dict, 
            os.path.join(pytorch_dump_folder_path, "model.safetensors")
        )
        logger.info("Model saved in safetensors format")
    except ImportError:
        logger.warning("safetensors not available, falling back to PyTorch format")
        # Fall back to PyTorch format if safetensors is not available
        torch.save(model.state_dict(), os.path.join(pytorch_dump_folder_path, "pytorch_model.bin"))
    
    # Save the config
    config_path = os.path.join(pytorch_dump_folder_path, "config.json")
    with open(config_path, "w") as f:
        f.write(config.to_json_string())
    
    # If specified, save the config with a custom name
    if config_name and config_name != "config":
        custom_config_path = os.path.join(pytorch_dump_folder_path, f"{config_name}.json")
        with open(custom_config_path, "w") as f:
            f.write(config.to_json_string())
    
    logger.info("Conversion complete!")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fairseq_checkpoint_path",
        type=str,
        required=True,
        help="Path to the fairseq checkpoint",
    )
    parser.add_argument(
        "--pytorch_dump_folder_path",
        type=str,
        required=True,
        help="Path to the output PyTorch model directory",
    )
    parser.add_argument(
        "--config_name",
        type=str,
        default=None,
        help="Optional name for the config file (default: config.json)",
    )
    parser.add_argument(
        "--config_type",
        type=str,
        default=None,
        help="Optional name for the config file (default: config.json)",
    )

    args = parser.parse_args()
    
    convert_tokenizer_fairseq_checkpoint_to_hf(
        args.fairseq_checkpoint_path,
        args.pytorch_dump_folder_path,
        args.config_name,
        args.config_type,
    )

if __name__ == "__main__":
    main()

# python src/transformers/models/vibepod/convert_viebpod_tokenizer_fairseq_checkpoint_to_pytorch.py     --fairseq_checkpoint_path /mnt/conversationhub/zhiliang/exp/sptok_vae_nchunk/meta_rms_depth-3-3-3-3-3-3-8_3200x_vae64-std0.5_klw3e-2_stream_pod-em-sp-au-mu-4-2-2-1-1_dbnorm_160s_lr3e-4_g8_300k/checkpoints/clean_checkpoint_1_300000.pt     --pytorch_dump_folder_path /tmp/viebpod_acoustic_tokenizer --config_type acoustic

# python src/transformers/models/vibepod/convert_viebpod_tokenizer_fairseq_checkpoint_to_pytorch.py     --fairseq_checkpoint_path /mnt/conversationhub/zhiliang/exp/sptok_vae_nchunk/meta_rms_depth-3-3-3-3-3-3-8_stream_pod-em-sp-mu-4-2-1-1_vae128_acous-init_onlydist-1_qwen1.5b_dbnorm_1k-tok_lr1e-4_g8_300k/checkpoints/clean_checkpoint_1_300000.pt     --pytorch_dump_folder_path /tmp/viebpod_semantic_tokenizer   --config_type semantic