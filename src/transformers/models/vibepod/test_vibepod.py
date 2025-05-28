#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
"""
Test script for VibePod model - Podcast TTS generation
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
import torch
import soundfile as sf
import librosa

from transformers.models.vibepod.modeling_vibepod import VibePodForConditionalGeneration
from transformers.models.vibepod.vibepod_processor import VibePodProcessor
from transformers.models.vibepod.vibepod_tokenizer_processor import VibePodTokenizerProcessor, AudioNormalizer
from transformers.models.vibepod.modular_vibepod_text_tokenizer import VibePodTextTokenizer, VibePodTextTokenizerFast

from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VibePod TTS generation test script")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--input_sample",
        type=str,
        required=True,
        help="Path to input file (.json or .txt format)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./tts_results",
        help="Directory to save generated audio files",
    )
    parser.add_argument(
        "--voice_samples",
        type=str,
        nargs="+",
        default=None,
        help="List of audio files to use as voice samples for each speaker",
    )
    parser.add_argument(
        "--cfg_scale",
        type=float,
        nargs="+",
        default=[1.3],
        help="List of cfg_scale values to use for generation",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=None,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--speech_tok_compress_ratio",
        type=int,
        default=3200,
        help="Speech tokenization compression ratio",
    )
    parser.add_argument(
        "--db_normalize",
        action="store_true",
        help="Whether to apply decibel normalization to audio inputs",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run the model on",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bf16",
        choices=["fp32", "fp16", "bf16"],
        help="Data type for model computation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility",
    )
    
    return parser.parse_args()


def set_seed(seed: int):
    """Set random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_audio(audio_path: str, target_sr: int = 24000) -> np.ndarray:
    """Load and resample audio file"""
    wav, sr = sf.read(audio_path)
    if len(wav.shape) > 1:
        wav = np.mean(wav, axis=1)
    if sr != target_sr:
        wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    return wav.astype(np.float32)


def main():
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Set device and dtype
    device = torch.device(args.device)
    if args.dtype == "fp32":
        dtype = torch.float32
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.bfloat16
    
    print(f"Loading model from {args.model_path}")
    print(f"Using device: {device}, dtype: {dtype}")
    
    # Load processor (which handles tokenizer and audio processor)
    processor = VibePodProcessor.from_pretrained(
        args.model_path,
        cache_dir="/mnt/msranlp/zliang/hf_ckpt"  # For Qwen tokenizer caching
    )
    
    # Override speech_tok_compress_ratio if provided
    if args.speech_tok_compress_ratio != processor.speech_tok_compress_ratio:
        print(f"Overriding speech_tok_compress_ratio from {processor.speech_tok_compress_ratio} to {args.speech_tok_compress_ratio}")
        processor.speech_tok_compress_ratio = args.speech_tok_compress_ratio
    
    # Override db_normalize if different
    if args.db_normalize != processor.db_normalize:
        print(f"Overriding db_normalize from {processor.db_normalize} to {args.db_normalize}")
        processor.db_normalize = args.db_normalize
        processor.audio_normalizer = AudioNormalizer() if args.db_normalize else None
    
    # Load model
    model = VibePodForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    
    model.set_ddpm_inference_steps(num_steps=10)

    # Default voice samples if not provided
    default_voice_samples = [
        "/mnt/conversationhub/zhiliang/other/man_voice.wav",
        "/mnt/conversationhub/zhiliang/other/woman_voice.wav",
    ]
    
    voice_samples = args.voice_samples if args.voice_samples else default_voice_samples
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    base_name = Path(args.input_sample).stem

    # Generate for each cfg_scale
    for cfg_scale in args.cfg_scale:
        print(f"\nGenerating with cfg_scale={cfg_scale}", flush=True)
        
        # Process inputs with the simplified interface
        # The processor now handles everything: file loading, parsing, tokenization, and preparation
        inputs = processor(
            text=args.input_sample,  # Can be file path or script content
            voice_samples=voice_samples,
            return_tensors="pt",
            device=device,
            dtype=dtype,
            prepare_for_generation=True,  # This splits input_ids into prompt_ids and last_ids
        )
        
        # Extract prepared inputs
        prompt_ids = inputs["prompt_ids"]
        last_ids = inputs["last_ids"]
        speech_tensors = inputs.get("speech_tensors")
        speech_masks = inputs.get("speech_masks")
        speech_input_mask = inputs.get("speech_input_mask")
        
        # Determine max tokens if not specified
        if args.max_new_tokens is None:
            # Estimate based on script length
            max_seq_len = model.config.decoder_config.max_position_embeddings
            args.max_new_tokens = max_seq_len - len(prompt_ids) - 1
            print(f"Auto-set max_new_tokens to {args.max_new_tokens}")

        # Log parsed script info
        if "parsed_script" in inputs:
            print(f"Processing script with {len(inputs['all_speakers'])} speakers", flush=True)
            script_preview = "\n".join([f"Speaker {sid}: {text}..." for sid, text in inputs["parsed_script"]])
            print(f"Script preview:\n{script_preview}", flush=True)

        # Generate speech
        print("Starting generation...")
        with torch.no_grad():
            # Use semantic-acoustic generation if available
            output_tokens, output_speeches = model.generate_multiple_speeches_semantic_acoustic(
                prompt_ids=prompt_ids,
                last_ids=last_ids,
                tokenizer=processor.tokenizer,
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_input_mask=speech_input_mask,
                cfg_scale=cfg_scale,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                verbose=True,
            )
        
        print(f"Generated {len(output_speeches)} speech segments")
        
        # Save generated audio
        if output_speeches:
            processor.save_audio(
                torch.cat(output_speeches, dim=-1),
                output_path=os.path.join(args.output_dir, f"{base_name}_cfg{cfg_scale}_new.wav"),
            )
        
        else:
            logger.warning("No audio segments were generated")
    
    print("\nGeneration complete!")


if __name__ == "__main__":
    main()

# Example usage:
# python test_vibepod.py \
#     --model_path /tmp/vibepod-model \
#     --input_sample /home/pengzhiliang/speech_qwen/evals/speech/podcast/speech_tokenizer.txt \
#     --output_dir ./tts_results \
#     --cfg_scale 1.2 1.3 \
#     --temperature 0.8 \
#     --db_normalize