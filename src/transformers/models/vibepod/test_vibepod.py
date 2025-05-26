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


def convert_json_to_script(json_file: str) -> Tuple[str, List[int]]:
    """
    Convert JSON format to script format.
    Expected JSON format:
    [
        {"speaker": "1", "text": "Hello everyone..."},
        {"speaker": "2", "text": "Great to be here..."}
    ]
    """
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    all_speakers = set()
    script_lines = []
    
    for item in data:
        speaker = item.get('speaker')
        text = item.get('text')
        if speaker and text:
            all_speakers.add(int(speaker))
            script_lines.append(f"Speaker {speaker}: {text}")
    
    return "\n".join(script_lines), sorted(list(all_speakers))


def convert_text_to_script(text_file: str) -> Tuple[str, List[int]]:
    """
    Convert text file to script format.
    Supports:
    1. Already formatted as "Speaker X: text"
    2. Plain text (assigns to Speaker 1)
    """
    with open(text_file, 'r') as f:
        lines = f.readlines()
    
    all_speakers = set()
    script_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        if line.startswith("Speaker"):
            try:
                speaker_part, text = line.split(":", 1)
                speaker_id = int(speaker_part.split()[1])
                all_speakers.add(speaker_id)
                script_lines.append(f"Speaker {speaker_id}: {text.strip()}")
            except:
                # If parsing fails, treat as plain text
                all_speakers.add(1)
                script_lines.append(f"Speaker 1: {line}")
        else:
            # Plain text - assign to Speaker 1
            all_speakers.add(1)
            script_lines.append(f"Speaker 1: {line}")
    
    return "\n".join(script_lines), sorted(list(all_speakers))


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
    
    logger.info(f"Loading model from {args.model_path}")
    logger.info(f"Using device: {device}, dtype: {dtype}")
    
    # Load processor (which handles tokenizer and audio processor)
    processor = VibePodProcessor.from_pretrained(
        args.model_path,
        cache_dir="/mnt/msranlp/zliang/hf_ckpt"  # For Qwen tokenizer caching
    )
    
    # Override speech_tok_compress_ratio if provided
    if args.speech_tok_compress_ratio != processor.speech_tok_compress_ratio:
        logger.info(f"Overriding speech_tok_compress_ratio from {processor.speech_tok_compress_ratio} to {args.speech_tok_compress_ratio}")
        processor.speech_tok_compress_ratio = args.speech_tok_compress_ratio
    
    # Override db_normalize if different
    if args.db_normalize != processor.db_normalize:
        logger.info(f"Overriding db_normalize from {processor.db_normalize} to {args.db_normalize}")
        processor.db_normalize = args.db_normalize
        processor.audio_normalizer = AudioNormalizer() if args.db_normalize else None
    
    # Load model
    model = VibePodForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map=device,
    )
    model.eval()
    
    # Process input file
    if args.input_sample.endswith('.json'):
        script, speaker_ids = convert_json_to_script(args.input_sample)
        base_name = Path(args.input_sample).stem
    elif args.input_sample.endswith('.txt'):
        script, speaker_ids = convert_text_to_script(args.input_sample)
        base_name = Path(args.input_sample).stem
    else:
        raise ValueError("Input file must be .json or .txt format")
    
    logger.info(f"Processing script with {len(speaker_ids)} speakers")
    logger.info(f"Script preview:\n{script[:500]}...")
    
    # Default voice samples if not provided
    default_voice_samples = [
        "/mnt/conversationhub/zhiliang/other/man_voice.wav",
        "/mnt/conversationhub/zhiliang/other/woman_voice.wav",
    ]
    
    voice_samples = args.voice_samples if args.voice_samples else default_voice_samples
    
    # Check if we have enough voice samples
    if len(voice_samples) < len(speaker_ids):
        logger.warning(f"Not enough voice samples. Need {len(speaker_ids)}, got {len(voice_samples)}")
        # Repeat the last sample if needed
        while len(voice_samples) < len(speaker_ids):
            voice_samples.append(voice_samples[-1])
    
    # Only use needed voice samples
    voice_samples = voice_samples[:len(speaker_ids)]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # breakpoint()

    # Generate for each cfg_scale
    for cfg_scale in args.cfg_scale:
        logger.info(f"\nGenerating with cfg_scale={cfg_scale}")
        
        # Process the script with processor
        inputs = processor.process_podcast_script(
            script=script,
            speaker_samples=voice_samples,
            return_tensors="pt"
        )
        
        # Move inputs to device
        input_ids = torch.tensor(inputs["input_ids"], device=device)
        speech_input_mask = torch.tensor(inputs["speech_input_mask"], device=device, dtype=torch.bool)
        
        # Prepare speech inputs if available
        if inputs["speech_inputs"]:
            speech_dict = processor.prepare_speech_inputs(
                inputs["speech_inputs"],
                return_tensors="pt",
                device=device,
                dtype=dtype
            )
            speech_tensors = speech_dict["padded_speeches"]
            speech_masks = speech_dict["speech_masks"]
        else:
            speech_tensors = None
            speech_masks = None
        
        # Split input_ids for generation
        prompt_ids = input_ids[:-1]
        last_ids = input_ids[-1].unsqueeze(0)
        
        # Determine max tokens if not specified
        if args.max_new_tokens is None:
            # Estimate based on script length
            max_seq_len = model.config.decoder_config.max_position_embeddings
            args.max_new_tokens = max_seq_len - len(input_ids)
            logger.info(f"Auto-set max_new_tokens to {args.max_new_tokens}")
        
        # Generate speech
        logger.info("Starting generation...")
        
        with torch.no_grad():
            # Use semantic-acoustic generation if available
            output_tokens, output_speeches = model.generate_multiple_speeches_semantic_acoustic(
                prompt_ids=prompt_ids,
                last_ids=last_ids,
                tokenizer=processor.tokenizer,
                speech_tensors=speech_tensors,
                speech_masks=speech_masks,
                speech_input_mask=speech_input_mask[:-1],  # Exclude last token
                cfg_scale=cfg_scale,
                temperature=args.temperature,
                max_new_tokens=args.max_new_tokens,
                verbose=True,
            )
            
        
        logger.info(f"Generated {len(output_speeches)} speech segments")
        
        # Save generated audio
        if output_speeches:
            # Concatenate all audio segments
            full_audio_list = []
            for audio in output_speeches:
                if isinstance(audio, torch.Tensor):
                    full_audio_list.append(audio.detach().float().cpu().numpy())
                else:
                    full_audio_list.append(audio)
            
            # Concatenate along time dimension
            full_audio = np.concatenate(full_audio_list, axis=-1)
            full_audio = np.clip(full_audio, -1, 1)
            
            # Save audio file
            os.makedirs(args.output_dir, exist_ok=True)
            output_filename = f"{base_name}_cfg{cfg_scale}.wav"
            output_path = os.path.join(args.output_dir, output_filename)
            
            sf.write(output_path, full_audio.squeeze(), 24000)
            logger.info(f"Saved audio to {output_path}")
            
            # # Also save the generated text tokens
            # generated_text = processor.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
            # text_output_path = output_path.replace('.wav', '.txt')
            # with open(text_output_path, 'w') as f:
            #     f.write(f"Script:\n{script}\n\n")
            #     f.write(f"Generated tokens:\n{generated_text}\n")
            # logger.info(f"Saved generated text to {text_output_path}")
        else:
            logger.warning("No audio segments were generated")
    
    logger.info("\nGeneration complete!")


if __name__ == "__main__":
    main()

# Example usage:
# python test_vibepod.py \
#     --model_path /tmp/vibepod-model \
#     --input_sample speech_tokenizer.txt \
#     --output_dir ./tts_results \
#     --cfg_scale 2.0 3.0 \
#     --temperature 0.8 \
#     --db_normalize