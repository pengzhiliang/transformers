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
Test script for VibePod Processor - Batch Processing Test
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Union, Dict, Any

import numpy as np
import torch
import soundfile as sf
import librosa

from transformers.models.vibepod.vibepod_processor import VibePodProcessor
from transformers.models.vibepod.vibepod_tokenizer_processor import AudioNormalizer
from transformers.tokenization_utils_base import PaddingStrategy
from transformers.utils import logging


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="VibePod Processor Batch Test")
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--test_single",
        action="store_true",
        help="Test single input processing",
    )
    parser.add_argument(
        "--test_batch",
        action="store_true", 
        help="Test batch input processing",
    )
    parser.add_argument(
        "--test_padding",
        action="store_true",
        help="Test different padding strategies",
    )
    parser.add_argument(
        "--test_truncation",
        action="store_true",
        help="Test truncation functionality",
    )
    parser.add_argument(
        "--test_tensors",
        action="store_true",
        help="Test different tensor return types",
    )
    parser.add_argument(
        "--voice_samples_dir",
        type=str,
        default=None,
        help="Directory containing voice samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for tensor tests",
    )
    
    return parser.parse_args()


def create_test_scripts() -> Dict[str, Any]:
    """Create test scripts for different scenarios"""
    
    # Single speaker script
    single_speaker_script = """Speaker 1: Hello everyone, welcome to our podcast.
Speaker 1: Today we'll be discussing artificial intelligence.
Speaker 1: Let's dive right in."""
    
    # Multi-speaker script
    multi_speaker_script = """Speaker 1: Welcome to Tech Talk podcast!
Speaker 2: Thanks for having me, I'm excited to be here.
Speaker 1: Today we're discussing the future of AI.
Speaker 2: It's such a fascinating topic with so many implications."""
    
    # Long script for truncation testing
    long_script = "\n".join([
        f"Speaker {i % 2 + 1}: This is sentence number {i} in our very long conversation about various topics."
        for i in range(50)
    ])
    
    # JSON format test
    json_data = [
        {"speaker": "1", "text": "Hello from JSON format"},
        {"speaker": "2", "text": "This tests JSON parsing"},
        {"speaker": "1", "text": "Great to test different formats"},
    ]
    
    return {
        "single_speaker": single_speaker_script,
        "multi_speaker": multi_speaker_script,
        "long_script": long_script,
        "json_data": json_data,
    }


def test_single_input(processor: VibePodProcessor, scripts: Dict[str, Any], voice_samples: List[str] = None):
    """Test single input processing"""
    print("\n" + "="*50)
    print("Testing Single Input Processing")
    print("="*50)
    
    # Test with string input
    print("\n1. Testing with direct script string:")
    result = processor(
        text=scripts["single_speaker"],
        voice_samples=voice_samples[:1] if voice_samples else None,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    print(f"   - Input IDs shape: {result['input_ids'].shape}")
    print(f"   - Attention mask shape: {result['attention_mask'].shape}")
    print(f"   - Speech input mask shape: {result['speech_input_mask'].shape}")
    if result.get('speech_tensors') is not None:
        print(f"   - Speech tensors shape: {result['speech_tensors'].shape}")
        print(f"   - Speech masks shape: {result['speech_masks'].shape}")
    
    # Test with multi-speaker script
    print("\n2. Testing with multi-speaker script:")
    result = processor(
        text=scripts["multi_speaker"],
        voice_samples=voice_samples[:2] if voice_samples else None,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    print(f"   - Input IDs shape: {result['input_ids'].shape}")
    print(f"   - Number of speakers: {len(result['all_speakers_list'][0])}")
    
    return True


def test_batch_input(processor: VibePodProcessor, scripts: Dict[str, Any], voice_samples: List[str] = None):
    """Test batch input processing"""
    print("\n" + "="*50)
    print("Testing Batch Input Processing")
    print("="*50)
    
    # Create batch of scripts
    batch_texts = [
        scripts["single_speaker"],
        scripts["multi_speaker"],
        "Speaker 1: Short script for testing",
    ]
    
    # Create corresponding voice samples
    batch_voice_samples = None
    if voice_samples:
        batch_voice_samples = [
            voice_samples[:1],  # 1 speaker
            voice_samples[:2],  # 2 speakers
            voice_samples[:1],  # 1 speaker
        ]
    
    print("\n1. Testing batch processing with padding:")
    result = processor(
        text=batch_texts,
        voice_samples=batch_voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    print(f"   - Batch size: {result['input_ids'].shape[0]}")
    print(f"   - Max sequence length: {result['input_ids'].shape[1]}")
    print(f"   - Attention mask shape: {result['attention_mask'].shape}")
    
    # Check padding
    print("\n2. Checking padding correctness:")
    for i in range(len(batch_texts)):
        non_pad_tokens = (result['attention_mask'][i] == 1).sum().item()
        print(f"   - Sample {i}: {non_pad_tokens} non-padded tokens")
    
    # Test without padding
    print("\n3. Testing batch without padding:")
    result_no_pad = processor(
        text=batch_texts,
        voice_samples=batch_voice_samples,
        padding=False,
        return_attention_mask=True,
    )
    
    print(f"   - Result is list: {isinstance(result_no_pad['input_ids'], list)}")
    print(f"   - Sequence lengths: {[len(seq) for seq in result_no_pad['input_ids']]}")
    
    return True


def test_padding_strategies(processor: VibePodProcessor, scripts: Dict[str, Any]):
    """Test different padding strategies"""
    print("\n" + "="*50)
    print("Testing Padding Strategies")
    print("="*50)
    
    batch_texts = [
        "Speaker 1: Short",
        "Speaker 1: Medium length script with more content",
        scripts["multi_speaker"],
    ]
    
    # Test longest padding
    print("\n1. Testing 'longest' padding:")
    result = processor(
        text=batch_texts,
        padding="longest",
        return_tensors="pt",
    )
    print(f"   - Padded to length: {result['input_ids'].shape[1]}")
    
    # Test max_length padding
    print("\n2. Testing 'max_length' padding:")
    max_len = 100
    result = processor(
        text=batch_texts,
        padding="max_length",
        max_length=max_len,
        return_tensors="pt",
    )
    print(f"   - Padded to length: {result['input_ids'].shape[1]} (requested: {max_len})")
    
    # Test no padding
    print("\n3. Testing no padding:")
    result = processor(
        text=batch_texts,
        padding=False,
    )
    print(f"   - Sequence lengths: {[len(seq) for seq in result['input_ids']]}")
    
    return True


def test_truncation(processor: VibePodProcessor, scripts: Dict[str, Any]):
    """Test truncation functionality"""
    print("\n" + "="*50)
    print("Testing Truncation")
    print("="*50)
    
    # Use long script
    long_text = scripts["long_script"]
    
    # Process without truncation
    print("\n1. Without truncation:")
    result_full = processor(
        text=long_text,
        truncation=False,
        return_tensors="pt",
    )
    full_length = result_full['input_ids'].shape[1]
    print(f"   - Full sequence length: {full_length}")
    
    # Process with truncation
    max_len = 200
    print(f"\n2. With truncation (max_length={max_len}):")
    result_trunc = processor(
        text=long_text,
        truncation=True,
        max_length=max_len,
        padding="max_length",
        return_tensors="pt",
    )
    print(f"   - Truncated sequence length: {result_trunc['input_ids'].shape[1]}")
    
    # Test batch truncation
    print("\n3. Batch truncation:")
    batch_texts = [scripts["long_script"], "Speaker 1: Short text"]
    result_batch = processor(
        text=batch_texts,
        truncation=True,
        max_length=150,
        padding=True,
        return_tensors="pt",
    )
    print(f"   - Batch shape: {result_batch['input_ids'].shape}")
    
    return True


def test_tensor_types(processor: VibePodProcessor, scripts: Dict[str, Any]):
    """Test different tensor return types"""
    print("\n" + "="*50)
    print("Testing Tensor Return Types")
    print("="*50)
    
    test_text = scripts["single_speaker"]
    
    # PyTorch tensors
    print("\n1. PyTorch tensors:")
    result_pt = processor(
        text=test_text,
        return_tensors="pt",
    )
    print(f"   - Type: {type(result_pt['input_ids'])}")
    print(f"   - Device: {result_pt['input_ids'].device}")
    
    # NumPy arrays
    print("\n2. NumPy arrays:")
    result_np = processor(
        text=test_text,
        return_tensors="np",
    )
    print(f"   - Type: {type(result_np['input_ids'])}")
    print(f"   - Dtype: {result_np['input_ids'].dtype}")
    
    # Python lists
    print("\n3. Python lists (no return_tensors):")
    result_list = processor(
        text=test_text,
        return_tensors=None,
    )
    print(f"   - Type: {type(result_list['input_ids'])}")
    
    # Test with device placement (PyTorch only)
    if torch.cuda.is_available():
        print("\n4. Testing device placement:")
        device = torch.device("cuda:0")
        result_cuda = processor(
            text=test_text,
            return_tensors="pt",
        )
        # Move to CUDA after processing
        result_cuda['input_ids'] = result_cuda['input_ids'].to(device)
        print(f"   - Device after moving: {result_cuda['input_ids'].device}")
    
    return True


def test_edge_cases(processor: VibePodProcessor):
    """Test edge cases and error handling"""
    print("\n" + "="*50)
    print("Testing Edge Cases")
    print("="*50)
    
    # Empty input
    print("\n1. Testing empty input:")
    try:
        result = processor(text="", return_tensors="pt")
        print(f"   - Handled empty input, shape: {result['input_ids'].shape}")
    except Exception as e:
        print(f"   - Error (expected): {type(e).__name__}: {e}")
    
    # Mixed batch with empty
    print("\n2. Testing batch with empty string:")
    try:
        result = processor(
            text=["Speaker 1: Hello", "", "Speaker 2: World"],
            padding=True,
            return_tensors="pt",
        )
        print(f"   - Handled mixed batch, shape: {result['input_ids'].shape}")
    except Exception as e:
        print(f"   - Error: {type(e).__name__}: {e}")
    
    # Very long speaker ID
    print("\n3. Testing unusual speaker IDs:")
    unusual_script = "Speaker 999: Testing large speaker ID\nSpeaker 0: Testing zero ID"
    result = processor(text=unusual_script, return_tensors="pt")
    print(f"   - Processed successfully, shape: {result['input_ids'].shape}")
    
    return True


def main():
    args = parse_args()
    
    print(f"Loading processor from {args.model_path}")
    
    # Load processor
    processor = VibePodProcessor.from_pretrained(
        args.model_path,
        cache_dir="/mnt/msranlp/zliang/hf_ckpt"
    )
    
    # Load voice samples if provided
    voice_samples = None
    if args.voice_samples_dir and os.path.exists(args.voice_samples_dir):
        voice_files = [f for f in os.listdir(args.voice_samples_dir) if f.endswith('.wav')]
        voice_samples = [os.path.join(args.voice_samples_dir, f) for f in voice_files[:3]]
        print(f"Loaded {len(voice_samples)} voice samples")
    
    # Create test scripts
    scripts = create_test_scripts()
    
    # Run tests based on arguments
    all_tests = not any([args.test_single, args.test_batch, args.test_padding, 
                         args.test_truncation, args.test_tensors])
    
    if all_tests or args.test_single:
        test_single_input(processor, scripts, voice_samples)
    
    if all_tests or args.test_batch:
        test_batch_input(processor, scripts, voice_samples)
    
    if all_tests or args.test_padding:
        test_padding_strategies(processor, scripts)
    
    if all_tests or args.test_truncation:
        test_truncation(processor, scripts)
    
    if all_tests or args.test_tensors:
        test_tensor_types(processor, scripts)
    
    # Always test edge cases
    test_edge_cases(processor)
    
    print("\n" + "="*50)
    print("All tests completed!")
    print("="*50)


if __name__ == "__main__":
    main()

# Example usage:
# python test_vibepod_processor.py --model_path /tmp/vibepod-model
# python test_vibepod_processor.py --model_path /tmp/vibepod-model --test_batch
# python test_vibepod_processor.py --model_path /tmp/vibepod-model --voice_samples_dir /path/to/voices