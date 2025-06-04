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
import time

import numpy as np
import torch
import soundfile as sf
import librosa

from transformers.models.vibepod.configuration_vibepod import VibePodConfig
from transformers.models.vibepod.modeling_vibepod import VibePodForConditionalGeneration

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
        default="/tmp/vibepod-model",
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--voice_samples_dir",
        type=str,
        default='/mnt/conversationhub/zhiliang/other/',
        help="Directory containing voice samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for tensor tests",
    )
    
    return parser.parse_args()

def test_batch_input(processor: VibePodProcessor, scripts: Dict[str, Any], voice_samples: List[str] = None):
    """Test batch input processing"""
    print("\n" + "="*50)
    print("Testing Batch Input Processing")
    print("="*50)
    
    # Create batch of scripts
    batch_texts = scripts
    
    # Create corresponding voice samples
    batch_voice_samples = None
    if voice_samples:
        batch_voice_samples = [voice_samples] * len(batch_texts)  # Repeat voice samples for each text script
    
    print("\n1. Testing batch processing with padding:")
    result = processor(
        text=batch_texts,
        voice_samples=batch_voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    
    return result

def main():
    args = parse_args()
    
    print(f"Loading processor from {args.model_path}")
    
    # Load processor
    processor = VibePodProcessor.from_pretrained(
        '/tmp/vibepod-model',
        cache_dir="/mnt/msranlp/zliang/hf_ckpt"
    )

    # Load model
    model = VibePodForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )

    model.eval()
    
    model.set_ddpm_inference_steps(num_steps=10)
    
    # Load voice samples if provided
    voice_samples = None
    if args.voice_samples_dir and os.path.exists(args.voice_samples_dir):
        voice_files = [f for f in os.listdir(args.voice_samples_dir) if f.endswith('.wav')]
        voice_samples = [os.path.join(args.voice_samples_dir, f) for f in voice_files[:3]]
        print(f"Loaded {len(voice_samples)} voice samples")
    
    # Create test scripts
    scripts = [
        "/home/pengzhiliang/speech_qwen/evals/speech/podcast/speech_tokenizer.txt",
        # "/home/pengzhiliang/speech_qwen/evals/speech/podcast/valle.txt",
        # "/home/pengzhiliang/speech_qwen/evals/speech/podcast/msft.txt",
        # "/home/pengzhiliang/speech_qwen/evals/speech/podcast/deepseekR1.json",
        # "/home/pengzhiliang/speech_qwen/evals/speech/podcast/ai_trend.txt",
    ]
    

    # if all_tests or args.test_single:
    #     test_single_input(processor, scripts, voice_samples)
    
    # if all_tests or args.test_batch:
    inputs = test_batch_input(processor, scripts, voice_samples)
    
    # if all_tests or args.test_padding:
    #     test_padding_strategies(processor, scripts)
    
    # ========================================================
    # add time logger
    start_time = time.time()
    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=1.5,
        tokenizer=processor.tokenizer,
    )
    print(f"Generation time: {time.time() - start_time:.2f} seconds")

    processor.save_audio(
        outputs.speech_outputs,
        output_path="./vibepod_outputs",
    )

    # Always test edge cases
    # test_edge_cases(processor)
    
    # print("\n" + "="*50)
    # print("All tests completed!")
    # print("="*50)


if __name__ == "__main__":
    main()

# Example usage:
# python test_vibepod_processor.py --model_path /tmp/vibepod-model
# python test_vibepod_processor.py --model_path /tmp/vibepod-model --test_batch
# python test_vibepod_processor.py --model_path /tmp/vibepod-model --voice_samples_dir /path/to/voices