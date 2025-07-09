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


# import sys
# sys.path.insert(0, "/data/yaoyaochang/code/speech/nano-vllm")
from nanovllm.utils.loader import load_model

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
        "--scp_path",
        type=str,
        default='/mnt/conversationhub/zhiliang/exp/podcast_eval/select_mosset/transcripts/transcript_small.scp',
        help="Directory containing voice samples",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for tensor tests",
    )
    
    return parser.parse_args()

def main():
    args = parse_args()


    save_names = []
    scripts = []
    voice_samples = []
    if args.scp_path.endswith('.json'):
        # If the input is a single JSON 
        save_names.append(os.path.basename(args.scp_path).split('.')[0])
        with open(args.scp_path, 'r') as json_file:
            data = json.load(json_file)

            prompts = data['prompts']
            sample_voices = []
            # Sort keys by converting to integers and get audio paths in order
            for k in sorted(prompts.keys(), key=int):
                sample_voices.append(prompts[k]['audio_path'])
            voice_samples.append(sample_voices)

            transcript = data['transcript']
            sample_scripts = []
            for sample_sentence in transcript:
                speaker_id = sample_sentence['speaker']
                sample_scripts.append(f"Speaker {speaker_id}: {sample_sentence['text']}")
            scripts.append('\n'.join(sample_scripts))
    else:
        with open(args.scp_path, 'r') as f:
            scp_lines = f.readlines()
            # deepseekR1 /mnt/conversationhub/zhiliang/exp/podcast_eval/select_mosset/transcripts/simulated_sample/deepseekR1.json
            for line in scp_lines:
                parts = line.strip().split()
                if len(parts) != 2:
                    print(f"Skipping invalid line: {line.strip()}")
                    continue
                    
                save_name, json_path = parts
                print(f"Processing file: {json_path}")
                # Check if file exists and is readable
                if not os.path.exists(json_path):
                    breakpoint()
                    print(f"File not found: {json_path}")
                    continue

                save_names.append(save_name)
                with open(json_path, 'r') as json_file:
                    data = json.load(json_file)

                    prompts = data['prompts']
                    sample_voices = []
                    # Sort keys by converting to integers and get audio paths in order
                    for k in sorted(prompts.keys(), key=int):
                        sample_voices.append(prompts[k]['audio_path'])
                    voice_samples.append(sample_voices)

                    transcript = data['transcript']
                    sample_scripts = []
                    for sample_sentence in transcript:
                        speaker_id = sample_sentence['speaker']
                        sample_scripts.append(f"Speaker {speaker_id}: {sample_sentence['text']}")
                    scripts.append('\n'.join(sample_scripts))
    
    # Load processor
    print(f"Loading processor & model from {args.model_path}")
    processor = VibePodProcessor.from_pretrained(
        args.model_path,
        cache_dir="/mnt/msranlp/zliang/hf_ckpt"
    )

    # Load model
    model = VibePodForConditionalGeneration.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        attn_implementation="flash_attention_2",
    )
    # print("load model in eval_vibepod.py")
    # load_model(model.model.llm.model_runner.model, "/data/yaoyaochang/code/speech/data/Qwen/Qwen2.5-1.5B")
    # load_model(model.model.language_model, "/data/yaoyaochang/code/speech/data/Qwen/Qwen2.5-1.5B-from-vibepod")

    model.eval()
    model.set_ddpm_inference_steps(num_steps=5)

    inputs = processor(
        text=scripts,
        voice_samples=voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    model_name = args.model_path.split('/')[-2]
    cfg_scale = 1.3
    print(f"model name: {model_name}")

    # ==================================================================
    # start_time = time.time()
    # outputs = model.generate_negative_with_start_end_token(
    #     **inputs,
    #     max_new_tokens=None,
    #     cfg_scale=cfg_scale,
    #     tokenizer=processor.tokenizer,
    # )
    # print(f"Generation time: {time.time() - start_time:.2f} seconds")

    # for i, save_name in enumerate(save_names):
    #     output_path = f"/mnt/conversationhub/zhiliang/exp/podcast_eval/{model_name}/batch_negative_with_start_end_token_cfg_{cfg_scale}_step_5/{save_name}.wav"
    #     # output_path = f"./vibepod_outputs/cfg_1.5_step_5_negative_with_SE_{save_name}.wav"
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     processor.save_audio(
    #         outputs.speech_outputs[i],
    #         output_path=output_path,
    #     )
    #     print(f"Saved output to {output_path}")
    
    # ==================================================================
    # start_time = time.time()
    # outputs = model.generate_negative_without_start_end_token(
    #     **inputs,
    #     max_new_tokens=None,
    #     cfg_scale=cfg_scale,
    #     tokenizer=processor.tokenizer,
    # )
    # print(f"Generation time: {time.time() - start_time:.2f} seconds")

    # for i, save_name in enumerate(save_names):
    #     output_path = f"/mnt/conversationhub/zhiliang/exp/podcast_eval/transformers_batch_generate_negative_without_start_end_token/{save_name}.wav"
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     processor.save_audio(
    #         outputs.speech_outputs[i],
    #         output_path=output_path,
    #     )
    #     print(f"Saved output to {output_path}")

    # ==================================================================
    # start_time = time.time()
    # for sample_script, sample_voice, sample_name in zip(scripts, voice_samples, save_names):
    #     print(f"Processing sample: {sample_name}")
    #     sample_input = processor(
    #         text=[sample_script],
    #         voice_samples=[sample_voice],
    #         padding=True,
    #         return_tensors="pt",
    #         return_attention_mask=True,
    #     )
    #     sample_outputs = model.generate_negative_without_start_end_token(
    #         **sample_input,
    #         max_new_tokens=None,
    #         cfg_scale=cfg_scale,
    #         tokenizer=processor.tokenizer,
    #     )
    #     output_path = f"/mnt/conversationhub/zhiliang/exp/podcast_eval/transformers_single_generate_negative_without_start_end_token/{sample_name}.wav"
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     processor.save_audio(
    #         sample_outputs.speech_outputs[0],
    #         output_path=output_path,
    #     )
    #     print(f"Saved output to {output_path}")
    # print(f"Single sample generation time: {time.time() - start_time:.2f} seconds")

    # ==================================================================
    start_time = time.time()
    outputs = model.generate_refresh_negative(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
    )
    print(f"Generation time: {time.time() - start_time:.2f} seconds")

    for i, save_name in enumerate(save_names):
        # output_path = f"/mnt/conversationhub/zhiliang/exp/podcast_eval/transformers_batch_generate_refresh_negative/{save_name}.wav"
        # output_path = f"/mnt/conversationhub/zhiliang/exp/podcast_eval/{model_name}/batch_refresh_negative_cfg_{cfg_scale}_step_5/{save_name}.wav"
        output_path = f"./vibepod_outputs/cfg_{cfg_scale}_{save_name}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processor.save_audio(
            outputs.speech_outputs[i],
            output_path=output_path,
        )
        print(f"Saved output to {output_path}")

    # ==================================================================
    # start_time = time.time()
    # for sample_script, sample_voice, sample_name in zip(scripts, voice_samples, save_names):
    #     print(f"Processing sample: {sample_name}")
    #     sample_input = processor(
    #         text=[sample_script],
    #         voice_samples=[sample_voice],
    #         padding=True,
    #         return_tensors="pt",
    #         return_attention_mask=True,
    #     )
    #     sample_outputs = model.generate_refresh_negative(
    #         **sample_input,
    #         max_new_tokens=None,
    #         cfg_scale=cfg_scale,
    #         tokenizer=processor.tokenizer,
    #     )
    #     output_path = f"/mnt/conversationhub/zhiliang/exp/podcast_eval/transformers_single_generate_refresh_negative/{sample_name}.wav"
    #     os.makedirs(os.path.dirname(output_path), exist_ok=True)
    #     processor.save_audio(
    #         sample_outputs.speech_outputs[0],
    #         output_path=output_path,
    #     )
    #     print(f"Saved output to {output_path}")
    # print(f"Single sample generation time: {time.time() - start_time:.2f} seconds")


    # ==================================================================
    # test one sample

    # sample_input = processor(
    #     text=[scripts[8], scripts[9]],
    #     voice_samples=[voice_samples[8], voice_samples[9]],
    #     padding=True,
    #     return_tensors="pt",
    #     return_attention_mask=True,
    # )
    # # generate_negative_without_start_end_token
    # # generate_refresh_negative
    # sample_outputs = model.generate_refresh_negative(
    #     **sample_input,
    #     max_new_tokens=None,
    #     cfg_scale=cfg_scale,
    #     tokenizer=processor.tokenizer,
    # )
    # processor.save_audio(
    #         sample_outputs.speech_outputs,
    #         output_path='./vibepod_outputs',
    #     )

if __name__ == "__main__":
    main()

# /mnt/conversationhub/zhiliang/exp/sp_mllm/qwen_1.5b_stream_podcast_v2_4096_text-1_ddpm-diff-5_acous-seman-tok3200x64_lr1e-4_bsz4m_8n_100k/hf_ckpt
# /mnt/conversationhub/zhiliang/exp/sp_mllm/qwen_1.5b_stream_en-zh-1-1_cfg0.1_12288_text-1_ddpm-diff-5_acous-seman-tok3200x64_lr1e-4_bsz3m_2n_40k/hf_ckpt_step30k
# /mnt/conversationhub/zhiliang/exp/sp_mllm/qwen_1.5b_stream_en-zh-1-1_12288_text-1_ddpm-diff-5_acous-seman-tok3200x64_lr1e-4_bsz3m_2n_40k/hf_ckpt_step10k
# /mnt/conversationhub/zhiliang/exp/sp_mllm/qwen_1.5b_stream_en-zh-1-1_12288_text-1_ddpm-diff-5_acous-seman-tok3200x64_lr1e-4_bsz3m_2n_40k/hf_ckpt_step40k
