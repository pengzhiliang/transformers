import os
import argparse
import json
import torch 
from transformers import AutoTokenizer
import sys
sys.path.insert(0, "/data/yaoyaochang/code/speech/transformers/src/transformers/models/vibepod/nano-vllm")
from nanovllm import LLM, SamplingParams

from transformers.models.vibepod.modular_vibepod_text_tokenizer import VibePodTextTokenizer, VibePodTextTokenizerFast
from transformers.models.vibepod.vibepod_processor import VibePodProcessor
from transformers.utils import logging
from transformers.tokenization_utils_base import PaddingStrategy


logging.set_verbosity_info()
logger = logging.get_logger(__name__)


# import torch

# _orig_tensor_repr = torch.Tensor.__repr__
# def _tensor_repr(self):
#     shape_str = f"shape:{tuple(self.shape)}, content:"
#     content_str = _orig_tensor_repr(self)
#     return f"{shape_str}{content_str}"
# torch.Tensor.__repr__ = _tensor_repr


def tensor_preview(t, k=6):
    # 展示前 k 个元素
    with torch.no_grad():
        flat = t.detach().cpu().flatten()
    return f"Tensor(shape={tuple(t.shape)}, dtype={t.dtype}, device={t.device}, " \
           f"preview={flat[:k].tolist()}{'…' if flat.numel()>k else ''})"

import torch, builtins
def _tensor_repr(self):
    return f"shape:{tuple(self.shape)}, dtype:{self.dtype}, device:{self.device}"
torch.Tensor.__repr__ = _tensor_repr


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

def read_scp(scp_path: str):
    
    save_names = []
    scripts = []
    voice_samples = []
    if scp_path.endswith('.json'):
        # If the input is a single JSON 
        save_names.append(os.path.basename(scp_path).split('.')[0])
        with open(scp_path, 'r') as json_file:
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
        with open(scp_path, 'r') as f:
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
    return save_names, scripts, voice_samples


def main():
    args = parse_args()
    print(f"Arguments: {args}")
    save_names, scripts, voice_samples = read_scp(args.scp_path)
    
    print(f"Loading processor & model from {args.model_path}")
    processor = VibePodProcessor.from_pretrained(
        args.model_path,
        cache_dir="/mnt/msranlp/zliang/hf_ckpt"
    )

    # tokenizer = VibePodTextTokenizerFast.from_pretrained("Qwen/Qwen2.5-1.5B", cache_dir='/mnt/msranlp/zliang/hf_ckpt')
    qwen_path = "/data/yaoyaochang/code/speech/data/Qwen/Qwen2.5-1.5B"
    cuda_start_idx = 2
    llm = LLM(vibepod_path=args.model_path, qwen_path=qwen_path, enforce_eager=False, tensor_parallel_size=1, cuda_start_idx=cuda_start_idx, tokenizer=processor.tokenizer)
    
    
    inputs = processor(
        text=scripts,
        voice_samples=voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )
    cfg_scale = 1.3
    inputs['cfg_scale'] = cfg_scale
    inputs['max_new_tokens'] = None
    inputs['tokenizer'] = processor.tokenizer
    inputs['return_speech'] = True
    inputs['processor'] = processor
    outputs = llm.generate(inputs)

    for i, save_name in enumerate(save_names):
        output_path = f"/data/yaoyaochang/code/speech/data/vibepod_outputs/nano_cfg_{cfg_scale}_{save_name}.wav"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processor.save_audio(
            outputs.speech_outputs[i],
            output_path=output_path,
        )
        print(f"Saved output to {output_path}")


if __name__ == "__main__":
    main()
