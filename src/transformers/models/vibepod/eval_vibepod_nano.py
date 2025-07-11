import os
from transformers import AutoTokenizer
import sys
sys.path.insert(0, "/data/yaoyaochang/code/speech/transformers/src/transformers/models/vibepod/nano-vllm")
from nanovllm import LLM, SamplingParams

from transformers.models.vibepod.modular_vibepod_text_tokenizer import VibePodTextTokenizer, VibePodTextTokenizerFast



def main():
    vibepod_path = "/data/yaoyaochang/code/speech/data/model/qwen_1.5b_stream_podcast_v2_4096_text-1_ddpm-diff-5_acous-seman-tok3200x64_lr1e-4_bsz4m_8n_100k"
    
    tokenizer = VibePodTextTokenizerFast.from_pretrained("Qwen/Qwen2.5-1.5B", cache_dir='/mnt/msranlp/zliang/hf_ckpt')
    qwen_path = "/data/yaoyaochang/code/speech/data/Qwen/Qwen2.5-1.5B"
    llm = LLM(vibepod_path=vibepod_path, qwen_path=qwen_path, enforce_eager=False, tensor_parallel_size=1, cuda_start_idx=2, tokenizer=tokenizer)

    sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
    prompts = [
        "introduce yourself",
        "list all prime numbers within 100",
    ]
    prompts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True
        )
        for prompt in prompts
    ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
