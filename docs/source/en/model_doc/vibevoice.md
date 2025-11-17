# VibeVoice

## Overview

VibeVoice is a novel framework for synthesizing high-fidelity, long-form speech with multiple speakers by employing a next-token diffusion approach within a Large Language Model (LLM) structure. It's designed to capture the authentic conversational "vibe" and is particularly suited for generating audio content like podcasts and multi-participant audiobooks.

**Core Architecture**

The VibeVoice framework integrates three key components:
- **Speech Tokenizers:** Utilize specialized acoustic and semantic tokenizers, where the acoustic tokenizer uses a $\sigma$-VAE to achieve ultra-low compression (7.5 tokens/sec, 3200x) for scalability and fidelity, and the semantic tokenizer uses an ASR proxy task for content-centric feature extraction.
- **Large Language Model (LLM):** Use Qwen2.5 (in 1.5B and 7B versions) as its core sequence model.
- **Token-Level Diffusion Head:** condition on the LLM's hidden state and be responsible for predicting the continuous VAE features in a streaming way.


## Key Features

- **Long-Form Synthesis**: Can synthesize multi-speaker conversational speech for up to 90 minutes.
- **Multi-Speaker Dialogue**: Capable of synthesizing audio with a maximum of 4 speakers.
- **State-of-the-Art Quality**: Outperforms baselines on both subjective and objective metrics.
- **High Compression**: Achieved by a novel acoustic tokenizer operating at an ultra-low 7.5 Hz frame rate.
- **Scalable LLM**: Scaling the core LLM from 1.5B to 7B significantly improves perceptual quality.

## Usage

```python
import os
import re
import time

from huggingface_hub import snapshot_download
import diffusers
import numpy as np
import torch
from tqdm import tqdm

from transformers import AutoProcessor, VibeVoiceForConditionalGeneration
from transformers.audio_utils import load_audio_librosa


# set seed for deterministic
torch.manual_seed(42)
np.random.seed(42)

model_path = "bezzam/VibeVoice-1.5B"
sampling_rate = 24000
max_new_tokens = None  # None for full generation
output_dir = "./vibevoice_output"
torch_device = "cuda" if torch.cuda.is_available() else "cpu"
bfloat16 = False
batch_flag = False
voice_sample_repo = "yingwanghf/vibe_voice_sample"

def input_process(conversation: list, voice_paths: list, sampling_rate: int = 24000) -> list:
    added_voices = set()
    for entry in conversation:
        role = entry["role"]
        voice_index = int(role)
        if voice_index in added_voices:
            continue
        else:
            try:
                voice_path = voice_paths[voice_index]
                voice_content = load_audio_librosa(voice_path, sampling_rate=sampling_rate)
                voice = {'type': 'audio', 'audio': voice_content}
                entry["content"].append(voice)
                added_voices.add(voice_index)
            except IndexError:
                print(f"Warning: No audio path provided for role '{role}'. Skipping audio assignment for this entry.")

    return conversation

def main():
    repo_dir = snapshot_download(
        repo_id=voice_sample_repo,
        repo_type="dataset",
    )

    # Only five voices for use
    # 'en-Alice_woman', 'en-Ben_man', 'en-Carter_man', 'en-Maya_woman', 'in-Samuel_man'
    voice_fn_1 = [
        "en-Alice_woman.wav", 
        "en-Ben_man.wav"
    ]
    # Please follow the format
    conversation_1 = [
	    {"role": "0", "content": [{"type": "text", "text": "Hello, how are you?"}]},
	    {"role": "1", "content": [{"type": "text", "text": "I'm fine, thank you! And you?"}]},
	    {"role": "0", "content": [{"type": "text", "text": "I'm doing well, thanks for asking."}]},
	    {"role": "1", "content": [{"type": "text", "text": "That's great to hear. What have you been up to lately?"}]},
	    {"role": "0", "content": [{"type": "text", "text": "Just working and spending time with family."}]}
	]
    voice_paths_1 = [f"{repo_dir}/{fn}" for fn in voice_fn_1]
    conversation_1 = input_process(conversation_1, voice_paths_1, sampling_rate=sampling_rate)

    voice_fn_2 = [
        "en-Maya_woman.wav", 
        "en-Carter_man.wav"
    ]

    conversation_2 = [
	    {"role": "0", "content": [{"type": "text", "text": "Hey, remember 'See You Again'?"}]},
	    {"role": "1", "content": [{"type": "text", "text": "Yeah… from Furious 7, right? That song always hits deep."}]},
	    {"role": "0", "content": [{"type": "text", "text": "Let me try to sing a part of it for you."}]},
	    {"role": "0", "content": [{"type": "text", "text": "It's been a long day… without you, my friend. And I'll tell you all about it when I see you again…"}]},
	    {"role": "1", "content": [{"type": "text", "text": "Wow… that line. Every time."}]},
	]
    voice_paths_2 = [f"{repo_dir}/{fn}" for fn in voice_fn_2]
    conversation_2 = input_process(conversation_2, voice_paths_2, sampling_rate=sampling_rate)

    if batch_flag:
        conversations = [conversation_1, conversation_2]
    else:
        conversations = [conversation_1]

    processor = AutoProcessor.from_pretrained(model_path)
    model = VibeVoiceForConditionalGeneration.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if bfloat16 else None,
        device_map=torch_device,
    ).eval()

    # determine model dtype
    model_dtype = next(model.parameters()).dtype

    # Prepare inputs for the model
    inputs = processor.apply_chat_template(
        conversations, 
        tokenize=True,
        return_dict=True
    ).to(torch_device, dtype=model_dtype)
    print("\ninput_ids shape : ", inputs["input_ids"].shape)

    # Generate audio
    start_time = time.time()

    # Noise scheduler from diffusers library
    noise_scheduler = getattr(diffusers, model.generation_config.noise_scheduler)(
        **model.generation_config.noise_scheduler_config
    )

    # Define a callback to monitor the progress of the generation
    completed_samples = set()
    with tqdm(desc="Generating") as pbar:
        def monitor_progress(p_batch):
            # p_batch format: [current_step, max_step, completion_step] for each sample
            finished_samples = (p_batch[:, 0] == p_batch[:, 1]).nonzero(as_tuple=False).squeeze(1)
            if finished_samples.numel() > 0:
                for sample_idx in finished_samples.tolist():
                    if sample_idx not in completed_samples:
                        completed_samples.add(sample_idx)
                        completion_step = int(p_batch[sample_idx, 2])
                        print(f"Sample {sample_idx} completed at step {completion_step}", flush=True)

            active_samples = p_batch[:, 0] < p_batch[:, 1]
            if active_samples.any():
                active_progress = p_batch[active_samples]
                max_active_idx = torch.argmax(active_progress[:, 0])
                p = active_progress[max_active_idx].detach().cpu()
            else:
                p = p_batch[0].detach().cpu()

            pbar.total = int(p[1])
            pbar.n = int(p[0])
            pbar.update()

        # Generate!
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            noise_scheduler=noise_scheduler,
            monitor_progress=monitor_progress,
            return_dict_in_generate=True,
        )
    generation_time = time.time() - start_time
    print(f"Generation time: {generation_time:.2f} seconds")

    # Calculate audio duration and additional metrics
    if outputs.speech_outputs and outputs.speech_outputs[0] is not None:
        audio_samples = outputs.speech_outputs[0].shape[-1] if len(outputs.speech_outputs[0].shape) > 0 else len(outputs.speech_outputs[0])
        audio_duration = audio_samples / sampling_rate
        rtf = generation_time / audio_duration if audio_duration > 0 else float('inf')

        print(f"Generated audio duration: {audio_duration:.2f} seconds")
        print(f"RTF (Real Time Factor): {rtf:.2f}x")
    else:
        print("No audio output generated")

    # Calculate token metrics
    input_tokens = inputs['input_ids'].shape[1]  # Number of input tokens
    output_tokens = outputs.sequences.shape[1]  # Total tokens (input + generated)
    generated_tokens = output_tokens - input_tokens

    print(f"Prefilling tokens: {input_tokens}")
    print(f"Generated tokens: {generated_tokens}")
    print(f"Total tokens: {output_tokens}")

    # Save output (processor handles device internally)
    txt_filename = "conversation"
    os.makedirs(output_dir, exist_ok=True)

    for i, speech in enumerate(outputs.speech_outputs):
        output_path = os.path.join(output_dir, f"{txt_filename}_generated_{i}.wav")
        processor.save_audio(
            speech,
            output_path,
        )
        print(f"Saved output to {output_path}")

if __name__ == "__main__":
    main()
```

