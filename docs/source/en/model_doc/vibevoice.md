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
### Basic Usage
```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
import torch
import re
import os
from huggingface_hub import hf_hub_download
from typing import List, Tuple, Union, Dict, Any

class VoiceMapper:
    def __init__(self):
        self.setup_voice_presets()
        new_dict = {}
        for name, path in self.voice_presets.items():
            
            if '_' in name:
                name = name.split('_')[0]
            
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        REPO_ID = "yingwanghf/vibe_voice_sample"
        REPO_TYPE = "dataset" 
        voices_dir = os.path.join(os.path.dirname(__file__), "voice_samples")
        os.makedirs(voices_dir, exist_ok=True)
        audio_files = ["en-Alice_woman.wav", "en-Ben_man.wav", "en-Carter_man.wav", "en-Maya_woman.wav", "in-Samuel_man.wav"]
        for audio in audio_files:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=audio,
                repo_type=REPO_TYPE,
                local_dir=voices_dir,
            )
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        self.voice_presets = {}
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice

def format_conversation_to_script(conversation_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    script_lines = []
    speaker_ids = []
    for turn in conversation_data:
        role_id = turn.get("role")
        content_parts = [item.get("text", "") for item in turn.get("content", []) if item.get("type") == "text"]
        text = " ".join(content_parts).strip()
        if role_id and text:
            script_lines.append(f"Speaker {role_id}: {text}")
            speaker_ids.append(role_id)

    return script_lines, speaker_ids

def script_to_chat_template(scripts: str) -> list:
    conversation = []
    lines = [line.strip() for line in scripts.strip().split('\n') if line.strip()]
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    current_speaker = None
    current_text = ""
    for line in lines:
        try:
            match = re.match(speaker_pattern, line, re.IGNORECASE)
            if match:
                if current_speaker and current_text:
                    chat_entry = {
                        "role": current_speaker,
                        "content": [{"type": "text", "text": current_text.strip()}]
                    }
                    
                    conversation.append(chat_entry)
                # Start new speaker
                current_speaker = match.group(1).strip()
                current_text = match.group(2).strip()
            else:
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
        except Exception as e:
            print(f"Error processing line '{line}': {e}")
            continue

    if current_speaker and current_text:
        chat_entry = {
            "role": current_speaker,
            "content": [{"type": "text", "text": current_text.strip()}]
        }
        
        conversation.append(chat_entry)
    return conversation

def input_process(txt_content: str, voices: List[str]) -> Tuple[str, List[str]]:
    voice_mapper = VoiceMapper()
    scripts, speaker_numbers = format_conversation_to_script(txt_content)

    full_script = '\n'.join(scripts)

    speaker_name_mapping = {}
    speaker_names_list = voices
    for i, name in enumerate(speaker_names_list):
        speaker_name_mapping[str(i)] = name
    voice_samples = []

    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)

    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_path = voice_mapper.get_voice_path(speaker_name)
        voice_samples.append(voice_path)

    return full_script, voice_samples

def main():
    model_path = "microsoft/VibeVoice-1.5b"
    cfg_scale = 1.3
    generated_audio_path = os.path.join(os.path.dirname(__file__), "outputs", "generated_audio.wav")
    os.makedirs(os.path.dirname(generated_audio_path), exist_ok=True)
    # Please follow the format below to add new scripts and voices
    conversation = [
	    {"role": "0", "content": [{"type": "text", "text": "Hello, how are you?"}]},
	    {"role": "1", "content": [{"type": "text", "text": "I'm fine, thank you! And you?"}]},
	    {"role": "0", "content": [{"type": "text", "text": "I'm doing well, thanks for asking."}]},
	    {"role": "1", "content": [{"type": "text", "text": "That's great to hear. What have you been up to lately?"}]},
	    {"role": "0", "content": [{"type": "text", "text": "Just working and spending time with family."}]},
	]

    # Only five voices for use
    # 'en-Alice_woman', 'en-Ben_man', 'en-Carter_man', 'en-Maya_woman', 'in-Samuel_man'
    voices = ["en-Alice_woman", "en-Carter_man"]
    full_script, voice_samples = input_process(conversation, voices)

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        attn_implementation="flash_attention_2"
    )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)
    inputs = processor(
        text=[full_script], 
        voice_samples=[voice_samples], 
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False},
        verbose=True,
    )

    processor.save_audio(
        outputs.speech_outputs[0],  # First (and only) batch item
        output_path=generated_audio_path,
    )

if __name__ == "__main__":
    main()
```

### Batch Usage
```python
from vibevoice.modular.modeling_vibevoice_inference import VibeVoiceForConditionalGenerationInference
from vibevoice.processor.vibevoice_processor import VibeVoiceProcessor
import torch
import re
import os
from huggingface_hub import hf_hub_download
from typing import List, Tuple, Union, Dict, Any
from transformers import set_seed

class VoiceMapper:  
    def __init__(self):
        self.setup_voice_presets()
        new_dict = {}
        for name, path in self.voice_presets.items():
            
            if '_' in name:
                name = name.split('_')[0]
            
            if '-' in name:
                name = name.split('-')[-1]

            new_dict[name] = path
        self.voice_presets.update(new_dict)

    def setup_voice_presets(self):
        REPO_ID = "yingwanghf/vibe_voice_sample"
        REPO_TYPE = "dataset" 
        voices_dir = os.path.join(os.path.dirname(__file__), "voice_samples")
        os.makedirs(voices_dir, exist_ok=True)
        audio_files = ["en-Alice_woman.wav", "en-Ben_man.wav", "en-Carter_man.wav", "en-Maya_woman.wav", "in-Samuel_man.wav"]
        for audio in audio_files:
            local_path = hf_hub_download(
                repo_id=REPO_ID,
                filename=audio,
                repo_type=REPO_TYPE,
                local_dir=voices_dir,
            )
        if not os.path.exists(voices_dir):
            print(f"Warning: Voices directory not found at {voices_dir}")
            self.voice_presets = {}
            self.available_voices = {}
            return
        self.voice_presets = {}
        wav_files = [f for f in os.listdir(voices_dir) 
                    if f.lower().endswith('.wav') and os.path.isfile(os.path.join(voices_dir, f))]
        for wav_file in wav_files:
            name = os.path.splitext(wav_file)[0]
            full_path = os.path.join(voices_dir, wav_file)
            self.voice_presets[name] = full_path
        self.voice_presets = dict(sorted(self.voice_presets.items()))
        self.available_voices = {
            name: path for name, path in self.voice_presets.items()
            if os.path.exists(path)
        }

    def get_voice_path(self, speaker_name: str) -> str:
        if speaker_name in self.voice_presets:
            return self.voice_presets[speaker_name]
        speaker_lower = speaker_name.lower()
        for preset_name, path in self.voice_presets.items():
            if preset_name.lower() in speaker_lower or speaker_lower in preset_name.lower():
                return path
        default_voice = list(self.voice_presets.values())[0]
        print(f"Warning: No voice preset found for '{speaker_name}', using default voice: {default_voice}")
        return default_voice

def format_conversation_to_script(conversation_data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    script_lines = []
    speaker_ids = []
    for turn in conversation_data:
        role_id = turn.get("role")
        content_parts = [item.get("text", "") for item in turn.get("content", []) if item.get("type") == "text"]
        text = " ".join(content_parts).strip()
        if role_id and text:
            script_lines.append(f"Speaker {role_id}: {text}")
            speaker_ids.append(role_id)

    return script_lines, speaker_ids

def script_to_chat_template(scripts: str) -> list:
    conversation = []
    lines = [line.strip() for line in scripts.strip().split('\n') if line.strip()]
    speaker_pattern = r'^Speaker\s+(\d+):\s*(.*)$'
    current_speaker = None
    current_text = ""
    for line in lines:
        try:
            match = re.match(speaker_pattern, line, re.IGNORECASE)
            if match:
                if current_speaker and current_text:
                    chat_entry = {
                        "role": current_speaker,
                        "content": [{"type": "text", "text": current_text.strip()}]
                    }
                    
                    conversation.append(chat_entry)
                # Start new speaker
                current_speaker = match.group(1).strip()
                current_text = match.group(2).strip()
            else:
                if current_text:
                    current_text += " " + line
                else:
                    current_text = line
        except Exception as e:
            print(f"Error processing line '{line}': {e}")
            continue

    if current_speaker and current_text:
        chat_entry = {
            "role": current_speaker,
            "content": [{"type": "text", "text": current_text.strip()}]
        }
        
        conversation.append(chat_entry)
    return conversation

def input_process(txt_content: str, voices: List[str]) -> Tuple[str, List[str]]:
    voice_mapper = VoiceMapper()
    scripts, speaker_numbers = format_conversation_to_script(txt_content)

    full_script = '\n'.join(scripts)

    speaker_name_mapping = {}
    speaker_names_list = voices
    for i, name in enumerate(speaker_names_list):
        speaker_name_mapping[str(i)] = name
    voice_samples = []

    unique_speaker_numbers = []
    seen = set()
    for speaker_num in speaker_numbers:
        if speaker_num not in seen:
            unique_speaker_numbers.append(speaker_num)
            seen.add(speaker_num)

    for speaker_num in unique_speaker_numbers:
        speaker_name = speaker_name_mapping.get(speaker_num, f"Speaker {speaker_num}")
        voice_path = voice_mapper.get_voice_path(speaker_name)
        voice_samples.append(voice_path)

    return full_script, voice_samples

def process_batch(batch, model, processor, cfg_scale):

    batch_save_names = batch['save_names']
    batch_scripts = batch['scripts']
    batch_voice_samples = batch['voice_samples']
    
    inputs = processor(
        text=batch_scripts,
        voice_samples=batch_voice_samples,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    device = next(model.parameters()).device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=None,
        cfg_scale=cfg_scale,
        tokenizer=processor.tokenizer,
        generation_config={'do_sample': False, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 0},
        max_length_times=2,
    )

    reach_max_step_sample = outputs.reach_max_step_sample if hasattr(outputs, 'reach_max_step_sample') else None

    original_failed_indices = []
    retry_count = 0
    max_retries = 5
    
    while reach_max_step_sample is not None and reach_max_step_sample.any() and retry_count < max_retries:
        retry_count += 1
        current_failed_indices = torch.where(reach_max_step_sample)[0].tolist()
        
        if original_failed_indices:
            failed_indices = [original_failed_indices[idx] for idx in current_failed_indices]
            original_failed_indices = failed_indices
        else:
            failed_indices = current_failed_indices
            original_failed_indices = failed_indices
        
        retry_seed = 42 + retry_count * 1000 
        set_seed(retry_seed)

        failed_scripts = [batch_scripts[idx] for idx in failed_indices]
        failed_voice_samples = [batch_voice_samples[idx] for idx in failed_indices]
        
        retry_inputs = processor(
            text=failed_scripts,
            voice_samples=failed_voice_samples,
            padding=True,
            return_tensors="pt",
            return_attention_mask=True,
        )
        
        for key in retry_inputs:
            if isinstance(retry_inputs[key], torch.Tensor):
                retry_inputs[key] = retry_inputs[key].to(device)
        
        retry_outputs = model.generate(
            **retry_inputs,
            max_new_tokens=None,
            cfg_scale=cfg_scale,
            tokenizer=processor.tokenizer,
            generation_config={'do_sample': False, 'temperature': 0.95, 'top_p': 0.95, 'top_k': 0},
            max_length_times=2,
            verbose=True, 
        )

        for i, original_idx in enumerate(failed_indices):
            outputs.speech_outputs[original_idx] = retry_outputs.speech_outputs[i]
        
        reach_max_step_sample = retry_outputs.reach_max_step_sample if hasattr(retry_outputs, 'reach_max_step_sample') else None
    
    set_seed(42)

    for i, save_name in enumerate(batch_save_names):
        output_path = os.path.join(os.path.dirname(__file__), "outputs", f"{save_name}.wav")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        processor.save_audio(
            outputs.speech_outputs[i],
            output_path=output_path,
        )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def main():
    model_path = "microsoft/VibeVoice-1.5b"
    cfg_scale = 1.3
    # Please follow the format below to add new scripts and voices
    conversation_1 = [
	    {"role": "0", "content": [{"type": "text", "text": "Hello, how are you?"}]},
	    {"role": "1", "content": [{"type": "text", "text": "I'm fine, thank you! And you?"}]},
	    {"role": "0", "content": [{"type": "text", "text": "I'm doing well, thanks for asking."}]},
	    {"role": "1", "content": [{"type": "text", "text": "That's great to hear. What have you been up to lately?"}]},
	    {"role": "0", "content": [{"type": "text", "text": "Just working and spending time with family."}]},
	]

    # Only five voices for use
    # 'en-Alice_woman', 'en-Ben_man', 'en-Carter_man', 'en-Maya_woman', 'in-Samuel_man'
    voices_1 = ["en-Alice_woman", "en-Ben_man"]
    full_script_1, voice_samples_1 = input_process(conversation_1, voices_1)

    # Please follow the format below to add new scripts and voices
    conversation_2 = [
	    {"role": "0", "content": [{"type": "text", "text": "Hey, remember 'See You Again'?"}]},
	    {"role": "1", "content": [{"type": "text", "text": "Yeah… from Furious 7, right? That song always hits deep."}]},
	    {"role": "0", "content": [{"type": "text", "text": "Let me try to sing a part of it for you."}]},
	    {"role": "0", "content": [{"type": "text", "text": "It's been a long day… without you, my friend. And I'll tell you all about it when I see you again…"}]},
	    {"role": "1", "content": [{"type": "text", "text": "Wow… that line. Every time."}]},
	]

    # Only five voices for use
    # 'en-Alice_woman', 'en-Ben_man', 'en-Carter_man', 'en-Maya_woman', 'in-Samuel_man'
    voices_2 = ["en-Carter_man", "en-Maya_woman"]
    full_script_2, voice_samples_2 = input_process(conversation_2, voices_2)

    batch_data = {
        'save_names': ['conversation_1', 'conversation_2'],
        'scripts': [full_script_1, full_script_2],
        'voice_samples': [voice_samples_1, voice_samples_2],
    }

    set_seed(42)

    processor = VibeVoiceProcessor.from_pretrained(model_path)
    model = VibeVoiceForConditionalGenerationInference.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map='cuda',
        attn_implementation="flash_attention_2"
    )

    model.eval()
    model.set_ddpm_inference_steps(num_steps=10)

    process_batch(batch_data, model, processor, cfg_scale)

if __name__ == "__main__":
    main()
```