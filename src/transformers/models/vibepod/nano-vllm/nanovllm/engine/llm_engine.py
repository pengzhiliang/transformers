import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union, Callable
import time

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from transformers.generation import GenerationMixin, GenerationConfig, LogitsProcessor, LogitsProcessorList, StoppingCriteriaList
from transformers.models.vibepod.modular_vibepod_tokenizer import VibePodTokenizerStreamingCache
from transformers.modeling_outputs import ModelOutput


from datetime import datetime, timezone, timedelta
def get_beijing_time():
    bj_time = datetime.utcnow().replace(tzinfo=timezone.utc).astimezone(timezone(timedelta(hours=8))).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3] 
    return bj_time
import builtins
original_print = builtins.print
def custom_print(*args, **kwargs):
    original_print(get_beijing_time(), *args, **kwargs)
builtins.print = custom_print


@dataclass
class VibePodGenerationOutput(ModelOutput):
    """
    Output type for VibePod generation.
    
    Args:
        sequences (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
            The generated sequences. 
        speech_outputs (`List[torch.FloatTensor]`, *optional*):
            List of generated speech waveforms or latents for each speech segment.
    """
    sequences: torch.LongTensor = None
    seqs: List[Sequence] = None
    speech_outputs: Optional[List[torch.FloatTensor]] = None


class LLMEngine:

    def __init__(self, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        print(f"LLMEngine: config_kwargs={config_kwargs}")
        config = Config(kwargs['qwen_path'], **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        # 主进程必开一个model_runner，如果tensor_parallel_size > 1，则还需要开进程跑model_runner。并且会监听exit事件
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = kwargs.get("tokenizer")
        self.logits_processor = LogitsProcessorList()
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt):
        # 把一个prompt（字符串或token id列表）转换为Sequence对象，并添加到scheduler中
        seq = Sequence(prompt)
        self.scheduler.add(seq)

    def step(self, model_kwargs, input_ids, negative_kwargs, generation_config):
        """
        1. 跑一次scheduler，获取一批Sequence对象
        2. 跑model，得到token id列表
        3. 把返回token 写回各个Sequence
        4. 收集已经完成的sequence，return出去
        """
        seqs, is_prefill = self.scheduler.schedule()
        print(f"LLMEngine: Scheduled {len(seqs)} sequences, is_prefill={is_prefill}")
        hidden_states, logits = self.model_runner.call("run", seqs, is_prefill, model_kwargs)
        
        # Get next token logits and scores
        next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=hidden_states.device)
        next_token_scores = self.logits_processor(input_ids, next_token_logits)
        next_tokens = torch.argmax(next_token_scores, dim=-1)
        input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
        
        self.scheduler.postprocess(seqs, hidden_states, next_tokens)
        
        
        diffusion_indices = (next_tokens == generation_config.speech_diffusion_id).nonzero(as_tuple=False).squeeze(1)
        
        
        
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs) # 预填充阶段的token数与解码阶段的token数。后者是负数，取相反数得到正常值。
        return outputs, num_tokens, input_ids

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(self, kwargs):
        """
        把所有prompt都添加到scheduler中。
        不停调用 step()，直到所有的prompt都生成完毕。
        实时统计prefill和decode的吞吐量。
        """
        print(f"LLMEngine: Generating with kwargs.keys()={kwargs.keys()}")
        # for part in kwargs['input_ids'].split(1, dim=0):
        #     self.add_request(part.tolist())
        processor = kwargs.get('processor') # only use for save_audio here
        cfg_scale = kwargs.get('cfg_scale')
        for i_sample in range(kwargs['input_ids'].shape[0]):
            input_id = kwargs['input_ids'][i_sample].tolist()
            self.add_request(input_id)
        print(f"LLMEngine: Added {len(kwargs['input_ids'])} requests to scheduler.")
        
        print(f"kwargs.keys(): {kwargs.keys()}")
        generation_config, model_kwargs, input_ids = self.model_runner.vibepod._build_generate_config_model_kwargs(
            None, **kwargs
        )
        
        tokenizer = kwargs.get('tokenizer')        
        negative_kwargs = {
            'input_ids': torch.full((kwargs['input_ids'].shape[0], 1), tokenizer.speech_start_id, dtype=torch.long, device=kwargs['input_ids'].device),
            'attention_mask':  torch.ones((kwargs['input_ids'].shape[0], 1), dtype=torch.long, device=kwargs['input_ids'].device),
            'max_new_tokens': kwargs.get('max_new_tokens', 100) 
        }
        negative_generation_config, negative_model_kwargs, negative_input_ids = self.model_runner.vibepod._build_generate_config_model_kwargs(
            None, tokenizer, **negative_kwargs
        )
        
        
        logits_processor = LogitsProcessorList()

        acoustic_cache = VibePodTokenizerStreamingCache()
        semantic_cache = VibePodTokenizerStreamingCache()
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        finished_tags = torch.zeros(batch_size, dtype=torch.bool, device=device)
        correct_cnt = torch.zeros(batch_size, dtype=torch.long, device=device)
        is_prefill = True
        inputs_embeds = None

        # Initialize audio chunks storage for each sample
        audio_chunks = [[] for _ in range(batch_size)]

        initial_length = input_ids.shape[-1]
        max_steps = min(generation_config.max_length - initial_length, int(2 * initial_length))

        step_id = 0
        # while not self.is_finished():
        while True:
            if finished_tags.all():
                break
            print("\n")
            print(f"LLMEngine: Running step {step_id}...")

            seqs, is_prefill = self.scheduler.schedule()
            print(f"LLMEngine: Scheduled {len(seqs)} sequences, is_prefill={is_prefill}")
            for seq in seqs:
                print(f"LLMEngine: seq_id={seq.seq_id}, num_tokens={seq.num_tokens}, num_prompt_tokens={seq.num_prompt_tokens}, num_cached_tokens={seq.num_cached_tokens}, block_table={seq.block_table}")
            last_hidden_states, logits = self.model_runner.call("run", seqs, is_prefill, model_kwargs)
            
            # Get next token logits and scores
            next_token_logits = logits[:, -1, :].to(copy=True, dtype=torch.float32, device=last_hidden_states.device)
            next_token_scores = self.logits_processor(input_ids, next_token_logits)
            next_tokens = torch.argmax(next_token_scores, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            
            print(f"LLMEngine: next_tokens={next_tokens} {next_tokens.tolist()}")
            
            
            # reached end of generation
            if generation_config.eos_token_id is not None and (next_tokens == generation_config.eos_token_id).any():
                print(f"LLMEngine: status=EOS token at step={step_id}, next_tokens.shape={next_tokens.shape} {next_tokens.tolist()}")
                eos_indices = (next_tokens == generation_config.eos_token_id).nonzero(as_tuple=False).squeeze(1)
                # Only print for samples that are newly finished (not already marked as finished)
                new_eos_indices = eos_indices[~finished_tags[eos_indices]]
                if new_eos_indices.numel() > 0:
                    # print(f"Reached EOS at indices: {new_eos_indices.tolist()}")
                    finished_tags[new_eos_indices] = True
                    
            # speech_end
            diffusion_end_indices = (next_tokens == generation_config.speech_end_id).nonzero(as_tuple=False).squeeze(1)
            if diffusion_end_indices.numel() > 0:
                # Clear tokenizer caches for samples that reached speech end
                print(f"LLMEngine: status=speech end at step={step_id}, next_tokens.shape={next_tokens.shape} {next_tokens.tolist()}, indices: {diffusion_end_indices.tolist()}")
                acoustic_cache.set_to_zero(diffusion_end_indices)
                semantic_cache.set_to_zero(diffusion_end_indices)
            
            # speech_begin
            diffusion_start_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens == generation_config.speech_start_id)]
            if diffusion_start_indices.numel() > 0:
                print(f"LLMEngine: status=speech start at step={step_id}, next_tokens.shape={next_tokens.shape} {next_tokens.tolist()}, indices: {diffusion_start_indices.tolist()}")
                # pdb.set_trace()
                # update attention mask
                for i, sample_idx in enumerate(diffusion_start_indices.tolist()):
                    negative_model_kwargs['attention_mask'][sample_idx, :] = 0
                    negative_model_kwargs['attention_mask'][sample_idx, -1] = 1
                # update past key values
                for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                        negative_model_kwargs['past_key_values'].value_cache)):
                    # Process each non-diffusion sample
                    for sample_idx in diffusion_start_indices.tolist():
                        # Shift cache for this sample (from positive to negative )
                        k_cache[sample_idx, :, -1, :] = k_cache[sample_idx, :, 0, :].clone()
                        v_cache[sample_idx, :, -1, :] = v_cache[sample_idx, :, 0, :].clone()
                # update negative_input_ids
                for sample_idx in diffusion_start_indices.tolist():
                    negative_input_ids[sample_idx, -1] = generation_config.speech_start_id
                # pdb.set_trace()
            
            # Prepare inputs_embeds for next iteration
            # Initialize with default embeddings for all tokens
            next_inputs_embeds = self.model_runner.vibepod.model.get_input_embeddings()(next_tokens).unsqueeze(1)  # [batch_size, 1, hidden_size]
            
            
            diffusion_indices = (next_tokens == generation_config.speech_diffusion_id).nonzero(as_tuple=False).squeeze(1)
            
            if diffusion_indices.numel() > 0:
                print(f"LLMEngine: status=diffusion at step={step_id}, next_tokens.shape={next_tokens.shape} {next_tokens.tolist()}, indices: {diffusion_indices.tolist()}")
                # pdb.set_trace()
                negative_model_inputs = self.model_runner.vibepod.prepare_inputs_for_generation(negative_input_ids, **negative_model_kwargs)
                # Forward negative pass through the model
                if negative_model_inputs['inputs_embeds'] is None and inputs_embeds is not None:
                    negative_model_inputs['inputs_embeds'] = inputs_embeds
                    negative_model_inputs['input_ids'] = None

                negative_outputs = self.model_runner.vibepod(
                    **negative_model_inputs, logits_to_keep=0, return_dict=True, output_attentions=False, output_hidden_states=False,
                )
                negative_model_kwargs = self.model_runner.vibepod._update_model_kwargs_for_generation(
                    negative_outputs, negative_model_kwargs, is_encoder_decoder=False,
                )
                negative_input_ids = torch.cat([negative_input_ids, next_tokens[:, None]], dim=-1)
                
                print("after negative")
                # correct the non-diffusion indices
                # we forward add samples' negative outputs even if 
                #   they are not in diffusion mode to keep the cache consistent
                # So we need to correct the kv cache of non-diffusion samples
                if diffusion_indices.numel() < batch_size - finished_tags.sum():
                    # not all samples are in diffusion mode, but we calculate all negative outputs
                    non_diffusion_indices = torch.arange(batch_size, device=device)[~finished_tags & (next_tokens != generation_config.speech_diffusion_id)]
                    start_indices = correct_cnt[non_diffusion_indices]
                    # print(f"Non-diffusion indices: {non_diffusion_indices.tolist()}, Start indices: {start_indices.tolist()}")

                    # 1. Update attention_mask - need to handle each sample separately
                    seq_len = negative_model_kwargs['attention_mask'].shape[1]
                    for i, (sample_idx, start_idx) in enumerate(zip(non_diffusion_indices.tolist(), start_indices.tolist())):
                        # Shift the attention mask for this sample
                        if start_idx + 1 < seq_len - 1:
                            negative_model_kwargs['attention_mask'][sample_idx, start_idx+1:] = \
                                negative_model_kwargs['attention_mask'][sample_idx, start_idx:-1].clone()
                        negative_model_kwargs['attention_mask'][sample_idx, start_idx] = 0

                    # 2. Update past_key_values
                    for layer_idx, (k_cache, v_cache) in enumerate(zip(negative_model_kwargs['past_key_values'].key_cache, 
                                                                        negative_model_kwargs['past_key_values'].value_cache)):
                        # Process each non-diffusion sample
                        for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                            if start_idx + 1 < k_cache.shape[2] - 1:
                                # Shift cache for this sample
                                k_cache[sample_idx, :, start_idx+1:, :] = k_cache[sample_idx, :, start_idx:-1, :].clone()
                                v_cache[sample_idx, :, start_idx+1:, :] = v_cache[sample_idx, :, start_idx:-1, :].clone()
                    
                    # 3. Update negative_input_ids
                    for sample_idx, start_idx in zip(non_diffusion_indices.tolist(), start_indices.tolist()):
                        if start_idx + 1 < negative_input_ids.shape[1] - 1:
                            negative_input_ids[sample_idx, start_idx+1:] = \
                                negative_input_ids[sample_idx, start_idx:-1].clone()
                                
                    correct_cnt[non_diffusion_indices] += 1
                    print("after non-diffusion correction")

                positive_condition = last_hidden_states[diffusion_indices, -1, :]
                negative_condition = negative_outputs.last_hidden_state[diffusion_indices, -1, :]
                
                speech_latent = self.model_runner.vibepod.sample_speech_tokens(
                    positive_condition,
                    negative_condition,
                    cfg_scale=cfg_scale,
                ).unsqueeze(1)
                
                print("after sample_speech_tokens")
                # Decode acoustic latent to audio using acoustic streaming cache
                scaled_latent = speech_latent / self.model_runner.vibepod.model.speech_scaling_factor - self.model_runner.vibepod.model.speech_bias_factor
                audio_chunk = self.model_runner.vibepod.model.acoustic_tokenizer.decode(
                    scaled_latent,
                    cache=acoustic_cache,  # Use acoustic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                )
                print("after acoustic_tokenizer.decode")
                
                # Store audio chunks for each sample
                for i, sample_idx in enumerate(diffusion_indices):
                    idx = sample_idx.item()
                    # Only append audio chunk if the sample is not finished
                    if not finished_tags[idx]:
                        audio_chunks[idx].append(audio_chunk[i])

                for i_sample, sample_chunks in enumerate(audio_chunks):
                    if sample_chunks:
                        output_path = f"/data/yaoyaochang/code/speech/data/vibepod_outputs/nano_tmpshort_{i_sample}.wav"
                        concatenated_audio = torch.cat(sample_chunks, dim=-1)
                        processor.save_audio(
                            concatenated_audio,
                            output_path=output_path,
                        )
                        print(f"Saved output to {output_path}")
                

                # Encode audio to semantic features using semantic streaming cache
                semantic_features = self.model_runner.vibepod.model.semantic_tokenizer.encode(
                    audio_chunk,
                    cache=semantic_cache,  # Use semantic-specific cache
                    sample_indices=diffusion_indices,
                    use_cache=True,
                    debug=False
                ).mean
                print("after semantic_tokenizer.encode")
                
                # Combine acoustic and semantic features for next input
                acoustic_embed = self.model_runner.vibepod.model.acoustic_connector(speech_latent)
                semantic_embed = self.model_runner.vibepod.model.semantic_connector(semantic_features)
                diffusion_embeds = acoustic_embed + semantic_embed

                # Update embeddings for diffusion indices
                next_inputs_embeds[diffusion_indices] = diffusion_embeds
             
            self.scheduler.postprocess(seqs, next_inputs_embeds, next_tokens)
                   
            # Set inputs_embeds for next iteration
            inputs_embeds = next_inputs_embeds
            
            step_id += 1

            
        # Concatenate audio chunks for each sample
        final_audio_outputs = []
        for sample_chunks in audio_chunks:
            if sample_chunks:
                # Concatenate all chunks along the time dimension (assumed to be the last dimension)
                concatenated_audio = torch.cat(sample_chunks, dim=-1)
                final_audio_outputs.append(concatenated_audio)
            else:
                # If no audio was generated for this sample, append None
                final_audio_outputs.append(None)
                
        return VibePodGenerationOutput(
            sequences=input_ids,
            seqs=seqs,
            speech_outputs=final_audio_outputs if kwargs['return_speech'] else None
        )