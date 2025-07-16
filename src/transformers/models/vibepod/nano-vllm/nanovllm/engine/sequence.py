from copy import copy
from enum import Enum, auto
from itertools import count
import torch

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids : list[int] | None = None, embeddings: list[torch.Tensor] | None = None, sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.embeddings = copy(embeddings)
        self.last_token = token_ids[-1] if token_ids else None
        self.last_embedding = embeddings[-1] if embeddings else None
        self.num_tokens = len(token_ids) if token_ids else len(embeddings)
        self.num_prompt_tokens = len(token_ids) if token_ids else 0 
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        print(f"Sequence: seq_id={self.seq_id}, num_tokens={self.num_tokens}, num_prompt_tokens={self.num_prompt_tokens}, num_cached_tokens={self.num_cached_tokens}, block_table={self.block_table}")

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def set_embeddings(self, embeddings: list[torch.Tensor] | torch.Tensor):
        """
        设置每个token的embedding。
        注意：如果是解码阶段，最后一个token的embedding可能是None。
        """
        # transforme embedding to list if it's a tensor
        if isinstance(embeddings, torch.Tensor):
            embeddings = list(embeddings) # 沿着第0维展开成list
        self.embeddings = embeddings
        if len(embeddings) != len(self.token_ids):
            raise ValueError(f"Length of embeddings {len(embeddings)} does not match length of token_ids {len(self.token_ids)}.")
        self.last_embedding = embeddings[-1]

    def append_token(self, token_id: int, embedding: torch.Tensor | None = None):
        
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        if embedding is not None:
            if self.embeddings is None:
                self.embeddings = []
            self.embeddings.append(embedding)
            assert len(self.embeddings) == len(self.token_ids)
            self.last_embedding = embedding

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
