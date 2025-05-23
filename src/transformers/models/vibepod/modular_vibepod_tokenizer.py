# Copyright (c) Meta Platforms, Inc. and affiliates.
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
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

import math
import typing as tp
from functools import partial
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...configuration_utils import PretrainedConfig
from ...utils import logging
from ...modeling_utils import PreTrainedModel

from .configuration_vibepod import VibePodAcousticTokenizerConfig, VibePodSemanticTokenizerConfig

logger = logging.get_logger(__name__)


# Normalization modules
class ConvLayerNorm(nn.LayerNorm):
    """
    Convolution-friendly LayerNorm that moves channels to last dimensions
    before running the normalization and moves them back to original position right after.
    """
    def __init__(self, normalized_shape: tp.Union[int, tp.List[int], torch.Size], **kwargs):
        super().__init__(normalized_shape, **kwargs)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        x = nn.functional.layer_norm(x.float(), self.normalized_shape, self.weight.float(), self.bias.float(), self.eps).type_as(x) 
        x = x.transpose(1, 2)  # b t ... -> b ... t
        return x


# Could use LlamaRMSNorm
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            weight_shape = (dim,) if weight_shape is None else weight_shape
            self.weight = nn.Parameter(torch.ones(weight_shape))
        else:
            self.register_parameter('weight', None)

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        return output

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}, elementwise_affine={self.elementwise_affine}'


class ConvRMSNorm(RMSNorm):
    def __init__(self, dim: int, eps: float = 1e-5, elementwise_affine=True, weight_shape=None):
        super().__init__(dim, eps, elementwise_affine, weight_shape)

    def forward(self, x):
        x = x.transpose(1, 2)  # b ... t -> b t ...
        output = self._norm(x.float()).type_as(x)
        if self.weight is not None:
            output = output * self.weight
        output = output.transpose(1, 2)  # b t ... -> b ... t
        return output


# Convolutional layers and utilities
CONV_NORMALIZATIONS = frozenset(['none', 'weight_norm', 'spectral_norm',
                                'time_layer_norm', 'layer_norm', 'time_group_norm'])


def apply_parametrization_norm(module: nn.Module, norm: str = 'none') -> nn.Module:
    assert norm in CONV_NORMALIZATIONS
    if norm == 'weight_norm':
        return nn.utils.weight_norm(module)
    elif norm == 'spectral_norm':
        return nn.utils.spectral_norm(module)
    else:
        # We already check was in CONV_NORMALIZATION, so any other choice
        # doesn't need reparametrization.
        return module


def get_norm_module(module: nn.Module, causal: bool = False, norm: str = 'none', **norm_kwargs) -> nn.Module:
    """Return the proper normalization module. If causal is True, this will ensure the returned
    module is causal, or return an error if the normalization doesn't support causal evaluation.
    """
    assert norm in CONV_NORMALIZATIONS
    if norm == 'layer_norm':
        assert isinstance(module, nn.modules.conv._ConvNd)
        return ConvLayerNorm(module.out_channels, **norm_kwargs)
    elif norm == 'time_group_norm':
        if causal:
            raise ValueError("GroupNorm doesn't support causal evaluation.")
        assert isinstance(module, nn.modules.conv._ConvNd)
        return nn.GroupNorm(1, module.out_channels, **norm_kwargs)
    else:
        return nn.Identity()


def get_extra_padding_for_conv1d(x: torch.Tensor, kernel_size: int, stride: int,
                                padding_total: int = 0) -> int:
    """Calculate extra padding needed for convolution to have the same output length"""
    length = x.shape[-1]
    n_frames = (length - kernel_size + padding_total) / stride + 1
    ideal_length = (math.ceil(n_frames) - 1) * stride + (kernel_size - padding_total)
    return ideal_length - length


def pad1d(x: torch.Tensor, paddings: tp.Tuple[int, int], mode: str = 'zero', value: float = 0.):
    """Pad 1D input with handling for small inputs in reflect mode"""
    length = x.shape[-1]
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    if mode == 'reflect':
        max_pad = max(padding_left, padding_right)
        extra_pad = 0
        if length <= max_pad:
            extra_pad = max_pad - length + 1
            x = F.pad(x, (0, extra_pad))
        padded = F.pad(x, paddings, mode, value)
        end = padded.shape[-1] - extra_pad
        return padded[..., :end]
    else:
        return F.pad(x, paddings, mode, value)


def unpad1d(x: torch.Tensor, paddings: tp.Tuple[int, int]):
    """Remove padding from x, handling properly zero padding. Only for 1d!"""
    padding_left, padding_right = paddings
    assert padding_left >= 0 and padding_right >= 0, (padding_left, padding_right)
    assert (padding_left + padding_right) <= x.shape[-1]
    end = x.shape[-1] - padding_right
    return x[..., padding_left: end]


class NormConv1d(nn.Module):
    """Wrapper around Conv1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.conv = apply_parametrization_norm(nn.Conv1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.conv, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        return x


class NormConvTranspose1d(nn.Module):
    """Wrapper around ConvTranspose1d and normalization applied to this conv"""
    def __init__(self, *args, causal: bool = False, norm: str = 'none',
                norm_kwargs: tp.Dict[str, tp.Any] = {}, **kwargs):
        super().__init__()
        self.convtr = apply_parametrization_norm(nn.ConvTranspose1d(*args, **kwargs), norm)
        self.norm = get_norm_module(self.convtr, causal, norm, **norm_kwargs)
        self.norm_type = norm

    def forward(self, x):
        x = self.convtr(x)
        x = self.norm(x)
        return x


class SConv1d(nn.Module):
    """Conv1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, dilation: int = 1,
                groups: int = 1, bias: bool = True, causal: bool = False,
                norm: str = 'none', norm_kwargs: tp.Dict[str, tp.Any] = {},
                pad_mode: str = 'reflect'):
        super().__init__()
        self.conv = NormConv1d(in_channels, out_channels, kernel_size, stride,
                            dilation=dilation, groups=groups, bias=bias, causal=causal,
                            norm=norm, norm_kwargs=norm_kwargs)
        self.causal = causal
        self.pad_mode = pad_mode

        # use for streaming
        self.streaming = False 
        self.init_streaming_state = False
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.stride = stride
        # Compute effective kernel size considering dilation
        self.effective_kernel_size = (kernel_size - 1) * dilation + 1
        # Compute total padding required for causal convolution
        self.padding_total = (kernel_size - 1) * dilation - (stride - 1)
        self.state = None
    
    def set_streaming(self, streaming: bool, init_streaming_state: bool = False):
        """Set streaming mode and initialize state if required"""
        self.streaming = streaming
        self.init_streaming_state = init_streaming_state

    def reset_state(self):
        """Reset streaming state"""
        self.state = None
    
    def init_state(self, batch_size, device, dtype=torch.float32):
        """Initialize state buffer for streaming convolution"""
        if self.causal:
            # For causal convolution, we need to keep padding_total samples
            return torch.zeros(
                batch_size,
                self.conv.conv.in_channels,
                self.padding_total,
                device=device,
                dtype=dtype
            )
        else:
            return None  # Non-causal streaming not fully supported yet
        
    def forward(self, x):
        if not self.streaming:
            B, C, T = x.shape
            kernel_size = self.conv.conv.kernel_size[0]
            stride = self.conv.conv.stride[0]
            dilation = self.conv.conv.dilation[0]
            padding_total = (kernel_size - 1) * dilation - (stride - 1)
            extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)
            if self.causal:
                # Left padding for causal
                if self.pad_mode == 'constant':
                    x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode, value=0)
                else:
                    x = pad1d(x, (padding_total, extra_padding), mode=self.pad_mode)
            else:
                # Asymmetric padding required for odd strides
                padding_right = padding_total // 2
                padding_left = padding_total - padding_right
                x = pad1d(x, (padding_left, padding_right + extra_padding), mode=self.pad_mode)
            return self.conv(x)
        else:
            assert self.causal, "Streaming mode is only supported for causal convolutions"

            batch_size = x.size(0)
            if self.init_streaming_state:
                self.state = self.init_state(batch_size, x.device, x.dtype)

            # Get key parameters
            kernel_size = self.kernel_size
            stride = self.stride
            dilation = self.dilation
            padding_total = self.padding_total
            extra_padding = get_extra_padding_for_conv1d(x, kernel_size, stride, padding_total)

            input_with_state = torch.cat([self.state, x], dim=2)
        
            # Apply padding (only on the right for streaming mode)
            padded_input = F.pad(input_with_state, (0, extra_padding), mode='constant')
            
            # Apply the convolution
            output = self.conv(padded_input)
            
            # Update state for next chunk - we need padding_total samples
            # We take samples from the input, not the padded version
            new_state_start = max(0, input_with_state.size(2) - padding_total)
            self.state = input_with_state[:, :, new_state_start:].detach()
            
            # For causal convolutions, all the valid outputs correspond to the new input,
            # since we've provided the needed left context via the state
            return output


class SConvTranspose1d(nn.Module):
    """ConvTranspose1d with built-in handling of asymmetric or causal padding and normalization."""
    def __init__(self, in_channels: int, out_channels: int,
                kernel_size: int, stride: int = 1, causal: bool = False,
                norm: str = 'none', trim_right_ratio: float = 1.,
                norm_kwargs: tp.Dict[str, tp.Any] = {}, bias: bool = True):
        super().__init__()
        self.convtr = NormConvTranspose1d(in_channels, out_channels, kernel_size, stride,
                                        causal=causal, norm=norm, norm_kwargs=norm_kwargs, bias=bias)
        self.causal = causal
        self.trim_right_ratio = trim_right_ratio
        assert self.causal or self.trim_right_ratio == 1., \
            "`trim_right_ratio` != 1.0 only makes sense for causal convolutions"
        assert self.trim_right_ratio >= 0. and self.trim_right_ratio <= 1.

        # use for streaming
        self.streaming = False 
        self.init_streaming_state = False
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding_total = kernel_size - stride
        self.position = 0
        self.input_buffer = None
    
    def set_streaming(self, streaming: bool, init_streaming_state: bool = False):
        """Set streaming mode and initialize state if required"""
        self.streaming = streaming
        self.init_streaming_state = init_streaming_state

    def reset_state(self):
        """Reset streaming state buffers"""
        self.input_buffer = None
        self.position = 0
    
    def process_nonstreaming(self, x):
        """Process in non-streaming mode"""
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride

        y = self.convtr(x)

        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            y = unpad1d(y, (padding_left, padding_right))
        return y
    
    def get_padding(self):
        """Calculate padding values based on configuration"""
        kernel_size = self.convtr.convtr.kernel_size[0]
        stride = self.convtr.convtr.stride[0]
        padding_total = kernel_size - stride
        
        if self.causal:
            padding_right = math.ceil(padding_total * self.trim_right_ratio)
            padding_left = padding_total - padding_right
        else:
            padding_right = padding_total // 2
            padding_left = padding_total - padding_right
            
        return padding_left, padding_right
    
    def forward(self, x):
        if not self.streaming:
            return self.process_nonstreaming(x)
        
        batch_size = x.size(0)
        if self.init_streaming_state:
            self.reset_state()
        
        if self.input_buffer is None:
            self.input_buffer = x
        else:
            # Concatenate new input with saved buffer
            self.input_buffer = torch.cat([self.input_buffer, x], dim=2)
            
        # Calculate how much output we should produce
        output_length = x.size(2) * self.stride
        
        # Check if we have enough input data to produce the expected output
        # For transposed conv, we need to ensure we have enough context to the left
        min_input_needed = math.ceil(output_length / self.stride)
        
        if self.input_buffer.size(2) >= min_input_needed:
            # Process the entire buffer
            with torch.no_grad():
                # Apply the transposed convolution to the entire buffer
                full_output = self.process_nonstreaming(self.input_buffer)
            
            # Determine which part of the output corresponds to the current input chunk
            start_pos = max(0, full_output.size(2) - output_length)
            
            # Extract the relevant part of the output
            output = full_output[:, :, start_pos:]
            
            # Keep only the minimum required context for the next chunk
            keep_size = min(self.kernel_size - 1, self.input_buffer.size(2))
            self.input_buffer = self.input_buffer[:, :, -keep_size:] if keep_size > 0 else None
            
            return output
        else:
            # Not enough data yet, return empty tensor matching expected shape
            return torch.zeros(batch_size, self.convtr.convtr.out_channels, 0, device=x.device)


# FFN and Convolution layers
class FFN(nn.Module):
    def __init__(
        self,
        embed_dim,
        ffn_dim,
        bias=False,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.linear1 = nn.Linear(self.embed_dim, ffn_dim, bias=bias) 
        self.gelu = nn.GELU()
        self.linear2 = nn.Linear(ffn_dim, self.embed_dim, bias=bias)

    def forward(self, x):
        x = self.linear1(x)
        x = self.gelu(x)
        x = self.linear2(x)
        return x


class Convlayer(nn.Module):
    def __init__(
            self, 
            in_channels, 
            out_channels, 
            kernel_size, 
            stride=1, 
            dilation=1, 
            groups=1, 
            bias=True, 
            pad_mode='zeros', 
            norm='weight_norm', 
            causal=True, 
        ):
        super().__init__()
        self.conv = SConv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, 
                           groups=groups, bias=bias, pad_mode=pad_mode, norm=norm, causal=causal)

    def forward(self, x):
        return self.conv(x)

class Block1D(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=0., mixer_layer='conv',  
                layer_scale_init_value=1e-6, **kwargs):
        super().__init__()
        
        if kwargs.get('layernorm', 'LN') == 'LN':
            self.norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvLayerNorm(dim, eps=kwargs.get('eps', 1e-6))               
        elif kwargs.get('layernorm', 'RMSNorm') == 'RMSNorm':
            self.norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))
            self.ffn_norm = ConvRMSNorm(dim, eps=kwargs.get('eps', 1e-6))

        if mixer_layer == 'conv':
            self.mixer = Convlayer(dim, dim, groups=kwargs.get('groups', 1),
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        elif mixer_layer == 'depthwise_conv':
            self.mixer = Convlayer(dim, dim, groups=dim,
                                kernel_size=kernel_size, 
                                pad_mode=kwargs.get('pad_mode', 'reflect'), 
                                norm=kwargs.get('norm', 'none'), 
                                causal=kwargs.get('causal', True), 
                                bias=kwargs.get('bias', True),
                                )
        else:
            raise ValueError(f"Unsupported mixer layer: {mixer_layer}")
        
        self.ffn = FFN(
            dim, 
            kwargs.get('ffn_expansion', 4) * dim, 
            bias=kwargs.get('bias', False),
        )
        self.drop_path = nn.Identity() if drop_path <= 0. else nn.modules.DropPath(drop_path)

        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
            self.ffn_gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        else:
            self.gamma = None
            self.ffn_gamma = None

    def forward(self, x):
        # mixer
        residual = x
        x = self.norm(x)
        x = self.mixer(x)
        if self.gamma is not None:
            x = x * self.gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        # ffn
        residual = x
        x = self.ffn_norm(x)
        x = x.permute(0, 2, 1)
        x = self.ffn(x)
        x = x.permute(0, 2, 1)
        if self.ffn_gamma is not None:
            x = x * self.ffn_gamma.unsqueeze(-1)
        x = residual + self.drop_path(x)

        return x


class TokenizerEncoder(nn.Module):
    """
    Encoder component for the VibePod tokenizer that converts audio to latent representations.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.channels = config.channels
        self.dimension = config.dimension
        self.n_filters = config.n_filters
        self.ratios = list(reversed(config.ratios))
        self.depths = config.depths
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)
        
        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and intermediate downsampling conv layers
        stem = nn.Sequential(
                SConv1d(self.channels, self.n_filters, kernel_size, norm=norm, norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.downsample_layers = nn.ModuleList()
        self.downsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** i)
            out_ch = self.n_filters * (2 ** (i + 1))
            downsample_layer = nn.Sequential(
                SConv1d(in_ch, out_ch, kernel_size=self.ratios[i] * 2, stride=self.ratios[i], causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
            )
            self.downsample_layers.append(downsample_layer)

        # configure the transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )
        
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0

        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** i)
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]
        
        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.dimension, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
        
        self.streaming = False
        self.init_streaming_state = False

    def forward_features(self, x, streaming=False, init_streaming_state=False):
        # Set global attributes
        if streaming != self.streaming or init_streaming_state != self.init_streaming_state:
            # update the streaming and init_streaming_state attributes
            self.streaming = streaming
            self.init_streaming_state = init_streaming_state
            for layer in self.modules():
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    if hasattr(layer, 'set_streaming'):
                        layer.set_streaming(streaming, init_streaming_state)

        for i in range(len(self.depths)):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x)

    def forward(self, x, streaming=False, init_streaming_state=False):
        x = self.forward_features(x, streaming=streaming, init_streaming_state=init_streaming_state)
        x = self.head(x)
        return x


class TokenizerDecoder(nn.Module):
    """
    Decoder component for the VibePod tokenizer that converts latent representations back to audio.
    
    Args:
        config: Configuration object with model parameters
    """
    def __init__(self, config):
        super().__init__()
        
        # Extract parameters from config
        self.dimension = config.dimension
        self.channels = config.channels
        self.n_filters = config.n_filters
        self.ratios = config.ratios
        
        # IMPORTANT CHANGE: Don't reverse depths again since they're already reversed in VibePodAcousticTokenizerModel
        self.depths = config.depths  # Changed from list(reversed(config.depths))
        
        self.n_residual_layers = getattr(config, "n_residual_layers", 1)
        self.hop_length = np.prod(self.ratios)
        self.causal = config.causal
        
        # Additional config parameters with defaults
        kernel_size = getattr(config, "kernel_size", 7)
        last_kernel_size = getattr(config, "last_kernel_size", 7)
        norm = getattr(config, "norm", "none")
        norm_params = getattr(config, "norm_params", {})
        pad_mode = getattr(config, "pad_mode", "reflect")
        bias = getattr(config, "bias", True)
        layernorm = getattr(config, "layernorm", "LN")
        layernorm_eps = getattr(config, "layernorm_eps", 1e-6)
        trim_right_ratio = getattr(config, "trim_right_ratio", 1.0)
        layernorm_elementwise_affine = getattr(config, "layernorm_elementwise_affine", True)
        drop_path_rate = getattr(config, "drop_path_rate", 0.0)
        mixer_layer = getattr(config, "mixer_layer", "conv")
        layer_scale_init_value = getattr(config, "layer_scale_init_value", 0)
        disable_last_norm = getattr(config, "disable_last_norm", False)

        # determine the norm type based on layernorm
        if layernorm == 'LN':
            norm_type = ConvLayerNorm
        elif layernorm == 'RMSNorm':
            norm_type = partial(ConvRMSNorm, elementwise_affine=layernorm_elementwise_affine)
        else:
            raise ValueError(f"Unsupported norm type: {layernorm}")
        
        # stem and upsampling layers
        stem = nn.Sequential(
                SConv1d(self.dimension, self.n_filters * 2 ** (len(self.depths) - 1), kernel_size, norm=norm, 
                        norm_kwargs=norm_params, causal=self.causal, pad_mode=pad_mode, bias=bias),
            )
        
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(stem)
        for i in range(len(self.ratios)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            out_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i - 1))
            upsample_layer = nn.Sequential(
                SConvTranspose1d(in_ch, out_ch,
                                kernel_size=self.ratios[i] * 2, stride=self.ratios[i],
                                norm=norm, norm_kwargs=norm_params, bias=bias,
                                causal=self.causal, trim_right_ratio=trim_right_ratio),
            )
            self.upsample_layers.append(upsample_layer)

        # configure transformer blocks
        layer_type = partial(
            Block1D,
            mixer_layer=mixer_layer,
            layernorm=layernorm,
            eps=layernorm_eps,
            causal=self.causal,
            pad_mode=pad_mode,
            norm=norm,
            bias=bias,
            layer_scale_init_value=layer_scale_init_value,
        )

        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))] 
        cur = 0
        
        # Create stages in the same order as the original model
        for i in range(len(self.depths)):
            in_ch = self.n_filters * (2 ** (len(self.depths) - 1 - i))
            stage = nn.Sequential(
                *[layer_type(dim=in_ch, drop_path=dp_rates[cur + j]) for j in range(self.depths[i])]
            )
            self.stages.append(stage)
            cur += self.depths[i]

        if not disable_last_norm:
            self.norm = norm_type(in_ch, eps=layernorm_eps)
        else:
            self.norm = nn.Identity()
        self.head = SConv1d(in_ch, self.channels, kernel_size=last_kernel_size, causal=self.causal, pad_mode=pad_mode, norm=norm, bias=bias)
        
        self.streaming = False
        self.init_streaming_state = False

    def forward_features(self, x, streaming=False, init_streaming_state=False):
        # pdb.set_trace()
        # Set global attributes
        if streaming != self.streaming or init_streaming_state != self.init_streaming_state:
            # update the streaming and init_streaming_state attributes
            self.streaming = streaming
            self.init_streaming_state = init_streaming_state
            for layer in self.modules():
                if isinstance(layer, (SConv1d, SConvTranspose1d)):
                    if hasattr(layer, 'set_streaming'):
                        # print(f"Set streaming for {layer.__class__.__name__}: {layer}")
                        layer.set_streaming(streaming, init_streaming_state)

        for i in range(len(self.depths)):
            x = self.upsample_layers[i](x)
            x = self.stages[i](x)

        return self.norm(x)
    
    def forward(self, x, streaming=False, init_streaming_state=False):
        x = self.forward_features(x, streaming=streaming, init_streaming_state=init_streaming_state)
        x = self.head(x)
        return x
    
@dataclass
class VibePodTokenizerEncoderOutput:
    """
    Output of VibePod tokenizer encoder, representing a Gaussian distribution with fixed variance.
    
    Args:
        mean (`torch.FloatTensor`): The mean parameters of the distribution.
        std (`float` or `torch.FloatTensor`): Fixed standard deviation value.
    """
    mean: torch.Tensor
    std: Optional[Union[float, torch.Tensor]] = None
    
    def sample(self, dist_type='fix'):
        """
        Sample from the distribution.
        
        Args:
            dist_type (`str`): Sampling method, either 'fix' or 'gaussian'.
                
        Returns:
            `torch.FloatTensor`: Sampled values.
            `torch.FloatTensor` (optional): Standard deviation used (only when dist_type='gaussian').
        """
        if dist_type == 'fix':
            x = self.mean + self.std * torch.randn_like(self.mean)
            return x, self.std
        elif dist_type == 'gaussian':
            batch_size = self.mean.size(0)
            value = self.std / 0.8
            std = torch.randn(batch_size, device=self.mean.device, dtype=self.mean.dtype) * value

            while std.dim() < self.mean.dim():
                std = std.unsqueeze(-1)

            x = self.mean + std * torch.randn_like(self.mean)
            return x, std
        else:
            return self.mean, self.std

    def kl(self):
        """Compute KL divergence between this distribution and a standard normal."""
        target = torch.zeros_like(self.mean)
        return F.mse_loss(self.mean, target, reduction='none')

    def mode(self):
        """Return the distribution mode (which is the mean for Gaussian)."""
        return self.mean
    
class VibePodAcousticTokenizerModel(PreTrainedModel):
    """VibePod speech tokenizer model combining encoder and decoder for acoustic tokens"""
    
    config_class = VibePodAcousticTokenizerConfig
    base_model_prefix = "vibepod_acoustic_tokenizer"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Register fix_std as a buffer (similar to your original implementation)
        self.register_buffer('fix_std', torch.tensor(config.fix_std))
        self.std_dist_type = getattr(config, "std_dist_type", "fix")
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
            
        # Parse decoder depths if provided
        if config.decoder_depths is not None and isinstance(config.decoder_depths, str):
            decoder_depths = [int(d) for d in config.decoder_depths.split('-')]
        else:
            # Default: use reversed encoder depths if decoder_depths is None
            decoder_depths = list(reversed(encoder_depths))
        
        # Create encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        
        # Create decoder config
        decoder_config = copy.deepcopy(config)
        decoder_config.dimension = config.vae_dim
        decoder_config.n_filters = config.decoder_n_filters
        decoder_config.ratios = config.decoder_ratios
        decoder_config.depths = decoder_depths
        decoder_config.norm = config.conv_norm
        decoder_config.pad_mode = config.pad_mode
        decoder_config.bias = config.conv_bias
        decoder_config.layernorm_eps = config.layernorm_eps
        decoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        decoder_config.mixer_layer = config.mixer_layer
        decoder_config.layer_scale_init_value = config.layer_scale_init_value
        decoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        self.decoder = TokenizerDecoder(decoder_config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def encode(self, audio, streaming=False, init_streaming_state=False):
        """Convert audio to latent representations"""
        latents = self.encoder(audio, streaming=streaming, init_streaming_state=init_streaming_state)
        return VibePodTokenizerEncoderOutput(mean=latents.permute(0, 2, 1), std=self.fix_std)
    
    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        """Sample from the encoder output distribution"""
        dist_type = dist_type or self.std_dist_type
    
        if dist_type == 'fix':
            return encoder_output.sample(dist_type='fix')
        elif dist_type == 'gaussian':
            return encoder_output.sample(dist_type='gaussian')
        else:
            raise ValueError(f"Unsupported dist_type: {dist_type}, expected 'fix' or 'gaussian'")
    
    @torch.no_grad()
    def decode(self, latents, streaming=False, init_streaming_state=False):
        """Convert latent representations back to audio"""
        # audio = self.decoder(latents, streaming=streaming, init_streaming_state=init_streaming_state)
        # return audio
        if latents.shape[1] == self.config.vae_dim:
            pass
        else:
            latents = latents.permute(0, 2, 1)

        audio = self.decoder(latents, streaming=streaming, init_streaming_state=init_streaming_state)
        return audio

    def forward(self, audio, streaming=False, init_streaming_state=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, streaming=streaming, init_streaming_state=init_streaming_state)
        sampled_latents, _ = self.sampling(encoder_output)
        reconstructed = self.decode(sampled_latents, streaming=streaming, init_streaming_state=init_streaming_state)
        return reconstructed, sampled_latents


class VibePodSemanticTokenizerModel(PreTrainedModel):
    """VibePod speech tokenizer model with only encoder for semantic tokens"""
    
    config_class = VibePodSemanticTokenizerConfig
    base_model_prefix = "vibepod_semantic_tokenizer"
    
    def __init__(self, config):
        super().__init__(config)
        
        # Parse encoder depths
        if isinstance(config.encoder_depths, str):
            encoder_depths = [int(d) for d in config.encoder_depths.split('-')]
        else:
            encoder_depths = config.encoder_depths
        
        # Create encoder config
        encoder_config = copy.deepcopy(config)
        encoder_config.dimension = config.vae_dim
        encoder_config.n_filters = config.encoder_n_filters
        encoder_config.ratios = config.encoder_ratios
        encoder_config.depths = encoder_depths
        encoder_config.norm = config.conv_norm
        encoder_config.pad_mode = config.pad_mode
        encoder_config.bias = config.conv_bias
        encoder_config.layernorm_eps = config.layernorm_eps
        encoder_config.layernorm_elementwise_affine = config.layernorm_elementwise_affine
        encoder_config.mixer_layer = config.mixer_layer
        encoder_config.layer_scale_init_value = config.layer_scale_init_value
        encoder_config.disable_last_norm = config.disable_last_norm
        
        # Initialize encoder and decoder
        self.encoder = TokenizerEncoder(encoder_config)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for the model"""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv1d):
            nn.init.normal_(module.weight, std=self.config.weight_init_value)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    @torch.no_grad()
    def encode(self, audio, streaming=False, init_streaming_state=False):
        """Convert audio to latent representations"""
        latents = self.encoder(audio, streaming=streaming, init_streaming_state=init_streaming_state)
        return VibePodTokenizerEncoderOutput(mean=latents.permute(0, 2, 1))
    
    @torch.no_grad()
    def sampling(self, encoder_output, dist_type=None):
        """Sample from the encoder output distribution"""
        return encoder_output.sample(dist_type='none')

    def forward(self, audio, streaming=False, init_streaming_state=False):
        """Full forward pass: encode audio to latents, then decode back to audio"""
        encoder_output = self.encode(audio, streaming=streaming, init_streaming_state=init_streaming_state)
        sampled_latents, _ = self.sampling(encoder_output, dist_type='none')
        return None, sampled_latents

__all__ = [
    "VibePodAcousticTokenizerModel",
    "VibePodSemanticTokenizerModel",
]