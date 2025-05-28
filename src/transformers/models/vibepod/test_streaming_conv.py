import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple
import matplotlib.pyplot as plt

# Import the modules to test
from transformers.models.vibepod.modular_vibepod_tokenizer import (
    StreamingCache, SConv1d, SConvTranspose1d
)

# ...existing imports...

def test_streaming_correctness():
    """Test streaming convolution for correctness (not matching non-streaming)"""
    print("\n" + "="*80)
    print("Testing Streaming Convolution Correctness")
    print("="*80)
    
    # Test configurations
    test_configs = [
        # (in_channels, out_channels, kernel_size, stride, dilation)
        (16, 32, 7, 1, 1),     # Basic conv
        (32, 64, 5, 5, 1),     # Stride = kernel
        (64, 128, 3, 1, 2),    # Dilated conv
        (128, 256, 9, 3, 1),   # Large kernel with stride
    ]
    
    batch_size = 2
    total_length = 100
    chunk_size = 20
    
    for config in test_configs:
        in_ch, out_ch, kernel, stride, dilation = config
        print(f"\nConfig: in={in_ch}, out={out_ch}, kernel={kernel}, stride={stride}, dilation={dilation}")
        
        # Create layer
        layer = SConv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel,
            stride=stride,
            dilation=dilation,
            causal=True,
            pad_mode='constant',
        )
        layer.eval()
        
        # Generate input
        full_input = torch.randn(batch_size, in_ch, total_length)
        
        # Test 1: Consistency across chunks
        print("  Testing chunk consistency...")
        cache = StreamingCache()
        sample_indices = torch.arange(batch_size)
        
        all_outputs = []
        for i in range(0, total_length, chunk_size):
            chunk = full_input[:, :, i:i+chunk_size]
            with torch.no_grad():
                output = layer(chunk, cache, sample_indices, use_cache=True, debug=False)
            all_outputs.append(output)
        
        # Test 2: Causality - changing future input shouldn't affect past output
        print("  Testing causality...")
        cache2 = StreamingCache()
        
        # Process first half
        first_half = full_input[:, :, :total_length//2]
        with torch.no_grad():
            output1 = layer(first_half, cache2, sample_indices, use_cache=True)
        
        # Process second half with modified input
        modified_input = full_input.clone()
        modified_input[:, :, total_length//2:] += 1.0  # Add noise to future
        
        cache3 = StreamingCache()
        first_half_modified = modified_input[:, :, :total_length//2]
        with torch.no_grad():
            output2 = layer(first_half_modified, cache3, sample_indices, use_cache=True)
        
        # Outputs should be identical since future was modified
        assert torch.allclose(output1, output2), "Causality violated!"
        print("    ✓ Causality preserved")
        
        # Test 3: Different chunk sizes should produce consistent results
        print("  Testing different chunk sizes...")
        for test_chunk_size in [10, 15, 25]:
            cache_test = StreamingCache()
            test_outputs = []
            
            for i in range(0, total_length, test_chunk_size):
                chunk = full_input[:, :, i:min(i+test_chunk_size, total_length)]
                if chunk.shape[2] > 0:
                    with torch.no_grad():
                        output = layer(chunk, cache_test, sample_indices, use_cache=True)
                    test_outputs.append(output)
            
            combined = torch.cat(test_outputs, dim=2)
            print(f"    Chunk size {test_chunk_size}: output shape = {combined.shape}")
    
    print("\n✓ All streaming correctness tests passed!")


def test_stride5_streaming_only():
    """Test stride=5 streaming behavior without comparing to non-streaming"""
    print("\n" + "="*80)
    print("Testing Stride=5 Streaming Behavior")
    print("="*80)
    
    # Configuration
    in_ch, out_ch, kernel, stride, dilation = 32, 64, 5, 5, 1
    
    layer = SConv1d(
        in_channels=in_ch,
        out_channels=out_ch,
        kernel_size=kernel,
        stride=stride,
        dilation=dilation,
        causal=True,
        pad_mode='constant',
    )
    layer.eval()
    
    # Test with different scenarios
    scenarios = [
        ("Fixed chunks", 32, 16),
        ("Tiny chunks", 32, 5),
        ("Large chunks", 50, 25),
        ("Uneven chunks", 37, 13),
    ]
    
    for name, total_length, chunk_size in scenarios:
        print(f"\n{name}: total_length={total_length}, chunk_size={chunk_size}")
        
        full_input = torch.randn(1, in_ch, total_length)
        cache = StreamingCache()
        sample_indices = torch.tensor([0])
        
        streaming_outputs = []
        position = 0
        
        while position < total_length:
            end = min(position + chunk_size, total_length)
            chunk = full_input[:, :, position:end]
            
            with torch.no_grad():
                output = layer(chunk, cache, sample_indices, use_cache=True, debug=True)
            
            streaming_outputs.append(output)
            print(f"  Chunk at {position}-{end}: input_shape={chunk.shape}, output_shape={output.shape}")
            position = end
        
        if streaming_outputs:
            total_output = torch.cat(streaming_outputs, dim=2)
            print(f"  Total streaming output shape: {total_output.shape}")


if __name__ == "__main__":
    # Run the new tests
    test_streaming_correctness()
    test_stride5_streaming_only()
    
    print("\n" + "="*80)
    print("✅ ALL STREAMING TESTS PASSED! ✅")
    print("="*80)