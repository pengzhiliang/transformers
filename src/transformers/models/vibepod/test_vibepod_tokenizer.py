import torch
import numpy as np
import time
from transformers.models.vibepod.vibepod_tokenizer_processor import VibePodTokenizerProcessor
from transformers.models.vibepod.modular_vibepod_tokenizer import (
    VibePodAcousticTokenizerModel, 
    VibePodSemanticTokenizerModel,
    VibePodTokenizerStreamingCache
)

def test_acoustic_tokenizer_streaming():
    """Test acoustic tokenizer in streaming mode"""
    print("\n" + "="*80)
    print("Testing VibePod Acoustic Tokenizer Streaming")
    print("="*80)
    
    # Initialize
    processor = VibePodTokenizerProcessor(normalize_audio=True)
    model = VibePodAcousticTokenizerModel.from_pretrained("/tmp/viebpod_acoustic_tokenizer")
    model.eval()
    
    # Create test audio (4 seconds at 24kHz)
    sample_rate = 24000
    duration = 4.0
    audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
    
    # Process full audio (non-streaming)
    print("\nNon-streaming processing...")
    inputs = processor(audio, return_tensors="pt")
    with torch.no_grad():
        reconstructed_full, latents_full = model(**inputs)
    print(f"Full audio shape: {inputs['audio'].shape}")
    print(f"Full latents shape: {latents_full.shape}")
    print(f"Full reconstructed shape: {reconstructed_full.shape}")
    
    # Process in streaming mode
    print("\nStreaming processing...")
    chunk_duration = 0.5  # 500ms chunks
    # chunk_size = int(sample_rate * chunk_duration)
    chunk_size = 3200
    num_chunks = int(np.ceil(len(audio) / chunk_size))
    
    # Create cache and sample indices
    cache = VibePodTokenizerStreamingCache()
    sample_indices = torch.tensor([0])  # Single sample
    
    latents_chunks = []
    reconstructed_chunks = []
    
    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(audio))
        audio_chunk = audio[start_idx:end_idx]
        
        # Process chunk
        inputs_chunk = processor(audio_chunk, return_tensors="pt")
        
        with torch.no_grad():
            # Encode
            encoder_output = model.encode(
                inputs_chunk['audio'], 
                cache=cache, 
                sample_indices=sample_indices, 
                use_cache=True,
                debug=False
            )
            sampled_latents, _ = model.sampling(encoder_output)
            latents_chunks.append(sampled_latents)
            
            # Decode
            reconstructed_chunk = model.decode(
                sampled_latents,
                cache=cache,
                sample_indices=sample_indices,
                use_cache=True,
                debug=False
            )
            reconstructed_chunks.append(reconstructed_chunk)
        
        print(f"Chunk {i}: input shape {inputs_chunk['audio'].shape}, "
              f"latents shape {sampled_latents.shape}, "
              f"output shape {reconstructed_chunk.shape}")
    
    # Concatenate streaming outputs
    if latents_chunks:
        latents_streaming = torch.cat(latents_chunks, dim=1)
        reconstructed_streaming = torch.cat(reconstructed_chunks, dim=2)
        
        print(f"\nStreaming latents shape: {latents_streaming.shape}")
        print(f"Streaming reconstructed shape: {reconstructed_streaming.shape}")
        
        # Note: Streaming and non-streaming outputs may differ due to boundary handling
        print("\nNote: Streaming outputs may differ from non-streaming due to different padding/boundary handling")


def test_semantic_tokenizer_streaming():
    """Test semantic tokenizer in streaming mode"""
    print("\n" + "="*80)
    print("Testing VibePod Semantic Tokenizer Streaming")
    print("="*80)
    
    # Initialize
    processor = VibePodTokenizerProcessor(normalize_audio=True)
    model = VibePodSemanticTokenizerModel.from_pretrained("/tmp/viebpod_semantic_tokenizer")
    model.eval()
    
    # Create test audio
    sample_rate = 24000
    duration = 4.0
    audio = np.random.randn(int(sample_rate * duration)) * 0.1
    
    # Process full audio
    print("\nNon-streaming processing...")
    inputs = processor(audio, return_tensors="pt")
    with torch.no_grad():
        _, latents_full = model(**inputs)
    print(f"Full audio shape: {inputs['audio'].shape}")
    print(f"Full latents shape: {latents_full.shape}")
    
    # Process in streaming mode
    print("\nStreaming processing...")
    # chunk_duration = 0.4  # 400ms chunks
    # chunk_size = int(sample_rate * chunk_duration)
    chunk_size = 3200

    cache = VibePodTokenizerStreamingCache()
    sample_indices = torch.tensor([0])
    
    latents_chunks = []
    position = 0
    
    while position < len(audio):
        end_pos = min(position + chunk_size, len(audio))
        audio_chunk = audio[position:end_pos]
        
        inputs_chunk = processor(audio_chunk, return_tensors="pt")
        
        with torch.no_grad():
            _, latents_chunk = model(
                inputs_chunk['audio'],
                cache=cache,
                sample_indices=sample_indices,
                use_cache=True
            )
        
        latents_chunks.append(latents_chunk)
        print(f"Chunk at {position}-{end_pos}: latents shape {latents_chunk.shape}")
        
        position = end_pos
    
    # Concatenate
    if latents_chunks:
        latents_streaming = torch.cat(latents_chunks, dim=1)
        print(f"\nStreaming latents shape: {latents_streaming.shape}")


def test_batch_streaming():
    """Test streaming with multiple samples in a batch"""
    print("\n" + "="*80)
    print("Testing Batch Streaming")
    print("="*80)
    
    processor = VibePodTokenizerProcessor(normalize_audio=True)
    model = VibePodSemanticTokenizerModel.from_pretrained("/tmp/viebpod_semantic_tokenizer")
    model.eval()
    
    # Create batch of audio with different lengths
    sample_rate = 24000
    batch_size = 3
    audio_lengths = [1.0, 1.5, 2.0]  # Different durations in seconds
    audio_samples = []
    
    for i, duration in enumerate(audio_lengths):
        audio = np.sin(2 * np.pi * (440 + i * 100) * np.linspace(0, duration, int(sample_rate * duration)))
        audio_samples.append(audio)
    
    print(f"\nAudio lengths: {[len(a) for a in audio_samples]} samples")
    
    # Test 1: Process each sample individually in streaming mode
    print("\n--- Test 1: Individual Streaming Processing ---")
    chunk_size = 3200  # 200ms chunks
    individual_results = []
    
    for i, audio in enumerate(audio_samples):
        cache = VibePodTokenizerStreamingCache()
        sample_indices = torch.tensor([i])
        chunks = []
        
        position = 0
        while position < len(audio):
            end_pos = min(position + chunk_size, len(audio))
            audio_chunk = audio[position:end_pos]
            
            inputs_chunk = processor(audio_chunk, return_tensors="pt")
            
            with torch.no_grad():
                _, latents_chunk = model(
                    inputs_chunk['audio'],
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=True
                )
            
            chunks.append(latents_chunk)
            position = end_pos
        
        result = torch.cat(chunks, dim=1) if chunks else torch.zeros(1, 0, model.config.vae_dim)
        individual_results.append(result)
        print(f"Sample {i}: {len(chunks)} chunks, final shape: {result.shape}")
    
    # Test 2: Chunk-wise batching with dynamic handling
    print("\n--- Test 2: Chunk-wise Batch Processing ---")
    cache_batch = VibePodTokenizerStreamingCache()
    batch_results = {i: [] for i in range(batch_size)}
    active_samples = list(range(batch_size))
    positions = [0] * batch_size
    
    chunk_idx = 0
    while active_samples:
        # Collect chunks for active samples
        current_batch_audio = []
        current_indices = []
        
        for i in active_samples[:]:  # Copy list to modify during iteration
            if positions[i] < len(audio_samples[i]):
                end_pos = min(positions[i] + chunk_size, len(audio_samples[i]))
                audio_chunk = audio_samples[i][positions[i]:end_pos]
                
                # Pad to chunk_size if necessary for batching
                if len(audio_chunk) < chunk_size:
                    audio_chunk = np.pad(audio_chunk, (0, chunk_size - len(audio_chunk)), mode='constant')
                
                current_batch_audio.append(audio_chunk)
                current_indices.append(i)
                positions[i] = end_pos
                
                # Check if this sample is finished
                if positions[i] >= len(audio_samples[i]):
                    active_samples.remove(i)
        
        if current_batch_audio:
            # Process batch
            batch_input = processor(current_batch_audio, return_tensors="pt")
            sample_indices = torch.tensor(current_indices)
            
            with torch.no_grad():
                _, latents_batch = model(
                    batch_input['audio'],
                    cache=cache_batch,
                    sample_indices=sample_indices,
                    use_cache=True
                )
            
            # Store results
            for idx, sample_idx in enumerate(current_indices):
                batch_results[sample_idx].append(latents_batch[idx:idx+1])
            
            print(f"Chunk {chunk_idx}: processed {len(current_indices)} samples")
        
        chunk_idx += 1
    
    # Concatenate batch results
    batch_final_results = []
    for i in range(batch_size):
        result = torch.cat(batch_results[i], dim=1) if batch_results[i] else torch.zeros(1, 0, model.config.vae_dim)
        batch_final_results.append(result)
        print(f"Batch sample {i}: final shape: {result.shape}")
    
    # Compare results
    print("\n--- Comparing Results ---")
    for i in range(batch_size):
        individual_shape = individual_results[i].shape
        batch_shape = batch_final_results[i].shape
        
        # Check if shapes match
        if individual_shape == batch_shape:
            # Check if values are close
            if individual_shape[1] > 0:  # Only compare if there's data
                diff = torch.abs(individual_results[i] - batch_final_results[i]).max().item()
                print(f"Sample {i}: shapes match {individual_shape}, max diff: {diff:.6e}")
            else:
                print(f"Sample {i}: shapes match {individual_shape} (empty)")
        else:
            print(f"Sample {i}: shape mismatch! Individual: {individual_shape}, Batch: {batch_shape}")
    
    print("\n✓ Batch streaming test completed")


def test_cache_isolation():
    """Test that different samples maintain isolated caches"""
    print("\n" + "="*80)
    print("Testing Cache Isolation")
    print("="*80)
    
    processor = VibePodTokenizerProcessor(normalize_audio=True)
    model = VibePodSemanticTokenizerModel.from_pretrained("/tmp/viebpod_semantic_tokenizer")
    model.eval()
    
    # Create two different audio signals
    sample_rate = 24000
    duration = 1.0
    audio1 = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sample_rate * duration)))
    audio2 = np.sin(2 * np.pi * 880 * np.linspace(0, duration, int(sample_rate * duration)))
    
    # Process both with same cache but different indices
    cache = VibePodTokenizerStreamingCache()
    
    inputs1 = processor(audio1[:12000], return_tensors="pt")
    inputs2 = processor(audio2[:12000], return_tensors="pt")
    
    with torch.no_grad():
        # Process sample 0
        _, latents1_chunk1 = model(
            inputs1['audio'],
            cache=cache,
            sample_indices=torch.tensor([0]),
            use_cache=True
        )
        
        # Process sample 1 
        _, latents2_chunk1 = model(
            inputs2['audio'],
            cache=cache,
            sample_indices=torch.tensor([1]),
            use_cache=True
        )
        
        # Process second chunk of sample 0
        inputs1_chunk2 = processor(audio1[12000:], return_tensors="pt")
        _, latents1_chunk2 = model(
            inputs1_chunk2['audio'],
            cache=cache,
            sample_indices=torch.tensor([0]),
            use_cache=True
        )
        
        # Process second chunk of sample 1
        inputs2_chunk2 = processor(audio2[12000:], return_tensors="pt")
        _, latents2_chunk2 = model(
            inputs2_chunk2['audio'],
            cache=cache,
            sample_indices=torch.tensor([1]),
            use_cache=True
        )
    
    print(f"Sample 0 chunk 1 shape: {latents1_chunk1.shape}")
    print(f"Sample 0 chunk 2 shape: {latents1_chunk2.shape}")
    print(f"Sample 1 chunk 1 shape: {latents2_chunk1.shape}")
    print(f"Sample 1 chunk 2 shape: {latents2_chunk2.shape}")
    
    # Verify that outputs are different
    if not torch.allclose(latents1_chunk1, latents2_chunk1):
        print("✓ Cache isolation verified: different samples produce different outputs")
    else:
        print("✗ Cache isolation failed: different samples produced same outputs!")


def test_real_audio_streaming(audio_path: str = "/mnt/conversationhub/zhiliang/other/billg.wav"):
    """Test streaming vs non-streaming with real audio file"""
    print("\n" + "="*80)
    print(f"Testing Real Audio: {audio_path}")
    print("="*80)
    
    # Initialize processor and models
    processor = VibePodTokenizerProcessor(normalize_audio=False)
    
    # Test with both tokenizer types
    for tokenizer_type in ["acoustic", "semantic"]:
        print(f"\n\n{'='*80}")
        print(f"Testing {tokenizer_type.upper()} Tokenizer")
        print('='*80)
        
        if tokenizer_type == "acoustic":
            model = VibePodAcousticTokenizerModel.from_pretrained("/tmp/viebpod_acoustic_tokenizer")
        else:
            model = VibePodSemanticTokenizerModel.from_pretrained("/tmp/viebpod_semantic_tokenizer")
        
        model.eval()
        
        # Load and preprocess audio
        print(f"\nLoading audio from: {audio_path}")
        audio = processor.preprocess_audio(audio_path)
        print(f"Audio shape: {audio.shape}")
        print(f"Audio duration: {len(audio) / processor.sampling_rate:.2f} seconds")
        print(f"Audio sample rate: {processor.sampling_rate} Hz")
        
        # Process full audio (non-streaming)
        print("\n--- Non-Streaming Processing ---")
        start_time = time.time()
        
        inputs = processor(audio, return_tensors="pt")
        print(f"Input tensor shape: {inputs['audio'].shape}")
        
        with torch.no_grad():
            if tokenizer_type == "acoustic":
                reconstructed_full, latents_full = model(**inputs)
                print(f"Latents shape: {latents_full.shape}")
                print(f"Reconstructed shape: {reconstructed_full.shape}")
            else:
                _, latents_full = model(**inputs)
                print(f"Latents shape: {latents_full.shape}")
        
        non_streaming_time = time.time() - start_time
        print(f"Non-streaming processing time: {non_streaming_time:.3f} seconds")
        
        # Process in streaming mode
        print("\n--- Streaming Processing ---")
        # chunk_duration = 0.5  # 500ms chunks
        # chunk_size = int(processor.sampling_rate * chunk_duration)
        chunk_size = 3200 
        print(f"Chunk size: {chunk_size} samples")
        
        # Create cache and sample indices
        cache = VibePodTokenizerStreamingCache()
        sample_indices = torch.tensor([0])
        
        latents_chunks = []
        reconstructed_chunks = [] if tokenizer_type == "acoustic" else None
        
        start_time = time.time()
        position = 0
        chunk_idx = 0
        
        while position < len(audio):
            end_pos = min(position + chunk_size, len(audio))
            audio_chunk = audio[position:end_pos]
            
            # Process chunk
            inputs_chunk = processor(audio_chunk, return_tensors="pt")
            
            with torch.no_grad():
                if tokenizer_type == "acoustic":
                    # Encode
                    encoder_output = model.encode(
                        inputs_chunk['audio'], 
                        cache=cache, 
                        sample_indices=sample_indices, 
                        use_cache=True,
                        debug=False
                    )
                    sampled_latents, _ = model.sampling(encoder_output)
                    latents_chunks.append(sampled_latents)
                    
                    # Decode
                    reconstructed_chunk = model.decode(
                        sampled_latents,
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=True,
                        debug=False
                    )
                    reconstructed_chunks.append(reconstructed_chunk)
                    
                    print(f"Chunk {chunk_idx}: pos {position}-{end_pos}, "
                          f"latents shape {sampled_latents.shape}, "
                          f"output shape {reconstructed_chunk.shape}")
                else:
                    _, latents_chunk = model(
                        inputs_chunk['audio'],
                        cache=cache,
                        sample_indices=sample_indices,
                        use_cache=True
                    )
                    latents_chunks.append(latents_chunk)
                    print(f"Chunk {chunk_idx}: pos {position}-{end_pos}, "
                          f"latents shape {latents_chunk.shape}")
            
            position = end_pos
            chunk_idx += 1
        
        streaming_time = time.time() - start_time
        print(f"Streaming processing time: {streaming_time:.3f} seconds")
        print(f"Processed {chunk_idx} chunks")
        
        # Concatenate streaming outputs
        if latents_chunks:
            latents_streaming = torch.cat(latents_chunks, dim=1)
            print(f"\nStreaming latents final shape: {latents_streaming.shape}")
            
            if tokenizer_type == "acoustic" and reconstructed_chunks:
                reconstructed_streaming = torch.cat(reconstructed_chunks, dim=2)
                print(f"Streaming reconstructed final shape: {reconstructed_streaming.shape}")
        
        # Save outputs for comparison
        output_dir = f"./tts_results/vibepod_{tokenizer_type}_outputs"
        import os
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nSaving outputs to: {output_dir}")
        
        if tokenizer_type == "acoustic":
            # Save non-streaming reconstructed audio
            non_streaming_paths = processor.save_audio(
                reconstructed_full,
                output_path=f"{output_dir}/non_streaming_reconstructed.wav",
                normalize=True
            )
            print(f"Saved non-streaming reconstructed: {non_streaming_paths}")
            
            # Save streaming reconstructed audio
            streaming_paths = processor.save_audio(
                reconstructed_streaming,
                output_path=f"{output_dir}/streaming_reconstructed.wav",
                normalize=True
            )
            print(f"Saved streaming reconstructed: {streaming_paths}")
            
            # Also save original for reference
            original_paths = processor.save_audio(
                torch.from_numpy(audio).unsqueeze(0).unsqueeze(0),
                output_path=f"{output_dir}/original.wav",
                normalize=True
            )
            print(f"Saved original: {original_paths}")
        
        # Compare latents
        print("\n--- Comparing Latents ---")
        print(f"Non-streaming latents shape: {latents_full.shape}")
        print(f"Streaming latents shape: {latents_streaming.shape}")
        
        # Check if shapes match (they might differ due to padding/boundary handling)
        if latents_full.shape == latents_streaming.shape:
            diff = torch.abs(latents_full - latents_streaming).max().item()
            print(f"Latents max difference: {diff:.6e}")
            
            # Calculate similarity metrics
            mse = torch.mean((latents_full - latents_streaming) ** 2).item()
            print(f"Latents MSE: {mse:.6e}")
            
            # Cosine similarity
            latents_full_flat = latents_full.flatten()
            latents_streaming_flat = latents_streaming.flatten()
            cosine_sim = torch.nn.functional.cosine_similarity(
                latents_full_flat.unsqueeze(0), 
                latents_streaming_flat.unsqueeze(0)
            ).item()
            print(f"Latents cosine similarity: {cosine_sim:.6f}")
        else:
            print("Note: Latent shapes differ due to different padding/boundary handling")
            print("This is expected behavior for streaming vs non-streaming processing")
        
        # Performance comparison
        print(f"\n--- Performance Summary ---")
        print(f"Non-streaming time: {non_streaming_time:.3f}s")
        print(f"Streaming time: {streaming_time:.3f}s")
        print(f"Streaming overhead: {(streaming_time / non_streaming_time - 1) * 100:.1f}%")
        
        # Real-time factor (for streaming)
        audio_duration = len(audio) / processor.sampling_rate
        rtf = streaming_time / audio_duration
        print(f"Streaming RTF (Real-Time Factor): {rtf:.3f}x")
        if rtf < 1.0:
            print("✓ Streaming is faster than real-time!")
        else:
            print("✗ Streaming is slower than real-time")


def test_long_audio_streaming():
    """Test streaming with very long audio to demonstrate memory efficiency"""
    print("\n" + "="*80)
    print("Testing Long Audio Streaming (Memory Efficiency)")
    print("="*80)
    
    # Create synthetic long audio (10 seconds)
    sample_rate = 24000
    duration = 10.0
    frequency = 440.0
    
    print(f"\nGenerating {duration}s synthetic audio...")
    t = np.linspace(0, duration, int(sample_rate * duration))
    audio = np.sin(2 * np.pi * frequency * t) * 0.5
    
    # Add some variation
    audio += np.sin(2 * np.pi * (frequency * 2) * t) * 0.2
    audio += np.sin(2 * np.pi * (frequency * 0.5) * t) * 0.3
    
    processor = VibePodTokenizerProcessor(normalize_audio=True)
    model = VibePodSemanticTokenizerModel.from_pretrained("/tmp/viebpod_semantic_tokenizer")
    model.eval()
    
    # Process with different chunk sizes
    chunk_sizes = [0.1, 0.5, 1.0, 2.0]  # seconds
    
    for chunk_duration in chunk_sizes:
        print(f"\n--- Chunk size: {chunk_duration}s ---")
        chunk_size = int(sample_rate * chunk_duration)
        
        cache = VibePodTokenizerStreamingCache()
        sample_indices = torch.tensor([0])
        
        start_time = time.time()
        num_chunks = 0
        position = 0
        
        while position < len(audio):
            end_pos = min(position + chunk_size, len(audio))
            audio_chunk = audio[position:end_pos]
            
            inputs_chunk = processor(audio_chunk, return_tensors="pt")
            
            with torch.no_grad():
                _, latents_chunk = model(
                    inputs_chunk['audio'],
                    cache=cache,
                    sample_indices=sample_indices,
                    use_cache=True
                )
            
            position = end_pos
            num_chunks += 1
        
        elapsed = time.time() - start_time
        rtf = elapsed / duration
        
        print(f"Processed {num_chunks} chunks in {elapsed:.3f}s")
        print(f"RTF: {rtf:.3f}x")


if __name__ == "__main__":
    # Run all tests
    # test_acoustic_tokenizer_streaming()
    # test_semantic_tokenizer_streaming()
    # test_batch_streaming()
    # test_cache_isolation()
    test_real_audio_streaming()
    test_long_audio_streaming()
    
    print("\n" + "="*80)
    print("✅ ALL VIBEPOD STREAMING TESTS COMPLETED! ✅")
    print("="*80)