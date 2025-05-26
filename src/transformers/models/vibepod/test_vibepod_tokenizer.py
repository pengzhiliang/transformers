from transformers.models.vibepod.vibepod_tokenizer_processor import VibePodTokenizerProcessor
from transformers.models.vibepod.modular_vibepod_tokenizer import VibePodAcousticTokenizerModel, VibePodSemanticTokenizerModel

import numpy as np

# 初始化
processor = VibePodTokenizerProcessor(normalize_audio=True)
# model = VibePodAcousticTokenizerModel.from_pretrained("/tmp/viebpod_acoustic_tokenizer")
model = VibePodSemanticTokenizerModel.from_pretrained("/tmp/viebpod_semantic_tokenizer")

# 处理单个音频
inputs = processor("/mnt/conversationhub/zhiliang/other/billg.wav", return_tensors="pt")

breakpoint()
reconstructed, sampled_latents = model(**inputs)

# save the reconstructed audio
processor.save_audio(reconstructed, output_path="reconstructed2.wav")

# # 流式处理
# inputs_streaming = processor(audio, return_tensors="pt", streaming=True)
# outputs_streaming = model.encode(**inputs_streaming, streaming=True, init_streaming_state=True)

# # 批量处理（不同长度）
# audio_batch = [np.random.randn(24000), np.random.randn(72000)]  # 1秒和3秒
# inputs_batch = processor(audio_batch, return_tensors="pt")

# # 禁用标准化
# inputs_raw = processor(audio, normalize_audio=False, return_tensors="pt")