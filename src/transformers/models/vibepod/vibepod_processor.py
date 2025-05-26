import math
import warnings
from typing import List, Optional, Union, Dict, Any, Tuple
import os

import numpy as np
import torch

from ...tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from ...utils import TensorType, logging
from .vibepod_tokenizer_processor import AudioNormalizer


logger = logging.get_logger(__name__)


class VibePodProcessor:
    r"""
    Constructs a VibePod processor which wraps a VibePod tokenizer and audio processor into a single processor.

    [`VibePodProcessor`] offers all the functionalities of [`VibePodTokenizer`] and [`VibePodTokenizerProcessor`]. 
    See the [`~VibePodProcessor.__call__`] and [`~VibePodProcessor.decode`] for more information.

    Args:
        tokenizer (`VibePodTextTokenizer` or `VibePodTextTokenizerFast`):
            The tokenizer for text processing.
        audio_processor (`VibePodTokenizerProcessor`):
            The audio processor for speech processing.
        speech_tok_compress_ratio (`int`, *optional*, defaults to 3200):
            The compression ratio for speech tokenization.
        db_normalize (`bool`, *optional*, defaults to True):
            Whether to apply decibel normalization to audio inputs.
    """

    def __init__(self, tokenizer=None, audio_processor=None, speech_tok_compress_ratio=3200, db_normalize=True, **kwargs):
        self.tokenizer = tokenizer
        self.audio_processor = audio_processor
        self.speech_tok_compress_ratio = speech_tok_compress_ratio
        self.db_normalize = db_normalize
        self.audio_normalizer = AudioNormalizer() if db_normalize else None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        """
        Instantiate a VibePodProcessor from a pretrained VibePod processor.

        Args:
            pretrained_model_name_or_path (`str` or `os.PathLike`):
                This can be either:
                - a string, the *model id* of a pretrained model
                - a path to a *directory* containing processor config

        Returns:
            [`VibePodProcessor`]: The processor object instantiated from pretrained model.
        """
        import os
        import json
        from .vibepod_tokenizer_processor import VibePodTokenizerProcessor
        from .modular_vibepod_text_tokenizer import VibePodTextTokenizer, VibePodTextTokenizerFast
        
        # Load processor configuration
        config_path = os.path.join(pretrained_model_name_or_path, "preprocessor_config.json")
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            logger.warning(f"No preprocessor_config.json found at {pretrained_model_name_or_path}, using defaults")
            config = {
                "speech_tok_compress_ratio": 3200,
                "db_normalize": True,
            }
        
        # Extract main processor parameters
        speech_tok_compress_ratio = config.get("speech_tok_compress_ratio", 3200)
        db_normalize = config.get("db_normalize", True)
        
        # Load tokenizer - try from model path first, then fallback to Qwen
        cache_dir = kwargs.pop("cache_dir", None)
        tokenizer = VibePodTextTokenizerFast.from_pretrained(
            config.get("language_model_pretrained_name", "Qwen/Qwen2.5-1.5B"),
            cache_dir=cache_dir,
            **kwargs
        )
        
        # Load audio processor
        if "audio_processor" in config:
            # Create audio processor from config
            audio_config = config["audio_processor"]
            audio_processor = VibePodTokenizerProcessor(
                sampling_rate=audio_config.get("sampling_rate", 24000),
                normalize_audio=audio_config.get("normalize_audio", True),
                target_dB_FS=audio_config.get("target_dB_FS", -25),
                eps=audio_config.get("eps", 1e-6),
            )
        else:
            # Create default audio processor
            audio_processor = VibePodTokenizerProcessor()
        
        # Create and return the processor
        return cls(
            tokenizer=tokenizer,
            audio_processor=audio_processor,
            speech_tok_compress_ratio=speech_tok_compress_ratio,
            db_normalize=db_normalize,
        )
    
    def save_pretrained(self, save_directory: Union[str, os.PathLike], **kwargs):
        """
        Save a processor to a directory, so that it can be re-loaded using the
        [`~VibePodProcessor.from_pretrained`] class method.

        Args:
            save_directory (`str` or `os.PathLike`):
                Directory where the processor will be saved.
        """
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Save processor configuration
        processor_config = {
            "processor_class": "VibePodProcessor",
            "speech_tok_compress_ratio": self.speech_tok_compress_ratio,
            "db_normalize": self.db_normalize,
            "audio_processor": {
                "feature_extractor_type": "VibePodTokenizerProcessor",
                "sampling_rate": getattr(self.audio_processor, 'sampling_rate', 24000),
                "normalize_audio": getattr(self.audio_processor, 'normalize_audio', True),
                "target_dB_FS": getattr(self.audio_processor, 'target_dB_FS', -25),
                "eps": getattr(self.audio_processor, 'eps', 1e-6),
            }
        }
        
        config_path = os.path.join(save_directory, "preprocessor_config.json")
        with open(config_path, 'w') as f:
            json.dump(processor_config, f, indent=2)
        
        logger.info(f"Processor configuration saved in {config_path}")
    
    def process_podcast_script(
        self,
        script: str,
        speaker_samples: List[Union[str, np.ndarray]] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Process a podcast script with speaker voice samples for TTS generation.

        Args:
            script (`str`):
                The podcast script in format "Speaker 1: text\nSpeaker 2: text\n..."
            speaker_samples (`List[Union[str, np.ndarray]]`, *optional*):
                List of audio samples (file paths or arrays) for each speaker's voice.
            return_tensors (`str` or [`~utils.TensorType`], *optional*):
                If set, will return tensors of a particular framework. Acceptable values are:
                - `'tf'`: Return TensorFlow `tf.constant` objects.
                - `'pt'`: Return PyTorch `torch.Tensor` objects.
                - `'np'`: Return NumPy `np.ndarray` objects.

        Returns:
            [`BatchEncoding`]: A [`BatchEncoding`] with the following fields:
                - **input_ids** -- Token IDs including special tokens for speech
                - **speech_inputs** -- Processed audio tensors for voice samples
                - **speech_input_mask** -- Boolean mask indicating speech token positions
                - **parsed_script** -- List of (speaker_id, text) tuples
        """
        # Parse the script
        parsed_lines = self._parse_script(script)
        all_speakers = list(set(speaker_id for speaker_id, _ in parsed_lines))
        
        # Create system prompt
        system_prompt = " Transform the text provided by various speakers into speech output, utilizing the distinct voice of each respective speaker.\n"
        system_tokens = self.tokenizer.encode(system_prompt, add_special_tokens=False)
        
        # Process voice samples if provided
        if speaker_samples:
            voice_tokens, voice_speech_inputs, voice_speech_masks = self._create_voice_prompt(speaker_samples)
        else:
            voice_tokens, voice_speech_inputs, voice_speech_masks = [], [], []
        
        # Build full token sequence
        full_tokens = system_tokens + voice_tokens
        speech_input_mask = [False] * len(system_tokens) + voice_speech_masks
        
        # Add text input section
        full_tokens += self.tokenizer.encode(' Text input:\n', add_special_tokens=False)
        speech_input_mask += [False] * len(self.tokenizer.encode(' Text input:\n', add_special_tokens=False))
        
        for speaker_id, speaker_text in parsed_lines:
            speaker_text_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:{speaker_text}\n", add_special_tokens=False)
            full_tokens += speaker_text_tokens
            speech_input_mask += [False] * len(speaker_text_tokens)
        
        # Add speech output section
        full_tokens += self.tokenizer.encode(' Speech output:\n', add_special_tokens=False) + [self.tokenizer.speech_start_id]
        speech_input_mask += [False] * (len(self.tokenizer.encode(' Speech output:\n', add_special_tokens=False)) + 1)
        
        # Prepare outputs
        encoding = BatchEncoding()
        encoding["input_ids"] = full_tokens
        encoding["speech_inputs"] = voice_speech_inputs if voice_speech_inputs else None
        encoding["speech_input_mask"] = speech_input_mask
        encoding["parsed_script"] = parsed_lines
        encoding["all_speakers"] = all_speakers
        
        # Convert to tensors if requested
        if return_tensors is not None:
            # Handle tensor conversion manually for complex types
            if return_tensors == "pt":
                encoding["input_ids"] = torch.tensor(encoding["input_ids"])
                encoding["speech_input_mask"] = torch.tensor(encoding["speech_input_mask"], dtype=torch.bool)
                # Don't convert speech_inputs here - they need padding first
            elif return_tensors == "np":
                encoding["input_ids"] = np.array(encoding["input_ids"])
                encoding["speech_input_mask"] = np.array(encoding["speech_input_mask"], dtype=np.bool_)
            elif return_tensors == "tf":
                import tensorflow as tf
                encoding["input_ids"] = tf.constant(encoding["input_ids"])
                encoding["speech_input_mask"] = tf.constant(encoding["speech_input_mask"], dtype=tf.bool)
            else:
                # For other fields that can be converted normally
                encoding = encoding.convert_to_tensors(return_tensors, prepend_batch_axis=False)
            
        return encoding

    def _parse_script(self, script: str) -> List[Tuple[int, str]]:
        """Parse script into list of (speaker_id, text) tuples."""
        lines = script.strip().split("\n")
        parsed_lines = []
        
        for line in lines:
            if not line.strip():
                continue
            try:
                speaker_part, text = line.split(":", 1)
                speaker_id = int(speaker_part.strip().split(" ")[1])
                text = ' ' + text.strip()
                # Normalize speaker IDs to start from 0
                parsed_lines.append((speaker_id - 1 if speaker_id > 0 else 0, text))
            except Exception as e:
                logger.warning(f"Error parsing line: '{line}' - {str(e)}")
                
        return parsed_lines

    def _create_voice_prompt(
        self, 
        speaker_samples: List[Union[str, np.ndarray]]
    ) -> Tuple[List[int], List[np.ndarray], List[bool]]:
        """
        Create voice prompt tokens and process audio samples.
        
        Returns:
            tuple: (voice_tokens, voice_speech_inputs, voice_speech_masks)
        """
        vae_token_id = self.tokenizer.speech_diffusion_id
        
        voice_full_tokens = self.tokenizer.encode(' Voice input:\n', add_special_tokens=False)
        voice_speech_inputs = []
        voice_speech_masks = [False] * len(voice_full_tokens)
        
        for speaker_id, speaker_audio in enumerate(speaker_samples):
            prefix_tokens = self.tokenizer.encode(f" Speaker {speaker_id}:", add_special_tokens=False)
            
            # Process audio
            if isinstance(speaker_audio, str):
                # Load audio from file
                wav = self.audio_processor._load_audio_from_path(speaker_audio)
            else:
                wav = np.array(speaker_audio, dtype=np.float32)
            
            # Apply normalization if needed
            if self.db_normalize and self.audio_normalizer:
                wav = self.audio_normalizer(wav)
            
            # Calculate token length based on compression ratio
            vae_tok_len = math.ceil(wav.shape[0] / self.speech_tok_compress_ratio)
            
            # Build tokens and masks
            speaker_tokens = (prefix_tokens + 
                            [self.tokenizer.speech_start_id] + 
                            [vae_token_id] * vae_tok_len + 
                            [self.tokenizer.speech_end_id] + 
                            self.tokenizer.encode('\n', add_special_tokens=False))
            
            vae_input_mask = ([False] * len(prefix_tokens) + 
                            [False] + 
                            [True] * vae_tok_len + 
                            [False] + 
                            [False])
            
            voice_full_tokens.extend(speaker_tokens)
            voice_speech_masks.extend(vae_input_mask)
            voice_speech_inputs.append(wav)
            
        return voice_full_tokens, voice_speech_inputs, voice_speech_masks

    def prepare_speech_inputs(
        self,
        speech_inputs: List[np.ndarray],
        return_tensors: Optional[Union[str, TensorType]] = None,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> Dict[str, Any]:
        """
        Prepare speech inputs for model consumption.
        
        Args:
            speech_inputs: List of speech arrays
            return_tensors: Output tensor type
            device: Device to place tensors on
            dtype: Data type for tensors
            
        Returns:
            Dictionary with padded_speeches and speech_masks
        """
        if not speech_inputs:
            return {"padded_speeches": None, "speech_masks": None}
        
        # Calculate sequence lengths
        vae_tok_seqlens = [math.ceil(s.shape[0] / self.speech_tok_compress_ratio) for s in speech_inputs]
        max_speech_length = max(s.shape[0] for s in speech_inputs)
        
        # Pad speeches
        padded_speeches = np.full((len(speech_inputs), max_speech_length), fill_value=0, dtype=np.float32)
        speech_masks = np.zeros((len(speech_inputs), max(vae_tok_seqlens)), dtype=np.bool_)
        
        for i, (speech, vae_tok_length) in enumerate(zip(speech_inputs, vae_tok_seqlens)):
            padded_speeches[i, :len(speech)] = speech
            speech_masks[i, :vae_tok_length] = True
        
        result = {
            "padded_speeches": padded_speeches,
            "speech_masks": speech_masks,
        }
        
        # Convert to tensors if requested
        if return_tensors == "pt":
            result["padded_speeches"] = torch.tensor(padded_speeches, device=device, dtype=dtype)
            result["speech_masks"] = torch.tensor(speech_masks, device=device)
        elif return_tensors == "tf":
            import tensorflow as tf
            result["padded_speeches"] = tf.constant(padded_speeches, dtype=tf.float32)
            result["speech_masks"] = tf.constant(speech_masks, dtype=tf.bool)
            
        return result

    def __call__(
        self,
        text: Optional[Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]]] = None,
        audio: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
        script: Optional[str] = None,
        speaker_samples: Optional[List[Union[str, np.ndarray]]] = None,
        add_special_tokens: bool = True,
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Union[bool, str, TruncationStrategy] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = None,
        return_attention_mask: Optional[bool] = None,
        **kwargs,
    ) -> BatchEncoding:
        """
        Main method to prepare inputs for VibePod models.
        
        This method can handle different input scenarios:
        1. Text-only input for language modeling
        2. Audio-only input for speech processing
        3. Podcast script with speaker samples for TTS generation
        4. Combined text and audio inputs
        
        Args:
            text: Text input(s) to encode
            audio: Audio input(s) to process
            script: Podcast script for TTS generation
            speaker_samples: Voice samples for each speaker
            Other arguments are standard tokenizer arguments
            
        Returns:
            BatchEncoding with processed inputs
        """
        # Handle podcast script processing
        if script is not None:
            return self.process_podcast_script(
                script=script,
                speaker_samples=speaker_samples,
                return_tensors=return_tensors,
                **kwargs
            )
        
        # Handle text input
        if text is not None:
            text_inputs = self.tokenizer(
                text,
                add_special_tokens=add_special_tokens,
                padding=padding,
                truncation=truncation,
                max_length=max_length,
                return_tensors=return_tensors,
                return_attention_mask=return_attention_mask,
                **kwargs,
            )
        else:
            text_inputs = None
        
        # Handle audio input
        if audio is not None:
            audio_inputs = self.audio_processor(
                audio,
                return_tensors=return_tensors,
                **kwargs,
            )
        else:
            audio_inputs = None
        
        # Combine inputs
        if text_inputs is not None and audio_inputs is not None:
            # Merge text and audio inputs
            return self._merge_inputs(text_inputs, audio_inputs)
        elif text_inputs is not None:
            return text_inputs
        elif audio_inputs is not None:
            return BatchEncoding(audio_inputs)
        else:
            raise ValueError("You must provide either text, audio, or script input")

    def _merge_inputs(self, text_inputs: BatchEncoding, audio_inputs: Dict) -> BatchEncoding:
        """Merge text and audio inputs into a single BatchEncoding."""
        # Start with text inputs
        merged = BatchEncoding(text_inputs)
        
        # Add audio-specific fields
        if "audio" in audio_inputs:
            merged["speech_inputs"] = audio_inputs["audio"]
        if "streaming" in audio_inputs:
            merged["streaming"] = audio_inputs["streaming"]
            
        return merged

    def batch_decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibePodTextTokenizer's [`~PreTrainedTokenizer.batch_decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.batch_decode(*args, **kwargs)

    def decode(self, *args, **kwargs):
        """
        This method forwards all its arguments to VibePodTextTokenizer's [`~PreTrainedTokenizer.decode`].
        Please refer to the docstring of this method for more information.
        """
        return self.tokenizer.decode(*args, **kwargs)

    @property
    def model_input_names(self):
        """
        Return the list of inputs accepted by the model.
        """
        tokenizer_input_names = self.tokenizer.model_input_names
        audio_processor_input_names = self.audio_processor.model_input_names
        return list(dict.fromkeys(tokenizer_input_names + audio_processor_input_names + ["speech_inputs", "speech_input_mask"]))

__all__ = [
    "VibePodProcessor",
]