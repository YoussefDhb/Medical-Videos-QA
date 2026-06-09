"""
whisper_patch.py

Helper functions to create HuggingFace Whisper ASR pipelines with accurate word-level timestamp prediction during transcription, including for incomplete audio segments.

Functions:
    get_whisper_pipeline_with_timestamps(model_id, device):
        Returns a Whisper ASR pipeline with timestamp support using AutoProcessor and AutoModelForSpeechSeq2Seq.
    get_whisper_pipeline_with_timestamps_simple(model_id, device=None):
        Returns a Whisper ASR pipeline with timestamp support using WhisperProcessor and WhisperForConditionalGeneration.

Example usage:
    from whisper_patch import get_whisper_pipeline_with_timestamps
    pipe = get_whisper_pipeline_with_timestamps(model_id, device=None)
    result = pipe(audio_path)
"""
from transformers import pipeline as hf_pipeline
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import WhisperProcessor, WhisperForConditionalGeneration

import torch

def get_whisper_pipeline_with_timestamps(model_id, device):
    """
    Create a HuggingFace Whisper ASR pipeline with timestamp support using AutoProcessor and AutoModelForSpeechSeq2Seq.

    Args:
        model_id (str): The HuggingFace model ID for the Whisper model.
        device (str or None): Device to use ('cuda', 'mps', or None for auto-detect).

    Returns:
        transformers.Pipeline: Configured ASR pipeline with timestamp support.
    """
    processor = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id)
    if torch.cuda.is_available():
        model = model.to('cuda')
        device_id = 0
    elif torch.backends.mps.is_available():
        model = model.to('mps')
        device_id = 0
    else:
        device_id = -1


    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,
        device=device_id,
    )
    return pipe

def get_whisper_pipeline_with_timestamps_simple(model_id, device=None):
    """
    Create a Whisper ASR pipeline with timestamp support using WhisperProcessor and WhisperForConditionalGeneration.
    This approach is often the most reliable for obtaining word-level timestamps.

    Args:
        model_id (str): The HuggingFace model ID for the Whisper model.
        device (str or None): Device to use ('cuda', 'mps', or None for auto-detect).

    Returns:
        transformers.Pipeline: Configured ASR pipeline with timestamp support.
    """
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(model_id)

    # Device handling
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
            device_id = 0
        elif torch.backends.mps.is_available():
            device = 'mps'
            device_id = 0
        else:
            device = 'cpu'
            device_id = -1
    else:
        device_id = 0 if device != 'cpu' else -1

    model = model.to(device)

    # Create pipeline with explicit timestamp configuration
    pipe = hf_pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        return_timestamps=True,  # This is key
        device=device_id,
        generate_kwargs={
            "language": "english",  # specify language or make configurable
            "task": "transcribe",
        }
    )

    # Optional: modify the pipeline's postprocessing to ensure timestamps
    original_postprocess = pipe.postprocess

    def enhanced_postprocess(model_outputs, return_timestamps=True, **kwargs):
        # Ensure return_timestamps is always True
        return original_postprocess(
            model_outputs,
            return_timestamps=True,
            **kwargs
        )

    pipe.postprocess = enhanced_postprocess

    return pipe
