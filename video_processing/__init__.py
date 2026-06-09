"""
video_processing package

Modular video processing pipeline for medical video QA:
- ASR (Automatic Speech Recognition)
- Text embeddings (BiomedCLIP Text Encoder)
- Visual embeddings (BiomedCLIP Vision Encoder)
- Deduplication utilities
- Pipeline orchestration
"""

from .asr import extract_audio_ffmpeg, transcribe_with_asr
from .text_embeddings import (
    load_ner_and_embed_models,
    extract_entities_and_embed,
    extract_entities_and_embed_optimized
)
from .visual_embeddings import (
    compute_frame_sharpness,
    extract_frames_and_embed
)
from .deduplication import deduplicate_embeddings_similarity
from .pipeline import VideoProcessor

__all__ = [
    'extract_audio_ffmpeg',
    'transcribe_with_asr',
    'load_ner_and_embed_models',
    'extract_entities_and_embed',
    'extract_entities_and_embed_optimized',
    'compute_frame_sharpness',
    'extract_frames_and_embed',
    'deduplicate_embeddings_similarity',
    'VideoProcessor'
]
