"""
Automatic Speech Recognition (ASR) module

Handles audio extraction from videos and transcription using Whisper models.
"""
import os
import subprocess
import tempfile
from whisper_patch import get_whisper_pipeline_with_timestamps_simple


def extract_audio_ffmpeg(video_path, audio_path):
    """
    Extract audio from video using ffmpeg.

    Args:
        video_path: Path to input video file
        audio_path: Path to output audio file (WAV format)

    Raises:
        subprocess.CalledProcessError: If ffmpeg command fails
    """
    # Look for ffmpeg in PATH or common custom cluster locations
    ffmpeg_cmd = 'ffmpeg'
    
    # Check if a custom path is provided via environment (set by clariden_job.sh)
    if 'FFMPEG_PREFIX' in os.environ:
        custom_ffmpeg = os.path.join(os.environ['FFMPEG_PREFIX'], 'bin', 'ffmpeg')
        if os.path.exists(custom_ffmpeg):
            ffmpeg_cmd = custom_ffmpeg

    cmd = [
        ffmpeg_cmd, '-y', '-hide_banner', '-loglevel', 'error',
        '-i', video_path, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path
    ]
    try:
        result = subprocess.run(cmd, check=True, stderr=subprocess.PIPE, text=True)
        print(f"Audio extracted: {os.path.basename(video_path)} -> {os.path.basename(audio_path)}")
    except subprocess.CalledProcessError as e:
        print(f"Audio extraction failed for {video_path}")
        if e.stderr:
            print(f"  Error: {e.stderr}")
        raise


def transcribe_with_asr(video_path, asr_model_id="openai/whisper-tiny"):
    """
    Transcribe video audio using Whisper ASR model.

    Args:
        video_path: Path to video file
        asr_model_id: HuggingFace model ID for Whisper (default: openai/whisper-tiny)

    Returns:
        List of transcript chunks with word-level timestamps
        Each chunk contains: {'text': str, 'timestamp': (start, end)}
    """
    import torch
    import gc

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
        extract_audio_ffmpeg(video_path, tmp_audio.name)
        pipe = get_whisper_pipeline_with_timestamps_simple(asr_model_id, device=None)

    print(f"Transcribing audio with ASR: {video_path} using model: {asr_model_id}")
    result = pipe(tmp_audio.name)
    os.remove(tmp_audio.name)

    # result['chunks'] contains word-level timestamps
    chunks = result['chunks']
    return chunks
