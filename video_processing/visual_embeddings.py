"""
Visual Embedding Module

Handles frame extraction, quality filtering, and visual embeddings using BiomedCLIP.
"""
import tempfile
import cv2
import torch
import numpy as np
import open_clip
from PIL import Image
from tqdm import tqdm


def compute_frame_sharpness(frame):
    """
    Compute frame sharpness using Laplacian variance.
    Higher values indicate sharper frames.

    Args:
        frame: OpenCV frame (BGR format)

    Returns:
        float: Sharpness score
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def extract_frames_and_embed(
    video_path, text_segments, video_id,
    model_id='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
    frames_per_segment=2,
    sampling_strategy='adaptive',
    min_frames=1,
    max_frames=10,
    aggregation_method='mean',
    quality_filter=False
):
    """
    Extract frames and visual embeddings aligned with text segments with multiple sampling strategies.

    Args:
        video_path: Path to video file
        text_segments: List of text segment dicts (from extract_entities_and_embed)
        video_id: Video identifier
        model_id: BiomedCLIP model ID
        frames_per_segment: Number of frames to sample per text segment (HYPERPARAMETER)
        sampling_strategy: 'uniform', 'adaptive', or 'quality_based'
            - uniform: Fixed number of frames per segment
            - adaptive: Varies based on segment duration (1 frame per 3 seconds)
            - quality_based: Sample more frames then select highest quality ones
        min_frames: Minimum frames for very short segments
        max_frames: Maximum frames for very long segments (computational cap)
        aggregation_method: How to combine multiple frame embeddings ('mean', 'max')
        quality_filter: If True, filter frames by sharpness before encoding

    Returns:
        List of visual embeddings with segment_id for multimodal linking
    """
    model, _, preprocess_val = open_clip.create_model_and_transforms(model_id)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    results = []

    if fps == 0:
        print(f"Warning: Could not get FPS for video {video_path}")
        cap.release()
        return results

    print(f"Extracting frames and visual embeddings: {video_path} (FPS: {fps})")
    print(f"Strategy: {sampling_strategy}, Frames/Segment: {frames_per_segment}, Aggregation: {aggregation_method}")

    for segment in tqdm(text_segments, desc="Frame extraction", unit="segment"):
        ts = segment['timestamp']
        segment_id = segment.get('segment_id')  # Link to text segment

        if not isinstance(ts, (tuple, list)) or len(ts) != 2:
            continue

        start_time, end_time = ts
        segment_duration = end_time - start_time

        # Determine number of frames based on strategy
        if sampling_strategy == 'adaptive':
            # Adaptive: 1 frame per 3 seconds, bounded by min/max
            num_frames = max(min_frames, min(int(segment_duration / 3), max_frames))
        elif sampling_strategy == 'quality_based':
            # Quality-based: Sample 2x frames, then select best quality ones
            num_frames = min(frames_per_segment * 2, max_frames)
        else:  # uniform
            num_frames = frames_per_segment

        # Sample frame times
        if segment_duration <= 0:
            frame_times = [start_time]
        elif num_frames == 1:
            frame_times = [start_time]
        else:
            # Uniformly sample across segment duration
            frame_times = [start_time + (i * segment_duration / (num_frames - 1))
                          for i in range(num_frames)]

        # Extract frames and optionally filter by quality
        frames_data = []  # Store (frame, frame_time, sharpness_score)
        for frame_time in frame_times:
            frame_idx = int(frame_time * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            if quality_filter or sampling_strategy == 'quality_based':
                sharpness = compute_frame_sharpness(frame)
                frames_data.append((frame, frame_time, sharpness))
            else:
                frames_data.append((frame, frame_time, 0))

        # Quality-based: Select top-k sharpest frames
        if sampling_strategy == 'quality_based' and len(frames_data) > frames_per_segment:
            frames_data.sort(key=lambda x: x[2], reverse=True)
            frames_data = frames_data[:frames_per_segment]
            # Re-sort by time for temporal consistency
            frames_data.sort(key=lambda x: x[1])

        # Encode frames to embeddings
        embeddings = []
        for frame, frame_time, sharpness in frames_data:
            # Process frame in-memory
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=True) as tmp_img:
                cv2.imwrite(tmp_img.name, frame)
                image = Image.open(tmp_img.name).convert('RGB')
                image_tensor = preprocess_val(image).unsqueeze(0)

                with torch.no_grad():
                    emb = model.encode_image(image_tensor).squeeze().cpu().numpy()
                embeddings.append(emb)

        # Aggregate embeddings from sampled frames
        if embeddings:
            if aggregation_method == 'mean':
                final_emb = np.mean(embeddings, axis=0)
            elif aggregation_method == 'max':
                final_emb = np.max(embeddings, axis=0)
            else:
                final_emb = np.mean(embeddings, axis=0)

            results.append({
                "video_id": video_id,
                "segment_id": segment_id,  # Segment ID for linking text and visual
                "timestamp": ts,
                "frame_path": "in-memory",
                "num_frames_averaged": len(embeddings),
                "sampling_strategy": sampling_strategy,
                "aggregation_method": aggregation_method,
                "embedding": final_emb
            })

    cap.release()
    print(f"Generated {len(results)} visual embeddings from {len(text_segments)} text segments")
    return results
