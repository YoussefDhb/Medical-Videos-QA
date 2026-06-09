#!/usr/bin/env python3
"""
Rebuild Visual Embeddings Only

This script rebuilds ONLY the visual embeddings with improved configuration,
preserving the existing text embeddings which are already working well.

Usage:
    python rebuild_visual_embeddings.py --split train
    python rebuild_visual_embeddings.py --split test --video_dir videos_test/
"""
import os
import argparse
import glob
from video_processing import VideoProcessor
from video_processing.pipeline import VideoProcessorConfig

def rebuild_visual_for_split(video_dir, split, output_dir="feature_extraction"):
    """
    Rebuild visual embeddings for a specific split.

    Args:
        video_dir: Directory containing video files
        split: Dataset split (train/test/val)
        output_dir: Base directory for feature storage
    """
    # Output directories
    text_feat_dir = os.path.join(output_dir, "textual", split)
    visual_feat_dir = os.path.join(output_dir, "visual", split)

    # Create visual output directory
    os.makedirs(visual_feat_dir, exist_ok=True)

    # Find all videos
    video_files = glob.glob(os.path.join(video_dir, "*.mp4")) + \
                  glob.glob(os.path.join(video_dir, "*.avi")) + \
                  glob.glob(os.path.join(video_dir, "*.mov"))

    if not video_files:
        print(f"No video files found in {video_dir}")
        return

    print(f"\n{'='*80}")
    print(f"REBUILDING VISUAL EMBEDDINGS - {split.upper()} SPLIT")
    print(f"{'='*80}")
    print(f"Video directory: {video_dir}")
    print(f"Text features: {text_feat_dir} (preserving existing)")
    print(f"Visual features: {visual_feat_dir} (rebuilding)")
    print(f"Total videos: {len(video_files)}")
    print(f"\nNEW CONFIGURATION:")
    print(f"  - frames_per_segment: 6 (was: 2)")
    print(f"  - sampling_strategy: quality_based (was: adaptive)")
    print(f"  - aggregation_method: max (was: mean)")
    print(f"  - quality_filter: True (was: False)")
    print(f"  - min_frames: 3, max_frames: 12")
    print(f"{'='*80}\n")

    # Configure processor with improved visual settings
    config = VideoProcessorConfig(
        frames_per_segment=6,
        sampling_strategy='quality_based',
        quality_filter=True,
        aggregation_method='max',
        min_frames=3,
        max_frames=12
    )

    processor = VideoProcessor(config=config)

    # Process each video
    processed = 0
    skipped = 0
    errors = 0

    for video_path in video_files:
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Check if text features exist
        text_json = os.path.join(text_feat_dir, f"{video_id}.json")
        if not os.path.exists(text_json):
            print(f"Skipping {video_id}: text features not found")
            skipped += 1
            continue

        try:
            print(f"\n[{processed+1}/{len(video_files)}] Processing {video_id}...")

            # Load existing text embeddings
            import json
            with open(text_json, 'r') as f:
                text_data = json.load(f)

            # Extract text segments (needed for visual embedding alignment)
            text_segments = []
            for item in text_data:
                text_segments.append({
                    'segment_id': item['segment_id'],
                    'timestamp': item['timestamp'],
                    'text': item.get('text', '')
                })

            # Generate visual embeddings only
            from video_processing import extract_frames_and_embed
            visual_results = extract_frames_and_embed(
                video_path, text_segments, video_id=video_id,
                frames_per_segment=config.frames_per_segment,
                sampling_strategy=config.sampling_strategy,
                quality_filter=config.quality_filter,
                aggregation_method=config.aggregation_method,
                min_frames=config.min_frames,
                max_frames=config.max_frames
            )

            # Apply visual deduplication
            from video_processing import deduplicate_embeddings_similarity
            visual_results = deduplicate_embeddings_similarity(
                visual_results,
                similarity_threshold=config.visual_similarity_threshold
            )

            # Save visual features
            import numpy as np
            visual_json = os.path.join(visual_feat_dir, f"{video_id}.json")
            visual_serializable = []
            for item in visual_results:
                item_copy = {k: v for k, v in item.items() if k != 'embedding'}
                item_copy['embedding'] = item['embedding'].tolist() if isinstance(item['embedding'], np.ndarray) else item['embedding']
                visual_serializable.append(item_copy)

            with open(visual_json, 'w') as f:
                json.dump(visual_serializable, f, indent=2)

            print(f"Processed {video_id}: {len(visual_results)} visual segments")
            processed += 1

        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            errors += 1

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {processed}/{len(video_files)} videos")
    print(f"Skipped (no text features): {skipped}")
    print(f"Errors: {errors}")
    print(f"\nVisual features saved to: {visual_feat_dir}")
    print(f"\nNext step: Rebuild FAISS visual index:")
    print(f"   python embedding_storage.py --feature_dir {output_dir} --split {split} --modality visual")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rebuild visual embeddings with improved configuration")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test", "val"],
                       help="Dataset split to process")
    parser.add_argument("--video_dir", type=str, default=None,
                       help="Video directory (auto-detected if not specified)")
    parser.add_argument("--output_dir", type=str, default="feature_extraction",
                       help="Base output directory for features")

    args = parser.parse_args()

    # Auto-detect video directory if not specified
    if args.video_dir is None:
        video_dirs = {
            "train": "videos_train",
            "test": "videos_test",
            "val": "videos_val"
        }
        args.video_dir = video_dirs.get(args.split, f"videos_{args.split}")

    if not os.path.exists(args.video_dir):
        print(f"Video directory not found: {args.video_dir}")
        exit(1)

    rebuild_visual_for_split(args.video_dir, args.split, args.output_dir)
