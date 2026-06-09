"""
Utility functions for search module.
"""
import os
import json
from typing import List, Dict, Optional


def load_segments_from_json_dir(json_dir: str = "feature_extraction/", split: Optional[str] = None) -> List[Dict]:
    """
    Load all segments from JSON feature files, recursively searching through subdirectories.

    Args:
        json_dir: Base directory containing JSON files (default: 'feature_extraction/')
                  Will recursively search all subdirectories (train/test/val) for JSON files
        split: Optional split name to filter specific directory (e.g., 'train', 'test', 'val')
               If None, loads from all subdirectories

    Returns:
        List of segment dictionaries with text and metadata
    """
    # Handle specific split if provided
    if split:
        json_dir = os.path.join(json_dir, split)

    if not os.path.exists(json_dir):
        raise FileNotFoundError(f"JSON directory not found: {json_dir}")

    print(f"Loading segments from {json_dir} (searching recursively)...")

    segments = []
    json_files_found = 0

    # Recursively walk through all subdirectories
    for root, dirs, files in os.walk(json_dir):
        for filename in files:
            if filename.endswith('.json'):
                filepath = os.path.join(root, filename)
                json_files_found += 1
                try:
                    with open(filepath, 'r') as f:
                        video_segments = json.load(f)

                        # Handle both list and dict formats
                        if isinstance(video_segments, list):
                            segments.extend(video_segments)
                        elif isinstance(video_segments, dict):
                            segments.append(video_segments)

                except Exception as e:
                    print(f"Warning: Failed to load {filepath}: {e}")

    print(f"Loaded {len(segments)} segments from {json_files_found} JSON files in {json_dir}")
    return segments
