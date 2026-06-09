"""
data_preparation.py
Data Preparation for MedVidQA VideoRAG Pipeline.

1. Load MedVidQA dataset.
2. Download YouTube videos.
3. Clean dataset by removing entries with failed downloads.
"""

import os
import json
import re
import shutil
import time
import random
from pytubefix import YouTube
from pytubefix.cli import on_progress

CLEANED_DIR = "MedVidQA_cleaned"
DATASET_DIR = "MedVidQA"
VIDEO_TRAIN_DIR = "videos_train"
VIDEO_TEST_DIR = "videos_test"
VIDEO_VAL_DIR = "videos_val"

# Path to the shared repository containing the original videos
# Replace or override by setting the environment variable `MEDVIDQA_SHARED_DIR`
SHARED_VIDEO_DIR = os.environ.get("MEDVIDQA_SHARED_DIR", "/capstor/store/cscs/swissai/a127/medvidqa_shared")

def load_medvidqa_dataset(json_path):
    """Load MedVidQA dataset from JSON file."""
    with open(json_path, "r") as f:
        data = json.load(f)
    return data

def find_video_in_shared(video_id: str, shared_dir: str) -> str | None:
    """Search the shared directory recursively for a file that contains the video_id in its filename.

    Returns the absolute path to the matched file or None if not found.
    """
    if not shared_dir or not os.path.exists(shared_dir):
        return None
    for root, _dirs, files in os.walk(shared_dir):
        for fname in files:
            # match if video_id is contained in filename (covers id.mp4, id_full.mp4, etc.)
            if video_id in fname:
                return os.path.join(root, fname)
    return None


def link_or_copy(src: str, dst: str) -> bool:
    """Try to create a hard link, fallback to symlink, then copy. Returns True on success."""
    try:
        # Prefer hard link (fast, space efficient) when possible
        os.link(src, dst)
        return True
    except Exception:
        try:
            os.symlink(src, dst)
            return True
        except Exception:
            try:
                shutil.copy2(src, dst)
                return True
            except Exception as e:
                print(f"Failed to link/copy from {src} to {dst}: {e}")
                return False

def download_video(video_url, save_dir, video_id, retries=3):
    """Download YouTube video using pytubefix. Returns True if successful.
    
    Includes exponential backoff and random jitter to avoid 'Too Many Requests' (429) errors.
    Attempts only once if the video is private or unavailable.
    """
    clients = ['ANDROID', 'WEB', 'MWEB']
    
    for attempt in range(retries):
        client = clients[attempt % len(clients)]
        try:
            yt = YouTube(
                video_url, 
                on_progress_callback=on_progress, 
                use_oauth=True, 
                allow_oauth_cache=True,
                client=client
            )
            
            stream = yt.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
            
            if stream:
                stream.download(output_path=save_dir, filename=f"{video_id}.mp4")
                print(f"  -> Downloaded {video_id} successfully.")
                time.sleep(random.uniform(10, 15)) 
                return True
            else:
                print(f"  -> No suitable stream found for {video_id}")
                return False
                
        except Exception as e:
            err_str = str(e).lower()
            # If video is private, unavailable, or deleted, don't bother retrying
            if any(x in err_str for x in ["private", "unavailable", "deleted", "age restricted", "removed"]):
                print(f"  -> Video {video_id} is {err_str}. Skipping further attempts.")
                return False
                
            if "429" in err_str or "too many requests" in err_str:
                wait_time = (2 ** attempt) * 60 + random.uniform(10, 30)
                print(f"  -> Rate limited (429). Waiting {wait_time:.1f}s before retry {attempt+1}/{retries}...")
                time.sleep(wait_time)
            else:
                print(f"  -> Attempt {attempt+1} failed for {video_id}: {e}")
                time.sleep(random.uniform(5, 10))
    
    return False

def get_unique_videos(dataset):
    """Return a dict of unique video_id -> video_url from dataset."""
    video_map = {}
    for entry in dataset:
        vid = entry.get("video_id")
        url = entry.get("video_url")
        if vid and url and vid not in video_map:
            video_map[vid] = url
    return video_map

def download_unique_videos(video_map, save_dir):
    """Download each unique video only once and track failures. Log unique video count and iteration. Skip if already downloaded."""
    failed_ids = []
    os.makedirs(save_dir, exist_ok=True)
    print(f"Found {len(video_map)} unique video ids. Starting processing in {save_dir} ...")
    for idx, (video_id, video_url) in enumerate(video_map.items(), 1):
        video_path = os.path.join(save_dir, f"{video_id}.mp4")
        if os.path.exists(video_path):
            print(f"[{idx}] {video_id} already downloaded. Skipping.")
            continue
        print(f"[{idx} / {len(video_map)}] Locating video: {video_id}")
        # 1. Try to find in shared repository first
        shared_file = find_video_in_shared(video_id, SHARED_VIDEO_DIR)
        if shared_file:
            print(f"  -> Found in shared dir: {shared_file}. Linking...")
            if link_or_copy(shared_file, video_path):
                continue
            else:
                print(f"  -> Failed to link/copy shared file for {video_id}")

        # 2. Fallback to YouTube download
        print(f"  -> Not found in shared dir. Downloading from YouTube: {video_url}")
        if not download_video(video_url, save_dir, video_id):
            print(f"  -> FAILED to get video {video_id}")
            failed_ids.append(video_id)
    return failed_ids

def clean_dataset(dataset, failed_ids):
    """Remove entries with failed video downloads."""
    return [entry for entry in dataset if entry.get("video_id") not in failed_ids]

def filter_by_embeddings(dataset, split, model_name=None):
    """Keep only entries for which both textual and visual embeddings exist.

    Always checks the `feature_extraction` directory (not model-specific directories).
    This ensures we filter based on the current pipeline run, not old cached embeddings.
    """
    # Always use the base feature_extraction directory
    base_dir = "feature_extraction"

    textual_dir = os.path.join(base_dir, "textual", split)
    visual_dir = os.path.join(base_dir, "visual", split)
    # Get set of video_ids for which both textual and visual embedding files exist
    textual_ids = set()
    visual_ids = set()
    if os.path.exists(textual_dir):
        for fname in os.listdir(textual_dir):
            if fname.endswith(".json"):
                vid = fname.split(".")[0]
                textual_ids.add(vid)
    if os.path.exists(visual_dir):
        for fname in os.listdir(visual_dir):
            if fname.endswith(".json"):
                vid = fname.split(".")[0]
                visual_ids.add(vid)
    valid_ids = textual_ids & visual_ids
    # Only keep entries whose video_id is in valid_ids
    return [entry for entry in dataset if entry.get("video_id") in valid_ids]

def _sanitize_for_filename(s: str) -> str:
    """Create a filesystem-safe string from a model name for filenames.

    Example: 'openai/whisper-small' -> 'openai_whisper_small'
    """
    return re.sub(r"[^A-Za-z0-9]+", "_", s)

def filter_json_by_embeddings(model_name: str):
    """Filter dataset JSON files to keep only entries with both textual and visual embeddings.

    This function reads the base {split}_cleaned.json files and filters them to keep
    only entries for which BOTH textual AND visual embedding JSON files exist in the
    feature_extraction directories. This ensures the filtered file contains only
    successfully processed videos.

    Writes new cleaned files in `CLEANED_DIR` named like
    `{split}_{sanitized_model}_cleaned.json`.

    Args:
        model_name: Name of the ASR model used for embeddings (e.g., 'openai/whisper-tiny')
    """
    sanitized = _sanitize_for_filename(model_name)
    os.makedirs(CLEANED_DIR, exist_ok=True)

    for split in ['train', 'val', 'test']:
        cleaned_path = os.path.join(CLEANED_DIR, f"{split}_cleaned.json")
        if not os.path.exists(cleaned_path):
            print(f"Warning: base cleaned file not found: {cleaned_path}. Skipping {split}.")
            continue

        # Load the base cleaned dataset
        with open(cleaned_path, "r") as f:
            data = json.load(f)
        print(f"\nProcessing {split}: {len(data)} entries in base cleaned file")

        # Filter by embeddings - keeps only entries with both textual AND visual embeddings
        # Note: model_name parameter is ignored - we always check 'feature_extraction' directory
        filtered = filter_by_embeddings(data, split)
        print(f"After filtering by embeddings: {len(filtered)} entries (successfully processed)")

        # Write the filtered file
        out_path = os.path.join(CLEANED_DIR, f"{split}_{sanitized}_cleaned.json")
        with open(out_path, "w") as out_f:
            json.dump(filtered, out_f, indent=2)
        print(f"Wrote filtered file: {out_path}")

def main():
    splits = ["train", "val", "test"]
    video_dirs = {"train": VIDEO_TRAIN_DIR, "val": VIDEO_VAL_DIR, "test": VIDEO_TEST_DIR}
    cleaned_data = {}

    for split in splits:
        json_path = os.path.join(DATASET_DIR, f"{split}.json")
        data = load_medvidqa_dataset(json_path)
        video_map = get_unique_videos(data)
        failed_ids = download_unique_videos(video_map, video_dirs[split])
        cleaned_data[split] = clean_dataset(data, failed_ids)

    os.makedirs(CLEANED_DIR, exist_ok=True)
    for split in splits:
        out_path = os.path.join(CLEANED_DIR, f"{split}_cleaned.json")
        with open(out_path, "w") as f:
            json.dump(cleaned_data[split], f, indent=2)

if __name__ == "__main__":
    main()
