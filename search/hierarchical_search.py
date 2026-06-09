"""
Hierarchical Search Module

Fine-grained timestamp refinement and context expansion for search results.
"""
import os
import json
from typing import List, Dict, Optional, Union


def refine_with_precise_timestamps(search_results: List[Dict],
                                   segments_data: Union[List[Dict], Dict[str, List[Dict]]],
                                   result_format: str = 'query_faiss') -> List[Dict]:
    """
    Core hierarchical search logic: refine coarse search results with precise timestamps.

    This function works with in-memory segment data and can be used by both the
    multimodal pipeline (single video) and query system (multi-video).

    Args:
        search_results: List of search results from FAISS
            For query_faiss format: [{'meta': {...}, 'score': float, ...}, ...]
            For multimodal format: [{'metadata': {...}, 'distance': float, ...}, ...]
        segments_data: List of segment dicts with overlapping_timestamps, or dict mapping video_id -> segments
        result_format: 'query_faiss' or 'multimodal' - determines input/output format

    Returns:
        List of refined results with precise_timestamp and additional fields
    """
    refined_results = []

    # Handle dict format (video_id -> segments mapping)
    if isinstance(segments_data, dict):
        segments_lookup = segments_data
    else:
        # Convert list to lookup dict for efficiency
        segments_lookup = {}
        for seg in segments_data:
            video_id = seg.get('video_id')
            if video_id:
                if video_id not in segments_lookup:
                    segments_lookup[video_id] = []
                segments_lookup[video_id].append(seg)

    for result in search_results:
        # Extract metadata based on format
        if result_format == 'query_faiss':
            meta = result.get('meta', {}) or {}
            score_key = 'score'
            score_value = result.get(score_key, float('inf'))
        else:  # multimodal format
            meta = result.get('metadata', {})
            if meta is None:
                refined_results.append(result)
                continue
            score_key = 'distance'
            score_value = result.get(score_key, float('inf'))

        video_id = meta.get('video_id')
        segment_id = meta.get('segment_id')

        if not segment_id:
            refined_results.append(result)
            continue

        # Find corresponding segment in data
        video_segments = segments_lookup.get(video_id, segments_data if not isinstance(segments_data, dict) else [])
        segment = next((s for s in video_segments if s.get('segment_id') == segment_id), None)

        if not segment:
            refined_results.append(result)
            continue

# Extract precise timestamp from overlapping_timestamps
        overlapping_ts = segment.get('overlapping_timestamps', [])

        if overlapping_ts and len(overlapping_ts) > 0:
            # ACCURATE INTERSECTION LOGIC:
            # Filter out None values that Whisper ASR might have returned for empty chunks
            starts = [ts[0] for ts in overlapping_ts if ts[0] is not None]
            ends = [ts[1] for ts in overlapping_ts if ts[1] is not None]
            
            # Fallback if all timestamps were corrupted/None
            if not starts or not ends:
                refined_results.append(result)
                continue
            
            # The most concentrated visual/textual information is in the overlapping core
            core_start = max(starts)
            core_end = min(ends)
            
            # If windows don't perfectly overlap internally, use the centroid of the tightest cluster
            if core_start >= core_end:
                core_start = sum(starts) / len(starts)
                core_end = sum(ends) / len(ends)
                
            precise_ts = [round(core_start, 2), round(core_end, 2)]

            if result_format == 'query_faiss':
                result_copy = result.copy()
                result_copy['precise_timestamp'] = precise_ts
                result_copy['all_timestamps'] = overlapping_ts
                result_copy['full_segment_text'] = segment.get('text', '')
                result_copy['window_info'] = segment.get('window_info', {})
                refined_results.append(result_copy)
            else:  # multimodal format
                refined_results.append({
                    score_key: score_value,
                    'metadata': meta,
                    'segment_id': segment_id,
                    'video_id': segment.get('video_id'),
                    'precise_timestamp': precise_ts,
                    'full_timestamp_range': segment.get('timestamp'),
                    'text': segment.get('text', ''),
                    'window_info': segment.get('window_info', {}),
                    'all_timestamps': overlapping_ts
                })
        else:
            refined_results.append(result)

    return refined_results


def load_segments_from_json_files(video_ids: Union[List[str], set],
                                  json_dir: str) -> Dict[str, List[Dict]]:
    """
    Load segment data from JSON files for multiple videos.
    Recursively searches through subdirectories (train/test/val) if needed.

    Args:
        video_ids: List of video IDs or set of video IDs
        json_dir: Directory containing JSON feature files (will search recursively)

    Returns:
        Dict mapping video_id -> list of segments
    """
    segments_by_video = {}

    for video_id in video_ids:
        # First try direct path
        json_path = os.path.join(json_dir, f"{video_id}.json")

        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    segments = json.load(f)
                    segments_by_video[video_id] = segments
                continue
            except Exception as e:
                print(f"Warning: Could not load segments for {video_id}: {e}")
                continue

        # If not found, search recursively through subdirectories
        # Common structure: feature_extraction/textual/{split}/{video_id}.json
        found = False
        for root, dirs, files in os.walk(json_dir):
            filename = f"{video_id}.json"
            if filename in files:
                json_path = os.path.join(root, filename)
                try:
                    with open(json_path, 'r') as f:
                        segments = json.load(f)
                        segments_by_video[video_id] = segments
                    found = True
                    break
                except Exception as e:
                    print(f"Warning: Could not load segments for {video_id} from {json_path}: {e}")
                    continue

        if not found:
            print(f"Warning: JSON file not found for video_id: {video_id} in {json_dir}")

    return segments_by_video


def hierarchical_search(query_emb, faiss_db, segments_data: Union[List[Dict], Dict[str, List[Dict]]],
                       top_k: int = 5, enable_fine_grained: bool = True,
                       result_format: str = 'multimodal') -> List[Dict]:
    """
    Complete hierarchical search: coarse FAISS search + fine-grained timestamp refinement.

    This is a convenience function that combines FAISS search with timestamp refinement.
    Can be used by both multimodal pipeline and query system.

    Args:
        query_emb: Query embedding vector
        faiss_db: FAISS database object with search() method
        segments_data: List of segments or dict mapping video_id -> segments
        top_k: Number of final results to return
        enable_fine_grained: If True, refine with precise timestamps
        result_format: 'query_faiss' or 'multimodal' - determines output format

    Returns:
        List of refined search results with precise timestamps
    """
    # Step 1: Coarse search - get more candidates than needed
    coarse_results = faiss_db.search(query_emb, top_k=top_k * 3)

    if not enable_fine_grained:
        return coarse_results[:top_k]

    # Step 2: Fine-grained refinement
    refined_results = refine_with_precise_timestamps(
        coarse_results,
        segments_data,
        result_format=result_format
    )

    # Step 3: Sort by score/distance and return top-k
    if result_format == 'query_faiss':
        refined_results.sort(key=lambda x: x.get('score', float('inf')))
    else:  # multimodal format
        refined_results.sort(key=lambda x: x.get('distance', float('inf')))

    return refined_results[:top_k]


def get_extended_context(segment_id: str,
                        segments_data: Union[List[Dict], Dict[str, List[Dict]]],
                        context_windows: int = 1) -> Optional[Dict]:
    """
    Retrieve surrounding windows for richer context.

    Args:
        segment_id: Current segment ID
        segments_data: List of all segments or dict mapping video_id -> segments
        context_windows: Number of adjacent windows to include on each side

    Returns:
        Dict with extended text and time range, or None if segment not found
    """
    # Convert dict to list if needed
    if isinstance(segments_data, dict):
        all_segments = []
        for segs in segments_data.values():
            all_segments.extend(segs)
        segments_data = all_segments

    current_segment = next((s for s in segments_data if s['segment_id'] == segment_id), None)
    if not current_segment:
        return None

    current_window_info = current_segment.get('window_info', {})
    current_end_token = current_window_info.get('end_token')
    current_start_token = current_window_info.get('start_token')
    video_id = current_segment.get('video_id')

    context_segments = [current_segment]

    # Get previous windows
    for _ in range(context_windows):
        prev_segment = None
        for segment in segments_data:
            if (segment.get('video_id') == video_id and
                segment.get('window_info', {}).get('end_token') == current_start_token):
                prev_segment = segment
                break

        if prev_segment:
            context_segments.insert(0, prev_segment)
            current_start_token = prev_segment.get('window_info', {}).get('start_token')
        else:
            break

    # Reset for next window search
    current_end_token = current_segment.get('window_info', {}).get('end_token')

    # Get next windows
    for _ in range(context_windows):
        next_segment = None
        for segment in segments_data:
            if (segment.get('video_id') == video_id and
                segment.get('window_info', {}).get('start_token') == current_end_token):
                next_segment = segment
                break

        if next_segment:
            context_segments.append(next_segment)
            current_end_token = next_segment.get('window_info', {}).get('end_token')
        else:
            break

    # Combine text
    full_context = " ".join([s.get('text', '') for s in context_segments])

    # Get time range
    all_timestamps = []
    for seg in context_segments:
        all_timestamps.extend(seg.get('overlapping_timestamps', []))

    if all_timestamps:
        time_range = (all_timestamps[0][0], all_timestamps[-1][1])
    else:
        time_range = current_segment.get('timestamp', (0, 0))

    return {
        'extended_text': full_context,
        'time_range': time_range,
        'segments': context_segments,
        'num_segments': len(context_segments)
    }
