"""
Aggregation Module

Segment and video-level result aggregation for multimodal search.
"""
import json
import os
from collections import defaultdict
import numpy as np
from search.hierarchical_search import load_segments_from_json_files, refine_with_precise_timestamps


def load_feedback_data(feedback_file='artifacts/human_feedback.json'):
    """Load and structure feedback data for easy lookup."""
    if not os.path.exists(feedback_file):
        return defaultdict(set)
    try:
        with open(feedback_file, 'r', encoding='utf-8') as f:
            feedbacks = json.load(f)
    except (json.JSONDecodeError, IOError):
        return defaultdict(set)

    feedback_map = defaultdict(set)
    for entry in feedbacks:
        query = entry.get('query', '').lower().strip()
        segment_id = entry.get('segment_id')
        if query and segment_id:
            feedback_map[query].add(segment_id)
    return feedback_map


def format_timestamp(ts):
    """
    Format timestamp tuple (start, end) to readable string.
    """
    if not ts or not isinstance(ts, (list, tuple)) or len(ts) != 2:
        return "unknown time"
    start, end = ts
    
    # SAFEFGUARD: Prevent string formatting crash on None
    if start is None or end is None:
        return "unknown time"
        
    return f"{start:.2f}s - {end:.2f}s"


def extract_metadata(result):
    """
    Extract useful metadata fields from a search result.

    Args:
        result: Search result dictionary

    Returns:
        Dictionary with extracted metadata fields
    """
    meta = result.get("meta", {}) or {}

    # Common fields across text and visual results
    video_id = meta.get("video_id", "unknown")
    timestamp = format_timestamp(meta.get("timestamp"))

    # Text-specific fields
    text = meta.get("text", "")
    entities = meta.get("entities", [])
    entity_text = ", ".join(ent[0] for ent in entities) if entities else ""

    # Sliding window specific metadata
    window_info = meta.get("window_info", {})
    processing_type = meta.get("processing_type", "unknown")

    # Handle both chunk_id (hybrid) and window_id (sliding window)
    chunk_id = meta.get("chunk_id")
    window_id = meta.get("window_id")
    mini_window_id = meta.get("mini_window_id")

    result_dict = {
        "video_id": video_id,
        "timestamp": timestamp,
        "text": text,
        "entities": entity_text,
        "processing_type": processing_type
    }

    # Add window information if available
    if window_info:
        result_dict["window_size"] = window_info.get("window_size")
        result_dict["window_range"] = f"tokens {window_info.get('start_token', 'N/A')}-{window_info.get('end_token', 'N/A')}"

    if chunk_id is not None:
        result_dict["chunk_id"] = chunk_id
    if window_id is not None:
        result_dict["window_id"] = window_id
    if mini_window_id is not None:
        result_dict["mini_window_id"] = mini_window_id

    return result_dict


def hierarchical_refine_timestamps(text_results, json_files_dir):
    """
    Refine search results with precise timestamps from overlapping_timestamps.

    Uses shared utility from hierarchical_search_utils.

    Args:
        text_results: List of raw search results
        json_files_dir: Directory containing JSON feature files

    Returns:
        List of results with precise_timestamp field added
    """
    # Extract unique video IDs from results
    video_ids = set()
    for result in text_results:
        meta = result.get('meta', {}) or {}
        video_id = meta.get('video_id')
        if video_id:
            video_ids.add(video_id)

    # Load segments for all relevant videos
    segments_by_video = load_segments_from_json_files(video_ids, json_files_dir)

    # Use shared refinement utility
    return refine_with_precise_timestamps(
        text_results,
        segments_by_video,
        result_format='query_faiss'
    )


def aggregate_results_by_segment(
    text_results,
    visual_results,
    top_k=10,
    text_weight=0.6,
    visual_weight=0.4,
    enable_hierarchical=False,
    json_dir=None,
    query: str = None,
    feedback_boost: float = 1.5
):
    """
    Aggregate multimodal search results by segment_id for precise multimodal linking.

    This leverages the segment_id created during embedding generation to precisely
    match text and visual embeddings from the same video segment.

    Args:
        text_results: List of dicts from textual index search
        visual_results: List of dicts from visual index search
        top_k: Number of top segments to return
        text_weight: Weight for textual similarity (default: 0.6)
        visual_weight: Weight for visual similarity (default: 0.4)
        enable_hierarchical: If True, refine with precise timestamps from JSON files
        json_dir: Directory containing JSON feature files (required if enable_hierarchical=True)
        query: The user's search query, used to retrieve relevant feedback.
        feedback_boost: The multiplier to apply to scores of user-endorsed segments.

    Returns:
        List of segment contexts sorted by relevance, each containing:
        - video_id
        - segment_id (unique identifier linking text and visual)
        - combined_score (weighted average when both modalities available, otherwise single modality)
        - text_evidence (text content, entities, similarity)
        - visual_evidence (frame info, similarity)
        - timestamp
        - precise_timestamp (if hierarchical search enabled)
    """
    # Apply hierarchical refinement if enabled
    if enable_hierarchical and json_dir:
        print("Applying hierarchical timestamp refinement...")
        text_results = hierarchical_refine_timestamps(text_results, json_dir)
        visual_results = hierarchical_refine_timestamps(visual_results, json_dir)

    # Load feedback data if a query is provided
    positive_segments = set()
    if query:
        feedback_map = load_feedback_data()
        normalized_query = query.lower().strip()
        positive_segments = feedback_map.get(normalized_query, set())

    # First, group by segment_id for precise multimodal linking
    segment_contexts = defaultdict(lambda: {
        "video_id": None,
        "segment_id": None,
        "split": "unknown",
        "timestamp": None,
        "text_evidence": None,
        "visual_evidence": None,
        "text_score": 0.0,
        "visual_score": 0.0,
        "combined_score": 0.0,
        "has_both_modalities": False
    })

    # Process textual results
    for result in text_results:
        # Compatibility check: handle both FAISS raw results and Hybrid Search fused results
        source_index = result.get("source_index", "")
        meta = result.get("meta", {}) or result.get("metadata", {}) or {}
        
        # If source_index is missing or generic (like 'hybrid'), try to get it from meta
        if (not source_index or source_index == "hybrid") and "source_index" in meta:
            source_index = meta["source_index"]

        split = "unknown"
        if "train" in source_index: split = "train"
        elif "val" in source_index: split = "val"
        elif "test" in source_index: split = "test"

        if 'combined_score' in result:
            # Result from HybridSearchEngine
            meta = result.get("metadata", {}) or {}
            similarity = result.get('combined_score', 0.0)
            dist = result.get('dense_score', 0.0) # Using dense_score as a distance proxy if needed
        else:
            # Raw result from FaissIndex
            meta = result.get("meta", {}) or {}
            # Normalize L2 distance to a 0-1 similarity score.
            # Max L2 distance for normalized vectors is 2.
            dist = result.get("raw_score", float('inf'))
            similarity = max(0.0, 1 - (dist / 2.0)) if np.isfinite(dist) else 0.0

        segment_id = meta.get("segment_id", f"{meta.get('video_id', 'unknown')}_seg_unknown")
        video_id = meta.get("video_id", "unknown")

        # Only update if this is a better text match for this segment
        if similarity > segment_contexts[segment_id]["text_score"]:
            segment_contexts[segment_id].update({
                "video_id": video_id,
                "segment_id": segment_id,
                "split": split, # Add split info
                "timestamp": meta.get("timestamp", "unknown"),
                "text_score": similarity,
                "text_evidence": {
                    "text": meta.get("text", ""),
                    "entities": meta.get("entities", []),
                    "similarity": similarity,
                    "raw_distance": dist,
                    "window_info": meta.get("window_info", {}),
                    "precise_timestamp": result.get('precise_timestamp') if enable_hierarchical else None,
                    "all_timestamps": result.get('all_timestamps') if enable_hierarchical else None
                }
            })

    # Process visual results (VISUAL DICTATES TIMESTAMP)
    for result in visual_results:
        meta = result.get("meta", {}) or {}
        segment_id = meta.get("segment_id", f"{meta.get('video_id', 'unknown')}_seg_unknown")
        video_id = meta.get("video_id", "unknown")
        
        source_index = result.get("source_index", "")
        if not source_index and "source_index" in meta:
            source_index = meta["source_index"]

        split = "unknown"
        if "train" in source_index: split = "train"
        elif "val" in source_index: split = "val"
        elif "test" in source_index: split = "test"

        dist = result.get("raw_score", float('inf'))
        similarity = max(0.0, 1 - (dist / 2.0)) if np.isfinite(dist) else 0.0

        if similarity > segment_contexts[segment_id]["visual_score"]:
            if segment_contexts[segment_id]["video_id"] is None:
                segment_contexts[segment_id]["video_id"] = video_id
                segment_contexts[segment_id]["segment_id"] = segment_id
                segment_contexts[segment_id]["split"] = split
                
            # CRITICAL: Always overwrite the timestamp with the visual timestamp.
            # Visual frames are instantaneous and precise, whereas text is a broad window.
            segment_contexts[segment_id]["timestamp"] = meta.get("timestamp", "unknown")

            segment_contexts[segment_id].update({
                "visual_score": similarity,
                "visual_evidence": {
                    "frame_info": meta.get("frame_path", "in-memory"),
                    "num_frames": meta.get("num_frames_averaged", 1),
                    "similarity": similarity,
                    "raw_distance": dist,
                    "sampling_strategy": meta.get("sampling_strategy", "unknown"),
                    "aggregation_method": meta.get("aggregation_method", "unknown")
                }
            })

    # Calculate combined scores (STRICT VISUAL DOMINANCE)
    for segment_id, ctx in segment_contexts.items():
        has_text = ctx["text_score"] > 0
        has_visual = ctx["visual_score"] > 0

        if has_text and has_visual:
            # Overrule standard defaults if they are passed. Force Visual Dominance.
            adaptive_text_weight = 0.2
            adaptive_visual_weight = 0.8

            ctx["fusion_mode"] = "visual_heavy_enforced"

            ctx["combined_score"] = (adaptive_text_weight * ctx["text_score"] +
                                    adaptive_visual_weight * ctx["visual_score"])
            ctx["weights_used"] = {
                "text": adaptive_text_weight,
                "visual": adaptive_visual_weight
            }
            ctx["has_both_modalities"] = True
        elif has_text:
            ctx["combined_score"] = ctx["text_score"] * 0.5  # Penalize text-only matches
            ctx["fusion_mode"] = "text_only_penalized"
        elif has_visual:
            ctx["combined_score"] = ctx["visual_score"]
            ctx["fusion_mode"] = "visual_only"
        else:
            ctx["combined_score"] = 0.0
            ctx["fusion_mode"] = "no_evidence"

        # Apply feedback boost if applicable
        if segment_id in positive_segments:
            ctx["combined_score"] = min(1.0, ctx["combined_score"] * feedback_boost)
            ctx["feedback_boosted"] = True
            ctx["fusion_mode"] = "feedback_boosted"

    # Sort by combined score and return top-k
    sorted_segments = sorted(
        [ctx for ctx in segment_contexts.values() if ctx["combined_score"] > 0],
        key=lambda x: (x.get("feedback_boosted", False), x["has_both_modalities"], x["combined_score"]),  # Prioritize boosted, then multimodal
        reverse=True
    )

    return sorted_segments[:top_k]


def aggregate_results_by_video(text_results, visual_results, top_k=5, text_weight=0.6, visual_weight=0.4, query: str = None):
    """
    Aggregate multimodal search results by video_id for video-level context.

    This groups all segments from the same video and provides comprehensive evidence.

    Args:
        text_results: List of dicts from textual index search
        visual_results: List of dicts from visual index search
        top_k: Number of top videos to return
        text_weight: Weight for textual similarity
        visual_weight: Weight for visual similarity
        query: The user's query for feedback-informed ranking.

    Returns:
        List of video contexts sorted by relevance with all matching segments
    """
    video_contexts = defaultdict(lambda: {
        "video_id": None,
        "segments": [],
        "best_combined_score": 0.0,
        "num_text_matches": 0,
        "num_visual_matches": 0,
        "num_multimodal_matches": 0
    })

    # First get segment-level aggregation
    segment_results = aggregate_results_by_segment(
        text_results, visual_results,
        top_k=len(text_results) + len(visual_results),  # Get all segments first
        text_weight=text_weight,
        visual_weight=visual_weight,
        query=query
    )

    # Group segments by video
    for segment in segment_results:
        video_id = segment["video_id"]

        if video_contexts[video_id]["video_id"] is None:
            video_contexts[video_id]["video_id"] = video_id

        video_contexts[video_id]["segments"].append(segment)
        video_contexts[video_id]["best_combined_score"] = max(
            video_contexts[video_id]["best_combined_score"],
            segment["combined_score"]
        )

        if segment["has_both_modalities"]:
            video_contexts[video_id]["num_multimodal_matches"] += 1
        if segment["text_evidence"]:
            video_contexts[video_id]["num_text_matches"] += 1
        if segment["visual_evidence"]:
            video_contexts[video_id]["num_visual_matches"] += 1

    # Sort segments within each video
    for ctx in video_contexts.values():
        ctx["segments"] = sorted(
            ctx["segments"],
            key=lambda x: x["combined_score"],
            reverse=True
        )

    # Sort videos by best segment score and multimodal coverage
    sorted_videos = sorted(
        video_contexts.values(),
        key=lambda x: (x["num_multimodal_matches"], x["best_combined_score"]),
        reverse=True
    )

    return sorted_videos[:top_k]


def print_segment_results(segment_contexts, query=None, output_file="multimodal_search_results.json"):
    """
    Pretty print segment-level multimodal search results.

    Args:
        segment_contexts: List of segment context dictionaries
        query: Original search query (optional)
        output_file: Path to save JSON results
    """
    if not segment_contexts:
        print("No segment contexts found.")
        return

    print("\n" + "=" * 100)
    print("MULTIMODAL SEGMENT-LEVEL SEARCH RESULTS")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Found {len(segment_contexts)} relevant segments")
    print("=" * 100)

    for i, ctx in enumerate(segment_contexts, 1):
        video_id = ctx["video_id"]
        segment_id = ctx["segment_id"]
        combined_score = ctx["combined_score"]
        timestamp = ctx["timestamp"]
        has_both = ctx["has_both_modalities"]

        modality_indicator = "🔗 [TEXT+VISUAL]" if has_both else ("📝 [TEXT]" if ctx["text_evidence"] else "🖼️ [VISUAL]")

        print(f"\n{i}. {modality_indicator} Video: {video_id} | Segment: {segment_id}")
        print(f"   Combined Score: {combined_score:.4f} (text: {ctx['text_score']:.4f}, visual: {ctx['visual_score']:.4f})")
        print(f"   ⏱️  Timestamp: {format_timestamp(timestamp) if isinstance(timestamp, (list, tuple)) else timestamp}")

        # Show textual evidence
        if ctx["text_evidence"]:
            text_ev = ctx["text_evidence"]
            print(f"\n   📝 Text Evidence (similarity: {text_ev['similarity']:.4f}):")

            # Show precise timestamp if available (hierarchical search)
            if text_ev.get('precise_timestamp'):
                precise = text_ev['precise_timestamp']
                print(f"      Precise timestamp: {precise[0]:.2f}s - {precise[1]:.2f}s")

            snippet = text_ev['text'][:200] + "..." if len(text_ev['text']) > 200 else text_ev['text']
            print(f"      {snippet}")

            if text_ev.get('entities'):
                entities_str = ", ".join([ent[0] if isinstance(ent, (list, tuple)) else str(ent)
                                         for ent in text_ev['entities'][:5]])
                print(f"      Medical Entities: {entities_str}")

            # Show coverage info if available
            window_info = text_ev.get('window_info', {})
            if window_info and window_info.get('coverage_contribution'):
                print(f"      Coverage: {window_info.get('coverage_contribution')} new tokens (window: {window_info.get('start_token')}-{window_info.get('end_token')})")

        # Show visual evidence
        if ctx["visual_evidence"]:
            vis_ev = ctx["visual_evidence"]
            print(f"\n   🖼️  Visual Evidence (similarity: {vis_ev['similarity']:.4f}):")
            print(f"      Frames averaged: {vis_ev['num_frames']}")

            # Display hyperparameters for analysis (useful during tuning)
            if 'sampling_strategy' in vis_ev or 'aggregation_method' in vis_ev:
                strategy = vis_ev.get('sampling_strategy', 'N/A')
                aggregation = vis_ev.get('aggregation_method', 'N/A')
                print(f"      Hyperparameters: {strategy} sampling, {aggregation} aggregation")

        print("-" * 100)

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump({
            "query": query,
            "total_segments": len(segment_contexts),
            "aggregation_mode": "segment",
            "results": segment_contexts
        }, f, indent=2, default=str)

    print(f"\n Results saved to: {output_file}")


def print_video_contexts(video_contexts, query=None, output_file="multimodal_search_results.json"):
    """
    Pretty print video-aggregated search results showing all matching segments.

    Args:
        video_contexts: List of video context dictionaries
        query: Original search query (optional)
        output_file: Path to save JSON results
    """
    if not video_contexts:
        print("No video contexts found.")
        return

    print("\n" + "=" * 100)
    print("MULTIMODAL VIDEO-LEVEL SEARCH RESULTS")
    print("=" * 100)
    print(f"Query: {query}")
    print(f"Found {len(video_contexts)} relevant videos")
    print("=" * 100)

    for i, ctx in enumerate(video_contexts, 1):
        video_id = ctx["video_id"]
        best_score = ctx["best_combined_score"]
        num_segments = len(ctx["segments"])
        num_multimodal = ctx["num_multimodal_matches"]

        print(f"\n{i}. 🎥 Video: {video_id}")
        print(f"   Best Score: {best_score:.4f} | Segments: {num_segments} (multimodal: {num_multimodal})")

        # Show top 3 segments from this video
        print(f"\n   🔝 Top Segments:")
        for j, seg in enumerate(ctx["segments"][:3], 1):
            timestamp = format_timestamp(seg["timestamp"]) if isinstance(seg["timestamp"], (list, tuple)) else seg["timestamp"]
            modality_str = "TEXT+VISUAL" if seg["has_both_modalities"] else ("TEXT" if seg["text_evidence"] else "VISUAL")

            print(f"\n      {j}. [{timestamp}] {modality_str} (score: {seg['combined_score']:.4f})")

            if seg["text_evidence"]:
                snippet = seg["text_evidence"]["text"][:120] + "..." if len(seg["text_evidence"]["text"]) > 120 else seg["text_evidence"]["text"]
                print(f"         📝 {snippet}")

            if seg["visual_evidence"]:
                print(f"         🖼️  {seg['visual_evidence']['num_frames']} frames averaged")

        if num_segments > 3:
            print(f"\n      ... and {num_segments - 3} more segments")

        print("-" * 100)

    # Save to JSON
    with open(output_file, "w") as f:
        json.dump({
            "query": query,
            "total_videos": len(video_contexts),
            "aggregation_mode": "video",
            "results": video_contexts
        }, f, indent=2, default=str)

    print(f"\n Results saved to: {output_file}")