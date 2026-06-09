"""
query_faiss.py

Load FAISS indices and metadata from `faiss_db/` and run retrieval for a user query.

Features:
- Embed queries with BiomedCLIP text encoder to match training embeddings
- Embed queries with BiomedCLIP vision encoder (open_clip) to search visual FAISS indexes for cross-modal retrieval
- Support for sliding window, chunk overlapping, and hybrid embedding strategies
- CLI: choose index paths, top_k, and mode (text / visual / both)

Usage examples:
python query_faiss.py --query "laparoscopic surgery" --text_index faiss_db/textual_train.index --visual_index faiss_db/visual_train.index --top_k 5 --mode both

"""

from html import parser
import os
import json
import argparse
import numpy as np
import torch

# Import from search module
from search import (
    FaissIndex,
    EmbeddingModels,
    aggregate_results_by_segment,
    aggregate_results_by_video,
    print_segment_results,
    print_video_contexts,
    extract_metadata,
    HybridSearchEngine,
    load_segments_from_json_dir
)

# Import evaluation module
from evaluation import AnswerEvaluator

def find_ground_truth(query, dataset_path):
    """Find ground truth for a query in MedVidQA dataset"""
    try:
        with open(dataset_path, 'r') as f:
            dataset = json.load(f)

        # Find matching query (case-insensitive)
        query_lower = query.lower().strip()
        for sample in dataset:
            if sample['question'].lower().strip() == query_lower:
                return {
                    'video_id': sample['video_id'],
                    'answer_start': sample['answer_start_second'],
                    'answer_end': sample['answer_end_second'],
                    'sample_id': sample.get('sample_id')
                }

        print(f"No matching ground truth found for query: {query}")
        return None
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return None


def evaluate_retrieval(segments, ground_truth):
    """Evaluate retrieval quality against ground truth"""
    evaluator = AnswerEvaluator()

    # Extract timestamps from segments
    predicted_timestamps = []
    for seg in segments:
        # Check top-level timestamp (from aggregation)
        if 'timestamp' in seg and seg['timestamp'] != 'unknown':
            ts = seg['timestamp']
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                if ts[0] is not None and ts[1] is not None:  # <-- ADDED SAFEGUARD
                    predicted_timestamps.append((ts[0], ts[1]))
        # Fallback to meta.timestamp
        elif 'meta' in seg and 'timestamp' in seg.get('meta', {}):
            meta = seg['meta']
            ts = meta['timestamp']
            if isinstance(ts, (list, tuple)) and len(ts) == 2:
                if ts[0] is not None and ts[1] is not None:  # <-- ADDED SAFEGUARD
                    predicted_timestamps.append((ts[0], ts[1]))

    # Evaluate temporal overlap
    temporal_metrics = evaluator.evaluate_temporal_overlap(
        predicted_timestamps=predicted_timestamps,
        ground_truth_start=ground_truth['answer_start'],
        ground_truth_end=ground_truth['answer_end']
    )

    # Check if correct video retrieved
    correct_video_retrieved = False
    for seg in segments:
        video_id = seg.get('video_id') or seg.get('meta', {}).get('video_id')
        if video_id == ground_truth['video_id']:
            correct_video_retrieved = True
            break

    # Determine relevant segments: same video AND temporal overlap with ground truth
    relevant_segment_ids = set()
    gt_start = ground_truth['answer_start']
    gt_end = ground_truth['answer_end']

    for seg in segments:
        video_id = seg.get('video_id') or seg.get('meta', {}).get('video_id')
        seg_id = seg.get('segment_id') or seg.get('meta', {}).get('segment_id')

        # Check if same video
        if video_id == ground_truth['video_id'] and seg_id:
            # Check temporal overlap
            ts = seg.get('timestamp') or seg.get('meta', {}).get('timestamp')
            if ts and isinstance(ts, list) and len(ts) == 2:
                seg_start, seg_end = ts[0], ts[1]
                
                # <-- ADDED SAFEGUARD BEFORE COMPARING
                if seg_start is not None and seg_end is not None: 
                    # Check if there's any overlap
                    overlap_start = max(seg_start, gt_start)
                    overlap_end = min(seg_end, gt_end)
                    if overlap_start < overlap_end:  # There is overlap
                        relevant_segment_ids.add(seg_id)

    # Compute retrieval metrics (Precision@K, Recall@K, F1@K, mAP, nDCG@K)
    retrieval_metrics = {}
    if relevant_segment_ids:
        retrieval_metrics = evaluator.evaluate_retrieval(
            retrieved_segments=segments,
            relevant_segment_ids=relevant_segment_ids,
            k_values=[5, 10]
        )
    else:
        # No relevant segments found - return zeros
        retrieval_metrics = {
            'precision@5': 0.0,
            'recall@5': 0.0,
            'f1@5': 0.0,
            'precision@10': 0.0,
            'recall@10': 0.0,
            'f1@10': 0.0,
            'mAP': 0.0,
            'nDCG@5': 0.0,
            'nDCG@10': 0.0
        }

    return {
        'temporal_metrics': temporal_metrics,
        'retrieval_metrics': retrieval_metrics,
        'correct_video_retrieved': correct_video_retrieved,
        'num_retrieved': len(segments),
        'num_with_timestamps': len(predicted_timestamps),
        'num_relevant_segments': len(relevant_segment_ids)
    }


def print_evaluation_metrics(eval_result, ground_truth):
    """Print formatted evaluation metrics"""
    print(f"\n{'='*80}")
    print("EVALUATION METRICS")
    print(f"{'='*80}")

    print(f"\nGround Truth:")
    print(f"  Video ID: {ground_truth['video_id']}")
    print(f"  Answer Time: {ground_truth['answer_start']:.1f}s - {ground_truth['answer_end']:.1f}s")
    print(f"  Duration: {ground_truth['answer_end'] - ground_truth['answer_start']:.1f}s")

    print(f"\nRetrieval Statistics:")
    print(f"  Segments retrieved: {eval_result['num_retrieved']}")
    print(f"  Segments with timestamps: {eval_result['num_with_timestamps']}")
    print(f"  Relevant segments (with temporal overlap): {eval_result['num_relevant_segments']}")
    print(f"  Correct video retrieved: {'Yes' if eval_result['correct_video_retrieved'] else 'No'}")

    # Ranking Metrics
    rm = eval_result['retrieval_metrics']
    print(f"\nRanking Quality Metrics:")
    print(f"  mAP (Mean Average Precision): {rm['mAP']:.4f}")
    print(f"    → Measures ranking quality across all positions")
    print(f"    → Range: 0.0 (worst) to 1.0 (perfect ranking)")

    print(f"\n  nDCG@5: {rm['nDCG@5']:.4f}")
    print(f"  nDCG@10: {rm['nDCG@10']:.4f}")
    print(f"    → Normalized Discounted Cumulative Gain")
    print(f"    → Penalizes relevant items at lower ranks")
    print(f"    → Range: 0.0 (worst) to 1.0 (ideal ranking)")

    print(f"\n  Precision@5: {rm['precision@5']:.4f} ({rm['precision@5']*100:.1f}% of top-5 are relevant)")
    print(f"  Recall@5: {rm['recall@5']:.4f} ({rm['recall@5']*100:.1f}% of relevant segments found in top-5)")
    print(f"  F1@5: {rm['f1@5']:.4f}")

    print(f"\n  Precision@10: {rm['precision@10']:.4f} ({rm['precision@10']*100:.1f}% of top-10 are relevant)")
    print(f"  Recall@10: {rm['recall@10']:.4f} ({rm['recall@10']*100:.1f}% of relevant segments found in top-10)")
    print(f"  F1@10: {rm['f1@10']:.4f}")

    tm = eval_result['temporal_metrics']
    print(f"\nTemporal Accuracy:")
    print(f"  mIoU (Mean Intersection over Union): {tm['iou']:.4f}")
    print(f"    → Measures overlap between predicted and ground truth intervals")
    print(f"    → Range: 0.0 (no overlap) to 1.0 (perfect match)")

    print(f"\n  Temporal Precision: {tm['temporal_precision']:.4f}")
    print(f"    → Percentage of predicted time that overlaps with ground truth")
    print(f"    → {tm['temporal_precision']*100:.1f}% of retrieved segments are relevant")

    print(f"\n  Temporal Recall: {tm['temporal_recall']:.4f}")
    print(f"    → Percentage of ground truth time covered by predictions")
    print(f"    → Found {tm['temporal_recall']*100:.1f}% of the relevant content")

    print(f"\n  Temporal F1: {tm['temporal_f1']:.4f}")
    print(f"    → Harmonic mean of precision and recall")
    print(f"    → Best overall metric for temporal accuracy")

    if tm['mean_distance'] != float('inf'):
        print(f"\n  Mean Distance: {tm['mean_distance']:.2f}s")
        print(f"    → Average distance from ground truth center")
        print(f"    → Lower is better (closer to target)")

    # Performance assessment
    print(f"\n{'='*80}")
    print("PERFORMANCE ASSESSMENT")
    print(f"{'='*80}")

    if not eval_result['correct_video_retrieved']:
        print("CRITICAL: Correct video not retrieved")
        print("   → Consider: Increase top_k, adjust search weights, or review embeddings")
    elif tm['iou'] >= 0.7:
        print("EXCELLENT: High temporal overlap (IoU >= 0.7)")
        print("   → Retrieved segments closely match ground truth")
    elif tm['iou'] >= 0.5:
        print("GOOD: Moderate temporal overlap (IoU >= 0.5)")
        print("   → Retrieved segments partially match ground truth")
    elif tm['iou'] >= 0.3:
        print("FAIR: Low temporal overlap (IoU >= 0.3)")
        print("   → Consider: Adjust alpha, increase local_k, or enable hierarchical search")
    else:
        print("POOR: Very low temporal overlap (IoU < 0.3)")
        print("   → Recommendations:")
        print("      • Try hybrid search: --hybrid --alpha 0.3")
        print("      • Increase retrieval: --local_k 100")
        print("      • Enable hierarchical: --hierarchical")

    print(f"\n{'='*80}\n")


def pretty_print_results(results, query=None):
    """Format search results with rich metadata context"""
    if not results:
        print("No results found.")
        return []

    out = []
    print("\nTop Results:")
    print("=" * 100)

    for i, r in enumerate(results, 1):
        metadata = extract_metadata(r)
        modal = r.get("modal", "unknown")
        sim = float(r.get("sim", 0.0))
        source = os.path.basename(r.get("source_index", "unknown"))

        entry = {
            "rank": i,
            "similarity": f"{sim:.3f}",
            "modality": modal,
            "video_id": metadata["video_id"],
            "timestamp": metadata["timestamp"],
            "source": source,
            "processing_type": metadata.get("processing_type", "N/A")
        }

        # Add text snippet for textual results
        if modal == "text" and metadata["text"]:
            entry["transcript"] = metadata["text"][:200]
        if metadata["entities"]:
            entry["entities"] = metadata["entities"]

        # Add window information for debugging/analysis
        if "window_size" in metadata:
            entry["window_info"] = {
                "size": metadata["window_size"],
                "range": metadata.get("window_range", "N/A")
            }

        out.append(entry)

        # Print human-friendly format
        print(f"\n{i}. Video: {metadata['video_id']} ({modal}, sim={sim:.3f})")
        print(f"   Time: {metadata['timestamp']}")
        print(f"   Source: {source}")
        print(f"   Processing: {metadata.get('processing_type', 'N/A')}")

        if modal == "text" and metadata["text"]:
            # Show snippet with context
            snippet = metadata["text"][:250] + "..." if len(metadata["text"]) > 250 else metadata["text"]
            print(f"   Transcript: {snippet}")

        if metadata["entities"]:
            print(f"   Medical Entities: {metadata['entities']}")

        if "window_size" in metadata:
            print(f"   Window Info: {metadata.get('window_range', 'N/A')} (size: {metadata['window_size']})")

        print("-" * 100)

    # Save structured output to JSON
    output_file = "search_results.json"
    print(f"\nStructured results saved to: {output_file}")
    with open(output_file, "w") as f:
        json.dump({
            "query": query,
            "total_results": len(out),
            "results": out
        }, f, indent=2)

    return out


def perform_search(query, split="all", top_k=10, hybrid=True, models=None, text_weight=0.2, visual_weight=0.8):
    """
    Programmatic entry point for searching.
    Returns structured results for use in APIs/Web apps.
    """
    if models is None:
        models = EmbeddingModels()
    
    # Auto-discover indexes
    import glob
    
    if split == "unseen":
        # Unseen videos are those in val and test splits
        text_index_paths = sorted(glob.glob(os.path.join("faiss_db", "textual_val.index")) + 
                                 glob.glob(os.path.join("faiss_db", "textual_test.index")))
        visual_index_paths = sorted(glob.glob(os.path.join("faiss_db", "visual_val.index")) + 
                                   glob.glob(os.path.join("faiss_db", "visual_test.index")))
    elif split == "all":
        text_index_paths = sorted(glob.glob(os.path.join("faiss_db", "textual_*.index")))
        visual_index_paths = sorted(glob.glob(os.path.join("faiss_db", "visual_*.index")))
    else:
        # Standard splits: train, val, test
        text_index_paths = sorted(glob.glob(os.path.join("faiss_db", f"textual_{split}.index")))
        visual_index_paths = sorted(glob.glob(os.path.join("faiss_db", f"visual_{split}.index")))

    def search_index_pool(index_paths, query_vec, local_k=150):
        pool = []
        for p in index_paths:
            try:
                idx = FaissIndex(p)
                idx.load()
                res = idx.search(query_vec, top_k=local_k)
                for r in res:
                    pool.append({
                        "raw_score": r.get("score"),
                        "meta": r.get("meta"),
                        "source_index": p
                    })
            except Exception as e:
                print(f"Warning: failed to search index {p}: {e}")
        return pool

    q_vec_text = models.embed_text_bio(query) if text_index_paths else None
    q_vec_clip = models.embed_text_clip(query) if visual_index_paths else None

    # Use larger pool for better candidate coverage before reranking
    text_pool = search_index_pool(text_index_paths, q_vec_text, local_k=150) if q_vec_text is not None else []
    visual_pool = search_index_pool(visual_index_paths, q_vec_clip, local_k=150) if q_vec_clip is not None else []

    if hybrid and text_pool:
        # HybridSearchEngine requires segments_data for BM25 initialization
        # We extract metadata from the text_pool to provide this data
        segments_data = [r['meta'] for r in text_pool if r.get('meta')]
        if segments_data:
            # OPTIMIZATION: Use balanced alpha and RRF fusion for more robust hybrid retrieval
            engine = HybridSearchEngine(segments_data=segments_data, alpha=0.45)
            text_pool = engine.search(query, text_pool, top_k=100, fusion='rrf', expand_query=True)
        else:
            print("Warning: No metadata found in text_pool, skipping hybrid search enhancement.")

    # Multi-modal aggregation: Use weights provided
    segments = aggregate_results_by_segment(
        text_pool, visual_pool, top_k=top_k, 
        text_weight=text_weight, visual_weight=visual_weight,
        enable_hierarchical=True, json_dir="feature_extraction/",
        query=query
    )

    return {
        "query": query,
        "results": segments
    }

def main():
    parser = argparse.ArgumentParser(description="Query FAISS textual/visual indices using user query embeddings")
    parser.add_argument("--query", type=str, required=False, help="Text query to search (if omitted you'll be prompted)")
    parser.add_argument("--split", type=str, required=True, choices=["train", "test", "val", "all"], help="REQUIRED: Dataset split to search. Use 'test' for test set, 'train' for training set, 'val' for validation, or 'all' for demo/exploration (NOT recommended for evaluation - causes data mixing). Always specify split to ensure clear search scope.")
    parser.add_argument("--text_index", type=str, default=None, help="Path to textual FAISS index (optional; auto-discover in faiss_db/ if omitted)")
    parser.add_argument("--visual_index", type=str, default=None, help="Path to visual FAISS index (optional; auto-discover in faiss_db/ if omitted)")
    parser.add_argument("--final_k", type=int, default=10, help="FINAL number of combined results to return (default: 10)")
    parser.add_argument("--local_k", type=int, default=150, help="Number of results to fetch from each index before merging (default: 150)")
    parser.add_argument("--device", type=str, choices=["cpu","cuda","mps"], default=None, help="Device to run models on; if omitted auto-detects")
    parser.add_argument("--max_length", type=int, default=512, help="Maximum sequence length for query embedding (default: 512)")
    parser.add_argument("--mode", type=str, choices=["segment", "video"], default="segment", help="Result aggregation mode: 'segment' for segment-level multimodal linking (default), 'video' for video-level aggregation")
    parser.add_argument("--top_videos", type=int, default=5, help="Number of top videos to return in video aggregation mode (default: 5)")
    parser.add_argument("--text_weight", type=float, default=0.2, help="Weight for textual similarity in video aggregation (default: 0.2)")     
    parser.add_argument("--visual_weight", type=float, default=0.8, help="Weight for visual similarity in video aggregation (default: 0.8)")
    parser.add_argument("--filter_sampling", type=str, default=None, choices=["uniform", "adaptive", "quality_based"], help="Filter results by sampling strategy (optional)")
    parser.add_argument("--filter_aggregation", type=str, default=None, choices=["mean", "max"], help="Filter results by aggregation method (optional)")
    parser.add_argument("--hierarchical", default=True, action="store_true", help="Enable hierarchical search with fine-grained timestamp refinement")
    parser.add_argument("--json_dir", type=str, default="feature_extraction/", help="Directory containing JSON feature files (will search recursively through train/test/val)")

    # Hybrid search arguments
    parser.add_argument("--hybrid", action="store_false", default=True, help="Enable hybrid search (BM25 + dense embeddings) by default. Use --no-hybrid to disable.")
    parser.add_argument("--alpha", type=float, default=0.85, help="Weight for dense retrieval in hybrid search (0-1, default: 0.85). Strongly favors visual/dense embeddings over BM25 text.")
    parser.add_argument("--fusion", type=str, choices=["linear", "rrf"], default="rrf", help="Fusion strategy: 'linear' (score-based) or 'rrf' (rank-based, default). RRF is generally more robust.")
    parser.add_argument("--expand_query", action="store_false", default=True, help="Expand query with medical synonyms in BM25 search (default: True)")
    parser.add_argument("--analyze_fusion", action="store_true", help="Show detailed fusion analysis (BM25 vs dense contribution)")
    parser.add_argument("--output", type=str, default=None, help="Output file path for search results (default: auto-generated based on search type)")


    # Evaluation arguments
    parser.add_argument("--eval", action="store_true", help="Enable evaluation mode - provide ground truth to calculate metrics")
    parser.add_argument("--video_id", type=str, default=None, help="Ground truth video ID for evaluation")
    parser.add_argument("--answer_start", type=float, default=None, help="Ground truth answer start time (seconds) for evaluation")
    parser.add_argument("--answer_end", type=float, default=None, help="Ground truth answer end time (seconds) for evaluation")
    parser.add_argument("--dataset", type=str, default=None, help="Path to MedVidQA JSON file to auto-find ground truth for query")

    args = parser.parse_args()

    # Auto-generate output filename based on search type if not specified
    if args.output is None:
        if args.hybrid:
            fusion_suffix = f"_{args.fusion}_a{args.alpha:.1f}"
            args.output = f"multimodal_search_results_hybrid{fusion_suffix}.json"
        else:
            args.output = "multimodal_search_results_dense.json"

    # If query not provided, prompt the user
    if not args.query:
        try:
            args.query = input("Enter search query: ")
        except Exception:
            raise ValueError("No query provided")

    # Auto-discover indexes in faiss_db/ if explicit paths are not provided
    def discover_indexes(prefix, split="all"):
        import glob
        if split == "all":
            # Load all splits
            paths = sorted(glob.glob(os.path.join("faiss_db", f"{prefix}_*.index")))
        else:
            # Load only specific split to prevent data leakage
            paths = sorted(glob.glob(os.path.join("faiss_db", f"{prefix}_{split}.index")))
        return paths

    text_index_paths = [args.text_index] if args.text_index else discover_indexes("textual", args.split)
    visual_index_paths = [args.visual_index] if args.visual_index else discover_indexes("visual", args.split)

    # Print which indices are being loaded
    print(f"\n{'='*80}")
    if args.split == "all":
        print(f"SEARCH SCOPE: ALL SPLITS (train+val+test combined)")
        print(f"This mixes different dataset partitions - use for demo/exploration only")
        print(f"For evaluation, use --split test (or --split train/val)")
    else:
        print(f"SEARCH SCOPE: {args.split.upper()} SPLIT ONLY")
        print(f"Clean split isolation - results are from {args.split} set only")
    print(f"{'='*80}")

    if not text_index_paths and not visual_index_paths:
        raise ValueError("No FAISS indexes found in faiss_db/. Place textual_*.index or visual_*.index or pass explicit paths.")

    device = None
    if args.device == "cuda":
        device = torch.device("cuda")
    elif args.device == "mps":
        device = torch.device("mps")
    elif args.device == "cpu":
        device = torch.device("cpu")

    print(f"Loading embedding models on device: {device or 'auto'}")
    models = EmbeddingModels(device=device)

    # Search each index for a larger local pool, normalize distances to similarities, and merge both modalities
    def search_index_pool(index_paths, query_vec, local_k=150):
        pool = []
        for p in index_paths:
            try:
                idx = FaissIndex(p)
                idx.load()
                res = idx.search(query_vec, top_k=local_k)
                for r in res:
                    meta = r.get("meta")
                    if meta:
                        meta["source_index"] = p
                    pool.append({
                        "raw_score": r.get("score"),
                        "meta": meta,
                        "source_index": p
                    })
            except Exception as e:
                print(f"Warning: failed to search index {p}: {e}")
        return pool

    # Generate query embeddings
    print(f"\nQuery: '{args.query}'")
    print(f"Generating embeddings with max_length={args.max_length}...")

    q_vec_text = models.embed_text_bio(args.query, max_length=args.max_length) if text_index_paths else None
    q_vec_clip = models.embed_text_clip(args.query) if visual_index_paths else None

    print(f"Searching indices with local_k={args.local_k} per index...")
    text_pool = search_index_pool(text_index_paths, q_vec_text, local_k=args.local_k) if q_vec_text is not None else []
    visual_pool = search_index_pool(visual_index_paths, q_vec_clip, local_k=args.local_k) if q_vec_clip is not None else []

    print(f"Retrieved {len(text_pool)} text results and {len(visual_pool)} visual results")

    # Apply hybrid search if enabled
    if args.hybrid and text_pool:
        print(f"\n{'='*80}")
        print("HYBRID SEARCH ENABLED (BM25 + Dense Embeddings)")
        print(f"{'='*80}")
        print(f"Configuration: alpha={args.alpha:.2f} (dense weight), fusion={args.fusion}")
        print(f"Strategy: {int((1-args.alpha)*100)}% BM25 + {int(args.alpha*100)}% Dense")

        try:
            # Load segments for BM25 index - respect split to prevent data leakage
            segments_data = []
            if args.split == 'all':
                segments_data = load_segments_from_json_dir(args.json_dir)
                print(f"   Loading BM25 segments from all splits")
            else:
                # Load only from the specific split
                for modality in ['textual', 'visual']:
                    split_dir = os.path.join(args.json_dir, modality, args.split)
                    if os.path.exists(split_dir):
                        split_segments = load_segments_from_json_dir(split_dir)
                        segments_data.extend(split_segments)
                print(f"   Loading BM25 segments from {args.split} split only")

            # Initialize hybrid search engine
            hybrid_engine = HybridSearchEngine(
                segments_data=segments_data,
                alpha=args.alpha
            )

            # Perform hybrid search on text results
            hybrid_text_results = hybrid_engine.hybrid_search(
                query=args.query,
                dense_results=text_pool,
                top_k=args.local_k,
                fusion=args.fusion,
                expand_query=args.expand_query
            )

            # Show fusion analysis if requested
            if args.analyze_fusion:
                hybrid_engine.analyze_fusion_contribution(hybrid_text_results, top_k=args.final_k)

            # Replace text_pool with hybrid results
            text_pool = []
            for hybrid_result in hybrid_text_results:
                meta = hybrid_result.get('metadata', {})
                # Convert hybrid score to raw_score format
                combined_score = hybrid_result.get('combined_score', 0.0)
                # Convert back to distance-like metric (lower is better)
                raw_score = -np.log(combined_score + 1e-10)

                text_pool.append({
                    'raw_score': raw_score,
                    'meta': meta,
                    'source_index': 'hybrid',
                    'hybrid_info': {
                        'dense_score': hybrid_result.get('dense_score', 0.0),
                        'bm25_score': hybrid_result.get('bm25_score', 0.0),
                        'fusion_method': hybrid_result.get('fusion_method', 'unknown')
                    }
                })

            print(f"Hybrid search complete: {len(text_pool)} results")

        except Exception as e:
            print(f"Hybrid search failed: {e}")
            print("Falling back to dense-only search...")

    # Normalize raw distances to similarity in (0,1] using sim = 1/(1+dist)
    combined = []
    for item in text_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0
        combined.append({
            "sim": sim,
            "meta": item.get("meta"),
            "source_index": item.get("source_index"),
            "modal": "text",
            "raw_score": dist,
            "hybrid_info": item.get("hybrid_info")  # Preserve hybrid search info
        })

    for item in visual_pool:
        dist = item.get("raw_score", float('inf'))
        sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0

        # Optional: Filter by hyperparameters (useful for comparing experiments)
        meta = item.get("meta", {}) or {}
        if args.filter_sampling and meta.get("sampling_strategy") != args.filter_sampling:
            continue
        if args.filter_aggregation and meta.get("aggregation_method") != args.filter_aggregation:
            continue

        combined.append({
            "sim": sim,
            "meta": item.get("meta"),
            "source_index": item.get("source_index"),
            "modal": "visual",
            "raw_score": dist
        })

    # Choose result aggregation mode
    if args.mode == "video":
        # Video-level aggregation: group by video_id for comprehensive context
        print(f"\nAggregating results by video (top {args.top_videos} videos)")
        print(f"Using weights: text={args.text_weight}, visual={args.visual_weight}")

        video_contexts = aggregate_results_by_video(
            text_pool,
            visual_pool,
            top_k=args.top_videos,
            text_weight=args.text_weight,
            visual_weight=args.visual_weight
        )

        print_video_contexts(video_contexts, query=args.query, output_file=args.output)

    else:
        # Segment-level aggregation: precise multimodal linking via segment_id
        print(f"\nAggregating results by segment (top {args.final_k} segments)")
        print(f"Using weights: text={args.text_weight}, visual={args.visual_weight}")

        segment_contexts = aggregate_results_by_segment(
            text_pool,
            visual_pool,
            top_k=args.final_k,
            text_weight=args.text_weight,
            visual_weight=args.visual_weight,
            enable_hierarchical=args.hierarchical,
            json_dir=args.json_dir if args.hierarchical else None
        )

        print_segment_results(segment_contexts, query=args.query, output_file=args.output)

        # Evaluate retrieval quality if ground truth provided
        ground_truth = None

        # Try to find ground truth from dataset if provided
        if args.dataset:
            ground_truth = find_ground_truth(args.query, args.dataset)

        # Or use manually provided ground truth
        elif args.eval and args.video_id and args.answer_start is not None and args.answer_end is not None:
            ground_truth = {
                'video_id': args.video_id,
                'answer_start': args.answer_start,
                'answer_end': args.answer_end
            }

        # Perform evaluation if ground truth available
        if ground_truth:
            eval_result = evaluate_retrieval(segment_contexts, ground_truth)
            print_evaluation_metrics(eval_result, ground_truth)
        elif args.eval:
            print(f"\n\u26a0\ufe0f  Evaluation mode enabled but no ground truth provided.")
            print("   Use --dataset <path> to auto-find, or provide:")
            print("   --video_id <id> --answer_start <sec> --answer_end <sec>")

if __name__ == "__main__":
    main()