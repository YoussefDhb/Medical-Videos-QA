"""
run_full_evaluation.py

Batch evaluation script for Medical VideoRAG system.
Evaluates retrieval performance across entire dataset splits.

Usage:
    # Evaluate test split only
    python run_full_evaluation.py --split test

    # Evaluate all splits
    python run_full_evaluation.py --split all

    # Evaluate with custom settings
    python run_full_evaluation.py --split test --alpha 0.3

    # Quick test on first 10 queries
    python run_full_evaluation.py --split test --max-queries 10
"""

from email import parser
import os
import json
import argparse
import numpy as np
import time
from datetime import datetime
from typing import List, Dict, Optional
from tqdm import tqdm
from pathlib import Path

# Import search and retrieval modules
from search import (
    FaissIndex,
    EmbeddingModels,
    aggregate_results_by_segment,
    HybridSearchEngine,
    load_segments_from_json_dir
)

# Import evaluation module
from evaluation import AnswerEvaluator



class FullEvaluator:
    """
    Batch evaluation runner for Medical VideoRAG.

    Loads models once and processes all queries efficiently.
    """

    def __init__(
        self,
        text_indices: List[str],
        visual_indices: List[str],
        split: str = 'all',
        device: Optional[str] = None,
        enable_hybrid: bool = True,
        alpha: float = 0.3,
        local_k: int = 50,
        final_k: int = 10
    ):
        """
        Initialize evaluator with models and indices.

        Args:
            text_indices: Paths to textual FAISS indices
            visual_indices: Paths to visual FAISS indices
            split: Dataset split ('train', 'test', 'val', or 'all')
            device: Device for models ('cpu', 'cuda', 'mps', or None for auto)
            enable_hybrid: Enable hybrid search (BM25 + dense)
            alpha: Weight for dense retrieval in hybrid search
            local_k: Number of results to fetch from each index
            final_k: Final number of combined results to return
        """
        self.device = device
        self.enable_hybrid = enable_hybrid
        self.alpha = alpha
        self.local_k = local_k
        self.final_k = final_k

        print("="*80)
        print("INITIALIZING FULL EVALUATION PIPELINE")
        print("="*80)

        # Load embedding models ONCE
        print("\n[1/4] Loading embedding models...")
        self.models = EmbeddingModels(device=device)
        print("Embedding models loaded")

        # Load FAISS indices ONCE
        print("\n[2/4] Loading FAISS indices...")
        self.text_indices = []
        for path in text_indices:
            idx = FaissIndex(path)
            idx.load()
            self.text_indices.append(idx)
            print(f"  ✅ Loaded {os.path.basename(path)}")

        self.visual_indices = []
        for path in visual_indices:
            idx = FaissIndex(path)
            idx.load()
            self.visual_indices.append(idx)
            print(f"  ✅ Loaded {os.path.basename(path)}")

        # Initialize hybrid search engine ONCE
        self.hybrid_engine = None
        if enable_hybrid:
            print("\n[3/4] Building BM25 index for hybrid search...")
            # Load segments for BM25 - must match the split to avoid data leakage
            # Since BM25 has 70% weight (1-alpha), using all splits would leak train/val data into test
            segments_data = []

            if split == 'all':
                # Load from all splits
                segments_data = load_segments_from_json_dir("feature_extraction/")
                print(f"  Loaded {len(segments_data)} segments for BM25 (all splits)")
            else:
                # Load only from the specific split to prevent data leakage
                # Structure: feature_extraction/textual/{split}/ and feature_extraction/visual/{split}/
                for modality in ['textual', 'visual']:
                    split_dir = os.path.join("feature_extraction", modality, split)
                    if os.path.exists(split_dir):
                        split_segments = load_segments_from_json_dir(split_dir)
                        segments_data.extend(split_segments)

                print(f"  Loaded {len(segments_data)} segments for BM25 ({split} split only)")
                print(f"  BM25 and FAISS both use {split.upper()} split - no data leakage")

            self.hybrid_engine = HybridSearchEngine(
                segments_data=segments_data,
                alpha=alpha
            )
            print("Hybrid search engine initialized")
        else:
            print("\n[3/4] Hybrid search disabled, skipping BM25 indexing")

        # Initialize evaluator ONCE
        print("\n[4/4] Initializing evaluator...")
        self.evaluator = AnswerEvaluator()
        print("Evaluator initialized")

        print("\n" + "="*80)
        print("INITIALIZATION COMPLETE - Ready for batch evaluation")
        print("="*80 + "\n")

    def search_query(self, query: str) -> List[Dict]:
        """
        Search for a single query across all indices.

        Args:
            query: User query text

        Returns:
            List of aggregated segment contexts
        """
        # Generate query embeddings
        q_vec_text = self.models.embed_text_bio(query, max_length=512)
        q_vec_clip = self.models.embed_text_clip(query)

        # Search text indices
        text_pool = []
        for idx in self.text_indices:
            res = idx.search(q_vec_text, top_k=self.local_k)
            for r in res:
                text_pool.append({
                    "raw_score": r.get("score"),
                    "meta": r.get("meta"),
                    "source_index": idx.index_path
                })

        # Search visual indices
        visual_pool = []
        for idx in self.visual_indices:
            res = idx.search(q_vec_clip, top_k=self.local_k)
            for r in res:
                visual_pool.append({
                    "raw_score": r.get("score"),
                    "meta": r.get("meta"),
                    "source_index": idx.index_path
                })

        # Apply hybrid search if enabled
        if self.enable_hybrid and text_pool:
            hybrid_text_results = self.hybrid_engine.hybrid_search(
                query=query,
                dense_results=text_pool,
                top_k=self.local_k,
                fusion='linear',
                expand_query=True
            )

            # Convert to text_pool format
            text_pool = []
            for hybrid_result in hybrid_text_results:
                meta = hybrid_result.get('metadata', {})
                combined_score = hybrid_result.get('combined_score', 0.0)
                raw_score = -np.log(combined_score + 1e-10)

                text_pool.append({
                    'raw_score': raw_score,
                    'meta': meta,
                    'source_index': 'hybrid',
                    'hybrid_info': {
                        'dense_score': hybrid_result.get('dense_score', 0.0),
                        'bm25_score': hybrid_result.get('bm25_score', 0.0),
                        'fusion_method': hybrid_result.get('fusion_method', 'linear')
                    }
                })

        # Normalize scores to similarities
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
                "hybrid_info": item.get("hybrid_info")
            })

        for item in visual_pool:
            dist = item.get("raw_score", float('inf'))
            sim = 1.0 / (1.0 + dist) if np.isfinite(dist) else 0.0
            combined.append({
                "sim": sim,
                "meta": item.get("meta"),
                "source_index": item.get("source_index"),
                "modal": "visual",
                "raw_score": dist
            })

        # Split combined results back into text and visual for proper aggregation
        text_only = [r for r in combined if r.get("modal") == "text"]
        visual_only = [r for r in combined if r.get("modal") == "visual"]

        # Aggregate by segment with separate text and visual results
        segment_contexts = aggregate_results_by_segment(
            text_results=text_only,
            visual_results=visual_only,
            top_k=self.final_k,
            text_weight=0.2,  # Let visual dictation shine in metrics
            visual_weight=0.8,
            enable_hierarchical=True,
            json_dir="feature_extraction/"
        )

        return segment_contexts

    def evaluate_query(self, query_data: Dict) -> Dict:
        """
        Evaluate a single query through the full pipeline.

        Args:
            query_data: Dictionary with 'question', 'video_id', 'answer_start_second', 'answer_end_second'

        Returns:
            Comprehensive evaluation results
        """
        query = query_data['question']
        ground_truth = {
            'video_id': query_data['video_id'],
            'answer_start': query_data['answer_start_second'],
            'answer_end': query_data['answer_end_second']
        }

        # Search
        search_start = time.time()
        segment_contexts = self.search_query(query)
        search_time = time.time() - search_start

        # Evaluate retrieval
        retrieval_metrics = self.evaluate_retrieval(segment_contexts, ground_truth)

        return {
            'query': query,
            'video_id': query_data['video_id'],
            'sample_id': query_data.get('sample_id'),
            'ground_truth': ground_truth,
            'retrieval_results': segment_contexts,
            'metrics': {
                'retrieval': retrieval_metrics,
                'timing': {
                    'search_time': search_time,
                    'total_time': search_time
                }
            }
        }

    def evaluate_retrieval(self, segments: List[Dict], ground_truth: Dict) -> Dict:
        """Evaluate retrieval quality against ground truth."""
        # Extract timestamps
        predicted_timestamps = []
        for seg in segments:
            if 'timestamp' in seg and seg['timestamp'] != 'unknown':
                ts = seg['timestamp']
                if isinstance(ts, list) and len(ts) == 2:
                    if ts[0] is not None and ts[1] is not None:  # SAFEGUARD ADDED
                        predicted_timestamps.append((ts[0], ts[1]))
            # Fallback to meta.timestamp
            elif 'meta' in seg and 'timestamp' in seg.get('meta', {}):
                meta = seg['meta']
                ts = meta['timestamp']
                if isinstance(ts, (list, tuple)) and len(ts) == 2:
                    if ts[0] is not None and ts[1] is not None:  # <-- ADDED SAFEGUARD
                        predicted_timestamps.append((ts[0], ts[1]))

        # Check correct video
        correct_video_retrieved = False
        for seg in segments:
            video_id = seg.get('video_id') or seg.get('meta', {}).get('video_id')
            if video_id == ground_truth['video_id']:
                correct_video_retrieved = True
                break

        # Determine relevant segments
        relevant_segment_ids = set()
        gt_start = ground_truth['answer_start']
        gt_end = ground_truth['answer_end']

        for seg in segments:
            video_id = seg.get('video_id') or seg.get('meta', {}).get('video_id')
            seg_id = seg.get('segment_id') or seg.get('meta', {}).get('segment_id')

            # Check if same video
            if video_id == ground_truth['video_id'] and seg_id:
                ts = seg.get('timestamp') or seg.get('meta', {}).get('timestamp')
                if ts and isinstance(ts, list) and len(ts) == 2:
                    seg_start, seg_end = ts[0], ts[1]
                    
                    # SAFEGUARD ADDED: Only compare if neither is None
                    if seg_start is not None and seg_end is not None:
                        overlap_start = max(seg_start, gt_start)
                        overlap_end = min(seg_end, gt_end)
                        if overlap_start < overlap_end:
                            relevant_segment_ids.add(seg_id)

        # Ranking metrics
        retrieval_metrics = {}
        if relevant_segment_ids:
            retrieval_metrics = self.evaluator.evaluate_retrieval(
                retrieved_segments=segments,
                relevant_segment_ids=relevant_segment_ids,
                k_values=[5, 10]
            )
        else:
            retrieval_metrics = {
                'precision@5': 0.0, 'recall@5': 0.0, 'f1@5': 0.0,
                'precision@10': 0.0, 'recall@10': 0.0, 'f1@10': 0.0,
                'mAP': 0.0, 'nDCG@5': 0.0, 'nDCG@10': 0.0
            }

        # Evaluate temporal metrics
        temporal_metrics = self.evaluator.evaluate_temporal_overlap(
            predicted_timestamps=predicted_timestamps,
            ground_truth_start=ground_truth['answer_start'],
            ground_truth_end=ground_truth['answer_end']
        )

        return {
            **temporal_metrics,
            **retrieval_metrics,
            'correct_video_retrieved': correct_video_retrieved,
            'num_retrieved': len(segments),
            'num_with_timestamps': len(predicted_timestamps),
            'num_relevant_segments': len(relevant_segment_ids)
        }


def load_dataset(split: str) -> List[Dict]:
    """
    Load dataset from MedVidQA_cleaned folder.

    Args:
        split: 'train', 'test', 'val', or 'all'

    Returns:
        List of question dictionaries
    """
    if split == 'all':
        splits = ['train', 'test', 'val']
    else:
        splits = [split]

    dataset = []
    for s in splits:
        filepath = f"MedVidQA_cleaned/{s}_openai_whisper_tiny_cleaned.json"
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                dataset.extend(data)
            print(f"  Loaded {len(data)} questions from {s} split")
        else:
            print(f"  File not found: {filepath}")

    return dataset


def aggregate_statistics(results: List[Dict]) -> Dict:
    """
    Compute aggregate statistics across all results.

    Args:
        results: List of per-query evaluation results

    Returns:
        Summary statistics
    """
    def compute_stats(values):
        """Compute mean, std, median, min, max."""
        if not values:
            return {'mean': 0.0, 'std': 0.0, 'median': 0.0, 'min': 0.0, 'max': 0.0, 'count': 0}
        arr = np.array(values)
        return {
            'mean': float(np.mean(arr)),
            'std': float(np.std(arr)),
            'median': float(np.median(arr)),
            'min': float(np.min(arr)),
            'max': float(np.max(arr)),
            'count': len(values)
        }

    # Extract all retrieval metrics
    retrieval_metrics = {}
    metric_names = ['recall@5', 'recall@10', 'precision@5', 'precision@10',
                   'f1@5', 'f1@10', 'mAP', 'nDCG@5', 'nDCG@10',
                   'iou', 'temporal_precision', 'temporal_recall', 'temporal_f1']

    for metric in metric_names:
        values = [r['metrics']['retrieval'].get(metric, 0.0) for r in results]
        retrieval_metrics[metric] = compute_stats(values)

    # Correct video retrieved rate
    correct_video_count = sum(1 for r in results if r['metrics']['retrieval'].get('correct_video_retrieved', False))
    retrieval_metrics['correct_video_rate'] = correct_video_count / len(results) if results else 0.0

    # Timing
    timing_metrics = {}
    for metric in ['search_time', 'total_time']:
        values = [r['metrics']['timing'].get(metric, 0.0) for r in results]
        timing_metrics[metric] = compute_stats(values)

    return {
        'retrieval': retrieval_metrics,
        'timing': timing_metrics
    }


def save_results(results: List[Dict], summary: Dict, output_dir: str, split: str, timestamp: str):
    """Save evaluation results and summary."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Save detailed results
    results_file = os.path.join(output_dir, f'evaluation_results_{split}_{timestamp}.json')
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {results_file}")

    # Save summary
    summary_file = os.path.join(output_dir, f'evaluation_summary_{split}_{timestamp}.json')
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"Summary statistics saved to: {summary_file}")

    # Generate markdown report
    report_file = os.path.join(output_dir, f'evaluation_report_{split}_{timestamp}.md')
    with open(report_file, 'w') as f:
        f.write(generate_report(summary, split, len(results)))
    print(f"Human-readable report saved to: {report_file}")


def generate_report(summary: Dict, split: str, num_queries: int) -> str:
    """Generate human-readable markdown report."""
    stats = summary.get('statistics', summary)  # Handle both nested and flat structures

    report = f"""# Evaluation Report: {split.upper()} Split

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Total Queries:** {num_queries}

---

## 1. Retrieval Performance

### Ranking Metrics
| Metric | Mean | Std | Median | Min | Max |
|--------|------|-----|--------|-----|-----|
| Recall@5 | {stats['retrieval']['recall@5']['mean']:.4f} | {stats['retrieval']['recall@5']['std']:.4f} | {stats['retrieval']['recall@5']['median']:.4f} | {stats['retrieval']['recall@5']['min']:.4f} | {stats['retrieval']['recall@5']['max']:.4f} |
| Recall@10 | {stats['retrieval']['recall@10']['mean']:.4f} | {stats['retrieval']['recall@10']['std']:.4f} | {stats['retrieval']['recall@10']['median']:.4f} | {stats['retrieval']['recall@10']['min']:.4f} | {stats['retrieval']['recall@10']['max']:.4f} |
| Precision@5 | {stats['retrieval']['precision@5']['mean']:.4f} | {stats['retrieval']['precision@5']['std']:.4f} | {stats['retrieval']['precision@5']['median']:.4f} | {stats['retrieval']['precision@5']['min']:.4f} | {stats['retrieval']['precision@5']['max']:.4f} |
| mAP | {stats['retrieval']['mAP']['mean']:.4f} | {stats['retrieval']['mAP']['std']:.4f} | {stats['retrieval']['mAP']['median']:.4f} | {stats['retrieval']['mAP']['min']:.4f} | {stats['retrieval']['mAP']['max']:.4f} |
| nDCG@10 | {stats['retrieval']['nDCG@10']['mean']:.4f} | {stats['retrieval']['nDCG@10']['std']:.4f} | {stats['retrieval']['nDCG@10']['median']:.4f} | {stats['retrieval']['nDCG@10']['min']:.4f} | {stats['retrieval']['nDCG@10']['max']:.4f} |

### Temporal Accuracy
| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| IoU | {stats['retrieval']['iou']['mean']:.4f} | {stats['retrieval']['iou']['std']:.4f} | {stats['retrieval']['iou']['median']:.4f} |
| Temporal Precision | {stats['retrieval']['temporal_precision']['mean']:.4f} | {stats['retrieval']['temporal_precision']['std']:.4f} | {stats['retrieval']['temporal_precision']['median']:.4f} |
| Temporal Recall | {stats['retrieval']['temporal_recall']['mean']:.4f} | {stats['retrieval']['temporal_recall']['std']:.4f} | {stats['retrieval']['temporal_recall']['median']:.4f} |
| Temporal F1 | {stats['retrieval']['temporal_f1']['mean']:.4f} | {stats['retrieval']['temporal_f1']['std']:.4f} | {stats['retrieval']['temporal_f1']['median']:.4f} |

**Correct Video Retrieved Rate:** {stats['retrieval']['correct_video_rate']:.2%}

---

## 2. Performance Timing

| Metric | Mean | Std | Median |
|--------|------|-----|--------|
| Search Time (s) | {stats['timing']['search_time']['mean']:.2f} | {stats['timing']['search_time']['std']:.2f} | {stats['timing']['search_time']['median']:.2f} |
| Total Time (s) | {stats['timing']['total_time']['mean']:.2f} | {stats['timing']['total_time']['std']:.2f} | {stats['timing']['total_time']['median']:.2f} |

**Total Evaluation Time:** {stats['timing']['total_time']['mean'] * num_queries / 60:.1f} minutes

---

## Summary

- **Queries Processed:** {num_queries}
- **Average Performance:** {stats['retrieval']['temporal_f1']['mean']:.2%} Temporal F1

"""
    return report


def main():
    parser = argparse.ArgumentParser(description="Batch evaluation for Medical VideoRAG")
    parser.add_argument("--split", type=str, required=True, choices=['train', 'test', 'val', 'all'],
                       help="Dataset split to evaluate")
    parser.add_argument("--alpha", type=float, default=0.85, 
                       help="Weight for dense retrieval...")
    parser.add_argument("--local-k", type=int, default=50,
                       help="Number of results from each index (default: 50)")
    parser.add_argument("--final-k", type=int, default=10,
                       help="Final number of combined results (default: 10)")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum number of queries to evaluate (for testing)")
    parser.add_argument("--output-dir", type=str, default="artifacts/evaluation_runs",
                       help="Output directory for results (default: artifacts/evaluation_runs)")
    parser.add_argument("--checkpoint-interval", type=int, default=50,
                       help="Save checkpoint every N queries (default: 50)")
    parser.add_argument("--device", type=str, default=None,
                       help="Device for models (cpu, cuda, mps, or None for auto)")

    args = parser.parse_args()

    # Generate timestamp for this evaluation run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("\n" + "="*80)
    print("MEDICAL VIDEORAG - FULL EVALUATION")
    print("="*80)
    print(f"Split: {args.split}")
    print(f"Hybrid Search: alpha={args.alpha}")
    print("="*80 + "\n")

    # Load dataset
    print("Loading dataset...")
    dataset = load_dataset(args.split)

    if args.max_queries:
        dataset = dataset[:args.max_queries]
        print(f"  Limited to first {args.max_queries} queries for testing\n")

    print(f"  Total queries to evaluate: {len(dataset)}\n")

    # Discover indices - load only the correct split's indices
    import glob
    if args.split == 'all':
        # For 'all' split, load all indices
        text_indices = sorted(glob.glob("faiss_db/textual_*.index"))
        visual_indices = sorted(glob.glob("faiss_db/visual_*.index"))
        print(f"  Loading ALL indices (train, val, test)\n")
    else:
        # For specific split, load only that split's indices
        text_indices = sorted(glob.glob(f"faiss_db/textual_{args.split}.index"))
        visual_indices = sorted(glob.glob(f"faiss_db/visual_{args.split}.index"))
        print(f"  Loading {args.split.upper()} indices only\n")

    if not text_indices or not visual_indices:
        print(f"\nERROR: No indices found for split '{args.split}'")
        print(f"   Expected: faiss_db/textual_{args.split}.index and faiss_db/visual_{args.split}.index")
        return

    # Initialize evaluator
    evaluator = FullEvaluator(
        text_indices=text_indices,
        visual_indices=visual_indices,
        split=args.split,
        device=args.device,
        enable_hybrid=True,
        alpha=args.alpha,
        local_k=args.local_k,
        final_k=args.final_k
    )

    # Run evaluation
    results = []
    errors = []

    print("\nStarting evaluation...")
    print("="*80 + "\n")

    for i, query_data in enumerate(tqdm(dataset, desc=f"Evaluating {args.split}")):
        try:
            result = evaluator.evaluate_query(query_data)
            results.append(result)

            # Checkpoint
            if (i + 1) % args.checkpoint_interval == 0:
                checkpoint_file = os.path.join(args.output_dir, f'checkpoint_{args.split}_{timestamp}_{i+1}.json')
                Path(args.output_dir).mkdir(parents=True, exist_ok=True)
                with open(checkpoint_file, 'w') as f:
                    json.dump(results, f, indent=2)
                print(f"\n  Checkpoint saved: {checkpoint_file}")

        except Exception as e:
            print(f"\n  Error on query {i}: {str(e)}")
            errors.append({
                'query_idx': i,
                'query': query_data.get('question', 'unknown'),
                'error': str(e)
            })

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"  Successful: {len(results)}/{len(dataset)}")
    print(f"  Errors: {len(errors)}")

    # Compute aggregate statistics
    print("\nComputing aggregate statistics...")
    summary = {
        'dataset_info': {
            'split': args.split,
            'total_questions': len(dataset),
            'successful_evaluations': len(results),
            'failed_evaluations': len(errors),
            'timestamp': datetime.now().isoformat()
        },
        'configuration': {
            'alpha': args.alpha,
            'local_k': args.local_k,
            'final_k': args.final_k
        },
        'statistics': aggregate_statistics(results),
        'errors': errors
    }

    # Save results
    save_results(results, summary, args.output_dir, args.split, timestamp)

    print("\n" + "="*80)
    print("All results saved successfully!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
