"""
evaluation.py

Retrieval and temporal evaluation metrics for Medical VideoRAG.

Metrics:
- Precision@K: % relevant segments in top-K
- Recall@K: % of relevant segments retrieved in top-K
- F1@K: Harmonic mean of Precision@K and Recall@K
- mAP: Mean Average Precision
- nDCG@K: Normalized Discounted Cumulative Gain
- Temporal IoU/Precision/Recall/F1

Usage:
    from evaluation import AnswerEvaluator

    evaluator = AnswerEvaluator()
    metrics = evaluator.evaluate_temporal_overlap(
        predicted_timestamps=[(10, 20)],
        ground_truth_start=12,
        ground_truth_end=18
    )
"""

import numpy as np
from typing import List, Dict, Optional, Set, Tuple
import warnings

warnings.filterwarnings('ignore')


class AnswerEvaluator:
    """Evaluator for retrieval quality and temporal accuracy."""

    def __init__(self):
        """Initialize retrieval evaluator."""
        return

    def evaluate_retrieval(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str],
        k_values: Optional[List[int]] = None
    ) -> Dict:
        """
        Evaluate retrieval quality using Precision@K, Recall@K, F1@K, mAP, and nDCG@K.

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs
            k_values: List of K values to evaluate (default: [5, 10])

        Returns:
            {
                'precision@5': float,
                'recall@5': float,
                'f1@5': float,
                'precision@10': float,
                'recall@10': float,
                'f1@10': float,
                'mAP': float,
                'nDCG@5': float,
                'nDCG@10': float
            }
        """
        if k_values is None:
            k_values = [5, 10]

        results = {}

        # Precision@K, Recall@K, F1@K for each k
        for k in k_values:
            metrics = self._compute_retrieval_metrics(retrieved_segments, relevant_segment_ids, k)
            results[f'precision@{k}'] = metrics['precision']
            results[f'recall@{k}'] = metrics['recall']
            results[f'f1@{k}'] = metrics['f1']

        # Mean Average Precision
        results['mAP'] = self.compute_average_precision(
            retrieved_segments,
            relevant_segment_ids
        )

        # nDCG@K for each k
        for k in k_values:
            results[f'nDCG@{k}'] = self.compute_ndcg(
                retrieved_segments,
                relevant_segment_ids,
                k=k
            )

        return results

    def evaluate_temporal_overlap(
        self,
        predicted_timestamps: List[Tuple[float, float]],
        ground_truth_start: float,
        ground_truth_end: float
    ) -> Dict[str, float]:
        """
        Evaluate temporal overlap between predicted and ground truth timestamps.

        Args:
            predicted_timestamps: List of (start, end) tuples from retrieved segments
            ground_truth_start: Ground truth answer start time (seconds)
            ground_truth_end: Ground truth answer end time (seconds)

        Returns:
            {
                'iou': float,           # Intersection over Union (0-1)
                'temporal_precision': float,  # % of predicted time that overlaps
                'temporal_recall': float,     # % of ground truth time covered
                'temporal_f1': float,   # Harmonic mean of precision/recall
                'mean_distance': float  # Average distance from ground truth
            }
        """
        # Filter out invalid or None timestamps to prevent TypeError
        valid_timestamps = [
            ts for ts in predicted_timestamps
            if ts and len(ts) == 2 and all(isinstance(t, (int, float)) for t in ts)
        ]
        if not valid_timestamps:
            return {
                'iou': 0.0,
                'temporal_precision': 0.0,
                'temporal_recall': 0.0,
                'temporal_f1': 0.0,
                'mean_distance': float('inf')
            }

        gt_interval = (ground_truth_start, ground_truth_end)
        gt_duration = ground_truth_end - ground_truth_start

        # Calculate union of all predicted intervals
        predicted_union = self._merge_intervals(valid_timestamps)
        pred_duration = sum(end - start for start, end in predicted_union)

        # Calculate intersection with ground truth
        intersection_duration = 0.0
        for pred_start, pred_end in predicted_union:
            overlap_start = max(pred_start, ground_truth_start)
            overlap_end = min(pred_end, ground_truth_end)
            if overlap_start < overlap_end:
                intersection_duration += (overlap_end - overlap_start)

        # IoU (Intersection over Union)
        union_duration = pred_duration + gt_duration - intersection_duration
        iou = intersection_duration / union_duration if union_duration > 0 else 0.0

        # Temporal Precision: how much of predicted time overlaps with GT
        temporal_precision = intersection_duration / pred_duration if pred_duration > 0 else 0.0

        # Temporal Recall: how much of GT time is covered by predictions
        temporal_recall = intersection_duration / gt_duration if gt_duration > 0 else 0.0

        # Temporal F1
        if temporal_precision + temporal_recall > 0:
            temporal_f1 = 2 * (temporal_precision * temporal_recall) / (temporal_precision + temporal_recall)
        else:
            temporal_f1 = 0.0

        # Mean distance from ground truth center
        gt_center = (ground_truth_start + ground_truth_end) / 2
        distances = []
        for start, end in valid_timestamps:
            pred_center = (start + end) / 2
            distances.append(abs(pred_center - gt_center))
        mean_distance = np.mean(distances) if distances else float('inf')

        return {
            'iou': iou,
            'temporal_precision': temporal_precision,
            'temporal_recall': temporal_recall,
            'temporal_f1': temporal_f1,
            'mean_distance': mean_distance
        }

    def _merge_intervals(self, intervals: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        """Merge overlapping time intervals"""
        if not intervals:
            return []

        # Sort by start time
        sorted_intervals = sorted(intervals, key=lambda x: x[0])
        merged = [sorted_intervals[0]]

        for current in sorted_intervals[1:]:
            last = merged[-1]
            # If intervals overlap, merge them
            if current[0] <= last[1]:
                merged[-1] = (last[0], max(last[1], current[1]))
            else:
                merged.append(current)

        return merged

    def _parse_timestamp(self, timestamp_str: str) -> float:
        """Parse timestamp string (MM:SS or HH:MM:SS) to seconds"""
        try:
            parts = timestamp_str.strip().split(':')
            if len(parts) == 2:  # MM:SS
                minutes, seconds = parts
                return int(minutes) * 60 + int(seconds)
            elif len(parts) == 3:  # HH:MM:SS
                hours, minutes, seconds = parts
                return int(hours) * 3600 + int(minutes) * 60 + int(seconds)
            else:
                return float(timestamp_str)
        except Exception:
            return 0.0

    def compute_average_precision(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str]
    ) -> float:
        """
        Compute Average Precision for ranked retrieval results.

        AP = sum(P@k * rel(k)) / number_of_relevant_items

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs

        Returns:
            Average Precision score (0.0 to 1.0)
        """
        if not relevant_segment_ids:
            return 0.0

        num_relevant = len(relevant_segment_ids)
        precision_at_k = []
        num_relevant_seen = 0

        for k, seg in enumerate(retrieved_segments, 1):
            seg_id = seg.get('segment_id') or seg.get('evidence_id') or seg.get('meta', {}).get('segment_id')

            if seg_id in relevant_segment_ids:
                num_relevant_seen += 1
                precision_at_k.append(num_relevant_seen / k)

        if not precision_at_k:
            return 0.0

        return sum(precision_at_k) / num_relevant

    def compute_ndcg(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str],
        k: int = 10
    ) -> float:
        """
        Compute Normalized Discounted Cumulative Gain at K.

        Accounts for ranking quality - relevant items ranked higher get more weight.

        Args:
            retrieved_segments: List of retrieved segments (ordered by relevance)
            relevant_segment_ids: Set of ground truth relevant segment IDs
            k: Cutoff position (default: 10)

        Returns:
            nDCG@K score (0.0 to 1.0)
        """
        import math

        # DCG calculation
        dcg = 0.0
        for i, seg in enumerate(retrieved_segments[:k], 1):
            seg_id = seg.get('segment_id') or seg.get('evidence_id') or seg.get('meta', {}).get('segment_id')
            relevance = 1.0 if seg_id in relevant_segment_ids else 0.0
            dcg += relevance / math.log2(i + 1)

        # IDCG (ideal DCG - all relevant items at top)
        num_relevant = min(len(relevant_segment_ids), k)
        idcg = sum(1.0 / math.log2(i + 2) for i in range(num_relevant))

        return dcg / idcg if idcg > 0 else 0.0

    def _compute_retrieval_metrics(
        self,
        retrieved_segments: List[Dict],
        relevant_segment_ids: Set[str],
        k: int
    ) -> Dict[str, float]:
        """Compute Precision@K, Recall@K, F1@K"""

        # Extract segment IDs from retrieved segments
        retrieved_ids = set()
        for seg in retrieved_segments:
            seg_id = seg.get('segment_id') or seg.get('evidence_id')
            if seg_id:
                retrieved_ids.add(seg_id)

        # Calculate metrics
        relevant_in_topk = retrieved_ids & relevant_segment_ids

        precision = len(relevant_in_topk) / k if k > 0 else 0.0
        recall = len(relevant_in_topk) / len(relevant_segment_ids) if relevant_segment_ids else 0.0
        f1 = (2 * precision * recall) / (precision + recall + 1e-10)

        return {
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


# Example usage
if __name__ == "__main__":
    # This block now runs a live demonstration using a real query from the dataset
    # to provide a meaningful example of how the temporal evaluation metrics work.

    import argparse
    import os
    import json
    import sys
    from pathlib import Path
    from search import EmbeddingModels, aggregate_results_by_segment, FaissIndex

    parser = argparse.ArgumentParser(description="Demonstration of the AnswerEvaluator class's temporal evaluation.")
    parser.add_argument("--split", type=str, default="test", choices=["train", "test", "val"], help="Dataset split to use for the demo.")
    parser.add_argument("--query_index", type=int, default=0, help="Index of the query to use from the dataset.")
    args = parser.parse_args()

    print("Evaluation Demo")
    print("=" * 80)

    # 1. Initialize models and evaluator
    print("Initializing models...")
    models = EmbeddingModels()
    evaluator = AnswerEvaluator()

    # 2. Load a sample query from the specified dataset split
    print(f"Loading sample query from '{args.split}' split...")
    dataset_path = os.path.join('MedVidQA_cleaned', f'{args.split}_openai_whisper_tiny_cleaned.json')

    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}. Please run data_preparation.py first.")
        sys.exit(1)
    
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    if args.query_index >= len(dataset):
        print(f"Error: --query_index {args.query_index} is out of bounds for dataset with {len(dataset)} samples.")
        sys.exit(1)

    sample_item = dataset[args.query_index]
    query = sample_item['question']
    ground_truth = {
        'video_id': sample_item['video_id'],
        'answer_start': sample_item['answer_start_second'],
        'answer_end': sample_item['answer_end_second']
    }
    
    print(f"Using Query: '{query}'")
    print(f"Ground Truth: Video '{ground_truth['video_id']}', Time {ground_truth['answer_start']:.1f}s - {ground_truth['answer_end']:.1f}s")


    # 3. Perform a live search to get retrieval results
    print("\nPerforming live retrieval to get timestamps...")
    
    # Discover indices for the split
    faiss_dir = 'faiss_db'
    text_index_paths = [str(p) for p in Path(faiss_dir).glob(f"textual_{args.split}.index")]
    visual_index_paths = [str(p) for p in Path(faiss_dir).glob(f"visual_{args.split}.index")]

    q_vec_text = models.embed_text_bio(query)
    q_vec_clip = models.embed_text_clip(query)

    def search_pool(paths, vec, k=10):
        pool = []
        for p in paths:
            try:
                idx = FaissIndex(p)
                idx.load()
                pool.extend(r for r in idx.search(vec, k) if r.get("meta"))
            except Exception as e:
                print(f"Warning: Failed to search {p}: {e}")
        return pool

    text_pool = search_pool(text_index_paths, q_vec_text)
    visual_pool = search_pool(visual_index_paths, q_vec_clip)
    
    retrieved_segments = aggregate_results_by_segment(text_pool, visual_pool, top_k=10)
    print(f"Retrieved {len(retrieved_segments)} segments.")


    # 4. Demonstrate Temporal Overlap Evaluation on live results
    print("\n1. Temporal Overlap Evaluation")
    print("-" * 80)
    
    predicted_timestamps = [
        tuple(seg['timestamp']) for seg in retrieved_segments 
        if seg.get('timestamp') and isinstance(seg['timestamp'], list) and len(seg['timestamp']) == 2
    ]
    
    if not predicted_timestamps:
        print("No valid timestamps were retrieved for this query.")
    else:
        temporal_metrics = evaluator.evaluate_temporal_overlap(
            predicted_timestamps, 
            ground_truth_start=ground_truth['answer_start'], 
            ground_truth_end=ground_truth['answer_end']
        )

        for metric, value in temporal_metrics.items():
            print(f"  {metric}: {value:.4f}")

    print("\n" + "=" * 80)
    print("Demo complete!")
