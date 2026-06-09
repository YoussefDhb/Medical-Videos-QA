"""
hyperparameter_tuning.py
Automated hyperparameter optimization for multimodal pipeline.

Tests different combinations of:
- Text embedding: window_size, stride, min_coverage_contribution, deduplication_mode
- Visual embedding: frames_per_segment, sampling_strategy, quality_filter, aggregation_method

Evaluates on metrics like:
- Coverage percentage
- Number of embeddings generated
- Search relevance (if ground truth available)
- Processing time
"""

import os
import json
import itertools
import time
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Fix for FFmpeg path on Clariden cluster
if 'FFMPEG_PREFIX' in os.environ:
    ffmpeg_bin = os.path.join(os.environ['FFMPEG_PREFIX'], 'bin')
    if os.path.exists(ffmpeg_bin) and ffmpeg_bin not in os.environ['PATH']:
        os.environ['PATH'] = f"{ffmpeg_bin}:{os.environ['PATH']}"
        print(f"Added {ffmpeg_bin} to PATH")

# Import your pipeline components
from video_processing.asr import transcribe_with_asr
from video_processing.text_embeddings import load_ner_and_embed_models, extract_entities_and_embed
from video_processing.visual_embeddings import extract_frames_and_embed
from video_processing.deduplication import deduplicate_embeddings_similarity
from embedding_storage import FaissDB
from search import aggregate_results_by_segment

# Hyperparameter search space
HYPERPARAMETER_GRID = {
    # Text embedding parameters
    'window_size': [128, 256, 384, 512],
    'stride': [64, 96, 128, 192, 256],
    'min_coverage_contribution': [0.05, 0.10, 0.15, 0.20, 0.25],
    'deduplication_mode': ['coverage', 'similarity', 'none'],

    # Visual embedding parameters
    'frames_per_segment': [1, 2, 3, 4, 5],
    'sampling_strategy': ['uniform', 'adaptive', 'quality_based'],
    'quality_filter': [True, False],
    'aggregation_method': ['mean', 'max']
}

# Evaluation metrics
class HyperparameterEvaluator:
    def __init__(self, video_path: str, ground_truth_queries: List[Dict] = None):
        """
        Initialize evaluator.

        Args:
            video_path: Path to test video
            ground_truth_queries: Optional list of {'query': str, 'relevant_segments': [...]}
        """
        self.video_path = video_path
        self.video_id = os.path.splitext(os.path.basename(video_path))[0]
        self.ground_truth_queries = ground_truth_queries or []
        self.results = []

    def evaluate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a single hyperparameter configuration.

        Returns metrics dict with:
        - text_coverage: % of tokens covered
        - text_embeddings_count: Number of text embeddings
        - visual_embeddings_count: Number of visual embeddings
        - processing_time: Total time in seconds
        - search_relevance: Average relevance score (if ground truth provided)
        """
        print(f"\n{'='*80}")
        print(f"Testing configuration:")
        for k, v in config.items():
            print(f"  {k}: {v}")
        print(f"{'='*80}")

        start_time = time.time()

        try:
            # Step 1: ASR (same for all configs)
            transcript_chunks = transcribe_with_asr(self.video_path, asr_model_id="openai/whisper-tiny")

            # Step 2: Text embeddings with current config using BiomedCLIP
            nlp, clip_model, clip_tokenizer = load_ner_and_embed_models()

            text_results = extract_entities_and_embed(
                transcript_chunks, nlp, clip_model, clip_tokenizer,
                video_id=self.video_id,
                window_size=config['window_size'],
                stride=config['stride'],
                min_coverage_contribution=config['min_coverage_contribution'],
                deduplication_mode=config['deduplication_mode']
            )

            # Extract coverage metrics from text results
            if text_results and 'window_info' in text_results[0]:
                total_tokens = text_results[0]['window_info']['total_tokens']
                covered_tokens = set()
                for r in text_results:
                    start = r['window_info']['start_token']
                    end = r['window_info']['end_token']
                    covered_tokens.update(range(start, end))
                text_coverage = (len(covered_tokens) / total_tokens * 100) if total_tokens > 0 else 0
            else:
                text_coverage = 0

            # Step 3: Visual embeddings with current config
            visual_results = extract_frames_and_embed(
                self.video_path, text_results, self.video_id,
                frames_per_segment=config['frames_per_segment'],
                sampling_strategy=config['sampling_strategy'],
                quality_filter=config['quality_filter'],
                aggregation_method=config['aggregation_method']
            )

            # Step 4: Apply visual deduplication
            visual_results = deduplicate_embeddings_similarity(visual_results, similarity_threshold=0.98)

            processing_time = time.time() - start_time

            # Step 5: Evaluate search relevance (if ground truth provided)
            search_relevance = None
            if self.ground_truth_queries:
                search_relevance = self._evaluate_search_quality(
                    text_results, visual_results, config, clip_model, clip_tokenizer
                )

            # Compile metrics
            metrics = {
                'config': config,
                'text_coverage': text_coverage,
                'text_embeddings_count': len(text_results),
                'visual_embeddings_count': len(visual_results),
                'processing_time': processing_time,
                'search_relevance': search_relevance,
                'text_dim': text_results[0]['embedding'].shape[0] if text_results else 0,
                'visual_dim': visual_results[0]['embedding'].shape[0] if visual_results else 0,
                'timestamp': datetime.now().isoformat()
            }

            # Print summary
            print(f"\nConfiguration completed:")
            print(f"  - Text coverage: {text_coverage:.2f}%")
            print(f"  - Text embeddings: {len(text_results)}")
            print(f"  - Visual embeddings: {len(visual_results)}")
            print(f"  - Processing time: {processing_time:.2f}s")
            if search_relevance is not None:
                print(f"  - Search relevance: {search_relevance:.4f}")

            self.results.append(metrics)
            return metrics

        except Exception as e:
            print(f"Configuration failed: {e}")
            error_metrics = {
                'config': config,
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
            self.results.append(error_metrics)
            return error_metrics

    def _evaluate_search_quality(self, text_results, visual_results, config, clip_model, clip_tokenizer):
        """
        Evaluate search quality using ground truth queries.

        Returns average relevance score across all queries (MRR).
        """
        if not self.ground_truth_queries:
            return None

        # Build temporary FAISS indices
        temp_dir = f"temp_faiss_{self.video_id}"
        os.makedirs(temp_dir, exist_ok=True)

        text_path = os.path.join(temp_dir, "text.index")
        visual_path = os.path.join(temp_dir, "visual.index")

        import torch

        # Create FAISS DBs
        text_db, visual_db = None, None
        if text_results:
            text_db = FaissDB(dim=text_results[0]['embedding'].shape[0], index_path=text_path)
            text_db.add([r['embedding'] for r in text_results], text_results)

        if visual_results:
            visual_db = FaissDB(dim=visual_results[0]['embedding'].shape[0], index_path=visual_path)
            visual_db.add([r['embedding'] for r in visual_results], visual_results)

        # Evaluate each query
        relevance_scores = []
        top_k = 5
        for query_info in self.ground_truth_queries:
            query = query_info['query']
            relevant_segments = set(query_info.get('relevant_segments', []))

            # Encode query
            tokens = clip_tokenizer([query])
            with torch.no_grad():
                query_emb = clip_model.encode_text(tokens).squeeze(0).cpu().numpy()

            # Search text DB
            raw_text_res = []
            if text_db:
                raw_text_res = text_db.search(query_emb, top_k=top_k, save_results=False)
            
            # Search visual DB
            raw_vis_res = []
            if visual_db:
                raw_vis_res = visual_db.search(query_emb, top_k=top_k, save_results=False)

            # Format for aggregate_results_by_segment
            fmt_text = [{'meta': r['metadata'], 'raw_score': r['distance']} for r in raw_text_res]
            fmt_vis = [{'meta': r['metadata'], 'raw_score': r['distance']} for r in raw_vis_res]

            segments = aggregate_results_by_segment(
                fmt_text, fmt_vis, top_k=top_k, text_weight=0.6, visual_weight=0.4
            )

            # Calculate MRR for this query
            score = 0.0
            for rank, seg in enumerate(segments, 1):
                if seg['segment_id'] in relevant_segments:
                    score = 1.0 / rank
                    break
            
            relevance_scores.append(score)

        # Cleanup temp files
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)

        return np.mean(relevance_scores) if relevance_scores else None

    def save_results(self, output_path: str):
        """Save all results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"\nResults saved to {output_path}")

    def get_best_config(self, metric: str = 'text_coverage', higher_is_better: bool = True) -> Dict:
        """
        Find best configuration based on a metric.

        Args:
            metric: Metric name to optimize ('text_coverage', 'search_relevance', etc.)
            higher_is_better: Whether higher values are better

        Returns:
            Best configuration dict
        """
        valid_results = [r for r in self.results if 'error' not in r and r.get(metric) is not None]

        if not valid_results:
            print("No valid results to analyze!")
            return None

        if higher_is_better:
            best = max(valid_results, key=lambda x: x[metric])
        else:
            best = min(valid_results, key=lambda x: x[metric])

        print(f"\n{'='*80}")
        print(f"BEST CONFIGURATION (optimizing {metric}):")
        print(f"{'='*80}")
        for k, v in best['config'].items():
            print(f"  {k}: {v}")
        print(f"\nMetrics:")
        for k, v in best.items():
            if k != 'config':
                print(f"  {k}: {v}")

        return best

    def plot_results(self, output_dir: str = "tuning_plots"):
        """Generate visualization plots for hyperparameter tuning results."""
        os.makedirs(output_dir, exist_ok=True)

        valid_results = [r for r in self.results if 'error' not in r]

        if not valid_results:
            print("No valid results to plot!")
            return

        # Plot 1: Coverage vs Number of Embeddings
        plt.figure(figsize=(10, 6))
        coverages = [r['text_coverage'] for r in valid_results]
        counts = [r['text_embeddings_count'] for r in valid_results]
        plt.scatter(counts, coverages, alpha=0.6)
        plt.xlabel('Number of Text Embeddings')
        plt.ylabel('Text Coverage (%)')
        plt.title('Coverage vs Embedding Count')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'coverage_vs_count.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 2: Window size impact
        window_sizes = {}
        for r in valid_results:
            ws = r['config']['window_size']
            if ws not in window_sizes:
                window_sizes[ws] = []
            window_sizes[ws].append(r['text_coverage'])

        plt.figure(figsize=(10, 6))
        plt.boxplot([window_sizes[ws] for ws in sorted(window_sizes.keys())],
                   labels=[str(ws) for ws in sorted(window_sizes.keys())])
        plt.xlabel('Window Size')
        plt.ylabel('Text Coverage (%)')
        plt.title('Impact of Window Size on Coverage')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'window_size_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 3: Stride impact
        strides = {}
        for r in valid_results:
            s = r['config']['stride']
            if s not in strides:
                strides[s] = []
            strides[s].append(r['text_coverage'])

        plt.figure(figsize=(10, 6))
        plt.boxplot([strides[s] for s in sorted(strides.keys())],
                   labels=[str(s) for s in sorted(strides.keys())])
        plt.xlabel('Stride')
        plt.ylabel('Text Coverage (%)')
        plt.title('Impact of Stride on Coverage')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(output_dir, 'stride_impact.png'), dpi=300, bbox_inches='tight')
        plt.close()

        # Plot 4: Processing time comparison
        plt.figure(figsize=(12, 6))
        times = [r['processing_time'] for r in valid_results]
        labels = [f"Config {i+1}" for i in range(len(valid_results))]
        plt.bar(range(len(times)), times)
        plt.xlabel('Configuration')
        plt.ylabel('Processing Time (s)')
        plt.title('Processing Time Comparison')
        plt.xticks(range(len(times)), labels, rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'processing_time.png'), dpi=300, bbox_inches='tight')
        plt.close()

        print(f"\nPlots saved to {output_dir}/")


def grid_search(evaluator: HyperparameterEvaluator,
                param_subset: Dict[str, List] = None,
                max_configs: int = None) -> List[Dict]:
    """
    Perform grid search over hyperparameter space.

    Args:
        evaluator: HyperparameterEvaluator instance
        param_subset: Optional subset of parameters to search (uses HYPERPARAMETER_GRID if None)
        max_configs: Maximum number of configurations to test (random sample if exceeded)

    Returns:
        List of evaluation results
    """
    grid = param_subset or HYPERPARAMETER_GRID

    # Generate all combinations
    keys = list(grid.keys())
    values = list(grid.values())
    all_combinations = list(itertools.product(*values))

    print(f"\n{'='*80}")
    print(f"GRID SEARCH: {len(all_combinations)} total configurations")
    print(f"{'='*80}")

    # Sample if too many configs
    if max_configs and len(all_combinations) > max_configs:
        print(f"Sampling {max_configs} random configurations...")
        import random
        all_combinations = random.sample(all_combinations, max_configs)

    # Test each configuration
    for idx, combo in enumerate(all_combinations, 1):
        config = dict(zip(keys, combo))
        print(f"\n[{idx}/{len(all_combinations)}] Testing configuration...")
        evaluator.evaluate_config(config)

    return evaluator.results


def random_search(evaluator: HyperparameterEvaluator,
                 n_iterations: int = 20,
                 param_subset: Dict[str, List] = None) -> List[Dict]:
    """
    Perform random search over hyperparameter space.

    Args:
        evaluator: HyperparameterEvaluator instance
        n_iterations: Number of random configurations to test
        param_subset: Optional subset of parameters to search

    Returns:
        List of evaluation results
    """
    import random

    grid = param_subset or HYPERPARAMETER_GRID

    print(f"\n{'='*80}")
    print(f"RANDOM SEARCH: {n_iterations} random configurations")
    print(f"{'='*80}")

    for i in range(n_iterations):
        # Sample random configuration
        config = {k: random.choice(v) for k, v in grid.items()}
        print(f"\n[{i+1}/{n_iterations}] Testing configuration...")
        evaluator.evaluate_config(config)

    return evaluator.results


def quick_test_configs(evaluator: HyperparameterEvaluator) -> List[Dict]:
    """
    Test a few pre-selected configurations for quick validation.

    Includes:
    - Baseline (conservative)
    - Fast (minimal processing)
    - Quality (maximize coverage/quality)
    - Balanced (compromise)
    """
    configs = [
        {
            'name': 'Baseline',
            'window_size': 256,
            'stride': 192,
            'min_coverage_contribution': 0.15,
            'deduplication_mode': 'coverage',
            'frames_per_segment': 2,
            'sampling_strategy': 'uniform',
            'quality_filter': False,
            'aggregation_method': 'mean'
        },
        {
            'name': 'Fast',
            'window_size': 128,
            'stride': 128,
            'min_coverage_contribution': 0.20,
            'deduplication_mode': 'aggressive',
            'frames_per_segment': 1,
            'sampling_strategy': 'uniform',
            'quality_filter': False,
            'aggregation_method': 'mean'
        },
        {
            'name': 'Quality',
            'window_size': 384,
            'stride': 96,
            'min_coverage_contribution': 0.10,
            'deduplication_mode': 'coverage',
            'frames_per_segment': 4,
            'sampling_strategy': 'quality_based',
            'quality_filter': True,
            'aggregation_method': 'mean'
        },
        {
            'name': 'Balanced',
            'window_size': 256,
            'stride': 128,
            'min_coverage_contribution': 0.15,
            'deduplication_mode': 'coverage',
            'frames_per_segment': 2,
            'sampling_strategy': 'adaptive',
            'quality_filter': False,
            'aggregation_method': 'mean'
        }
    ]

    print(f"\n{'='*80}")
    print(f"QUICK TEST: {len(configs)} preset configurations")
    print(f"{'='*80}")

    for config in configs:
        name = config.pop('name')
        print(f"\nTesting '{name}' configuration...")
        evaluator.evaluate_config(config)

    return evaluator.results


# Example usage
def main():
    """Example hyperparameter tuning workflow."""

    # Configuration
    # Find first available video in train split
    video_dir = "videos_train"
    available_videos = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    
    if not available_videos:
        print(f"No videos found in {video_dir}. Please run data_preparation.py first.")
        return

    VIDEO_PATH = os.path.join(video_dir, available_videos[0])
    print(f" usando test video: {VIDEO_PATH}")
    
    OUTPUT_DIR = "hyperparameter_tuning_results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Ground truth queries (optional, disabled by default for tuning)
    ground_truth_queries = []

    # Initialize evaluator
    evaluator = HyperparameterEvaluator(VIDEO_PATH, ground_truth_queries)

    # Choose search strategy:

    # Option 1: Quick test with preset configs (recommended for initial exploration)
    print("\n" + "="*80)
    print("Running QUICK TEST with preset configurations...")
    print("="*80)
    quick_test_configs(evaluator)

    # Option 2: Random search (good balance of coverage and speed)
    # print("\n" + "="*80)
    # print("Running RANDOM SEARCH...")
    # print("="*80)
    # random_search(evaluator, n_iterations=20)

    # Option 3: Grid search (exhaustive but slow)
    # WARNING: This will test ALL combinations - can be very slow!
    # Consider using param_subset or max_configs to limit scope
    # print("\n" + "="*80)
    # print("Running GRID SEARCH...")
    # print("="*80)
    # limited_grid = {
    #     'window_size': [256, 384],
    #     'stride': [128, 192],
    #     'min_coverage_contribution': [0.15],
    #     'deduplication_mode': ['coverage'],
    #     'frames_per_segment': [2, 3],
    #     'sampling_strategy': ['uniform', 'adaptive'],
    #     'quality_filter': [False],
    #     'aggregation_method': ['mean']
    # }
    # grid_search(evaluator, param_subset=limited_grid)

    # Save results
    results_path = os.path.join(OUTPUT_DIR, f"tuning_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    evaluator.save_results(results_path)

    # Analyze results
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)

    # Find best configs for different metrics
    best_coverage = evaluator.get_best_config(metric='text_coverage', higher_is_better=True)
    best_speed = evaluator.get_best_config(metric='processing_time', higher_is_better=False)

    if evaluator.ground_truth_queries:
        best_relevance = evaluator.get_best_config(metric='search_relevance', higher_is_better=True)

    # Generate plots
    evaluator.plot_results(output_dir=os.path.join(OUTPUT_DIR, "plots"))

    print(f"\n{'='*80}")
    print("TUNING COMPLETE!")
    print(f"{'='*80}")
    print(f"Results saved to: {results_path}")
    print(f"Plots saved to: {os.path.join(OUTPUT_DIR, 'plots')}/")


if __name__ == "__main__":
    main()
