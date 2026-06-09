"""
search package

Modular search components for medical video QA:
- Dense search (FAISS-based vector search)
- Sparse search (BM25 keyword-based search)
- Hybrid fusion (combining dense + sparse results)
- Aggregation (segment/video-level result merging)
"""

from .dense_search import FaissIndex, EmbeddingModels
from .aggregation import (
    aggregate_results_by_segment,
    aggregate_results_by_video,
    print_segment_results,
    print_video_contexts,
    format_timestamp,
    extract_metadata
)
from .sparse_search import BM25Search, MedicalQueryExpander
from .hybrid_fusion import HybridFusion, HybridSearchEngine
from .utils import load_segments_from_json_dir
from .hierarchical_search import (
    refine_with_precise_timestamps,
    load_segments_from_json_files,
    hierarchical_search,
    get_extended_context
)

__all__ = [
    # Dense search
    'FaissIndex',
    'EmbeddingModels',

    # Sparse search
    'BM25Search',
    'MedicalQueryExpander',

    # Hybrid fusion
    'HybridFusion',
    'HybridSearchEngine',

    # Hierarchical search
    'refine_with_precise_timestamps',
    'load_segments_from_json_files',
    'hierarchical_search',
    'get_extended_context',

    # Aggregation
    'aggregate_results_by_segment',
    'aggregate_results_by_video',
    'print_segment_results',
    'print_video_contexts',
    'format_timestamp',
    'extract_metadata',

    # Utils
    'load_segments_from_json_dir'
]