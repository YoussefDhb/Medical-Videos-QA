"""
Hybrid Fusion Module

Combines dense (FAISS) and sparse (BM25) search results using different fusion strategies.
"""
import numpy as np
from typing import List, Dict


class HybridFusion:
    """
    Hybrid search result fusion combining dense and sparse retrieval.

    Supports multiple fusion strategies:
    - Linear: Weighted score combination
    - RRF: Reciprocal Rank Fusion

    Args:
        alpha: Weight for dense retrieval (0-1), (1-alpha) for sparse
    """

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha  # Weight for dense retrieval

    def fuse(self, dense_results: List[Dict], sparse_results: List[Dict],
             top_k: int = 10, strategy: str = 'linear', rrf_k: int = 60) -> List[Dict]:
        """
        Fuse dense and sparse search results.

        Args:
            dense_results: Results from dense retrieval (FAISS) with 'raw_score' and 'meta'
            sparse_results: Results from sparse retrieval (BM25) with 'bm25_score' and 'segment_id'
            top_k: Number of final results to return
            strategy: Fusion strategy ('linear' or 'rrf')
            rrf_k: Parameter for RRF (default: 60)

        Returns:
            List of fused results sorted by combined score
        """
        if strategy == 'rrf':
            return self._fuse_rrf(dense_results, sparse_results, top_k, rrf_k)
        else:
            return self._fuse_linear(dense_results, sparse_results, top_k)

    def _fuse_linear(self, dense_results: List[Dict], sparse_results: List[Dict],
                     top_k: int) -> List[Dict]:
        """
        Linear score-based fusion.

        Combined score = alpha * dense_score + (1-alpha) * sparse_score
        """
        # Normalize sparse (BM25) scores to [0, 1]
        if sparse_results:
            max_sparse = max(r['bm25_score'] for r in sparse_results)
            min_sparse = min(r['bm25_score'] for r in sparse_results)
            sparse_range = max_sparse - min_sparse

            if sparse_range > 0:
                for r in sparse_results:
                    r['bm25_score_normalized'] = (r['bm25_score'] - min_sparse) / sparse_range
            else:
                for r in sparse_results:
                    r['bm25_score_normalized'] = 0.0

        # Normalize dense scores (convert distance to similarity)
        for r in dense_results:
            dist = r.get('raw_score', float('inf'))
            # Use exponential decay for better score distribution
            r['dense_score_normalized'] = np.exp(-dist) if np.isfinite(dist) else 0.0

        # Create mapping of segment_id to scores
        sparse_scores = {r['segment_id']: r['bm25_score_normalized'] for r in sparse_results}

        # Combine scores
        combined_results = {}

        # Process dense results
        for r in dense_results:
            meta = r.get('meta', {}) or {}
            segment_id = meta.get('segment_id', meta.get('video_id', 'unknown'))

            dense_score = r['dense_score_normalized']
            sparse_score = sparse_scores.get(segment_id, 0.0)

            combined_score = self.alpha * dense_score + (1 - self.alpha) * sparse_score

            combined_results[segment_id] = {
                'segment_id': segment_id,
                'combined_score': combined_score,
                'dense_score': dense_score,
                'bm25_score': sparse_score,
                'metadata': meta,
                'fusion_method': 'linear'
            }

        # Add sparse-only results (not in dense results)
        for r in sparse_results:
            segment_id = r['segment_id']
            if segment_id not in combined_results:
                sparse_score = r['bm25_score_normalized']
                combined_score = (1 - self.alpha) * sparse_score

                combined_results[segment_id] = {
                    'segment_id': segment_id,
                    'combined_score': combined_score,
                    'dense_score': 0.0,
                    'bm25_score': sparse_score,
                    'metadata': r['metadata'],
                    'fusion_method': 'linear'
                }

        # Sort by combined score
        sorted_results = sorted(combined_results.values(),
                               key=lambda x: x['combined_score'],
                               reverse=True)

        return sorted_results[:top_k]

    def _fuse_rrf(self, dense_results: List[Dict], sparse_results: List[Dict],
                  top_k: int, k: int = 60) -> List[Dict]:
        """
        Reciprocal Rank Fusion (RRF).

        RRF score = sum(1 / (k + rank)) across all ranking methods

        Args:
            k: Constant for RRF (default: 60, standard in literature)
        """
        def get_rrf_score(rank: int, k: int) -> float:
            return 1.0 / (k + rank)

        combined_scores = {}

        # Process dense results (by rank)
        for rank, r in enumerate(dense_results):
            meta = r.get('meta', {}) or {}
            segment_id = meta.get('segment_id', meta.get('video_id', 'unknown'))

            if segment_id not in combined_scores:
                combined_scores[segment_id] = {
                    'segment_id': segment_id,
                    'rrf_score': 0.0,
                    'dense_rank': None,
                    'bm25_rank': None,
                    'metadata': meta
                }

            combined_scores[segment_id]['rrf_score'] += self.alpha * get_rrf_score(rank, k)
            combined_scores[segment_id]['dense_rank'] = rank + 1

        # Process sparse (BM25) results (by rank)
        for rank, r in enumerate(sparse_results):
            segment_id = r['segment_id']

            if segment_id not in combined_scores:
                combined_scores[segment_id] = {
                    'segment_id': segment_id,
                    'rrf_score': 0.0,
                    'dense_rank': None,
                    'bm25_rank': None,
                    'metadata': r['metadata']
                }

            combined_scores[segment_id]['rrf_score'] += (1 - self.alpha) * get_rrf_score(rank, k)
            combined_scores[segment_id]['bm25_rank'] = rank + 1

        # Sort by RRF score
        sorted_results = sorted(combined_scores.values(),
                               key=lambda x: x['rrf_score'],
                               reverse=True)

        # Normalize RRF scores to [0, 1] for compatibility with thresholds
        if sorted_results:
            max_rrf = sorted_results[0]['rrf_score']
            min_rrf = sorted_results[-1]['rrf_score']
            rrf_range = max_rrf - min_rrf
            
            for r in sorted_results:
                if rrf_range > 0:
                    r['rrf_score_normalized'] = (r['rrf_score'] - min_rrf) / rrf_range
                else:
                    r['rrf_score_normalized'] = 1.0 if max_rrf > 0 else 0.0
                
                r['combined_score'] = r['rrf_score_normalized']
                r['fusion_method'] = 'rrf'
        
        return sorted_results[:top_k]

    def analyze_contribution(self, hybrid_results: List[Dict], top_k: int = 10):
        """
        Analyze how sparse and dense retrieval contribute to final results.
        Useful for tuning alpha parameter.

        Args:
            hybrid_results: Fused results from fuse()
            top_k: Number of top results to analyze
        """
        print(f"\n{'='*80}")
        print("FUSION ANALYSIS (Top {})".format(top_k))
        print(f"{'='*80}")

        for i, result in enumerate(hybrid_results[:top_k], 1):
            segment_id = result['segment_id']
            combined = result['combined_score']

            if result['fusion_method'] == 'linear':
                dense = result.get('dense_score', 0.0)
                sparse = result.get('bm25_score', 0.0)

                dense_contrib = self.alpha * dense
                sparse_contrib = (1 - self.alpha) * sparse

                print(f"\n{i}. {segment_id}")
                print(f"   Combined: {combined:.4f}")
                print(f"   Dense contribution: {dense_contrib:.4f} (raw: {dense:.4f})")
                print(f"   Sparse (BM25) contribution: {sparse_contrib:.4f} (raw: {sparse:.4f})")

                if dense > sparse:
                    print(f"   → Dense-driven result")
                elif sparse > dense:
                    print(f"   → Sparse (BM25)-driven result")
                else:
                    print(f"   → Balanced result")

            else:  # RRF
                rrf = result.get('rrf_score', 0.0)
                dense_rank = result.get('dense_rank', 'N/A')
                sparse_rank = result.get('bm25_rank', 'N/A')

                print(f"\n{i}. {segment_id}")
                print(f"   RRF score: {rrf:.4f}")
                print(f"   Dense rank: {dense_rank}")
                print(f"   Sparse (BM25) rank: {sparse_rank}")

        print(f"\n{'='*80}")


class HybridSearchEngine:
    """
    Complete hybrid search engine combining dense and sparse retrieval.

    Convenience wrapper that integrates BM25Search and HybridFusion.

    Args:
        segments_data: List of segment dictionaries
        alpha: Weight for dense retrieval (default: 0.7)
        bm25_k1: BM25 k1 parameter (default: 1.5)
        bm25_b: BM25 b parameter (default: 0.75)
    """

    def __init__(self, segments_data: List[Dict], alpha: float = 0.7,
                 bm25_k1: float = 1.5, bm25_b: float = 0.75):
        from search.sparse_search import BM25Search, MedicalQueryExpander

        self.segments_data = segments_data
        self.alpha = alpha

        # Initialize BM25 search
        self.bm25_search = BM25Search(segments_data, k1=bm25_k1, b=bm25_b)

        # Initialize fusion engine
        self.fusion = HybridFusion(alpha=alpha)

        # Query expander
        self.query_expander = MedicalQueryExpander()

    def search(self, query: str, dense_results: List[Dict], top_k: int = 10,
               fusion: str = 'linear', expand_query: bool = True, rrf_k: int = 60) -> List[Dict]:
        """
        Perform hybrid search.

        Args:
            query: Search query
            dense_results: Results from dense retrieval (FAISS)
            top_k: Number of results to return
            fusion: Fusion strategy ('linear' or 'rrf')
            expand_query: Whether to expand medical query terms
            rrf_k: Parameter for RRF

        Returns:
            List of hybrid search results
        """
        # Expand query if requested
        if expand_query:
            expanded_query = self.query_expander.expand_query(query)
            print(f"Original query: {query}")
            print(f"Expanded query: {expanded_query}")
            query_to_use = expanded_query
        else:
            query_to_use = query

        # Get BM25 results
        sparse_results = self.bm25_search.search(query_to_use, top_k=len(self.segments_data))

        print(f"\nHybrid Search Stats:")
        print(f"  BM25 results (non-zero scores): {len(sparse_results)}")
        print(f"  Dense results: {len(dense_results)}")
        print(f"  Fusion strategy: {fusion}")
        print(f"  Alpha (dense weight): {self.alpha}")

        # Fuse results
        return self.fusion.fuse(dense_results, sparse_results, top_k, fusion, rrf_k)

    # Alias for backward compatibility
    def hybrid_search(self, query: str, dense_results: List[Dict], top_k: int = 10,
                     fusion: str = 'linear', expand_query: bool = True, rrf_k: int = 60) -> List[Dict]:
        """Alias for search() to maintain backward compatibility."""
        return self.search(query, dense_results, top_k, fusion, expand_query, rrf_k)

    def analyze_fusion_contribution(self, hybrid_results: List[Dict], top_k: int = 10):
        """Analyze fusion contribution (delegates to HybridFusion)."""
        self.fusion.analyze_contribution(hybrid_results, top_k)
