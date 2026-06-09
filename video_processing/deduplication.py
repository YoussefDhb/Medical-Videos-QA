"""
Deduplication Module

Provides embedding deduplication utilities using similarity-based methods.
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def deduplicate_embeddings_similarity(embeddings_list, similarity_threshold=0.95):
    """
    Similarity-based deduplication for embeddings (hash-based already done during generation).

    Args:
        embeddings_list: List of embedding dictionaries (already hash-deduplicated)
        similarity_threshold: Cosine similarity threshold for near-duplicate detection

    Returns:
        Deduplicated list of embeddings
    """
    if len(embeddings_list) <= 1:
        return embeddings_list

    print(f"Applying similarity-based deduplication to {len(embeddings_list)} embeddings...")
    embeddings_matrix = np.vstack([emb_data['embedding'] for emb_data in embeddings_list])

    # Compute cosine similarity matrix
    similarity_matrix = cosine_similarity(embeddings_matrix)

    # Find near-duplicates
    to_remove = set()

    for i in range(len(similarity_matrix)):
        if i in to_remove:
            continue
        for j in range(i + 1, len(similarity_matrix)):
            if j in to_remove:
                continue
            if similarity_matrix[i][j] > similarity_threshold:
                # Keep the one with more entities or earlier timestamp
                emb_i = embeddings_list[i]
                emb_j = embeddings_list[j]

                entities_i = len(emb_i.get('entities', []))
                entities_j = len(emb_j.get('entities', []))

                if entities_i >= entities_j:
                    to_remove.add(j)
                else:
                    to_remove.add(i)

    # Remove near-duplicates
    final_deduplicated = [emb_data for i, emb_data in enumerate(embeddings_list)
                         if i not in to_remove]

    print(f"Similarity deduplication: Removed {len(to_remove)} near-duplicates")
    print(f"Final result: {len(final_deduplicated)} unique embeddings")

    return final_deduplicated
