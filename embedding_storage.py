"""
embedding_storage.py
Helpers to persist extracted embeddings and metadata.

Provides:
- save_video_features: write per-video JSON files for text and visual features
- save_faiss_indices_from_lists: build and save FAISS indices from in-memory lists
"""
import os
import json
import numpy as np
from typing import List

import faiss


def _to_serializable(results: List[dict]):
    """Convert numpy arrays to lists recursively in a list of dictionaries."""
    def convert_item(item):
        if isinstance(item, np.ndarray):
            return item.tolist()
        elif isinstance(item, dict):
            return {k: convert_item(v) for k, v in item.items()}
        elif isinstance(item, list):
            return [convert_item(x) for x in item]
        elif isinstance(item, tuple):
            return tuple(convert_item(x) for x in item)
        else:
            return item

    out = []
    for r in results:
        if r is None:
            out.append(None)
        else:
            out.append(convert_item(r))
    return out


def save_video_features(video_id: str, text_results: List[dict], visual_results: List[dict], text_feat_dir: str, visual_feat_dir: str):
    """Write per-video JSON files for the given results.

    Args:
        video_id: base filename without extension
        text_results: list of dicts produced by extract_entities_and_embed
        visual_results: list of dicts produced by extract_frames_and_embed
        text_feat_dir: directory for textual JSONs
        visual_feat_dir: directory for visual JSONs
    """
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    text_json_path = os.path.join(text_feat_dir, f"{video_id}.json")
    visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json")

    text_serializable = _to_serializable(text_results)
    visual_serializable = _to_serializable(visual_results)

    with open(text_json_path, "w") as f:
        json.dump(text_serializable, f)

    with open(visual_json_path, "w") as f:
        json.dump(visual_serializable, f)


class FaissDB:
    """Minimal helper to build and save a FAISS IndexFlatL2 and metadata file."""
    def __init__(self, dim: int, index_path: str):
        self.dim = dim
        self.index_path = index_path
        self.index = faiss.IndexFlatL2(dim)
        self.metadata = []

    def add(self, embeddings, metadata):
        embeddings = np.stack([np.array(e) / np.linalg.norm(e) for e in embeddings])
        self.index.add(embeddings)
        self.metadata.extend(metadata)

    def load(self):
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)

    def save(self):
        # Ensure parent directory exists before attempting to write index file
        parent_dir = os.path.dirname(self.index_path)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)
        faiss.write_index(self.index, self.index_path)

        # Use _to_serializable to handle all numpy arrays recursively
        sanitized_metadata = _to_serializable(self.metadata)

        with open(self.index_path + ".meta.json", "w") as f:
            json.dump(sanitized_metadata, f)

    def search(self, query_vec, top_k=5, save_results=False, results_file="search_results.json"):
        """Search for top-k similar embeddings and return serializable results.

        Args:
            query_vec: 1D numpy array representing the query vector
            top_k: number of top results to return
            save_results: whether to save results to a JSON file (default: False for clean index directories)
            results_file: name of the file to save results (default: "search_results.json")

        Returns:
            List of dictionaries with 'distance' and 'metadata' keys, sorted by distance (ascending)
        """
        # query_vec should be 1D numpy array
        q = query_vec.astype(np.float32)
        # normalize as the pipeline used normalized vectors
        norm = np.linalg.norm(q)
        if norm == 0:
            raise ValueError("Query vector has zero norm.")
        q = q / norm

        # Perform FAISS search
        D, I = self.index.search(q.reshape(1, -1), top_k)

        # Build results list with metadata
        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                results.append({"distance": float(dist), "metadata": None})
            else:
                result_meta = self.metadata[idx].copy() if self.metadata[idx] else None
                results.append({"distance": float(dist), "metadata": result_meta})

        # Sort by distance (lower is better for L2 distance)
        sorted_results = sorted(results, key=lambda x: x["distance"])

        # Use _to_serializable to ensure all numpy arrays are converted to lists
        serializable_results = _to_serializable(sorted_results)

        # Save results to JSON file if requested
        if save_results:
            # Determine the directory to save results (same as index directory)
            index_dir = os.path.dirname(self.index_path) if os.path.dirname(self.index_path) else "."
            results_path = os.path.join(index_dir, results_file)

            # Create search results with metadata
            search_output = {
                "query_info": {
                    "top_k": top_k,
                    "num_results": len(serializable_results),
                    "index_path": self.index_path
                },
                "results": serializable_results
            }

            with open(results_path, "w") as f:
                json.dump(search_output, f, indent=2)

            print(f"Search results saved to: {results_path}")

        return serializable_results


def save_faiss_indices_from_lists(all_text_embs, all_text_meta, all_visual_embs, all_visual_meta, faiss_text_path, faiss_visual_path):
    """Build and save FAISS indices from provided lists.

    This function is convenient when extraction code collects embeddings in memory
    and then wants to persist them as indices.
    """
    if all_text_embs:
        # all_text_embs is expected to be a list/sequence of array-like embeddings
        first_emb = np.array(all_text_embs[0])
        dim = int(first_emb.shape[0])
        text_db = FaissDB(dim=dim, index_path=faiss_text_path)
        text_db.add(all_text_embs, all_text_meta)
        text_db.save()

    if all_visual_embs:
        first_emb = np.array(all_visual_embs[0])
        dim = int(first_emb.shape[0])
        visual_db = FaissDB(dim=dim, index_path=faiss_visual_path)
        visual_db.add(all_visual_embs, all_visual_meta)
        visual_db.save()

    print("FAISS indices saved.")


def build_indices_from_json_dir(feature_dir: str, output_dir: str, split: str, modality: str = "both"):
    """
    Build FAISS indices from JSON feature files.

    Args:
        feature_dir: Base directory containing textual/ and visual/ subdirectories
        output_dir: Directory to save FAISS indices
        split: Dataset split (train/test/val)
        modality: Which indices to build: "both", "text", or "visual"
    """
    import glob

    text_feat_dir = os.path.join(feature_dir, "textual", split)
    visual_feat_dir = os.path.join(feature_dir, "visual", split)

    os.makedirs(output_dir, exist_ok=True)

    # Build text index if requested
    if modality in ["both", "text"]:
        print(f"\n{'='*80}")
        print(f"BUILDING TEXT INDEX - {split.upper()}")
        print(f"{'='*80}")

        text_json_files = glob.glob(os.path.join(text_feat_dir, "*.json"))
        if not text_json_files:
            print(f"No text feature files found in {text_feat_dir}")
        else:
            all_text_embs = []
            all_text_meta = []

            for json_file in text_json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for item in data:
                    emb = np.array(item['embedding'])
                    all_text_embs.append(emb)

                    # Store metadata without embedding
                    meta = {k: v for k, v in item.items() if k != 'embedding'}
                    all_text_meta.append(meta)

            if all_text_embs:
                text_index_path = os.path.join(output_dir, f"textual_{split}.index")
                first_emb = np.array(all_text_embs[0])
                dim = int(first_emb.shape[0])

                print(f"Building text index from {len(text_json_files)} JSON files...")
                print(f"  - Total embeddings: {len(all_text_embs)}")
                print(f"  - Embedding dimension: {dim}")

                text_db = FaissDB(dim=dim, index_path=text_index_path)
                text_db.add(all_text_embs, all_text_meta)
                text_db.save()

                print(f"Text index saved: {text_index_path}")
                print(f"Metadata saved: {text_index_path}.meta.json")

    # Build visual index if requested
    if modality in ["both", "visual"]:
        print(f"\n{'='*80}")
        print(f"BUILDING VISUAL INDEX - {split.upper()}")
        print(f"{'='*80}")

        visual_json_files = glob.glob(os.path.join(visual_feat_dir, "*.json"))
        if not visual_json_files:
            print(f"No visual feature files found in {visual_feat_dir}")
        else:
            all_visual_embs = []
            all_visual_meta = []

            for json_file in visual_json_files:
                with open(json_file, 'r') as f:
                    data = json.load(f)
                for item in data:
                    emb = np.array(item['embedding'])
                    all_visual_embs.append(emb)

                    # Store metadata without embedding
                    meta = {k: v for k, v in item.items() if k != 'embedding'}
                    all_visual_meta.append(meta)

            if all_visual_embs:
                visual_index_path = os.path.join(output_dir, f"visual_{split}.index")
                first_emb = np.array(all_visual_embs[0])
                dim = int(first_emb.shape[0])

                print(f"Building visual index from {len(visual_json_files)} JSON files...")
                print(f"  - Total embeddings: {len(all_visual_embs)}")
                print(f"  - Embedding dimension: {dim}")

                visual_db = FaissDB(dim=dim, index_path=visual_index_path)
                visual_db.add(all_visual_embs, all_visual_meta)
                visual_db.save()

                print(f"Visual index saved: {visual_index_path}")
                print(f"Metadata saved: {visual_index_path}.meta.json")

    print(f"\n{'='*80}")
    print("INDEX BUILD COMPLETE")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build FAISS indices from JSON feature files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build both text and visual indices for train split
  python embedding_storage.py --split train

  # Build only visual index for train split
  python embedding_storage.py --split train --modality visual

  # Build indices from custom directories
  python embedding_storage.py --split train --feature_dir my_features/ --output_dir my_indices/
        """
    )

    parser.add_argument("--split", type=str, required=True,
                       choices=["train", "test", "val"],
                       help="Dataset split to build indices for")
    parser.add_argument("--feature_dir", type=str, default="feature_extraction",
                       help="Base directory containing textual/ and visual/ subdirectories (default: feature_extraction)")
    parser.add_argument("--output_dir", type=str, default="faiss_db",
                       help="Directory to save FAISS indices (default: faiss_db)")
    parser.add_argument("--modality", type=str, default="both",
                       choices=["both", "text", "visual"],
                       help="Which indices to build: both, text, or visual (default: both)")

    args = parser.parse_args()

    build_indices_from_json_dir(
        feature_dir=args.feature_dir,
        output_dir=args.output_dir,
        split=args.split,
        modality=args.modality
    )
