"""
Dense Search Module

FAISS-based dense vector search for text and visual embeddings.
"""
import os
import json
import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel
import open_clip


class FaissIndex:
    """
    FAISS index wrapper for loading and searching vector embeddings.
    """

    def __init__(self, index_path):
        """
        Initialize FAISS index loader.

        Args:
            index_path: Path to FAISS index file

        Raises:
            FileNotFoundError: If index or metadata file doesn't exist
        """
        self.index_path = index_path
        self.index = None
        self.metadata = []

        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index not found: {index_path}")
        if not os.path.exists(index_path + ".meta.json"):
            raise FileNotFoundError(
                f"Metadata JSON file not found for index: {index_path}. "
                f"Expected: {index_path}.meta.json"
            )

    def load(self):
        """Load FAISS index and associated metadata from disk."""
        self.index = faiss.read_index(self.index_path)
        with open(self.index_path + ".meta.json", "r") as f:
            self.metadata = json.load(f)

    def search(self, query_vec, top_k=5):
        """
        Search FAISS index with normalized query vector.

        Args:
            query_vec: 1D numpy array (should already be normalized)
            top_k: Number of results to return

        Returns:
            List of dicts with score and metadata
        """
        q = query_vec.astype(np.float32).reshape(1, -1)
        D, I = self.index.search(q, top_k)

        results = []
        for dist, idx in zip(D[0], I[0]):
            if idx < 0 or idx >= len(self.metadata):
                results.append({"score": float(dist), "meta": None})
            else:
                results.append({"score": float(dist), "meta": self.metadata[idx]})
        return results


class EmbeddingModels:
    """
    Embedding models for query encoding.

    Uses BiomedCLIP text encoder for both text and visual queries to ensure
    embedding space alignment. BiomedCLIP's text and vision encoders are trained
    together, enabling true cross-modal retrieval.
    """

    def __init__(
        self,
        device=None,
        biomedclip_model_id="hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
    ):
        """
        Initialize embedding models with BiomedCLIP for unified embedding space.

        Args:
            device: torch device (auto-detected if None)
            biomedclip_model_id: HuggingFace model ID for BiomedCLIP
        """
        if device is None:
            if torch.cuda.is_available():
                device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                device = torch.device("mps")
            else:
                device = torch.device("cpu")
        self.device = device

        # Load BiomedCLIP for unified text+visual embedding space
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            biomedclip_model_id,
            pretrained=True
        )
        self.clip_model = self.clip_model.to(self.device)
        self.clip_model.eval()

        # Store tokenizer for text encoding
        self.clip_tokenizer = open_clip.get_tokenizer(biomedclip_model_id)

    def embed_text_bio(self, text, max_length=77):
        """
        Generate text embedding using BiomedCLIP text encoder.

        This ensures embedding space alignment with both text and visual embeddings
        generated during the video processing pipeline.

        Args:
            text: Input text query
            max_length: Maximum sequence length (default 77, CLIP's standard)

        Returns:
            Normalized numpy array of shape (embedding_dim,)
        """
        tokens = self.clip_tokenizer([text]).to(self.device)

        with torch.no_grad():
            emb = self.clip_model.encode_text(tokens).squeeze(0)

        # Normalize embedding to match FAISS index normalization
        vec = emb.cpu().numpy().astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec

    def embed_text_clip(self, text):
        """
        Generate text embedding using BiomedCLIP for cross-modal retrieval.

        Note: This is now identical to embed_text_bio() since we use BiomedCLIP
        for all text embeddings. Kept for backward compatibility.

        Args:
            text: Input text query

        Returns:
            Normalized numpy array of shape (embedding_dim,)
        """
        return self.embed_text_bio(text)
