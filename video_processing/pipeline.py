"""
Video Processing Pipeline Orchestrator

Provides a high-level VideoProcessor class that coordinates ASR, text embedding,
visual embedding, and deduplication steps.
"""
import os
import json
import numpy as np
from .asr import transcribe_with_asr
from .text_embeddings import load_ner_and_embed_models, extract_entities_and_embed
from .visual_embeddings import extract_frames_and_embed
from .deduplication import deduplicate_embeddings_similarity


class VideoProcessorConfig:
    """Configuration for high-precision, visual-dominant video processing pipeline."""

    def __init__(
        self,
        asr_model_id="openai/whisper-tiny",
        
        # 1. TIGHTEN TEXT WINDOWS FOR EXACT BOUNDARIES
        window_size=64,                  # Reduced from 256. Forces temporal boundaries to be ~10-15s max.
        stride=16,                       # Reduced from 128. High overlap ensures we catch the exact start of an action.
        min_coverage_contribution=0.01,  
        deduplication_mode='none',       # Disable dedup to prevent losing crucial exact-second anchors.
        
        # 2. PURE VISUAL DOMINANCE SETTINGS
        frames_per_segment=8,            # Increased to capture every micro-movement.
        sampling_strategy='uniform',     # Uniform guarantees strict, mathematically even temporal coverage.
        quality_filter=False,           
        aggregation_method='max',        # CRITICAL: Changed from 'mean'. 'Max' preserves the peak visual signal of the exact frame. 'Mean' blurs it.
        min_frames=3,
        max_frames=15,
        
        # 3. PRESERVE ALL VISUAL DATA
        enable_visual_deduplication=False, # We want every exact frame boundary, do not deduplicate visuals.
        visual_similarity_threshold=0.99
    ):
        """
        Initialize configuration with default hyperparameters.

        Text embedding hyperparameters:
            window_size: Token window size (default: 256)
            stride: Stride between windows (default: 192)
            min_coverage_contribution: Minimum new token coverage (default: 0.05)
            deduplication_mode: 'coverage', 'similarity', 'aggressive', or 'none'

        Visual embedding hyperparameters:
            frames_per_segment: Number of frames per segment (default: 6, optimized for quality)
            sampling_strategy: 'uniform', 'adaptive', or 'quality_based' (default: 'quality_based')
            quality_filter: Enable frame quality filtering (default: True)
            aggregation_method: 'mean' or 'max' (default: 'max' - captures best frame)
            min_frames: Minimum frames for short segments (default: 3)
            max_frames: Maximum frames for long segments (default: 12)
            enable_visual_deduplication: Enable visual deduplication (default: True)
            visual_similarity_threshold: Cosine similarity threshold for dedup (default: 0.98)
        """
        # ASR
        self.asr_model_id = asr_model_id

        # Text embedding
        self.window_size = window_size
        self.stride = stride
        self.min_coverage_contribution = min_coverage_contribution
        self.deduplication_mode = deduplication_mode

        # Visual embedding
        self.frames_per_segment = frames_per_segment
        self.sampling_strategy = sampling_strategy
        self.quality_filter = quality_filter
        self.aggregation_method = aggregation_method
        self.min_frames = min_frames
        self.max_frames = max_frames
        self.enable_visual_deduplication = enable_visual_deduplication
        self.visual_similarity_threshold = visual_similarity_threshold


class VideoProcessor:
    """
    High-level video processing pipeline orchestrator.

    Coordinates ASR, text embedding, visual embedding, and deduplication.
    """

    def __init__(self, config=None):
        """
        Initialize video processor with configuration.

        Args:
            config: VideoProcessorConfig instance (uses defaults if None)
        """
        self.config = config or VideoProcessorConfig()
        self.nlp = None
        self.clip_model = None
        self.clip_tokenizer = None

    def _load_models(self):
        """Lazy load NER and embedding models."""
        if self.clip_model is None:
            print("Loading NER and BiomedCLIP embedding models...")
            self.nlp, self.clip_model, self.clip_tokenizer = load_ner_and_embed_models()

    def process_video(self, video_path, video_id, text_feat_dir=None, visual_feat_dir=None,
                     skip_if_exists=True):
        """
        Process a single video through the full pipeline.

        Args:
            video_path: Path to video file
            video_id: Unique identifier for the video
            text_feat_dir: Directory to save text features (optional)
            visual_feat_dir: Directory to save visual features (optional)
            skip_if_exists: If True, skip processing if JSON files already exist

        Returns:
            tuple: ((text_embeddings, text_metadata), (visual_embeddings, visual_metadata))
        """
        text_json_path = os.path.join(text_feat_dir, f"{video_id}.json") if text_feat_dir else None
        visual_json_path = os.path.join(visual_feat_dir, f"{video_id}.json") if visual_feat_dir else None

        # Check if both JSON files exist and skip_if_exists is True
        if skip_if_exists and text_json_path and visual_json_path:
            if os.path.exists(text_json_path) and os.path.exists(visual_json_path):
                print(f"[SKIP] {video_id}: Loading existing embeddings from JSON files...")
                return self._load_from_json(video_id, text_json_path, visual_json_path)

        # Step 1: ASR - Transcribe audio
        print(f"\n[1/4] Transcribing audio for {video_id}...")
        transcript_chunks = transcribe_with_asr(video_path, self.config.asr_model_id)
        print(f"Transcription complete: {len(transcript_chunks)} chunks")

        # Step 2: Text Embedding - Generate text embeddings with sliding windows using BiomedCLIP
        print(f"\n[2/4] Generating text embeddings for {video_id}...")
        self._load_models()

        text_results = extract_entities_and_embed(
            transcript_chunks, self.nlp, self.clip_model, self.clip_tokenizer,
            video_id=video_id,
            window_size=self.config.window_size,
            stride=self.config.stride,
            deduplication_mode=self.config.deduplication_mode,
            min_coverage_contribution=self.config.min_coverage_contribution
        )
        print(f"Text embeddings: {len(text_results)} segments")

        # Save text features if directory provided
        if text_json_path:
            self._save_to_json(text_results, text_json_path)

        # Step 3: Visual Embedding - Extract frames and generate visual embeddings
        print(f"\n[3/4] Generating visual embeddings for {video_id}...")
        visual_results = extract_frames_and_embed(
            video_path, text_results, video_id=video_id,
            frames_per_segment=self.config.frames_per_segment,
            sampling_strategy=self.config.sampling_strategy,
            quality_filter=self.config.quality_filter,
            aggregation_method=self.config.aggregation_method,
            min_frames=self.config.min_frames,
            max_frames=self.config.max_frames
        )
        print(f"Visual embeddings: {len(visual_results)} segments")

        # Step 4: Deduplication - Apply similarity-based deduplication to visual embeddings (optional)
        if self.config.enable_visual_deduplication:
            print(f"\n[4/4] Deduplicating visual embeddings for {video_id}...")
            visual_results_before = len(visual_results)
            visual_results = deduplicate_embeddings_similarity(
                visual_results,
                similarity_threshold=self.config.visual_similarity_threshold
            )
            print(f"Visual embeddings after dedup: {len(visual_results)} segments (removed {visual_results_before - len(visual_results)})")
        else:
            print(f"\n[4/4] Skipping visual deduplication (disabled)")

        # Save visual features if directory provided
        if visual_json_path:
            self._save_to_json(visual_results, visual_json_path)

        # Prepare output format
        text_embs = [r['embedding'] for r in text_results]
        text_meta = [{"video_id": video_id, **r} for r in text_results]
        visual_embs = [r['embedding'] for r in visual_results]
        visual_meta = [{"video_id": video_id, **r} for r in visual_results]

        print(f"\nProcessing complete for {video_id}")
        print(f"   Text: {len(text_results)} embeddings (dim={len(text_embs[0]) if text_embs else 0})")
        print(f"   Visual: {len(visual_results)} embeddings (dim={len(visual_embs[0]) if visual_embs else 0})")

        return (text_embs, text_meta), (visual_embs, visual_meta)

    def _save_to_json(self, results, json_path):
        """Save embeddings to JSON file."""
        os.makedirs(os.path.dirname(json_path), exist_ok=True)

        results_serializable = []
        for r in results:
            r_copy = r.copy()
            if isinstance(r_copy.get('embedding'), np.ndarray):
                r_copy['embedding'] = r_copy['embedding'].tolist()
            results_serializable.append(r_copy)

        with open(json_path, 'w') as f:
            json.dump(results_serializable, f)

    def _load_from_json(self, video_id, text_json_path, visual_json_path):
        """Load embeddings from existing JSON files."""
        try:
            with open(text_json_path, 'r') as f:
                text_json = json.load(f)
            with open(visual_json_path, 'r') as f:
                visual_json = json.load(f)

            # Reconstruct embeddings and metadata
            text_embs = [np.array(r['embedding']) for r in text_json]
            text_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}}
                        for r in text_json]

            visual_embs = [np.array(r['embedding']) for r in visual_json]
            visual_meta = [{"video_id": video_id, **{k: v for k, v in r.items() if k != 'embedding'}}
                          for r in visual_json]

            print(f"Loaded {len(text_embs)} text and {len(visual_embs)} visual embeddings")
            return (text_embs, text_meta), (visual_embs, visual_meta)
        except Exception as e:
            print(f"Failed to load existing JSONs: {e}. Reprocessing video...")
            raise
