"""
multimodal_pipeline.py
Modular pipeline for multimodal medical video processing:
- ASR (openai/whisper-tiny)
- Entity recognition & text embeddings (BiomedCLIP Text encoder, SciSpacy/HF NER)
- Frame extraction & visual embeddings (BiomedCLIP Vision encoder)
- FAISS DB integration
- Demo pipeline
"""
import os
import gc
import torch
import numpy as np
import open_clip
import getpass
from concurrent.futures import ProcessPoolExecutor, as_completed
from dotenv import load_dotenv
from huggingface_hub import login as hf_login
from video_processing import VideoProcessor
from video_processing.pipeline import VideoProcessorConfig
from embedding_storage import FaissDB, save_faiss_indices_from_lists
from data_preparation import filter_json_by_embeddings
from transformers import AutoTokenizer, AutoModel
import multiprocessing as mp


try:
    from search import (
        aggregate_results_by_segment,
        print_segment_results,
        HybridSearchEngine,
        load_segments_from_json_dir
    )
except ImportError:
    print("Warning: Could not import from search module. Search functionality may not be available.")
    aggregate_results_by_segment = None
    print_segment_results = None
    HybridSearchEngine = None
    load_segments_from_json_dir = None

# Parallel processing flag
ENABLE_PARALLEL = True  # Set to False to disable parallel processing

# Flag to use Colab secrets for HF_TOKEN (default: True)
USE_COLAB_SECRETS = True

# Load HuggingFace token from .env, Colab secrets, or prompt
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# if HF_TOKEN is None and USE_COLAB_SECRETS:
#     try:
#         HF_TOKEN = userdata.get('HF_TOKEN')
#     except Exception:
#         HF_TOKEN = None

if HF_TOKEN is None:
    try:
        HF_TOKEN = getpass.getpass("Enter your HuggingFace token (for gated models): ")
        hf_login(HF_TOKEN)
    except Exception as e:
        print("Warning: Could not login to HuggingFace. If you get a 401 error, set HF_TOKEN env variable or login manually.")
else:
    hf_login(HF_TOKEN)

def process_video_batch(batch_fnames, video_dir, text_feat_dir, visual_feat_dir, asr_model_id,
                       **processor_kwargs):
    """
    Process a batch of videos using either the new VideoProcessor or legacy implementation.

    Args:
        batch_fnames: List of video filenames to process
        video_dir: Directory containing videos
        text_feat_dir: Directory to save text features
        visual_feat_dir: Directory to save visual features
        asr_model_id: ASR model identifier
        **processor_kwargs: Additional configuration parameters for VideoProcessor

    Returns:
        Dictionary mapping filenames to (text_data, visual_data, error) tuples
    """
    batch_results = {}

    config = VideoProcessorConfig(
        asr_model_id=asr_model_id,
        window_size=processor_kwargs.get('window_size', 256),
        stride=processor_kwargs.get('stride', 128),
        min_coverage_contribution=processor_kwargs.get('min_coverage_contribution', 0.15),
        deduplication_mode=processor_kwargs.get('deduplication_mode', 'coverage'),
        frames_per_segment=processor_kwargs.get('frames_per_segment', 2),
        sampling_strategy=processor_kwargs.get('sampling_strategy', 'adaptive'),
        quality_filter=processor_kwargs.get('quality_filter', False),
        aggregation_method=processor_kwargs.get('aggregation_method', 'mean'),
        enable_visual_deduplication=processor_kwargs.get('enable_visual_deduplication', True),
        visual_similarity_threshold=processor_kwargs.get('visual_similarity_threshold', 0.98)
    )

    processor = VideoProcessor(config)

    for fname in batch_fnames:
        if not fname.endswith('.mp4'):
            batch_results[fname] = (None, None, 'Not an mp4 file')
            continue

        video_id = os.path.splitext(fname)[0]
        video_path = os.path.join(video_dir, fname)

        try:
            # Clear GPU memory before processing each video
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

            print(f"\nProcessing {fname} with VideoProcessor...")

            (text_embs, text_meta), (visual_embs, visual_meta) = processor.process_video(
                video_path=video_path,
                video_id=video_id,
                text_feat_dir=text_feat_dir,
                visual_feat_dir=visual_feat_dir,
                skip_if_exists=True
            )

            batch_results[fname] = ((text_embs, text_meta), (visual_embs, visual_meta), None)

        except Exception as e:
            batch_results[fname] = (None, None, str(e))

    return batch_results

def parallel_process_videos(fnames, video_dir, text_feat_dir, visual_feat_dir, asr_model_id="openai/whisper-tiny", batch_size=1, max_workers=2, **processor_kwargs):
    """
    Full pipeline parallel processing with batching: ASR, NER, embedding, JSON saving.
    """
    # Split fnames into batches
    batches = [fnames[i:i+batch_size] for i in range(0, len(fnames), batch_size)]
    results = {}
    total = len(fnames)
    processed = 0
    ctx = mp.get_context('spawn')
    with ProcessPoolExecutor(max_workers=max_workers, mp_context=ctx) as executor:
        futures = {executor.submit(process_video_batch, batch, video_dir, text_feat_dir, visual_feat_dir, asr_model_id, **processor_kwargs): tuple(batch) for batch in batches}
        for fut in as_completed(futures):
            batch_result = fut.result()
            for fname, (text, visual, error) in batch_result.items():
                processed += 1
                if error == 'JSONs already exist':
                    print(f"[SKIP] {fname}: {error} | Progress: {processed}/{total} videos processed.")
                elif error:
                    print(f"[ERROR] {fname}: {error} | Progress: {processed}/{total} videos processed.")
                else:
                    print(f"[DONE] {fname} | Progress: {processed}/{total} videos processed.")
                results[fname] = (text, visual)
    return results

# 4. Demo pipeline
def demo_pipeline(video_path, text_feat_dir, visual_feat_dir, faiss_text_path, faiss_visual_path,
                 window_size=256, stride=192, min_coverage_contribution=0.05,
                 deduplication_mode='coverage', frames_per_segment=2,
                 sampling_strategy='adaptive', quality_filter=False, aggregation_method='mean',
                 enable_visual_deduplication=True):
    """
    Demo pipeline with configurable hyperparameters using refactored VideoProcessor.

    Text embedding hyperparameters:
        window_size: Token window size (default: 256)
        stride: Stride between windows (default: 192)
        min_coverage_contribution: Minimum new token coverage to keep window (default: 0.05)
        deduplication_mode: 'coverage', 'similarity', 'aggressive', or 'none' (default: 'coverage')

    Visual embedding hyperparameters:
        frames_per_segment: Number of frames per segment (default: 2)
        sampling_strategy: 'uniform', 'adaptive', or 'quality_based' (default: 'adaptive')
        quality_filter: Enable frame quality filtering (default: False)
        aggregation_method: 'mean' or 'max' (default: 'mean')
        enable_visual_deduplication: Enable visual deduplication (default: True)
    """
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)

    fname = os.path.basename(video_path)
    video_dir = os.path.dirname(video_path)

    if not fname.endswith('.mp4'):
        return None, None

    video_id = os.path.splitext(fname)[0]
    video_path = os.path.join(video_dir, fname)

    print(f"Processing video: {video_path}")
    print(f"\nUsing VideoProcessor with hyperparameters:")
    print(f"  Text: window_size={window_size}, stride={stride}, dedup={deduplication_mode}")
    print(f"  Visual: frames={frames_per_segment}, strategy={sampling_strategy}, agg={aggregation_method}")

    # Create VideoProcessor configuration
    config = VideoProcessorConfig(
        window_size=window_size,
        stride=stride,
        min_coverage_contribution=min_coverage_contribution,
        deduplication_mode=deduplication_mode,
        frames_per_segment=frames_per_segment,
        sampling_strategy=sampling_strategy,
        quality_filter=quality_filter,
        aggregation_method=aggregation_method,
        enable_visual_deduplication=enable_visual_deduplication
    )

    # Process video with VideoProcessor
    processor = VideoProcessor(config)
    (text_embs, text_meta), (visual_embs, visual_meta) = processor.process_video(
        video_path=video_path,
        video_id=video_id,
        text_feat_dir=text_feat_dir,
        visual_feat_dir=visual_feat_dir,
        skip_if_exists=True
    )

    # Convert back to results format for FAISS storage
    text_results = [meta for meta in text_meta]
    for i, emb in enumerate(text_embs):
        text_results[i]['embedding'] = emb

    visual_results = [meta for meta in visual_meta]
    for i, emb in enumerate(visual_embs):
        visual_results[i]['embedding'] = emb

    # Save to FAISS
    text_db = FaissDB(dim=len(text_embs[0]) if text_embs else 768, index_path=faiss_text_path)
    visual_db = FaissDB(dim=len(visual_embs[0]) if visual_embs else 512, index_path=faiss_visual_path)
    text_db.add(text_embs, text_results)
    visual_db.add(visual_embs, visual_results)
    text_db.save()
    visual_db.save()
    print("Databases saved.")

    # Enhanced logging for demo results
    print(f"\nFinal Results Summary:")
    print(f"- Video ID: {video_id}")
    print(f"- Text embeddings: {len(text_results)}")
    print(f"- Visual embeddings: {len(visual_results)}")
    print(f"- Text FAISS index dimension: {len(text_embs[0]) if text_embs else 'N/A'}")
    print(f"- Visual FAISS index dimension: {len(visual_embs[0]) if visual_embs else 'N/A'}")

    # Query demo with multimodal search (matching query_faiss.py output format)
    query = "How to do a mouth cancer check at home?"

    # Load BiomedCLIP for unified text+visual search with embedding space alignment
    print("Loading BiomedCLIP for unified embedding space search...")
    clip_model, _, _ = open_clip.create_model_and_transforms(
        'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224',
        pretrained=True
    )
    clip_model.eval()

    # Tokenize with BiomedCLIP tokenizer
    tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')
    tokens = tokenizer([query])
    with torch.no_grad():
        query_emb = clip_model.encode_text(tokens).squeeze().cpu().numpy()

    # Normalize embedding for both text and visual search (same embedding space)
    norm = np.linalg.norm(query_emb)
    if norm > 0:
        query_emb = query_emb / norm

    print(f"\n{'='*80}")
    print(f"DEMO QUERY: {query}")
    print(f"{'='*80}")

    # Perform multimodal search (using same query embedding for both since embeddings are aligned)
    print("\n[1] Searching textual index...")
    text_search_results = text_db.search(query_emb, top_k=50)  # Get more for hybrid search

    print("[2] Searching visual index...")
    visual_search_results = visual_db.search(query_emb, top_k=10)

    print(f"\nFound {len(text_search_results)} text results and {len(visual_search_results)} visual results")

    # Apply hybrid search to text results if available
    use_hybrid = True  # Set to False to disable hybrid search in demo
    if use_hybrid and HybridSearchEngine is not None and load_segments_from_json_dir is not None:
        print(f"\n{'='*80}")
        print("APPLYING HYBRID SEARCH (BM25 + Dense Embeddings)")
        print(f"{'='*80}")

        try:
            # Load segments from the text feature directory
            segments_data = load_segments_from_json_dir(text_feat_dir)

            # Initialize hybrid search engine
            hybrid_engine = HybridSearchEngine(
                segments_data=segments_data,
                alpha=0.3  # 30% dense, 70% BM25 (optimized for medical terminology)
            )

            # Transform FAISS results to format expected by hybrid search
            text_pool = []
            for result in text_search_results:
                text_pool.append({
                    'raw_score': result.get('distance', float('inf')),
                    'meta': result.get('metadata', {}),
                    'source_index': faiss_text_path
                })

            # Perform hybrid search
            hybrid_text_results = hybrid_engine.hybrid_search(
                query=query,
                dense_results=text_pool,
                top_k=50,
                fusion='linear',
                expand_query=True
            )

            # Show fusion analysis
            hybrid_engine.analyze_fusion_contribution(hybrid_text_results, top_k=10)

            # Replace text results with hybrid results
            text_search_results = []
            for hybrid_result in hybrid_text_results:
                meta = hybrid_result.get('metadata', {})
                # Convert hybrid score back to distance format
                combined_score = hybrid_result.get('combined_score', 0.0)
                distance = -np.log(combined_score + 1e-10)

                text_search_results.append({
                    'metadata': meta,
                    'distance': distance
                })

            print(f"Hybrid search applied: Re-ranked {len(text_search_results)} text results")

        except Exception as e:
            print(f"Hybrid search failed: {e}")
            print("Falling back to dense-only search...")
    elif use_hybrid:
        print("\nHybrid search not available (module not imported)")

    # Transform results to match query_faiss format
    def transform_faiss_results(results, is_visual=False):
        """Transform FaissDB search results to query_faiss format"""
        transformed = []
        for result in results:
            meta = result.get('metadata', {})
            dist = result.get('distance', float('inf'))
            transformed.append({
                'meta': meta,
                'raw_score': dist,
                'distance': dist
            })
        return transformed

    text_results_formatted = transform_faiss_results(text_search_results, is_visual=False)
    visual_results_formatted = transform_faiss_results(visual_search_results, is_visual=True)

    # Apply segment-level aggregation (matching query_faiss.py)
    if aggregate_results_by_segment is not None:
        print("\n[3] Aggregating multimodal results by segment...")
        # Use feature_extraction/ directory for hierarchical search (contains overlapping_timestamps)
        hierarchical_json_dir = "feature_extraction/"
        segment_contexts = aggregate_results_by_segment(
            text_results=text_results_formatted,
            visual_results=visual_results_formatted,
            top_k=10,
            text_weight=0.6,
            visual_weight=0.4,
            enable_hierarchical=True,
            json_dir=hierarchical_json_dir
        )

        # Print and save results in the same format as query_faiss.py
        # Determine output filename based on hybrid search usage
        output_file = "multimodal_search_results_hybrid.json" if use_hybrid else "multimodal_search_results_dense.json"

        if print_segment_results is not None:
            print_segment_results(segment_contexts, query=query, output_file=output_file)
        else:
            print(f"\nFound {len(segment_contexts)} multimodal segments")
            for i, ctx in enumerate(segment_contexts[:3], 1):
                print(f"\nSegment {i}: {ctx['video_id']} | Score: {ctx['combined_score']:.4f}")
                print(f"  Text: {ctx['text_evidence']['text'][:150] if ctx['text_evidence'] else 'N/A'}...")
    else:
        print("\nWarning: Multimodal aggregation not available. Install query_faiss dependencies.")
        print("Showing individual search results instead...")

        # Fallback: show individual results
        print("\nTop Text Results:")
        for i, result in enumerate(text_search_results[:3], 1):
            meta = result.get('metadata', {})
            print(f"  {i}. {meta.get('segment_id', 'N/A')} - Score: {result.get('distance', 'N/A'):.4f}")

        print("\nTop Visual Results:")
        for i, result in enumerate(visual_search_results[:3], 1):
            meta = result.get('metadata', {})
            print(f"  {i}. {meta.get('segment_id', 'N/A')} - Score: {result.get('distance', 'N/A'):.4f}")

def process_video(fname, video_dir, text_feat_dir, visual_feat_dir,
                 window_size=256, stride=192, min_coverage_contribution=0.05,
                 deduplication_mode='coverage', frames_per_segment=2,
                 sampling_strategy='adaptive', quality_filter=False, aggregation_method='mean',
                 enable_visual_deduplication=True):
    """
    Process a single video with configurable hyperparameters.
    See demo_pipeline docstring for parameter descriptions.
    """
    if not fname.endswith('.mp4'):
        return None, None

    video_id = os.path.splitext(fname)[0]
    video_path = os.path.join(video_dir, fname)

    config = VideoProcessorConfig(
        window_size=window_size,
        stride=stride,
        min_coverage_contribution=min_coverage_contribution,
        deduplication_mode=deduplication_mode,
        frames_per_segment=frames_per_segment,
        sampling_strategy=sampling_strategy,
        quality_filter=quality_filter,
        aggregation_method=aggregation_method,
        enable_visual_deduplication=enable_visual_deduplication
    )

    processor = VideoProcessor(config)

    return processor.process_video(
        video_path, video_id,
        text_feat_dir=text_feat_dir,
        visual_feat_dir=visual_feat_dir,
        skip_if_exists=True
    )

def process_split(split, video_dir, text_feat_dir, visual_feat_dir, faiss_text_path, faiss_visual_path,
                 window_size=256, stride=192, min_coverage_contribution=0.05,
                 deduplication_mode='coverage', frames_per_segment=2,
                 sampling_strategy='adaptive', quality_filter=False, aggregation_method='mean',
                 enable_visual_deduplication=True):
    """
    Process a full split (train/val/test) with configurable hyperparameters.

    See demo_pipeline docstring for parameter descriptions.
    """
    print(f"\nProcessing split '{split}' with hyperparameters:")
    print(f"  Text: window_size={window_size}, stride={stride}, min_cov={min_coverage_contribution}, dedup={deduplication_mode}")
    print(f"  Visual: frames={frames_per_segment}, strategy={sampling_strategy}, quality={quality_filter}, agg={aggregation_method}")

    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    all_text_embs, all_text_meta = [], []
    all_visual_embs, all_visual_meta = [], []
    fnames = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]
    batch_size = 4  # Tune for your hardware
    max_workers = 2 if torch.cuda.is_available() or torch.backends.mps.is_available() else os.cpu_count()
    if ENABLE_PARALLEL:
        print(f"Parallel processing enabled: batch_size={batch_size}, max_workers={max_workers}")
        results = parallel_process_videos(fnames, video_dir, text_feat_dir, visual_feat_dir, batch_size=batch_size, max_workers=max_workers,
                                         window_size=window_size, stride=stride,
                                         min_coverage_contribution=min_coverage_contribution,
                                         deduplication_mode=deduplication_mode,
                                         frames_per_segment=frames_per_segment,
                                         sampling_strategy=sampling_strategy,
                                         quality_filter=quality_filter,
                                         aggregation_method=aggregation_method,
                                         enable_visual_deduplication=enable_visual_deduplication)
        for fname in fnames:
            text, visual = results.get(fname, (None, None))
            if text and all(item is not None for item in text):
                all_text_embs.extend(text[0])
                all_text_meta.extend(text[1])
            if visual and all(item is not None for item in visual):
                all_visual_embs.extend(visual[0])
                all_visual_meta.extend(visual[1])
            # Free up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()
    else:
        total = len(fnames)
        done = 0
        for idx, fname in enumerate(fnames, 1):
            text, visual = process_video(
                fname, video_dir, text_feat_dir, visual_feat_dir,
                window_size=window_size, stride=stride,
                min_coverage_contribution=min_coverage_contribution,
                deduplication_mode=deduplication_mode,
                frames_per_segment=frames_per_segment,
                sampling_strategy=sampling_strategy,
                quality_filter=quality_filter,
                aggregation_method=aggregation_method,
                enable_visual_deduplication=enable_visual_deduplication
            )
            if text:
                all_text_embs.extend(text[0])
                all_text_meta.extend(text[1])
            if visual:
                all_visual_embs.extend(visual[0])
                all_visual_meta.extend(visual[1])
            done += 1
            print(f"Progress: {done}/{total} videos processed.")
    # Delegate FAISS index building / saving to embedding_storage
    try:
        save_faiss_indices_from_lists(all_text_embs, all_text_meta, all_visual_embs, all_visual_meta, faiss_text_path, faiss_visual_path)
    except Exception:
        # Fallback: try previous logic if the helper fails
        if all_text_embs:
            print("Saving text embeddings to FAISS (fallback)...")
            text_db = FaissDB(dim=len(all_text_embs[0]), index_path=faiss_text_path)
            text_db.add(all_text_embs, all_text_meta)
            text_db.save()
        if all_visual_embs:
            print("Saving visual embeddings to FAISS (fallback)...")
            visual_db = FaissDB(dim=len(all_visual_embs[0]), index_path=faiss_visual_path)
            visual_db.add(all_visual_embs, all_visual_meta)
            visual_db.save()
    print(f"Done processing split: {split}")

def test_single_video(video_path, text_feat_dir, visual_feat_dir):
    os.makedirs(text_feat_dir, exist_ok=True)
    os.makedirs(visual_feat_dir, exist_ok=True)
    fname = os.path.basename(video_path)
    (text_embs, text_meta), (visual_embs, visual_meta) = process_video(fname, os.path.dirname(video_path), text_feat_dir, visual_feat_dir)
    print(f"Text embeddings: {len(text_embs)}; Visual embeddings: {len(visual_embs)}")
    print(f"Sample text meta: {text_meta[0] if text_meta else None}")
    print(f"Sample visual meta: {visual_meta[0] if visual_meta else None}")
    save_faiss_indices_from_lists(text_embs, text_meta, visual_embs, visual_meta,
                                  faiss_text_path='faiss_db/textual_single.index',
                                  faiss_visual_path='faiss_db/visual_single.index')

def main():
    # Test mode for a single video
    # test_single_video("videos_train/_6csIJAWj_s.mp4", "feature_extraction/textual/test_single", "feature_extraction/visual/test_single")

    # Test demo pipeline
    # demo_pipeline(
    #     video_path="videos_train/_6csIJAWj_s.mp4",
    #     text_feat_dir="feature_extraction/textual/demo",
    #     visual_feat_dir="feature_extraction/visual/demo",
    #     faiss_text_path="faiss_db/textual_demo.index",
    #     faiss_visual_path="faiss_db/visual_demo.index"
    # )

    # Uncomment above and set your video path to test single video or run a demo pipeline
    splits = [
        ("train", "videos_train"),
        ("val", "videos_val"),
        ("test", "videos_test")
    ]

    for split, video_dir in splits:
        print(f"Processing split: {split}")
        process_split(
            split=split,
            video_dir=video_dir,
            text_feat_dir=f"feature_extraction/textual/{split}",
            visual_feat_dir=f"feature_extraction/visual/{split}",
            faiss_text_path=f"faiss_db/textual_{split}.index",
            faiss_visual_path=f"faiss_db/visual_{split}.index",
            # Strict Visual Dominance Config
            window_size=64,
            stride=16,
            min_coverage_contribution=0.01,
            deduplication_mode='none',
            frames_per_segment=8,
            sampling_strategy='uniform',
            quality_filter=False,
            aggregation_method='max',
            enable_visual_deduplication=False
        )

    # After all splits are processed, filter JSON files based on successfully generated embeddings
    print("\n" + "="*80)
    print("Filtering JSON files to keep only entries with both textual AND visual embeddings...")
    print("="*80)
    filter_json_by_embeddings(model_name="openai/whisper-tiny")

if __name__ == "__main__":
    import multiprocessing as mp
    try:
        # Enforce 'spawn' globally right at script launch
        mp.set_start_method('spawn', force=True)
        print("Multiprocessing start method successfully set to 'spawn'.")
    except RuntimeError as e:
        print(f"Multiprocessing start method warning: {e}")
        
    main()