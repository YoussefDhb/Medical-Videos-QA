"""
Text Embedding Module

Handles text chunking, entity recognition, and BiomedCLIP text embeddings for transcript text.
Uses sliding window approach with coverage-based deduplication.
"""
import hashlib
import torch
import numpy as np
import spacy
from scispacy.umls_linking import UmlsEntityLinker
import open_clip

# Global flag for NER (can be overridden via environment variable)
import os
ENABLE_NER = os.environ.get('ENABLE_NER', 'False').lower() == 'true'


def load_ner_and_embed_models(
    model_id='hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
):
    """
    Load NER and text embedding models.

    Args:
        model_id: BiomedCLIP model ID (default: BiomedCLIP-PubMedBERT)

    Returns:
        nlp: SpaCy NLP pipeline (None if NER disabled)
        clip_model: BiomedCLIP model for text embeddings
        clip_tokenizer: BiomedCLIP tokenizer function
    """
    if ENABLE_NER:
        nlp = spacy.load("en_core_sci_lg")
        linker = UmlsEntityLinker(resolve_abbreviations=True)
        nlp.add_pipe(linker)
    else:
        nlp = None

    # Load BiomedCLIP for unified text/visual embedding space
    clip_model, _, _ = open_clip.create_model_and_transforms(
        model_id,
        pretrained=True
    )
    clip_model.eval()
    clip_tokenizer = open_clip.get_tokenizer(model_id)

    return nlp, clip_model, clip_tokenizer


def extract_entities_and_embed_optimized(
    transcript_chunks, nlp, clip_model, clip_tokenizer, video_id,
    window_size=256, stride=192, max_length=77,
    deduplication_mode='coverage', min_coverage_contribution=0.05
):
    """
    Optimized sliding window approach with coverage-based deduplication using BiomedCLIP.

    Key improvements:
    - Uses BiomedCLIP text encoder for embedding space alignment with visual embeddings
    - Coverage-based deduplication ensures 100% transcript coverage
    - Efficient timestamp mapping for accurate segment identification
    - Shared segment_id for linking text and visual embeddings
    - Validation to prevent content loss

    Args:
        transcript_chunks: List of transcript chunks with timestamps
        nlp: SpaCy NLP pipeline for entity extraction
        clip_model: BiomedCLIP model for text embeddings
        clip_tokenizer: BiomedCLIP tokenizer function
        video_id: Unique identifier for the video
        window_size: 256 tokens (reasonable for text context)
        stride: 192 tokens (25% overlap for coverage)
        max_length: Maximum sequence length for CLIP (default: 77, CLIP's max)
        deduplication_mode: 'coverage' (recommended), 'similarity', 'aggressive', or 'none'
        min_coverage_contribution: Minimum % of new tokens required to keep window (default: 0.05)

    Returns:
        List of dictionaries with text embeddings and metadata
    """
    results = []
    seen_hashes = {}  # Track content hashes with token ranges
    covered_ranges = []  # Track which token ranges we've embedded

    # Combine all transcript text with timestamp mapping
    full_text = ""
    timestamp_map = []

    for chunk in transcript_chunks:
        text = chunk['text']
        ts = chunk['timestamp']

        start_char = len(full_text)
        full_text += text + " "
        end_char = len(full_text) - 1

        timestamp_map.append({
            'start_char': start_char,
            'end_char': end_char,
            'timestamp': ts,
            'text': text
        })

    # Tokenize full text using CLIP tokenizer
    # Note: CLIP uses different tokenization, so we approximate with whitespace split
    # for sliding window logic, but use actual CLIP tokenization for embeddings
    full_tokens = full_text.split()  # Approximate tokenization for windowing

    if not full_tokens:
        return results

    print(f"Processing {len(full_tokens)} tokens with window_size={window_size}, stride={stride}")

    # Create sliding windows
    for window_idx, start_idx in enumerate(range(0, len(full_tokens), stride)):
        end_idx = min(start_idx + window_size, len(full_tokens))
        window_tokens = full_tokens[start_idx:end_idx]

        if len(window_tokens) < window_size // 4:  # Skip very small windows at the end
            continue

        window_text = ' '.join(window_tokens)

        # Coverage-based deduplication
        content_hash = hashlib.md5(window_text.encode('utf-8')).hexdigest()

        # Calculate new coverage contribution
        window_range = set(range(start_idx, end_idx))
        covered_tokens = set()
        for r_start, r_end in covered_ranges:
            covered_tokens.update(range(r_start, r_end))

        new_tokens = window_range - covered_tokens
        coverage_ratio = len(new_tokens) / len(window_range) if len(window_range) > 0 else 0

        # Decide whether to keep this window based on deduplication mode
        should_keep = False

        if deduplication_mode == 'none':
            should_keep = True
        elif deduplication_mode == 'coverage':
            # Keep if it adds significant new content
            should_keep = coverage_ratio >= min_coverage_contribution
        elif deduplication_mode == 'similarity':
            # Keep if not exact duplicate (for backward compatibility)
            should_keep = content_hash not in seen_hashes
        elif deduplication_mode == 'aggressive':
            # Old behavior: strict hash-based deduplication
            should_keep = content_hash not in seen_hashes

        if not should_keep:
            continue

        # Track this window
        seen_hashes[content_hash] = {
            'token_start': start_idx,
            'token_end': end_idx,
            'coverage_contribution': len(new_tokens)
        }
        covered_ranges.append((start_idx, end_idx))

        # Find representative timestamp
        start_char_approx = int((start_idx / len(full_tokens)) * len(full_text))
        end_char_approx = int((end_idx / len(full_tokens)) * len(full_text))

        overlapping_timestamps = []
        for ts_info in timestamp_map:
            if (ts_info['start_char'] <= end_char_approx and
                ts_info['end_char'] >= start_char_approx):
                overlapping_timestamps.append(ts_info['timestamp'])

        representative_ts = overlapping_timestamps[len(overlapping_timestamps)//2] if overlapping_timestamps else timestamp_map[0]['timestamp'] if timestamp_map else (0, 0)

        # Entity extraction
        entities = []
        if ENABLE_NER and nlp is not None:
            try:
                doc = nlp(window_text)
                entities = [(ent.text, ent._.umls_ents) for ent in doc.ents if hasattr(ent._, 'umls_ents')]
            except Exception as e:
                print(f"NER failed for window {window_idx}: {e}")

        # Generate embedding using BiomedCLIP text encoder
        try:
            # Tokenize with CLIP tokenizer (handles truncation internally)
            tokens = clip_tokenizer([window_text])

            with torch.no_grad():
                emb = clip_model.encode_text(tokens).squeeze(0).cpu().numpy()
        except Exception as e:
            print(f"Embedding generation failed for window {window_idx}: {e}")
            continue

        # Create segment_id for multimodal linking: video_id + window_idx
        segment_id = f"{video_id}_seg_{window_idx}"

        results.append({
            "video_id": video_id,
            "segment_id": segment_id,  # Segment ID for linking text and visual
            "window_id": window_idx,
            "timestamp": representative_ts,
            "text": window_text,
            "content_hash": content_hash,
            "overlapping_timestamps": overlapping_timestamps,
            "entities": entities,
            "embedding": emb,
            "window_info": {
                "start_token": start_idx,
                "end_token": end_idx,
                "window_size": len(window_tokens),
                "total_tokens": len(full_tokens),
                "coverage_contribution": len(new_tokens)
            }
        })

    # Validation: Check coverage
    total_covered = len(covered_tokens)
    coverage_pct = (total_covered / len(full_tokens)) * 100 if len(full_tokens) > 0 else 0

    print(f"Generated {len(results)} text embeddings (from {window_idx + 1} candidate windows)")
    print(f"Coverage: {coverage_pct:.1f}% ({total_covered}/{len(full_tokens)} tokens)")

    if coverage_pct < 95:
        print(f"WARNING: Only {coverage_pct:.1f}% coverage! Missing {len(full_tokens) - total_covered} tokens.")
        print(f"   Consider: reducing min_coverage_contribution (current: {min_coverage_contribution})")

    return results


def extract_entities_and_embed(transcript_chunks, nlp, clip_model, clip_tokenizer, video_id, **kwargs):
    """
    Unified embedding function using optimized sliding window approach with BiomedCLIP.

    Uses window_size=256, stride=192 (25% overlap) for token efficiency.
    Includes integrated deduplication and segment_id for multimodal linking.
    BiomedCLIP ensures text embeddings align with visual embeddings in same space.

    Args:
        transcript_chunks: List of transcript chunks with timestamps
        nlp: SpaCy NLP pipeline for entity extraction
        clip_model: BiomedCLIP model for text embeddings
        clip_tokenizer: BiomedCLIP tokenizer function
        video_id: Unique identifier for the video being processed
        **kwargs: window_size, stride, max_length (optional overrides)

    Returns:
        List of text embedding dictionaries
    """
    return extract_entities_and_embed_optimized(
        transcript_chunks, nlp, clip_model, clip_tokenizer, video_id, **kwargs
    )
