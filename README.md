# Medical Video Question Answering through Multimodal Retrieval

A system for retrieving relevant medical video segments that answer natural language questions using multimodal embeddings and hybrid search.

## Quick Start

### 1. Install

```bash
# Create environment
conda create -n medvidenv python=3.11 && conda activate medvidenv

# Install dependencies
pip install -r requirements.txt

# Optional: Configure HuggingFace token
echo "HF_TOKEN=your_token" > .env
```

### 2. Run

**Option A: Full Pipeline** (Recommended first-time)
```bash
python main.py  # Downloads data → Extracts features → Builds indices → Demo query
```

**Option B: Web App** (Interactive UI)
```bash
python app.py  # Visit http://127.0.0.1:5000
```

**Option C: CLI Query** (Single query)
```bash
python query_faiss.py \
  --query "How to do a mouth cancer check at home?" \
  --split test \
  --hybrid \
  --alpha 0.45
```

**Option D: Batch Evaluation** (Full dataset)
```bash
python run_full_evaluation.py --split test
```

---

## Key Features

| Feature | Details |
|---------|---------|
| **ASR** | OpenAI Whisper (tiny) with word-level timestamps |
| **Text Embeddings** | BiomedCLIP with sliding windows (256 tokens, 64 stride) |
| **Visual Embeddings** | BiomedCLIP vision encoder with adaptive frame sampling |
| **Dense Search** | FAISS L2-normalized similarity on 768/512-dim embeddings |
| **Sparse Search** | BM25 with medical term expansion |
| **Hybrid Fusion** | Linear/RRF combining dense + sparse (default: α=0.45, RRF) |
| **Evaluation** | Recall@K, mAP, nDCG, temporal IoU/F1, video hit rate |
| **Dataset** | MedVidQA: 800+ videos, 3000+ QA pairs |

---

## System Architecture

```
Video Input
    ↓
[Phase 1] ASR + Text Embeddings (sliding windows) + Visual Embeddings (adaptive frames)
    ↓
[Phase 2] Normalize embeddings (L2) → Build FAISS indices
    ↓
[Phase 3] Query embedding → Dense search (FAISS) + Sparse search (BM25)
    ↓
[Phase 4] Hybrid fusion (linear/RRF) → Multimodal aggregation by segment
    ↓
Retrieved Timestamps + Confidence Scores
```

**Multimodal Design**: Fixed visual dominance (0.2 text weight, 0.8 visual weight) enforced in the platform for consistent retrieval.

---

## Usage Guide

### Data Preparation
```bash
python data_preparation.py  # Download videos, clean dataset, organize into train/val/test
```

### Feature Extraction & Indexing
```bash
python multimodal_pipeline_with_sliding_window.py  # Extract embeddings, build FAISS indices
```

Customize parameters by editing the script:
- Text: `window_size=256, stride=64, deduplication_mode='coverage'`
- Visual: `frames_per_segment=2, sampling_strategy='adaptive'`
- Parallel: `batch_size=4, max_workers=2`

### Querying

**Basic (Dense Search)**:
```bash
python query_faiss.py --query "..." --split test --final_k 10
```

**Hybrid (Recommended)**:
```bash
python query_faiss.py --query "..." --split test --hybrid --alpha 0.45 --fusion rrf
```

**With Evaluation**:
```bash
python query_faiss.py --query "..." --split test --hybrid --eval \
  --video_id "..." --answer_start 35 --answer_end 96
```

**Batch Evaluation**:
```bash
python run_full_evaluation.py --split test --alpha 0.45 --max_queries 50
```

### Web Application

```bash
python app.py
```

Access at `http://127.0.0.1:5000`. The web app uses **fixed weights** (text=0.2, visual=0.8) for consistent platform behavior. For research/experimentation with different weights, use the CLI.

---

## Configuration

### Environment Variables (`.env`)
```bash
HF_TOKEN=hf_...  # HuggingFace token for gated models
```

### Common Hyperparameters

**Text Embeddings**:
- `window_size=256` – Token window size
- `stride=64` – Overlap (75%)
- `deduplication_mode='coverage'` – 'coverage', 'similarity', 'aggressive', 'none'

**Visual Embeddings**:
- `frames_per_segment=2` – Frames per segment
- `sampling_strategy='adaptive'` – 'uniform', 'adaptive', 'quality_based'

**Search** (edit scripts directly):
- `alpha=0.45` – Dense weight in RRF (0-1)
- `fusion='rrf'` – 'linear' or 'rrf'
- `local_k=50` – Results per index before merging
- `final_k=10` – Final combined results
- `text_weight=0.2, visual_weight=0.8` – **CLI-only, fixed in web app**

---

## Project Structure

```
.
├── app.py                          # Flask web app (fixed weights: 0.2/0.8)
├── query_faiss.py                  # CLI for querying + evaluation
├── run_full_evaluation.py          # Batch evaluation script
├── multimodal_pipeline_with_sliding_window.py  # Feature extraction
├── data_preparation.py             # Download & prepare data
│
├── search/                         # Search modules
│   ├── dense_search.py             # FAISS + BiomedCLIP
│   ├── sparse_search.py            # BM25 + medical expansion
│   ├── hybrid_fusion.py            # Linear/RRF fusion
│   ├── aggregation.py              # Multimodal aggregation
│   └── utils.py
│
├── video_processing/               # Feature extraction modules
│   ├── pipeline.py                 # Orchestrator
│   ├── asr.py                      # Whisper ASR
│   ├── text_embeddings.py          # Sliding windows + BiomedCLIP
│   ├── visual_embeddings.py        # Frame sampling + BiomedCLIP
│   └── deduplication.py
│
├── templates/index.html            # Web app UI
├── requirements.txt                # Dependencies
├── .env                            # API keys (user-created)
└── README.md
```

---

## Performance (Test Set, 102 queries)

| Metric | Value |
|--------|-------|
| Recall@10 | 65.7% |
| Precision@5 | 23.5% |
| mAP | 28.3% |
| nDCG@10 | 37.8% |
| Temporal F1 | 15.9% |
| Video Hit Rate | 94.1% |
| Avg Search Time | 0.27s |

---

## Example Workflows

### Workflow 1: First-Time Setup
```bash
pip install -r requirements.txt
python data_preparation.py
python multimodal_pipeline_with_sliding_window.py
python query_faiss.py --query "How to check mouth cancer at home?" --split train --hybrid
```

### Workflow 2: Web App with Existing Indices
```bash
# (Assumes indices already built)
python app.py
# Open http://127.0.0.1:5000
```

### Workflow 3: Batch Evaluation
```bash
python run_full_evaluation.py --split test --alpha 0.3
python run_full_evaluation.py --split test --alpha 0.5
# Compare results in artifacts/evaluation_runs/
```

### Workflow 4: CLI Experimentation (Different Weights)
```bash
# Test with different text/visual weight combinations (CLI-only)
python query_faiss.py --query "..." --split test --text_weight 0.3 --visual_weight 0.7
python query_faiss.py --query "..." --split test --text_weight 0.5 --visual_weight 0.5
```

---

## Installation Details

### System Requirements
- Python 3.11+
- ffmpeg (system package)
- ~20GB disk space for videos + indices

### OS-Specific Setup

**macOS:**
```bash
brew install ffmpeg
pip install -r requirements.txt
```

**Ubuntu/Debian:**
```bash
sudo apt-get install ffmpeg
pip install -r requirements.txt
```

**Windows:**
- Download ffmpeg from https://ffmpeg.org/download.html and add to PATH
- Then: `pip install -r requirements.txt`

### GPU Acceleration (Optional)
```bash
# NVIDIA GPUs
pip uninstall faiss-cpu
pip install faiss-gpu>=1.7.4

# Apple Silicon: Automatic (uses MPS)
```

---

## Advanced Topics

### Rebuild Specific Components
```bash
# Visual embeddings only (no text reprocessing)
python rebuild_visual_embeddings.py --split train

# Custom evaluation
python evaluation.py  # See source for detailed metrics
```

### Hyperparameter Optimization
```bash
python hyperparameter_tuning.py  # Grid search + visualization
```

### Compare Search Methods
```bash
python compare_search_methods.py --query "..." --expected_video "..." --top_k 10
```

---

## Notes

- **Web App**: Uses fixed weights (0.2/0.8) for visual dominance. Optimal for medical video retrieval.
- **CLI**: Supports weight customization for research/experimentation.
- **Dataset**: MedVidQA (https://medvidqa.github.io) - 800+ videos, 3000+ QA pairs
- **Models**: BiomedCLIP (text: 768-dim, visual: 512-dim), Whisper-tiny ASR, BM25 sparse search

---

## References

- [MedVidQA Dataset](https://medvidqa.github.io)
- [BiomedCLIP Paper](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [OpenAI Whisper](https://github.com/openai/whisper)
