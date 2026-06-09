#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/users/ydhouib/projetcass}"
SCRATCH_ROOT="${SCRATCH_ROOT:-/iopsstor/scratch/cscs/${USER}/projetcass}"
HF_HOME="${HF_HOME:-/iopsstor/scratch/cscs/${USER}/hf}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$SCRATCH_ROOT/pip-cache}"
TMPDIR="${TMPDIR:-$SCRATCH_ROOT/tmp}"
NPM_PREFIX="${NPM_PREFIX:-$SCRATCH_ROOT/npm-global}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH_ROOT/xdg-cache}"
NODE_PREFIX="${NODE_PREFIX:-$SCRATCH_ROOT/node}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
RUN_TARGET="${RUN_TARGET:-full}"
SCI_SPA_CY_MODEL_URL="https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_core_sci_sm-0.5.4.tar.gz"
FFMPEG_PREFIX="${FFMPEG_PREFIX:-$SCRATCH_ROOT/ffmpeg}"
export FFMPEG_PREFIX

if [[ -n "${PYTHON_BIN:-}" ]]; then
  :
elif command -v python3.11 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3.11)"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "python3.11 or python3 not found on this node."
  exit 1
fi

ensure_ffmpeg() {
  if command -v ffmpeg >/dev/null 2>&1; then
    return 0
  fi

  if type module >/dev/null 2>&1; then
    for module_name in ffmpeg FFmpeg ffmpeg/6.1 ffmpeg/7.0; do
      if module load "$module_name" >/dev/null 2>&1 && command -v ffmpeg >/dev/null 2>&1; then
        echo "Loaded ffmpeg via module: $module_name"
        return 0
      fi
    done
  fi

  if command -v conda >/dev/null 2>&1 || command -v mamba >/dev/null 2>&1; then
    local conda_cmd="conda"
    if ! command -v conda >/dev/null 2>&1 && command -v mamba >/dev/null 2>&1; then
      conda_cmd="mamba"
    fi

    if [[ ! -x "$FFMPEG_PREFIX/bin/ffmpeg" ]]; then
      mkdir -p "$FFMPEG_PREFIX"
      "$conda_cmd" install -y -p "$FFMPEG_PREFIX" -c conda-forge ffmpeg
    fi

    export PATH="$FFMPEG_PREFIX/bin:$PATH"
    if command -v ffmpeg >/dev/null 2>&1; then
      echo "Using scratch-local ffmpeg at $FFMPEG_PREFIX"
      return 0
    fi
  fi

  if [[ -x "$PYTHON_BIN" ]]; then
    local ffmpeg_archive="$FFMPEG_PREFIX/ffmpeg-master-latest-linuxarm64-lgpl.tar.xz"
    local ffmpeg_install_dir="$FFMPEG_PREFIX/ffmpeg-static"
    local ffmpeg_binary

    if [[ ! -x "$ffmpeg_install_dir/bin/ffmpeg" ]]; then
      mkdir -p "$ffmpeg_install_dir"
      "$PYTHON_BIN" - <<PY
import os
import shutil
import tarfile
import urllib.request

archive = os.path.expanduser(r"$ffmpeg_archive")
install_dir = os.path.expanduser(r"$ffmpeg_install_dir")
url = "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-master-latest-linuxarm64-lgpl.tar.xz"

os.makedirs(install_dir, exist_ok=True)
if not os.path.exists(archive):
    with urllib.request.urlopen(url, timeout=300) as response, open(archive, "wb") as output_file:
        shutil.copyfileobj(response, output_file)

with tarfile.open(archive, mode="r:xz") as archive_file:
    archive_file.extractall(path=install_dir)
PY
    fi

    ffmpeg_binary="$(find "$ffmpeg_install_dir" -type f -name ffmpeg -perm -u+x | head -n 1)"
    if [[ -x "$ffmpeg_binary" ]]; then
      mkdir -p "$FFMPEG_PREFIX/bin"
      ln -sf "$ffmpeg_binary" "$FFMPEG_PREFIX/bin/ffmpeg"
      if [[ -x "$(dirname "$ffmpeg_binary")/ffprobe" ]]; then
        ln -sf "$(dirname "$ffmpeg_binary")/ffprobe" "$FFMPEG_PREFIX/bin/ffprobe"
      fi
      export PATH="$FFMPEG_PREFIX/bin:$PATH"
      if command -v ffmpeg >/dev/null 2>&1; then
        echo "Using static ffmpeg binary at $ffmpeg_binary"
        return 0
      fi
    fi
  fi

  return 1
}

ensure_node() {
  if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
    return 0
  fi

  if type module >/dev/null 2>&1; then
    for module_name in node Node.js nodejs nodejs/22 nodejs/24; do
      if module load "$module_name" >/dev/null 2>&1 && command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
        echo "Loaded Node.js via module: $module_name"
        return 0
      fi
    done
  fi

  if [[ -x "$PYTHON_BIN" ]]; then
    local node_install_dir="$NODE_PREFIX/node-static"
    local node_archive="$NODE_PREFIX/node-arm64-lts.tar.xz"
    local node_binary

    if [[ ! -x "$node_install_dir/bin/npm" ]]; then
      mkdir -p "$node_install_dir"
      "$PYTHON_BIN" - <<PY
import json
import os
import shutil
import tarfile
import urllib.request

index_url = "https://nodejs.org/dist/index.json"
install_dir = os.path.expanduser(r"$node_install_dir")
archive = os.path.expanduser(r"$node_archive")

with urllib.request.urlopen(index_url, timeout=300) as response:
    releases = json.load(response)

release = next(item for item in releases if item.get("lts") and "linux-arm64" in item.get("files", []))
version = release["version"]
download_url = "https://nodejs.org/dist/{}/node-{}-linux-arm64.tar.xz".format(version, version)

if not os.path.exists(archive):
    with urllib.request.urlopen(download_url, timeout=300) as response, open(archive, "wb") as output_file:
        shutil.copyfileobj(response, output_file)

with tarfile.open(archive, mode="r:xz") as archive_file:
    archive_file.extractall(path=install_dir)
PY
    fi

    node_binary="$(find "$node_install_dir" -type f -path '*/bin/node' -perm -u+x | head -n 1)"
    if [[ -x "$node_binary" ]]; then
      export PATH="$(dirname "$node_binary"):$PATH"
      if command -v node >/dev/null 2>&1 && command -v npm >/dev/null 2>&1; then
        echo "Using static Node.js binary at $node_binary"
        return 0
      fi
    fi
  fi

  return 1
}

# Conda/micromamba bootstrap: create a scratch-local env with build tools
CONDA_PREFIX="${CONDA_PREFIX:-$SCRATCH_ROOT/conda_env}"
MICROMAMBA_BIN="${MICROMAMBA_BIN:-$SCRATCH_ROOT/micromamba/micromamba}"

ensure_micromamba() {
  if [[ -x "$CONDA_PREFIX/bin/python" ]]; then
    export PYTHON_BIN="$CONDA_PREFIX/bin/python"
    echo "Using existing conda env python at $PYTHON_BIN"
    return 0
  fi

  local mamba_cmd=""
  if command -v micromamba >/dev/null 2>&1; then
    mamba_cmd="$(command -v micromamba)"
  elif command -v mamba >/dev/null 2>&1; then
    mamba_cmd="$(command -v mamba)"
  elif command -v conda >/dev/null 2>&1; then
    mamba_cmd="$(command -v conda)"
  else
    mkdir -p "$(dirname "$MICROMAMBA_BIN")"
    if [[ ! -x "$MICROMAMBA_BIN" ]]; then
      echo "Downloading and extracting micromamba to $MICROMAMBA_BIN"
      "$PYTHON_BIN" - <<PY
import urllib.request, tarfile, os, io
url = 'https://micro.mamba.pm/api/micromamba/linux-aarch64/latest'
dst = os.path.expanduser(r"$MICROMAMBA_BIN")
os.makedirs(os.path.dirname(dst), exist_ok=True)
with urllib.request.urlopen(url, timeout=300) as r:
    with tarfile.open(fileobj=io.BytesIO(r.read()), mode='r:bz2') as tar:
        for member in tar.getmembers():
            if member.name.endswith('bin/micromamba'):
                f = tar.extractfile(member)
                with open(dst, 'wb') as out:
                    out.write(f.read())
                break
PY
      chmod +x "$MICROMAMBA_BIN" || true
    fi
    if [[ -x "$MICROMAMBA_BIN" ]]; then
      mamba_cmd="$MICROMAMBA_BIN"
    fi
  fi

  if [[ -z "$mamba_cmd" ]]; then
    echo "No micromamba/conda/mamba available and download failed"
    return 1
  fi

  echo "Creating conda env at $CONDA_PREFIX using $mamba_cmd (this may take a few minutes)"
  mkdir -p "$CONDA_PREFIX"
  
  # Required by micromamba to store pkg caches and metadata
  export MAMBA_ROOT_PREFIX="${SCRATCH_ROOT}/micromamba_root"
  mkdir -p "$MAMBA_ROOT_PREFIX"
  
  # Added gcc_linux-aarch64 and gxx_linux-aarch64 to provide a native toolchain
  PKGS=(python=3.11 pip pybind11 numpy cmake gcc_linux-aarch64 gxx_linux-aarch64 sysroot_linux-aarch64)
  
  "$mamba_cmd" create -y -p "$CONDA_PREFIX" -c conda-forge "${PKGS[@]}" || true
  
  if [[ -x "$CONDA_PREFIX/bin/python" ]]; then
    "$mamba_cmd" install -y -p "$CONDA_PREFIX" -c conda-forge nmslib || true
  fi

  if [[ -x "$CONDA_PREFIX/bin/python" ]]; then
    export PYTHON_BIN="$CONDA_PREFIX/bin/python"
    echo "Conda env ready; using $PYTHON_BIN"
    return 0
  fi

  echo "Conda env creation failed"
  return 1
}

mkdir -p "$SCRATCH_ROOT" "$PIP_CACHE_DIR" "$TMPDIR" "$HF_HOME"
mkdir -p "$NPM_PREFIX"
cd "$PROJECT_DIR"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

set -a
source "$ENV_FILE"
set +a

export HF_HOME
export PIP_CACHE_DIR
export TMPDIR
export XDG_CACHE_HOME
export NPM_CONFIG_PREFIX="$NPM_PREFIX"
export PATH="$NPM_PREFIX/bin:$PATH"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1

if ! ensure_ffmpeg; then
  echo "ffmpeg not found on this node."
  exit 1
fi

if ! ensure_node; then
  echo "node/npm not found on this node."
  exit 1
fi

if ! command -v youtube-po-token-generator >/dev/null 2>&1; then
  npm install -g youtube-po-token-generator
fi

# Ensure micromamba runs BEFORE creating the venv so we use Conda's Python 3.11
if ! ensure_micromamba; then
  echo "Failed to bootstrap conda/micromamba environment."
  exit 1
fi

if [[ -x "$VENV_DIR/bin/python" ]]; then
  venv_version="$($VENV_DIR/bin/python - <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
 )"
  if [[ "$venv_version" != "3.11" ]]; then
    rm -rf "$VENV_DIR"
  fi
fi

if [[ ! -d "$VENV_DIR" ]]; then
  "$PYTHON_BIN" -m venv --system-site-packages "$VENV_DIR"
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip "setuptools<82.0.0" wheel

# Filter out nmslib-metabrainz from requirements since we pre-installed nmslib binary via conda
python -m pip install -r requirements.txt

python -c "import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')"

python - <<'PY'
import torch
import faiss
import transformers
import cv2
import spacy
import scispacy
import open_clip
print('Core imports verified')
PY

case "$RUN_TARGET" in
  smoke)
    python - <<'PY'
print('Smoke setup complete. Ready to run pipeline commands.')
PY
    ;;
  data)
    python data_preparation.py
    ;;
  pipeline)
    python multimodal_pipeline_with_sliding_window.py
    ;;
  evaluate)
    python run_full_evaluation.py --split test --enable_curation --enable_attribution --output-dir artifacts/evaluation_runs
    ;;
  full)
    python main.py
    ;;
  *)
    echo "Unknown RUN_TARGET: $RUN_TARGET"
    echo "Use one of: smoke, data, pipeline, evaluate, full"
    exit 1
    ;;
esac