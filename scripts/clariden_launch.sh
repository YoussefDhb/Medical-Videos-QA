#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ACCOUNT="${ACCOUNT:-a127}"
PARTITION="${PARTITION:-normal}"
JOB_TIME="${JOB_TIME:-12:00:00}"
GPUS="${GPUS:-1}"
CPUS_PER_TASK="${CPUS_PER_TASK:-16}"
MEMORY="${MEMORY:-64G}"
ENV_FILE="${ENV_FILE:-$PROJECT_DIR/.env}"

if [[ ! -f "$ENV_FILE" ]]; then
  echo "Missing env file: $ENV_FILE"
  exit 1
fi

srun \
  --account="$ACCOUNT" \
  --partition="$PARTITION" \
  --time="$JOB_TIME" \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task="$CPUS_PER_TASK" \
  --gres="gpu:$GPUS" \
  --mem="$MEMORY" \
  --export=ALL \
  --pty bash "$PROJECT_DIR/scripts/clariden_job.sh"
