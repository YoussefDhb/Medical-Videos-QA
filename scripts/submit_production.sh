#!/bin/bash
#SBATCH --job-name=medvidrag_full
#SBATCH --output=/users/$USER/projetcass/logs/R-%x.%j.out
#SBATCH --error=/users/$USER/projetcass/logs/R-%x.%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH -A a127
#SBATCH --partition=normal

set -eo pipefail
set -x

# Ensure log directory exists
mkdir -p /users/$USER/projetcass/logs

# Load the environment and run the requested stage
# You will launch this script multiple times for different stages
export RUN_TARGET=${RUN_TARGET:-full}

srun --environment=/users/$USER/.edf/multimodal.toml bash ./scripts/clariden_job.sh