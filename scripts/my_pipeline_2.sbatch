#!/bin/bash
#SBATCH --account=a127
#SBATCH --partition=normal
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --job-name=multimodal_pipeline
#SBATCH --output=slurm-%j.out

# Execute your exact commands
export RUN_TARGET=smoke
source scripts/clariden_job.sh
python app.py
