#!/bin/bash
#SBATCH --job-name=pk_sweep
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=32G
#SBATCH --output=logs/sweep.out
#SBATCH --error=logs/sweep.err

source ~/.bashrc
cd ~/proton-kaon
mkdir -p logs

PYTHONPATH=~/proton-kaon \
  uv run python scripts/run_sweep.py \
    --sweep configs/sweep_transforms.yaml \
    --overrides configs/csf.yaml \
    --local \
    --resume \
    --keep-going
