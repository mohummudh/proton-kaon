#!/bin/bash
#$ -cwd
#$ -N pk_sweep
#$ -l gpu=1
#$ -l h_rt=48:00:00
#$ -l h_vmem=32G
#$ -o logs/sweep.out
#$ -e logs/sweep.err

source ~/.bashrc
cd ~/proton-kaon
mkdir -p logs

PYTHONPATH=~/proton-kaon \
  python scripts/run_sweep.py \
    --sweep configs/sweep_transforms.yaml \
    --overrides configs/csf.yaml \
    --local \
    --resume \
    --keep-going
