#!/bin/bash
set -e  # stop immediately if any command fails

CONFIG="configs/run_0066_model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p.yaml"
CSDA_KAON_PATH="/Volumes/easystore/proton-kaon/images/csv_kaon_48x48_raw.pt"

# uv run python scripts/extract_csv_kaons.py
uv run python scripts/run_inference.py --config "$CONFIG" --csda-kaon-path "$CSDA_KAON_PATH"
uv run python scripts/compute_features.py --config "$CONFIG" --csda-kaons --include-muons
uv run python scripts/analyse_latents.py --config "$CONFIG" --csda-kaons --analyses feature_auc logistic
uv run python scripts/extra/plot_umap_all.py --config "$CONFIG"
