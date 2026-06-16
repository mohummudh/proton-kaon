#!/bin/bash
set -e  # stop immediately if any command fails

CONFIG="configs/run_0066_model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p.yaml"
CSDA_KAON_PATH="/Volumes/easystore/proton-kaon/images/csv_kaon_48x48_raw_clean.pt"

# Step 1: extract all matched clusters (multiple per event) from ROOT
# uv run python scripts/extract_csv_kaons.py

# Step 2: score every cluster with a feature-based RF and keep the most
#         kaon-like one per event; writes csv_kaon_*_clean.pkl + *_raw_clean.pt
# uv run python scripts/extra/csda_kaon_cleaning.py --config "$CONFIG"

# Step 3: inference on the cleaned images
uv run python scripts/run_inference.py --config "$CONFIG" --csda-kaon-path "$CSDA_KAON_PATH"

# Step 4: compute features (uses csv_kaon_col_clean.pkl via --csda-kaons flag)
uv run python scripts/compute_features.py --config "$CONFIG" --csda-kaons --include-muons

# Step 5: analyse latents
uv run python scripts/analyse_latents.py --config "$CONFIG" --include-muons --csda-kaons --analyses feature_auc logistic

# Step 6: UMAP plot
uv run python scripts/extra/plot_umap_all.py --config "$CONFIG"
