#!/bin/bash
set -e  # stop immediately if any command fails

CONFIG="configs/run_0066_model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p.yaml"
MUON_PATH="/Volumes/easystore/proton-kaon/images/muon_48x48_log1p.pt"

# uv run python scripts/image_making_muons_art.py
# uv run python scripts/image_making.py --muon
# uv run python scripts/run_inference.py --config "$CONFIG" --include-muons --muon-image-path "$MUON_PATH"
uv run python scripts/compute_features.py --config "$CONFIG" --include-muons
uv run python scripts/analyse_latents.py --config "$CONFIG" --include-muons
uv run python scripts/plot_umap_all.py --config "$CONFIG"