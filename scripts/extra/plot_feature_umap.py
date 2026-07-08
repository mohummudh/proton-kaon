#!/usr/bin/env python3
"""
scripts/extra/plot_feature_umap.py

UMAP scatter plots coloured by feature value.
Layout: one row per feature, columns = particle sets.

Edit DATA and STYLE at the top; pass --config to point at a model YAML.

Usage:
    python scripts/extra/plot_feature_umap.py --config configs/run_XXXX.yaml
    python scripts/extra/plot_feature_umap.py --config configs/run_XXXX.yaml --out figs/umap_features.pdf
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA  —  what to plot
# ══════════════════════════════════════════════════════════════════════════════

# Features to plot — one row each, top to bottom
FEATURES = ["mean_adc", "solidity"]

# Columns to show — any subset of: "train", "val", "kaon", "muon"
COLUMNS = ["train", "muon", "kaon"]

# Column display labels
COL_LABELS = {
    "train": "Protons",
    "val":   "Val Protons",
    "kaon":  "Kaon Candidates",
    "muon":  "Muons",
}

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════

CMAP        = "viridis"     # per-scatter colormap
POINT_SIZE  = 6             # marker size (pt²)
POINT_ALPHA = 0.5           # marker opacity
SHARED_CBAR = True          # one colorbar per row spanning all columns

# Colorbar range per feature — set to (vmin, vmax) to fix, or None for auto
CLIM = {
    "mean_adc": None,
    "solidity": None,
}

# Figure
PANEL_W  = 4.5    # inches per column
PANEL_H  = 3.5    # inches per row
HSPACE   = 0.45
WSPACE   = 0.15

# Axis labels
XLABEL = "UMAP 1"
YLABEL = "UMAP 2"

# Colorbar label suffix — appended to feature name; set "" to use feature name only
CBAR_SUFFIX = ""

# Font
FONT_FAMILY = "serif"
FONT_SERIF  = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE   = 14

DPI = 300

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

plt.rcParams.update({
    "font.family":         FONT_FAMILY,
    "font.serif":          FONT_SERIF,
    "font.size":           FONT_SIZE,
    "axes.labelsize":      FONT_SIZE,
    "xtick.labelsize":     FONT_SIZE - 1,
    "ytick.labelsize":     FONT_SIZE - 1,
    "axes.linewidth":      0.5,
    "xtick.major.width":   0.5,
    "ytick.major.width":   0.5,
    "xtick.major.size":    2.5,
    "ytick.major.size":    2.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.dpi":          DPI,
    "savefig.dpi":         DPI,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
})


def build_model_name(cfg):
    return (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
        f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
        f"_tx{cfg['data'].get('transform', 'none')}"
        + ("_speciesall" if cfg["data"].get("proton") == "all" else "")
    )


def load_embeddings(cfg, model_name):
    """Load or compute UMAP embeddings. Returns dict key→(N,2) array."""
    try:
        import umap as umap_lib
    except ImportError:
        print("umap-learn not installed — install with: pip install umap-learn")
        sys.exit(1)

    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name

    # Try pre-saved embeddings cache first (written by plot_umap_all.py)
    cache = inf_dir.parent.parent / "figs" / model_name / "latents-features" / "cache_umap.npz"
    if cache.exists():
        print(f"  Loading cached UMAP embeddings from {cache}")
        data = np.load(cache)
        return {k: data[k] for k in data.files}

    # Fall back: load latents, fit/load reducer, transform
    print("  No embedding cache found — transforming latents...")
    latents = {}
    for key in ["train", "val", "kaon", "muon"]:
        p = inf_dir / f"{key}.npz"
        if p.exists():
            latents[key] = np.load(p)["latents"]

    reducer_path = inf_dir / "reducer.pkl"
    if reducer_path.exists():
        with open(reducer_path, "rb") as f:
            reducer = pickle.load(f)
        print(f"  Loaded UMAP reducer from {reducer_path}")
    else:
        print("  Fitting UMAP reducer...")
        all_l = np.vstack(list(latents.values()))
        reducer = umap_lib.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        reducer.fit(all_l)
        with open(reducer_path, "wb") as f:
            pickle.dump(reducer, f)

    return {k: reducer.transform(v) for k, v in latents.items()}


def load_features(cfg, _model_name):
    """Returns dict key→DataFrame for each particle set."""
    features_path = (
        cfg.get("data", {}).get("features_path")
        or "/Volumes/easystore/proton-kaon/features/features.pkl"
    )
    split_path = Path(cfg["output"]["splits_dir"]) / "split_p.npz"

    feat = pd.read_pickle(features_path)
    idx  = np.load(split_path)

    protons = feat[feat["particle_type"] == "proton"]
    return {
        "train": protons.iloc[idx["train_idx"]].reset_index(drop=True),
        "val":   protons.iloc[idx["val_idx"]].reset_index(drop=True),
        "kaon":  feat[feat["particle_type"] == "kaon"].reset_index(drop=True),
        "muon":  feat[feat["particle_type"] == "muon"].reset_index(drop=True),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = build_model_name(cfg)
    print(f"Model: {model_name}")

    embeddings = load_embeddings(cfg, model_name)
    features   = load_features(cfg, model_name)

    # filter to requested columns that have data
    cols = [c for c in COLUMNS if c in embeddings and c in features]
    if not cols:
        print("No matching columns found in embeddings/features.")
        sys.exit(1)

    n_cols = len(cols)
    out_dir = PROJECT_ROOT / "figs" / model_name / "latents-features"
    out_dir.mkdir(parents=True, exist_ok=True)

    for feat_name in FEATURES:
        clim = CLIM.get(feat_name)
        if clim is None:
            all_vals = np.concatenate([
                features[c][feat_name].dropna().values
                for c in cols if feat_name in features[c].columns
            ])
            clim = (float(all_vals.min()), float(all_vals.max()))

        fig, axes = plt.subplots(
            1, n_cols,
            figsize=(PANEL_W * n_cols, PANEL_H),
            squeeze=False,
        )
        fig.subplots_adjust(wspace=WSPACE)

        sc_last = None
        for ci, col in enumerate(cols):
            ax = axes[0][ci]
            feat_df = features[col]
            emb     = embeddings[col]

            if feat_name not in feat_df.columns:
                ax.set_visible(False)
                continue

            vals   = feat_df[feat_name].values.astype(float)
            finite = np.isfinite(vals)

            sc = ax.scatter(
                emb[finite, 0], emb[finite, 1],
                c=vals[finite],
                cmap=CMAP,
                vmin=clim[0], vmax=clim[1],
                s=POINT_SIZE, alpha=POINT_ALPHA,
                linewidths=0, rasterized=True,
            )
            sc_last = sc

            ax.set_xlabel(XLABEL)
            ax.set_ylabel(YLABEL if ci == 0 else "")
            ax.set_title(COL_LABELS.get(col, col))
            ax.spines[["top", "right"]].set_visible(False)

        if SHARED_CBAR and sc_last is not None:
            cbar = fig.colorbar(sc_last, ax=axes[0], pad=0.02, shrink=0.85)
            label = feat_name + (f" {CBAR_SUFFIX}" if CBAR_SUFFIX else "")
            cbar.set_label(label, fontsize=FONT_SIZE)
            cbar.ax.tick_params(labelsize=FONT_SIZE - 1)

        plt.savefig(out_dir / f"feature_umap_{feat_name}.pdf")
        plt.savefig(out_dir / f"feature_umap_{feat_name}.png", dpi=DPI)
        print(f"Saved feature_umap_{feat_name}.pdf  +  .png")
        plt.close()

    plt.close()


if __name__ == "__main__":
    main()
