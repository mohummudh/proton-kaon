#!/usr/bin/env python3
"""
scripts/extra/plot_mean_adc_vs_solidity.py

Scatter plots of mean_adc vs solidity for every event (protons, kaons, muons),
read from the precomputed features table (scripts/compute_features.py output):
  - one combined plot with all three species overlaid
  - three separate plots, one per species

Usage:
    python scripts/extra/plot_mean_adc_vs_solidity.py
    python scripts/extra/plot_mean_adc_vs_solidity.py --no-muons
    python scripts/extra/plot_mean_adc_vs_solidity.py --out-dir figs/mean_adc_vs_solidity_v2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATURES_PKL = Path("/Volumes/easystore/proton-kaon/features/features.pkl")
OUT_DIR      = Path(__file__).resolve().parent.parent.parent / "figs" / "mean_adc_vs_solidity"

COLOURS = {
    "proton": "#4C78A8",
    "kaon":   "#F58518",
    "muon":   "#9467BD",
}

DPI = 150

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       12,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "savefig.bbox":    "tight",
})


def scatter_panel(ax, mean_adc, solidity, particle):
    finite = np.isfinite(mean_adc) & np.isfinite(solidity)
    ax.scatter(
        solidity[finite], mean_adc[finite],
        s=8, alpha=0.35, color=COLOURS[particle],
        edgecolors="none", label=f"{particle} (n={finite.sum()})",
    )
    ax.set_xlabel("Solidity")
    ax.set_ylabel("Mean ADC")
    ax.set_xlim(0, 1)
    ax.spines[["top", "right"]].set_visible(False)


def plot_combined(data_by_particle, out_dir):
    fig, ax = plt.subplots(figsize=(6, 5))
    for particle, (mean_adc, solidity) in data_by_particle.items():
        scatter_panel(ax, mean_adc, solidity, particle)
    ax.set_title("Mean ADC vs Solidity", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

    path = out_dir / "mean_adc_vs_solidity_combined.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def plot_separate(data_by_particle, out_dir):
    for particle, (mean_adc, solidity) in data_by_particle.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        scatter_panel(ax, mean_adc, solidity, particle)
        ax.set_title(f"{particle.capitalize()} — Mean ADC vs Solidity",
                     fontsize=13, fontweight="bold", color=COLOURS[particle])
        ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

        path = out_dir / f"mean_adc_vs_solidity_{particle}.png"
        fig.savefig(path)
        plt.close(fig)
        print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default=str(FEATURES_PKL),
                        help="Path to features.pkl (default: %(default)s)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: figs/mean_adc_vs_solidity)")
    parser.add_argument("--no-muons", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.features}…")
    feat_df = pd.read_pickle(args.features)

    species = ["proton", "kaon"] + ([] if args.no_muons else ["muon"])
    data_by_particle = {}
    for particle in species:
        rows = feat_df[feat_df["particle_type"] == particle]
        if rows.empty:
            print(f"  {particle}: no rows found, skipping")
            continue
        mean_adc = rows["mean_adc"].to_numpy()
        solidity = rows["solidity"].to_numpy()
        data_by_particle[particle] = (mean_adc, solidity)
        finite = np.isfinite(mean_adc) & np.isfinite(solidity)
        print(f"  {particle:6s}: {finite.sum()}/{len(rows)} finite  "
              f"mean_adc={mean_adc[finite].mean():.1f}  solidity={solidity[finite].mean():.3f}")

    print("\nPlotting combined scatter…")
    plot_combined(data_by_particle, out_dir)

    print("Plotting per-species scatter…")
    plot_separate(data_by_particle, out_dir)

    print(f"\nDone. All plots saved to {out_dir}")


if __name__ == "__main__":
    main()
