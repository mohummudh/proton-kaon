#!/usr/bin/env python3
"""
scripts/inspect_n_local_maxima.py

Shows how n_local_maxima processes column_maxes:
  raw profile → smoothed (uniform_filter1d, size=5) → detected peaks.

Plots a random sample of tracks for protons, kaons, and muons side-by-side,
plus a histogram of the peak count distribution per particle type.

Usage:
    python scripts/inspect_n_local_maxima.py
    python scripts/inspect_n_local_maxima.py --n-tracks 12 --seed 7
    python scripts/inspect_n_local_maxima.py --no-muons
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
from scipy.signal import find_peaks

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cuts import image_cuts

COL_PKL      = Path("/Volumes/easystore/proton-kaon/clusters/col.pkl")
IND_PKL      = Path("/Volumes/easystore/proton-kaon/clusters/ind.pkl")
MUON_COL_PKL = Path("/Volumes/easystore/proton-kaon/clusters/muon_col.pkl")
OUT_DIR      = PROJECT_ROOT / "figs" / "inspect_n_local_maxima"

SMOOTH_SIZE = 15  # must match topology.n_local_maxima

COLOURS = {
    "proton": "#4C78A8",
    "kaon":   "#F58518",
    "muon":   "#9467BD",
}


def process(column_maxes):
    cm = uniform_filter1d(column_maxes.astype(float), size=SMOOTH_SIZE)
    peaks, _ = find_peaks(cm)
    return cm, peaks


def plot_tracks(rows, particle, n_tracks, rng, out_dir):
    """Grid of individual track profiles: raw vs smoothed + peaks marked."""
    sample = rows.sample(min(n_tracks, len(rows)), random_state=rng).reset_index(drop=True)
    ncols = 4
    nrows = int(np.ceil(len(sample) / ncols))

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 3.8, nrows * 2.6),
                             constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)
    colour = COLOURS[particle]

    for idx, row in sample.iterrows():
        ax = axes[idx // ncols, idx % ncols]
        cm_raw = np.array(row["column_maxes"])
        cm_smooth, peaks = process(cm_raw)
        x = np.arange(len(cm_raw))

        ax.plot(x, cm_raw, color=colour, alpha=0.35, linewidth=1.0, label="raw")
        ax.plot(x, cm_smooth, color=colour, linewidth=1.6, label=f"smoothed (w={SMOOTH_SIZE})")
        if len(peaks):
            ax.scatter(peaks, cm_smooth[peaks],
                       color="red", zorder=5, s=40, marker="v",
                       label=f"peaks ({len(peaks)})")
        ax.set_title(f"n_peaks={len(peaks)}", fontsize=9)
        ax.set_xlabel("wire (column)", fontsize=8)
        ax.set_ylabel("max ADC", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.spines[["top", "right"]].set_visible(False)
        if idx == 0:
            ax.legend(fontsize=7, framealpha=0.8)

    # hide unused axes
    for idx in range(len(sample), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    fig.suptitle(f"{particle.capitalize()}s — raw vs smoothed dE/dx profile with detected peaks",
                 fontsize=12, fontweight="bold")
    path = out_dir / f"tracks_{particle}.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def plot_histogram(data_by_particle, out_dir):
    """Distribution of n_local_maxima counts per particle type."""
    max_peaks = max(v.max() for v in data_by_particle.values())
    bins = np.arange(-0.5, max_peaks + 1.5, 1)

    fig, ax = plt.subplots(figsize=(8, 4))
    for particle, counts in data_by_particle.items():
        ax.hist(counts, bins=bins, density=True,
                color=COLOURS[particle], alpha=0.55,
                label=f"{particle} (n={len(counts)})", histtype="stepfilled")
        ax.hist(counts, bins=bins, density=True,
                color=COLOURS[particle], alpha=0.9,
                histtype="step", linewidth=1.4)

    ax.set_xlabel("n_local_maxima", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of peak count per particle type", fontsize=12, fontweight="bold")
    all_ticks = np.arange(0, int(max_peaks) + 1)
    step = max(1, int(np.ceil(len(all_ticks) / 20)))
    ax.set_xticks(all_ticks[::step])
    ax.tick_params(axis="x", rotation=45)
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    path = out_dir / "histogram.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def compute_counts(rows):
    return np.array([
        len(process(np.array(row["column_maxes"]))[1])
        for _, row in rows.iterrows()
    ])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tracks", type=int, default=16,
                        help="Number of example tracks to plot per particle (default 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-muons", action="store_true",
                        help="Skip muon data even if available")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("Loading col.pkl…")
    col = pd.read_pickle(COL_PKL)
    ind = pd.read_pickle(IND_PKL)
    col, ind = image_cuts(col, ind, lower=10)
    print(f"  {len(col)} rows after cuts")

    protons = col[col["particle_type"] == "proton"]
    kaons   = col[col["particle_type"] == "kaon"]
    print(f"  protons: {len(protons)}  kaons: {len(kaons)}")

    particles = {"proton": protons, "kaon": kaons}

    if not args.no_muons and MUON_COL_PKL.exists():
        muons = pd.read_pickle(MUON_COL_PKL)
        print(f"  muons:   {len(muons)}")
        particles["muon"] = muons
    elif not args.no_muons:
        print(f"  muon pkl not found at {MUON_COL_PKL}, skipping")

    # ── per-particle track grids ──
    print("\nPlotting track grids…")
    for particle, rows in particles.items():
        plot_tracks(rows, particle, args.n_tracks, args.seed, OUT_DIR)

    # ── peak count histograms ──
    print("\nComputing peak count distributions…")
    counts_by_particle = {}
    for particle, rows in particles.items():
        counts = compute_counts(rows)
        counts_by_particle[particle] = counts
        mean, med = counts.mean(), np.median(counts)
        print(f"  {particle:6s}  mean={mean:.2f}  median={med:.0f}  "
              f"max={counts.max()}  frac>1={100*(counts>1).mean():.1f}%")

    plot_histogram(counts_by_particle, OUT_DIR)

    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
