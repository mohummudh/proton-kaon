#!/usr/bin/env python3
"""
scripts/extra/plot_proxy_hists.py

Counts-based histograms of the two physics proxies, read from the precomputed
features table (scripts/compute_features.py output):

  - mean_adc  → "Calorimetry Proxy"
  - solidity  → "Topology Proxy"

Each species is a solid step outline; behind them a light grey filled histogram
gives the total count per bin across all species. No median line, y-axis is raw
counts (not density). Each figure is written as both .pdf (paper) and .png
(quick preview).

Usage:
    python scripts/extra/plot_proxy_hists.py
    python scripts/extra/plot_proxy_hists.py --bins 60
    python scripts/extra/plot_proxy_hists.py --full-range
    python scripts/extra/plot_proxy_hists.py --features mean_adc solidity total_adc
    python scripts/extra/plot_proxy_hists.py --out-dir figs/proxy_hists_v2
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT    = Path(__file__).resolve().parent.parent.parent
FEATURES_PKL = Path("/Volumes/easystore/proton-kaon/features/features.pkl")
OUT_DIR      = REPO_ROOT / "figs" / "proxy_hists"

# fixed species order — Paul Tol palette, matching scripts/extra/plot_umap_all.py
SPECIES = [
    ("proton",    "Proton",     "#0077BB"),
    ("kaon",      "Kaon",       "#EE7733"),
    ("muon",      "MIPs",      "#AA3377"),
    ("csda_kaon", "CSDA-Kaon",  "#CC3311"),
]

# feature → x-axis label
PROXY_LABELS = {
    "mean_adc": "Calorimetry Proxy",
    "solidity": "Topology Proxy",
}

# feature → legend corner (whichever one the distributions leave empty)
LEGEND_LOC = {
    "solidity": "upper left",
}
DEFAULT_LEGEND_LOC = "upper right"

TOTAL_FILL = "#D9D9D9"
TOTAL_EDGE = "#BDBDBD"
DPI = 300

# journal column widths (inches) — same constants as plot_umap_all.py
SINGLE_COL = 3.375   # ~86 mm  — single column (that script's default width)
DOUBLE_COL = 6.875   # ~175 mm — double / full width


def apply_style(fig_w):
    """Publication settings from plot_umap_all.py, scaled to the figure width.

    That script draws 9/8 pt text on a SINGLE_COL-wide figure. Text and strokes
    are scaled by fig_w / SINGLE_COL so a wider figure keeps the same *relative*
    look — 9 pt on a 3.375" figure reads far larger than 9 pt on a 6.875" one.
    """
    s = fig_w / SINGLE_COL
    plt.rcParams.update({
        # Font
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          9 * s,
        "axes.labelsize":     9 * s,
        "xtick.labelsize":    8 * s,
        "ytick.labelsize":    8 * s,
        "legend.fontsize":    8 * s,
        "legend.title_fontsize": 8 * s,
        # Lines / ticks
        "axes.linewidth":     0.6 * s,
        "xtick.major.width":  0.6 * s,
        "ytick.major.width":  0.6 * s,
        "xtick.major.size":   3.0 * s,
        "ytick.major.size":   3.0 * s,
        "xtick.minor.visible": False,
        "ytick.minor.visible": False,
        # Output
        "figure.dpi":         DPI,
        "savefig.dpi":        DPI,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    })
    return s


def proxy_hist(feat_df, feature, bins=50, trim=True, trim_quantile=0.999,
               fig_w=SINGLE_COL, aspect=1.25):
    """Counts histogram: grey total behind, one solid step outline per species."""
    present = []
    for key, label, colour in SPECIES:
        vals = feat_df.loc[feat_df["particle_type"] == key, feature]
        vals = vals[np.isfinite(vals)].to_numpy()
        if len(vals):
            present.append((label, colour, vals))
    if not present:
        raise ValueError(f"no finite values for '{feature}'")

    all_vals   = np.concatenate([v for _, _, v in present])
    bin_edges  = np.linspace(all_vals.min(), all_vals.max(), bins + 1)
    centres    = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    total = np.zeros(bins)
    per_species = []
    for label, colour, vals in present:
        counts, _ = np.histogram(vals, bins=bin_edges)
        total += counts
        per_species.append((label, colour, counts))

    s = apply_style(fig_w)
    fig, ax = plt.subplots(figsize=(fig_w, round(fig_w / aspect, 3)))

    # background: total counts regardless of species
    ax.stairs(total, bin_edges, fill=True, color=TOTAL_FILL,
              edgecolor=TOTAL_EDGE, linewidth=0.5 * s, zorder=1, label="All species")

    # foreground: solid outline per species
    for label, colour, counts in per_species:
        ax.stairs(counts, bin_edges, color=colour, linewidth=0.9 * s,
                  zorder=3, label=label)

    ax.set_xlabel(PROXY_LABELS.get(feature, feature))
    ax.set_ylabel("Counts")

    if trim:
        # view-only trim: bins (and therefore counts) still span the full range
        cum   = np.cumsum(total) / total.sum()
        upper = bin_edges[1:][cum >= trim_quantile]
        ax.set_xlim(bin_edges[0], upper[0] if len(upper) else bin_edges[-1])
    else:
        ax.set_xlim(bin_edges[0], bin_edges[-1])
    # headroom so the legend clears the bars — the legend takes a larger share of
    # a small figure, so the smaller the figure the more headroom it needs
    ax.set_ylim(0, total.max() * (1 + 0.35 / s))

    ax.legend(loc=LEGEND_LOC.get(feature, DEFAULT_LEGEND_LOC))
    fig.tight_layout()
    return fig, ax, centres


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--features", nargs="+", default=["mean_adc", "solidity"],
                   help="features to plot (default: mean_adc solidity)")
    p.add_argument("--features-pkl", type=Path, default=FEATURES_PKL)
    p.add_argument("--out-dir", type=Path, default=OUT_DIR)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--width", type=float, default=SINGLE_COL,
                   help=f"figure width in inches (default {SINGLE_COL}, the "
                        "plot_umap_all.py single-column default); text and "
                        "strokes scale with it")
    p.add_argument("--aspect", type=float, default=1.25,
                   help="width / height (default 1.25, as in plot_umap_all.py)")
    p.add_argument("--double-col", action="store_true",
                   help=f"shorthand for --width {DOUBLE_COL}")
    p.add_argument("--full-range", action="store_true",
                   help="do not trim empty x-axis tail (default trims the view "
                        "to the 99.9%% quantile of the total)")
    args = p.parse_args()

    feat_df = pd.read_pickle(args.features_pkl)
    print(f"Loaded {len(feat_df)} rows from {args.features_pkl}")
    print(feat_df["particle_type"].value_counts().to_string())

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for feature in args.features:
        if feature not in feat_df.columns:
            print(f"  skipped {feature}: not in features table")
            continue
        fig, _, _ = proxy_hist(feat_df, feature, bins=args.bins,
                               trim=not args.full_range,
                               fig_w=DOUBLE_COL if args.double_col else args.width,
                               aspect=args.aspect)
        out = args.out_dir / f"{feature}_counts"
        fig.savefig(out.with_suffix(".pdf"))
        fig.savefig(out.with_suffix(".png"), dpi=300)
        plt.close(fig)
        print(f"  saved {out.name}.pdf  +  {out.name}.png")


if __name__ == "__main__":
    main()
