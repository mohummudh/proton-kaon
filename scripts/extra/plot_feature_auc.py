#!/usr/bin/env python3
"""
scripts/plot_feature_auc.py

Standalone plot — feature AUC-ROC grouped horizontal bars.
Edit the DATA and STYLE sections at the top; the rest takes care of itself.

Usage:
    python scripts/plot_feature_auc.py
    python scripts/plot_feature_auc.py --out figs/feature_auc_v2.pdf
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA  —  edit these numbers
# ══════════════════════════════════════════════════════════════════════════════

# Each entry: (feature_label, category, {particle: auc, ...})
# category is used only for the y-axis label suffix — "calo" or "topo"
# Add/remove entries to change what's plotted; order here = order on plot (top → bottom)
DATA = [
    ("mean_adc",  "calo", {"Proton": 0.944, "Kaon": 0.795, "Muon": 0.792}),
    ("solidity",  "topo", {"Proton": 0.763, "Kaon": 0.826, "Muon": 0.752}),
    ("total_adc", "calo", {"Proton": 0.963, "Kaon": 0.748, "Muon": 0.620}),
]

# Particle display order (top bar to bottom bar within each feature group)
PARTICLE_ORDER = ["Proton", "Kaon", "Muon"]

# ══════════════════════════════════════════════════════════════════════════════
# STYLE  —  tweak visuals here
# ══════════════════════════════════════════════════════════════════════════════

COLORS = {
    "Proton": "#0077BB",
    "Kaon":   "#EE7733",
    "Muon":   "#AA3377",
}

# Figure dimensions (inches)
FIG_W = 6.875   # double-column width
FIG_H = 4.6     # increase if labels overlap

# Bar geometry
BAR_H        = 0.07    # height of each individual bar
GROUP_PAD    = 0.30    # vertical space between feature groups (centres)
BAR_ALPHA    = 0.88    # bar fill opacity

# Label on the right of each bar
LABEL_OFFSET = 0.006   # gap between bar end and value text
LABEL_SIZE   = 14
LABEL_BOLD   = True

# Y-axis tick labels
YTICK_SIZE   = 14

# Axis
XLIM         = (0.40, 1.03)   # x range
CHANCE_LINE  = 0.5            # dashed reference; set to None to hide

# Legend
LEGEND_LOC        = "center right"
LEGEND_MARKER_SIZE = 28       # pt²

# Font
FONT_FAMILY = "serif"
FONT_SERIF  = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE   = 12

# Output
DPI = 300

# ══════════════════════════════════════════════════════════════════════════════
# PLOT  —  no need to edit below this line
# ══════════════════════════════════════════════════════════════════════════════

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

plt.rcParams.update({
    "font.family":         FONT_FAMILY,
    "font.serif":          FONT_SERIF,
    "font.size":           FONT_SIZE,
    "axes.labelsize":      14,
    "xtick.labelsize":     FONT_SIZE - 1,
    "ytick.labelsize":     YTICK_SIZE,
    "legend.fontsize":     FONT_SIZE - 1,
    "axes.linewidth":      0.6,
    "xtick.major.width":   0.6,
    "ytick.major.width":   0.6,
    "xtick.major.size":    3.0,
    "ytick.major.size":    3.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "figure.dpi":          DPI,
    "savefig.dpi":         DPI,
    "savefig.bbox":        "tight",
    "savefig.pad_inches":  0.02,
})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=None,
                        help="Output path (default: figs/feature_auc.pdf + .png)")
    args = parser.parse_args()

    particles_present = [p for p in PARTICLE_ORDER if any(p in d[2] for d in DATA)]
    n_bars   = len(particles_present)
    n_groups = len(DATA)

    # Centre y-positions for each feature group
    group_centres = np.arange(n_groups, dtype=float) * GROUP_PAD

    # Offsets within each group: spread bars symmetrically.
    # With invert_yaxis active, a lower y-value places the bar higher on the plot,
    # so the first particle in PARTICLE_ORDER gets the most negative offset → top bar.
    total_span = (n_bars - 1) * BAR_H
    offsets = {p: -total_span / 2 + i * BAR_H for i, p in enumerate(particles_present)}

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    for gi, (feat, cat, aucs) in enumerate(DATA):
        yc = group_centres[gi]
        for particle in particles_present:
            if particle not in aucs:
                continue
            val = aucs[particle]
            ypos = yc + offsets[particle]
            ax.barh(
                ypos, val,
                height=BAR_H * 0.88,
                color=COLORS[particle],
                alpha=BAR_ALPHA,
                edgecolor="none",
            )
            ax.text(
                val + LABEL_OFFSET, ypos,
                f"{val:.3f}",
                va="center", ha="left",
                fontsize=LABEL_SIZE,
                color=COLORS[particle],
                fontweight="bold" if LABEL_BOLD else "normal",
            )

    # Y-axis ticks at group centres
    ax.set_yticks(group_centres)
    ax.set_yticklabels(
        [f"{feat}\n({cat})" for feat, cat, _ in DATA],
        fontsize=YTICK_SIZE,
    )
    ax.invert_yaxis()

    ax.set_xlim(*XLIM)
    ax.set_xlabel("AUC-ROC")

    if CHANCE_LINE is not None:
        ax.axvline(CHANCE_LINE, color="grey", linestyle="--", linewidth=1.0,
                   label=f"Chance ({CHANCE_LINE})")

    # Grid
    ax.xaxis.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    # Legend
    handles = [
        mpatches.Patch(facecolor=COLORS[p], alpha=BAR_ALPHA, label=p)
        for p in particles_present
    ]
    if CHANCE_LINE is not None:
        handles.append(
            plt.Line2D([0], [0], color="grey", linestyle="--",
                       linewidth=1.0, label=f"Chance ({CHANCE_LINE})")
        )
    leg = ax.legend(
        handles=handles,
        loc=LEGEND_LOC,
        frameon=True, framealpha=0.85, edgecolor="0.75",
        handlelength=1.0, handletextpad=0.4,
        borderpad=0.5, labelspacing=0.35,
    )
    for lh in leg.legend_handles:
        if hasattr(lh, "set_height"):
            lh.set_height(10)

    plt.tight_layout()

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        if out.suffix != ".png":
            plt.savefig(out.with_suffix(".png"), dpi=DPI)
        print(f"Saved {out}  +  {out.with_suffix('.png')}")
    else:
        out_dir = Path(__file__).resolve().parent.parent.parent / "figs"
        out_dir.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_dir / "feature_auc.pdf")
        plt.savefig(out_dir / "feature_auc.png", dpi=DPI)
        print(f"Saved {out_dir}/feature_auc.pdf  +  .png")

    plt.close()


if __name__ == "__main__":
    main()
