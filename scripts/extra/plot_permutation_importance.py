#!/usr/bin/env python3
"""
scripts/plot_permutation_importance.py

Standalone permutation-importance heatmap.
Edit DATA and STYLE at the top; nothing else needs touching.

Usage:
    python scripts/plot_permutation_importance.py
    python scripts/plot_permutation_importance.py --out figs/perm_imp.pdf
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA  —  R² drop on permutation, per feature × latent dim
# ══════════════════════════════════════════════════════════════════════════════

# Latent dimension labels (x-axis)
Z_DIMS = ["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"]

# Each panel is a dict:
#   title    : subplot title string
#   features : ordered list of feature names (y-axis, top → bottom)
#   values   : list of rows — one list per feature, aligned to Z_DIMS
#
# Values are R² drop (permutation importance). Positive = that dim matters.

PANELS = [
    {
        "title":    "Permutation importance — calorimetry (protons)",
        "features": ["mean_adc", "total_adc"],
        "values": [
            # z0     z1     z2     z3     z4     z5     z6     z7
            [0.017, 0.008, 0.238, 0.069, 0.612, 0.186, 0.589, 1.144],  # mean_adc
            [0.017, 0.024, 0.833, 0.042, 0.157, 0.234, 0.260, 0.223],  # total_adc
        ],
    },
    {
        "title":    "Permutation importance — topology (protons)",
        "features": ["solidity"],
        "values": [
            [0.162, 0.169, 0.891, 0.275, 1.225, 0.201, 0.649, 0.974],  # solidity
        ],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════

# Figure
FIG_W         = 6.875   # inches — double column
ROW_H         = 0.72    # inches per feature row
TITLE_H       = 0.5     # inches of extra height per panel for the title
PANEL_PAD     = 0.6     # inches of vertical padding between panels
HSPACE        = 0.55    # gridspec hspace (relative)

# Colormap — sequential feels right for non-negative importance values
CMAP          = "YlOrRd"   # alternatives: "Reds", "OrRd", "RdBu_r" (diverging, center=0)
CMAP_CENTER   = None        # set to 0 if using a diverging cmap like RdBu_r
CMAP_VMIN     = 0.0         # clamp low end; set None for auto
CMAP_VMAX     = None        # clamp high end; None = max across all panels

# Cell annotations
ANNOT_SIZE    = 12
ANNOT_FMT     = ".3f"

# Grid lines between cells
LINEWIDTHS    = 0.4
LINECOLOR     = "white"

# Colorbar label
CBAR_LABEL    = "$R^2$ drop on permutation"
CBAR_SHRINK   = 0.8     # shrink colorbar height relative to axes

# Axis labels
XLABEL        = "Latent dimension"
YLABEL        = "Feature"
TITLE_SIZE    = 9.5
TITLE_WEIGHT  = "semibold"

# Font
FONT_FAMILY   = "serif"
FONT_SERIF    = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE     = 14

DPI           = 300

# ══════════════════════════════════════════════════════════════════════════════
# PLOT
# ══════════════════════════════════════════════════════════════════════════════

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.family":         FONT_FAMILY,
    "font.serif":          FONT_SERIF,
    "font.size":           FONT_SIZE,
    "axes.labelsize":      FONT_SIZE,
    "xtick.labelsize":     FONT_SIZE - 1,
    "ytick.labelsize":     FONT_SIZE - 1,
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
                        help="Output path (default: figs/permutation_importance.pdf + .png)")
    args = parser.parse_args()

    # ── compute global vmax across all panels ─────────────────────────────────
    all_vals = np.concatenate([np.array(p["values"]).ravel() for p in PANELS])
    vmax = CMAP_VMAX if CMAP_VMAX is not None else float(all_vals.max())
    vmin = CMAP_VMIN if CMAP_VMIN is not None else None

    # ── figure size: each panel gets height proportional to its feature count ─
    panel_heights = [len(p["features"]) * ROW_H + TITLE_H for p in PANELS]
    total_h = sum(panel_heights) + PANEL_PAD * (len(PANELS) - 1)
    fig = plt.figure(figsize=(FIG_W, total_h))
    gs  = fig.add_gridspec(
        len(PANELS), 1,
        height_ratios=panel_heights,
        hspace=HSPACE,
    )

    axes = []
    ims  = []
    for pi, panel in enumerate(PANELS):
        ax  = fig.add_subplot(gs[pi])
        mat = np.array(panel["values"], dtype=float)
        n_rows, n_cols = mat.shape

        im = ax.imshow(
            mat,
            aspect="auto",
            cmap=CMAP,
            vmin=vmin if vmin is not None else mat.min(),
            vmax=vmax,
            origin="upper",
        )
        axes.append(ax)
        ims.append(im)

        # cell annotations
        thresh = (im.norm.vmin + im.norm.vmax) / 2
        for r in range(n_rows):
            for c in range(n_cols):
                val = mat[r, c]
                color = "white" if val > thresh else "black"
                ax.text(c, r, f"{val:{ANNOT_FMT}}", ha="center", va="center",
                        fontsize=ANNOT_SIZE, color=color)

        # cell grid lines
        for c in range(n_cols + 1):
            ax.axvline(c - 0.5, color=LINECOLOR, linewidth=LINEWIDTHS)
        for r in range(n_rows + 1):
            ax.axhline(r - 0.5, color=LINECOLOR, linewidth=LINEWIDTHS)

        # ticks
        ax.set_xticks(range(n_cols))
        ax.set_xticklabels(Z_DIMS, rotation=0)
        ax.set_yticks(range(n_rows))
        ax.set_yticklabels(panel["features"], rotation=0)
        ax.tick_params(length=0)

        ax.set_xlabel(XLABEL)

    # shared colorbar spanning all axes
    cbar = fig.colorbar(ims[0], ax=axes, shrink=CBAR_SHRINK, pad=0.02)
    cbar.set_label(CBAR_LABEL, fontsize=FONT_SIZE - 1)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 2)

    out_dir = Path(__file__).resolve().parent.parent.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out)
        if out.suffix != ".png":
            plt.savefig(out.with_suffix(".png"), dpi=DPI)
        print(f"Saved {out}  +  {out.with_suffix('.png')}")
    else:
        plt.savefig(out_dir / "permutation_importance.pdf")
        plt.savefig(out_dir / "permutation_importance.png", dpi=DPI)
        print(f"Saved {out_dir}/permutation_importance.pdf  +  .png")

    plt.close()


if __name__ == "__main__":
    main()
