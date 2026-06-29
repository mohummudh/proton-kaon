#!/usr/bin/env python3
"""
scripts/extra/plot_disentanglement.py

Standalone disentanglement heatmap — Spearman ρ between latent dims and features.
Diverging colormap, horizontal line separating calo from topo, subplot labels.

Edit DATA and STYLE at the top; nothing else needs touching.

Usage:
    python scripts/extra/plot_disentanglement.py
    python scripts/extra/plot_disentanglement.py --out figs/disentanglement.pdf
"""

# ══════════════════════════════════════════════════════════════════════════════
# DATA  —  Spearman ρ values
# ══════════════════════════════════════════════════════════════════════════════

Z_DIMS = ["z0", "z1", "z2", "z3", "z4", "z5", "z6", "z7"]

# Each panel:
#   label    : subplot letter shown below, e.g. "(a)"
#   title    : text above the heatmap; set "" to hide
#   calo     : list of (feature_name, [ρ per z-dim])  — above the separator line
#   topo     : list of (feature_name, [ρ per z-dim])  — below the separator line

PANELS = [
    {
        "label": "(a)",
        "title": "VAE Latent Disentanglement: Feature Correlation (Kaons only)",
        "calo": [
            ("mean_adc",  [ 0.15,  0.05, -0.16,  0.08, -0.36,  0.01, -0.32, -0.28]),
            ("total_adc", [ 0.01, -0.21, -0.21,  0.07,  0.34,  0.03,  0.09,  0.24]),
        ],
        "topo": [
            ("solidity",  [ 0.11,  0.16, -0.12,  0.04, -0.54,  0.11, -0.10, -0.28]),
        ],
    },
    {
        "label": "(b)",
        "title": "Muon VAE Latent Disentanglement: Feature Correlation",
        "calo": [
            ("mean_adc",  [ 0.43, -0.02,  0.46,  0.34,  0.31, -0.08, -0.54, -0.02]),
            ("total_adc", [-0.11, -0.17,  0.02, -0.04,  0.03, -0.00,  0.05,  0.19]),
        ],
        "topo": [
            ("solidity",  [-0.15,  0.20, -0.44, -0.07, -0.49,  0.09,  0.18, -0.08]),
        ],
    },
]

# ══════════════════════════════════════════════════════════════════════════════
# STYLE
# ══════════════════════════════════════════════════════════════════════════════

# Figure
FIG_W       = 8.0     # inches
ROW_H       = 0.35    # inches per feature row
TITLE_H     = 0.55    # extra height per panel for title
LABEL_H     = 0.40    # extra height per panel for the (a)/(b) label below
HSPACE      = 0.65    # vertical space between panels (relative)

# Colormap — diverging, centred at 0
CMAP        = "RdBu_r"
VMIN, VMAX  = -1.0, 1.0

# Cell annotations
ANNOT_SIZE  = 12
ANNOT_FMT   = ".2f"

# Grid lines between cells
LINEWIDTHS  = 0.5
LINECOLOR   = "white"

# Separator line between calo and topo rows
SEP_COLOR     = "black"
SEP_LINEWIDTH = 2.0

# Colorbar
CBAR_LABEL  = "Spearman $\\rho$"
CBAR_SHRINK = 0.85

# Subplot label style (the "(a)", "(b)" text)
SUBLABEL_SIZE   = 10
SUBLABEL_WEIGHT = "normal"
SUBLABEL_Y      = -0.18   # fraction below axes

# Axis labels
XLABEL      = "Latent Dimensions"
YLABEL      = "Features"
TITLE_SIZE  = 10
TITLE_PAD   = 6

# Font
FONT_FAMILY = "serif"
FONT_SERIF  = ["Times New Roman", "DejaVu Serif"]
FONT_SIZE   = 14

DPI = 300

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


def draw_panel(panel, fig, ax):
    feats  = [name for name, _ in panel["calo"]] + [name for name, _ in panel["topo"]]
    values = [vals  for _, vals  in panel["calo"]] + [vals  for _, vals  in panel["topo"]]
    mat    = np.array(values, dtype=float)
    n_rows, n_cols = mat.shape
    n_calo = len(panel["calo"])

    im = ax.imshow(mat, aspect="auto", cmap=CMAP, vmin=VMIN, vmax=VMAX, origin="upper")

    for r in range(n_rows):
        for c in range(n_cols):
            val = mat[r, c]
            intensity = abs(val) / max(abs(VMIN), abs(VMAX))
            color = "white" if intensity > 0.55 else "black"
            ax.text(c, r, f"{val:{ANNOT_FMT}}", ha="center", va="center",
                    fontsize=ANNOT_SIZE, color=color)

    for c in range(n_cols + 1):
        ax.axvline(c - 0.5, color=LINECOLOR, linewidth=LINEWIDTHS)
    for r in range(n_rows + 1):
        ax.axhline(r - 0.5, color=LINECOLOR, linewidth=LINEWIDTHS)

    ax.axhline(n_calo - 0.5, color=SEP_COLOR, linewidth=SEP_LINEWIDTH)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(Z_DIMS, rotation=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(feats, rotation=0)
    ax.tick_params(length=0)
    ax.set_xlabel(XLABEL)

    cbar = fig.colorbar(im, ax=ax, shrink=CBAR_SHRINK, pad=0.02)
    cbar.set_label(CBAR_LABEL, fontsize=FONT_SIZE)
    cbar.ax.tick_params(labelsize=FONT_SIZE - 1)


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    out_dir = Path(__file__).resolve().parent.parent.parent / "figs"
    out_dir.mkdir(parents=True, exist_ok=True)

    for panel in PANELS:
        n_rows = len(panel["calo"]) + len(panel["topo"])
        fig_h  = n_rows * ROW_H + TITLE_H
        fig, ax = plt.subplots(figsize=(FIG_W, fig_h))

        draw_panel(panel, fig, ax)

        stem = f"disentanglement_{panel['label'].strip('()')}"
        plt.savefig(out_dir / f"{stem}.pdf")
        plt.savefig(out_dir / f"{stem}.png", dpi=DPI)
        print(f"Saved {stem}.pdf  +  .png")
        plt.close()


if __name__ == "__main__":
    main()
