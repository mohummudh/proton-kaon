#!/usr/bin/env python3
"""
scripts/extra/plot_latent_densities.py

Per-dimension distribution of the latent space, decomposed by beam species.

WHAT THIS ANSWERS
    The UMAP and t-SNE figures show species occupying different regions, but a
    projection cannot say WHERE that separation lives. This does: it plots each
    latent coordinate's marginal, so you can see directly which dimensions carry
    species information and which are shared.

    Expect the marginals to OVERLAP heavily. Every one of the eight is unimodal
    on its own (Hartigan dip p = 0.71-1.00), and the species separation in this
    space is carried by direction and covariance rather than by any single axis
    -- which is exactly why a full-covariance mixture beats k-means here. This
    figure is the honest picture of that: no axis is a discriminant, yet the
    joint distribution is strongly species-aligned.

GREY IS WHAT THE MODEL SEES
    The grey total is the pooled distribution over all events -- the only thing
    the VAE ever had access to. The coloured curves are the decomposition by
    beam tag, which it was never told. Same convention as the proxy histograms
    and the reconstruction-error figure, so grey means the same thing in every
    figure in the paper.

    Under --density each curve integrates to 1, so the coloured curves do NOT
    sum to the grey one; use --counts for the additive version.

ETA-SQUARED
    Each panel is annotated with the fraction of that dimension's variance
    explained by species (one-way, between-group over total). It is bounded
    [0, 1] and reads directly as "how much of this coordinate is species".

OUTPUTS (under figs/<model_name>/latents-features/)
    latent_densities.{png,pdf}

Usage:
    python scripts/extra/plot_latent_densities.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_latent_densities.py --config ... --counts
    python scripts/extra/plot_latent_densities.py --config ... --sort-by-separation
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, SINGLE_COL, SPECIES,
                        apply_style, figure_dir, load_beam_data, load_config,
                        savefig, select)

TOTAL_FILL, TOTAL_EDGE = "#D9D9D9", "#BDBDBD"


def eta_squared(x, species):
    """Fraction of x's variance explained by species (one-way, between/total)."""
    grand = x.mean()
    between = sum(((x[species == s].mean() - grand) ** 2) * (species == s).sum()
                  for s in SPECIES)
    total = ((x - grand) ** 2).sum()
    return float(between / total) if total > 0 else 0.0


def draw_dim(ax, x, species, edges, scale, density):
    """Grey pooled distribution behind, one coloured step outline per species."""
    kw = dict(bins=edges, density=density)
    total, _ = np.histogram(x, **kw)
    ax.stairs(total, edges, fill=True, color=TOTAL_FILL, edgecolor=TOTAL_EDGE,
              linewidth=0.5 * scale, zorder=1, label="All species")
    for s in SPECIES:
        v = x[species == s]
        if len(v):
            c, _ = np.histogram(v, **kw)
            ax.stairs(c, edges, color=COLOURS[s], linewidth=0.9 * scale,
                      zorder=3, label=DISPLAY[s])
    return total


def plot_densities(Z, df, out_dir, bins=60, density=True, sort_by_sep=False,
                   ncols=4):
    species = df["species"].to_numpy()
    D = Z.shape[1]
    etas = np.array([eta_squared(Z[:, i], species) for i in range(D)])
    order = np.argsort(-etas) if sort_by_sep else np.arange(D)

    nrows = int(np.ceil(D / ncols))
    scale = apply_style(DOUBLE_COL / ncols)   # one PANEL width, not the figure
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL, DOUBLE_COL / ncols * nrows * 1.05),
                             squeeze=False)
    for k, dim in enumerate(order):
        ax = axes[k // ncols][k % ncols]
        x = Z[:, dim]
        # common edges from robust percentiles: a few extreme events otherwise
        # squeeze the entire distribution into two or three bins
        lo, hi = np.percentile(x, [0.2, 99.8])
        edges = np.linspace(lo, hi, bins + 1)
        draw_dim(ax, x, species, edges, scale, density)
        ax.set_xlabel(f"$z_{{{dim}}}$")
        if k % ncols == 0:
            ax.set_ylabel("Density" if density else "Counts")
        ax.set_xlim(lo, hi)
        ax.tick_params(labelsize=7 * scale)
        ax.text(0.03, 0.95, rf"$\eta^2={etas[dim]:.2f}$", transform=ax.transAxes,
                va="top", ha="left", fontsize=7.5 * scale)
    for k in range(D, nrows * ncols):          # blank any unused cell
        axes[k // ncols][k % ncols].axis("off")

    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    print("  eta^2 by dimension (variance explained by species):")
    for dim in np.argsort(-etas):
        print(f"    z{dim}: {etas[dim]:.3f}")
    return savefig(fig, out_dir, "latent_densities")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model YAML (all-species run)")
    ap.add_argument("--bins", type=int, default=60)
    ap.add_argument("--counts", action="store_true",
                    help="raw counts instead of density; coloured curves then sum "
                         "to the grey total")
    ap.add_argument("--sort-by-separation", action="store_true",
                    help="order panels by eta^2 rather than by dimension index")
    ap.add_argument("--picky", type=int, choices=[0, 1], default=None)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    if args.picky is not None:
        Z, df = select(Z, df, picky=args.picky)
    out_dir = args.out_dir or figure_dir(cfg, "latents-features")
    print(f"latent densities: {len(Z)} events x {Z.shape[1]}D -> {out_dir}")
    plot_densities(Z, df, out_dir, bins=args.bins, density=not args.counts,
                   sort_by_sep=args.sort_by_separation, ncols=args.ncols)


if __name__ == "__main__":
    main()
