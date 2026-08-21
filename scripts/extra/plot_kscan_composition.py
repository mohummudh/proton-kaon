#!/usr/bin/env python3
"""
scripts/extra/plot_kscan_composition.py

Species composition of every cluster, one figure per k.

READING ORDER
    Clusters are sorted by descending (proton - MIP), the one key that makes
    both requested orderings hold together: more proton moves a bar left, more
    MIP moves it right. So proton falls and MIP rises across the axis at the
    same time, and kaon-dominated clusters -- low in both -- settle in the
    middle. Cluster numbering on the x axis is positional, not the mixture's own
    component index.

    No ordering can make both hold exactly, because kaon absorbs whatever proton
    and MIP do not; this is the one that minimises the tension. --sort
    proton-then-mip gives the plain lexicographic rule instead, which is the
    literal reading but scatters the kaon-dominated clusters.

    Within each bar: proton at the bottom, kaon in the middle, MIPs on top --
    the same vertical order as the sort, and the same colours used everywhere
    else in the project.

NO LABEL ENTERS THE FIT
    The mixture is fitted on the raw 8D latents. Beam tags are read back only to
    colour the bars, exactly as in cluster_latents.py, so the k=3 figure
    reproduces the composition already in the paper.

UNIFORM WIDTH HIDES CLUSTER SIZE; --mosaic FIXES IT
    With equal-width bars a 100%-pure cluster of 63 events looks exactly like
    one of 2500, and at high k that distinction is the whole question. --mosaic
    scales each bar's width by its event count and packs the bars adjacently, so
    the x axis becomes the cumulative fraction of the dataset and every coloured
    block's AREA is the number of events of that species in that cluster. The
    full rectangle is then the whole sample, and a thin sliver is visibly a
    sliver rather than an equal citizen.

    Bars holding more than 3% of the sample are labelled with that percentage
    (bare number, no symbol); the rest are identifiable by position and in
    kscan_clusters.csv.

--kaon-mass: THE COMPOSITION AND THE MASS RESULT IN ONE FIGURE
    Shades each bar's kaon segment by the median beamline mass of the
    kaon-tagged events in that cluster, instead of flat orange. Mass is measured
    by the spectrometer and never seen by the VAE, so this overlays an external
    measurement on an unsupervised partition: proton-rich clusters should carry
    visibly heavier kaon-tagged events than the pure-kaon ones.

    The colour scale is shared across every k drawn in one invocation, so the
    figures are comparable. Clusters with fewer than --min-kaon kaon-tagged
    events are drawn grey -- their median is not meaningfully determined.

    Bound on the reading: the kaon TAG is itself a mass selection (348.9-648.2
    MeV, hard edges), so a heavy segment means "sits at the heavy edge of the
    kaon window", not "these are protons".

OUTPUTS (under figs/<model_name>/kscan/composition/)
    composition_k<k>.{png,pdf}          one per k
    mosaic/composition_k<k>.{png,pdf}   with --mosaic

Labels are cached per (k, seed) next to the inference files, so re-running to
restyle does not refit 48 mixtures.

Usage:
    python scripts/extra/plot_kscan_composition.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_kscan_composition.py --config ... --kmin 12 --kmax 12
    python scripts/extra/plot_kscan_composition.py --config ... --width-by-size
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.patches import Patch
from sklearn.mixture import GaussianMixture

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PDG_MASS, SINGLE_COL,
                        SPECIES, apply_style, build_model_name, figure_dir,
                        load_beam_data, load_config, savefig)
from cluster_k_sweep import composition, labels_for

# bottom -> top, matching the sort order
STACK = ["proton", "kaon", "muon"]


def plot_one(rows, k, out_dir, mosaic=False, kaon_mass=None, norm=None,
             cmap="YlOrRd"):
    n_bar = len(rows)
    fig_w = float(np.clip(0.13 * n_bar + 1.4, SINGLE_COL, DOUBLE_COL))
    scale = apply_style(min(fig_w, SINGLE_COL * 1.3))
    fig, ax = plt.subplots(figsize=(fig_w, fig_w * 0.62))
    sizes = np.array([r["n"] for r in rows], dtype=float)

    if mosaic:
        # width proportional to size, packed adjacently: area == events
        frac = sizes / sizes.sum()
        gap = 0.0016                      # constant sliver, keeps edges readable
        w = np.maximum(frac - gap, frac * 0.35)
        edges = np.concatenate([[0.0], np.cumsum(frac)])
        x = edges[:-1] + frac / 2
        x_lo, x_hi = -0.004, 1.004
    else:
        w = np.full(n_bar, 0.82)
        x = np.arange(n_bar, dtype=float)
        x_lo, x_hi = -0.7, n_bar - 0.3

    cm = plt.get_cmap(cmap)
    bottom = np.zeros(n_bar)
    for sp in STACK:
        h = np.array([r[sp] for r in rows]) * 100
        if kaon_mass is not None and sp == "kaon":
            col = [cm(norm(kaon_mass[r["cluster"]]))
                   if np.isfinite(kaon_mass.get(r["cluster"], np.nan)) else "0.78"
                   for r in rows]
        else:
            col = COLOURS[sp]
        ax.bar(x, h, bottom=bottom, width=w, color=col,
               edgecolor="white" if mosaic else "0.25",
               linewidth=(0.3 if mosaic else 0.4) * scale, label=DISPLAY[sp])
        bottom += h

    ax.set_ylim(0, 100)
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylabel("Cluster composition [%]")
    ax.set_yticks([0, 25, 50, 75, 100])
    # mosaic labels sit above the bars, so the title needs to clear them
    ax.set_title(f"$k$ = {k}" + (f"   ({int(sizes.sum()):,} events)" if mosaic else ""),
                 loc="left", fontsize=9 * scale, pad=(13 if mosaic else 3) * scale)

    if mosaic:
        ax.set_xlabel("Cumulative fraction of events")
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_xticklabels(["0", "25%", "50%", "75%", "100%"],
                           fontsize=6.5 * scale)
        # share of the whole sample, which is what the bar width already
        # encodes -- the label just makes it readable
        for xi, wi in zip(x, frac):
            if wi > 0.03:
                ax.text(xi, 101.5, f"{wi * 100:.0f}", ha="center", va="bottom",
                        fontsize=5.8 * scale, color="0.35")
    else:
        ax.set_xlabel("Cluster (descending proton, ascending MIP)")
        step = 1 if n_bar <= 20 else 5
        ax.set_xticks(x[::step])
        ax.set_xticklabels([str(i) for i in range(0, n_bar, step)],
                           fontsize=6.5 * scale)
    # figure-level legend with reserved space rather than an axes-fraction
    # offset: the same offset is a different number of inches at SINGLE_COL and
    # DOUBLE_COL, so a fixed one collided with the x label on narrow figures
    handles = []
    for sp in reversed(STACK):                                # top of bar first
        if kaon_mass is not None and sp == "kaon":
            handles.append(Patch(facecolor=cm(0.6), edgecolor="0.25",
                                 linewidth=0.4 * scale,
                                 label=f"{DISPLAY[sp]} (shaded by mass)"))
        else:
            handles.append(Patch(facecolor=COLOURS[sp], edgecolor="0.25",
                                 linewidth=0.4 * scale, label=DISPLAY[sp]))
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               fontsize=7 * scale, handlelength=1.2, handletextpad=0.4,
               columnspacing=1.4, bbox_to_anchor=(0.5, -0.01))
    if kaon_mass is not None:
        cb = fig.colorbar(ScalarMappable(norm=norm, cmap=cm), ax=ax,
                          fraction=0.038, pad=0.02)
        cb.set_label("Median beamline mass of\nkaon-tagged events [MeV]",
                     fontsize=7 * scale)
        cb.ax.tick_params(labelsize=6.5 * scale)
        cb.ax.axhline(PDG_MASS["kaon"], color="0.15", lw=0.9 * scale)
        cb.outline.set_linewidth(0.6 * scale)
    fig.tight_layout(rect=(0, 0.11, 1, 1))
    return savefig(fig, out_dir, f"composition_k{k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sort", choices=["gradient", "proton-then-mip"],
                    default="gradient", help="bar ordering; see the module docstring")
    ap.add_argument("--kaon-mass", action="store_true",
                    help="shade kaon segments by the median beamline mass of "
                         "their kaon-tagged events (implies --mosaic)")
    ap.add_argument("--min-kaon", type=int, default=20,
                    help="--kaon-mass: clusters with fewer kaon-tagged events "
                         "than this are drawn grey")
    ap.add_argument("--mosaic", action="store_true",
                    help="bar width proportional to cluster size, packed "
                         "adjacently, so block area is the event count")
    ap.add_argument("--recompute", action="store_true", help="ignore cached labels")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    species = df["species"].to_numpy()
    cache_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    out_dir = (Path(args.out_dir) if args.out_dir
               else figure_dir(cfg, "kscan") / "composition")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"composition figures for k={args.kmin}..{args.kmax} -> {out_dir}")
    ks = list(range(args.kmin, args.kmax + 1))
    labs = {k: labels_for(Z, k, args.seed, cache_dir, args.recompute) for k in ks}

    masses, norm = None, None
    if args.kaon_mass:
        # one colour scale across every k drawn here, so the figures compare
        mass = df["beamline_mass"].to_numpy(float)
        kt = (species == "kaon") & np.isfinite(mass)
        masses = {}
        for k in ks:
            masses[k] = {}
            for c in range(k):
                v = mass[(labs[k] == c) & kt]
                masses[k][c] = (float(np.median(v)) if len(v) >= args.min_kaon
                                else np.nan)
        allv = [v for m in masses.values() for v in m.values() if np.isfinite(v)]
        norm = Normalize(vmin=float(np.min(allv)), vmax=float(np.max(allv)))
        print(f"  mass colour scale {norm.vmin:.0f}-{norm.vmax:.0f} MeV "
              f"(shared across k={args.kmin}..{args.kmax})")

    for k in ks:
        plot_one(composition(labs[k], species, k, args.sort), k, out_dir,
                 mosaic=args.mosaic or args.kaon_mass,
                 kaon_mass=masses[k] if masses else None, norm=norm)


if __name__ == "__main__":
    main()
