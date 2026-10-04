#!/usr/bin/env python3
"""
scripts/extra/plot_kscan_embeddings.py

How the unsupervised partition subdivides the latent space as k grows, shown on
both 2D projections.

WHAT IS AND IS NOT BEING VARIED
    The EMBEDDINGS ARE FIXED -- one cached UMAP and one cached t-SNE, identical
    in every panel. Only the colouring changes. So differences between panels are
    differences in the CLUSTERING, never in the projection, which is the whole
    point of laying them out this way.

    Every mixture is fitted in the native 8D latents, never on the 2D embedding:
    clustering the projection scores roughly half the ARI (0.18-0.19 vs 0.37)
    because the embedding discards the covariance structure a full-covariance
    mixture depends on.

TWO COLOURINGS, ANSWERING DIFFERENT QUESTIONS
    --colour cluster  each component its own colour. Shows how the space
                      subdivides: which regions split first, which stay intact.
    --colour species  each component takes the colour of the species that
                      dominates it. Shows how the partition relates to physics:
                      whether new components are sub-populations of one species
                      or fresh mixtures.

    Both are worth looking at. The first shows the geometry, the second shows
    whether the geometry means anything.

READ REGION MEMBERSHIP, NOT GAPS. Both projections manufacture whitespace by
construction; the latent space is a continuum with no species-aligned density
modes. Colour tells you which events group together, and that is all.

OUTPUTS (under figs/<model_name>/kscan/)
    kscan_tsne_<colour>.{png,pdf}
    kscan_umap_<colour>.{png,pdf}

Usage:
    python scripts/extra/plot_kscan_embeddings.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_kscan_embeddings.py --config ... --kmax 12 --colour species
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, SINGLE_COL, SPECIES,
                        apply_style, build_model_name, figure_dir,
                        load_beam_data, load_config, load_embedding, savefig)


def fit_all(Z, ks, seed=0):
    """One full-covariance mixture per k, fitted in the native 8D space."""
    out = {}
    for k in ks:
        out[k] = GaussianMixture(k, covariance_type="full", n_init=20,
                                 random_state=seed).fit_predict(Z)
        print(f"    k={k:2d} fitted", flush=True)
    return out


def cluster_colours(labels, truth, mode):
    """Per-event colours plus the legend handles that go with them."""
    k = int(labels.max()) + 1
    if mode == "species":
        cols = []
        for c in range(k):
            cnt = np.bincount(truth[labels == c], minlength=3)
            cols.append(COLOURS[SPECIES[int(cnt.argmax())]])
        return np.array(cols)[labels]
    cmap = plt.get_cmap("tab20")
    return np.array([cmap(c % 20) for c in range(k)], dtype=object)[labels]


def grid(E, labels_by_k, truth, name, out_dir, mode, ncols=4, s=0.9):
    ks = sorted(labels_by_k)
    nrows = int(np.ceil(len(ks) / ncols))
    scale = apply_style(DOUBLE_COL / ncols)      # one PANEL width
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(DOUBLE_COL, DOUBLE_COL / ncols * nrows * 1.06),
                             squeeze=False, sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    order = rng.permutation(len(E))              # no component drawn on top
    for i, k in enumerate(ks):
        ax = axes[i // ncols][i % ncols]
        lab = labels_by_k[k]
        cols = cluster_colours(lab, truth, mode)
        ax.scatter(E[order, 0], E[order, 1], s=s * scale,
                   c=list(cols[order]), linewidths=0, alpha=0.55)
        ax.set_title(f"$k$ = {k}   (ARI {adjusted_rand_score(truth, lab):.2f})",
                     pad=2.5, fontsize=7.5 * scale)
        ax.set_xticks([]); ax.set_yticks([])
    for j in range(len(ks), nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    if mode == "species":
        fig.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4 * scale,
                                   color=COLOURS[sp],
                                   label=f"majority {DISPLAY[sp]}")
                            for sp in SPECIES],
                   loc="lower center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.5, -0.015))
        fig.tight_layout(rect=(0, 0.035, 1, 1))
    else:
        fig.tight_layout()
    return savefig(fig, out_dir, f"kscan_{name}_{mode}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=18)
    ap.add_argument("--colour", choices=["cluster", "species", "both"], default="both")
    ap.add_argument("--perplexity", type=float, default=30.0,
                    help="which cached t-SNE embedding to use")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ncols", type=int, default=4)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    truth = np.array([SPECIES.index(s) for s in df["species"]])
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(cfg, "kscan")

    inf = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    tsne_path = inf / f"tsne_p{args.perplexity:g}_seed{args.seed}.npy"
    if not tsne_path.exists():
        raise FileNotFoundError(
            f"{tsne_path.name} not found. Run plot_tsne.py first so both figures "
            f"share one embedding rather than fitting a second, different one.")
    embeddings = {"tsne": np.load(tsne_path), "umap": load_embedding(cfg, Z)}

    ks = list(range(args.kmin, args.kmax + 1))
    print(f"fitting {len(ks)} mixtures on {len(Z)} events x {Z.shape[1]}D", flush=True)
    labels_by_k = fit_all(Z, ks, args.seed)

    modes = ["cluster", "species"] if args.colour == "both" else [args.colour]
    for name, E in embeddings.items():
        for mode in modes:
            grid(E, labels_by_k, truth, name, out_dir, mode, ncols=args.ncols)


if __name__ == "__main__":
    main()
