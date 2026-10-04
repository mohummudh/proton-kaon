#!/usr/bin/env python3
"""
scripts/extra/plot_tsne.py

t-SNE of the 8D VAE latent space, as a second opinion on the UMAP figures.

WHY A SECOND PROJECTION AT ALL
    Every 2D view of an 8D space is a lie of some particular kind, and the two
    methods lie differently. UMAP optimises a fuzzy topological representation
    and tends to preserve global arrangement at the cost of exaggerating gaps;
    t-SNE optimises neighbourhood probabilities and preserves local structure at
    the cost of global distances. A feature that survives both is a property of
    the latent space; a feature that appears in only one is a property of the
    projection. The specific question this was written for: protons and MIPs sit
    close in the UMAP, and it matters whether that is real or an artefact.

    Do NOT read cluster SIZE or BETWEEN-cluster distance off a t-SNE. Neither is
    meaningful. Read only which points are near which.

PERPLEXITY IS A SCALE KNOB, NOT A TUNING PARAMETER
    It sets roughly how many neighbours each point is fitted against, so it
    picks out structure at that scale. A single perplexity is a single scale, so
    --sweep draws several and the honest answer is whatever holds across them.

OUTPUTS (under figs/<model_name>/tsne/)
    tsne_all_species.{png,pdf}   species by beam tag, the t-SNE analogue of
                                 umap_all_species
    tsne_species_panel.{png,pdf} one panel per species against a grey backdrop
    tsne_proxy_<name>.{png,pdf}  the embedding coloured by each physics proxy
    tsne_proxies.{png,pdf}       both proxies over the same embedding, side by side
    tsne_perplexity.{png,pdf}    the same data at several perplexities (--sweep)
    tsne_clusters.{png,pdf}      beam tag vs unsupervised GMM assignment (--clusters),
                                 optionally plus the anchored fit (--anchored)

Embeddings are cached per (perplexity, seed) next to the inference files, so
re-running to restyle a figure does not refit. t-SNE has no transform() for new
points, so unlike the UMAP reducer what is cached is the embedding itself, which
is only valid for this exact Z.

Usage:
    python scripts/extra/plot_tsne.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_tsne.py --config ... --sweep
    python scripts/extra/plot_tsne.py --config ... --perplexity 50 --recompute
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PROXY_LABELS, SINGLE_COL,
                        SPECIES, apply_style, build_model_name, figure_dir,
                        load_beam_data, load_config, savefig, select)

BACKDROP = "#DDDDDD"

# One real collection-plane event per species, chosen to make the two visual
# cues used throughout the paper easy to see: a proton with a bright stopping
# end, a kaon with secondary activity, and a straight MIP.  Event keys are used
# instead of tensor row numbers so the selection remains auditable if the
# latent blocks are reordered.
EXAMPLE_EVENTS = {
    "proton": (8596, 287, 26620),
    "kaon": (8758, 306, 33642),
    "muon": (8629, 150, 11637),
}
HIGH_RES_PK = Path("/Volumes/easystore/proton-kaon/images/"
                   "pk_256x256_raw_10-179wires.pt")
HIGH_RES_MIP = Path("/Volumes/easystore/proton-kaon/images/muon_256x256_raw.pt")


def embed(Z, perplexity, seed, cache_dir, recompute=False, standardise=False):
    """2D t-SNE of Z, cached per (perplexity, seed).

    init='pca' rather than 'random': random init gives t-SNE no global anchor at
    all, so the arrangement of the clusters changes run to run and nothing about
    the overall layout can be trusted. PCA init at least fixes the coarse frame.
    """
    tag = f"tsne_p{perplexity:g}_seed{seed}{'_std' if standardise else ''}.npy"
    path = Path(cache_dir) / tag
    if path.exists() and not recompute:
        E = np.load(path)
        if len(E) == len(Z):
            print(f"  loaded cached {path.name}")
            return E
        print(f"  cached {path.name} has {len(E)} rows, need {len(Z)} — refitting")

    X = (Z - Z.mean(0)) / Z.std(0) if standardise else Z
    print(f"  fitting t-SNE: n={len(X)}, perplexity={perplexity}, seed={seed}"
          f"{', standardised' if standardise else ''} ...", flush=True)
    E = TSNE(n_components=2, perplexity=perplexity, init="pca",
             learning_rate="auto", random_state=seed, max_iter=1000).fit_transform(X)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, E)
    print(f"  cached {path.name}")
    return E


def gmm_labels(Z, k=3, seed=0):
    """Full-covariance mixture fit in the NATIVE 8D space, no labels involved.

    Deliberately not fitted on the 2D embedding: clustering the projection scores
    roughly half the ARI of clustering the latents (0.18-0.19 vs 0.37), because
    the embedding discards the covariance structure a full-covariance mixture
    relies on. The projection is for looking at, not for fitting.
    """
    from sklearn.mixture import GaussianMixture
    return GaussianMixture(k, covariance_type="full", n_init=20,
                           random_state=seed).fit_predict(Z)


def match_to_species(labels, species):
    """Relabel clusters to the species they mostly contain, for DISPLAY ONLY.

    Cluster indices from a mixture fit are arbitrary, so without this the colours
    would be meaningless. Uses an optimal one-to-one assignment rather than a
    greedy per-cluster majority, which can hand the same species to two clusters
    and leave a third unnamed.
    """
    from scipy.optimize import linear_sum_assignment
    truth = np.array([SPECIES.index(s) for s in species])
    k = int(labels.max()) + 1
    conf = np.zeros((k, len(SPECIES)), dtype=int)
    for c in range(k):
        conf[c] = np.bincount(truth[labels == c], minlength=len(SPECIES))
    rows, cols = linear_sum_assignment(-conf)
    mapping = dict(zip(rows, cols))
    return np.array([mapping.get(c, c) for c in labels]), conf


def _scatter_by_code(ax, E, codes, scale, s, rng):
    order = rng.permutation(len(E))
    cols = np.array([COLOURS[SPECIES[c]] for c in codes])[order]
    ax.scatter(E[order, 0], E[order, 1], s=s * scale, c=cols,
               linewidths=0, alpha=0.55)


def fig_clusters(E, Z, df, out_dir, seed=0, s=1.6, with_anchored=False):
    """t-SNE coloured by beam tag, by unsupervised GMM, and optionally anchored."""
    from sklearn.metrics import adjusted_rand_score

    species = df["species"].to_numpy()
    truth = np.array([SPECIES.index(x) for x in species])
    panels = [("Beam tag", truth, None)]

    raw = gmm_labels(Z, 3, seed)
    matched, conf = match_to_species(raw, species)
    purity = conf.max(axis=1).sum() / len(truth)
    panels.append((f"Unsupervised GMM\nARI {adjusted_rand_score(truth, raw):.2f}, "
                   f"purity {purity:.2f}", matched, None))
    print(f"  unsupervised GMM: ARI {adjusted_rand_score(truth, raw):.3f}, "
          f"purity {purity:.3f}")

    if with_anchored:
        from anchored_clustering import anchored_fit
        anch, _, _ = anchored_fit(Z, df["species"])
        a_pur = sum(np.bincount(truth[anch == c], minlength=3).max()
                    for c in np.unique(anch)) / len(truth)
        panels.append((f"Anchored (semi-sup.)\nARI "
                       f"{adjusted_rand_score(truth, anch):.2f}, purity {a_pur:.2f}",
                       anch, None))
        print(f"  anchored: ARI {adjusted_rand_score(truth, anch):.3f}, "
              f"purity {a_pur:.3f}")

    n = len(panels)
    scale = apply_style(DOUBLE_COL / n)
    fig, axes = plt.subplots(1, n, figsize=(DOUBLE_COL, DOUBLE_COL / n + 0.45),
                             sharex=True, sharey=True)
    rng = np.random.default_rng(0)
    for ax, (title, codes, _) in zip(np.atleast_1d(axes), panels):
        _scatter_by_code(ax, E, codes, scale, s, rng)
        ax.set_title(title, pad=4, fontsize=8 * scale)
        _bare(ax)
        ax.set_ylabel("")
    np.atleast_1d(axes)[0].set_ylabel("t-SNE 2")
    fig.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4 * scale,
                               color=COLOURS[sp], label=DISPLAY[sp])
                        for sp in SPECIES],
               loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.03))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    return savefig(fig, out_dir, "tsne_clusters")


def _bare(ax):
    """t-SNE axes carry no units worth labelling; keep the frame, drop the numbers."""
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")


def fig_all_species(E, df, out_dir, s=1.6):
    scale = apply_style(SINGLE_COL)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    order = np.random.default_rng(0).permutation(len(E))  # no species drawn on top
    cols = np.array([COLOURS[sp] for sp in df["species"]])[order]
    ax.scatter(E[order, 0], E[order, 1], s=s * scale, c=cols, linewidths=0, alpha=0.55)
    _bare(ax)
    counts = df["species"].value_counts()
    ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4 * scale,
                              color=COLOURS[sp], label=f"{DISPLAY[sp]} ({counts[sp]})")
                       for sp in SPECIES],
              loc="best", frameon=False, handletextpad=0.3, borderpad=0.2)
    return savefig(fig, out_dir, "tsne_all_species")


def _example_images(cfg, df):
    """Return row-aligned, raw collection-plane examples for Figure 1.

    Proton latents are stored train-then-validation, whereas the image tensor is
    in its original order.  Kaon and MIP blocks retain their original order.
    Reconstructing that mapping here prevents a plausible-looking but wrong
    image from being paired with a species spotlight.
    """
    pk_images = torch.load(HIGH_RES_PK, map_location="cpu", weights_only=False,
                           mmap=True)
    mip_images = torch.load(HIGH_RES_MIP, map_location="cpu", weights_only=False,
                            mmap=True)
    images = {"p": pk_images["p"], "k": pk_images["k"],
              "m": mip_images["m"]}

    inference_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    split = np.load(inference_dir / "species_split.npz")
    proton_order = np.concatenate([split["p_train_idx"], split["p_val_idx"]])
    starts = {
        "proton": 0,
        "kaon": int((df["species"] == "proton").sum()),
        "muon": int((df["species"] != "muon").sum()),
    }
    tensor_key = {"proton": "p", "kaon": "k", "muon": "m"}

    selected = {}
    for sp in SPECIES:
        run, subrun, event = EXAMPLE_EVENTS[sp]
        match = ((df["species"] == sp) & (df["run"] == run) &
                 (df["subrun"] == subrun) & (df["event"] == event))
        rows = np.flatnonzero(match.to_numpy())
        if len(rows) != 1:
            raise ValueError(f"expected one {sp} example for {(run, subrun, event)}, "
                             f"found {len(rows)}")
        local = int(rows[0] - starts[sp])
        image_row = int(proton_order[local]) if sp == "proton" else local
        selected[sp] = images[tensor_key[sp]][image_row, 0].numpy()
    return selected


def _display_examples(examples, pad=4):
    """Orient examples and apply one shared crop to their blank wire margin.

    Every image uses the same row limits.  This keeps the spatial scale common
    across species while avoiding a strip dominated by detector background.
    """
    full = {sp: examples[sp].T for sp in SPECIES}
    active = [np.flatnonzero((image > 0).any(axis=1)) for image in full.values()]
    active = np.concatenate([rows for rows in active if len(rows)])
    lo = max(0, int(active.min()) - pad)
    hi = min(256, int(active.max()) + pad + 1)
    return {sp: image[lo:hi] for sp, image in full.items()}


def fig_species_panel(E, df, out_dir, cfg=None, s=1.6):
    """Species spotlights grounded by one real detector image per panel."""
    scale = apply_style(SINGLE_COL)
    if cfg is None:
        # Preserve the old plotting API for callers that only have an embedding.
        fig, latent_axes = plt.subplots(
            1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3 + 0.15),
            sharex=True, sharey=True)
        image_axes = None
    else:
        fig, axes = plt.subplots(
            2, 3, figsize=(DOUBLE_COL, 2.84),
            gridspec_kw={"height_ratios": [0.37, 1.0], "hspace": 0.06,
                         "wspace": 0.08})
        image_axes, latent_axes = axes
        examples = _example_images(cfg, df)
        shown = _display_examples(examples)
        nonzero = np.concatenate([im[im > 0] for im in shown.values()])
        vmax = float(np.percentile(nonzero, 99.5))
        for ax, sp in zip(image_axes, SPECIES):
            ax.imshow(shown[sp], cmap="viridis", origin="lower", vmin=0,
                      vmax=vmax, interpolation="nearest", aspect="auto",
                      rasterized=True)
            ax.set_title(f"{DISPLAY[sp]}", pad=3, fontsize=9 * scale,
                         fontweight="normal")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("0.75")
                spine.set_linewidth(0.5 * scale)

    for ax, sp in zip(latent_axes, SPECIES):
        m = (df["species"] == sp).to_numpy()
        ax.scatter(E[~m, 0], E[~m, 1], s=s * scale * 0.7, c=BACKDROP,
                   linewidths=0, alpha=0.5)
        ax.scatter(E[m, 0], E[m, 1], s=s * scale, c=COLOURS[sp],
                   linewidths=0, alpha=0.65)
        if cfg is None:
            ax.set_title(f"{DISPLAY[sp]} ({m.sum()})", pad=3)
        _bare(ax)
        ax.set_xlim(E[:, 0].min(), E[:, 0].max())
        ax.set_ylim(E[:, 1].min(), E[:, 1].max())
        ax.set_ylabel("")
    latent_axes[0].set_ylabel("t-SNE 2")
    if cfg is None:
        fig.tight_layout()
    else:
        fig.subplots_adjust(left=0.055, right=0.995, bottom=0.075, top=0.93)
    return savefig(fig, out_dir, "tsne_species_panel")


def fig_proxy(E, df, feat, out_dir, s=1.6, pct=(2, 98)):
    """Colour by a physics proxy. Percentile clip so a long tail does not flatten
    the whole colour range into one bin."""
    scale = apply_style(SINGLE_COL)
    v = df[feat].to_numpy(float)
    ok = np.isfinite(v)
    lo, hi = np.percentile(v[ok], pct)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL))
    order = np.random.default_rng(0).permutation(int(ok.sum()))
    sc = ax.scatter(E[ok][order, 0], E[ok][order, 1], s=s * scale, c=v[ok][order],
                    cmap="viridis", vmin=lo, vmax=hi, linewidths=0, alpha=0.7)
    _bare(ax)
    cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(PROXY_LABELS.get(feat, feat))
    cb.outline.set_linewidth(0.6 * scale)
    return savefig(fig, out_dir, f"tsne_proxy_{feat}")


def fig_proxy_pair(E, df, feats, out_dir, s=1.6, pct=(2, 98)):
    """Both physics proxies over the SAME embedding, side by side.

    The per-proxy figures above are each self-contained, but the question the
    paper actually asks is whether calorimetry and topology are laid out along
    the same direction of the map or along different ones. That is a comparison,
    and a comparison read across two separately-scaled figures on two pages is a
    comparison the reader has to do from memory. Here the geometry is literally
    identical between panels -- same E, same clip rule, same draw order -- so
    only the colour differs and the eye does the work.

    Separate colourbars, not a shared one: the proxies are different quantities
    in different units, and a common scale would be meaningless.
    """
    scale = apply_style(DOUBLE_COL / len(feats))
    fig, axes = plt.subplots(1, len(feats),
                             figsize=(DOUBLE_COL, DOUBLE_COL / len(feats) + 0.15),
                             sharex=True, sharey=True)
    order = np.random.default_rng(0).permutation(len(E))  # one order for both panels
    for ax, feat in zip(np.atleast_1d(axes), feats):
        v = df[feat].to_numpy(float)[order]
        ok = np.isfinite(v)
        lo, hi = np.percentile(v[ok], pct)
        sc = ax.scatter(E[order][ok, 0], E[order][ok, 1], s=s * scale, c=v[ok],
                        cmap="viridis", vmin=lo, vmax=hi, linewidths=0, alpha=0.7)
        ax.set_title(PROXY_LABELS.get(feat, feat), pad=4, fontsize=9 * scale)
        _bare(ax)
        ax.set_ylabel("")
        cb = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.03)
        cb.outline.set_linewidth(0.6 * scale)
    np.atleast_1d(axes)[0].set_ylabel("t-SNE 2")
    fig.tight_layout()
    return savefig(fig, out_dir, "tsne_proxies")


def fig_perplexity(Z, df, perps, seed, cache_dir, out_dir, recompute, s=1.2):
    scale = apply_style(DOUBLE_COL / len(perps))
    fig, axes = plt.subplots(1, len(perps),
                             figsize=(DOUBLE_COL, DOUBLE_COL / len(perps) + 0.2))
    rng = np.random.default_rng(0)
    for ax, p in zip(np.atleast_1d(axes), perps):
        E = embed(Z, p, seed, cache_dir, recompute)
        order = rng.permutation(len(E))
        ax.scatter(E[order, 0], E[order, 1], s=s * scale,
                   c=np.array([COLOURS[sp] for sp in df["species"]])[order],
                   linewidths=0, alpha=0.5)
        ax.set_title(f"perplexity {p:g}", pad=3)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    return savefig(fig, out_dir, "tsne_perplexity")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model YAML (all-species run)")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--picky", type=int, choices=[0, 1], default=None)
    ap.add_argument("--standardise", action="store_true",
                    help="z-score each latent dim before embedding (default: raw Z, "
                         "matching the UMAP figures)")
    ap.add_argument("--clusters", action="store_true",
                    help="also draw beam tag vs unsupervised GMM (fit in 8D)")
    ap.add_argument("--anchored", action="store_true",
                    help="add the anchored semi-supervised fit as a third panel")
    ap.add_argument("--sweep", action="store_true",
                    help="also draw the same data at several perplexities")
    ap.add_argument("--perps", type=float, nargs="+", default=[5, 15, 30, 50, 100])
    ap.add_argument("--recompute", action="store_true", help="ignore cached embeddings")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    if args.picky is not None:
        Z, df = select(Z, df, picky=args.picky)
    cache_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(cfg, "tsne")
    print(f"t-SNE on {len(Z)} events x {Z.shape[1]}D -> {out_dir}")

    E = embed(Z, args.perplexity, args.seed, cache_dir, args.recompute,
              args.standardise)
    fig_all_species(E, df, out_dir)
    fig_species_panel(E, df, out_dir, cfg=cfg)
    proxies = [f for f in PROXY_LABELS if f in df.columns]
    for feat in proxies:
        fig_proxy(E, df, feat, out_dir)
    if len(proxies) > 1:
        fig_proxy_pair(E, df, proxies, out_dir)
    if args.clusters or args.anchored:
        fig_clusters(E, Z, df, out_dir, seed=args.seed, with_anchored=args.anchored)
    if args.sweep:
        fig_perplexity(Z, df, args.perps, args.seed, cache_dir, out_dir, args.recompute)


if __name__ == "__main__":
    main()
