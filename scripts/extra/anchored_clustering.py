#!/usr/bin/env python3
"""
scripts/extra/anchored_clustering.py

Semi-supervised clustering of the latent space: two mixture components are
ANCHORED to the samples we trust, and the third is left free.

WHAT IS AND IS NOT SUPERVISED HERE
    Uses:     the proton and MIP beam tags, to build fixed density templates.
              Both samples are ~100% pure in this dataset.
    Does NOT use: the kaon tag, at any point. The kaon component is learned
              from the kaon-tagged sample without ever being told which of
              those events are really kaons.

    So this is supervised on the two classes we believe and unsupervised on the
    one we don't. It CANNOT support the "structure emerges without labels"
    claim -- cluster_latents.py is the script for that. Use this one when you
    want the best available per-event assignment, e.g. for measuring the beam
    composition or selecting a cleaner kaon sample.

WHY IT BEATS THE UNSUPERVISED FIT
    The species have very unequal latent spread (generalised variance ~1.7 for
    MIP against ~5.4 for proton). With no labels, EM maximises likelihood by
    letting the tight MIP component annex the low-density edge of the proton
    cloud, which is why ~13% of protons land in the muon cluster and proton
    recall stalls near 0.67. Pinning the two clean components removes exactly
    that failure mode.

THE ALGORITHM
    1. Fit q_P on the pure protons and q_M on the pure MIPs. These never move.
    2. Initialise q_K on the kaon-tagged sample.
    3. Iterate: compute each kaon-tagged event's responsibility for the free
       component, then refit q_K on the sample resampled by that responsibility
       so events already explained by q_P or q_M stop shaping it. This is what
       stops the free component from simply absorbing the contamination.
    4. Assign every event to the component with the highest weighted density.

OUTPUTS (under figs/<model_name>/anchored/)
    anchored_umap.{png,pdf}         beam labels | unsupervised GMM | anchored
    anchored_recall.{png,pdf}       per-species recall, the three methods
                                    plus the supervised ceiling
    anchored_proxy_hists.{png,pdf}  both physics proxies by anchored assignment
    metrics.json

Usage:
    python scripts/extra/anchored_clustering.py --config configs/run_0093_*.yaml
    python scripts/extra/anchored_clustering.py --config ... --picky 1
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import cross_val_predict

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PROXY_LABELS, SINGLE_COL,
                        SPECIES, apply_style, figure_dir, load_beam_data,
                        load_config, load_embedding, savefig)

TOTAL_FILL, TOTAL_EDGE = "#D9D9D9", "#BDBDBD"


def fit_density(X, n_components, seed=0):
    return GaussianMixture(n_components, covariance_type="full", n_init=6,
                           random_state=seed).fit(X)


def anchored_fit(Z, species, n_anchor_comp=6, n_free_comp=6, n_iter=6, seed=0):
    """Two components anchored to the clean samples, the third learned.

    Returns (labels, weights, densities) where labels index SPECIES order.
    """
    rng = np.random.default_rng(seed)
    is_kaon = (species == "kaon").to_numpy()

    q_proton = fit_density(Z[(species == "proton").to_numpy()], n_anchor_comp, seed)
    q_muon = fit_density(Z[(species == "muon").to_numpy()], n_anchor_comp, seed)
    q_kaon = fit_density(Z[is_kaon], n_free_comp, seed)

    Z_kaon = Z[is_kaon]
    counts = np.array([(species == s).sum() for s in SPECIES], dtype=float)
    weights = counts / counts.sum()

    for _ in range(n_iter):
        log_joint = np.column_stack([q_proton.score_samples(Z_kaon),
                                     q_kaon.score_samples(Z_kaon),
                                     q_muon.score_samples(Z_kaon)]) + np.log(weights)
        resp = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
        resp /= resp.sum(axis=1, keepdims=True)

        # Refit the free component on events that still look like kaons, so it
        # stops being shaped by the contamination the anchors already explain.
        kaon_resp = resp[:, 1]
        if kaon_resp.sum() < 50:
            break
        draw = rng.choice(len(Z_kaon), len(Z_kaon), p=kaon_resp / kaon_resp.sum())
        q_kaon = fit_density(Z_kaon[draw], n_free_comp, seed)

    log_joint = np.column_stack([q_proton.score_samples(Z),
                                 q_kaon.score_samples(Z),
                                 q_muon.score_samples(Z)]) + np.log(weights)
    return log_joint.argmax(axis=1), weights, (q_proton, q_kaon, q_muon)


def score_partition(labels, species, name):
    """ARI / NMI / purity plus per-species recall, over every event."""
    truth = pd.Categorical(species, categories=SPECIES).codes
    n_clusters = int(labels.max()) + 1
    purity = sum(np.bincount(truth[labels == c], minlength=3).max()
                 for c in range(n_clusters)) / len(truth)
    recall = {s: float(np.bincount(labels[truth == i], minlength=n_clusters).max()
                       / (truth == i).sum()) for i, s in enumerate(SPECIES)}
    return {"method": name,
            "ari": float(adjusted_rand_score(truth, labels)),
            "nmi": float(normalized_mutual_info_score(truth, labels)),
            "purity": float(purity), **{f"recall_{s}": recall[s] for s in SPECIES}}


def plot_umap_comparison(embedding, df, out_dir):
    """Three panels on the same projection: truth, unsupervised, anchored."""
    s = apply_style(SINGLE_COL)  # each PANEL is single-column wide
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3 * 1.02))
    panels = [("species", "(a) Beam labels (truth)"),
              ("unsupervised_name", "(b) Unsupervised GMM"),
              ("anchored_name", "(c) Anchored (semi-supervised)")]
    for ax, (column, title) in zip(axes, panels):
        for name in ["proton", "muon", "kaon"]:
            sel = (df[column] == name).to_numpy()
            ax.scatter(embedding[sel, 0], embedding[sel, 1], s=0.5, c=COLOURS[name],
                       alpha=0.30, lw=0, label=DISPLAY[name], rasterized=True)
        legend = ax.legend(markerscale=10, handletextpad=0.2, borderpad=0.3,
                           loc="lower left", fontsize=6.2 * s)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        ax.set_title(title, loc="left", fontsize=8.5 * s, pad=3 * s)
        ax.set_xlabel("UMAP 1")
        ax.set_xticks([]); ax.set_yticks([])
    axes[0].set_ylabel("UMAP 2")
    fig.tight_layout()
    savefig(fig, out_dir, "anchored_umap")


def plot_recall(rows, out_dir):
    """Per-species recall for each method, with the supervised ceiling behind."""
    s = apply_style(SINGLE_COL)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.55, DOUBLE_COL * 0.55 / 1.5))
    methods = [r["method"] for r in rows]
    width = 0.8 / len(methods)
    shades = ["0.75", COLOURS["kaon"], "0.35"]
    positions = np.arange(len(SPECIES))
    for i, (r, shade) in enumerate(zip(rows, shades)):
        offset = (i - (len(methods) - 1) / 2) * width
        values = [r[f"recall_{sp}"] for sp in SPECIES]
        bars = ax.bar(positions + offset, values, width=width * 0.92,
                      color=shade, label=r["method"])
        for bar, value in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.012, f"{value:.2f}",
                    ha="center", fontsize=5.6 * s)
    ax.set_xticks(positions)
    ax.set_xticklabels([DISPLAY[sp] for sp in SPECIES])
    ax.set_ylabel("Recall (fraction in its own cluster)")
    ax.set_ylim(0, 1.14)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
              fontsize=6.2 * s, columnspacing=1.0, handlelength=1.2)
    fig.tight_layout()
    savefig(fig, out_dir, "anchored_recall")


def plot_proxy_hists(df, out_dir, bins=50):
    """Both proxies decomposed by the anchored assignment."""
    s = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2 / 1.25))
    for ax, feature in zip(axes, PROXY_LABELS):
        vals = df[feature].to_numpy()
        edges = np.linspace(np.nanmin(vals), np.nanmax(vals), bins + 1)
        counts = {}
        for name in SPECIES:
            v = df.loc[df["anchored_name"] == name, feature].to_numpy()
            v = v[np.isfinite(v)]
            if len(v):
                counts[name] = np.histogram(v, bins=edges)[0]
        total = np.sum(list(counts.values()), axis=0)
        ax.stairs(total, edges, fill=True, color=TOTAL_FILL, edgecolor=TOTAL_EDGE,
                  linewidth=0.5 * s, zorder=1, label="All species")
        for name, c in counts.items():
            ax.stairs(c, edges, color=COLOURS[name], linewidth=0.9 * s, zorder=3,
                      label=DISPLAY[name])
        ax.set_xlabel(PROXY_LABELS[feature]); ax.set_ylabel("Counts")
        cum = np.cumsum(total) / total.sum()
        upper = edges[1:][cum >= 0.999]
        ax.set_xlim(edges[0], upper[0] if len(upper) else edges[-1])
        ax.set_ylim(0, total.max() * (1 + 0.35 / s))
        ax.legend(loc="upper left" if feature == "solidity" else "upper right")
    fig.tight_layout()
    savefig(fig, out_dir, "anchored_proxy_hists")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--features-pkl", default=None)
    ap.add_argument("--picky-csv", default=None)
    ap.add_argument("--picky", type=int, choices=[0, 1], default=None,
                    help="restrict the KAON sample to picky (1) or non-picky (0)")
    ap.add_argument("--n-iter", type=int, default=6,
                    help="anchored EM outer iterations (refits of the free component)")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kwargs = {k: v for k, v in [("features_path", args.features_pkl),
                                ("picky_path", args.picky_csv)] if v}
    Z, df = load_beam_data(cfg, **kwargs)
    embedding = None if args.no_umap else load_embedding(cfg, Z)

    if args.picky is not None:
        # Filter the kaon sample only -- protons and MIPs are 100% p=1, so
        # filtering them on p=0 would empty the anchors.
        keep = ((df["species"] != "kaon") | (df["picky"] == args.picky)).to_numpy()
        Z, df = Z[keep], df.loc[keep].reset_index(drop=True)
        embedding = embedding[keep] if embedding is not None else None
        print(f"  kaon sample restricted to picky={args.picky}: "
              f"{int((df['species'] == 'kaon').sum())} kaon events")

    out_dir = args.out_dir or figure_dir(cfg, "anchored")

    anchored, weights, _ = anchored_fit(Z, df["species"], n_iter=args.n_iter, seed=args.seed)
    unsupervised = GaussianMixture(3, covariance_type="full", n_init=20,
                                   random_state=args.seed).fit_predict(Z)
    truth = pd.Categorical(df["species"], categories=SPECIES).codes
    ceiling = cross_val_predict(QuadraticDiscriminantAnalysis(), Z, truth, cv=3)

    # Name the unsupervised clusters by majority tag so the UMAP colours line up.
    unsup_names = {}
    for c in range(3):
        vc = df.loc[unsupervised == c, "species"].value_counts()
        unsup_names[c] = vc.index[0] if len(vc) else SPECIES[c]
    df["unsupervised_name"] = pd.Series(unsupervised).map(unsup_names)
    df["anchored_name"] = pd.Series(anchored).map(dict(enumerate(SPECIES)))

    rows = [score_partition(unsupervised, df["species"], "unsupervised GMM"),
            score_partition(anchored, df["species"], "anchored (semi-sup.)"),
            score_partition(ceiling, df["species"], "supervised QDA [ceiling]")]
    table = pd.DataFrame(rows)
    print("\nEvery event scored; recall is the fraction of a species landing in its own cluster.")
    print(table.round(3).to_string(index=False))

    confusion = pd.crosstab(df["anchored_name"], df["species"]).reindex(
        index=SPECIES, columns=SPECIES, fill_value=0)
    print("\nanchored assignment vs beam tag (rows = assigned):")
    print(confusion.to_string())
    print("\ncolumn-normalised (per-species recall):")
    print(confusion.div(confusion.sum(axis=0), axis=1).round(3).to_string())
    print(f"\nfitted component weights (proton, kaon, MIP): {weights.round(3)}")
    print("\nNOTE: the kaon column is NOT a clean benchmark -- that sample is a mixture. "
          "Read the proton and MIP columns for method quality, and use "
          "estimate_contamination.py for the kaon composition.")
    print("NOTE: kaon-tagged events landing on the MIP component are NOT a light-particle "
          "contamination estimate. The MIP template is built from through-going tracks "
          "(176-226 wires) while the kaon sample is 11-178, so a stopping pion cannot "
          "resemble it; and those events sit ABOVE the kaon peak in spectrometer mass, "
          "the wrong sign for pion leakage. They look like decay-in-flight kaons whose "
          "imaged last 50 wires are the daughter muon.")

    plot_recall(rows, out_dir)
    plot_proxy_hists(df, out_dir)
    if embedding is not None:
        plot_umap_comparison(embedding, df, out_dir)

    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump({"picky": args.picky, "n_iter": args.n_iter,
                   "component_weights": weights.tolist(),
                   "comparison": rows, "confusion": confusion.to_dict()}, fh, indent=2)
    print(f"  saved {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
