#!/usr/bin/env python3
"""
scripts/extra/cluster_latents.py

Unsupervised clustering of the 8D VAE latent space, with no labels used at any
point in the fit. The beam tags are read back afterwards only to score and
describe the clusters.

WHY A FULL-COVARIANCE GAUSSIAN MIXTURE
    The three species have very unequal latent spread -- generalised variance
    (det Sigma)^(1/D) is roughly 10.8 (kaon), 5.4 (proton), 1.7 (MIP). k-means
    and diagonal/tied mixtures assume comparable, axis-aligned scatter and so
    carve the diffuse kaon cloud apart while gluing the tight MIP core to its
    neighbours. Full covariance is not a refinement here, it is the whole gap
    (ARI ~0.37 vs ~0.17 for k-means on this model).

CHOOSING k
    k is SPECIFIED (default 3, from the physics: three beam species), not
    discovered. BIC falls monotonically well past k=12 and stability selection
    is flat across k=2,3,4, so no internal criterion picks 3 on its own. Pass
    --scan-k to reproduce that and report it honestly rather than implying the
    data chose k.

OUTPUTS (under figs/<model_name>/clustering/)
    proxy_hists_beam_vs_cluster.{png,pdf}   2x2: beam labels vs cluster labels,
                                            for both physics proxies
    <proxy>_counts_cluster.{png,pdf}        single-column drop-in versions
    cluster_composition.{png,pdf}           stacked composition per cluster
    metrics.json                            ARI/NMI/purity, confusion matrix,
                                            per-cluster proxy medians, k scan

Usage:
    python scripts/extra/cluster_latents.py --config configs/run_0093_*.yaml
    python scripts/extra/cluster_latents.py --config ... --picky 1
    python scripts/extra/cluster_latents.py --config ... --scan-k 2 8
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PROXY_LABELS, SINGLE_COL,
                        SPECIES, apply_style, figure_dir, load_beam_data,
                        load_config, load_embedding, savefig, select)

TOTAL_FILL, TOTAL_EDGE = "#D9D9D9", "#BDBDBD"


def fit_clusters(Z, k, seed=0, n_init=20):
    """Fit a k-component full-covariance Gaussian mixture. No labels involved."""
    gmm = GaussianMixture(k, covariance_type="full", n_init=n_init, random_state=seed)
    return gmm.fit_predict(Z), gmm


def name_clusters(labels, species, k):
    """Map each cluster to its majority beam tag, so colours line up with the
    beam-label plots. Purely cosmetic -- it does not affect any metric."""
    names = {}
    for c in range(k):
        counts = species[labels == c].value_counts()
        names[c] = counts.index[0] if len(counts) else SPECIES[c % 3]
    return names


def score_clusters(labels, species, k):
    """Agreement between the unsupervised partition and the beam tags.

    Purity is majority-vote: each cluster is credited with its most common tag.
    With k = n_species this is comparable to accuracy; for k > n_species it is
    an upper bound and should be read as such.
    """
    truth = pd.Categorical(species, categories=SPECIES).codes
    correct = sum(np.bincount(truth[labels == c], minlength=3).max() for c in range(k))
    return {
        "ari": float(adjusted_rand_score(truth, labels)),
        "nmi": float(normalized_mutual_info_score(truth, labels)),
        "purity": float(correct / len(truth)),
    }


def scan_k(Z, species, k_lo, k_hi, seed=0):
    """BIC and agreement across k, to show that no internal criterion picks 3."""
    rows = []
    for k in range(k_lo, k_hi + 1):
        gmm = GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed).fit(Z)
        labels = gmm.predict(Z)
        rows.append({"k": k, "bic": float(gmm.bic(Z)), "aic": float(gmm.aic(Z)),
                     **score_clusters(labels, species, k)})
    return pd.DataFrame(rows)


def benchmark(Z, embedding, species, seed=0):
    """Compare the chosen GMM against the obvious alternatives, scored fairly.

    THE SCORING TRAP THIS AVOIDS
        Density methods leave points unassigned. If you score only the assigned
        subset, they look excellent purely by declining the hard cases --
        HDBSCAN on the UMAP reports purity 0.92 that way, above the SUPERVISED
        ceiling, which should be an immediate tell. Here every event is scored:
        unassigned points get singleton labels, so abstaining costs you.

    The supervised rows are ceilings, not competitors: QDA is exactly this
    Gaussian mixture handed the labels, so the GMM-to-QDA gap is the price of
    not having them, and no unsupervised method can be expected to close it.
    """
    from sklearn.cluster import HDBSCAN, KMeans
    from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler

    truth = pd.Categorical(species, categories=SPECIES).codes

    def row(name, labels, unassigned=0.0):
        labels = np.asarray(labels).copy()
        noise = labels < 0
        if noise.any():  # singleton labels: abstaining is penalised, not rewarded
            labels[noise] = labels.max() + 1 + np.arange(noise.sum())
        purity = sum(np.bincount(truth[labels == c], minlength=3).max()
                     for c in np.unique(labels)) / len(truth)
        recall = [np.bincount(labels[truth == s], minlength=labels.max() + 1).max()
                  / (truth == s).sum() for s in range(3)]
        return {"method": name, "ari": adjusted_rand_score(truth, labels),
                "nmi": normalized_mutual_info_score(truth, labels), "purity": purity,
                "recall_proton": recall[0], "recall_kaon": recall[1],
                "recall_mip": recall[2], "unassigned": unassigned}

    rows = [row("GMM-full k=3, raw 8D  [chosen]",
                GaussianMixture(3, covariance_type="full", n_init=20,
                                random_state=seed).fit_predict(Z)),
            row("k-means k=3, raw 8D",
                KMeans(3, n_init=20, random_state=seed).fit_predict(Z)),
            row("GMM-diag k=3, raw 8D",
                GaussianMixture(3, covariance_type="diag", n_init=20,
                                random_state=seed).fit_predict(Z)),
            row("GMM-full k=3, z-scored",
                GaussianMixture(3, covariance_type="full", n_init=20, random_state=seed)
                .fit_predict(StandardScaler().fit_transform(Z)))]

    if embedding is not None:
        for mcs in (500, 1000):
            labels = HDBSCAN(min_cluster_size=mcs).fit(embedding).labels_
            frac = float((labels < 0).mean())
            rows.append(row(f"HDBSCAN on UMAP (mcs={mcs})", labels, frac))
            assigned = labels >= 0
            filled = labels.copy()
            filled[~assigned] = (KNeighborsClassifier(15)
                                 .fit(embedding[assigned], labels[assigned])
                                 .predict(embedding[~assigned]))
            rows.append(row(f"HDBSCAN on UMAP (mcs={mcs}) + kNN fill", filled))

    rows.append(row("[ceiling] QDA, supervised",
                    cross_val_predict(QuadraticDiscriminantAnalysis(), Z, truth, cv=3)))
    return pd.DataFrame(rows)


def _draw_proxy_hist(ax, df, feature, group_col, edges, scale):
    """Grey total behind, one solid step outline per group.

    The grey total is identical between the beam-label and cluster-label
    versions -- same events, only the decomposition changes -- which is what
    makes the two panels directly comparable.
    """
    counts = {}
    for name in SPECIES:
        vals = df.loc[df[group_col] == name, feature].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            counts[name] = np.histogram(vals, bins=edges)[0]
    total = np.sum(list(counts.values()), axis=0)

    ax.stairs(total, edges, fill=True, color=TOTAL_FILL, edgecolor=TOTAL_EDGE,
              linewidth=0.5 * scale, zorder=1, label="All species")
    for name, c in counts.items():
        ax.stairs(c, edges, color=COLOURS[name], linewidth=0.9 * scale, zorder=3,
                  label=DISPLAY[name])

    ax.set_xlabel(PROXY_LABELS.get(feature, feature))
    ax.set_ylabel("Counts")
    # View-only trim: bins still span the full range, we just hide the empty tail.
    cum = np.cumsum(total) / total.sum()
    upper = edges[1:][cum >= 0.999]
    ax.set_xlim(edges[0], upper[0] if len(upper) else edges[-1])
    ax.set_ylim(0, total.max() * (1 + 0.35 / scale))
    ax.legend(loc="upper left" if feature == "solidity" else "upper right")


def plot_proxy_comparison(df, out_dir, bins=50):
    """2x2 grid: rows are {beam label, cluster label}, columns are the proxies."""
    s = apply_style(SINGLE_COL)  # each PANEL is single-column wide
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.82))
    for col, feature in enumerate(PROXY_LABELS):
        vals = df[feature].to_numpy()
        edges = np.linspace(np.nanmin(vals), np.nanmax(vals), bins + 1)
        _draw_proxy_hist(axes[0, col], df, feature, "species", edges, s)
        _draw_proxy_hist(axes[1, col], df, feature, "cluster_name", edges, s)
        top = max(axes[0, col].get_ylim()[1], axes[1, col].get_ylim()[1])
        for row in (0, 1):
            axes[row, col].set_ylim(0, top)
            axes[row, col].set_xlim(axes[1, col].get_xlim())
    axes[0, 0].set_title("Beam labels (truth)", fontsize=9 * s, pad=5 * s, loc="left")
    axes[1, 0].set_title("Unsupervised GMM clusters", fontsize=9 * s, pad=5 * s, loc="left")
    fig.tight_layout()
    savefig(fig, out_dir, "proxy_hists_beam_vs_cluster")

    for feature in PROXY_LABELS:  # single-column drop-in replacements
        s = apply_style(SINGLE_COL)
        fig, ax = plt.subplots(figsize=(SINGLE_COL, round(SINGLE_COL / 1.25, 3)))
        vals = df[feature].to_numpy()
        edges = np.linspace(np.nanmin(vals), np.nanmax(vals), bins + 1)
        _draw_proxy_hist(ax, df, feature, "cluster_name", edges, s)
        fig.tight_layout()
        savefig(fig, out_dir, f"{feature}_counts_cluster")


def plot_cluster_umap(embedding, df, k, out_dir):
    """Side-by-side UMAP: beam labels vs the unsupervised partition.

    The projection is the shared cached reducer, so the point positions are
    identical in both panels -- only the colouring changes. Anywhere the two
    panels disagree is a place the latent geometry and the beam tag part ways.
    """
    s = apply_style(SINGLE_COL)  # each PANEL is single-column wide
    fig, (ax_beam, ax_cluster) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2))

    for name in ["proton", "muon", "kaon"]:
        sel = (df["species"] == name).to_numpy()
        ax_beam.scatter(embedding[sel, 0], embedding[sel, 1], s=0.5, c=COLOURS[name],
                        alpha=0.30, lw=0, label=DISPLAY[name], rasterized=True)

    cluster_colours = plt.get_cmap("tab10")
    for c in range(k):
        sel = (df["cluster"] == c).to_numpy()
        ax_cluster.scatter(embedding[sel, 0], embedding[sel, 1], s=0.5,
                           c=[cluster_colours(c)], alpha=0.30, lw=0,
                           label=f"cluster {c} ({DISPLAY[df.loc[sel, 'cluster_name'].iloc[0]]}-rich)",
                           rasterized=True)

    for ax, title in [(ax_beam, "Beam labels (truth)"),
                      (ax_cluster, f"Unsupervised GMM (k={k})")]:
        legend = ax.legend(markerscale=10, handletextpad=0.2, borderpad=0.3,
                           loc="lower left", fontsize=6.4 * s)
        for handle in legend.legend_handles:
            handle.set_alpha(1)
        ax.set_title(title, loc="left", fontsize=9 * s, pad=3 * s)
        ax.set_xlabel("UMAP 1")
        ax.set_xticks([]); ax.set_yticks([])
    ax_beam.set_ylabel("UMAP 2")
    fig.tight_layout()
    savefig(fig, out_dir, "cluster_umap")


def plot_composition(df, k, out_dir):
    """Stacked bar of beam-tag composition per cluster."""
    s = apply_style(SINGLE_COL)
    table = pd.crosstab(df["cluster"], df["species"]).reindex(columns=SPECIES, fill_value=0)
    frac = table.div(table.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(SINGLE_COL, round(SINGLE_COL / 1.3, 3)))
    bottom = np.zeros(len(frac))
    for name in SPECIES:
        ax.bar(frac.index, frac[name], bottom=bottom, color=COLOURS[name],
               label=DISPLAY[name], width=0.72, edgecolor="white", linewidth=0.5 * s)
        bottom += frac[name].to_numpy()
    ax.set_xlabel("Cluster"); ax.set_ylabel("Fraction of cluster")
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"{c}\n(n={table.loc[c].sum()})" for c in frac.index])
    ax.set_ylim(0, 1)
    # Stacked bars fill the axes, so the legend goes above rather than inside.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), ncol=3,
              fontsize=7 * s, frameon=False, columnspacing=1.2, handlelength=1.4)
    fig.tight_layout()
    savefig(fig, out_dir, "cluster_composition")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model YAML (must be an all-species run)")
    ap.add_argument("--features-pkl", default=None)
    ap.add_argument("--picky-csv", default=None)
    ap.add_argument("--k", type=int, default=3, help="number of clusters (specified, not learned)")
    ap.add_argument("--picky", type=int, choices=[0, 1], default=None,
                    help="restrict to picky (1) or non-picky (0) events; default keeps both")
    ap.add_argument("--scan-k", type=int, nargs=2, metavar=("LO", "HI"),
                    help="also report BIC and agreement across this range of k")
    ap.add_argument("--benchmark", action="store_true",
                    help="compare the chosen GMM against alternatives, scored over ALL events")
    ap.add_argument("--no-umap", action="store_true",
                    help="skip the UMAP panel (avoids loading/fitting a reducer)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kwargs = {k: v for k, v in [("features_path", args.features_pkl),
                                ("picky_path", args.picky_csv)] if v}
    Z, df = load_beam_data(cfg, **kwargs)
    embedding = None if args.no_umap else load_embedding(cfg, Z)
    if args.picky is not None:
        keep = (df["picky"] == args.picky).to_numpy()
        embedding = embedding[keep] if embedding is not None else None
        Z, df = select(Z, df, picky=args.picky)
        print(f"  restricted to picky={args.picky}: {len(Z)} events")

    out_dir = args.out_dir or figure_dir(cfg, "clustering")

    labels, gmm = fit_clusters(Z, args.k, seed=args.seed)
    df["cluster"] = labels
    df["cluster_name"] = pd.Series(labels).map(name_clusters(labels, df["species"], args.k))

    metrics = score_clusters(labels, df["species"], args.k)
    print(f"\nGMM(k={args.k}, full covariance) on the raw {Z.shape[1]}D latent")
    print(f"  ARI {metrics['ari']:.3f}   NMI {metrics['nmi']:.3f}   "
          f"majority purity {metrics['purity']:.3f}")

    confusion = pd.crosstab(df["cluster"], df["species"]).reindex(columns=SPECIES, fill_value=0)
    print("\ncluster composition (row-normalised):")
    print(confusion.div(confusion.sum(axis=1), axis=0).round(3).to_string())
    print("\nper-species recall (column-normalised):")
    print(confusion.div(confusion.sum(axis=0), axis=1).round(3).to_string())

    proxy_cols = list(PROXY_LABELS) + ["median_adc", "n_local_maxima", "max_ADC_position",
                                       "height", "recon_error"]
    print("\nper-cluster medians:")
    print(df.groupby("cluster")[proxy_cols].median().round(3).to_string())
    print("\nper-species medians (reference):")
    print(df.groupby("species")[proxy_cols].median().round(3).to_string())

    plot_proxy_comparison(df, out_dir)
    plot_composition(df, args.k, out_dir)
    if embedding is not None:
        plot_cluster_umap(embedding, df, args.k, out_dir)

    payload = {
        "k": args.k, "picky": args.picky, "n_events": int(len(Z)),
        **metrics,
        "weights": gmm.weights_.round(4).tolist(),
        "confusion": confusion.to_dict(),
        "cluster_medians": df.groupby("cluster")[proxy_cols].median().round(4).to_dict(),
        "species_medians": df.groupby("species")[proxy_cols].median().round(4).to_dict(),
    }
    if args.benchmark:
        table = benchmark(Z, embedding, df["species"], seed=args.seed)
        print("\nMETHOD COMPARISON (every event scored; unassigned points are penalised)")
        print(table.round(3).to_string(index=False))
        payload["benchmark"] = table.to_dict(orient="records")

    if args.scan_k:
        scan = scan_k(Z, df["species"], *args.scan_k, seed=args.seed)
        print(f"\nk scan -- note BIC keeps falling, so it does NOT select k={args.k}:")
        print(scan.round(3).to_string(index=False))
        payload["k_scan"] = scan.to_dict(orient="records")

    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump(payload, fh, indent=2)
    print(f"  saved {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
