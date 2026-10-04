#!/usr/bin/env python3
"""Sweep full-covariance GMM clustering over eight existing physics features."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, parallel_backend
from sklearn.impute import SimpleImputer
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from _beam_data import (SPECIES, figure_dir, load_beam_data, load_config)


FEATURES = [
    "n_local_maxima",
    "median_adc",
    "monotonic_rise_fraction",
    "bragg_rise_slope",
    "last_quartile_mean",
    "std_adc",
    "n_pixels",
    "max_ADC_position",
]


def fit_one(X, truth, k, seed, n_init):
    labels = GaussianMixture(
        k, covariance_type="full", n_init=n_init, random_state=seed
    ).fit_predict(X)
    counts = [np.bincount(truth[labels == c], minlength=len(SPECIES)) for c in range(k)]
    sizes = np.bincount(labels, minlength=k)
    cluster_purity = np.array([count.max() / count.sum() for count in counts])
    return {
        "k": k,
        "purity": sum(count.max() for count in counts) / len(truth),
        "ari": adjusted_rand_score(truth, labels),
        "nmi": normalized_mutual_info_score(truth, labels),
        "smallest_cluster": int(sizes.min()),
        "n_below_100": int((sizes < 100).sum()),
        "frac_in_85pct_clusters": float(sizes[cluster_purity >= 0.85].sum() / len(truth)),
    }


def plot(results, out_dir):
    fig, axes = plt.subplots(2, 1, figsize=(6.875, 5.2), sharex=True)
    axes[0].plot(results["k"], results["purity"], marker="o", markersize=3)
    axes[0].set_ylabel("Majority-vote purity")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].grid(alpha=0.25)
    axes[1].plot(results["k"], results["smallest_cluster"], marker="o", markersize=3)
    axes[1].axhline(100, color="0.5", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Smallest cluster (events)")
    axes[1].set_xlabel("GMM components, k")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(out_dir / f"feature_purity_k3_50.{suffix}", dpi=300)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-init", type=int, default=20)
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--out-dir")
    args = ap.parse_args()

    cfg = load_config(args.config)
    _, df = load_beam_data(cfg)
    truth = pd.Categorical(df["species"], categories=SPECIES).codes
    X = SimpleImputer(strategy="median").fit_transform(df[FEATURES])
    X = StandardScaler().fit_transform(X)
    ks = range(args.kmin, args.kmax + 1)
    with parallel_backend("loky", inner_max_num_threads=1):
        rows = Parallel(n_jobs=args.jobs, verbose=10)(
            delayed(fit_one)(X, truth, k, args.seed, args.n_init) for k in ks
        )
    results = pd.DataFrame(rows).sort_values("k")
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(cfg, "feature-kscan")
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "feature_kscan.csv", index=False)
    plot(results, out_dir)
    print(results.to_string(index=False))
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
