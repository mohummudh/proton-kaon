#!/usr/bin/env python3
"""
scripts/extra/plot_cluster_mass.py

Spectrometer mass of KAON-WINDOW events, cluster by cluster.

THE ARGUMENT THIS FIGURE MAKES
    Every event here comes from the same beamline mass window. What differs is
    only where the VAE put it. If the latent space is separating genuine kaons
    from contamination, then kaon-window events sitting in PROTON-RICH clusters
    should carry heavier spectrometer mass than those in kaon-rich ones.

    Mass is measured by the beamline spectrometer and is never seen by the VAE,
    and no label enters the mixture fit. So the ordering, if present, is an
    withheld check on an unsupervised result. It is not independent ground truth,
    because this same mass quantity defines the candidate window.

WHAT IT CANNOT SHOW
    The kaon category is ITSELF a spectrometer mass selection, so every event
    plotted already sits inside the kaon mass window. A positive trend means
    "these sit at the heavy edge of the window", NOT "these are protons". The
    proton PDG mass is drawn only for scale; nothing here should reach it.

OUTPUTS (under figs/<model_name>/clustering/)
    cluster_mass_k<k>.{png,pdf}

Usage:
    python scripts/extra/plot_cluster_mass.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_cluster_mass.py --config ... --k 20
"""

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from sklearn.mixture import GaussianMixture

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PDG_MASS, SINGLE_COL,
                        SPECIES, apply_style, figure_dir, load_beam_data,
                        load_config, savefig)

TOTAL_FILL, TOTAL_EDGE = "#D9D9D9", "#BDBDBD"
# The anchored fit's split, for reference: it reaches these medians using the
# proton and MIP beamline windows. The unsupervised spread should be compared to it.
ANCHORED_HEAVY, ANCHORED_LIGHT = 562.8, 459.4


def boot_median_ci(x, n=2000, seed=0):
    """Percentile bootstrap CI on a median -- cluster sizes here span 80 to 2500,
    so a fixed marker would misrepresent how well each median is determined."""
    rng = np.random.default_rng(seed)
    if len(x) < 5:
        return np.nan, np.nan
    m = np.median(rng.choice(x, (n, len(x))), axis=1)
    return np.percentile(m, [16, 84])


def collect(Z, df, k, seed=0, min_kaon=50):
    truth = np.array([SPECIES.index(s) for s in df["species"]])
    mass = df["beamline_mass"].to_numpy(float)
    kt = (truth == 1) & np.isfinite(mass)
    lab = GaussianMixture(k, covariance_type="full", n_init=20,
                          random_state=seed).fit_predict(Z)
    rows = []
    for c in range(k):
        m = lab == c
        sel = m & kt
        if sel.sum() < min_kaon:
            continue
        cnt = np.bincount(truth[m], minlength=3)
        v = mass[sel]
        lo, hi = boot_median_ci(v)
        rows.append({"cluster": c, "n": int(m.sum()), "n_kaon": int(sel.sum()),
                     "proton_frac": cnt[0] / m.sum(),
                     "majority": SPECIES[int(cnt.argmax())],
                     "median_mass": float(np.median(v)), "lo": lo, "hi": hi})
    return lab, kt, truth, mass, rows


def plot(rows, lab, kt, truth, mass, k, out_dir, panel="both", show_title=True):
    scale = apply_style(SINGLE_COL)
    if panel == "both":
        fig, (ax, bx) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.46))
    else:
        # a standalone panel is single-column: the pair is only wide because two
        # of them have to sit side by side
        fig, one = plt.subplots(figsize=(SINGLE_COL, SINGLE_COL * 0.82))
        ax = bx = one

    # ---- (a) per-cluster median mass against how proton-like the cluster is
    pf = np.array([r["proton_frac"] for r in rows])
    mm = np.array([r["median_mass"] for r in rows])
    nk = np.array([r["n_kaon"] for r in rows])
    rho, pv = spearmanr(pf, mm)
    if panel in ("both", "a"):
      for r in rows:
        ax.plot([r["proton_frac"]] * 2, [r["lo"], r["hi"]], color="0.6",
                lw=0.7 * scale, zorder=2)
      ax.scatter(pf, mm, s=12 + 90 * nk / nk.max(),
               c=[COLOURS[r["majority"]] for r in rows],
               edgecolors="0.25", linewidths=0.5 * scale, zorder=3)
      ax.axhline(PDG_MASS["kaon"], color="0.35", ls="--", lw=0.7 * scale, zorder=1)
      ax.text(0.99, PDG_MASS["kaon"] + 3, "PDG $K^+$", ha="right", va="bottom",
            fontsize=6.5 * scale, color="0.35", transform=ax.get_yaxis_transform())
      for y, nm, xa, ha in ((ANCHORED_HEAVY, "anchored $\\rightarrow$ proton", 0.01, "left"),
                          (ANCHORED_LIGHT, "anchored $\\rightarrow$ kaon", 0.99, "right")):
        ax.axhline(y, color="#AA3377", ls=":", lw=0.7 * scale, zorder=1)
        ax.text(xa, y + 3, nm, ha=ha, va="bottom", fontsize=6.5 * scale,
                color="#AA3377", transform=ax.get_yaxis_transform())
      ax.set_xlabel("Proton-tag fraction of cluster")
      ax.set_ylabel("Median beamline mass of\nkaon-selected events [MeV]")
      if show_title:
        ax.set_title(f"(a) $k$ = {k}, one point per cluster", loc="left",
                     fontsize=8.5 * scale, pad=3)
      ax.legend(handles=[Line2D([], [], marker="o", ls="none", ms=4 * scale,
                                color=COLOURS[s], label=f"{DISPLAY[s]}-majority")
                         for s in SPECIES],
                loc="upper left", frameon=False, fontsize=6.5 * scale,
                handletextpad=0.3, borderpad=0.2)

    if panel == "a":
        fig.tight_layout()
        print(f"  rho={rho:.3f} (p={pv:.2e})")
        return savefig(fig, out_dir, f"cluster_mass_k{k}_a")

    # ---- (b) the same events as distributions, split by cluster majority
    inP = np.isin(lab, [c for c in np.unique(lab)
                        if np.bincount(truth[lab == c], minlength=3).argmax() == 0])
    inK = np.isin(lab, [c for c in np.unique(lab)
                        if np.bincount(truth[lab == c], minlength=3).argmax() == 1])
    a, b = mass[kt & inP], mass[kt & inK]
    lo_w, hi_w = mass[kt].min(), mass[kt].max()
    edges = np.linspace(lo_w, hi_w, 45)
    bx.stairs(np.histogram(mass[kt], bins=edges, density=True)[0], edges, fill=True,
              color=TOTAL_FILL, edgecolor=TOTAL_EDGE, lw=0.5 * scale, zorder=1,
              label=f"all kaon-selected ({int(kt.sum())})")
    for v, col, lbl in ((a, COLOURS["proton"], f"in proton-tag-majority ({len(a)})"),
                        (b, COLOURS["kaon"], f"in kaon-tag-majority ({len(b)})")):
        bx.stairs(np.histogram(v, bins=edges, density=True)[0], edges, color=col,
                  lw=1.0 * scale, zorder=3, label=lbl)
        bx.axvline(np.median(v), color=col, ls="--", lw=0.7 * scale, zorder=2)
    bx.set_xlabel("Beamline mass [MeV]")
    bx.set_ylabel("Density")
    if show_title:
        bx.set_title(("(b) same events, no label used to split them" if panel == "both"
                      else f"$k$ = {k}: kaon-selected events, split by cluster"),
                     loc="left", fontsize=8.5 * scale, pad=3)
    bx.legend(loc="upper left", frameon=False, fontsize=6.2 * scale,
              handletextpad=0.4, borderpad=0.2)
    bx.set_ylim(0, bx.get_ylim()[1] * 1.28)
    bx.text(0.03, 0.72, f"$\\Delta$median = {np.median(a)-np.median(b):+.1f} MeV",
            transform=bx.transAxes, ha="left", va="top", fontsize=7 * scale)
    fig.tight_layout()
    print(f"  rho={rho:.3f} (p={pv:.2e}), split={np.median(a)-np.median(b):+.1f} MeV")
    stem = f"cluster_mass_k{k}" + ("" if panel == "both" else "_b")
    return savefig(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-kaon", type=int, default=50,
                    help="skip clusters with fewer kaon-window events than this; "
                         "their median mass is not meaningfully determined")
    ap.add_argument("--panel", choices=["both", "a", "b"], default="both",
                    help="draw both panels, or just one as a standalone "
                         "single-column figure")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--no-title", action="store_true",
                    help="omit the axes title for placement in a paper panel")
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    out_dir = args.out_dir or figure_dir(cfg, "clustering")
    lab, kt, truth, mass, rows = collect(Z, df, args.k, args.seed, args.min_kaon)
    print(f"k={args.k}: {len(rows)} clusters with >={args.min_kaon} kaon-window events")
    plot(rows, lab, kt, truth, mass, args.k, out_dir, panel=args.panel,
         show_title=not args.no_title)


if __name__ == "__main__":
    main()
