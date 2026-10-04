#!/usr/bin/env python3
"""
scripts/extra/plot_proxy_auc.py

How well a LINEAR readout of the latent space recovers each physics proxy, per
species, with bootstrap confidence intervals. The paper's central claim stated as
one number per (proxy, species).

REPLACES scripts/extra/plot_feature_auc.py, WHICH HARDCODED ITS NUMBERS
    That script carried the AUCs as literals in a DATA block at the top, and they
    had drifted from the model: it drew calorimetry 0.944 / 0.795 / 0.792 and
    topology 0.763 / 0.826 / 0.752 against the current model's actual 0.953 / 0.857
    / 0.823 and 0.804 / 0.908 / 0.788 — every value wrong, kaon topology by 0.082.
    Numbers in a figure should come from the artifact they describe, so this
    computes them from the latents on every run and there is nothing to go stale.

WHAT IS MEASURED
    For each species and each proxy, fit a cross-validated logistic regression on
    the latents to predict whether an event sits above or below THAT SPECIES'
    median of the proxy. AUC 0.5 means the latent space carries nothing about the
    proxy for that species; higher means the proxy is linearly encoded in the latent
    geometry.

    The median is taken within the species rather than globally, so the label is
    balanced by construction and a species whose values sit mostly on one side of
    the global median cannot produce a degenerate problem. It also means the number
    answers "does the latent space resolve this proxy WITHIN this species", which is
    the harder and more interesting question than separating species.

    The probe is imported from analyse_latents.make_feature_auc_probe, the same one
    the feature_auc analysis and both training sweeps use, so these numbers are
    directly comparable to those.

    Validation events only, for all-species models — the claim is about
    generalisation, not about events the encoder was fitted on.

ERROR BARS
    Percentile bootstrap over EVENTS, resampling the out-of-fold predictions the
    AUC was computed from. That is the relevant uncertainty here: it asks how much
    the number would move on a different draw of events from the same detector,
    which is what a reader wants when comparing 0.79 against 0.83. It does not
    capture uncertainty from the VAE fit itself — for that see the latent sweep,
    where three seeds put the between-seed sd at roughly 0.005.

TWO FIGURES, NOT TWO PANELS
    proxy_auc  — AUC per (proxy, species), with bootstrap intervals.
    proxy_r2   — linear against non-linear R^2 for the same cells.

    Separate files because they answer different questions and either may be cited
    alone. They share the row layout, so they line up if a document places them
    side by side.

WHY AUC IS KEPT, AND WHAT proxy_r2 ADDS THAT IT CANNOT SAY
    The median split is the questionable part of an AUC-on-a-continuous-quantity: it
    binarises at an arbitrary cut and discards every distinction within each half.
    That was checked rather than assumed. Cross-validated Spearman correlation
    between the predicted and true proxy value — continuous, threshold-free — ranks
    all six (proxy, species) cells in EXACTLY the same order as the AUC, rank
    correlation +1.000. So the binarisation costs information but does not distort
    the comparison, and AUC is retained for its familiarity and its unambiguous
    chance reference at 0.5.

    What AUC genuinely cannot express is the difference between a proxy being
    PRESENT in the latent space and being LINEARLY READABLE from it. Panel (b) is
    that figure: cross-validated R^2 from a ridge probe against the same from a small MLP.
    The gap is the non-linearly encoded part. It matters most for kaon calorimetry,
    where linear R^2 = 0.41 against non-linear 0.72 — a linear readout sees barely
    half of what is there — while proton topology has a gap of 0.02, so what is
    present is fully accessible. MIP topology has a slightly NEGATIVE gap, which is
    the signature of little real signal rather than of non-linear encoding.

    Read together: proxy_auc says how well the geometry orders events, proxy_r2 how much
    of that ordering a linear reader can reach.

LENGTH IS DELIBERATELY ABSENT
    The old figure carried a third row, "Length Proxy" (total_adc). It is dropped:
    total_adc is close to a restatement of track length, which the calorimetry proxy
    already partly carries, and three near-collinear rows invited reading them as
    three independent successes. Two proxies, calorimetry and topology, are what the
    paper defines and what the clustering and sweep figures use.

Usage:
    python scripts/extra/plot_proxy_auc.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_proxy_auc.py --config ... --n-boot 5000
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, SINGLE_COL, SPECIES,
                        apply_style, figure_dir, load_beam_data, load_config, savefig)
from _sweep_measure import PROJECT_ROOT, regression_scores, val_row_indices

sys.path.insert(0, str(PROJECT_ROOT))
from scripts.analyse_latents import make_feature_auc_probe  # noqa: E402

PROXIES = {"mean_adc": "Calorimetry proxy", "solidity": "Topology proxy"}


def bootstrap_auc(y, proba, n_boot, seed=0):
    """Percentile bootstrap CI for an AUC, resampling events.

    Resamples the out-of-fold predictions rather than refitting: refitting would mix
    the sampling uncertainty this is meant to measure with the fit-to-fit noise of
    the probe, and the two answer different questions.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        if yb.min() == yb.max():          # degenerate resample, no both-class labels
            continue
        out.append(roc_auc_score(yb, proba[idx]))
    a = np.asarray(out)
    return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)), float(a.std(ddof=1))


def _row_layout():
    """Shared y-axis geometry: species rows grouped by proxy, with a gap between
    groups. Both figures use it so their rows line up if placed side by side in a
    document."""
    ypos, ylab, seps = [], [], []
    y = 0.0
    for i, f in enumerate(PROXIES):
        if i:
            y += 0.6
            seps.append(y - 0.3)
        for sp in SPECIES:
            ypos.append(y); ylab.append(DISPLAY[sp]); y += 1
    order = [(f, sp) for f in PROXIES for sp in SPECIES]
    return order, ypos, ylab, seps


def _finish(fig, ax, ypos, ylab, seps, s, xlabel):
    ax.set_yticks(ypos); ax.set_yticklabels(ylab)
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    for sep in seps:
        ax.axhline(sep, color="0.85", lw=0.6, zorder=0)
    # Proxy names as group headers on the right, so the y-axis stays species-only.
    for i, f in enumerate(PROXIES):
        centre = np.mean(ypos[i * len(SPECIES):(i + 1) * len(SPECIES)])
        ax.annotate(PROXIES[f], xy=(1.0, centre), xycoords=("axes fraction", "data"),
                    xytext=(11, 0), textcoords="offset points", rotation=90,
                    ha="center", va="center", fontsize=7.4 * s, color="0.25")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()


def plot_auc(rows, out_dir, stem="proxy_auc"):
    """How well the latent geometry orders events within each species.

    Same visual grammar as the two-sample figure — horizontal, a marker with an
    interval, a reference line — so a reader who has parsed one has parsed the other.
    """
    s = apply_style(SINGLE_COL)
    df = pd.DataFrame(rows)
    order, ypos, ylab, seps = _row_layout()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.55, 0.30 * len(order) + 1.15))
    ax.axvline(0.5, color="0.35", lw=0.8, ls="--", zorder=1, label="chance")
    for (f, sp), yy in zip(order, ypos):
        r = df[(df.feature == f) & (df.species == sp)].iloc[0]
        ax.errorbar(r.auc, yy, xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o",
                    ms=5, capsize=2.5, lw=1.2, color=COLOURS[sp], zorder=3)
        ax.annotate(f"{r.auc:.3f}", xy=(r.hi, yy), xytext=(4, 0),
                    textcoords="offset points", va="center", fontsize=6.6 * s,
                    color="0.3")
    ax.set_xlim(min(df.lo.min(), 0.5) - 0.03, 1.0 + 0.07)
    ax.legend(fontsize=7 * s, frameon=True, framealpha=0.85, edgecolor="0.75",
              loc="lower right")
    _finish(fig, ax, ypos, ylab, seps, s,
            "AUC — linear readout of the latent space")
    savefig(fig, out_dir, stem)


def plot_auc_compact(rows, out_dir, stem="proxy_auc_compact"):
    """Paper layout with one panel per proxy and a comfortably cropped axis.

    Every measured AUC lies between 0.788 and 0.952. Showing the full interval
    from chance to one makes the labels unnecessarily small at workshop column
    width, so this version uses a plainly labelled restricted axis. The caption
    states that chance is 0.5 and lies outside the displayed range.
    """
    s = apply_style(SINGLE_COL * 1.20)
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, len(PROXIES), figsize=(DOUBLE_COL, 2.30),
                             sharex=True, sharey=True)
    ypos = np.arange(len(SPECIES))

    for ax, (feature, title) in zip(np.atleast_1d(axes), PROXIES.items()):
        for yy, sp in zip(ypos, SPECIES):
            r = df[(df.feature == feature) & (df.species == sp)].iloc[0]
            ax.errorbar(r.auc, yy,
                        xerr=[[r.auc - r.lo], [r.hi - r.auc]], fmt="o",
                        ms=6.2 * s, capsize=2.6 * s, lw=1.25 * s,
                        color=COLOURS[sp], zorder=3)
            ax.annotate(f"{r.auc:.3f}", xy=(r.hi, yy), xytext=(5, 0),
                        textcoords="offset points", va="center",
                        fontsize=7.3 * s, color="0.25")
        ax.set_title(title, fontsize=9.2 * s, pad=6 * s)
        ax.set_xlim(0.765, 0.985)
        ax.set_xticks([0.80, 0.85, 0.90, 0.95])
        ax.set_yticks(ypos)
        ax.set_yticklabels([DISPLAY[sp] for sp in SPECIES])
        ax.grid(axis="x", color="0.90", lw=0.55 * s, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    np.atleast_1d(axes)[0].invert_yaxis()
    fig.supxlabel("AUC from a linear readout of the latent space",
                  fontsize=8.8 * s, y=0.055)
    fig.subplots_adjust(left=0.12, right=0.975, bottom=0.27, top=0.82,
                        wspace=0.32)
    savefig(fig, out_dir, stem)


def plot_r2(rows, out_dir, stem="proxy_r2"):
    """How much of what the latent space encodes a LINEAR readout can reach.

    A separate figure from the AUC rather than a second panel of it, because it
    answers a different question and deserves to be cited on its own: the AUC says
    how well the geometry orders events, this says how much of that ordering is
    linearly accessible.

    The connecting line is the quantity of interest, not decoration — it is the
    non-linearly encoded part, the information present in the latent space that a
    linear probe cannot see.
    """
    s = apply_style(SINGLE_COL)
    df = pd.DataFrame(rows)
    order, ypos, ylab, seps = _row_layout()

    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.55, 0.30 * len(order) + 1.15))
    for (f, sp), yy in zip(order, ypos):
        r = df[(df.feature == f) & (df.species == sp)].iloc[0]
        ax.plot([r.r2_linear, r.r2_mlp], [yy, yy], "-", lw=1.6,
                color=COLOURS[sp], alpha=0.45, zorder=2)
        ax.plot(r.r2_linear, yy, "o", ms=5.5, color=COLOURS[sp], zorder=3)
        ax.plot(r.r2_mlp, yy, "o", ms=5.5, mfc="white", mew=1.3,
                color=COLOURS[sp], zorder=3)
        ax.annotate(f"{r.r2_mlp - r.r2_linear:+.2f}",
                    xy=(max(r.r2_mlp, r.r2_linear), yy), xytext=(5, 0),
                    textcoords="offset points", va="center", fontsize=6.6 * s,
                    color="0.3")
    ax.plot([], [], "o", ms=5.5, color="0.35", label="linear (ridge)")
    ax.plot([], [], "o", ms=5.5, mfc="white", mew=1.3, color="0.35",
            label="non-linear (MLP)")
    ax.set_xlim(0, max(df.r2_mlp.max(), df.r2_linear.max()) + 0.17)
    ax.legend(fontsize=7 * s, frameon=True, framealpha=0.85, edgecolor="0.75",
              loc="lower right")
    _finish(fig, ax, ypos, ylab, seps, s,
            r"Cross-validated $R^2$ — proxy value from the latents")
    savefig(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--no-r2", action="store_true",
                    help="skip panel (b), the linear-vs-nonlinear R2 comparison")
    ap.add_argument("--paper-compact", action="store_true",
                    help="also write a compact two-panel AUC figure for the paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(figure_dir(cfg, "latents-features")))
    out_dir.mkdir(parents=True, exist_ok=True)

    probe = make_feature_auc_probe(return_scores=True)
    rows_idx = val_row_indices(cfg, df)
    print(f"validation events per species: "
          f"{', '.join(f'{DISPLAY[s]} {len(i)}' for s, i in rows_idx.items())}")

    rows = []
    for feat in PROXIES:
        for sp in SPECIES:
            idx = rows_idx[sp]
            auc, med, y, proba = probe(Z[idx], df.iloc[idx].reset_index(drop=True), feat)
            if auc is None:
                print(f"  {feat} / {sp}: skipped")
                continue
            lo, hi, sd = bootstrap_auc(y, proba, args.n_boot, seed=args.seed)
            row = {"feature": feat, "species": sp, "auc": auc,
                   "lo": lo, "hi": hi, "boot_sd": sd,
                   "n": int(len(y)), "median": float(med)}
            if not args.no_r2:
                sub = df.iloc[idx].reset_index(drop=True)
                v = sub[feat].to_numpy(float)
                m = np.isfinite(v)
                r2l, r2n, rho = regression_scores(Z[idx][m], v[m], seed=args.seed)
                row.update({"r2_linear": r2l, "r2_mlp": r2n, "spearman": rho})
            rows.append(row)
            extra = ("" if args.no_r2 else
                     f"  R2 {row['r2_linear']:.3f}->{row['r2_mlp']:.3f} "
                     f"(gap {row['r2_mlp'] - row['r2_linear']:+.3f})"
                     f"  rho={row['spearman']:.3f}")
            print(f"  {PROXIES[feat]:19s} {DISPLAY[sp]:7s} AUC={auc:.4f} "
                  f"[{lo:.4f}, {hi:.4f}]  n={len(y)}{extra}")

    # The median split is the questionable step, so check it against the
    # threshold-free statistic rather than defending it in prose.
    if not args.no_r2:
        rf = pd.DataFrame(rows)
        agree = spearmanr(rf["auc"], rf["spearman"]).statistic
        print(f"\nAUC vs threshold-free Spearman, rank agreement over the "
              f"{len(rf)} cells: {agree:+.3f}"
              + ("  (identical ordering — the median split costs information but "
                 "does not distort the comparison)" if agree > 0.99 else
                 "  (ORDERING DIFFERS — do not rely on the median-split AUC alone)"))

    plot_auc(rows, out_dir)
    if args.paper_compact:
        plot_auc_compact(rows, out_dir)
    if not args.no_r2:
        plot_r2(rows, out_dir)
    pd.DataFrame(rows).to_csv(out_dir / "proxy_auc.csv", index=False)
    with open(out_dir / "proxy_auc.json", "w") as fh:
        json.dump({"n_boot": args.n_boot, "rows": rows}, fh, indent=2)
    print(f"  saved {out_dir}/proxy_auc.csv and .json")


if __name__ == "__main__":
    main()
