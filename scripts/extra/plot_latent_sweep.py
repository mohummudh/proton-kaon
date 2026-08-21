#!/usr/bin/env python3
"""
scripts/extra/plot_latent_sweep.py

Summarise the latent-capacity sweep (configs/sweep_latent_pool8227_tr50.yaml):
latent dim 4 to 128 in steps of 4, three seeds each, data split fixed at 50/50.

THE QUESTION, AND WHY IT IS NOT THE SAME AS THE SPLIT SWEEP'S
    plot_split_sweep.py asked how much DATA the latent space needs, and found the
    physics content saturated below 12k images. This asks how much CAPACITY it
    needs. The two can come apart: a representation can be data-saturated and still
    capacity-limited, or vice versa.

PANELS (a) AND (c): NOTHING COLLAPSES, AND HERE IS WHY
    "Latent dim 128" says what was made AVAILABLE, not what is USED, so the first
    thing to establish is the second. The expectation going in was the textbook
    beta-VAE one: surplus dimensions get switched off, their posterior mean pinned
    to a constant so they cost no KL, and the count of survivors leaves the diagonal
    at the point where capacity stops binding.

    That is NOT what these models do. Every dimension stays active: the
    Var(mu) > 0.01 count sits exactly on the diagonal at every point, and at latent
    64 even the smallest per-dimension variance is 0.27, twenty-seven times the
    threshold. Panel (a) plots it anyway, on the diagonal, because that is the
    evidence for the claim.

    Panel (c) is the explanation. The reconstruction term is a weighted MSE *summed*
    over 2x48x48 pixels while the KL is a mean over dimensions, so at beta = 0.5 the
    KL is 0.5% of the objective at latent 4 rising to only 5.4% at latent 64. There
    is almost no pressure to compress, which makes this a lightly regularised
    autoencoder rather than a tightly-constrained beta-VAE. Posterior collapse never
    arises, and the standard active-units diagnostic is therefore uninformative
    here.

    So read the participation ratio, (sum var)^2 / sum(var^2), which is continuous
    and threshold-free and does resolve the effect: effective dimensions grow
    sub-linearly with the nominal count (PR/latent falls from 0.95 at latent 4 to
    0.73 at latent 64), so surplus capacity is diluted rather than switched off.
    n_dims_95var is the coarser threshold-free companion.

THE OTHER PANELS
    (b) validation reconstruction loss, weighted -- the objective the VAE optimises.
        Note the unweighted per-pixel MSE is too insensitive to see changes here;
        see plot_split_sweep.py, where that was measured.
    (d) unsupervised GMM (k=3) agreement with the beam tags.
    (e), (f) calorimetry and topology proxy AUCs, val only -- the paper's central
        claim, that a linear readout of the latent space recovers the physics.

    Reading (b) against (d)-(f) is the point of the figure. Reconstruction more than
    halves across the sweep, so if the physics panels stay flat then latent 8 is not
    a reconstruction-optimal choice but may still be a physics-sufficient one, and
    the paper's use of it needs to be defended on those grounds rather than on
    reconstruction.

CAVEAT ON (e) AND (f): THE PROBE GROWS WITH THE LATENT SPACE
    The proxy AUC is a logistic regression on the latents, so its number of
    features is the latent dim. Comparing AUC across this sweep therefore varies
    the probe as well as the representation, which it does not in the split sweep.

    It is milder than it first looks. The AUC is cross-validated, so extra
    dimensions carrying no information cannot inflate it -- CV does not reward noise
    features. And this sweep runs at the 50/50 split, so each species contributes
    4113 validation events: even at latent 128 that is a 32:1 ratio of events to
    coefficients, comfortable for a regularised logistic regression. The
    events-to-features squeeze would matter at a 90/10 split, where the count drops
    to 823 per species, but not here.

    What remains is that a fall at the top end is weaker evidence than a rise,
    since added variance in the probe can only push the AUC down. Panels (a) and (c)
    have no such dependence at all.

CAVEAT ON (d): THE CLUSTERING ALSO GROWS
    A full-covariance k=3 mixture fits 3*D*(D+1)/2 covariance parameters, which at
    D=128 is ~24,800 against 25,418 events. The fit is regularised (sklearn's
    reg_covar) and does converge, but it is near the limit of what the sample
    supports, so a decline in ARI at high D is partly the mixture model running out
    of data rather than the latent space losing structure. n_init is held at 20,
    the same as cluster_latents.py, so the numbers stay comparable to the paper's.

Usage:
    python scripts/extra/plot_latent_sweep.py --write-configs   # then run inference
    python scripts/extra/plot_latent_sweep.py
    python scripts/extra/plot_latent_sweep.py --from-cache      # replot only
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, SINGLE_COL, SPECIES,
                        apply_style, savefig)
from _sweep_measure import (plot_r2_sweep, PROJECT_ROOT, PROXIES, agg, errline, measure_model,
                            read_log, run_id, rung_configs, write_local_configs)

BLUE, ORANGE, PURPLE, GREEN = "#0077BB", "#EE7733", "#AA3377", "#009988"
X = "latent"


def plot_sweep(df: pd.DataFrame, out_dir: Path) -> None:
    s = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL * 1.08, DOUBLE_COL * 0.62))

    # (a) how much of the capacity is actually used. The diagonal is "all of it";
    # n_active_dims sits exactly on it, which is the point — nothing collapses.
    ax = axes[0, 0]
    lo, hi = df[X].min(), df[X].max()
    ax.plot([lo, hi], [lo, hi], ls="--", lw=0.8, color="0.6", zorder=0,
            label="all available")
    errline(ax, df, "n_active_dims", X, PURPLE, label=r"Var$(\mu)>0.01$",
            marker="^", ls=":", ms=3.4)
    errline(ax, df, "n_dims_95var", X, GREEN, label="95% of variance",
            marker="s", ls="--")
    errline(ax, df, "participation_ratio", X, BLUE, label="participation ratio")
    ax.set_ylabel("Effective latent dimensions")
    ax.set_title("(a) capacity actually used", loc="left", fontsize=9 * s, pad=3)
    ax.legend(fontsize=6.4 * s, frameon=True, framealpha=0.85, edgecolor="0.75",
              loc="upper left")

    # (b) reconstruction, (c) KL's share of the objective — the latter is what
    # explains (a): a few percent of the loss cannot force compression.
    ax = axes[0, 1]
    errline(ax, df, "val_recon", X, BLUE)
    ax.set_ylabel("Validation reconstruction loss")
    ax.set_title("(b) reconstruction", loc="left", fontsize=9 * s, pad=3)

    ax = axes[0, 2]
    if errline(ax, df, "kl_share", X, ORANGE):
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_ylabel(r"$\beta\,$KL / total objective")
    ax.set_title("(c) KL share of the loss", loc="left", fontsize=9 * s, pad=3)

    # (d) unsupervised structure
    ax = axes[1, 0]
    drawn = errline(ax, df, "ari", X, ORANGE, label="ARI")
    drawn |= errline(ax, df, "purity", X, PURPLE, label="majority purity",
                     marker="s", ls="--")
    if drawn:
        ax.set_ylabel("Agreement with beam tags")
        ax.legend(fontsize=7 * s, frameon=True, framealpha=0.85, edgecolor="0.75")
    ax.set_title("(d) unsupervised GMM ($k=3$)", loc="left", fontsize=9 * s, pad=3)

    # (e), (f) proxy AUCs per species
    for j, (feat, label) in enumerate(PROXIES.items()):
        ax = axes[1, j + 1]
        drawn = False
        for sp in SPECIES:
            drawn |= errline(ax, df, f"auc_{feat}_{sp}", X, COLOURS[sp],
                             label=DISPLAY[sp])
        if drawn:
            # Autoscale rather than forcing chance (0.5) into view: that would put
            # every curve in the top fifth of the panel and hide whether the AUC
            # moves at all, which is what the panel is for.
            if ax.get_ylim()[0] <= 0.55:
                ax.axhline(0.5, color="0.5", lw=0.7, ls=":", zorder=0)
            ax.set_ylabel(f"{label} AUC")
            ax.legend(fontsize=6.5 * s, frameon=True, framealpha=0.85,
                      edgecolor="0.75", ncol=3, columnspacing=0.8,
                      handletextpad=0.3, loc="best")
        ax.set_title(f"({'ef'[j]}) {label.lower()}, val only",
                     loc="left", fontsize=9 * s, pad=3)

    for ax in axes.ravel():
        ax.set_xlabel("Latent dimension")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_dir, "latent_sweep")


def plot_recon_by_species(df: pd.DataFrame, out_dir: Path) -> None:
    """Reconstruction error split by species, and the train/val gap it hides.

    The pooled loss in the main figure's panel (b) conceals the two largest facts
    about reconstruction in this sweep.

    (a) The species are nowhere near equally hard. Kaons reconstruct 6.5-7.7x worse
        than protons at every latent dimension, and the gap narrows only slightly
        with capacity (7.3x at latent 4 to 6.5x at 128). Log y-axis, because a
        linear one puts protons and MIPs on the floor.

    (b) The val/train ratio, which is where capacity stops looking free. Protons sit
        flat near 1.07 across the whole sweep and MIPs rise only to 1.09, while
        KAONS climb from 1.09 to 1.38. Extra capacity is being spent memorising
        kaons specifically.

        That is a concrete argument for a modest latent dimension which does not
        appeal to interpretability: past roughly latent 48 the kaon reconstruction
        gain is small and increasingly comes from memorisation. It is also
        physically unsurprising — the kaon sample is the contaminated, most
        heterogeneous one, so it offers the most idiosyncratic detail to memorise.
    """
    s = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.82, DOUBLE_COL * 0.34))

    ax = axes[0]
    for sp in SPECIES:
        errline(ax, df, f"re_{sp}_val", X, COLOURS[sp], label=DISPLAY[sp])
    ax.set_yscale("log")
    ax.set_ylabel("Reconstruction MSE (val)")
    ax.set_title("(a) by species", loc="left", fontsize=9 * s, pad=3)
    ax.legend(fontsize=7 * s, frameon=True, framealpha=0.85, edgecolor="0.75")

    ax = axes[1]
    ax.axhline(1.0, color="0.5", lw=0.7, ls=":", zorder=0)
    for sp in SPECIES:
        errline(ax, df, f"re_{sp}_ratio", X, COLOURS[sp], label=DISPLAY[sp])
    ax.set_ylabel("val / train reconstruction MSE")
    ax.set_title("(b) generalisation gap", loc="left", fontsize=9 * s, pad=3)

    for ax in axes:
        ax.set_xlabel("Latent dimension")
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_dir, "latent_sweep_recon_by_species")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="configs/sweep_latent_pool8227_tr50.yaml")
    ap.add_argument("--no-clustering", action="store_true",
                    help="skip panel (d); it is the slow part at high latent dim")
    ap.add_argument("--from-cache", action="store_true",
                    help="replot from latent_sweep_runs.csv without recomputing")
    ap.add_argument("--write-configs", action="store_true",
                    help="write local per-run configs for run_inference.py, then exit")
    ap.add_argument("--configs-dir", default="configs/generated_latent_sweep")
    ap.add_argument("--no-r2", action="store_true",
                    help="skip the latent->proxy R2 figure; it is the slow part, "
                         "roughly 20s per model")
    ap.add_argument("--seed", type=int, default=0, help="GMM seed")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    sweep_path = Path(args.sweep)
    if not sweep_path.is_absolute() and not sweep_path.exists():
        sweep_path = PROJECT_ROOT / args.sweep
    rungs = rung_configs(sweep_path)

    if args.write_configs:
        d = Path(args.configs_dir)
        write_local_configs(rungs, d if d.is_absolute() else PROJECT_ROOT / d)
        return

    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "figs" / "latent_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "latent_sweep_runs.csv"

    if args.from_cache and cache.exists():
        df = pd.read_csv(cache)
        print(f"Loaded {len(df)} runs from {cache.name}")
    else:
        rows = []
        for _, cfg in rungs:
            ident = run_id(cfg)
            row = {**ident, **read_log(cfg)}
            if not row.get("n_train"):
                print(f"  latent {ident['latent']:3d} seed={ident['seed']}: "
                      f"no training log yet — skipped")
                continue
            row.update(measure_model(cfg, seed=args.seed,
                                     do_clustering=not args.no_clustering,
                                     do_r2=not args.no_r2))
            rows.append(row)
            nan = float("nan")
            print(f"  latent {ident['latent']:3d} seed={ident['seed']}: "
                  f"eff={row.get('participation_ratio', nan):5.1f} "
                  f"(95%: {row.get('n_dims_95var', nan):3.0f})  "
                  f"recon={row['val_recon']:8.1f}  "
                  f"kl_share={row['kl_share']:5.2%}"
                  + (f"  ARI={row['ari']:.3f}" if "ari" in row else ""))
        if not rows:
            print("No runs have finished training yet — nothing to plot.")
            return
        df = pd.DataFrame(rows).sort_values([X, "seed"]).reset_index(drop=True)
        df.to_csv(cache, index=False)
        print(f"  saved {cache}")

    plot_sweep(df, out_dir)
    if any(c.startswith("r2mlp_") for c in df.columns):
        plot_r2_sweep(df, out_dir, X, "Latent dimension", "latent_sweep_r2", savefig, apply_style,
                      COLOURS, DISPLAY, SPECIES, SINGLE_COL, DOUBLE_COL, logx=False)
    if any(c.startswith("re_") for c in df.columns):
        plot_recon_by_species(df, out_dir)

    show = [c for c in ["participation_ratio", "n_dims_95var", "val_recon", "val_kl",
                        "ari", "purity"] if c in df.columns]
    show += [c for c in df.columns if c.startswith(("auc_", "re_"))]
    summary = df.groupby(X)[show].agg(["mean", "std"]).round(4)
    summary.to_csv(out_dir / "latent_sweep_summary.csv")
    # Print whichever of the headline columns actually survived into `show`, rather
    # than a fixed list: the set has changed as metrics were added and removed, and
    # a hardcoded one silently becomes a KeyError the next time it does.
    headline = [c for c in ("participation_ratio", "n_dims_95var", "val_recon")
                if c in summary.columns.get_level_values(0)]
    print(f"\n{summary[headline].to_string()}")
    print(f"  saved {out_dir / 'latent_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
