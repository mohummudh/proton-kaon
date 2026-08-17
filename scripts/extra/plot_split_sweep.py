#!/usr/bin/env python3
"""
scripts/extra/plot_split_sweep.py

Summarise the training-statistics sweep in one figure: as the model sees more
data, does it reconstruct better, does the latent space organise into species,
and does it encode the physics proxies more strongly?

WHAT IS BEING COMPARED, AND WHY IT IS A FAIR COMPARISON
    Every rung draws its training set from the SAME fixed 8227-per-species pool,
    and the training sets NEST (50% is a subset of 55%, and so on -- see
    scripts/make_balanced_split.py). So two rungs differ by how many images the
    VAE saw and by nothing else: not by which images, and not by species mixture,
    since train and val are each exactly balanced at every rung.

    That is what licenses reading the x-axis as a dose. Without it a bumpy curve
    could just as easily be a different sample as a different sample size.

THE ERROR BARS ARE THE POINT
    Each ratio is trained at several seeds (configs/sweep_split_pool8227_seeded.yaml
    uses three), so the spread at fixed ratio measures training-run variance and
    the between-ratio differences finally have something to be significant
    against. The first version of this sweep had one unseeded run per ratio and
    could not distinguish a trend from a draw: it gave clustering ARI 0.393 /
    0.379 / 0.312 / 0.441 / 0.387 over five ratios, a 0.13 spread with no trend.

    Points are the mean over seeds; bars are +/- the sample sd across seeds (not
    the standard error -- with three seeds the sd is the honest description of how
    much a single run moves, which is what a reader wants to know).

    Read overlap, not ordering. If neighbouring error bars overlap, the ratios are
    not resolved, however suggestive the means look.

THE FOUR PANELS
    (a) validation reconstruction loss -- the model's own objective, and the thing
        that must improve if extra data is doing anything at all.
    (b) agreement between an unsupervised k=3 Gaussian mixture on the raw latent
        space and the beam tags (ARI and majority purity). Species structure
        emerging with no labels in the fit.
    (c) calorimetry proxy AUC and (d) topology proxy AUC -- can a linear readout of
        the latent space say whether an event is above or below its species' median
        for mean_adc / solidity? This is the paper's central claim stated as a
        number, and it is measured on validation events only, so it is about
        generalisation rather than memorisation.

    (c) and (d) use the same probe as the feature_auc analysis in
    analyse_latents.py -- imported from it, not reimplemented -- so the numbers are
    directly comparable to the ones quoted for the paper model.

PANEL (a): THE DIFFERING VALIDATION SETS WERE CHECKED AND DO NOT MATTER
    Each rung's val loss is measured on its own validation set, and those shrink
    from 12339 events to 2469 across the sweep, which looks like it should bias the
    comparison. It was tested rather than assumed: `common_val_mse` re-evaluates
    every model on the 2469 events of the highest rung's validation set, which the
    nesting makes a subset of every other rung's. Own-set and common-set values
    agree to better than 0.5% at every rung (e.g. 0.19338 vs 0.19436 at tr50), and
    both give the same verdict. The validation sets are interchangeable here.

    WHAT DOES MATTER IS WHICH RECONSTRUCTION METRIC
        The trend in (a) is present in the *weighted* loss the VAE actually
        optimises (Spearman rho = -0.86, p < 1e-5) and absent in the unweighted
        per-pixel MSE (rho = -0.24 to -0.32, p = 0.10 to 0.23). That is not a
        contradiction: src/losses/vae.py upweights signal pixels 10x, while an
        unweighted mean over a 48x48 image is dominated by the mostly-empty
        background, which every model reconstructs trivially well. The unweighted
        number is simply too insensitive to see the change. Panel (a) therefore
        plots the weighted loss, and `common_val_mse` stays in the CSV as the
        record of this check rather than being plotted as a rival curve.

    Panel (b) is computed over every event in the latent space, the same set at
    every rung. (c) and (d) are val-only and so do shrink with the ratio -- which
    shows up as wider error bars at high ratios, the honest signature of a smaller
    measurement sample.

REQUIRES
    Each rung's training log (synced by run_sweep.py into output.dir) and, for
    panels (b)-(d), its inference outputs. Run inference for every rung first:

        for cfg in configs/generated_split_sweep/pool8227_tr*_seed*.yaml; do
            python scripts/run_inference.py --config "$cfg"
        done

    Use those configs, NOT the ones run_sweep.py leaves in <models>/sweep_configs/:
    the sweep rewrites data.path and output.dir with the remote machine's paths, so
    locally they point at /home/mohammed/... and inference fails on a missing file.
    Generate them with --write-configs.

    Runs with no inference yet are simply left out of the affected panels, so this
    script is usable while the sweep is still going.

Usage:
    # write the local per-run configs the inference loop above needs
    python scripts/extra/plot_split_sweep.py --sweep configs/sweep_split_pool8227_seeded.yaml \
        --write-configs

    python scripts/extra/plot_split_sweep.py --sweep configs/sweep_split_pool8227_seeded.yaml
    python scripts/extra/plot_split_sweep.py --sweep ... --from-cache   # replot only
    python scripts/extra/plot_split_sweep.py --sweep ... --no-clustering
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
from _sweep_measure import (PROJECT_ROOT, PROXIES, agg, common_eval_indices,
                            common_val_mse, errline, measure_model, read_log,
                            run_id, rung_configs, write_local_configs)

BLUE, ORANGE, PURPLE = "#0077BB", "#EE7733", "#AA3377"
X = "n_train"

# ── plotting ──────────────────────────────────────────────────────────────────

def plot_sweep(df: pd.DataFrame, out_dir: Path, has_clusters: bool) -> None:
    s = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.72))

    # (a) the model's own objective: the weighted reconstruction term, which is
    # what the VAE was actually trained on. `common_val_mse` is computed and kept
    # in the CSV as a diagnostic but deliberately NOT plotted here — see the
    # docstring: it is the unweighted MSE, a different and much less sensitive
    # quantity, so putting the two on twin axes invites reading a metric change as
    # a sample-bias correction.
    ax = axes[0, 0]
    errline(ax, df, "val_recon", X, BLUE)
    ax.set_ylabel("Validation reconstruction loss (weighted)")
    ax.set_title("(a) reconstruction", loc="left", fontsize=9 * s, pad=3)

    # (b) unsupervised structure
    ax = axes[0, 1]
    if has_clusters:
        errline(ax, df, "ari", X, ORANGE, label="ARI")
        errline(ax, df, "purity", X, PURPLE, label="majority purity", marker="s", ls="--")
        ax.set_ylabel("Agreement with beam tags")
        ax.legend(fontsize=7.5 * s, frameon=True, framealpha=0.85, edgecolor="0.75")
    else:
        ax.axis("off")
    ax.set_title("(b) unsupervised GMM ($k=3$)", loc="left", fontsize=9 * s, pad=3)

    # (c), (d) proxy AUCs, per species
    for j, (feat, label) in enumerate(PROXIES.items()):
        ax = axes[1, j]
        drawn = False
        for sp in SPECIES:
            drawn |= errline(ax, df, f"auc_{feat}_{sp}", X, COLOURS[sp],
                              label=DISPLAY[sp])
        if drawn:
            # Let the axis follow the data. Forcing chance (0.5) into view puts
            # every curve in the top fifth of the panel, which buries the very
            # thing these panels exist to show — whether the AUC moves with
            # training size by more than the between-seed spread. The chance line
            # is drawn only when it is already in range; the AUC scale is
            # universally understood, so its absence costs the reader nothing.
            lo, hi = ax.get_ylim()
            if lo <= 0.55:
                ax.axhline(0.5, color="0.5", lw=0.7, ls=":", zorder=0)
            ax.set_ylabel(f"{label} AUC")
            ax.legend(fontsize=7 * s, frameon=True, framealpha=0.85,
                      edgecolor="0.75", ncol=3, columnspacing=0.9,
                      handletextpad=0.4, loc="best")
        else:
            ax.axis("off")
        ax.set_title(f"({'cd'[j]}) {label.lower()}, val only",
                     loc="left", fontsize=9 * s, pad=3)

    for ax in axes.ravel():
        if ax.axison:
            ax.set_xlabel("Training images")
            ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_dir, "split_sweep")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep", default="configs/sweep_split_pool8227_seeded.yaml")
    ap.add_argument("--no-clustering", action="store_true",
                    help="skip the GMM panel; it is the slow part")
    ap.add_argument("--no-common-eval", action="store_true",
                    help="plot each rung's own validation loss instead of re-evaluating "
                         "every model on the one set they all held out. The own-val "
                         "version compares different samples per rung, so prefer the "
                         "default unless you specifically want it.")
    ap.add_argument("--from-cache", action="store_true",
                    help="replot from split_sweep_runs.csv without recomputing")
    ap.add_argument("--write-configs", action="store_true",
                    help="write local per-run configs for run_inference.py, then exit")
    ap.add_argument("--configs-dir", default="configs/generated_split_sweep")
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

    out_dir = Path(args.out_dir) if args.out_dir else (PROJECT_ROOT / "figs" / "split_sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "split_sweep_runs.csv"

    if args.from_cache and cache.exists():
        df = pd.read_csv(cache)
        print(f"Loaded {len(df)} runs from {cache.name}")
    else:
        # The highest-train-fraction rung has the smallest validation set, and the
        # rungs nest, so it is the set every model held out.
        common = None
        if not args.no_common_eval:
            last_tag = rungs[-1][1]["data"]["tag"]
            common = common_eval_indices(rungs[-1][1], last_tag)
            print(f"Common evaluation set: {last_tag} validation, "
                  f"{sum(len(v) for v in common.values())} events "
                  f"({', '.join(f'{s} {len(v)}' for s, v in common.items())})")
        rows = []
        for params, cfg in rungs:
            tag = cfg["data"]["tag"]
            seed = (cfg.get("train") or {}).get("seed")
            row = {"tag": tag, "seed": seed, **read_log(cfg)}
            if not row.get("n_train"):
                print(f"  {tag} seed={seed}: no training log yet — skipped")
                continue
            if common is not None:
                row["common_val_mse"] = common_val_mse(cfg, common)
            row.update(measure_model(cfg, seed=args.seed,
                                     do_clustering=not args.no_clustering))
            rows.append(row)
            print(f"  {tag} seed={seed}: n_train={row['n_train']:6d}  "
                  f"val_recon={row['val_recon']:9.1f}"
                  + (f"  ARI={row['ari']:.3f}" if "ari" in row else "")
                  + (f"  calo={row.get('auc_mean_adc_proton', float('nan')):.3f}"
                     if "auc_mean_adc_proton" in row else ""))
        if not rows:
            print("No runs have finished training yet — nothing to plot.")
            return
        df = pd.DataFrame(rows).sort_values(["n_train", "seed"]).reset_index(drop=True)
        df.to_csv(cache, index=False)
        print(f"  saved {cache}")

    plot_sweep(df, out_dir, has_clusters="ari" in df.columns and df["ari"].notna().any())

    # Per-ratio summary: the means and the across-seed spreads side by side.
    show = [c for c in ["val_recon", "ari", "purity"] if c in df.columns]
    show += [c for c in df.columns if c.startswith("auc_")]
    summary = df.groupby(X)[show].agg(["mean", "std"]).round(4)
    print(f"\n{summary.to_string()}")
    summary.to_csv(out_dir / "split_sweep_summary.csv")
    print(f"  saved {out_dir / 'split_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
