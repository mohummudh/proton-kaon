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
import copy
import itertools
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, SINGLE_COL, SPECIES,
                        apply_style, load_beam_data, savefig)
from cluster_latents import fit_clusters, score_clusters

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.train.naming import model_name as build_model_name  # noqa: E402
from scripts.analyse_latents import make_feature_auc_probe  # noqa: E402

BLUE, ORANGE, PURPLE = "#0077BB", "#EE7733", "#AA3377"

# The two proxies the paper leads with. Keys are feature columns, values are the
# reader-facing names used everywhere else (see _beam_data.PROXY_LABELS).
PROXIES = {"mean_adc": "Calorimetry Proxy", "solidity": "Topology Proxy"}


def rung_configs(sweep_path: Path) -> list:
    """Rebuild every run's config the same way run_sweep.py does.

    Deliberately re-derived from the sweep YAML rather than read from the
    generated per-run configs: those live on the model drive and are rewritten by
    the next sweep, so reading them would make this script's answer depend on
    whatever ran last.

    Handles a grid with or without train.seed, so it still works on the older
    unseeded sweep config.
    """
    with open(sweep_path) as fh:
        sweep = yaml.safe_load(fh)
    base_path = Path(sweep["base"])
    if not base_path.is_absolute() and not base_path.exists():
        base_path = sweep_path.parent / base_path.name
    with open(base_path) as fh:
        base = yaml.safe_load(fh)

    grid = sweep.get("grid", {})
    keys = list(grid)
    out = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        cfg = copy.deepcopy(base)
        params = dict(zip(keys, combo))
        for dotted, value in params.items():
            section, key = dotted.split(".")
            cfg.setdefault(section, {})[key] = value
        out.append((params, cfg))
    return out


def read_log(cfg: dict) -> dict:
    """Best-epoch losses from a run's training log.

    `best val` is the epoch whose weights were actually saved: train() restores
    best_state before returning, so the checkpoint is the argmin of val loss, not
    the last epoch. Reporting the last epoch instead would understate every run by
    the patience window.
    """
    path = Path(cfg["output"]["dir"]) / (build_model_name(cfg) + ".json")
    if not path.exists():
        return {}
    with open(path) as fh:
        log = json.load(fh)
    hist = log.get("history", [])
    if not hist:
        return {}
    best = min(hist, key=lambda e: e["val_loss"])
    return {
        "n_train": log["dataset"]["n_train"],
        "n_val": log["dataset"]["n_val"],
        "epochs": len(hist),
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_recon": best["val_recon"],
        "train_loss": best["train_loss"],
    }


def val_row_indices(cfg: dict, df: pd.DataFrame) -> dict:
    """Row positions of each species' VALIDATION events within load_beam_data's Z.

    load_beam_data stacks [proton_train, proton_val, kaon_all, muon_all] and orders
    the proton block by concat(p_train_idx, p_val_idx), so proton validation is the
    tail of the proton block while kaon and muon are indexed directly by their
    within-species validation indices. Getting this wrong silently mispairs rows
    rather than raising, which is why it is derived here rather than assumed.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    ss = np.load(inf_dir / "species_split.npz")
    n_p_train = len(ss["p_train_idx"])
    n_p = n_p_train + len(ss["p_val_idx"])
    n_k = int((df["species"] == "kaon").sum())
    return {
        "proton": np.arange(n_p_train, n_p),
        "kaon": n_p + np.asarray(ss["k_val_idx"]),
        "muon": n_p + n_k + np.asarray(ss["m_val_idx"]),
    }


_SPECIES_SIZES = None


def species_sizes(features_path=None) -> dict:
    """Per-species row counts of the FULL image tensor, as {p, k, m}.

    Needed to decode a split file, whose indices address the concatenated
    [p, k, m] tensor and so depend on n_p to locate the kaon block. Taken from the
    features table rather than by loading the 510 MB image tensor, since the two
    are built row-for-row from the same clusters. Cached: every model in a sweep
    needs the same three numbers.
    """
    global _SPECIES_SIZES
    if _SPECIES_SIZES is None:
        from _beam_data import DEFAULT_FEATURES
        counts = pd.read_pickle(features_path or DEFAULT_FEATURES)["particle_type"].value_counts()
        _SPECIES_SIZES = {"p": int(counts["proton"]), "k": int(counts["kaon"]),
                          "m": int(counts["muon"])}
    return _SPECIES_SIZES


def common_eval_indices(cfg: dict, tag: str) -> dict:
    """Within-species indices of `tag`'s validation set, as {species: idx}.

    `tag` should be the highest-train-fraction rung, whose validation set is the
    smallest and — because the rungs nest — is contained in every other rung's
    validation set. That makes it the one set every model in the sweep has held
    out, and therefore the only fair basis for comparing their reconstruction.
    """
    n = species_sizes()
    val = np.load(Path(cfg["output"]["splits_dir"]) / f"split_all_{tag}.npz")["val_idx"]
    lo_k, lo_m = n["p"], n["p"] + n["k"]
    return {
        "proton": val[val < lo_k],
        "kaon": val[(val >= lo_k) & (val < lo_m)] - lo_k,
        "muon": val[val >= lo_m] - lo_m,
    }


def common_val_mse(cfg: dict, common: dict) -> float:
    """Mean per-event reconstruction MSE on the common held-out set.

    Why this exists: each rung's logged val loss is measured on its own validation
    set, and those shrink from 12339 events to 2469 across the sweep. Comparing
    them compares performance on different samples, which is enough to manufacture
    a bump that looks like a property of the model. This evaluates every model on
    one fixed set instead.

    Costs no forward passes: run_inference.py already stored a per-event MSE for
    every event in the pool, so this is a lookup. Note it is the unweighted
    per-pixel MSE from src/inference/inference.py, not the weighted objective the
    training log reports — a different scale, but the same for every model, which
    is all a comparison needs.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    if not (inf_dir / "train.npz").exists():
        return np.nan
    ss = np.load(inf_dir / "species_split.npz")
    n = species_sizes()

    # Protons are stored split across train.npz/val.npz in split order; rebuild an
    # array addressable by within-proton index. NaN-filled so that selecting an
    # event outside the pool would surface as NaN rather than as a wrong number.
    p_re = np.full(n["p"], np.nan)
    p_re[ss["p_train_idx"]] = np.load(inf_dir / "train.npz")["re"]
    p_re[ss["p_val_idx"]] = np.load(inf_dir / "val.npz")["re"]
    per_species = {
        "proton": p_re,
        "kaon": np.load(inf_dir / "kaon.npz")["re"],
        "muon": np.load(inf_dir / "muon.npz")["re"],
    }
    vals = np.concatenate([per_species[s][idx] for s, idx in common.items()])
    if np.isnan(vals).any():
        raise ValueError(f"{build_model_name(cfg)}: the common evaluation set "
                         f"includes events this model has no inference for")
    return float(vals.mean())


def measure_model(cfg: dict, seed: int = 0, do_clustering: bool = True) -> dict:
    """Clustering agreement and proxy AUCs for one trained model.

    Labels are used only to score the clustering, never to fit it — same protocol
    as cluster_latents.py, so the numbers are comparable to the paper model's.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    if not (inf_dir / "train.npz").exists():
        return {}
    Z, df = load_beam_data(cfg)
    out = {}

    if do_clustering:
        labels, _ = fit_clusters(Z, 3, seed=seed)
        out.update(score_clusters(labels, df["species"], 3))

    # Proxy AUCs on validation events only — the question is whether the latent
    # space encodes the proxy for events it was not fitted on.
    probe = make_feature_auc_probe()
    rows = val_row_indices(cfg, df)
    for feat in PROXIES:
        for sp, idx in rows.items():
            auc, _ = probe(Z[idx], df.iloc[idx].reset_index(drop=True), feat)
            if auc is not None:
                out[f"auc_{feat}_{sp}"] = auc
    return out


def write_local_configs(rungs: list, out_dir: Path) -> None:
    """Per-run configs carrying LOCAL paths, for run_inference.py and re-analysis."""
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ("# Generated by scripts/extra/plot_split_sweep.py --write-configs.\n"
              "# LOCAL paths, unlike the run_sweep.py-generated configs under\n"
              "# <models>/sweep_configs/, which are rewritten with the remote's paths\n"
              "# and so cannot be used for local inference or analysis.\n")
    for params, cfg in rungs:
        tag = cfg["data"]["tag"]
        seed = (cfg.get("train") or {}).get("seed")
        stem = tag + (f"_seed{seed}" if seed is not None else "")
        with open(out_dir / f"{stem}.yaml", "w") as fh:
            fh.write(header)
            yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"  wrote {len(rungs)} configs to {out_dir}")


# ── plotting ──────────────────────────────────────────────────────────────────

def _agg(df: pd.DataFrame, col: str) -> pd.DataFrame:
    """Mean and across-seed sd of `col` per training-set size."""
    sub = df[df[col].notna()]
    if sub.empty:
        return sub
    g = sub.groupby("n_train")[col]
    return pd.DataFrame({"n_train": g.mean().index, "mean": g.mean().values,
                         "sd": g.std(ddof=1).values, "n": g.size().values})


def _errline(ax, df, col, colour, label=None, marker="o", ls="-"):
    a = _agg(df, col)
    if a.empty:
        return False
    ax.errorbar(a["n_train"], a["mean"], yerr=np.nan_to_num(a["sd"]),
                fmt=marker, ls=ls, color=colour, ms=4.2, lw=1.1, capsize=2.5,
                elinewidth=0.9, label=label)
    return True


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
    _errline(ax, df, "val_recon", BLUE)
    ax.set_ylabel("Validation reconstruction loss (weighted)")
    ax.set_title("(a) reconstruction", loc="left", fontsize=9 * s, pad=3)

    # (b) unsupervised structure
    ax = axes[0, 1]
    if has_clusters:
        _errline(ax, df, "ari", ORANGE, label="ARI")
        _errline(ax, df, "purity", PURPLE, label="majority purity", marker="s", ls="--")
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
            drawn |= _errline(ax, df, f"auc_{feat}_{sp}", COLOURS[sp],
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
    summary = df.groupby("n_train")[show].agg(["mean", "std"]).round(4)
    print(f"\n{summary.to_string()}")
    summary.to_csv(out_dir / "split_sweep_summary.csv")
    print(f"  saved {out_dir / 'split_sweep_summary.csv'}")


if __name__ == "__main__":
    main()
