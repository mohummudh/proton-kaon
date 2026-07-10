#!/usr/bin/env python3
"""
scripts/extra/plot_physics_plane_nonlinear.py

Nonlinear projection of the 8D VAE latent space onto a 2D "physics plane"
(x = predicted solidity, y = predicted mean_adc -- matches the axis
convention in scripts/extra/plot_mean_adc_vs_solidity.py), fit on ALL THREE
species pooled (proton + kaon + muon) and then applied back to each species
individually.

Three regressors are fit on a stratified train/val split of the pooled,
all-species data and compared by held-out R^2 per target dimension:
    - LinearRegression            (baseline)
    - MLPRegressor (1 hidden layer, standardised inputs)
    - RandomForestRegressor       (one per target)
The split fraction and seed are controlled via --val-split/--random-seed
(default 90/10, seed 42).

The best-scoring model is then used to project Z_proton, Z_kaon, and Z_muon
into the 2D plane. For each (val_split, seed) run, six scatter plots are
saved: all species combined, one per species alone, and two pairwise plots
(proton+kaon, proton+muon) -- plus a metrics.json recording split sizes,
R^2 per model/target, the chosen best model, and predicted-vs-true Pearson
correlations per species.

--val-split and --random-seed each accept multiple values, which triggers a
sweep over every (val_split, seed) combination. Sweep runs are organised
under figs/physics_plane_nonlinear/sweep/valsplit<X.XX>_seed<N>/ (one folder
per combo, named from its params), with a sweep_summary.csv written at the
sweep root aggregating R^2 and correlations across all runs for comparison.

CAVEAT: this projection uses a flexible nonlinear regressor (MLP or random
forest) fit on pooled data from all species, so it is no longer a pure
"proton-trained, applied out-of-distribution" probe -- the regressor has seen
kaon and muon (latent, mean_adc/solidity) pairs directly during training. That
means any separation you see between species in the resulting plot is at
least partly *by construction* (the model was trained to predict mean_adc/
solidity using whatever species-correlated latent structure helps it fit the
pooled targets), not purely evidence of latent structure discovered from
protons alone. Treat this plot as a visualisation aid only -- it is not
evidence for the linear-probe claims made elsewhere in the analysis (see
scripts/analyse_latents.py), which train and test within species-appropriate
splits.

Usage:
    python scripts/extra/plot_physics_plane_nonlinear.py \
        --config configs/run_0066_model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p.yaml

    # sweep over val-split fractions and seeds:
    python scripts/extra/plot_physics_plane_nonlinear.py \
        --val-split 0.1 0.2 0.3 --random-seed 42 7
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
OUT_DIR = PROJECT_ROOT / "figs" / "physics_plane_nonlinear"
TARGET_NAMES = ["mean_adc", "solidity"]

COLOURS = {
    "proton": "#4C78A8",
    "kaon":   "#F58518",
    "muon":   "#9467BD",
}

DPI = 150
plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       12,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      DPI,
    "savefig.dpi":     DPI,
    "savefig.bbox":    "tight",
})


def build_model_name(cfg: dict) -> str:
    species_tag = "_speciesall" if cfg["data"].get("proton") == "all" else ""
    return (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
        f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
        f"_tx{cfg['data'].get('transform', 'none')}{species_tag}"
    )


def load_latents_and_features(cfg: dict, features_path: str):
    """Returns dict of species -> (Z, mean_adc, solidity), all positionally
    aligned (see scripts/analyse_latents.py::load_features_and_splits for the
    same alignment convention used across the analysis pipeline)."""
    model_name = build_model_name(cfg)
    inference_dir = Path(cfg["output"]["inference_dir"]) / model_name

    train_latents = np.load(inference_dir / "train.npz")["latents"]
    val_latents   = np.load(inference_dir / "val.npz")["latents"]
    kaon_latents  = np.load(inference_dir / "kaon.npz")["latents"]
    muon_npz = inference_dir / "muon.npz"
    muon_latents = np.load(muon_npz)["latents"] if muon_npz.exists() else None

    features = pd.read_pickle(features_path)
    split_p = np.load(Path(cfg["output"]["splits_dir"]) / "split_p.npz")

    all_proton = features[features["particle_type"] == "proton"].reset_index(drop=True)
    all_kaon   = features[features["particle_type"] == "kaon"].reset_index(drop=True)
    all_muon   = features[features["particle_type"] == "muon"].reset_index(drop=True)

    # Reassemble proton features in the same order as vstack([train, val]) latents,
    # using the saved split indices (proton latents are stored train-then-val).
    Z_proton = np.vstack([train_latents, val_latents])
    proton_order = np.concatenate([split_p["train_idx"], split_p["val_idx"]])
    proton_feats = all_proton.iloc[proton_order].reset_index(drop=True)

    data = {
        "proton": (Z_proton, proton_feats["mean_adc"].to_numpy(), proton_feats["solidity"].to_numpy()),
        "kaon":   (kaon_latents, all_kaon["mean_adc"].to_numpy(), all_kaon["solidity"].to_numpy()),
    }
    if muon_latents is not None and len(muon_latents) == len(all_muon):
        data["muon"] = (muon_latents, all_muon["mean_adc"].to_numpy(), all_muon["solidity"].to_numpy())
    else:
        print("  (muon latents unavailable or misaligned with features -- skipping muons)")

    return data


def fit_and_score(Z_tr, Y_tr, Z_val, Y_val):
    """Fit LinearRegression, MLPRegressor, and per-target RandomForestRegressor
    on the proton train split; return held-out R^2 (per target) for each."""
    results = {}

    # -- Linear baseline --
    lin = LinearRegression().fit(Z_tr, Y_tr)
    y_pred = lin.predict(Z_val)
    results["linear"] = {
        "model": lin,
        "r2": [r2_score(Y_val[:, j], y_pred[:, j]) for j in range(Y_val.shape[1])],
    }

    # -- MLP (single hidden layer, inputs standardised) --
    mlp = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16,),
            activation="relu",
            solver="adam",
            max_iter=2000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=15,
        )),
    ]).fit(Z_tr, Y_tr)
    y_pred = mlp.predict(Z_val)
    results["mlp"] = {
        "model": mlp,
        "r2": [r2_score(Y_val[:, j], y_pred[:, j]) for j in range(Y_val.shape[1])],
    }

    # -- Random forest, one model per target --
    rf_models = []
    rf_r2 = []
    for j in range(Y_tr.shape[1]):
        rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
        rf.fit(Z_tr, Y_tr[:, j])
        rf_models.append(rf)
        rf_r2.append(r2_score(Y_val[:, j], rf.predict(Z_val)))
    results["rf"] = {"model": rf_models, "r2": rf_r2}

    return results


class RFWrapper:
    """Wraps a pair of per-target RandomForestRegressors behind a single
    .predict(Z) -> (N, 2) interface, so it's interchangeable with the
    MLP/linear models below."""

    def __init__(self, models):
        self.models = models

    def predict(self, Z):
        return np.column_stack([m.predict(Z) for m in self.models])


def pick_best_model(results):
    """Pick the model with the best mean held-out R^2 across both targets
    (ties/near-ties default to the MLP, per the task spec)."""
    means = {name: np.mean(r["r2"]) for name, r in results.items()}
    best_name = max(means, key=means.get)
    if means["mlp"] >= means[best_name] - 0.01:
        best_name = "mlp"
    return best_name


def run_dir_name(val_split: float, seed: int) -> str:
    """Folder/file naming derived from the split params, so sweep runs don't
    clobber each other."""
    return f"valsplit{val_split:.2f}_seed{seed}"


def run_once(data: dict, val_split: float, seed: int, out_dir: Path) -> dict:
    """Fit/score/project/plot for one (val_split, seed) combo. Saves plots and
    a metrics.json into out_dir; returns the same metrics as a dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Step 1/2: pool ALL species for fitting (proton + kaon + muon), so the
    # regressors see the full latent manifold rather than just the proton
    # region. Stratified train/val split. ──
    species_list = list(data.keys())
    Z_all = np.vstack([data[s][0] for s in species_list])
    Y_all = np.vstack([np.column_stack([data[s][1], data[s][2]]) for s in species_list])
    species_labels = np.concatenate([[s] * len(data[s][0]) for s in species_list])

    Z_tr, Z_val, Y_tr, Y_val, lbl_tr, lbl_val = train_test_split(
        Z_all, Y_all, species_labels,
        test_size=val_split, random_state=seed, stratify=species_labels,
    )
    print(f"\n=== {run_dir_name(val_split, seed)} ===")
    print(f"Pooled (all species) split ({(1 - val_split):.0%}/{val_split:.0%}, seed={seed}): "
          f"{len(Z_tr)} train / {len(Z_val)} val")
    split_sizes = {}
    for s in species_list:
        split_sizes[s] = {"train": int(np.sum(lbl_tr == s)), "val": int(np.sum(lbl_val == s))}
        print(f"  {s:6s}: {split_sizes[s]['train']} train / {split_sizes[s]['val']} val")

    results = fit_and_score(Z_tr, Y_tr, Z_val, Y_val)

    print("Held-out R^2 on pooled (all-species) validation split:")
    print(f"{'model':10s} {'mean_adc':>10s} {'solidity':>10s}")
    r2_table = {}
    for name in ["linear", "mlp", "rf"]:
        r2 = results[name]["r2"]
        r2_table[name] = {"mean_adc": r2[0], "solidity": r2[1]}
        print(f"{name:10s} {r2[0]:10.3f} {r2[1]:10.3f}")

    best_name = pick_best_model(results)
    print(f"Using '{best_name}' for the 2D projection (best/near-best held-out R^2).")

    if best_name == "rf":
        best_model = RFWrapper(results["rf"]["model"])
    else:
        best_model = results[best_name]["model"]

    # ── Step 3: project all species into the 2D physics plane ──
    projections = {}
    for species, (Z, mean_adc, solidity) in data.items():
        Y_pred = best_model.predict(Z)
        projections[species] = (Y_pred, mean_adc, solidity)

    # ── Step 4: scatter plots, styled to match the linear mean_adc/solidity plot ──
    # x-axis = predicted solidity, y-axis = predicted mean ADC (matches
    # scripts/extra/plot_mean_adc_vs_solidity.py's axis convention).
    all_y = np.concatenate([Y_pred[:, 0] for Y_pred, _, _ in projections.values()])
    ylim = (all_y.min() - 1, all_y.max() + 1)  # shared y-range so per-species panels are comparable

    def _panel(ax, species, Y_pred):
        ax.scatter(
            Y_pred[:, 1], Y_pred[:, 0],
            s=8, alpha=0.35, color=COLOURS[species],
            edgecolors="none", label=f"{species} (n={len(Y_pred)})",
        )
        ax.set_xlabel("Predicted solidity")
        ax.set_ylabel("Predicted mean ADC")
        ax.set_xlim(0, 1)
        ax.set_ylim(*ylim)
        ax.spines[["top", "right"]].set_visible(False)

    fig, ax = plt.subplots(figsize=(6, 5))
    for species, (Y_pred, _, _) in projections.items():
        _panel(ax, species, Y_pred)
    ax.set_title(f"Nonlinear physics-plane projection ({best_name}, all-species-fit)",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

    path = out_dir / f"physics_plane_nonlinear_{best_name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    # ── per-species panels (same model/projection, one species each) ──
    for species, (Y_pred, _, _) in projections.items():
        fig, ax = plt.subplots(figsize=(6, 5))
        _panel(ax, species, Y_pred)
        ax.set_title(f"{species.capitalize()} — nonlinear physics-plane projection ({best_name})",
                     fontsize=13, fontweight="bold", color=COLOURS[species])
        ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

        species_path = out_dir / f"physics_plane_nonlinear_{best_name}_{species}.png"
        fig.savefig(species_path)
        plt.close(fig)
        print(f"Saved {species_path}")

    # ── pairwise panels: proton+kaon only, proton+muon only ──
    for pair in [("proton", "kaon"), ("proton", "muon")]:
        if not all(s in projections for s in pair):
            continue
        fig, ax = plt.subplots(figsize=(6, 5))
        for species in pair:
            _panel(ax, species, projections[species][0])
        ax.set_title(f"{pair[0].capitalize()} + {pair[1].capitalize()} — "
                     f"nonlinear physics-plane projection ({best_name})",
                     fontsize=13, fontweight="bold")
        ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

        pair_path = out_dir / f"physics_plane_nonlinear_{best_name}_{pair[0]}_{pair[1]}.png"
        fig.savefig(pair_path)
        plt.close(fig)
        print(f"Saved {pair_path}")

    # ── Step 5: correlation between projected and true feature values ──
    print("Pearson correlation (predicted vs. true), per species:")
    correlations = {}
    for species, (Y_pred, mean_adc, solidity) in projections.items():
        finite = np.isfinite(mean_adc) & np.isfinite(solidity)
        r_adc, _ = pearsonr(Y_pred[finite, 0], mean_adc[finite])
        r_sol, _ = pearsonr(Y_pred[finite, 1], solidity[finite])
        correlations[species] = {"mean_adc": r_adc, "solidity": r_sol}
        print(f"  {species:6s}: mean_adc r={r_adc:.3f}   solidity r={r_sol:.3f}")

    train_n = sum(v["train"] for v in split_sizes.values())
    val_n   = sum(v["val"] for v in split_sizes.values())
    metrics = {
        "val_split": val_split,
        "random_seed": seed,
        "split_sizes": split_sizes,
        "train_n": train_n,
        "val_n": val_n,
        "r2": r2_table,
        "best_model": best_name,
        "correlations": correlations,
    }
    metrics_path = out_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved {metrics_path}")

    print(
        "CAVEAT: the nonlinear projection above is fit on POOLED data from all "
        "three species (not protons alone), so the regressor has directly seen "
        "kaon/muon latent-to-feature pairs during training. Any species "
        "separation visible in the plot is therefore partly by construction, "
        "not purely evidence of latent structure discovered from protons -- "
        "treat it as a visualisation aid, not as evidence for the linear-probe "
        "separability claims made elsewhere."
    )

    return metrics


def plot_sweep_performance(summary_df: pd.DataFrame, sweep_root: Path) -> None:
    """Plot held-out R^2 vs. training-set size, to see how performance scales
    with data amount.

    Metric choice: held-out R^2 of the model actually used for the projection
    (r2_best_*), not the raw Pearson correlations printed elsewhere. R^2 is
    the right metric here for two reasons: (1) it's computed on the held-out
    validation split, i.e. it is exactly the generalisation performance that
    --val-split trades off against training-set size, whereas the
    predicted-vs-true correlations are computed over the *entire* species
    population (train+val combined) and so don't isolate the data-amount
    effect as cleanly; and (2) R^2 penalises scale/bias errors that a
    correlation coefficient can mask (a regressor that's off by a constant
    factor can still have r=1 but a poor R^2). Since 'best_model' can differ
    per sweep point in principle, r2_best_* always reflects the model that
    was actually deployed for that run's plots, making it the fair
    apples-to-apples y-axis across the sweep.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5))

    for tgt, marker, colour in [("mean_adc", "o", "#4C78A8"), ("solidity", "s", "#F58518")]:
        col = f"r2_best_{tgt}"
        # average across seeds at each train_n, show individual seeds as faint points
        grouped = summary_df.groupby("train_n")[col].agg(["mean", "std"]).sort_index()
        ax.errorbar(
            grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0),
            marker=marker, color=colour, label=tgt, capsize=3, linewidth=1.5, markersize=6,
        )
        ax.scatter(summary_df["train_n"], summary_df[col], color=colour, alpha=0.3, s=15, zorder=0)

    ax.set_xlabel("Training-set size (pooled, all species)")
    ax.set_ylabel("Held-out R$^2$ (best model)")
    ax.set_title("Sweep: performance vs. data amount", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75")

    path = sweep_root / "sweep_performance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=str(PROJECT_ROOT / "configs" /
                    "run_0066_model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p.yaml"),
        help="Path to model YAML config (default: latent-8 proton/kaon model, which has muon latents too)",
    )
    parser.add_argument(
        "--features",
        default="/Volumes/easystore/proton-kaon/features/features.pkl",
        help="Path to features.pkl",
    )
    parser.add_argument("--out-dir", default=None,
                        help="Base output directory (default: figs/physics_plane_nonlinear)")
    parser.add_argument(
        "--val-split", type=float, nargs="+", default=[0.1],
        help="Held-out validation fraction(s) for the pooled train/val split (default: 0.1). "
             "Pass multiple values (with --random-seed) to sweep.",
    )
    parser.add_argument(
        "--random-seed", type=int, nargs="+", default=[42],
        help="Random seed(s) for the train/val split (default: 42). "
             "Pass multiple values to sweep every val-split x seed combo.",
    )
    args = parser.parse_args()

    base_out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading latents + features for config {args.config} ...")
    data = load_latents_and_features(cfg, args.features)

    is_sweep = len(args.val_split) > 1 or len(args.random_seed) > 1
    sweep_root = base_out_dir / "sweep" if is_sweep else base_out_dir

    summary_rows = []
    for val_split in args.val_split:
        for seed in args.random_seed:
            run_out_dir = sweep_root / run_dir_name(val_split, seed) if is_sweep else sweep_root
            metrics = run_once(data, val_split, seed, run_out_dir)
            best = metrics["best_model"]
            summary_rows.append({
                "val_split": val_split,
                "random_seed": seed,
                "train_n": metrics["train_n"],
                "val_n": metrics["val_n"],
                "best_model": best,
                # held-out R^2 of the model actually used for the projection --
                # the metric to track "performance vs. data amount" with (see
                # plot_sweep_performance's docstring for why).
                "r2_best_mean_adc": metrics["r2"][best]["mean_adc"],
                "r2_best_solidity": metrics["r2"][best]["solidity"],
                **{f"r2_{model}_{tgt}": metrics["r2"][model][tgt]
                   for model in ["linear", "mlp", "rf"] for tgt in ["mean_adc", "solidity"]},
                **{f"corr_{species}_{tgt}": metrics["correlations"][species][tgt]
                   for species in metrics["correlations"] for tgt in ["mean_adc", "solidity"]},
            })

    if is_sweep:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = sweep_root / "sweep_summary.csv"
        if summary_path.exists():
            existing_df = pd.read_csv(summary_path)
            summary_df = pd.concat([existing_df, summary_df], ignore_index=True)
            summary_df = summary_df.drop_duplicates(
                subset=["val_split", "random_seed"], keep="last"
            )
        summary_df.to_csv(summary_path, index=False)
        print(f"\n=== Sweep summary ({len(summary_rows)} new runs, "
              f"{len(summary_df)} total) saved to {summary_path} ===")
        print(summary_df.to_string(index=False))

        plot_sweep_performance(summary_df, sweep_root)


if __name__ == "__main__":
    main()
