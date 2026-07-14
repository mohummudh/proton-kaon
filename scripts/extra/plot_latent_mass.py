#!/usr/bin/env python3
"""
scripts/extra/plot_latent_mass.py

Nonlinear projection of the 8D VAE latent space onto beamline mass
(`beamline_mass` from docs/picky+match.csv -- the spectrometer-measured mass,
the closest thing to ground truth in the project, see docs/METHODS.md §20),
fit on ALL THREE species pooled (proton + kaon + muon) and then applied back
to each species individually. Structurally mirrors
scripts/extra/plot_physics_plane_nonlinear.py, but with a single scalar
target (mass) instead of the 2D (mean_adc, solidity) physics plane.

Three regressors are fit on a stratified train/val split of the pooled,
all-species data and compared by held-out R^2:
    - LinearRegression            (baseline)
    - MLPRegressor (1 hidden layer, standardised inputs)
    - RandomForestRegressor
The split fraction and seed are controlled via --val-split/--random-seed
(default 90/10, seed 42).

The best-scoring model is then used to project Z_proton, Z_kaon, and Z_muon
into predicted mass. For each (val_split, seed) run, this saves:
  - a predicted-vs-true scatter (combined, per-species, and two pairwise
    panels: proton+kaon, proton+muon), with a y=x reference line
  - a predicted-mass histogram per species, with PDG mass reference lines,
    analogous to the notebooks/csda-length.ipynb beamline-mass spectrum
  - metrics.json recording split sizes, R^2 per model, the chosen best
    model, and predicted-vs-true Pearson correlations per species

--val-split and --random-seed each accept multiple values, which triggers a
sweep over every (val_split, seed) combination, mirroring
plot_physics_plane_nonlinear.py's sweep mechanics (sweep_summary.csv +
sweep_performance.png under figs/latent_mass/sweep/).

CAVEAT: this projection uses a flexible nonlinear regressor (MLP or random
forest) fit on pooled data from all species, so it is no longer a pure
"proton-trained, applied out-of-distribution" probe -- the regressor has seen
kaon and muon (latent, mass) pairs directly during training. Species
identity is trivially separable in this latent space (see
scripts/analyse_latents.py's logistic probe), and mass is strongly
species-correlated (proton ~938 MeV, kaon ~494 MeV, muon ~106 MeV), so a
pooled regressor can score a high R^2 largely by inferring species from the
latent and reporting that species' mean mass -- NOT by resolving
within-species mass variation. Treat this plot as a visualisation aid only;
it is not evidence that the latent encodes mass beyond what species identity
already implies.

Usage:
    python scripts/extra/plot_latent_mass.py \
        --config configs/run_0066_model_vae_latent8_ch32_64_128_256_beta0.5_lr0.001_epoch200_actrelu_kern5_stride2_pad2_hw48x48_txlog1p.yaml

    # sweep over val-split fractions and seeds:
    python scripts/extra/plot_latent_mass.py \
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
OUT_DIR = PROJECT_ROOT / "figs" / "latent_mass"
TARGET_NAME = "beamline_mass"

COLOURS = {
    "proton": "#4C78A8",
    "kaon":   "#F58518",
    "muon":   "#9467BD",
}

# PDG masses (MeV/c^2), for reference lines on the predicted-mass histogram.
PDG_MASS = {
    "proton": 938.272,
    "kaon":   493.677,
    "muon":   105.658,
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


def load_latents_and_mass(cfg: dict, features_path: str, picky_csv_path: str):
    """Returns dict of species -> (Z, mass), positionally aligned (see
    scripts/analyse_latents.py::load_features_and_splits for the same
    alignment convention used across the analysis pipeline). `mass` is
    `beamline_mass` from picky+match.csv, left-merged onto each species'
    feature rows by the (run, subrun, event) event key -- the same merge
    convention scripts/compute_features.py uses for log-likelihoods."""
    model_name = build_model_name(cfg)
    inference_dir = Path(cfg["output"]["inference_dir"]) / model_name

    train_latents = np.load(inference_dir / "train.npz")["latents"]
    val_latents   = np.load(inference_dir / "val.npz")["latents"]
    kaon_latents  = np.load(inference_dir / "kaon.npz")["latents"]
    muon_npz = inference_dir / "muon.npz"
    muon_latents = np.load(muon_npz)["latents"] if muon_npz.exists() else None

    features = pd.read_pickle(features_path)
    split_p = np.load(Path(cfg["output"]["splits_dir"]) / "split_p.npz")
    picky = pd.read_csv(picky_csv_path)[["run", "subrun", "event", "beamline_mass"]]

    all_proton = features[features["particle_type"] == "proton"].reset_index(drop=True)
    all_kaon   = features[features["particle_type"] == "kaon"].reset_index(drop=True)
    all_muon   = features[features["particle_type"] == "muon"].reset_index(drop=True)

    # Reassemble proton features in the same order as vstack([train, val]) latents,
    # using the saved split indices (proton latents are stored train-then-val).
    Z_proton = np.vstack([train_latents, val_latents])
    proton_order = np.concatenate([split_p["train_idx"], split_p["val_idx"]])
    proton_feats = all_proton.iloc[proton_order].reset_index(drop=True)

    def _attach_mass(feats_df):
        merged = feats_df[["run", "subrun", "event"]].merge(
            picky, on=["run", "subrun", "event"], how="left"
        )
        return merged["beamline_mass"].to_numpy()

    data = {
        "proton": (Z_proton, _attach_mass(proton_feats)),
        "kaon":   (kaon_latents, _attach_mass(all_kaon)),
    }
    if muon_latents is not None and len(muon_latents) == len(all_muon):
        data["muon"] = (muon_latents, _attach_mass(all_muon))
    else:
        print("  (muon latents unavailable or misaligned with features -- skipping muons)")

    for species, (_, mass) in data.items():
        n_matched = np.isfinite(mass).sum()
        print(f"  {species:6s}: {n_matched}/{len(mass)} matched to {TARGET_NAME}")

    return data


def fit_and_score(Z_tr, y_tr, Z_val, y_val):
    """Fit LinearRegression, MLPRegressor, and RandomForestRegressor on the
    pooled train split; return held-out R^2 for each."""
    results = {}

    lin = LinearRegression().fit(Z_tr, y_tr)
    results["linear"] = {"model": lin, "r2": r2_score(y_val, lin.predict(Z_val))}

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
    ]).fit(Z_tr, y_tr)
    results["mlp"] = {"model": mlp, "r2": r2_score(y_val, mlp.predict(Z_val))}

    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    rf.fit(Z_tr, y_tr)
    results["rf"] = {"model": rf, "r2": r2_score(y_val, rf.predict(Z_val))}

    return results


def pick_best_model(results):
    """Pick the model with the best held-out R^2 (ties/near-ties default to
    the MLP, per the same convention as plot_physics_plane_nonlinear.py)."""
    r2s = {name: r["r2"] for name, r in results.items()}
    best_name = max(r2s, key=r2s.get)
    if r2s["mlp"] >= r2s[best_name] - 0.01:
        best_name = "mlp"
    return best_name


def run_dir_name(val_split: float, seed: int) -> str:
    return f"valsplit{val_split:.2f}_seed{seed}"


def run_once(data: dict, val_split: float, seed: int, out_dir: Path) -> dict:
    """Fit/score/project/plot for one (val_split, seed) combo. Saves plots and
    a metrics.json into out_dir; returns the same metrics as a dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── pool ALL species (with a matched mass) for fitting; stratified
    # train/val split. ──
    species_list = list(data.keys())
    Z_parts, y_parts, label_parts = [], [], []
    for s in species_list:
        Z, mass = data[s]
        finite = np.isfinite(mass)
        Z_parts.append(Z[finite])
        y_parts.append(mass[finite])
        label_parts.append([s] * finite.sum())
    Z_all = np.vstack(Z_parts)
    y_all = np.concatenate(y_parts)
    species_labels = np.concatenate(label_parts)

    Z_tr, Z_val, y_tr, y_val, lbl_tr, lbl_val = train_test_split(
        Z_all, y_all, species_labels,
        test_size=val_split, random_state=seed, stratify=species_labels,
    )
    print(f"\n=== {run_dir_name(val_split, seed)} ===")
    print(f"Pooled (all species) split ({(1 - val_split):.0%}/{val_split:.0%}, seed={seed}): "
          f"{len(Z_tr)} train / {len(Z_val)} val")
    split_sizes = {}
    for s in species_list:
        split_sizes[s] = {"train": int(np.sum(lbl_tr == s)), "val": int(np.sum(lbl_val == s))}
        print(f"  {s:6s}: {split_sizes[s]['train']} train / {split_sizes[s]['val']} val")

    results = fit_and_score(Z_tr, y_tr, Z_val, y_val)

    print("Held-out R^2 on pooled (all-species) validation split:")
    r2_table = {}
    for name in ["linear", "mlp", "rf"]:
        r2_table[name] = results[name]["r2"]
        print(f"  {name:10s} {results[name]['r2']:10.3f}")

    best_name = pick_best_model(results)
    best_model = results[best_name]["model"]
    print(f"Using '{best_name}' for the mass projection (best/near-best held-out R^2).")

    # ── project all species into predicted mass ──
    projections = {}
    for species, (Z, mass) in data.items():
        y_pred = best_model.predict(Z)
        projections[species] = (y_pred, mass)

    # ── predicted-vs-true scatter, styled to match plot_physics_plane_nonlinear.py ──
    all_vals = np.concatenate(
        [np.concatenate([y_pred, mass[np.isfinite(mass)]]) for y_pred, mass in projections.values()]
    )
    lims = (all_vals.min() - 20, all_vals.max() + 20)

    def _scatter_panel(ax, species, y_pred, mass):
        finite = np.isfinite(mass)
        ax.scatter(
            mass[finite], y_pred[finite],
            s=8, alpha=0.35, color=COLOURS[species],
            edgecolors="none", label=f"{species} (n={finite.sum()})",
        )
        ax.set_xlabel(f"True {TARGET_NAME} (MeV/c$^2$)")
        ax.set_ylabel(f"Predicted {TARGET_NAME} (MeV/c$^2$)")
        ax.set_xlim(*lims)
        ax.set_ylim(*lims)
        ax.plot(lims, lims, color="0.5", linestyle="--", linewidth=1, zorder=0)
        ax.spines[["top", "right"]].set_visible(False)

    fig, ax = plt.subplots(figsize=(6, 5.5))
    for species, (y_pred, mass) in projections.items():
        _scatter_panel(ax, species, y_pred, mass)
    ax.set_title(f"Latent → mass, predicted vs. true ({best_name}, all-species-fit)",
                 fontsize=13, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

    path = out_dir / f"latent_mass_scatter_{best_name}.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"Saved {path}")

    for species, (y_pred, mass) in projections.items():
        fig, ax = plt.subplots(figsize=(6, 5.5))
        _scatter_panel(ax, species, y_pred, mass)
        ax.set_title(f"{species.capitalize()} — latent → mass, predicted vs. true ({best_name})",
                     fontsize=13, fontweight="bold", color=COLOURS[species])
        ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

        species_path = out_dir / f"latent_mass_scatter_{best_name}_{species}.png"
        fig.savefig(species_path)
        plt.close(fig)
        print(f"Saved {species_path}")

    for pair in [("proton", "kaon"), ("proton", "muon")]:
        if not all(s in projections for s in pair):
            continue
        fig, ax = plt.subplots(figsize=(6, 5.5))
        for species in pair:
            y_pred, mass = projections[species]
            _scatter_panel(ax, species, y_pred, mass)
        ax.set_title(f"{pair[0].capitalize()} + {pair[1].capitalize()} — "
                     f"latent → mass, predicted vs. true ({best_name})",
                     fontsize=13, fontweight="bold")
        ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)

        pair_path = out_dir / f"latent_mass_scatter_{best_name}_{pair[0]}_{pair[1]}.png"
        fig.savefig(pair_path)
        plt.close(fig)
        print(f"Saved {pair_path}")

    # ── predicted-mass histogram (a "mass spectrum" from the latent), with
    # PDG reference lines -- analogous to notebooks/csda-length.ipynb ──
    fig, ax = plt.subplots(figsize=(7, 5))
    bins = np.linspace(lims[0], lims[1], 60)
    for species, (y_pred, _) in projections.items():
        ax.hist(y_pred, bins=bins, alpha=0.5, color=COLOURS[species],
                 label=f"{species} (n={len(y_pred)})", density=True)
        ax.axvline(PDG_MASS[species], color=COLOURS[species], linestyle="--", linewidth=1.5)
    ax.set_xlabel(f"Predicted {TARGET_NAME} (MeV/c$^2$)")
    ax.set_ylabel("Density")
    ax.set_title(f"Predicted mass spectrum from latent ({best_name}, all-species-fit)\n"
                 "dashed lines = PDG mass", fontsize=12, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75")

    hist_path = out_dir / f"latent_mass_hist_{best_name}.png"
    fig.savefig(hist_path)
    plt.close(fig)
    print(f"Saved {hist_path}")

    # ── correlation + error between projected and true mass ──
    print("Pearson correlation (predicted vs. true) and mean predicted mass, per species:")
    correlations = {}
    for species, (y_pred, mass) in projections.items():
        finite = np.isfinite(mass)
        r, _ = pearsonr(y_pred[finite], mass[finite])
        rmse = float(np.sqrt(np.mean((y_pred[finite] - mass[finite]) ** 2)))
        correlations[species] = {
            "pearson_r": r,
            "rmse": rmse,
            "pred_mean": float(np.mean(y_pred)),
            "true_mean": float(np.mean(mass[finite])),
            "pdg_mass": PDG_MASS[species],
        }
        print(f"  {species:6s}: r={r:.3f}  rmse={rmse:7.2f}  "
              f"pred_mean={np.mean(y_pred):7.2f}  true_mean={np.mean(mass[finite]):7.2f}  "
              f"pdg={PDG_MASS[species]:.2f}")

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
        "CAVEAT: the mass projection above is fit on POOLED data from all three "
        "species (not protons alone), and mass is strongly species-correlated "
        "(proton ~938, kaon ~494, muon ~106 MeV). Since species identity is "
        "trivially separable in this latent space, a high R^2 here can largely "
        "reflect the regressor inferring species and reporting its mean mass, "
        "not resolving within-species mass variation. Treat this plot as a "
        "visualisation aid only, not as evidence the latent encodes mass beyond "
        "species identity."
    )

    return metrics


def plot_sweep_performance(summary_df: pd.DataFrame, sweep_root: Path) -> None:
    """Plot held-out R^2 (of the deployed model) vs. training-set size."""
    fig, ax = plt.subplots(figsize=(6.5, 5))

    grouped = summary_df.groupby("train_n")["r2_best"].agg(["mean", "std"]).sort_index()
    ax.errorbar(
        grouped.index, grouped["mean"], yerr=grouped["std"].fillna(0),
        marker="o", color="#4C78A8", capsize=3, linewidth=1.5, markersize=6,
    )
    ax.scatter(summary_df["train_n"], summary_df["r2_best"], color="#4C78A8", alpha=0.3, s=15, zorder=0)

    ax.set_xlabel("Training-set size (pooled, all species)")
    ax.set_ylabel("Held-out R$^2$ (best model)")
    ax.set_title("Sweep: latent → mass performance vs. data amount", fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)

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
    parser.add_argument(
        "--picky-csv",
        default="/Volumes/easystore/proton-kaon/docs/picky+match.csv",
        help="Path to picky+match.csv (source of beamline_mass)",
    )
    parser.add_argument("--out-dir", default=None,
                        help="Base output directory (default: figs/latent_mass)")
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

    print(f"Loading latents + mass for config {args.config} ...")
    data = load_latents_and_mass(cfg, args.features, args.picky_csv)

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
                "r2_best": metrics["r2"][best],
                **{f"r2_{model}": metrics["r2"][model] for model in ["linear", "mlp", "rf"]},
                **{f"corr_{species}": metrics["correlations"][species]["pearson_r"]
                   for species in metrics["correlations"]},
                **{f"rmse_{species}": metrics["correlations"][species]["rmse"]
                   for species in metrics["correlations"]},
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
