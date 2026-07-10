#!/usr/bin/env python3
"""
scripts/extra/plot_physics_plane_nonlinear.py

Nonlinear projection of the 8D VAE latent space onto a 2D "physics plane"
(mean_adc, solidity), fit on ALL THREE species pooled (proton + kaon + muon)
and then applied back to each species individually.

Three regressors are fit on a 90/10 train/val split of the pooled, all-species
data (stratified by species) and compared by held-out R^2 per target dimension:
    - LinearRegression            (baseline)
    - MLPRegressor (1 hidden layer, standardised inputs)
    - RandomForestRegressor       (one per target)

The best-scoring model (per the notes printed at the end of step 2) is then
used to project Z_proton, Z_kaon, and Z_muon into the 2D plane, and a scatter
plot coloured by species is saved.

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
"""

import argparse
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
    parser.add_argument("--out-dir", default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"Loading latents + features for config {args.config} ...")
    data = load_latents_and_features(cfg, args.features)

    # ── Step 1/2: pool ALL species for fitting (proton + kaon + muon), so the
    # regressors see the full latent manifold rather than just the proton
    # region. 90/10 train/val split, stratified by species. ──
    species_list = list(data.keys())
    Z_all = np.vstack([data[s][0] for s in species_list])
    Y_all = np.vstack([np.column_stack([data[s][1], data[s][2]]) for s in species_list])
    species_labels = np.concatenate([[s] * len(data[s][0]) for s in species_list])

    Z_tr, Z_val, Y_tr, Y_val, lbl_tr, lbl_val = train_test_split(
        Z_all, Y_all, species_labels, test_size=0.1, random_state=42, stratify=species_labels,
    )
    print(f"\nPooled (all species) split: {len(Z_tr)} train / {len(Z_val)} val")
    for s in species_list:
        print(f"  {s:6s}: {np.sum(lbl_tr == s)} train / {np.sum(lbl_val == s)} val")

    results = fit_and_score(Z_tr, Y_tr, Z_val, Y_val)

    print("\nHeld-out R^2 on pooled (all-species) validation split:")
    print(f"{'model':10s} {'mean_adc':>10s} {'solidity':>10s}")
    for name in ["linear", "mlp", "rf"]:
        r2 = results[name]["r2"]
        print(f"{name:10s} {r2[0]:10.3f} {r2[1]:10.3f}")

    best_name = pick_best_model(results)
    print(f"\nUsing '{best_name}' for the 2D projection (best/near-best held-out R^2).")

    if best_name == "rf":
        best_model = RFWrapper(results["rf"]["model"])
    else:
        best_model = results[best_name]["model"]

    # ── Step 3: project all species into the 2D physics plane ──
    projections = {}
    for species, (Z, mean_adc, solidity) in data.items():
        Y_pred = best_model.predict(Z)
        projections[species] = (Y_pred, mean_adc, solidity)

    # ── Step 4: scatter plot, styled to match the linear mean_adc/solidity plot ──
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
    print(f"\nSaved {path}")

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
    print("\nPearson correlation (predicted vs. true), per species:")
    for species, (Y_pred, mean_adc, solidity) in projections.items():
        finite = np.isfinite(mean_adc) & np.isfinite(solidity)
        r_adc, _  = pearsonr(Y_pred[finite, 0], mean_adc[finite])
        r_sol, _  = pearsonr(Y_pred[finite, 1], solidity[finite])
        print(f"  {species:6s}: mean_adc r={r_adc:.3f}   solidity r={r_sol:.3f}")

    print(
        "\nCAVEAT: the nonlinear projection above is fit on POOLED data from all "
        "three species (not protons alone), so the regressor has directly seen "
        "kaon/muon latent-to-feature pairs during training. Any species "
        "separation visible in the plot is therefore partly by construction, "
        "not purely evidence of latent structure discovered from protons -- "
        "treat it as a visualisation aid, not as evidence for the linear-probe "
        "separability claims made elsewhere."
    )


if __name__ == "__main__":
    main()
