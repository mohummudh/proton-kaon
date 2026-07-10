#!/usr/bin/env python3
"""
scripts/extra/linear_latent_projection.py

Fit a strictly linear map from the VAE's n-dim latent space down to a 2D
"physics plane" whose axes are (mean_adc, solidity), trained on all species
(protons, kaons, muons) pooled together, then apply it back to each species
and check how well it fits.

    z (n,)  --project()-->  (mean_adc_hat, solidity_hat)

project(Z) = Z @ W.T + b, with W (2 x n) and b (2,) built directly from the
coefficients of two independent sklearn LinearRegression fits (one per
target). No UMAP, no nonlinear models — this is meant to be interpretable.

Usage:
    python scripts/extra/linear_latent_projection.py --config configs/run_0003_....yaml
    python scripts/extra/linear_latent_projection.py --config configs/run_0003_....yaml \
        --features /Volumes/easystore/proton-kaon/features/features.pkl
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_FEATURES = Path("/Volumes/easystore/proton-kaon/features/features.pkl")
OUT_DIR = PROJECT_ROOT / "figs" / "linear_latent_projection"

COLOURS = {
    "proton": "#4C78A8",
    "kaon":   "#F58518",
    "muon":   "#9467BD",
}

plt.rcParams.update({
    "font.family":     "serif",
    "font.serif":      ["Times New Roman", "DejaVu Serif"],
    "font.size":       12,
    "axes.labelsize":  12,
    "legend.fontsize": 10,
    "figure.dpi":      150,
    "savefig.dpi":     150,
    "savefig.bbox":    "tight",
})


# ── config / model-name helpers (mirrors scripts/analyse_latents.py) ─────────

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


def load_species_split(cfg: dict, model_name: str):
    """All-species models: per-species train/val indices saved by run_inference."""
    if cfg["data"].get("proton") != "all":
        return None
    path = Path(cfg["output"]["inference_dir"]) / model_name / "species_split.npz"
    return np.load(path) if path.exists() else None


# ── load latents + matching physical features for each species ──────────────

def load_data(cfg: dict, features_path: Path):
    model_name = build_model_name(cfg)
    inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
    print(f"Model: {model_name}")
    print(f"Inference dir: {inference_dir}")

    train_latents = np.load(inference_dir / "train.npz")["latents"]
    val_latents   = np.load(inference_dir / "val.npz")["latents"]
    kaon_latents  = np.load(inference_dir / "kaon.npz")["latents"]
    muon_npz = inference_dir / "muon.npz"
    muon_latents = np.load(muon_npz)["latents"] if muon_npz.exists() else None

    feat_df = pd.read_pickle(features_path)

    ss = load_species_split(cfg, model_name)
    all_proton = feat_df[feat_df["particle_type"] == "proton"]
    if ss is not None:
        train_idx, val_idx = ss["p_train_idx"], ss["p_val_idx"]
    else:
        idx = np.load(Path(cfg["output"]["splits_dir"]) / "split_p.npz")
        train_idx, val_idx = idx["train_idx"], idx["val_idx"]
    train_features = all_proton.iloc[train_idx]
    val_features   = all_proton.iloc[val_idx]

    Z_proton = np.vstack([train_latents, val_latents])
    proton_features = pd.concat([train_features, val_features], ignore_index=True)
    assert len(Z_proton) == len(proton_features), \
        f"proton latents ({len(Z_proton)}) != proton features ({len(proton_features)})"

    kaon_features = feat_df[feat_df["particle_type"] == "kaon"]
    assert len(kaon_latents) == len(kaon_features), \
        f"kaon latents ({len(kaon_latents)}) != kaon features ({len(kaon_features)})"
    Z_kaon = kaon_latents

    Z_muon, muon_features = None, None
    if muon_latents is not None:
        muon_features = feat_df[feat_df["particle_type"] == "muon"]
        if len(muon_latents) == len(muon_features):
            Z_muon = muon_latents
        else:
            print(f"  ⚠ muon latents ({len(muon_latents)}) != muon features "
                  f"({len(muon_features)}), skipping muons")

    data = {
        "proton": (Z_proton, proton_features["mean_adc"].to_numpy(), proton_features["solidity"].to_numpy()),
        "kaon":   (Z_kaon, kaon_features["mean_adc"].to_numpy(), kaon_features["solidity"].to_numpy()),
    }
    if Z_muon is not None:
        data["muon"] = (Z_muon, muon_features["mean_adc"].to_numpy(), muon_features["solidity"].to_numpy())
    return data


# ── linear projection ─────────────────────────────────────────────────────────

def pool_species(data: dict):
    """Stack Z/mean_adc/solidity across all species, keeping a species label
    per row so the held-out split can be stratified and scored per species."""
    species_order = list(data.keys())
    Z_all    = np.concatenate([data[s][0] for s in species_order], axis=0)
    adc_all  = np.concatenate([data[s][1] for s in species_order], axis=0)
    sol_all  = np.concatenate([data[s][2] for s in species_order], axis=0)
    labels   = np.concatenate([[s] * len(data[s][0]) for s in species_order])
    return Z_all, adc_all, sol_all, labels


def fit_projection(Z_all, adc_all, sol_all, labels, test_size=0.1, seed=42):
    """Fit reg_adc and reg_sol on a 90/10 split of the POOLED (all-species) data.

    The split is stratified by species so every species is represented in both
    the training partition and the held-out partition. Returns the fitted
    regressors plus overall and per-species held-out R².
    """
    Z_tr, Z_te, adc_tr, adc_te, sol_tr, sol_te, lab_tr, lab_te = train_test_split(
        Z_all, adc_all, sol_all, labels,
        test_size=test_size, random_state=seed, stratify=labels,
    )

    reg_adc = LinearRegression().fit(Z_tr, adc_tr)
    reg_sol = LinearRegression().fit(Z_tr, sol_tr)

    r2_adc = r2_score(adc_te, reg_adc.predict(Z_te))
    r2_sol = r2_score(sol_te, reg_sol.predict(Z_te))

    r2_by_species = {}
    for species in np.unique(lab_te):
        mask = lab_te == species
        r2_by_species[species] = (
            r2_score(adc_te[mask], reg_adc.predict(Z_te[mask])),
            r2_score(sol_te[mask], reg_sol.predict(Z_te[mask])),
        )

    return reg_adc, reg_sol, r2_adc, r2_sol, r2_by_species


def build_matrix(reg_adc, reg_sol):
    """Stack the two regressors' coefficients into W (2 x n_dims), b (2,)."""
    W = np.vstack([reg_adc.coef_, reg_sol.coef_])
    b = np.array([reg_adc.intercept_, reg_sol.intercept_])
    return W, b


def project(Z, W, b):
    """Apply the linear map directly: for each row z, returns W @ z + b."""
    return Z @ W.T + b


# ── plotting ───────────────────────────────────────────────────────────────

def plot_projection(projected, out_dir):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for species, proj in projected.items():
        ax.scatter(
            proj[:, 0], proj[:, 1],
            s=8, alpha=0.35, color=COLOURS[species],
            edgecolors="none", label=f"{species} (n={len(proj)})",
        )
    ax.set_xlabel("Projected mean ADC")
    ax.set_ylabel("Projected solidity")
    ax.set_title("Linear latent projection onto the physics plane", fontsize=13, fontweight="bold")
    ax.legend(frameon=True, framealpha=0.85, edgecolor="0.75", markerscale=2)
    ax.spines[["top", "right"]].set_visible(False)

    path = out_dir / "linear_latent_projection.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model config yaml")
    parser.add_argument("--features", default=str(DEFAULT_FEATURES),
                        help="Path to features.pkl (default: %(default)s)")
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: figs/linear_latent_projection)")
    parser.add_argument("--test-size", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    data = load_data(cfg, Path(args.features))
    for species, (Z, adc, sol) in data.items():
        print(f"  {species:6s}: Z={Z.shape}  mean_adc={adc.shape}  solidity={sol.shape}")

    Z_all, adc_all, sol_all, labels = pool_species(data)
    print(f"\nPooled (all species): Z={Z_all.shape}")

    # ── 1. fit on ALL species pooled together, held-out R² ──────────────────
    reg_adc, reg_sol, r2_adc, r2_sol, r2_by_species = fit_projection(
        Z_all, adc_all, sol_all, labels,
        test_size=args.test_size, seed=args.seed,
    )
    print(f"\nHeld-out R² (pooled {int((1 - args.test_size) * 100)}/{int(args.test_size * 100)} split):")
    print(f"  reg_adc: R² = {r2_adc:.4f}")
    print(f"  reg_sol: R² = {r2_sol:.4f}")
    print("  per species:")
    for species, (r2_a, r2_s) in r2_by_species.items():
        print(f"    {species:6s}  mean_adc R²={r2_a:.4f}  solidity R²={r2_s:.4f}")

    # ── 2. stack coefficients into W, b ──────────────────────────────────────
    W, b = build_matrix(reg_adc, reg_sol)
    print(f"\nW (2 x {W.shape[1]}):\n{W}")
    print(f"b (2,): {b}")

    # ── 3. verify project() matches the fitted regressors ───────────────────
    proj_check = project(Z_all, W, b)
    ref_check = np.column_stack([reg_adc.predict(Z_all), reg_sol.predict(Z_all)])
    # loose tolerance: Z/W are float32, so matmul order differences accumulate
    # small rounding error over n_dims terms — not a correctness issue.
    assert np.allclose(proj_check, ref_check, rtol=1e-3, atol=1e-3), \
        "project() does not match fitted regressors"
    print("\nproject() verified against LinearRegression.predict() — OK")

    # ── 4. apply project() to all species ────────────────────────────────────
    projected = {species: project(Z, W, b) for species, (Z, _, _) in data.items()}

    # ── 5. scatter plot ──────────────────────────────────────────────────────
    plot_projection(projected, out_dir)

    # ── 6. correlation of projected vs true feature, per species ────────────
    print("\nProjected vs true-feature correlations (generalisation check):")
    for species, (Z, mean_adc_true, solidity_true) in data.items():
        proj = projected[species]
        pr_adc, _ = pearsonr(proj[:, 0], mean_adc_true)
        sr_adc, _ = spearmanr(proj[:, 0], mean_adc_true)
        pr_sol, _ = pearsonr(proj[:, 1], solidity_true)
        sr_sol, _ = spearmanr(proj[:, 1], solidity_true)
        print(f"  {species:6s}  mean_adc: pearson={pr_adc:.3f} spearman={sr_adc:.3f}   "
              f"solidity: pearson={pr_sol:.3f} spearman={sr_sol:.3f}")

    print(f"\nDone. Plot saved to {out_dir}")


if __name__ == "__main__":
    main()
