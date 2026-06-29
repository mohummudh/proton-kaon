#!/usr/bin/env python3
"""
scripts/analyse_latents.py

Run latent-feature analyses for a given model YAML config and save all plots
to figs/latents-features/<model_name>/.

Usage:
    python scripts/analyse_latents.py --config configs/run_XXXX.yaml
    python scripts/analyse_latents.py --config configs/run_XXXX.yaml \
        --analyses correlation traversal logistic nonlinear
    python scripts/analyse_latents.py --config configs/run_XXXX.yaml \
        --features /path/to/features.pkl

Available analyses: correlation  traversal  logistic  nonlinear
Default: all four.
"""

import argparse
import sys
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":              11,
    "axes.labelsize":         11,
    "xtick.labelsize":        10,
    "ytick.labelsize":        10,
    "legend.fontsize":        9,
    "legend.title_fontsize":  9,
    "axes.linewidth":         0.6,
    "xtick.major.width":      0.6,
    "ytick.major.width":      0.6,
    "xtick.major.size":       3.0,
    "ytick.major.size":       3.0,
    "xtick.minor.visible":    False,
    "ytick.minor.visible":    False,
    "figure.dpi":             300,
    "savefig.dpi":            300,
    "savefig.format":         "pdf",
    "savefig.bbox":           "tight",
    "savefig.pad_inches":     0.02,
})


import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.configVAE import VAE  # noqa: E402

# ── feature groups ─────────────────────────────────────────────────────────────
# CALO = [
#     "total_adc", "mean_adc", "median_adc", "max_adc", "std_adc", "adc_entropy",
#     "bragg_peak_height", "max_ADC_position", "bragg_peak_ratio", "bragg_peak_to_median",
#     "end_vs_start_ratio", "last_quartile_mean", "first_quartile_mean",
#     "bragg_rise_slope", "peak_integral_fraction", "bragg_peak_width",
#     "profile_cv", "monotonic_rise_fraction", "relative_peak_energy",
#     "profile_skewness", "profile_kurtosis",
# ]
# TOPO = ["height", "n_pixels", "fill_fraction", "solidity", "n_local_maxima"]

CALO = ["mean_adc", "total_adc"]
TOPO = ["solidity"]

BLUE   = "#0077BB"   # Paul Tol palette — colourblind-safe
ORANGE = "#EE7733"
PURPLE = "#AA3377"
GREEN  = "#CC3311"

DOUBLE_COL = 6.875   # ~175 mm — double-column width for paper figures


# ── shared helpers ─────────────────────────────────────────────────────────────

def build_model_name(cfg: dict) -> str:
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
        f"_tx{cfg['data'].get('transform', 'none')}"
    )


def load_latents(cfg: dict, model_name: str) -> tuple:
    """Returns (train_latents, val_latents, kaon_latents) as numpy arrays."""
    inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
    train = np.load(inference_dir / "train.npz")
    val   = np.load(inference_dir / "val.npz")
    kaon  = np.load(inference_dir / "kaon.npz")
    return train["latents"], val["latents"], kaon["latents"]


def load_features_and_splits(cfg: dict, features_path: str) -> tuple:
    """Returns (features_df, index_npz) where index_npz has train_idx / val_idx."""
    split_path = Path(cfg["output"]["splits_dir"]) / "split_p.npz"
    features   = pd.read_pickle(features_path)
    index      = np.load(split_path)
    return features, index


def latent_dim_names(n: int) -> list:
    return [f"z{i}" for i in range(n)]


def make_mlp_pipeline() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(16, 16),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-3,
        )),
    ])


def permutation_importance_mlp(pipe, X, y, n_repeats=10, random_state=42):
    rng      = np.random.default_rng(random_state)
    baseline = r2_score(y, pipe.predict(X))
    n_dims   = X.shape[1]
    importances = np.zeros(n_dims)
    for i in range(n_dims):
        drops = []
        for _ in range(n_repeats):
            X_perm       = X.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            drops.append(baseline - r2_score(y, pipe.predict(X_perm)))
        importances[i] = np.mean(drops)
    return importances


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple:
    """Return (MAE, MSE, RMSE, MAPE). MAPE skips |y_true| < 1e-8 to avoid div/0."""
    mae  = float(mean_absolute_error(y_true, y_pred))
    mse  = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    nz   = np.abs(y_true) > 1e-8
    mape = float(np.mean(np.abs((y_true[nz] - y_pred[nz]) / y_true[nz])) * 100) if nz.sum() > 0 else np.nan
    return mae, mse, rmse, mape


def _particle_metrics(mask, y_true, y_oof_lin, y_oof_mlp):
    """Compute (mae, mse, rmse, mape) for linear and MLP on a particle sub-mask."""
    if mask.sum() < 2:
        return (np.nan, np.nan, np.nan, np.nan), (np.nan, np.nan, np.nan, np.nan)
    return (
        compute_regression_metrics(y_true[mask], y_oof_lin[mask]),
        compute_regression_metrics(y_true[mask], y_oof_mlp[mask]),
    )


def _savefig(path: Path) -> None:
    """Save figure as both PNG (raster) and PDF (vector) for publication."""
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


def make_legend_kwargs(loc: str = "best") -> dict:
    return dict(
        frameon=True, framealpha=0.85, edgecolor="0.75", loc=loc,
        handlelength=1.0, handletextpad=0.4, borderpad=0.5, labelspacing=0.35,
    )


# ── analysis 1: correlation ────────────────────────────────────────────────────

def run_correlation(cfg, model_name, features_path, out_dir):
    print("\n=== Correlation analysis ===")

    train_latents, val_latents, kaon_latents = load_latents(cfg, model_name)
    n_dims   = train_latents.shape[1]
    dim_cols = latent_dim_names(n_dims)

    features, index = load_features_and_splits(cfg, features_path)

    all_proton = features[features["particle_type"] == "proton"]
    all_kaon   = features[features["particle_type"] == "kaon"]

    # restrict to p/k only so muon rows (latent_z=0) don't bias Spearman
    pk_features = features[features["particle_type"].isin(["proton", "kaon"])].copy()

    latent_z = np.zeros((len(features), n_dims))
    latent_z[all_proton.index[index["train_idx"]], :] = train_latents
    latent_z[all_proton.index[index["val_idx"]],   :] = val_latents
    latent_z[all_kaon.index,                        :] = kaon_latents

    for j, col in enumerate(dim_cols):
        pk_features[col] = latent_z[pk_features.index, j]

    calo = [f for f in CALO if f in pk_features.columns]
    topo = [f for f in TOPO if f in pk_features.columns]
    all_feats = calo + topo

    # ── Spearman heatmap ──
    corr_matrix = np.zeros((len(all_feats), n_dims))
    for i, feat in enumerate(all_feats):
        for j, lat in enumerate(dim_cols):
            valid = pk_features[[feat, lat]].notna().all(axis=1)
            if valid.sum() > 2:
                rho, _ = spearmanr(pk_features.loc[valid, feat], pk_features.loc[valid, lat])
                corr_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, len(all_feats) * 0.65 + 1))
    sns.heatmap(
        corr_matrix,
        xticklabels=dim_cols,
        yticklabels=all_feats,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "Spearman $\\rho$"},
        annot=True, fmt=".2f", annot_kws={"size": 10},
        ax=ax,
    )
    ax.axhline(y=len(calo), color="black", linewidth=2)
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Feature")
    plt.tight_layout()
    _savefig(out_dir / "disentanglement_heatmap.png")
    plt.close()
    print("  saved disentanglement_heatmap.png")

    for j, lat in enumerate(dim_cols):
        c = np.abs(corr_matrix[:len(calo), j])
        t = np.abs(corr_matrix[len(calo):, j])
        print(f"  {lat}: calo mean |ρ|={c.mean():.3f}, topo mean |ρ|={t.mean():.3f}, "
              f"specificity={c.mean()/t.mean():.2f}x")

    # ── feature-to-feature correlation ──
    feat_corr = pk_features[all_feats].corr(method="spearman")
    fig_f, ax_f = plt.subplots(figsize=(DOUBLE_COL, len(all_feats) * 0.75 + 1))
    sns.heatmap(
        feat_corr,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "Spearman $\\rho$"},
        annot=True, fmt=".2f", annot_kws={"size": 10},
        ax=ax_f,
    )
    plt.tight_layout()
    _savefig(out_dir / "feature_correlation.png")
    plt.close()
    print("  saved feature_correlation.png")

    # ── per-particle disentanglement heatmaps & feature-to-feature correlations ──
    for ptag in ["proton", "kaon"]:
        sub = pk_features[pk_features["particle_type"] == ptag]
        if len(sub) < 3:
            continue

        corr_sub = np.zeros((len(all_feats), n_dims))
        for i, feat in enumerate(all_feats):
            for j, lat in enumerate(dim_cols):
                valid = sub[[feat, lat]].notna().all(axis=1)
                if valid.sum() > 2:
                    rho, _ = spearmanr(sub.loc[valid, feat], sub.loc[valid, lat])
                    corr_sub[i, j] = rho

        fig, ax = plt.subplots(figsize=(DOUBLE_COL, len(all_feats) * 0.65 + 1))
        sns.heatmap(
            corr_sub,
            xticklabels=dim_cols,
            yticklabels=all_feats,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "Spearman $\\rho$"},
            annot=True, fmt=".2f", annot_kws={"size": 10},
            ax=ax,
        )
        ax.axhline(y=len(calo), color="black", linewidth=2)
        ax.set_xlabel("Latent dimension")
        ax.set_ylabel("Feature")
        plt.tight_layout()
        fname = f"disentanglement_heatmap_{ptag}.png"
        _savefig(out_dir / fname)
        plt.close()
        print(f"  saved {fname}")

        feat_corr_sub = sub[all_feats].corr(method="spearman")
        fig_f, ax_f = plt.subplots(figsize=(DOUBLE_COL, len(all_feats) * 0.75 + 1))
        sns.heatmap(
            feat_corr_sub,
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            cbar_kws={"label": "Spearman $\\rho$"},
            annot=True, fmt=".2f", annot_kws={"size": 10},
            ax=ax_f,
        )
        plt.tight_layout()
        fname = f"feature_correlation_{ptag}.png"
        _savefig(out_dir / fname)
        plt.close()
        print(f"  saved {fname}")

    # ── variance decomposition (linear R² per dim per category) ──
    pk_latent_z = latent_z[pk_features.index]
    records = []
    for i in range(n_dims):
        z = pk_latent_z[:, i].reshape(-1, 1)
        for feat in all_feats:
            y = pk_features[feat].values
            mask = np.isfinite(y)
            if mask.sum() < 10:
                continue
            r2 = LinearRegression().fit(z[mask], y[mask]).score(z[mask], y[mask])
            records.append({
                "dim":      dim_cols[i],
                "feature":  feat,
                "r2":       max(r2, 0.0),
                "category": "calorimetry" if feat in calo else "topology",
            })

    var_df = pd.DataFrame(records)
    summary = (
        var_df.groupby(["dim", "category"])["r2"]
        .mean()
        .unstack("category")
        [["calorimetry", "topology"]]
    )
    print("\n  Variance decomposition (mean linear R²):")
    print(summary.round(4).to_string())

    fig, ax = plt.subplots(figsize=(7.0, 3.5))
    summary.plot(
        kind="bar", stacked=True, ax=ax,
        color=[BLUE, ORANGE],
        edgecolor="white", linewidth=0.5, width=0.5,
    )
    ax.set_xlabel("Latent dimension")
    ax.set_ylabel("Mean $R^2$ (linear, univariate)")
    ax.set_xticklabels(dim_cols, rotation=0)
    ax.legend(
        title="Feature category", frameon=True, framealpha=0.85,
        edgecolor="0.8", handlelength=1.0, handletextpad=0.4, borderpad=0.5,
    )
    ax.set_ylim(0, summary.values.sum(axis=1).max() * 1.15)
    for idx, (_, row) in enumerate(summary.iterrows()):
        total = row.sum()
        ax.text(idx, total + 0.001, f"{total:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    _savefig(out_dir / "variance_decomposition.png")
    plt.close()
    print("  saved variance_decomposition.png")


# ── analysis 2: latent traversal ──────────────────────────────────────────────

def run_traversal(cfg, model_name, out_dir):
    print("\n=== Latent traversal ===")

    device = torch.device("mps" if torch.backends.mps.is_available() else
                          "cuda" if torch.cuda.is_available() else "cpu")

    model = VAE(
        input_hw=tuple(cfg["model"]["input_hw"]),
        latent=cfg["model"]["latent"],
        channels=cfg["model"]["channels"],
        kernel=cfg["model"]["kernel"],
        stride=cfg["model"]["stride"],
        padding=cfg["model"]["padding"],
        activation=cfg["model"]["activation"],
        p_enc=cfg["model"]["dropout"],
    ).to(device)
    model.load_state_dict(torch.load(
        Path(cfg["output"]["dir"]) / (model_name + ".pt"), map_location=device
    ))
    model.eval()

    train_npz    = np.load(Path(cfg["output"]["inference_dir"]) / model_name / "train.npz")
    train_latents = train_npz["latents"]

    mu  = train_latents[10]
    sig = train_latents.std(axis=0)

    N_STEPS = 9
    N_DIMS  = cfg["model"]["latent"]
    CHANNEL = 0
    steps   = np.linspace(-2, 2, N_STEPS)

    fig, axes = plt.subplots(
        N_DIMS, N_STEPS,
        figsize=(N_STEPS * 1.8, N_DIMS * 2.2),
        constrained_layout=True,
    )
    if N_DIMS == 1:
        axes = axes[np.newaxis, :]

    with torch.no_grad():
        for i in range(N_DIMS):
            z_batch = np.tile(mu, (N_STEPS, 1))
            z_batch[:, i] = mu[i] + steps * sig[i]

            z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(device)
            recon    = model.decode(z_tensor).cpu().numpy()
            images   = recon[:, CHANNEL, :, :]

            vmax = images.max()
            for j in range(N_STEPS):
                ax = axes[i, j]
                ax.imshow(images[j], origin="lower", cmap="viridis", vmin=0, vmax=vmax)
                ax.axis("off")
                if j == 0:
                    ax.set_ylabel(rf"$z_{{{i}}}$", rotation=0, labelpad=28, va="center")
                if i == 0:
                    ax.set_title(f"{steps[j]:+.1f}$\\sigma$", fontsize=8)

    _savefig(out_dir / "latent_traversal.png")
    plt.close()
    print("  saved latent_traversal.png")


# ── analysis 3: logistic regression ───────────────────────────────────────────

def _auto_subsets(n_dims: int) -> dict:
    """
    Generate latent subsets automatically:
      - each individual dimension
      - all pairs (only when n_dims <= 8 to keep it tractable)
      - all dims combined
    """
    dims = list(range(n_dims))
    subsets = {}
    for i in dims:
        subsets[f"z{i}"] = [i]
    if n_dims <= 8:
        for a, b in combinations(dims, 2):
            subsets[f"z{a}+z{b}"] = [a, b]
    subsets[f"All (z0–z{n_dims-1})"] = dims
    return subsets


def _run_logistic_hardcases(
    val_latents, kaon_latents, val_features, kaon_features,
    kaon_orig_indices, val_idx, cfg, out_dir,
    suffix="", kaon_label="",
    muon_latents=None, muon_features_df=None,
    from_cache=False,
):
    """
    Run LR+MLP binary classification (protons vs kaon subset) and save:
      linear_probe{suffix}.png, hard_cases{suffix}.png, image panels.

    kaon_orig_indices: integer positions of these kaons in the full kaon image
        tensor (data['k']).  For all kaons pass np.arange(len(kaon_latents)).
    val_idx: split_p.npz val_idx — needed to retrieve hard-proton images.
    kaon_label: short string appended to titles, e.g. "picky kaons (p=1)".
    """
    n_dims = val_latents.shape[1]
    X = np.concatenate([val_latents, kaon_latents], axis=0)
    y = np.concatenate([
        np.zeros(len(val_latents)),
        np.ones(len(kaon_latents)),
    ])
    tag = f" [{kaon_label}]" if kaon_label else ""
    print(f"  Protons (val): {len(val_latents)}, Kaons: {len(kaon_latents)}{tag}")

    import pickle as _pickle
    _cache_pkl   = out_dir / f"cache_logistic{suffix}.pkl"
    _use_lr_cache = from_cache and _cache_pkl.exists()
    if _use_lr_cache:
        print(f"  (--from-cache: loading precomputed logistic results from {_cache_pkl.name})")

    features_df = pd.concat([val_features, kaon_features], ignore_index=True)
    subsets = _auto_subsets(n_dims)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    mlp_clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            activation="relu",
            solver="adam",
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=10,
        )),
    ])

    # ── LR on all latent subsets ──
    results    = {}
    pred_proba = {}
    for label, dims in subsets.items():
        if _use_lr_cache:
            continue
        X_sub = X[:, dims]
        proba = cross_val_predict(
            lr_pipeline, X_sub, y, cv=cv, method="predict_proba"
        )[:, 1]
        auc  = roc_auc_score(y, proba)
        acc  = accuracy_score(y, (proba > 0.5).astype(int))
        results[label]    = {"AUC": auc, "Accuracy": acc}
        pred_proba[label] = proba
        print(f"  {label:25s}  AUC={auc:.3f}  Acc={acc:.3f}")

    # ── MLP classifier on all dims ──
    # Train with balanced sample weights to match LR's class_weight="balanced".
    # cross_val_predict doesn't support fit_params routing in older sklearn,
    # so we manually loop over folds.
    if _use_lr_cache:
        with open(_cache_pkl, "rb") as _f:
            _lr_state = _pickle.load(_f)
        results    = _lr_state["results"]
        pred_proba = _lr_state["pred_proba"]
        mlp_proba  = _lr_state["mlp_proba"]
        mlp_auc    = _lr_state["mlp_auc"]
        mlp_acc    = _lr_state["mlp_acc"]
        print(f"  Loaded {len(results)} LR subsets from cache.  MLP AUC={mlp_auc:.3f}")
    else:
        print("\n  MLP classifier (all dims):")
        sample_weights = compute_sample_weight("balanced", y)
        mlp_proba = np.zeros(len(y))
        for train_idx, test_idx in cv.split(X, y):
            sw = sample_weights[train_idx]
            mlp_clf_pipeline.fit(X[train_idx], y[train_idx], mlp__sample_weight=sw)
            mlp_proba[test_idx] = mlp_clf_pipeline.predict_proba(X[test_idx])[:, 1]
        mlp_auc = roc_auc_score(y, mlp_proba)
        mlp_acc = accuracy_score(y, (mlp_proba > 0.5).astype(int))
        print(f"  MLP (all dims):           AUC={mlp_auc:.3f}  Acc={mlp_acc:.3f}")
        with open(_cache_pkl, "wb") as _f:
            _pickle.dump({
                "results": results, "pred_proba": pred_proba,
                "mlp_proba": mlp_proba, "mlp_auc": mlp_auc, "mlp_acc": mlp_acc,
            }, _f)
        print(f"  Saved logistic cache → {_cache_pkl.name}")

    # ── event-level: LR(all dims) vs MLP ──
    all_label    = f"All (z0–z{n_dims-1})"
    lr_all_proba = pred_proba[all_label]
    lr_auc       = results[all_label]["AUC"]

    lr_pred  = (lr_all_proba > 0.5).astype(int)
    mlp_pred = (mlp_proba    > 0.5).astype(int)

    both_correct_mask  = (lr_pred == y) & (mlp_pred == y)
    lr_only_mask       = (lr_pred == y) & (mlp_pred != y)
    mlp_only_mask      = (lr_pred != y) & (mlp_pred == y)
    both_wrong_mask    = (lr_pred != y) & (mlp_pred != y)
    N = len(y)

    print(f"\n  LR (all dims) AUC={lr_auc:.3f}  vs  MLP AUC={mlp_auc:.3f}")
    print(f"  Event-level agreement (N={N}):")
    print(f"    Both correct:  {both_correct_mask.sum():4d}  ({100*both_correct_mask.mean():.1f}%)")
    print(f"    LR only:       {lr_only_mask.sum():4d}  ({100*lr_only_mask.mean():.1f}%)")
    print(f"    MLP only:      {mlp_only_mask.sum():4d}  ({100*mlp_only_mask.mean():.1f}%)")
    print(f"    Both wrong:    {both_wrong_mask.sum():4d}  ({100*both_wrong_mask.mean():.1f}%)")

    # hard cases: split by true label
    hard_kaon_mask   = both_wrong_mask & (y == 1)   # kaon candidates both classifiers call proton
    hard_proton_mask = both_wrong_mask & (y == 0)   # protons both classifiers call kaon
    easy_kaon_mask   = both_correct_mask & (y == 1)
    proton_mask      = (y == 0)

    print(f"\n  Hard cases breakdown:")
    print(f"    Hard kaons   (look like protons): {hard_kaon_mask.sum()}")
    print(f"    Hard protons (look like kaons):   {hard_proton_mask.sum()}")

    # ── event-level: two best single-dim classifiers ──
    single_dim_labels = [f"z{i}" for i in range(n_dims)]
    single_aucs = [(lbl, results[lbl]["AUC"]) for lbl in single_dim_labels]
    top2   = sorted(single_aucs, key=lambda x: x[1], reverse=True)[:2]
    lbl_a, lbl_b = top2[0][0], top2[1][0]

    pred_a = (pred_proba[lbl_a] > 0.5).astype(int)
    pred_b = (pred_proba[lbl_b] > 0.5).astype(int)

    agree_correct = int(((pred_a == y) & (pred_b == y)).sum())
    a_only        = int(((pred_a == y) & (pred_b != y)).sum())
    b_only        = int(((pred_a != y) & (pred_b == y)).sum())
    both_wrong_ab = int(((pred_a != y) & (pred_b != y)).sum())

    print(f"\n  Single-dim agreement — {lbl_a} vs {lbl_b} (N={N}):")
    print(f"    Both correct:        {agree_correct:4d}  ({100*agree_correct/N:.1f}%)")
    print(f"    {lbl_a} only correct: {a_only:4d}  ({100*a_only/N:.1f}%)")
    print(f"    {lbl_b} only correct: {b_only:4d}  ({100*b_only/N:.1f}%)")
    print(f"    Both wrong:          {both_wrong_ab:4d}  ({100*both_wrong_ab/N:.1f}%)")

    da, db = int(lbl_a[1:]), int(lbl_b[1:])

    # ── PLOT 1: linear_probe{suffix}.png ──
    labels  = list(results.keys())
    aucs    = [results[l]["AUC"] for l in labels]
    palette = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    colours = [palette[i % 10] for i in range(len(labels))]
    axes[0].bar(labels, aucs, color=colours, edgecolor="white", width=0.55)
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].tick_params(axis="x", rotation=30)
    for i, auc in enumerate(aucs):
        axes[0].text(i, auc + 0.01, f"{auc:.3f}", ha="center", fontsize=9)

    categories  = ["Both correct", f"{lbl_a} only", f"{lbl_b} only", "Both wrong"]
    counts      = [agree_correct, a_only, b_only, both_wrong_ab]
    bar_colours = ["#4CAF50", BLUE, ORANGE, "#e57373"]
    axes[1].bar(categories, counts, color=bar_colours, edgecolor="white", width=0.5)
    axes[1].set_ylabel("Events")
    for i, c in enumerate(counts):
        axes[1].text(i, c + 0.5, str(c), ha="center", fontsize=9)

    plt.tight_layout()
    _savefig(out_dir / f"linear_probe{suffix}.png")
    plt.close()
    print(f"  saved linear_probe{suffix}.png")

    # ── PLOT 2: hard_cases{suffix}.png ──
    # row 0: LR vs MLP comparison  |  row 1: latent scatter  |  row 2: feature dists
    phys_feats = [f for f in (CALO + TOPO) if f in features_df.columns]

    n_feat_cols = len(phys_feats)
    n_grid_cols = max(4, n_feat_cols)  # need ≥4 so row-0 axes [0:2] and [2:4] never overlap
    fig = plt.figure(figsize=(7.0, 9.0))
    gs  = gridspec.GridSpec(
        3, n_grid_cols,
        figure=fig, hspace=0.45, wspace=0.35,
        height_ratios=[1, 1.4, 1.4],
    )

    # ── row 0: LR vs MLP AUC bar + event-level breakdown ──
    ax_auc   = fig.add_subplot(gs[0, :2])
    ax_break = fig.add_subplot(gs[0, 2:4])

    # AUC comparison
    classifier_labels = ["LR (all dims)", "MLP (all dims)"]
    classifier_aucs   = [lr_auc, mlp_auc]
    bars = ax_auc.bar(
        classifier_labels, classifier_aucs,
        color=[BLUE, ORANGE], edgecolor="white", width=0.4,
    )
    ax_auc.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax_auc.set_ylim(0.4, 1.0)
    ax_auc.set_ylabel("AUC-ROC")
    for bar, val in zip(bars, classifier_aucs):
        ax_auc.text(
            bar.get_x() + bar.get_width() / 2,
            val + 0.01, f"{val:.3f}", ha="center", fontsize=9, fontweight="bold",
        )

    # Event breakdown: both correct / LR only / MLP only / both wrong
    breakdown_cats   = ["Both\ncorrect", "LR\nonly", "MLP\nonly", "Both\nwrong"]
    breakdown_counts = [
        int(both_correct_mask.sum()),
        int(lr_only_mask.sum()),
        int(mlp_only_mask.sum()),
        int(both_wrong_mask.sum()),
    ]
    bcolours = ["#4CAF50", BLUE, ORANGE, "#e57373"]
    ax_break.bar(breakdown_cats, breakdown_counts, color=bcolours, edgecolor="white", width=0.5)
    ax_break.set_ylabel("Events")
    for i, c in enumerate(breakdown_counts):
        pct = 100 * c / N
        ax_break.text(i, c + N * 0.005, f"{c}\n({pct:.1f}%)", ha="center", fontsize=9)

    # ── row 1: latent scatter (top-2 discriminating dims) ──
    ax_scatter = fig.add_subplot(gs[1, :])

    if muon_latents is not None and len(muon_latents) > 0:
        ax_scatter.scatter(
            muon_latents[:, da], muon_latents[:, db],
            c=PURPLE, marker="s", s=4, alpha=0.5,
            label=f"Muons (n={len(muon_latents)})", linewidths=0, zorder=0,
        )

    groups = {
        "Protons":         (proton_mask,      BLUE,      "o",  4,  0.5),
        "Easy kaons":      (easy_kaon_mask,   "#009988", "o",  6,  0.6),
        "Hard kaons\n(look like protons)": (hard_kaon_mask, "#CC3311", "D", 14, 1.0),
        "Hard protons\n(look like kaons)": (hard_proton_mask, ORANGE,  "^", 14, 1.0),
    }

    for label, (mask, colour, marker, size, alpha) in groups.items():
        if mask.sum() == 0:
            continue
        kw = dict(c=colour, marker=marker, s=size, alpha=alpha, linewidths=0)
        if alpha == 1.0:
            kw.update(linewidths=0.3, edgecolors="white")
        ax_scatter.scatter(
            X[mask, da], X[mask, db],
            label=f"{label} (n={mask.sum()})", **kw,
        )

    ax_scatter.set_xlabel(rf"$z_{{{da}}}$  (AUC={results[lbl_a]['AUC']:.3f})")
    ax_scatter.set_ylabel(rf"$z_{{{db}}}$  (AUC={results[lbl_b]['AUC']:.3f})")
    ax_scatter.legend(**make_legend_kwargs("upper right"), markerscale=1.2)
    ax_scatter.spines[["top", "right"]].set_visible(False)

    # ── row 2: feature distributions — protons / easy kaons / hard kaons / muons ──
    group_data = {
        "Protons":    features_df[proton_mask.astype(bool)],
        "Easy kaons": features_df[easy_kaon_mask.astype(bool)],
        "Hard kaons": features_df[hard_kaon_mask.astype(bool)],
    }
    group_colours = {"Protons": BLUE, "Easy kaons": "#009988", "Hard kaons": "#CC3311"}
    if muon_features_df is not None and len(muon_features_df) > 0:
        group_data["Muons"] = muon_features_df
        group_colours["Muons"] = PURPLE

    for fi, feat in enumerate(phys_feats):
        ax_f = fig.add_subplot(gs[2, fi])
        for grp_label, grp_df in group_data.items():
            vals = grp_df[feat].dropna().values if feat in grp_df.columns else np.array([])
            if len(vals) == 0:
                continue
            ax_f.hist(
                vals, bins=30, density=True,
                color=group_colours[grp_label], alpha=0.55,
                label=grp_label, histtype="stepfilled",
            )
            ax_f.hist(
                vals, bins=30, density=True,
                color=group_colours[grp_label], alpha=0.9,
                histtype="step", linewidth=1.2,
            )
        ax_f.set_xlabel(feat)
        ax_f.set_ylabel("Density" if fi == 0 else "")
        ax_f.spines[["top", "right"]].set_visible(False)
        if fi == 0:
            ax_f.legend(
                frameon=True, framealpha=0.85, edgecolor="0.8",
                handlelength=1.0, handletextpad=0.4, borderpad=0.5,
            )

    title_tag = f" — {kaon_label}" if kaon_label else ""
    _savefig(out_dir / f"hard_cases{suffix}.png")
    plt.close()
    print(f"  saved hard_cases{suffix}.png")

    # ── PLOT 3 & 4: raw images of hard kaons and hard protons ──
    pt_data = torch.load(cfg["data"]["path"], map_location="cpu")
    n_val   = len(val_latents)

    for case_label, mask, predicted_as in [
        ("kaon",   hard_kaon_mask,   "proton"),
        ("proton", hard_proton_mask, "kaon"),
    ]:
        hard_x_indices = np.where(mask)[0]
        if len(hard_x_indices) == 0:
            print(f"  no hard {case_label}s to plot — skipping")
            continue

        if case_label == "kaon":
            # map from position-in-subset back to position in the full kaon tensor
            dataset_indices = kaon_orig_indices[hard_x_indices - n_val]
            particle_data   = pt_data[cfg["data"]["kaon"]]
        else:
            # hard protons: X indices 0..n_val-1 map to val_idx in the proton tensor
            dataset_indices = val_idx[hard_x_indices]
            particle_data   = pt_data[cfg["data"]["proton"]]

        n_show = min(50, len(dataset_indices))
        dataset_indices = dataset_indices[:n_show]

        ncols = 10
        nrows = (n_show + ncols - 1) // ncols

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(ncols * 1.8, nrows * 2.0),
            constrained_layout=True,
        )
        axes = np.array(axes).reshape(nrows, ncols)

        for idx, ds_idx in enumerate(dataset_indices):
            row, col = divmod(idx, ncols)
            item = particle_data[int(ds_idx)]
            if isinstance(item, (tuple, list)):
                item = item[0]
            img = item.numpy() if isinstance(item, torch.Tensor) else np.array(item)
            if img.ndim == 3:
                img = img[0]  # first channel
            axes[row, col].imshow(img, origin="lower", cmap="viridis")
            axes[row, col].set_title(f"#{ds_idx}", fontsize=6)
            axes[row, col].axis("off")

        # hide unused axes in last row
        for idx in range(n_show, nrows * ncols):
            row, col = divmod(idx, ncols)
            axes[row, col].set_visible(False)

        fname = f"hard_cases_images_{case_label}{suffix}.png"
        _savefig(out_dir / fname)
        plt.close()
        print(f"  saved {fname}")


def run_logistic(cfg, model_name, features_path, out_dir, muon_latents=None, muon_features_df=None, from_cache=False):
    print("\n=== Logistic regression + MLP classifier ===")

    _, val_latents, kaon_latents = load_latents(cfg, model_name)

    features, index = load_features_and_splits(cfg, features_path)
    all_proton   = features[features["particle_type"] == "proton"]
    all_kaon     = features[features["particle_type"] == "kaon"].reset_index(drop=True)
    val_features = all_proton.iloc[index["val_idx"]]
    val_idx      = index["val_idx"]

    # ── run 1: all kaons ──
    _run_logistic_hardcases(
        val_latents, kaon_latents,
        val_features, all_kaon,
        kaon_orig_indices=np.arange(len(kaon_latents)),
        val_idx=val_idx,
        cfg=cfg, out_dir=out_dir,
        suffix="", kaon_label="",
        muon_latents=muon_latents, muon_features_df=muon_features_df,
        from_cache=from_cache,
    )

    # ── run 2: picky kaons (p=1 in picky+match.csv) ──
    picky_csv_path = Path("/Volumes/easystore/proton-kaon/docs/picky+match.csv")
    if not picky_csv_path.exists():
        print(f"  picky CSV not found at {picky_csv_path} — skipping picky run")
        return

    picky = pd.read_csv(picky_csv_path)
    picky_p1 = picky.loc[picky["p"] == 1, ["run", "subrun", "event"]].copy()
    picky_p1["_picky"] = True
    merged = all_kaon[["run", "subrun", "event"]].merge(
        picky_p1, on=["run", "subrun", "event"], how="left"
    )
    picky_mask    = merged["_picky"].notna().values
    picky_indices = np.where(picky_mask)[0]

    print(f"\n--- Picky kaon run: {picky_mask.sum()} / {len(all_kaon)} kaons (p=1) ---")

    _run_logistic_hardcases(
        val_latents, kaon_latents[picky_indices],
        val_features, all_kaon.iloc[picky_indices].reset_index(drop=True),
        kaon_orig_indices=picky_indices,
        val_idx=val_idx,
        cfg=cfg, out_dir=out_dir,
        suffix="_picky", kaon_label="picky kaons (p=1)",
        muon_latents=muon_latents, muon_features_df=muon_features_df,
        from_cache=from_cache,
    )


# ── analysis 4: non-linear ────────────────────────────────────────────────────

def run_nonlinear(cfg, model_name, features_path, out_dir, from_cache=False):
    print("\n=== Non-linear analysis ===")
    _cache_path_nl = out_dir / "cache_nonlinear.json"
    _use_nl_cache  = from_cache and _cache_path_nl.exists()
    if _use_nl_cache:
        print(f"  (--from-cache: loading precomputed results from {_cache_path_nl.name})")

    train_latents, val_latents, kaon_latents = load_latents(cfg, model_name)
    n_dims   = train_latents.shape[1]
    dim_cols = latent_dim_names(n_dims)
    imp_cols = [f"z{i}_imp" for i in range(n_dims)]

    features, index = load_features_and_splits(cfg, features_path)

    all_proton = features[features["particle_type"] == "proton"]
    all_kaon   = features[features["particle_type"] == "kaon"]
    all_muon   = features[features["particle_type"] == "muon"]

    inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
    muon_npz = inference_dir / "muon.npz"
    muon_latents_nl = np.load(muon_npz)["latents"] if muon_npz.exists() and len(all_muon) > 0 else None

    train_features = all_proton.iloc[index["train_idx"]]
    val_features   = all_proton.iloc[index["val_idx"]]
    kaon_features  = all_kaon

    if muon_latents_nl is not None:
        X           = np.vstack([train_latents, val_latents, kaon_latents, muon_latents_nl])
        features_df = pd.concat([train_features, val_features, kaon_features, all_muon], ignore_index=True)
        particle_labels = np.array(
            [0] * len(train_latents) + [0] * len(val_latents) + [1] * len(kaon_latents) + [2] * len(muon_latents_nl)
        )
    else:
        X           = np.vstack([train_latents, val_latents, kaon_latents])
        features_df = pd.concat([train_features, val_features, kaon_features], ignore_index=True)
        particle_labels = np.array(
            [0] * len(train_latents) + [0] * len(val_latents) + [1] * len(kaon_latents)
        )

    calo = [f for f in CALO if f in features_df.columns]
    topo = [f for f in TOPO if f in features_df.columns]
    all_feats = calo + topo

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    linear_pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

    print(f"  R² comparison: {len(all_feats)} features ({len(calo)} calo, {len(topo)} topo)")
    print(f"  N = {len(X)} events\n")

    records = []
    for i, feat in enumerate(all_feats, 1):
        if _use_nl_cache:
            continue
        category = "calorimetry" if feat in calo else "topology"
        print(f"  [{i:2d}/{len(all_feats)}] {feat}  ({category})")

        y    = features_df[feat].values.astype(float)
        mask = np.isfinite(y)
        Xm, ym, lm = X[mask], y[mask], particle_labels[mask]

        if mask.sum() < len(mask):
            print(f"         dropped {(~mask).sum()} non-finite  →  N={mask.sum()}")

        splits = list(cv.split(Xm, lm))

        r2_linear = max(0.0, cross_val_score(
            linear_pipeline, Xm, ym, cv=splits, scoring="r2"
        ).mean())

        mlp_pipeline = make_mlp_pipeline()
        r2_mlp = max(0.0, cross_val_score(
            mlp_pipeline, Xm, ym, cv=splits, scoring="r2"
        ).mean())
        gap = round(r2_mlp - r2_linear, 3)
        print(f"         linear R²={r2_linear:.3f}  mlp R²={r2_mlp:.3f}  gap={gap:+.3f}")

        # OOF predictions for error metrics and per-particle breakdown.
        # cross_val_predict clones the pipeline per fold — no data leakage.
        y_oof_lin = cross_val_predict(linear_pipeline,    Xm, ym, cv=splits)
        y_oof_mlp = cross_val_predict(make_mlp_pipeline(), Xm, ym, cv=splits)

        p_mask = lm == 0
        k_mask = lm == 1
        m_mask = lm == 2
        r2_p = max(0.0, r2_score(ym[p_mask], y_oof_mlp[p_mask]))
        r2_k = max(0.0, r2_score(ym[k_mask], y_oof_mlp[k_mask]))
        r2_m = max(0.0, r2_score(ym[m_mask], y_oof_mlp[m_mask])) if m_mask.sum() > 1 else np.nan

        # Fit once on all data solely for permutation importance
        # (importance is relative, so in-sample is acceptable here).
        pipe_full = make_mlp_pipeline()
        pipe_full.fit(Xm, ym)
        imp   = permutation_importance_mlp(pipe_full, Xm, ym)
        imp_p = permutation_importance_mlp(pipe_full, Xm[p_mask], ym[p_mask]) if p_mask.sum() > 5 else np.zeros(n_dims)
        imp_k = permutation_importance_mlp(pipe_full, Xm[k_mask], ym[k_mask]) if k_mask.sum() > 5 else np.zeros(n_dims)
        imp_str = "  ".join(f"z{j}:{imp[j]:.3f}" for j in range(n_dims))
        muon_str = f"  muon R²={r2_m:.3f}" if not np.isnan(r2_m) else ""
        print(f"         proton R²={r2_p:.3f}  kaon R²={r2_k:.3f}{muon_str}  perm: {imp_str}")

        # ── Error metrics (MAE / MSE / RMSE / MAPE) ──────────────────────────
        def _fm(v):   # format MAPE
            return f"{v:.1f}%" if np.isfinite(v) else "N/A"
        def _rnf(v, d=4):  # round if finite
            return round(float(v), d) if np.isfinite(float(v)) else np.nan

        lin_mae,  lin_mse,  lin_rmse,  lin_mape  = compute_regression_metrics(ym, y_oof_lin)
        mlp_mae,  mlp_mse,  mlp_rmse,  mlp_mape  = compute_regression_metrics(ym, y_oof_mlp)
        print(f"         Linear  MAE={lin_mae:.4f}  MSE={lin_mse:.4f}  RMSE={lin_rmse:.4f}  MAPE={_fm(lin_mape)}")
        print(f"         MLP     MAE={mlp_mae:.4f}  MSE={mlp_mse:.4f}  RMSE={mlp_rmse:.4f}  MAPE={_fm(mlp_mape)}")

        (lin_mae_p, lin_mse_p, lin_rmse_p, lin_mape_p), (mlp_mae_p, mlp_mse_p, mlp_rmse_p, mlp_mape_p) \
            = _particle_metrics(p_mask, ym, y_oof_lin, y_oof_mlp)
        (lin_mae_k, lin_mse_k, lin_rmse_k, lin_mape_k), (mlp_mae_k, mlp_mse_k, mlp_rmse_k, mlp_mape_k) \
            = _particle_metrics(k_mask, ym, y_oof_lin, y_oof_mlp)
        (lin_mae_m, lin_mse_m, lin_rmse_m, lin_mape_m), (mlp_mae_m, mlp_mse_m, mlp_rmse_m, mlp_mape_m) \
            = _particle_metrics(m_mask, ym, y_oof_lin, y_oof_mlp)

        print(f"         proton  Lin RMSE={lin_rmse_p:.4f} MAPE={_fm(lin_mape_p)}"
              f"  |  MLP RMSE={mlp_rmse_p:.4f} MAPE={_fm(mlp_mape_p)}")
        print(f"         kaon    Lin RMSE={lin_rmse_k:.4f} MAPE={_fm(lin_mape_k)}"
              f"  |  MLP RMSE={mlp_rmse_k:.4f} MAPE={_fm(mlp_mape_k)}")
        if m_mask.sum() > 1:
            print(f"         muon    Lin RMSE={lin_rmse_m:.4f} MAPE={_fm(lin_mape_m)}"
                  f"  |  MLP RMSE={mlp_rmse_m:.4f} MAPE={_fm(mlp_mape_m)}")
        print()

        row = {
            "feature":   feat,
            "category":  category,
            "linear_r2": round(r2_linear, 3),
            "mlp_r2":    round(r2_mlp, 3),
            "gap":       gap,
            "r2_proton": round(r2_p, 3),
            "r2_kaon":   round(r2_k, 3),
            "r2_muon":   round(r2_m, 3) if not np.isnan(r2_m) else np.nan,
            "gap_pk":    round(r2_p - r2_k, 3),
            # ── overall error metrics ──
            "linear_mae":  _rnf(lin_mae),
            "linear_mse":  _rnf(lin_mse),
            "linear_rmse": _rnf(lin_rmse),
            "linear_mape": _rnf(lin_mape, 2),
            "mlp_mae":     _rnf(mlp_mae),
            "mlp_mse":     _rnf(mlp_mse),
            "mlp_rmse":    _rnf(mlp_rmse),
            "mlp_mape":    _rnf(mlp_mape, 2),
            # ── per-particle RMSE ──
            "linear_rmse_proton": _rnf(lin_rmse_p),
            "linear_rmse_kaon":   _rnf(lin_rmse_k),
            "linear_rmse_muon":   _rnf(lin_rmse_m),
            "mlp_rmse_proton":    _rnf(mlp_rmse_p),
            "mlp_rmse_kaon":      _rnf(mlp_rmse_k),
            "mlp_rmse_muon":      _rnf(mlp_rmse_m),
            # ── per-particle MAE ──
            "linear_mae_proton": _rnf(lin_mae_p),
            "linear_mae_kaon":   _rnf(lin_mae_k),
            "linear_mae_muon":   _rnf(lin_mae_m),
            "mlp_mae_proton":    _rnf(mlp_mae_p),
            "mlp_mae_kaon":      _rnf(mlp_mae_k),
            "mlp_mae_muon":      _rnf(mlp_mae_m),
            # ── per-particle MAPE ──
            "linear_mape_proton": _rnf(lin_mape_p, 2),
            "linear_mape_kaon":   _rnf(lin_mape_k, 2),
            "linear_mape_muon":   _rnf(lin_mape_m, 2),
            "mlp_mape_proton":    _rnf(mlp_mape_p, 2),
            "mlp_mape_kaon":      _rnf(mlp_mape_k, 2),
            "mlp_mape_muon":      _rnf(mlp_mape_m, 2),
        }
        for j in range(n_dims):
            row[f"z{j}_imp"]        = round(imp[j],   4)
            row[f"z{j}_imp_proton"] = round(imp_p[j], 4)
            row[f"z{j}_imp_kaon"]   = round(imp_k[j], 4)
        records.append(row)

    if _use_nl_cache:
        results = pd.read_json(_cache_path_nl, orient="records")
        print("  Loaded nonlinear results from cache.")
    else:
        results = (
            pd.DataFrame(records)
            .sort_values("gap", ascending=False)
            .reset_index(drop=True)
        )
        results.to_json(_cache_path_nl, orient="records", indent=2)
        print(f"  Saved nonlinear cache → {_cache_path_nl.name}")
    print(results.to_string(index=False))

    # ── nonlinear_r2.png ──
    calo_df = results[results["category"] == "calorimetry"].sort_values("gap", ascending=True)
    topo_df = results[results["category"] == "topology"].sort_values("gap", ascending=True)

    PX_PER_FEAT = 0.50
    PAD         = 1.5
    calo_h = len(calo_df) * PX_PER_FEAT + PAD
    topo_h = len(topo_df) * PX_PER_FEAT + PAD

    fig  = plt.figure(figsize=(7.0, calo_h + topo_h + 0.5))
    gs   = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[calo_h, topo_h], hspace=0.15, top=0.95)
    ax_c = fig.add_subplot(gs[0])
    ax_t = fig.add_subplot(gs[1], sharex=ax_c)
    x_max = results["mlp_r2"].max() * 1.18

    for ax, df, title in zip([ax_c, ax_t], [calo_df, topo_df], ["Calorimetry", "Topology"]):
        y_pos       = np.arange(len(df))
        h           = 0.6
        linear_vals = df["linear_r2"].values
        gap_vals    = df["gap"].clip(lower=0).values
        mlp_vals    = df["mlp_r2"].values
        feats       = df["feature"].values

        ax.barh(y_pos, linear_vals, h, color=BLUE,   alpha=0.9, label="Linear $R^2$")
        ax.barh(y_pos, gap_vals,    h, color=ORANGE, alpha=0.9, left=linear_vals,
                label="Non-linear gain")

        for yi, (lin, gap_v, mlp) in enumerate(zip(linear_vals, gap_vals, mlp_vals)):
            ax.text(mlp + 0.008, yi, f"{mlp:.3f}", va="center", ha="left", fontsize=9)
            if gap_v > 0.04:
                ax.text(lin + gap_v / 2, yi, f"+{gap_v:.2f}",
                        va="center", ha="center", fontsize=9, color="white", fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Cross-validated $R^2$")
        ax.set_title(title, fontsize=9, fontweight="semibold", loc="left", pad=4)
        ax.grid(axis="x", alpha=0.3, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(
            loc="lower right", frameon=True, framealpha=0.85, edgecolor="0.8",
            handlelength=1.0, handletextpad=0.4, borderpad=0.5,
        )

    _savefig(out_dir / "nonlinear_r2.png")
    plt.close()
    print("  saved nonlinear_r2.png")

    # ── error_metrics.png (RMSE & MAE per feature × model × particle) ──
    _has_muon_em = results["mlp_rmse_muon"].notna().any()
    _particle_specs = [
        ("all particles", "linear_rmse",        "mlp_rmse",        "linear_mae",        "mlp_mae",        BLUE,      ORANGE),
        ("proton",        "linear_rmse_proton",  "mlp_rmse_proton", "linear_mae_proton", "mlp_mae_proton", "#5B9BD5", "#FCA15B"),
        ("kaon",          "linear_rmse_kaon",    "mlp_rmse_kaon",   "linear_mae_kaon",   "mlp_mae_kaon",   "#59A14F", "#EFCE5A"),
    ]
    if _has_muon_em:
        _particle_specs.append(
            ("muon", "linear_rmse_muon", "mlp_rmse_muon", "linear_mae_muon", "mlp_mae_muon", "#9467BD", "#C5B0D5")
        )
    _n_panels = len(_particle_specs)
    _fig_em, _axes_em = plt.subplots(
        2, _n_panels,
        figsize=(7.0, max(3, len(results) * 0.55 + 1.5)),
        sharey="row",
    )
    if _n_panels == 1:
        _axes_em = _axes_em[:, np.newaxis]

    _feat_order = results["feature"].values
    _feat_cat   = results.set_index("feature")["category"]
    _feat_labels = [
        f"{f}\n({_cat_abbr.get(_feat_cat[f], _feat_cat[f])})"
        for f in _feat_order
    ]
    _y_pos = np.arange(len(_feat_order))
    _h = 0.35

    for _pi, (_pname, _lin_rmse_col, _mlp_rmse_col, _lin_mae_col, _mlp_mae_col, _lc, _mc) in enumerate(_particle_specs):
        for _row_idx, (_metric_name, _lin_col, _mlp_col) in enumerate([
            ("RMSE", _lin_rmse_col, _mlp_rmse_col),
            ("MAE",  _lin_mae_col,  _mlp_mae_col),
        ]):
            _ax = _axes_em[_row_idx, _pi]
            _lin_vals = results[_lin_col].values.astype(float)
            _mlp_vals = results[_mlp_col].values.astype(float)
            _ax.barh(_y_pos - _h / 2, _lin_vals, _h, color=_lc, alpha=0.9, label="Linear (Ridge)")
            _ax.barh(_y_pos + _h / 2, _mlp_vals, _h, color=_mc, alpha=0.9, label="MLP")
            _ax.set_yticks(_y_pos)
            _ax.set_yticklabels(_feat_labels if _pi == 0 else [])
            _ax.set_xlabel(_metric_name)
            if _row_idx == 0:
                _ax.set_title(_pname, fontsize=8, fontweight="semibold")
            _ax.legend(
                frameon=True, framealpha=0.85, edgecolor="0.8",
                handlelength=1.0, handletextpad=0.4, borderpad=0.5,
            )
            _ax.spines[["top", "right"]].set_visible(False)
            _ax.grid(axis="x", alpha=0.3, linewidth=0.6)

    plt.tight_layout()
    _savefig(out_dir / "error_metrics.png")
    plt.close()
    print("  saved error_metrics.png")

    # ── permutation importance heatmaps ──
    rename = {f"z{i}_imp": f"z{i}" for i in range(n_dims)}
    for category in ["calorimetry", "topology"]:
        df_cat = (
            results[results["category"] == category]
            .set_index("feature")[imp_cols]
            .rename(columns=rename)
            .sort_values("z0")
        )
        _perm_h = max(2.5, len(df_cat) * 0.75 + 1.0)
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, _perm_h))
        sns.heatmap(
            df_cat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            linewidths=0.4, ax=ax,
            cbar_kws={"label": "$R^2$ drop on permutation"},
            annot_kws={"size": 10},
        )
        ax.set_xlabel("Latent dimension")
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        fname = f"permutation_importance_{category}.png"
        _savefig(out_dir / fname)
        plt.close()
        print(f"  saved {fname}")

    for ptag in ["proton", "kaon"]:
        imp_cols_pt = [f"z{i}_imp_{ptag}" for i in range(n_dims)]
        rename_pt   = {f"z{i}_imp_{ptag}": f"z{i}" for i in range(n_dims)}
        for category in ["calorimetry", "topology"]:
            df_cat = (
                results[results["category"] == category]
                .set_index("feature")[imp_cols_pt]
                .rename(columns=rename_pt)
                .sort_values("z0")
            )
            _perm_h = max(2.5, len(df_cat) * 0.75 + 1.0)
            fig, ax = plt.subplots(figsize=(DOUBLE_COL, _perm_h))
            sns.heatmap(
                df_cat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
                linewidths=0.4, ax=ax,
                cbar_kws={"label": "$R^2$ drop on permutation"},
                annot_kws={"size": 10},
            )
            ax.set_xlabel("Latent dimension")
            ax.tick_params(axis="y", rotation=0)
            plt.tight_layout()
            fname = f"permutation_importance_{category}_{ptag}.png"
            _savefig(out_dir / fname)
            plt.close()
            print(f"  saved {fname}")

    # ── mutual information heatmaps ──
    mi_records = []
    for feat in all_feats:
        category = "calorimetry" if feat in calo else "topology"
        y    = features_df[feat].values.astype(float)
        mask = np.isfinite(y)
        Zm, ym = X[mask], y[mask]

        mi_full = mutual_info_regression(Zm, ym, random_state=42)[0]
        mi_dims = [
            mutual_info_regression(Zm[:, j:j+1], ym, random_state=42)[0]
            for j in range(n_dims)
        ]
        best_single    = max(mi_dims)
        gap_mi         = mi_full - best_single
        frac_explained = best_single / mi_full if mi_full > 1e-6 else np.nan

        row = {
            "feature":        feat,
            "category":       category,
            "mi_full":        round(mi_full, 4),
            "best_single":    round(best_single, 4),
            "gap":            round(gap_mi, 4),
            "frac_explained": round(frac_explained, 3) if np.isfinite(frac_explained) else np.nan,
        }
        for j in range(n_dims):
            row[f"mi_z{j}"] = round(mi_dims[j], 4)
        mi_records.append(row)

    mi_df = pd.DataFrame(mi_records).set_index("feature")

    mi_dim_cols = [f"mi_z{j}" for j in range(n_dims)]
    mi_rename   = {f"mi_z{j}": f"z{j}" for j in range(n_dims)}
    vmax_mi     = mi_df[mi_dim_cols].values.max()

    for category in ["calorimetry", "topology"]:
        sub = mi_df[mi_df["category"] == category][mi_dim_cols].rename(columns=mi_rename)
        sub = sub.sort_values("z0", ascending=False)
        _mi_h = max(2.5, len(sub) * 0.75 + 1.0)
        fig, ax = plt.subplots(figsize=(DOUBLE_COL, _mi_h))
        sns.heatmap(
            sub, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "MI (nats)"},
            vmin=0, vmax=vmax_mi,
            annot_kws={"size": 10},
        )
        ax.set_xlabel("Latent dimension")
        ax.tick_params(axis="y", rotation=0)
        plt.tight_layout()
        fname = f"mutual_information_{category}.png"
        _savefig(out_dir / fname)
        plt.close()
        print(f"  saved {fname}")


# ── analysis 5: feature AUC ───────────────────────────────────────────────────

def run_feature_auc(cfg, model_name, features_path, out_dir, muon_latents=None, muon_features_df=None, csda_kaon_latents=None, csda_kaon_features_df=None, from_cache=False):
    print("\n=== Feature AUC analysis (per class) ===")
    _cache_path_fa  = out_dir / "cache_feature_auc.json"
    _use_feat_cache = from_cache and _cache_path_fa.exists()
    if _use_feat_cache:
        print(f"  (--from-cache: loading precomputed feature AUC from {_cache_path_fa.name})")

    train_latents, val_latents, kaon_latents = load_latents(cfg, model_name)
    features, index = load_features_and_splits(cfg, features_path)

    all_proton = features[features["particle_type"] == "proton"]
    all_kaon   = features[features["particle_type"] == "kaon"]

    proton_latents  = np.vstack([train_latents, val_latents])
    train_features  = all_proton.iloc[index["train_idx"]]
    val_features    = all_proton.iloc[index["val_idx"]]
    proton_features = pd.concat([train_features, val_features], ignore_index=True)
    kaon_features   = all_kaon.reset_index(drop=True)

    calo = [f for f in CALO if f in features.columns]
    topo = [f for f in TOPO if f in features.columns]
    all_feats = calo + topo

    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def _probe(latents, feat_df, feat_name):
        vals = feat_df[feat_name].values.astype(float)
        finite_mask = np.isfinite(vals)
        if finite_mask.sum() < 10:
            return None, None
        median_val = np.nanmedian(vals[finite_mask])
        y  = (vals > median_val).astype(int)
        Xm = latents[finite_mask]
        ym = y[finite_mask]
        if ym.sum() < 2 or (len(ym) - ym.sum()) < 2:
            return None, None
        proba = cross_val_predict(lr_pipeline, Xm, ym, cv=cv, method="predict_proba")[:, 1]
        return roc_auc_score(ym, proba), median_val

    has_muon = muon_latents is not None and muon_features_df is not None and len(muon_latents) > 0
    has_csda = csda_kaon_latents is not None and csda_kaon_features_df is not None and len(csda_kaon_latents) > 0

    records = []
    for feat in all_feats:
        if _use_feat_cache:
            continue
        category = "calorimetry" if feat in calo else "topology"

        auc_p,  med_p  = _probe(proton_latents,     proton_features,      feat)
        auc_k,  med_k  = _probe(kaon_latents,        kaon_features,        feat)
        auc_m,  med_m  = _probe(muon_latents,        muon_features_df,     feat) if has_muon else (None, None)
        auc_ck, med_ck = _probe(csda_kaon_latents,   csda_kaon_features_df, feat) if has_csda else (None, None)

        if auc_p is None and auc_k is None and auc_m is None and auc_ck is None:
            print(f"  {feat:25s}  skipped (all classes insufficient)")
            continue

        if auc_p is not None:
            print(f"  {feat:25s}  proton    ({category[:4]})  median={med_p:.3g}  AUC={auc_p:.3f}")
            records.append({"feature": feat, "category": category, "particle": "proton",    "auc": auc_p})
        if auc_k is not None:
            print(f"  {feat:25s}  kaon      ({category[:4]})  median={med_k:.3g}  AUC={auc_k:.3f}")
            records.append({"feature": feat, "category": category, "particle": "kaon",      "auc": auc_k})
        if auc_m is not None:
            print(f"  {feat:25s}  muon      ({category[:4]})  median={med_m:.3g}  AUC={auc_m:.3f}")
            records.append({"feature": feat, "category": category, "particle": "muon",      "auc": auc_m})
        if auc_ck is not None:
            print(f"  {feat:25s}  csda-kaon ({category[:4]})  median={med_ck:.3g}  AUC={auc_ck:.3f}")
            records.append({"feature": feat, "category": category, "particle": "csda_kaon", "auc": auc_ck})

    if _use_feat_cache:
        auc_df = pd.read_json(_cache_path_fa, orient="records")
        print("  Loaded feature AUC results from cache.")
    else:
        if not records:
            print("  No features produced a valid AUC — skipping plot.")
            return
        auc_df = pd.DataFrame(records)
        auc_df.to_json(_cache_path_fa, orient="records", indent=2)
        print(f"  Saved feature AUC cache → {_cache_path_fa.name}")

    print("\n  Summary (mean AUC across classes, sorted descending):")
    pivot = auc_df.pivot_table(index="feature", columns="particle", values="auc")
    pivot["mean"] = pivot.mean(axis=1)
    print(pivot.sort_values("mean", ascending=False).drop(columns="mean").to_string())

    # ── feature_auc.png — grouped horizontal bars ──
    feat_order = (
        auc_df.groupby("feature")["auc"].mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    feat_to_cat = dict(zip(auc_df["feature"], auc_df["category"]))

    PROTON_COL    = BLUE
    KAON_COL      = ORANGE
    MUON_COL      = PURPLE
    CSDA_KAON_COL = GREEN
    n_feats    = len(feat_order)

    particles_present = [p for p in ["proton", "kaon", "muon", "csda_kaon"] if p in auc_df["particle"].unique()]
    n_bars = len(particles_present)
    bar_h  = max(0.18, 0.72 / n_bars)
    step   = bar_h
    offsets = {p: (n_bars - 1) / 2 * step - idx * step for idx, p in enumerate(particles_present)}
    colors  = {"proton": PROTON_COL, "kaon": KAON_COL, "muon": MUON_COL, "csda_kaon": CSDA_KAON_COL}

    fig, ax = plt.subplots(figsize=(DOUBLE_COL, max(3.0, n_feats * 1.0 + 1.5)))

    for i, feat in enumerate(feat_order):
        sub = auc_df[auc_df["feature"] == feat]
        for particle in particles_present:
            row = sub[sub["particle"] == particle]
            if row.empty:
                continue
            v     = row["auc"].values[0]
            y_pos = i + offsets[particle]
            col   = colors[particle]
            ax.barh(y_pos, v, height=bar_h, color=col, edgecolor="white")
            ax.text(v + 0.004, y_pos, f"{v:.3f}",
                    va="center", ha="left", fontsize=9, color=col, fontweight="bold")

    ax.axvline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_xlim(0.4, 1.02)
    ax.set_yticks(np.arange(n_feats))
    ax.set_yticklabels(
        [f"{f}\n({_cat_abbr.get(feat_to_cat.get(f,''), '')})" for f in feat_order],
    )
    ax.set_xlabel("AUC-ROC")

    from matplotlib.patches import Patch
    _labels = {"proton": "Proton", "kaon": "Kaon", "muon": "Muon", "csda_kaon": "CSDA-Kaon"}
    legend_handles = [Patch(facecolor=colors[p], label=_labels.get(p, p.capitalize())) for p in particles_present]
    legend_handles.append(plt.Line2D([0], [0], color="grey", linestyle="--", linewidth=1, label="Chance (0.5)"))
    ax.legend(handles=legend_handles, **make_legend_kwargs())

    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.3, linewidth=0.6)
    plt.tight_layout()
    _savefig(out_dir / "feature_auc.png")
    plt.close()
    print("  saved feature_auc.png")


_cat_abbr = {"calorimetry": "calo", "topology": "topo"}


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latent-feature analysis for a VAE model.")
    parser.add_argument("--config",    required=True, help="Path to model YAML config")
    parser.add_argument(
        "--analyses", nargs="+",
        choices=["correlation", "traversal", "logistic", "nonlinear", "feature_auc"],
        default=["correlation", "traversal", "logistic", "nonlinear", "feature_auc"],
        help="Which analyses to run (default: all)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to features.pkl (overrides config; falls back to data.features_path in config)",
    )
    parser.add_argument(
        "--include-muons", action="store_true",
        help="Also run separate muon-only analysis (correlation, traversal, nonlinear)"
    )
    parser.add_argument(
        "--csda-kaons", action="store_true",
        help="Include csda-kaon latents/features in feature_auc and binary logistic probes",
    )
    parser.add_argument(
        "--from-cache", action="store_true",
        help=(
            "Skip ML training; load pre-computed numbers from cache files in the output dir.  "
            "Cache files are saved automatically on first run.  "
            "JSON caches (nonlinear, feature_auc) are human-editable to tweak individual numbers."
        ),
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # resolve features path: CLI arg > config field > hardcoded default
    features_path = (
        args.features
        or cfg.get("data", {}).get("features_path")
        or "/Volumes/easystore/proton-kaon/features/features.pkl"
    )

    model_name = build_model_name(cfg)
    print(f"Model: {model_name}")

    out_dir = PROJECT_ROOT / "figs" / model_name / "latents-features" 
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    # ── pre-load muon data once (used in logistic and muon-only sections) ──
    muon_latents = None
    muon_features = None
    if args.include_muons:
        inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
        muon_npz_path = inference_dir / "muon.npz"
        if muon_npz_path.exists():
            muon_latents = np.load(muon_npz_path)["latents"]
            _features, _ = load_features_and_splits(cfg, features_path)
            muon_features = _features[_features["particle_type"] == "muon"]
            if len(muon_features) == 0:
                muon_features = None

    # ── pre-load csda-kaon data ──
    csda_kaon_latents = None
    csda_kaon_features = None
    if args.csda_kaons:
        inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
        csda_npz_path = inference_dir / "csda_kaon.npz"
        if csda_npz_path.exists():
            csda_kaon_latents = np.load(csda_npz_path)["latents"]
            _features, _ = load_features_and_splits(cfg, features_path)
            csda_kaon_features = _features[_features["particle_type"] == "csda_kaon"].reset_index(drop=True)
            if len(csda_kaon_features) == 0:
                csda_kaon_features = None
            print(f"Loaded {len(csda_kaon_latents)} csda-kaon latents, "
                  f"{len(csda_kaon_features) if csda_kaon_features is not None else 0} csda-kaon features")
        else:
            print(f"csda-kaon latents not found at {csda_npz_path} — run run_inference.py --csda-kaon-path")

    if "correlation" in args.analyses:
        run_correlation(cfg, model_name, features_path, out_dir)

    if "traversal" in args.analyses:
        run_traversal(cfg, model_name, out_dir)

    if "logistic" in args.analyses:
        run_logistic(cfg, model_name, features_path, out_dir,
                     muon_latents=muon_latents, muon_features_df=muon_features,
                     from_cache=args.from_cache)

    if "nonlinear" in args.analyses:
        run_nonlinear(cfg, model_name, features_path, out_dir,
                      from_cache=args.from_cache)

    if "feature_auc" in args.analyses:
        run_feature_auc(cfg, model_name, features_path, out_dir,
                        muon_latents=muon_latents, muon_features_df=muon_features,
                        csda_kaon_latents=csda_kaon_latents, csda_kaon_features_df=csda_kaon_features,
                        from_cache=args.from_cache)

    # ── csda-kaon binary logistic probes ──────────────────────────────────────
    if args.csda_kaons and csda_kaon_latents is not None and "logistic" in args.analyses:
        print("\n=== CSDA-Kaon binary logistic probes (60/40 train/test) ===")
        train_l_ck, val_l_ck, kaon_l_ck = load_latents(cfg, model_name)
        proton_l_ck = np.vstack([train_l_ck, val_l_ck])

        lr_ck = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ])

        comparisons = [
            ("csda-kaon vs proton", np.vstack([csda_kaon_latents, proton_l_ck]),
             np.array([1] * len(csda_kaon_latents) + [0] * len(proton_l_ck))),
            ("csda-kaon vs kaon",   np.vstack([csda_kaon_latents, kaon_l_ck]),
             np.array([1] * len(csda_kaon_latents) + [0] * len(kaon_l_ck))),
        ]
        if muon_latents is not None:
            comparisons.append(
                ("csda-kaon vs muon", np.vstack([csda_kaon_latents, muon_latents]),
                 np.array([1] * len(csda_kaon_latents) + [0] * len(muon_latents)))
            )

        sss = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=42)

        results_ck = {}
        for label_b, X_b, y_b in comparisons:
            train_idx, test_idx = next(sss.split(X_b, y_b))
            lr_ck.fit(X_b[train_idx], y_b[train_idx])
            proba_test = lr_ck.predict_proba(X_b[test_idx])[:, 1]
            auc = roc_auc_score(y_b[test_idx], proba_test)
            acc = accuracy_score(y_b[test_idx], (proba_test > 0.5).astype(int))
            results_ck[label_b] = auc
            print(f"  {label_b:30s}  AUC={auc:.3f}  Acc={acc:.3f}  "
                  f"(train={len(train_idx)}  test={len(test_idx)})")

        n_bars  = len(results_ck)
        colors  = ([GREEN, ORANGE, "#009988"] + ["#888"] * n_bars)[:n_bars]
        fig, ax = plt.subplots(figsize=(7.0, 3.5))
        bars = ax.bar(list(results_ck.keys()), list(results_ck.values()),
                      color=colors, edgecolor="white", width=0.5)
        ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
        ax.set_ylim(0.4, 1.0)
        ax.set_ylabel("AUC-ROC")
        for bar, val in zip(bars, results_ck.values()):
            ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01,
                    f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
        plt.tight_layout()
        _savefig(out_dir / "csda_kaon_logistic_probes.png")
        plt.close()
        print("  saved csda_kaon_logistic_probes.png")

    print(f"\nDone. All figures saved to {out_dir}")

    # ── Muon-only analysis (if requested) ──
    if args.include_muons:
        print("\n" + "="*70)
        print("MUON ANALYSIS (separate, >=180 wires)")
        print("="*70)

        if muon_latents is None:
            inference_dir = Path(cfg["output"]["inference_dir"]) / model_name
            print(f"⚠ Muon latents not found at {inference_dir / 'muon.npz'}")
            print("Run: python scripts/run_inference.py --config {config} --include-muons")
            return

        if muon_features is None:
            print("⚠ No muon features found in features.pkl")
            print("Run: python scripts/compute_features.py --config {config} --include-muons")
            return

        print(f"Loaded {len(muon_latents)} muon latents, {len(muon_features)} muon features")

        muon_out_dir = PROJECT_ROOT / "figs" / model_name / "latents-features-muon"
        muon_out_dir.mkdir(parents=True, exist_ok=True)

        if "correlation" in args.analyses:
            print("\n--- Muon Correlation Analysis ---")
            n_dims = muon_latents.shape[1]
            dim_cols = latent_dim_names(n_dims)
            # muon_features rows = muon_col only (N), muon_latents N — verify alignment
            if len(muon_features) != len(muon_latents):
                print(f"  ⚠ muon features ({len(muon_features)}) != latents ({len(muon_latents)}), truncating to min")
                n = min(len(muon_features), len(muon_latents))
                muon_features = muon_features.iloc[:n].copy()
                muon_latents_corr = muon_latents[:n]
            else:
                muon_features = muon_features.copy()
                muon_latents_corr = muon_latents
            for j, col in enumerate(dim_cols):
                muon_features[col] = muon_latents_corr[:, j]

            calo = [f for f in CALO if f in muon_features.columns]
            topo = [f for f in TOPO if f in muon_features.columns]
            all_feats = calo + topo

            corr_matrix = np.zeros((len(all_feats), n_dims))
            for i, feat in enumerate(all_feats):
                for j, lat in enumerate(dim_cols):
                    valid = muon_features[[feat, lat]].notna().all(axis=1)
                    if valid.sum() > 2:
                        rho, _ = spearmanr(muon_features.loc[valid, feat], muon_features.loc[valid, lat])
                        corr_matrix[i, j] = rho

            fig, ax = plt.subplots(figsize=(DOUBLE_COL, len(all_feats) * 0.65 + 1))
            sns.heatmap(
                corr_matrix,
                xticklabels=dim_cols,
                yticklabels=all_feats,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                cbar_kws={"label": "Spearman $\\rho$"},
                annot=True, fmt=".2f", annot_kws={"size": 10},
                ax=ax,
            )
            ax.axhline(y=len(calo), color="black", linewidth=2)
            ax.set_xlabel("Latent dimension")
            ax.set_ylabel("Feature")
            plt.tight_layout()
            _savefig(muon_out_dir / "disentanglement_heatmap.png")
            plt.close()
            print("  saved disentanglement_heatmap.png")

            # ── Muon Feature-to-feature correlation ──
            feat_corr_muon = muon_features[all_feats].corr(method="spearman")
            fig_f, ax_f = plt.subplots(figsize=(DOUBLE_COL, len(all_feats) * 0.75 + 1))
            sns.heatmap(
                feat_corr_muon,
                cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                cbar_kws={"label": "Spearman $\\rho$"},
                annot=True, fmt=".2f", annot_kws={"size": 10},
                ax=ax_f,
            )
            plt.tight_layout()
            _savefig(muon_out_dir / "feature_correlation.png")
            plt.close()
            print("  saved feature_correlation.png")

        if "traversal" in args.analyses:
            print("\n--- Muon Latent Traversal ---")
            device = torch.device("mps" if torch.backends.mps.is_available() else
                                  "cuda" if torch.cuda.is_available() else "cpu")
            model = VAE(
                input_hw=tuple(cfg["model"]["input_hw"]),
                latent=cfg["model"]["latent"],
                channels=cfg["model"]["channels"],
                kernel=cfg["model"]["kernel"],
                stride=cfg["model"]["stride"],
                padding=cfg["model"]["padding"],
                activation=cfg["model"]["activation"],
                p_enc=cfg["model"]["dropout"],
            ).to(device)
            model.load_state_dict(torch.load(
                Path(cfg["output"]["dir"]) / (model_name + ".pt"), map_location=device
            ))
            model.eval()

            mu  = muon_latents.mean(axis=0)
            sig = muon_latents.std(axis=0)

            N_STEPS = 9
            N_DIMS  = cfg["model"]["latent"]
            CHANNEL = 0
            steps   = np.linspace(-2, 2, N_STEPS)

            fig, axes = plt.subplots(
                N_DIMS, N_STEPS,
                figsize=(N_STEPS * 1.8, N_DIMS * 2.2),
                constrained_layout=True,
            )
            if N_DIMS == 1:
                axes = axes[np.newaxis, :]

            with torch.no_grad():
                for i in range(N_DIMS):
                    z_batch = np.tile(mu, (N_STEPS, 1))
                    z_batch[:, i] = mu[i] + steps * sig[i]
                    z_tensor = torch.tensor(z_batch, dtype=torch.float32).to(device)
                    recon    = model.decode(z_tensor).cpu().numpy()
                    images   = recon[:, CHANNEL, :, :]
                    vmax = images.max()
                    for j in range(N_STEPS):
                        ax = axes[i, j]
                        ax.imshow(images[j], origin="lower", cmap="viridis", vmin=0, vmax=vmax)
                        ax.axis("off")
                        if j == 0:
                            ax.set_ylabel(rf"$z_{{{i}}}$", rotation=0, labelpad=28, va="center")
                        if i == 0:
                            ax.set_title(f"{steps[j]:+.1f}$\\sigma$", fontsize=8)

            _savefig(muon_out_dir / "latent_traversal.png")
            plt.close()
            print("  saved latent_traversal.png")

        # ── Muon logistic: binary probes vs proton and kaon ──
        if "logistic" in args.analyses:
            print("\n--- Muon Logistic Probes (binary) ---")
            train_l_log, val_l_log, kaon_l_log = load_latents(cfg, model_name)
            proton_latents_log = np.vstack([train_l_log, val_l_log])

            cv_log = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            lr_pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ])

            for label_b, X_b, name_b in [
                ("muon vs proton", np.vstack([muon_latents, proton_latents_log]),
                 np.array([1]*len(muon_latents) + [0]*len(proton_latents_log))),
                ("muon vs kaon",   np.vstack([muon_latents, kaon_l_log]),
                 np.array([1]*len(muon_latents) + [0]*len(kaon_l_log))),
            ]:
                proba = cross_val_predict(lr_pipe, X_b, name_b, cv=cv_log, method="predict_proba")[:, 1]
                auc   = roc_auc_score(name_b, proba)
                acc   = accuracy_score(name_b, (proba > 0.5).astype(int))
                print(f"  {label_b:20s}  AUC={auc:.3f}  Acc={acc:.3f}")

            # Bar chart
            results_log = {}
            for label_b, X_b, y_b in [
                ("muon vs proton", np.vstack([muon_latents, proton_latents_log]),
                 np.array([1]*len(muon_latents) + [0]*len(proton_latents_log))),
                ("muon vs kaon",   np.vstack([muon_latents, kaon_l_log]),
                 np.array([1]*len(muon_latents) + [0]*len(kaon_l_log))),
                ("proton vs kaon", np.vstack([proton_latents_log, kaon_l_log]),
                 np.array([0]*len(proton_latents_log) + [1]*len(kaon_l_log))),
            ]:
                proba = cross_val_predict(lr_pipe, X_b, y_b, cv=cv_log, method="predict_proba")[:, 1]
                results_log[label_b] = roc_auc_score(y_b, proba)

            fig, ax = plt.subplots(figsize=(7.0, 3.5))
            colours = [PURPLE, PURPLE, BLUE]
            bars = ax.bar(list(results_log.keys()), list(results_log.values()),
                          color=colours, edgecolor="white", width=0.5)
            ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
            ax.set_ylim(0.4, 1.0)
            ax.set_ylabel("AUC-ROC")
            for bar, val in zip(bars, results_log.values()):
                ax.text(bar.get_x() + bar.get_width()/2, val + 0.01,
                        f"{val:.3f}", ha="center", fontsize=8, fontweight="bold")
            plt.tight_layout()
            _savefig(muon_out_dir / "muon_logistic_probes.png")
            plt.close()
            print("  saved muon_logistic_probes.png")

        if "nonlinear" in args.analyses:
            print("\n--- Muon Non-linear Analysis (p/k/muon combined) ---")
            train_latents_nl, val_latents_nl, kaon_latents_nl = load_latents(cfg, model_name)
            features_nl, index_nl = load_features_and_splits(cfg, features_path)

            all_proton_nl = features_nl[features_nl["particle_type"] == "proton"]
            all_kaon_nl   = features_nl[features_nl["particle_type"] == "kaon"]
            train_f_nl    = all_proton_nl.iloc[index_nl["train_idx"]]
            val_f_nl      = all_proton_nl.iloc[index_nl["val_idx"]]

            X_nl = np.vstack([train_latents_nl, val_latents_nl, kaon_latents_nl, muon_latents])
            features_nl_df = pd.concat([train_f_nl, val_f_nl, all_kaon_nl, muon_features], ignore_index=True)
            particle_labels_nl = np.array(
                [0] * len(train_latents_nl) + [0] * len(val_latents_nl) +
                [1] * len(kaon_latents_nl) + [2] * len(muon_latents)
            )

            n_dims = muon_latents.shape[1]
            calo = [f for f in CALO if f in features_nl_df.columns]
            topo = [f for f in TOPO if f in features_nl_df.columns]
            all_feats = calo + topo

            from sklearn.model_selection import KFold
            cv_nl = KFold(n_splits=5, shuffle=True, random_state=42)
            linear_pipeline = Pipeline([("scaler", StandardScaler()), ("ridge", Ridge(alpha=1.0))])

            print(f"  N = {len(X_nl)} (p/k/muon combined), {len(all_feats)} features\n")

            records = []
            for i, feat in enumerate(all_feats, 1):
                category = "calorimetry" if feat in calo else "topology"
                y    = features_nl_df[feat].values.astype(float)
                mask = np.isfinite(y)
                Xm, ym, lm = X_nl[mask], y[mask], particle_labels_nl[mask]

                if mask.sum() < 10:
                    continue

                splits = list(cv_nl.split(Xm))
                r2_linear = max(0.0, cross_val_score(linear_pipeline, Xm, ym, cv=splits, scoring="r2").mean())
                mlp_pipeline = make_mlp_pipeline()
                r2_mlp = max(0.0, cross_val_score(mlp_pipeline, Xm, ym, cv=splits, scoring="r2").mean())
                gap = round(r2_mlp - r2_linear, 3)

                y_oof_lin_nl = cross_val_predict(linear_pipeline,    Xm, ym, cv=splits)
                y_oof_mlp_nl = cross_val_predict(make_mlp_pipeline(), Xm, ym, cv=splits)

                p_mask_nl = lm == 0
                k_mask_nl = lm == 1
                m_mask_nl = lm == 2
                r2_p = max(0.0, r2_score(ym[p_mask_nl], y_oof_mlp_nl[p_mask_nl]))
                r2_k = max(0.0, r2_score(ym[k_mask_nl], y_oof_mlp_nl[k_mask_nl]))
                r2_m = max(0.0, r2_score(ym[m_mask_nl], y_oof_mlp_nl[m_mask_nl])) if m_mask_nl.sum() > 1 else np.nan

                def _fm_nl(v):
                    return f"{v:.1f}%" if np.isfinite(v) else "N/A"
                def _rnf_nl(v, d=4):
                    return round(float(v), d) if np.isfinite(float(v)) else np.nan

                lin_mae_nl,  lin_mse_nl,  lin_rmse_nl,  lin_mape_nl  = compute_regression_metrics(ym, y_oof_lin_nl)
                mlp_mae_nl,  mlp_mse_nl,  mlp_rmse_nl,  mlp_mape_nl  = compute_regression_metrics(ym, y_oof_mlp_nl)

                (lin_mae_p_nl, _, lin_rmse_p_nl, lin_mape_p_nl), (mlp_mae_p_nl, _, mlp_rmse_p_nl, mlp_mape_p_nl) \
                    = _particle_metrics(p_mask_nl, ym, y_oof_lin_nl, y_oof_mlp_nl)
                (lin_mae_k_nl, _, lin_rmse_k_nl, lin_mape_k_nl), (mlp_mae_k_nl, _, mlp_rmse_k_nl, mlp_mape_k_nl) \
                    = _particle_metrics(k_mask_nl, ym, y_oof_lin_nl, y_oof_mlp_nl)
                (lin_mae_m_nl, _, lin_rmse_m_nl, lin_mape_m_nl), (mlp_mae_m_nl, _, mlp_rmse_m_nl, mlp_mape_m_nl) \
                    = _particle_metrics(m_mask_nl, ym, y_oof_lin_nl, y_oof_mlp_nl)

                print(f"  {feat:25s}  linear={r2_linear:.3f}  mlp={r2_mlp:.3f}  gap={gap:+.3f}  "
                      f"p={r2_p:.3f}  k={r2_k:.3f}  m={r2_m:.3f}")
                print(f"  {'':25s}  Linear  MAE={lin_mae_nl:.4f}  RMSE={lin_rmse_nl:.4f}  MAPE={_fm_nl(lin_mape_nl)}")
                print(f"  {'':25s}  MLP     MAE={mlp_mae_nl:.4f}  RMSE={mlp_rmse_nl:.4f}  MAPE={_fm_nl(mlp_mape_nl)}")
                print(f"  {'':25s}  proton  Lin RMSE={lin_rmse_p_nl:.4f}  MLP RMSE={mlp_rmse_p_nl:.4f}")
                print(f"  {'':25s}  kaon    Lin RMSE={lin_rmse_k_nl:.4f}  MLP RMSE={mlp_rmse_k_nl:.4f}")
                if m_mask_nl.sum() > 1:
                    print(f"  {'':25s}  muon    Lin RMSE={lin_rmse_m_nl:.4f}  MLP RMSE={mlp_rmse_m_nl:.4f}")

                records.append({
                    "feature": feat, "category": category,
                    "linear_r2": round(r2_linear, 3), "mlp_r2": round(r2_mlp, 3), "gap": gap,
                    "r2_proton": round(r2_p, 3), "r2_kaon": round(r2_k, 3),
                    "r2_muon": round(r2_m, 3) if not np.isnan(r2_m) else np.nan,
                    # ── overall error metrics ──
                    "linear_mae":  _rnf_nl(lin_mae_nl),  "linear_mse":  _rnf_nl(lin_mse_nl),
                    "linear_rmse": _rnf_nl(lin_rmse_nl), "linear_mape": _rnf_nl(lin_mape_nl, 2),
                    "mlp_mae":     _rnf_nl(mlp_mae_nl),  "mlp_mse":     _rnf_nl(mlp_mse_nl),
                    "mlp_rmse":    _rnf_nl(mlp_rmse_nl), "mlp_mape":    _rnf_nl(mlp_mape_nl, 2),
                    # ── per-particle RMSE & MAE ──
                    "linear_rmse_proton": _rnf_nl(lin_rmse_p_nl), "mlp_rmse_proton": _rnf_nl(mlp_rmse_p_nl),
                    "linear_rmse_kaon":   _rnf_nl(lin_rmse_k_nl), "mlp_rmse_kaon":   _rnf_nl(mlp_rmse_k_nl),
                    "linear_rmse_muon":   _rnf_nl(lin_rmse_m_nl), "mlp_rmse_muon":   _rnf_nl(mlp_rmse_m_nl),
                    "linear_mae_proton":  _rnf_nl(lin_mae_p_nl),  "mlp_mae_proton":  _rnf_nl(mlp_mae_p_nl),
                    "linear_mae_kaon":    _rnf_nl(lin_mae_k_nl),  "mlp_mae_kaon":    _rnf_nl(mlp_mae_k_nl),
                    "linear_mae_muon":    _rnf_nl(lin_mae_m_nl),  "mlp_mae_muon":    _rnf_nl(mlp_mae_m_nl),
                    # ── per-particle MAPE ──
                    "linear_mape_proton": _rnf_nl(lin_mape_p_nl, 2), "mlp_mape_proton": _rnf_nl(mlp_mape_p_nl, 2),
                    "linear_mape_kaon":   _rnf_nl(lin_mape_k_nl, 2), "mlp_mape_kaon":   _rnf_nl(mlp_mape_k_nl, 2),
                    "linear_mape_muon":   _rnf_nl(lin_mape_m_nl, 2), "mlp_mape_muon":   _rnf_nl(mlp_mape_m_nl, 2),
                })

            results = pd.DataFrame(records).sort_values("gap", ascending=False)

            # Plot
            calo_df = results[results["category"] == "calorimetry"].sort_values("gap", ascending=True)
            topo_df = results[results["category"] == "topology"].sort_values("gap", ascending=True)

            _calo_h_nl = len(calo_df) * 0.50 + 1.5
            _topo_h_nl = len(topo_df) * 0.50 + 1.5
            fig  = plt.figure(figsize=(7.0, _calo_h_nl + _topo_h_nl + 0.5))
            gs   = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[_calo_h_nl, _topo_h_nl], hspace=0.15, top=0.95)
            ax_c = fig.add_subplot(gs[0])
            ax_t = fig.add_subplot(gs[1], sharex=ax_c)

            for ax, df, title in zip([ax_c, ax_t], [calo_df, topo_df], ["Calorimetry", "Topology"]):
                if len(df) == 0:
                    continue
                y_pos = np.arange(len(df))
                h = 0.6
                linear_vals = df["linear_r2"].values
                gap_vals = df["gap"].clip(lower=0).values
                mlp_vals = df["mlp_r2"].values
                feats = df["feature"].values

                ax.barh(y_pos, linear_vals, h, color=BLUE, alpha=0.9, label="Linear $R^2$")
                ax.barh(y_pos, gap_vals, h, color=ORANGE, alpha=0.9, left=linear_vals, label="Non-linear gain")

                ax.set_yticks(y_pos)
                ax.set_yticklabels(feats)
                ax.set_xlabel("Cross-validated $R^2$")
                ax.set_title(title, fontsize=9, fontweight="semibold", loc="left", pad=4)
                ax.grid(axis="x", alpha=0.3, linewidth=0.7)
                ax.spines[["top", "right"]].set_visible(False)
                ax.legend(
                    loc="lower right", frameon=True, framealpha=0.85, edgecolor="0.8",
                    handlelength=1.0, handletextpad=0.4, borderpad=0.5,
                )

            _savefig(muon_out_dir / "nonlinear_r2.png")
            plt.close()
            print("  saved nonlinear_r2.png")

        print(f"\nMuon analysis saved to {muon_out_dir}")


if __name__ == "__main__":
    main()
