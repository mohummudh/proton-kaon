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
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import yaml
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_regression
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, r2_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.configVAE import VAE  # noqa: E402

# ── feature groups ─────────────────────────────────────────────────────────────
CALO = [
    "total_adc", "mean_adc", "median_adc", "max_adc", "std_adc", "adc_entropy",
    "bragg_peak_height", "bragg_peak_position", "bragg_peak_ratio", "bragg_peak_to_median",
    "end_vs_start_ratio", "last_quartile_mean", "first_quartile_mean",
    "bragg_rise_slope", "peak_integral_fraction", "bragg_peak_width",
    "profile_cv", "monotonic_rise_fraction", "relative_peak_energy",
    "profile_skewness", "profile_kurtosis",
]
TOPO = ["height", "n_pixels", "fill_fraction", "solidity", "n_local_maxima"]

BLUE   = "#4C78A8"
ORANGE = "#F58518"


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


# ── analysis 1: correlation ────────────────────────────────────────────────────

def run_correlation(cfg, model_name, features_path, out_dir):
    print("\n=== Correlation analysis ===")

    train_latents, val_latents, kaon_latents = load_latents(cfg, model_name)
    n_dims   = train_latents.shape[1]
    dim_cols = latent_dim_names(n_dims)

    features, index = load_features_and_splits(cfg, features_path)

    all_proton = features[features["particle_type"] == "proton"]
    all_kaon   = features[features["particle_type"] == "kaon"]

    latent_z = np.zeros((len(features), n_dims))
    latent_z[all_proton.index[index["train_idx"]], :] = train_latents
    latent_z[all_proton.index[index["val_idx"]],   :] = val_latents
    latent_z[all_kaon.index,                        :] = kaon_latents

    for j, col in enumerate(dim_cols):
        features[col] = latent_z[:, j]

    calo = [f for f in CALO if f in features.columns]
    topo = [f for f in TOPO if f in features.columns]
    all_feats = calo + topo

    # ── Spearman heatmap ──
    corr_matrix = np.zeros((len(all_feats), n_dims))
    for i, feat in enumerate(all_feats):
        for j, lat in enumerate(dim_cols):
            valid = features[[feat, lat]].notna().all(axis=1)
            if valid.sum() > 2:
                rho, _ = spearmanr(features.loc[valid, feat], features.loc[valid, lat])
                corr_matrix[i, j] = rho

    fig, ax = plt.subplots(figsize=(max(6, n_dims * 1.8), len(all_feats) * 0.55 + 1))
    sns.heatmap(
        corr_matrix,
        xticklabels=dim_cols,
        yticklabels=all_feats,
        cmap="RdBu_r", center=0, vmin=-1, vmax=1,
        cbar_kws={"label": "Spearman ρ"},
        annot=True, fmt=".2f", annot_kws={"size": 9},
        ax=ax,
    )
    ax.axhline(y=len(calo), color="black", linewidth=2)
    ax.set_title("VAE Latent Disentanglement: Feature Correlation", fontsize=12, weight="bold")
    ax.set_xlabel("Latent Dimensions", fontsize=11)
    ax.set_ylabel("Features", fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "disentanglement_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved disentanglement_heatmap.png")

    for j, lat in enumerate(dim_cols):
        c = np.abs(corr_matrix[:len(calo), j])
        t = np.abs(corr_matrix[len(calo):, j])
        print(f"  {lat}: calo mean |ρ|={c.mean():.3f}, topo mean |ρ|={t.mean():.3f}, "
              f"specificity={c.mean()/t.mean():.2f}x")

    # ── variance decomposition (linear R² per dim per category) ──
    records = []
    for i in range(n_dims):
        z = latent_z[:, i].reshape(-1, 1)
        for feat in all_feats:
            y = features[feat].values
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

    fig, ax = plt.subplots(figsize=(max(5, n_dims * 1.2), 4))
    summary.plot(
        kind="bar", stacked=True, ax=ax,
        color=["steelblue", "darkorange"],
        edgecolor="white", linewidth=0.5, width=0.5,
    )
    ax.set_xlabel("Latent dimension", fontsize=11)
    ax.set_ylabel("Mean R²  (linear, univariate)", fontsize=11)
    ax.set_title("Variance decomposition per latent dimension", fontsize=12)
    ax.set_xticklabels(dim_cols, rotation=0)
    ax.legend(title="Feature category", framealpha=0.8)
    ax.set_ylim(0, summary.values.sum(axis=1).max() * 1.15)
    for idx, (_, row) in enumerate(summary.iterrows()):
        total = row.sum()
        ax.text(idx, total + 0.001, f"{total:.3f}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(out_dir / "variance_decomposition.png", dpi=150, bbox_inches="tight")
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
                    ax.set_ylabel(f"z{i}", fontsize=11, rotation=0, labelpad=28, va="center")
                if i == 0:
                    ax.set_title(f"{steps[j]:+.1f}σ", fontsize=9)

    fig.suptitle(
        "Latent traversal — each row sweeps one z_i from −2σ to +2σ\n"
        "all other dimensions held at training mean",
        fontsize=11,
    )
    plt.savefig(out_dir / "latent_traversal.png", dpi=150, bbox_inches="tight")
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


def run_logistic(cfg, model_name, out_dir):
    print("\n=== Logistic regression ===")

    train_latents, val_latents, kaon_latents = load_latents(cfg, model_name)
    n_dims = val_latents.shape[1]

    X = np.concatenate([val_latents, kaon_latents], axis=0)
    y = np.concatenate([
        np.zeros(len(val_latents)),
        np.ones(len(kaon_latents)),
    ])
    print(f"  Protons (val): {len(val_latents)}, Kaons: {len(kaon_latents)}")

    subsets = _auto_subsets(n_dims)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])

    results    = {}
    pred_proba = {}
    for label, dims in subsets.items():
        X_sub = X[:, dims]
        proba = cross_val_predict(
            clf_pipeline, X_sub, y, cv=cv, method="predict_proba"
        )[:, 1]
        auc  = roc_auc_score(y, proba)
        acc  = accuracy_score(y, (proba > 0.5).astype(int))
        results[label]    = {"AUC": auc, "Accuracy": acc}
        pred_proba[label] = proba
        print(f"  {label:25s}  AUC={auc:.3f}  Acc={acc:.3f}")

    # ── event-level agreement: two best single-dim classifiers ──
    single_dim_labels = [f"z{i}" for i in range(n_dims)]
    single_aucs = [(lbl, results[lbl]["AUC"]) for lbl in single_dim_labels]
    top2 = sorted(single_aucs, key=lambda x: x[1], reverse=True)[:2]
    lbl_a, lbl_b = top2[0][0], top2[1][0]

    pred_a = (pred_proba[lbl_a] > 0.5).astype(int)
    pred_b = (pred_proba[lbl_b] > 0.5).astype(int)

    agree_correct = int(((pred_a == y) & (pred_b == y)).sum())
    a_only        = int(((pred_a == y) & (pred_b != y)).sum())
    b_only        = int(((pred_a != y) & (pred_b == y)).sum())
    both_wrong    = int(((pred_a != y) & (pred_b != y)).sum())
    N             = len(y)

    print(f"\n  Event-level agreement — {lbl_a} vs {lbl_b} (N={N}):")
    print(f"    Both correct:        {agree_correct:4d}  ({100*agree_correct/N:.1f}%)")
    print(f"    {lbl_a} only correct: {a_only:4d}  ({100*a_only/N:.1f}%)")
    print(f"    {lbl_b} only correct: {b_only:4d}  ({100*b_only/N:.1f}%)")
    print(f"    Both wrong:          {both_wrong:4d}  ({100*both_wrong/N:.1f}%)")

    # ── plots ──
    labels  = list(results.keys())
    aucs    = [results[l]["AUC"] for l in labels]
    palette = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(max(10, len(labels) * 0.9 + 4), 4))

    # AUC bar chart
    colours = [palette[i % 10] for i in range(len(labels))]
    axes[0].bar(labels, aucs, color=colours, edgecolor="white", width=0.55)
    axes[0].axhline(0.5, color="grey", linestyle="--", linewidth=1, label="Chance")
    axes[0].set_ylim(0.4, 1.0)
    axes[0].set_ylabel("AUC-ROC")
    axes[0].set_title("Linear probe AUC by latent subset")
    axes[0].tick_params(axis="x", rotation=30)
    for i, auc in enumerate(aucs):
        axes[0].text(i, auc + 0.01, f"{auc:.3f}", ha="center", fontsize=9)

    # event-level breakdown
    categories  = ["Both correct", f"{lbl_a} only", f"{lbl_b} only", "Both wrong"]
    counts      = [agree_correct, a_only, b_only, both_wrong]
    bar_colours = ["#4CAF50", "steelblue", "darkorange", "#e57373"]
    axes[1].bar(categories, counts, color=bar_colours, edgecolor="white", width=0.5)
    axes[1].set_ylabel("Number of events")
    axes[1].set_title(f"Where {lbl_a} and {lbl_b} agree / disagree")
    for i, c in enumerate(counts):
        axes[1].text(i, c + 0.5, str(c), ha="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(out_dir / "linear_probe.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved linear_probe.png")


# ── analysis 4: non-linear ────────────────────────────────────────────────────

def run_nonlinear(cfg, model_name, features_path, out_dir):
    print("\n=== Non-linear analysis ===")

    train_latents, val_latents, kaon_latents = load_latents(cfg, model_name)
    n_dims   = train_latents.shape[1]
    dim_cols = latent_dim_names(n_dims)
    imp_cols = [f"z{i}_imp" for i in range(n_dims)]

    features, index = load_features_and_splits(cfg, features_path)

    all_proton = features[features["particle_type"] == "proton"]
    all_kaon   = features[features["particle_type"] == "kaon"]

    train_features = all_proton.iloc[index["train_idx"]]
    val_features   = all_proton.iloc[index["val_idx"]]
    kaon_features  = all_kaon

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

        pipe_full = make_mlp_pipeline()
        pipe_full.fit(Xm, ym)
        y_pred = pipe_full.predict(Xm)

        p_mask = lm == 0
        k_mask = lm == 1
        r2_p = max(0.0, r2_score(ym[p_mask], y_pred[p_mask]))
        r2_k = max(0.0, r2_score(ym[k_mask], y_pred[k_mask]))

        imp = permutation_importance_mlp(pipe_full, Xm, ym)
        imp_str = "  ".join(f"z{j}:{imp[j]:.3f}" for j in range(n_dims))
        print(f"         proton R²={r2_p:.3f}  kaon R²={r2_k:.3f}  perm: {imp_str}\n")

        row = {
            "feature":   feat,
            "category":  category,
            "linear_r2": round(r2_linear, 3),
            "mlp_r2":    round(r2_mlp, 3),
            "gap":       gap,
            "r2_proton": round(r2_p, 3),
            "r2_kaon":   round(r2_k, 3),
            "gap_pk":    round(r2_p - r2_k, 3),
        }
        for j in range(n_dims):
            row[f"z{j}_imp"] = round(imp[j], 4)
        records.append(row)

    results = (
        pd.DataFrame(records)
        .sort_values("gap", ascending=False)
        .reset_index(drop=True)
    )
    print(results.to_string(index=False))

    # ── nonlinear_r2.png ──
    calo_df = results[results["category"] == "calorimetry"].sort_values("gap", ascending=True)
    topo_df = results[results["category"] == "topology"].sort_values("gap", ascending=True)

    PX_PER_FEAT = 0.38
    PAD         = 2.0
    calo_h = len(calo_df) * PX_PER_FEAT + PAD
    topo_h = len(topo_df) * PX_PER_FEAT + PAD

    fig  = plt.figure(figsize=(12, calo_h + topo_h + 1.0))
    gs   = gridspec.GridSpec(2, 1, figure=fig, height_ratios=[calo_h, topo_h], hspace=0.1, top=0.95)
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

        ax.barh(y_pos, linear_vals, h, color=BLUE,   alpha=0.9, label="Linear R²")
        ax.barh(y_pos, gap_vals,    h, color=ORANGE, alpha=0.9, left=linear_vals,
                label="Non-linear gain")

        for yi, (lin, gap_v, mlp) in enumerate(zip(linear_vals, gap_vals, mlp_vals)):
            ax.text(mlp + 0.008, yi, f"{mlp:.3f}", va="center", ha="left", fontsize=8)
            if gap_v > 0.04:
                ax.text(lin + gap_v / 2, yi, f"+{gap_v:.2f}",
                        va="center", ha="center", fontsize=7.5, color="white", fontweight="bold")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(feats, fontsize=9)
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Cross-validated R²", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="semibold", loc="left", pad=6)
        ax.grid(axis="x", alpha=0.3, linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.8)

    fig.suptitle(
        "Linear vs non-linear encoding of physics features in VAE latent space\n"
        "bar = MLP R²  |  blue = linearly accessible  |  orange = non-linear gain",
        fontsize=11, y=0.98,
    )
    plt.savefig(out_dir / "nonlinear_r2.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  saved nonlinear_r2.png")

    # ── permutation importance heatmaps ──
    rename = {f"z{i}_imp": f"z{i}" for i in range(n_dims)}
    for category in ["calorimetry", "topology"]:
        df_cat = (
            results[results["category"] == category]
            .set_index("feature")[imp_cols]
            .rename(columns=rename)
            .sort_values("z0")
        )
        fig, ax = plt.subplots(figsize=(max(4, n_dims * 1.1), len(df_cat) * 0.4 + 1))
        sns.heatmap(
            df_cat, annot=True, fmt=".3f", cmap="RdBu_r", center=0,
            linewidths=0.4, ax=ax,
            cbar_kws={"label": "R² drop on permutation"},
        )
        ax.set_title(f"Permutation importance — {category}", fontsize=11)
        ax.set_xlabel("Latent dimension")
        plt.tight_layout()
        fname = f"permutation_importance_{category}.png"
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
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
        fig, ax = plt.subplots(figsize=(max(4, n_dims * 1.1), len(sub) * 0.4 + 1))
        sns.heatmap(
            sub, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "MI (nats)"},
            vmin=0, vmax=vmax_mi,
        )
        ax.set_title(f"Per-dimension MI — {category}", fontsize=11)
        ax.set_xlabel("Latent dimension")
        plt.tight_layout()
        fname = f"mutual_information_{category}.png"
        plt.savefig(out_dir / fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  saved {fname}")


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Latent-feature analysis for a VAE model.")
    parser.add_argument("--config",    required=True, help="Path to model YAML config")
    parser.add_argument(
        "--analyses", nargs="+",
        choices=["correlation", "traversal", "logistic", "nonlinear"],
        default=["correlation", "traversal", "logistic", "nonlinear"],
        help="Which analyses to run (default: all)",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Path to features.pkl (overrides config; falls back to data.features_path in config)",
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

    out_dir = PROJECT_ROOT / "figs" / "latents-features" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    if "correlation" in args.analyses:
        run_correlation(cfg, model_name, features_path, out_dir)

    if "traversal" in args.analyses:
        run_traversal(cfg, model_name, out_dir)

    if "logistic" in args.analyses:
        run_logistic(cfg, model_name, out_dir)

    if "nonlinear" in args.analyses:
        run_nonlinear(cfg, model_name, features_path, out_dir)

    print(f"\nDone. All figures saved to {out_dir}")


if __name__ == "__main__":
    main()
