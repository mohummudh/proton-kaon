#!/usr/bin/env python3
"""
scripts/extra/_sweep_measure.py

Shared measurement layer for the training sweeps. Two sweeps ask different
questions of the same trained models:

    plot_split_sweep.py    how much DATA does the latent space need?
                           (configs/sweep_split_pool8227_seeded.yaml)
    plot_latent_sweep.py   how much CAPACITY does it need?
                           (configs/sweep_latent_pool8227_tr50.yaml)

Their plots differ, but every number behind them is computed the same way, so it
is computed here once. The alternative — a measurement block per plotting script —
is how two scripts end up quietly reporting different things under the same name,
which is the failure this module exists to prevent. It is the same reason the
feature-AUC probe lives in analyse_latents.make_feature_auc_probe and is imported
rather than copied.
"""

import copy
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.compose import TransformedTargetRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from _beam_data import load_beam_data
from cluster_latents import fit_clusters, score_clusters

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.train.naming import model_name as build_model_name  # noqa: E402
from scripts.analyse_latents import make_feature_auc_probe  # noqa: E402

# The two proxies the paper leads with. Keys are feature columns, values the
# reader-facing names used everywhere else (see _beam_data.PROXY_LABELS).
PROXIES = {"mean_adc": "Calorimetry Proxy", "solidity": "Topology Proxy"}

# A latent dimension counts as active when the variance of its posterior mean
# across the dataset clears this. The conventional beta-VAE threshold: a collapsed
# dimension has mu pinned near the prior mean for every event, so its variance goes
# to zero, while a used dimension carries O(1) variance. Threshold-dependent, which
# is why n_dims_95var is reported alongside it as a threshold-free companion.
ACTIVE_VAR_THRESHOLD = 0.01


def rung_configs(sweep_path: Path) -> list:
    """Rebuild every run's config the same way run_sweep.py does.

    Deliberately re-derived from the sweep YAML rather than read from the generated
    per-run configs: those live on the model drive and are rewritten by the next
    sweep, so reading them would make the answer depend on whatever ran last.

    Returns [(params, cfg)], params being the grid dict for that run. Works for any
    grid keys, so it handles both sweeps and the older unseeded configs.
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


def run_id(cfg: dict) -> dict:
    """The identifying axes of a run: split tag, latent dim, seed."""
    return {"tag": cfg["data"].get("tag"),
            "latent": cfg["model"]["latent"],
            "seed": (cfg.get("train") or {}).get("seed")}


def read_log(cfg: dict) -> dict:
    """Best-epoch losses from a run's training log.

    `best val` is the epoch whose weights were actually saved: train() restores
    best_state before returning, so the checkpoint is the argmin of val loss, not
    the last epoch. Reporting the last epoch would understate every run by the
    patience window.
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
    # KL's share of the objective. Worth carrying because it is what explains the
    # absence of posterior collapse: the reconstruction term is summed over pixels
    # while the KL is a mean over dimensions, so beta = 0.5 leaves KL a few percent
    # of the total and the model is under almost no pressure to compress.
    beta = cfg["train"].get("beta", 1.0)
    denom = best["val_recon"] + beta * best["val_kl"]
    return {
        "n_train": log["dataset"]["n_train"],
        "n_val": log["dataset"]["n_val"],
        "epochs": len(hist),
        "best_epoch": best["epoch"],
        "val_loss": best["val_loss"],
        "val_recon": best["val_recon"],
        "val_kl": best["val_kl"],
        "kl_share": float(beta * best["val_kl"] / denom) if denom else float("nan"),
        "train_loss": best["train_loss"],
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
    set, and those shrink across a split sweep. Comparing them compares performance
    on different samples, which is enough to manufacture a bump that looks like a
    property of the model. This evaluates every model on one fixed set instead.

    It was used to establish that the differing validation sets do NOT bias the
    split sweep (agreement better than 0.5% at every rung). Keep in mind it is the
    unweighted per-pixel MSE from src/inference/inference.py, not the weighted
    objective the training log reports — a much less sensitive quantity, so it is a
    check on sample bias and not a substitute for the logged loss.

    Costs no forward passes: run_inference.py already stored a per-event MSE for
    every event in the pool, so this is a lookup.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    if not (inf_dir / "train.npz").exists():
        return np.nan
    ss = np.load(inf_dir / "species_split.npz")
    n = species_sizes()

    # Protons are stored split across train.npz/val.npz in split order; rebuild an
    # array addressable by within-proton index. NaN-filled so that selecting an
    # event outside the pool surfaces as NaN rather than as a wrong number.
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


def species_recon_error(cfg: dict) -> dict:
    """Mean per-event reconstruction MSE per species, train and validation.

    The pooled loss in the training log hides that the three species are not
    equally hard: kaons reconstruct roughly 6.5-7.7x worse than protons at every
    latent dimension, which is the single largest effect in the sweep and invisible
    in any pooled number.

    Carrying train as well as val is what makes the val/train ratio available, and
    that ratio turns out to be species-specific — the interesting part.

    This is the unweighted per-pixel MSE from src/inference/inference.py, so it is
    not on the same scale as the weighted objective the log reports. Across the
    latent sweep it nevertheless tracks it closely, because the effect is large;
    across the split sweep it did not, because there the effect was ~6% and the
    unweighted mean is dominated by easily-reconstructed background. Compare within
    a sweep, never against a logged loss.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    if not (inf_dir / "train.npz").exists():
        return {}
    ss = np.load(inf_dir / "species_split.npz")
    kaon = np.load(inf_dir / "kaon.npz")["re"]
    muon = np.load(inf_dir / "muon.npz")["re"]
    out = {
        "re_proton_train": float(np.load(inf_dir / "train.npz")["re"].mean()),
        "re_proton_val": float(np.load(inf_dir / "val.npz")["re"].mean()),
        "re_kaon_train": float(kaon[ss["k_train_idx"]].mean()),
        "re_kaon_val": float(kaon[ss["k_val_idx"]].mean()),
        "re_muon_train": float(muon[ss["m_train_idx"]].mean()),
        "re_muon_val": float(muon[ss["m_val_idx"]].mean()),
    }
    for sp in ("proton", "kaon", "muon"):
        tr = out[f"re_{sp}_train"]
        out[f"re_{sp}_ratio"] = out[f"re_{sp}_val"] / tr if tr else float("nan")
    # Equal-weight mean over species: a pooled reconstruction number that does NOT
    # depend on the validation set's species mixture. Needed because kaons
    # reconstruct ~7x worse than protons, so a pooled mean over an unbalanced
    # validation set says as much about its composition as about the model. The
    # sweep's own validation sets are exactly balanced and so unaffected, but the
    # bal9419 model's is 40/28/32, which drops its pooled loss by ~150 relative to a
    # balanced set at identical per-species performance.
    out["re_balanced_val"] = float(np.mean([out[f"re_{sp}_val"]
                                            for sp in ("proton", "kaon", "muon")]))
    return out


def val_row_indices(cfg: dict, df: pd.DataFrame) -> dict:
    """Row positions of each species' VALIDATION events within load_beam_data's Z.

    load_beam_data stacks [proton_train, proton_val, kaon_all, muon_all] and orders
    the proton block by concat(p_train_idx, p_val_idx), so proton validation is the
    tail of the proton block while kaon and muon are indexed directly by their
    within-species validation indices. Getting this wrong silently mispairs rows
    rather than raising, which is why it is derived rather than assumed.
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


def active_dims(Z: np.ndarray) -> dict:
    """How much of the latent space the model actually uses.

    "Latent dim 128" states what was made available, not what is used, so a capacity
    sweep needs a measure of the latter. Three are returned because the obvious one
    turned out not to work on these models.

      participation_ratio  (sum var)^2 / sum(var^2) over the per-dimension variances
                     of mu: a continuous effective dimension count, scale-free and
                     threshold-free. This is the one to read.
      n_dims_95var   how many dimensions, largest-variance first, hold 95% of the
                     total variance in mu. Also threshold-free, coarser.
      n_active_dims  dimensions with Var(mu) > ACTIVE_VAR_THRESHOLD.

    WHY n_active_dims IS NOT THE HEADLINE, THOUGH THE LITERATURE USES IT
        It was expected to be: a beta-VAE handed surplus capacity normally switches
        it off, pinning those dimensions' posterior mean to a constant so they cost
        no KL, and counting the survivors is the standard diagnostic. On these
        models it is saturated and uninformative — it equals the nominal latent dim
        at every point measured, because NOTHING collapses. At latent 64 the
        smallest per-dimension variance is still 0.27, twenty-seven times the
        conventional 0.01 threshold.

        The reason is the loss balance, not the latent size. The reconstruction term
        is a weighted MSE *summed* over 2x48x48 pixels while the KL is a mean over
        dimensions, so with beta = 0.5 the KL is between 0.5% (latent 4) and 5.4%
        (latent 64) of the objective. There is essentially no pressure to compress,
        which makes this a lightly regularised autoencoder rather than a
        tightly-constrained beta-VAE, and posterior collapse simply does not arise.

        n_active_dims is still returned, because "nothing collapses" is a real
        finding about the model and this is the number that shows it.
    """
    var = Z.var(axis=0)
    order = np.sort(var)[::-1]
    total = order.sum()
    if total <= 0:
        return {"participation_ratio": 0.0, "n_dims_95var": 0,
                "n_active_dims": 0, "latent_var_total": 0.0}
    return {
        "participation_ratio": float(total ** 2 / (var ** 2).sum()),
        "n_dims_95var": int(np.searchsorted(np.cumsum(order) / total, 0.95) + 1),
        "n_active_dims": int((var > ACTIVE_VAR_THRESHOLD).sum()),
        "latent_var_total": float(total),
    }


def regression_scores(X, v, seed=0):
    """Cross-validated R^2 predicting the proxy VALUE from the latents.

    Returns (r2_linear, r2_mlp, spearman_of_linear). The PAIR is the point: ridge
    measures what a linear readout can reach, the MLP what is present at all, and
    the difference is the non-linearly encoded part. Reporting the MLP alone cannot
    separate "the latent space encodes a lot" from "the probe is flexible".

    Lives here rather than in a plotting script because three consumers need it --
    both sweeps and plot_proxy_auc.py -- and a second copy is how two of them end up
    reporting different numbers under the same name.

    THE TARGET MUST BE SCALED, AND THAT IS NOT COSMETIC
        MLPRegressor optimises a raw squared error at a fixed default learning rate,
        so an unscaled target silently decides whether it can fit at all. The two
        proxies differ by two orders of magnitude in spread -- solidity sd 0.09,
        mean_adc sd 15 -- and on solidity the raw-target version simply did not
        learn: R^2 = -1.2 at latent 128, worse than predicting the mean, while ridge
        on the same data gave +0.55. Scaling y through TransformedTargetRegressor
        fixes it (+0.37), and early stopping recovers the rest (+0.56). Early
        stopping also removes the non-convergence that was hitting 3 folds in 5 on
        mean_adc.

        Ridge is scale-free and so was unaffected, which is what made the bug
        visible: a smooth ridge curve beside an erratic, negative MLP curve is an
        optimisation failure, never a property of the representation. Do not run
        this under a blanket warnings filter -- the ConvergenceWarnings are the
        early symptom.
    """
    cv = KFold(5, shuffle=True, random_state=42)
    ridge = Pipeline([("s", StandardScaler()), ("m", Ridge(alpha=1.0))])
    mlp = TransformedTargetRegressor(
        regressor=Pipeline([
            ("s", StandardScaler()),
            ("m", MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=2000,
                               early_stopping=True, n_iter_no_change=15,
                               random_state=seed)),
        ]),
        transformer=StandardScaler(),
    )
    pred_l = cross_val_predict(ridge, X, v, cv=cv)
    pred_n = cross_val_predict(mlp, X, v, cv=cv)
    return (float(r2_score(v, pred_l)), float(r2_score(v, pred_n)),
            float(spearmanr(pred_l, v).statistic))


def proxy_r2(cfg: dict, Z, df, seed: int = 0) -> dict:
    """r2_linear / r2_mlp per (proxy, species), on validation events only."""
    out = {}
    rows = val_row_indices(cfg, df)
    for feat in PROXIES:
        for sp, idx in rows.items():
            sub = df.iloc[idx].reset_index(drop=True)
            if feat not in sub.columns:
                continue
            v = sub[feat].to_numpy(float)
            m = np.isfinite(v)
            if m.sum() < 50:
                continue
            r2l, r2n, _ = regression_scores(Z[idx][m], v[m], seed=seed)
            out[f"r2lin_{feat}_{sp}"] = r2l
            out[f"r2mlp_{feat}_{sp}"] = r2n
    return out


def measure_model(cfg: dict, seed: int = 0, do_clustering: bool = True,
                  do_proxies: bool = True, do_r2: bool = False) -> dict:
    """Active dimensions, clustering agreement and proxy AUCs for one model.

    Labels are used only to score the clustering, never to fit it — same protocol
    as cluster_latents.py, so the numbers are comparable to the paper model's.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    if not (inf_dir / "train.npz").exists():
        return {}
    Z, df = load_beam_data(cfg)
    out = active_dims(Z)
    out.update(species_recon_error(cfg))

    if do_clustering:
        labels, _ = fit_clusters(Z, 3, seed=seed)
        out.update(score_clusters(labels, df["species"], 3))

    if do_proxies:
        # Validation events only: the question is whether the latent space encodes
        # the proxy for events it was not fitted on.
        probe = make_feature_auc_probe()
        rows = val_row_indices(cfg, df)
        for feat in PROXIES:
            for sp, idx in rows.items():
                auc, _ = probe(Z[idx], df.iloc[idx].reset_index(drop=True), feat)
                if auc is not None:
                    out[f"auc_{feat}_{sp}"] = auc

    if do_r2:
        out.update(proxy_r2(cfg, Z, df, seed=seed))
    return out


def write_local_configs(rungs: list, out_dir: Path) -> None:
    """Per-run configs carrying LOCAL paths, for run_inference.py and re-analysis.

    Needed because run_sweep.py rewrites data.path and output.dir with the remote
    machine's paths in the configs it leaves under <models>/sweep_configs/, so those
    point at /home/mohammed/... and are useless locally.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    header = ("# Generated by a plot_*_sweep.py --write-configs run.\n"
              "# LOCAL paths, unlike the run_sweep.py-generated configs under\n"
              "# <models>/sweep_configs/, which carry the remote's paths.\n")
    # Only name the axes the sweep actually varies. Including a constant axis would
    # rename the split sweep's already-committed configs (pool8227_tr50_seed0 ->
    # ..._latent8_seed0) and leave two files per run behind after a regeneration.
    idents = [run_id(cfg) for _, cfg in rungs]
    vary_tag = len({i["tag"] for i in idents}) > 1
    vary_latent = len({i["latent"] for i in idents}) > 1
    for ident, (_, cfg) in zip(idents, rungs):
        stem = str(ident["tag"])
        if vary_latent or not vary_tag:
            stem += f"_latent{ident['latent']}" if vary_latent else ""
        if ident["seed"] is not None:
            stem += f"_seed{ident['seed']}"
        with open(out_dir / f"{stem}.yaml", "w") as fh:
            fh.write(header)
            yaml.safe_dump(cfg, fh, sort_keys=False)
    print(f"  wrote {len(rungs)} configs to {out_dir}")


# ── plotting helpers ──────────────────────────────────────────────────────────

def agg(df: pd.DataFrame, col: str, x: str) -> pd.DataFrame:
    """Mean and across-seed sd of `col` at each value of `x`."""
    if col not in df.columns:
        return pd.DataFrame()
    sub = df[df[col].notna()]
    if sub.empty:
        return sub
    g = sub.groupby(x)[col]
    return pd.DataFrame({x: g.mean().index, "mean": g.mean().values,
                         "sd": g.std(ddof=1).values, "n": g.size().values})


def plot_r2_sweep(df, out_dir, x: str, xlabel: str, stem: str, savefig, apply_style,
                  COLOURS, DISPLAY, SPECIES, SINGLE_COL, DOUBLE_COL, logx=False):
    """R^2 of latent -> proxy across a sweep: what is present, and what is linear.

    A separate figure from the sweep's main one because it carries a different
    claim. The main figure asks whether the latent space encodes the proxies at all;
    this asks how much of that encoding a LINEAR readout can reach, which is the
    form the paper's "positional relationships" claim actually takes.

    Filled markers and solid lines are the MLP, open markers and dotted lines the
    ridge. The vertical distance between a species' two curves is the non-linearly
    encoded part. Plotting the MLP alone would be a weaker figure: on its own it
    cannot separate "the latent space encodes a lot" from "the probe is flexible".
    """
    s = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL * 0.92, DOUBLE_COL * 0.36),
                             squeeze=False)
    for j, (feat, label) in enumerate(PROXIES.items()):
        ax = axes[0, j]
        drawn = False
        for sp in SPECIES:
            drawn |= errline(ax, df, f"r2mlp_{feat}_{sp}", x, COLOURS[sp],
                             label=DISPLAY[sp])
            errline(ax, df, f"r2lin_{feat}_{sp}", x, COLOURS[sp], marker="o", ls=":",
                    ms=3.4, mfc="white")
        if not drawn:
            ax.axis("off")
            continue
        ax.set_xlabel(xlabel)
        ax.set_ylabel(rf"{label} $R^2$")
        if logx:
            ax.set_xscale("log")
        if j == 0:
            ax.plot([], [], "o-", color="0.35", ms=4.2, label="MLP")
            ax.plot([], [], "o:", color="0.35", ms=3.4, mfc="white", label="ridge")
        ax.legend(fontsize=6.4 * s, frameon=True, framealpha=0.85, edgecolor="0.75",
                  ncol=2, columnspacing=0.8, handletextpad=0.4, loc="best")
        ax.set_title(f"({'ab'[j]}) {label.lower()}, val only",
                     loc="left", fontsize=9 * s, pad=3)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_dir, stem)


def errline(ax, df, col, x, colour, label=None, marker="o", ls="-", ms=4.2, mfc=None):
    """One mean-with-seed-sd series. Returns False if the column has no data."""
    a = agg(df, col, x)
    if a.empty:
        return False
    ax.errorbar(a[x], a["mean"], yerr=np.nan_to_num(a["sd"]),
                fmt=marker, ls=ls, color=colour, ms=ms, lw=1.1, capsize=2.5,
                elinewidth=0.9, label=label, markerfacecolor=mfc)
    return True
