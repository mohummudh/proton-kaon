#!/usr/bin/env python3
"""Audit the MIP reference selection and its sensitivity to light contamination.

This is a diagnostic study, not a calibrated species-composition estimator.  It
traces the dedicated MIP beamline selection through matched clusters and the
final image cuts, then asks whether low-mass candidates in the kaon window look
like that retained MIP reference in the paper-model latent space.

The cluster pickles contain large image arrays.  They are loaded one at a time
and immediately reduced to scalar metadata to keep peak memory manageable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml
from scipy.special import expit
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PROJECT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(PROJECT), str(PROJECT / "scripts" / "extra")]

from _beam_data import SPECIES, load_beam_data  # noqa: E402
from src.train.naming import model_name  # noqa: E402

KEYS = ["run", "subrun", "event"]
MASS_BINS = [350.0, 400.0, 450.0, 525.0, 575.0, 650.001]
MASS_LABELS = ["350–400", "400–450", "450–525", "525–575", "575–650"]
FEATURES = ["mean_adc", "median_adc", "solidity", "fill_fraction", "n_pixels", "height"]
GROUPS = {
    "MIP reference": ("muon", None),
    "kaon-window low side": ("kaon", (350.0, 425.0)),
    "kaon-window core": ("kaon", (475.0, 525.0)),
    "kaon-window high side": ("kaon", (575.0, 650.001)),
}


def metadata(path: str, plane: str) -> pd.DataFrame:
    """Load a large cluster pickle and retain only scalar selection metadata."""
    raw = pd.read_pickle(path)
    out = raw[KEYS + ["height", "width"]].copy()
    out["pair_index"] = raw.index.to_numpy()
    return out.rename(columns={"height": f"height_{plane}", "width": f"width_{plane}"})


def interval(values: pd.Series) -> dict[str, float | int]:
    q = values.dropna().quantile([0.1, 0.5, 0.9])
    return {"n": int(values.notna().sum()), "q10": float(q.loc[0.1]),
            "median": float(q.loc[0.5]), "q90": float(q.loc[0.9])}


def stage_summary(source: pd.DataFrame, pairs: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    matched_keys = pairs[KEYS].drop_duplicates().assign(matched=True)
    final_keys = pairs.loc[pairs["pass_final"], KEYS].drop_duplicates().assign(final=True)
    events = source.merge(matched_keys, on=KEYS, how="left").merge(final_keys, on=KEYS, how="left")
    events[["matched", "final"]] = events[["matched", "final"]].fillna(False).astype(bool)

    rows = []
    for stage, mask in [("beamline source", np.ones(len(events), dtype=bool)),
                        ("matched ≥176-wire pair", events["matched"]),
                        ("final width-selected image", events["final"])]:
        d = events.loc[mask]
        rows.append({"stage": stage, "unique_events": len(d),
                     "fraction_of_source": len(d) / len(events),
                     **{f"mass_{k}": v for k, v in interval(d["beamline_mass"]).items()},
                     **{f"momentum_{k}": v for k, v in interval(d["momentum"]).items()}})
    return events, pd.DataFrame(rows)


def binned_efficiency(events: pd.DataFrame, variable: str) -> pd.DataFrame:
    work = events.dropna(subset=[variable]).copy()
    work["bin"] = pd.qcut(work[variable], 10, duplicates="drop")
    rows = []
    for band, d in work.groupby("bin", observed=True):
        rows.append({"variable": variable, "lo": float(band.left), "hi": float(band.right),
                     "centre": float(d[variable].median()), "n_source": len(d),
                     "matched_efficiency": float(d["matched"].mean()),
                     "final_efficiency": float(d["final"].mean())})
    return pd.DataFrame(rows)


def add_paper_assignments(cfg: dict, Z: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    inf = Path(cfg["output"]["inference_dir"]) / model_name(cfg)
    labels = np.load(inf / "gmm_labels_k37_seed0.npy")
    if len(labels) != len(df):
        raise ValueError("k=37 label cache is not aligned to the loaded paper sample")
    majority = {int(c): df.loc[labels == c, "species"].value_counts().idxmax()
                for c in np.unique(labels)}
    out = df.copy()
    out["gmm_assignment"] = [majority[int(c)] for c in labels]
    return out


def group_mask(df: pd.DataFrame, name: str) -> np.ndarray:
    species, band = GROUPS[name]
    mask = (df["species"] == species).to_numpy(copy=True)
    if band is not None:
        mass = df["beamline_mass"].to_numpy()
        mask &= (mass >= band[0]) & (mass < band[1])
    return mask


def grouped_auc(X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> float:
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000,
                                                               class_weight="balanced"))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    pred = cross_val_predict(pipe, X, y, groups=groups, cv=cv,
                             method="decision_function")
    return float(roc_auc_score(y, pred))


def reference_separability(Z: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """How far each kaon-window mass region lies from the retained MIP reference."""
    rows = []
    mip = group_mask(df, "MIP reference")
    for candidate in ["kaon-window low side", "kaon-window core", "kaon-window high side"]:
        side = group_mask(df, candidate)
        keep = mip | side
        y = side[keep].astype(int)
        groups = df.loc[keep, KEYS].astype(str).agg("_".join, axis=1).to_numpy()
        rows.append({"comparison": f"MIP reference vs {candidate}",
                     "n_mip": int(mip.sum()), "n_candidate": int(side.sum()),
                     "latent_oof_auc": grouped_auc(Z[keep], y, groups)})
    return pd.DataFrame(rows)


def mass_side_probes(Z: np.ndarray, df: pd.DataFrame) -> pd.DataFrame:
    """Measure low- and high-side structure relative to the kaon peak core."""
    rows = []
    core = group_mask(df, "kaon-window core")
    feature_sets = {
        "latents": Z,
        "two paper proxies": df[["mean_adc", "solidity"]].to_numpy(float),
        "six reconstructed features": df[FEATURES].to_numpy(float),
    }
    for quality in ["all", "picky"]:
        qmask = np.ones(len(df), dtype=bool) if quality == "all" else (df["picky"] == 1).to_numpy()
        for candidate in ["kaon-window low side", "kaon-window high side"]:
            side = group_mask(df, candidate)
            keep = qmask & (core | side)
            y = side[keep].astype(int)
            groups = df.loc[keep, KEYS].astype(str).agg("_".join, axis=1).to_numpy()
            for feature_set, values in feature_sets.items():
                X = values[keep]
                finite = np.isfinite(X).all(axis=1)
                auc = grouped_auc(X[finite], y[finite], groups[finite])
                rows.append({"quality": quality, "candidate": candidate,
                             "feature_set": feature_set, "n": int(finite.sum()),
                             "oof_auc": auc, "orientation_free_auc": max(auc, 1 - auc)})
    return pd.DataFrame(rows)


def gmm_mass_scan(cfg: dict, df: pd.DataFrame) -> pd.DataFrame:
    """Test whether MIP-majority components carry the expected low-mass signature."""
    inf = Path(cfg["output"]["inference_dir"]) / model_name(cfg)
    kaon = (df["species"] == "kaon").to_numpy()
    rows = []
    for path in sorted(inf.glob("gmm_labels_k*_seed0.npy")):
        try:
            k = int(path.stem.split("_k", 1)[1].split("_", 1)[0])
        except (IndexError, ValueError):
            continue
        labels = np.load(path)
        if len(labels) != len(df):
            continue
        majority = {int(c): df.loc[labels == c, "species"].value_counts().idxmax()
                    for c in np.unique(labels)}
        assignment = np.array([majority[int(c)] for c in labels])
        for quality in ["all", "picky"]:
            base = kaon if quality == "all" else kaon & (df["picky"].to_numpy() == 1)
            for assigned in SPECIES:
                values = df.loc[base & (assignment == assigned), "beamline_mass"].dropna()
                rows.append({"k": k, "quality": quality, "assignment": assigned,
                             "n": len(values), "median_mass": float(values.median())})
    result = pd.DataFrame(rows)
    pivot = result[result["quality"] == "all"].pivot(index="k", columns="assignment",
                                                       values="median_mass")
    delta = (pivot.get("muon") - pivot.get("kaon")).rename("mip_minus_kaon_median_mass")
    return result.merge(delta, left_on="k", right_index=True, how="left")


def endpoint_image_support(df: pd.DataFrame, image_path: str) -> pd.DataFrame:
    """Quantify support differences that remain visible in the 48x48 model input."""
    tensors = torch.load(image_path, map_location="cpu", mmap=True, weights_only=True)
    by_species = {s: df[df["species"] == s].reset_index(drop=True) for s in SPECIES}
    tensor_key = {"kaon": "k", "muon": "m"}
    rows = []
    for group in GROUPS:
        species, band = GROUPS[group]
        if species not in tensor_key:
            continue
        meta = by_species[species]
        select = np.ones(len(meta), dtype=bool)
        if band is not None:
            mass = meta["beamline_mass"].to_numpy()
            select = (mass >= band[0]) & (mass < band[1])
        images = tensors[tensor_key[species]][torch.from_numpy(select)].float().clamp_min_(0).log1p_()
        active = images > 0
        flat = images.flatten(1)
        count = active.flatten(1).sum(1)
        active_sum = flat.sum(1)
        row_extent = active.any(dim=1).any(dim=2).sum(1) / images.shape[-2]
        col_extent = active.any(dim=1).any(dim=1).sum(1) / images.shape[-1]
        plane_count = active.flatten(2).sum(2)
        plane_imbalance = ((plane_count[:, 0] - plane_count[:, 1]).abs() /
                           plane_count.sum(1).clamp_min(1))
        values = {
            "image_mean_log1p": flat.mean(1),
            "active_mean_log1p": active_sum / count.clamp_min(1),
            "active_fraction": count / flat.shape[1],
            "row_extent_fraction": row_extent,
            "column_extent_fraction": col_extent,
            "plane_active_pixel_imbalance": plane_imbalance,
        }
        for metric, vector in values.items():
            a = vector.numpy()
            rows.append({"group": group, "metric": metric, "n": len(a),
                         "q10": float(np.quantile(a, .1)),
                         "median": float(np.median(a)),
                         "q90": float(np.quantile(a, .9))})
    return pd.DataFrame(rows)


def latent_reference_score(Z: np.ndarray, df: pd.DataFrame) -> tuple[np.ndarray, dict]:
    """Fit a proton-versus-MIP diagnostic score and apply it to all rows.

    This supervised score is used only to test reference support.  It is not the
    unsupervised paper result and is not interpreted as a species probability.
    """
    ref = df["species"].isin(["proton", "muon"]).to_numpy()
    y = (df.loc[ref, "species"] == "muon").astype(int).to_numpy()
    groups = df.loc[ref, KEYS].astype(str).agg("_".join, axis=1).to_numpy()
    pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    cv = StratifiedGroupKFold(5, shuffle=True, random_state=42)
    oof = cross_val_predict(pipe, Z[ref], y, groups=groups, cv=cv,
                            method="decision_function")
    auc = roc_auc_score(y, oof)
    pipe.fit(Z[ref], y)
    score = pipe.decision_function(Z)
    return score, {"reference_oof_auc": float(auc),
                   "reference_n_proton": int((y == 0).sum()),
                   "reference_n_mip": int((y == 1).sum())}


def kaon_mass_table(df: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    d = df.loc[df["species"] == "kaon"].copy()
    d["mip_reference_score"] = score[df["species"].to_numpy() == "kaon"]
    d["mip_reference_fraction"] = expit(d["mip_reference_score"])
    d["mass_band"] = pd.cut(d["beamline_mass"], MASS_BINS, labels=MASS_LABELS,
                            right=False, include_lowest=True)
    rows = []
    for quality in ["all", "picky"]:
        q = d if quality == "all" else d[d["picky"] == 1]
        for band, x in q.groupby("mass_band", observed=True):
            counts = x["gmm_assignment"].value_counts()
            rows.append({"quality": quality, "mass_band": str(band), "n": len(x),
                         "median_mass": float(x["beamline_mass"].median()),
                         "median_momentum": float(x["momentum"].median()),
                         "gmm_mip_fraction": float(counts.get("muon", 0) / len(x)),
                         "gmm_proton_fraction": float(counts.get("proton", 0) / len(x)),
                         "gmm_kaon_fraction": float(counts.get("kaon", 0) / len(x)),
                         "median_mip_reference_fraction": float(x["mip_reference_fraction"].median()),
                         "mip_reference_fraction_gt_half": float((x["mip_reference_score"] > 0).mean())})
    return pd.DataFrame(rows)


def feature_table(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["comparison_group"] = np.select(
        [d["species"] == "muon",
         (d["species"] == "kaon") & d["beamline_mass"].between(350, 425, inclusive="left"),
         (d["species"] == "kaon") & d["beamline_mass"].between(475, 525, inclusive="left"),
         (d["species"] == "kaon") & d["beamline_mass"].between(575, 650, inclusive="both")],
        ["MIP reference", "kaon-window low side", "kaon-window core", "kaon-window high side"],
        default="other")
    d = d[d["comparison_group"] != "other"]
    rows = []
    for group, x in d.groupby("comparison_group"):
        for feature in FEATURES:
            v = x[feature].replace([np.inf, -np.inf], np.nan).dropna()
            rows.append({"group": group, "feature": feature, "n": len(v),
                         "q10": float(v.quantile(.1)), "median": float(v.median()),
                         "q90": float(v.quantile(.9))})
    return pd.DataFrame(rows)


def make_plots(stage: pd.DataFrame, efficiency: pd.DataFrame, mass: pd.DataFrame,
               features: pd.DataFrame, out: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    axes[0].bar(stage["stage"], stage["fraction_of_source"], color=["0.4", "#4477AA", "#228833"])
    axes[0].set_ylabel("Fraction of beamline-selected events")
    axes[0].tick_params(axis="x", rotation=25)
    for i, r in stage.iterrows():
        axes[0].text(i, r["fraction_of_source"] + .025, f"{r['unique_events']:,}", ha="center")
    for variable, colour in [("beamline_mass", "#AA3377"), ("momentum", "#EE7733")]:
        x = efficiency[efficiency["variable"] == variable]
        ax = axes[1] if variable == "beamline_mass" else axes[2]
        ax.plot(x["centre"], x["matched_efficiency"], "o--", color="0.5", label="matched")
        ax.plot(x["centre"], x["final_efficiency"], "o-", color=colour, label="final")
        ax.set_xlabel("Beamline mass [MeV/$c^2$]" if variable == "beamline_mass" else "Momentum [MeV/$c$]")
        ax.set_ylabel("Event selection efficiency")
        ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "mip_cutflow.png", dpi=220)
    fig.savefig(out / "mip_cutflow.pdf")
    plt.close(fig)

    all_mass = mass[mass["quality"] == "all"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.2))
    x = np.arange(len(all_mass))
    bottom = np.zeros(len(all_mass))
    for col, label, colour in [("gmm_mip_fraction", "MIP-majority", "#AA3377"),
                               ("gmm_kaon_fraction", "kaon-majority", "#EE7733"),
                               ("gmm_proton_fraction", "proton-majority", "#0077BB")]:
        axes[0].bar(x, all_mass[col], bottom=bottom, label=label, color=colour)
        bottom += all_mass[col].to_numpy()
    axes[0].set_xticks(x, all_mass["mass_band"], rotation=30)
    axes[0].set_ylabel("Fraction of kaon-window candidates")
    axes[0].set_xlabel("Measured mass band [MeV/$c^2$]")
    axes[0].legend(frameon=False, fontsize=8)
    axes[1].plot(x, all_mass["median_mip_reference_fraction"], "o-", color="#AA3377",
                 label="median diagnostic score")
    axes[1].plot(x, all_mass["mip_reference_fraction_gt_half"], "s--", color="0.35",
                 label="fraction closer to MIP side")
    axes[1].set_xticks(x, all_mass["mass_band"], rotation=30)
    axes[1].set_ylim(-.02, 1.02)
    axes[1].set_ylabel("MIP-reference diagnostic")
    axes[1].set_xlabel("Measured mass band [MeV/$c^2$]")
    axes[1].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "kaon_window_mip_response.png", dpi=220)
    fig.savefig(out / "kaon_window_mip_response.pdf")
    plt.close(fig)

    med = features.pivot(index="group", columns="feature", values="median")
    # Normalise each feature by the MIP-reference median for a compact support comparison.
    rel = med.divide(med.loc["MIP reference"].replace(0, np.nan), axis=1)
    fig, ax = plt.subplots(figsize=(8.0, 3.5))
    rel.T.plot(kind="bar", ax=ax, width=.78)
    ax.axhline(1, color="0.2", lw=.8)
    ax.set_ylabel("Median / MIP-reference median")
    ax.set_xlabel("")
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(out / "mip_feature_support.png", dpi=220)
    fig.savefig(out / "mip_feature_support.pdf")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--out-dir", default=str(PROJECT / "output" / "mip_selection_study"))
    args = ap.parse_args()
    cfg_path = Path(args.config) if args.config else next((PROJECT / "configs").glob("run_0093*.yaml"))
    cfg = yaml.safe_load(cfg_path.read_text())
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    source_keys = pd.read_csv("/Volumes/easystore/proton-kaon/docs/picky_muons_50_300.csv").drop_duplicates(KEYS)
    beam = pd.read_csv("/Volumes/easystore/proton-kaon/docs/picky+match.csv").drop_duplicates(KEYS)
    momentum = pd.read_csv("/Volumes/easystore/proton-deuteron/momentum_tof.csv").drop_duplicates(KEYS)
    source = source_keys.merge(beam, on=KEYS, how="left", validate="one_to_one")
    source = source.merge(momentum, on=KEYS, how="left", validate="one_to_one")

    col = metadata("/Volumes/easystore/proton-kaon/clusters/muon_col.pkl", "col")
    ind = metadata("/Volumes/easystore/proton-kaon/clusters/muon_ind.pkl", "ind")
    pairs = col.merge(ind, on=KEYS + ["pair_index"], how="inner", validate="one_to_one")
    pairs["pass_col_width"] = pairs["width_col"] < 473
    pairs["pass_ind_width"] = pairs["width_ind"] < 473
    pairs["pass_final"] = pairs["pass_col_width"] & pairs["pass_ind_width"]

    events, stage = stage_summary(source, pairs)
    efficiency = pd.concat([binned_efficiency(events, "beamline_mass"),
                            binned_efficiency(events, "momentum")], ignore_index=True)

    Z, df = load_beam_data(cfg)
    extra_momentum = momentum[KEYS + ["momentum"]]
    df = df.merge(extra_momentum, on=KEYS, how="left", validate="many_to_one")
    df = add_paper_assignments(cfg, Z, df)
    score, diagnostic = latent_reference_score(Z, df)
    mass = kaon_mass_table(df, score)
    features = feature_table(df)
    separability = reference_separability(Z, df)
    probes = mass_side_probes(Z, df)
    gmm_scan = gmm_mass_scan(cfg, df)
    endpoint = endpoint_image_support(df, cfg["data"]["path"])

    kaon = df["species"] == "kaon"
    valid = kaon & np.isfinite(df["beamline_mass"].to_numpy())
    rho, p_value = spearmanr(df.loc[valid, "beamline_mass"], score[valid])
    diagnostic.update({"kaon_mass_vs_mip_score_spearman": float(rho),
                       "kaon_mass_vs_mip_score_p": float(p_value),
                       "pair_rows_matched": len(pairs),
                       "pair_rows_final": int(pairs["pass_final"].sum()),
                       "pair_fail_col_width": int((~pairs["pass_col_width"]).sum()),
                       "pair_fail_ind_width": int((~pairs["pass_ind_width"]).sum()),
                       "pair_fail_both_width": int((~pairs["pass_col_width"] & ~pairs["pass_ind_width"]).sum()),
                       "height_overlap_kaon_mip_rows": int(((df["species"] == "kaon") & df["height"].between(176, 178)).sum()),
                       "height_overlap_mip_rows": int(((df["species"] == "muon") & df["height"].between(176, 178)).sum())})

    stage.to_csv(out / "mip_cutflow.csv", index=False)
    efficiency.to_csv(out / "mip_selection_efficiency.csv", index=False)
    mass.to_csv(out / "kaon_mass_band_response.csv", index=False)
    features.to_csv(out / "feature_support.csv", index=False)
    separability.to_csv(out / "mip_reference_separability.csv", index=False)
    probes.to_csv(out / "mass_side_probe_auc.csv", index=False)
    gmm_scan.to_csv(out / "gmm_mip_mass_scan.csv", index=False)
    endpoint.to_csv(out / "endpoint_image_support.csv", index=False)
    with (out / "metrics.json").open("w") as fh:
        json.dump(diagnostic, fh, indent=2)
    make_plots(stage, efficiency, mass, features, out)

    print("\nMIP cutflow")
    print(stage.to_string(index=False))
    print("\nKaon-window response")
    print(mass.to_string(index=False))
    print("\nDiagnostics")
    print(json.dumps(diagnostic, indent=2))
    print(f"\nWrote study to {out}")


if __name__ == "__main__":
    main()
