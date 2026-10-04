#!/usr/bin/env python3
"""Reconstruct the kaon-window population before the 10–179 wire cut.

The regular cluster table contains only objects that survived the height and
beam-entry cuts.  This script returns to the 20,035-event kaon-window ROOT file,
stores only per-event cluster heights, and compares the spectrometer variables
for events removed as through-going with events having an in-window cluster.

The extraction is intentionally cached because reading every waveform takes
roughly ten minutes on the reference machine.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import uproot
from scipy.stats import mannwhitneyu, spearmanr

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))

from src.event import Event  # noqa: E402
from src.open_root import _select_tree, open_root  # noqa: E402

KEYS = ["run", "subrun", "event"]
ROOT = "/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root"
TREE = "ana/raw;352"


def extract(out: Path) -> pd.DataFrame:
    events = open_root(ROOT, tree_name=TREE)
    tree = _select_tree(uproot.open(ROOT), tree_name=TREE)
    rows, failed, start = [], 0, time.time()
    print(f"Extracting pre-cut heights for {len(events):,} kaon-window events", flush=True)
    for n, (_, row) in enumerate(events.iterrows(), 1):
        try:
            event = Event(tree=tree, filepath=row.file_path, index=row.event_index, plot=False)
            rec = {k: int(row[k]) for k in KEYS}
            for plane, image, threshold in (("col", event.collection, 15),
                                            ("ind", event.induction, 7)):
                _, regions = event.connectedregions(image, threshold=threshold)
                heights = ([int(r.bbox[2] - r.bbox[0]) for r in regions]
                           if regions is not None else [])
                rec[f"n_{plane}"] = len(heights)
                rec[f"max_h_{plane}"] = max(heights) if heights else 0
                rec[f"n_short_{plane}"] = sum(h <= 10 for h in heights)
                rec[f"n_window_{plane}"] = sum(10 < h < 179 for h in heights)
                rec[f"n_long_{plane}"] = sum(h >= 179 for h in heights)
            rows.append(rec)
        except Exception:
            failed += 1
        if n % 1000 == 0:
            elapsed = time.time() - start
            eta = (len(events) - n) * elapsed / n / 60
            print(f"  {n:,}/{len(events):,}; {eta:.1f} min remaining; {failed} failures",
                  flush=True)
    result = pd.DataFrame(rows)
    result.to_csv(out, index=False)
    print(f"Saved {len(result):,} rows to {out}; {failed} failures", flush=True)
    return result


def classify(row: pd.Series) -> str:
    if row["n_col"] == 0:
        return "no collection cluster"
    if row["n_window_col"] > 0:
        return "has 11–178 wire cluster"
    if row["n_long_col"] > 0:
        return "all collection clusters ≥179"
    return "all collection clusters ≤10"


def analyse(heights: pd.DataFrame, out: Path) -> None:
    beam = pd.read_csv("/Volumes/easystore/proton-kaon/docs/picky+match.csv").drop_duplicates(KEYS)
    momentum = pd.read_csv("/Volumes/easystore/proton-deuteron/momentum_tof.csv").drop_duplicates(KEYS)
    features = pd.read_pickle("/Volumes/easystore/proton-kaon/features/features.pkl")
    final = (features.loc[features["particle_type"] == "kaon", KEYS]
             .drop_duplicates().assign(in_final_dataset=True))
    d = heights.merge(beam, on=KEYS, how="left", validate="one_to_one")
    d = d.merge(momentum[KEYS + ["momentum"]], on=KEYS, how="left", validate="one_to_one")
    d = d.merge(final, on=KEYS, how="left", validate="one_to_one")
    d["in_final_dataset"] = d["in_final_dataset"].fillna(False).astype(bool)
    d["category"] = d.apply(classify, axis=1)
    d.to_csv(out / "kaon_precut_event_categories.csv", index=False)

    rows = []
    quality_masks = {"all": np.ones(len(d), dtype=bool),
                     "picky": (d["p"] == 1).to_numpy(),
                     "non-picky": (d["p"] == 0).to_numpy()}
    for quality, quality_mask in quality_masks.items():
        for category, x in d.loc[quality_mask].groupby("category"):
            mass = x["beamline_mass"].dropna()
            rows.append({"quality": quality, "category": category, "n": len(x),
                         "mass_median": float(mass.median()),
                         "mass_q25": float(mass.quantile(.25)),
                         "mass_q75": float(mass.quantile(.75)),
                         "below_380": float((mass < 380).mean()),
                         "below_400": float((mass < 400).mean()),
                         "below_425": float((mass < 425).mean()),
                         "momentum_median": float(x["momentum"].median()),
                         "picky_fraction": float((x["p"] == 1).mean()),
                         "final_dataset_fraction": float(x["in_final_dataset"].mean())})
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "kaon_precut_category_summary.csv", index=False)

    long = d.loc[d["category"] == "all collection clusters ≥179", "beamline_mass"].dropna()
    window = d.loc[d["category"] == "has 11–178 wire cluster", "beamline_mass"].dropna()
    has = d[d["n_col"] > 0].dropna(subset=["beamline_mass"])
    rho, rho_p = spearmanr(has["max_h_col"], has["beamline_mass"])
    metrics = {
        "n_root_events": len(heights),
        "n_with_mass": int(d["beamline_mass"].notna().sum()),
        "long_minus_window_median_mass": float(long.median() - window.median()),
        "long_vs_window_mann_whitney_p": float(mannwhitneyu(long, window).pvalue),
        "long_minus_window_below_380_fraction": float((long < 380).mean() - (window < 380).mean()),
        "long_minus_window_below_400_fraction": float((long < 400).mean() - (window < 400).mean()),
        "height_mass_spearman": float(rho),
        "height_mass_spearman_p": float(rho_p),
    }
    for quality, quality_mask in quality_masks.items():
        q = d.loc[quality_mask]
        q_long = q.loc[q["category"] == "all collection clusters ≥179", "beamline_mass"].dropna()
        q_window = q.loc[q["category"] == "has 11–178 wire cluster", "beamline_mass"].dropna()
        metrics[f"{quality}_long_minus_window_median_mass"] = float(q_long.median() - q_window.median())
        for threshold in (380, 400, 425):
            metrics[f"{quality}_long_minus_window_below_{threshold}_fraction"] = float(
                (q_long < threshold).mean() - (q_window < threshold).mean())
    with (out / "kaon_precut_metrics.json").open("w") as fh:
        json.dump(metrics, fh, indent=2)

    bins = [0, 10, 40, 80, 120, 179, 250, 10000]
    has = has.assign(height_band=pd.cut(has["max_h_col"], bins))
    height_summary = (has.groupby("height_band", observed=True)
                      .agg(n=("beamline_mass", "size"),
                           mass_median=("beamline_mass", "median"),
                           mass_q25=("beamline_mass", lambda x: x.quantile(.25)),
                           mass_q75=("beamline_mass", lambda x: x.quantile(.75)),
                           below_400=("beamline_mass", lambda x: (x < 400).mean()))
                      .reset_index())
    height_summary.to_csv(out / "kaon_precut_height_summary.csv", index=False)

    order = ["all collection clusters ≥179", "has 11–178 wire cluster",
             "all collection clusters ≤10", "no collection cluster"]
    plot_data = [d.loc[d["category"] == c, "beamline_mass"].dropna() for c in order]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4))
    axes[0].boxplot(plot_data, tick_labels=["all ≥179", "has 11–178", "all ≤10", "none"],
                    showfliers=False, patch_artist=True,
                    boxprops={"facecolor": "#DDEAF3"}, medianprops={"color": "#AA3377"})
    axes[0].axhline(493.677, color="0.35", ls=":", label="PDG $K^+$")
    axes[0].set_ylabel("Beamline mass [MeV/$c^2$]")
    axes[0].set_xlabel("Pre-cut collection-plane category")
    axes[0].legend(frameon=False)
    x = np.arange(len(height_summary))
    axes[1].errorbar(x, height_summary["mass_median"],
                     yerr=[height_summary["mass_median"] - height_summary["mass_q25"],
                           height_summary["mass_q75"] - height_summary["mass_median"]],
                     fmt="o-", color="#0077BB", capsize=3)
    axes[1].axvline(4.5, color="#AA3377", ls="--", label="179-wire boundary")
    axes[1].set_xticks(x, height_summary["height_band"].astype(str), rotation=30)
    axes[1].set_ylabel("Median beamline mass [MeV/$c^2$]")
    axes[1].set_xlabel("Tallest collection cluster [wires]")
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out / "kaon_precut_length_mass.png", dpi=220)
    fig.savefig(out / "kaon_precut_length_mass.pdf")
    plt.close(fig)

    print("\nPre-cut kaon-window categories")
    print(summary[summary["quality"] == "all"].to_string(index=False))
    print("\nMetrics")
    print(json.dumps(metrics, indent=2))


def main() -> None:
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default=str(PROJECT / "output" / "mip_selection_study"))
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache = out / "kaon_precut_heights.csv"
    heights = extract(cache) if args.refresh or not cache.exists() else pd.read_csv(cache)
    analyse(heights, out)


if __name__ == "__main__":
    main()
