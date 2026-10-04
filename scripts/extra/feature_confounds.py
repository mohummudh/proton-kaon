#!/usr/bin/env python3
"""
scripts/extra/feature_confounds.py

Is the species structure the model finds physics, or the beamline selection?

WHY THIS EXISTS
    The samples are defined by cuts that are not symmetric between species:
    protons and kaons were required to span 2-179 wires, MIPs >=176. So the MIP
    sample is disjoint in track length from the other two BY CONSTRUCTION, and a
    single threshold on `height` separates proton from MIP at 0.9993 -- it
    recovers the cut at 176. Any claim that the model "separates MIPs" has to be
    checked against that before it means anything.

    The check matters and the naive reading of it is wrong. `height` is the WIRE
    extent (src/clustering.py: bbox[2]-bbox[0]), and images are cropped to the
    last 50 wires, so the model can never see more than 50 -- a MIP at 193 wires
    and one at 500 are identical to it. The leak, if there is one, has to run
    through IMAGE extent: a proton at 42 wires fills less of the window than a
    MIP, which always fills all 50.

WHAT THIS SCRIPT MEASURES
  pairs     best accuracy from a single threshold on each handcrafted feature,
            for each species pair. Establishes which comparisons are cuts
            (proton/MIP 0.9993, kaon/MIP 0.9952) and which are physics
            (proton/kaon 0.686 on height -- no cut between them, they share one
            selection window).

  matched   the decisive test. Restrict to protons with height >= 50, which fill
            the wire window exactly as MIPs do, removing the geometric leak by
            construction. If the latent separation survives, it is calorimetric.
            It does: 0.915 -> 0.909 while dropping 58.6% of protons.

  image     extent measured directly off the 48x48 images the model receives.
            Nothing derived from image geometry reaches 0.77, against the
            model's 0.91, so the model is not using extent.

  threeway  best three-way accuracy from `height` alone, as context for the
            probe numbers. 0.784 with two thresholds. This is NOT a competing
            explanation for the probe's 0.868 -- the model has no access to
            height -- but a reviewer will compute it, so it belongs in the paper
            with that sentence attached.

OUTPUTS (under figs/<model_name>/latents-features/)
    feature_confounds.csv

Usage:
    python scripts/extra/feature_confounds.py --config configs/run_0093_*.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

from _beam_data import (DISPLAY, SPECIES, figure_dir, load_beam_data,
                        load_config)

FEATURES = ["height", "mean_adc", "solidity", "n_pixels", "total_adc"]


def threshold_accuracy(x, y, n_grid=500):
    """Best accuracy achievable from ONE threshold on x for binary y.

    An oracle threshold chosen against the labels, so this is a SUPERVISED
    baseline -- deliberately generous, because the point is to show what a
    trivial method can do before crediting anything to the representation.
    """
    ok = np.isfinite(x)
    x, y = x[ok], y[ok]
    if len(y) == 0 or y.all() or (~y).all():
        return np.nan
    best = 0.0
    for t in np.percentile(x, np.linspace(0.5, 99.5, n_grid)):
        best = max(best, max(((x < t) == y).sum(), ((x >= t) == y).sum()) / len(y))
    return float(best)


def pairwise(df, truth, rows):
    print("=== best single-threshold accuracy, per species pair ===")
    print(f"{'pair':>18} | " + " | ".join(f"{f:>10}" for f in FEATURES), flush=True)
    for a, b in (("proton", "muon"), ("proton", "kaon"), ("kaon", "muon")):
        m = np.isin(truth, [SPECIES.index(a), SPECIES.index(b)])
        y = truth[m] == SPECIES.index(a)
        cells = []
        for f in FEATURES:
            acc = threshold_accuracy(df.loc[m, f].to_numpy(float), y)
            cells.append(f"{acc:>10.4f}")
            rows.append({"test": "pairwise", "pair": f"{a}_vs_{b}", "feature": f,
                         "accuracy": acc})
        print(f"{DISPLAY[a] + ' vs ' + DISPLAY[b]:>18} | " + " | ".join(cells),
              flush=True)


def matched(Z, df, truth, rows, k=16):
    """Protons that fill the 50-wire window, versus MIPs."""
    h = df["height"].to_numpy(float)
    P, M = SPECIES.index("proton"), SPECIES.index("muon")
    lab = GaussianMixture(k, covariance_type="full", n_init=10,
                          random_state=0).fit_predict(Z)

    def gmm_pair(mask, ya):
        return sum(max(int((sel & ya).sum()), int((sel & ~ya).sum()))
                   for sel in ((lab == c)[mask] for c in np.unique(lab))) / mask.sum()

    for name, mask in (("all protons", (truth != SPECIES.index("kaon"))),
                       ("matched (height>=50)",
                        ((truth == P) & (h >= 50)) | (truth == M))):
        y = truth[mask] == P
        r = {"test": "matched", "subset": name, "n": int(mask.sum()),
             "height": threshold_accuracy(h[mask], y),
             "mean_adc": threshold_accuracy(df.loc[mask, "mean_adc"].to_numpy(float), y),
             "solidity": threshold_accuracy(df.loc[mask, "solidity"].to_numpy(float), y),
             f"gmm_k{k}_latents": gmm_pair(mask, y)}
        rows.append(r)
        print(f"\n=== proton vs MIP, {name} (n={r['n']}) ===")
        for key in ("height", "mean_adc", "solidity", f"gmm_k{k}_latents"):
            print(f"  {key:<22} {r[key]:.4f}", flush=True)


def threeway(df, truth, rows, n_grid=150):
    """Best three-way accuracy from `height` alone, two thresholds."""
    h = df["height"].to_numpy(float)
    qs = np.percentile(h, np.linspace(1, 99, n_grid))
    best, at = 0.0, None
    perms = [(0, 1, 2), (0, 2, 1), (1, 0, 2), (1, 2, 0), (2, 0, 1), (2, 1, 0)]
    for i, t1 in enumerate(qs):
        for t2 in qs[i + 1:]:
            pred = np.digitize(h, [t1, t2])
            for p in perms:
                a = float((np.array(p)[pred] == truth).mean())
                if a > best:
                    best, at = a, (float(t1), float(t2))
    rows.append({"test": "threeway_height", "accuracy": best,
                 "threshold_1": at[0], "threshold_2": at[1]})
    print(f"\n=== three-way species accuracy from `height` alone ===")
    print(f"  {best:.4f} at thresholds {at[0]:.0f}, {at[1]:.0f}")
    print("  The model has no access to height (images are cropped to the last")
    print("  50 wires), so this is context for the probe numbers, not a")
    print("  competing explanation for them.", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=16,
                    help="k for the latent-space comparison in the matched test")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    truth = pd.Categorical(df["species"], categories=SPECIES).codes
    out_dir = (Path(args.out_dir) if args.out_dir
               else figure_dir(cfg, "latents-features"))

    print("=== median values by species ===")
    print(df.groupby("species")[FEATURES].median().to_string(
        float_format=lambda v: f"{v:.1f}"), flush=True)
    print(f"  fraction of each species below 50 wires: " + ", ".join(
        f"{s} {(df.loc[df.species == s, 'height'] < 50).mean():.1%}"
        for s in SPECIES), flush=True)
    print()

    rows = []
    pairwise(df, truth, rows)
    matched(Z, df, truth, rows, args.k)
    threeway(df, truth, rows)
    pd.DataFrame(rows).to_csv(out_dir / "feature_confounds.csv", index=False)
    print(f"\nwrote {out_dir / 'feature_confounds.csv'}")


if __name__ == "__main__":
    main()
