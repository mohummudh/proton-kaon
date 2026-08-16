#!/usr/bin/env python3
"""
scripts/extra/plot_beamline_spectrum.py

The full beamline mass spectrum, and the three windows that define the proton,
kaon and MIP samples. This is the context figure for everything else: it shows
in one panel why the kaon sample is contaminated and the other two are not.

WHAT THE FIGURE SAYS
    The spectrometer assigns a mass to every particle. Plotting all of them
    gives three humps -- pi/mu near 140, K near 494, p near 938 -- and the
    species "tags" used throughout this project are simply windows cut on this
    axis. Each hump has tails, because the measurement is imperfect.

    The proton and MIP windows sit on top of their own huge peaks, so their
    tails contribute little and those samples come out ~pure. The kaon window
    sits in the VALLEY between two populations roughly ten times larger, so the
    pi/mu right-hand tail and the proton left-hand tail both leak into it. That
    geometry is the whole reason the kaon sample needs a contamination
    measurement and the other two do not.

NEGATIVE MASSES
    m = p * sqrt((c t / L)^2 - 1), so a time-of-flight fluctuation below the
    speed-of-light limit makes the bracket negative. Those events are given a
    signed mass and ~17% of the file lands below zero. They are unphysical
    reconstruction failures, not a light species, so the default view starts at
    m = 0; the fraction dropped is printed and recorded.

PICKY
    --split-picky overlays the p=1 subset. The picky peaks are visibly
    narrower, which is the same effect that makes fit_kaon_peak.py well posed
    on picky events and degenerate on the full sample.

OUTPUTS (under figs/beamline_mass_fit/ or --out-dir)
    beamline_spectrum.{png,pdf}
    metrics.json    counts per window, negative-mass fraction

Usage:
    python scripts/extra/plot_beamline_spectrum.py
    python scripts/extra/plot_beamline_spectrum.py --split-picky
    python scripts/extra/plot_beamline_spectrum.py --range 0 1600 --bins 400
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from _beam_data import (COLOURS, DEFAULT_PICKY, DOUBLE_COL, PROJECT_ROOT,
                        SINGLE_COL, apply_style, savefig)

# The selection windows that define the three samples, in MeV/c^2.
WINDOWS = {"muon": (0.0, 250.0), "kaon": (350.0, 650.0), "proton": (750.0, 1150.0)}
WINDOW_LABEL = {"muon": "MIPs\nwindow", "kaon": "kaon\nwindow", "proton": "proton\nwindow"}

# PDG masses to mark. The MIP window holds both muons and pions.
# (mass, label, horizontal nudge) -- mu and pi are only 34 MeV apart, so their
# labels are pushed to opposite sides of their lines to stop them colliding.
PDG_LINES = [(105.658, r"$\mu$", -1), (139.570, r"$\pi$", +1),
             (493.677, "$K$", +1), (938.272, "$p$", +1)]


def load_mass(picky_path):
    """Beamline mass per event, plus the fraction that came out negative."""
    picky = pd.read_csv(picky_path).drop_duplicates(["run", "subrun", "event"])
    mass = picky["beamline_mass"].to_numpy()
    flag = picky["p"].to_numpy()
    finite = np.isfinite(mass)
    return mass[finite], flag[finite]


def plot_spectrum(mass, flag, lo, hi, bins, split_picky, out_dir):
    s = apply_style(SINGLE_COL * 1.55)  # single wide panel
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.72, DOUBLE_COL * 0.72 / 1.55))
    edges = np.linspace(lo, hi, bins + 1)

    for name, (w_lo, w_hi) in WINDOWS.items():
        ax.axvspan(max(w_lo, lo), min(w_hi, hi), color=COLOURS[name], alpha=0.13, lw=0, zorder=0)

    ax.hist(mass, bins=edges, color="0.45", lw=0, label="all events" if split_picky else None)
    if split_picky:
        ax.hist(mass[flag == 1], bins=edges, histtype="step", color="#CC3311",
                lw=0.9 * s, label="picky ($p=1$)")

    top = np.histogram(mass, edges)[0].max()
    for x0, label, side in PDG_LINES:
        if lo <= x0 <= hi:
            ax.axvline(x0, ls="--", lw=0.7 * s, color="0.25", zorder=2)
            ax.text(x0 + side * (hi - lo) * 0.007, top * 0.55, label, fontsize=8 * s,
                    color="0.25", va="top", ha="left" if side > 0 else "right")

    for name, (w_lo, w_hi) in WINDOWS.items():
        centre = (max(w_lo, lo) + min(w_hi, hi)) / 2
        ax.text(centre, top * 1.9, WINDOW_LABEL[name], ha="center", va="top",
                fontsize=6.8 * s, color=COLOURS[name], linespacing=1.1)

    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(None, top * 3.2)
    ax.set_xlabel("Beamline mass [MeV/$c^2$]")
    ax.set_ylabel("Counts (log)")
    ax.set_title(f"Full beamline sample, $m \\geq 0$:  n = {len(mass):,}",
                 fontsize=9 * s, pad=3 * s)
    if split_picky:
        ax.legend(loc="upper right", fontsize=7 * s)
    fig.tight_layout()
    savefig(fig, out_dir, "beamline_spectrum")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picky-csv", default=DEFAULT_PICKY)
    ap.add_argument("--range", type=float, nargs=2, default=[0.0, 1400.0],
                    metavar=("LO", "HI"),
                    help="x-axis range; default starts at 0 to drop unphysical negative masses")
    ap.add_argument("--bins", type=int, default=350)
    ap.add_argument("--split-picky", action="store_true",
                    help="overlay the p=1 subset to show the resolution difference")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    lo, hi = args.range
    mass, flag = load_mass(args.picky_csv)
    n_negative = int((mass < 0).sum())
    print(f"beamline events with a finite mass: {len(mass)}")
    print(f"  negative (unphysical, TOF below the light limit): {n_negative} "
          f"({n_negative / len(mass):.2%}) -- excluded by the default view")

    shown = mass[(mass >= lo) & (mass <= hi)]
    print(f"  shown in [{lo:.0f},{hi:.0f}]: {len(shown)}")
    counts = {}
    for name, (w_lo, w_hi) in WINDOWS.items():
        inside = int(((mass > w_lo) & (mass < w_hi)).sum())
        picky_inside = int(((mass > w_lo) & (mass < w_hi) & (flag == 1)).sum())
        counts[name] = {"window": [w_lo, w_hi], "n": inside, "n_picky": picky_inside}
        print(f"  {name:7s} window [{w_lo:4.0f},{w_hi:4.0f}]: {inside:7d} events "
              f"({inside / len(mass):5.1%} of the beam), picky {picky_inside:6d}")
    print("\n  the kaon window sits in the valley between two populations ~10x larger,")
    print("  which is why it is the only one of the three that needs a contamination fit.")

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "figs" / "beamline_mass_fit"
    plot_spectrum(mass[mass >= lo] if lo >= 0 else mass, flag[mass >= lo] if lo >= 0 else flag,
                  lo, hi, args.bins, args.split_picky, out_dir)

    with open(Path(out_dir) / "spectrum_metrics.json", "w") as fh:
        json.dump({"n_finite": int(len(mass)), "n_negative": n_negative,
                   "negative_fraction": float(n_negative / len(mass)),
                   "range": [lo, hi], "windows": counts}, fh, indent=2)
    print(f"  saved {Path(out_dir) / 'spectrum_metrics.json'}")


if __name__ == "__main__":
    main()
