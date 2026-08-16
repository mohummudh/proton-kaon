#!/usr/bin/env python3
"""
scripts/extra/fit_kaon_peak.py

Three-component fit to the beamline mass spectrum around the kaon window, to
measure how much of that window is genuine K+ and how much is proton and
light-particle leakage. Uses only docs/picky+match.csv -- no TPC data, no
latents, no VAE -- so it is an independent cross-check on any contamination
estimate made from the latent space.

MODEL, over a window bracketing the peak (default 330-690 MeV):
    N(m) = A_light * exp(-(m - lo) / tau_light)     falling pi/mu tail
         + A_proton * exp(+(m - hi) / tau_proton)   rising proton tail
         + A_kaon * Gauss(m; mu, sigma)             the K+ peak
Fitted by binned Poisson maximum likelihood. Over this narrow range the two
tails are locally simple, so no global lineshape model is needed.

WHY --picky 1 IS THE DEFAULT AND MATTERS
    `p` in picky+match.csv is the beamline reconstruction quality flag. Kaon
    momenta run higher than proton/MIP momenta, and TOF-derived mass loses
    precision as beta -> 1, so non-picky events are heavily smeared: they fill
    the valley on both sides of the peak and the three components become
    degenerate (an all-events fit gives chi2/dof ~ 9 and a 74 MeV-wide "kaon"
    acting as filler). Restricting to p=1 leaves a clean isolated peak and the
    fit becomes well posed (chi2/dof ~ 0.9).

VALIDATION TO REPORT ALONGSIDE THE YIELDS
    The peak mean is free within [400, 600]. If it lands near PDG K+ = 493.7
    with a plausible width, the kaon component is fitting a real peak rather
    than absorbing leftover background. Quote it whenever you quote the
    composition -- it is the main evidence the decomposition is meaningful.

CAVEATS
    - This is the composition of the BEAMLINE window. A TPC-selected analysis
      sample (stopping tracks, 2-179 wires) will differ, since through-going
      light particles are preferentially removed.
    - Fitted on picky events; extending the result to a mostly non-picky
      analysis sample is an assumption that needs its own argument.

OUTPUTS (under figs/beamline_mass_fit/ or --out-dir)
    kaon_peak_fit.{png,pdf}
    metrics.json    yields, fractions with bootstrap intervals, peak mu/sigma,
                    chi2/dof

Usage:
    python scripts/extra/fit_kaon_peak.py
    python scripts/extra/fit_kaon_peak.py --picky 0 --range 300 720
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize

from _beam_data import (COLOURS, DEFAULT_PICKY, DOUBLE_COL, PDG_MASS,
                        PROJECT_ROOT, SINGLE_COL, apply_style, savefig)

# Analysis window the composition is quoted over (the kaon beam selection).
WINDOW = (350.0, 650.0)
MU_BOUNDS = (400.0, 600.0)
SIGMA_BOUNDS = (15.0, 120.0)


def components(theta, centres, lo, hi):
    """Unpack parameters into the three component histograms."""
    log_a_light, tau_light, log_a_proton, tau_proton, log_a_kaon, mu, sigma = theta
    light = np.exp(log_a_light) * np.exp(-(centres - lo) / abs(tau_light))
    proton = np.exp(log_a_proton) * np.exp((centres - hi) / abs(tau_proton))
    kaon = np.exp(log_a_kaon) * np.exp(-0.5 * ((centres - mu) / abs(sigma)) ** 2)
    return light, proton, kaon


def neg_log_likelihood(theta, counts, centres, lo, hi):
    """Binned Poisson NLL, with the peak position and width kept physical."""
    if not (MU_BOUNDS[0] < theta[5] < MU_BOUNDS[1]):
        return 1e12
    if not (SIGMA_BOUNDS[0] < abs(theta[6]) < SIGMA_BOUNDS[1]):
        return 1e12
    model = np.clip(sum(components(theta, centres, lo, hi)), 1e-9, None)
    return float(np.sum(model - counts * np.log(model)))


def fit_spectrum(counts, centres, lo, hi, n_starts=30, seed=0):
    """Multi-start Nelder-Mead; returns the best parameter vector."""
    rng = np.random.default_rng(seed)
    best = None
    for _ in range(n_starts):
        start = np.array([
            np.log(max(counts[0], 1.0)), 40 + 30 * rng.random(),
            np.log(max(counts[-1], 1.0)), 40 + 30 * rng.random(),
            np.log(max(counts.max(), 1.0)),
            PDG_MASS["kaon"] + 30 * rng.standard_normal(),
            45 + 15 * rng.random(),
        ])
        res = minimize(neg_log_likelihood, start, args=(counts, centres, lo, hi),
                       method="Nelder-Mead",
                       options=dict(maxiter=200000, maxfev=200000, xatol=1e-6, fatol=1e-6))
        if best is None or res.fun < best.fun:
            best = res
    return best.x


def window_fractions(theta, centres, lo, hi, window):
    """Integrate each component over the analysis window; return counts and fractions."""
    light, proton, kaon = components(theta, centres, lo, hi)
    inside = (centres > window[0]) & (centres < window[1])
    yields = {"kaon": float(kaon[inside].sum()),
              "light": float(light[inside].sum()),
              "proton": float(proton[inside].sum())}
    total = sum(yields.values())
    return yields, {k: v / total for k, v in yields.items()}, total


def bootstrap(theta, counts, centres, lo, hi, window, n_boot, seed=1):
    """Poisson-resample the fitted model and refit, for interval estimates."""
    rng = np.random.default_rng(seed)
    model = sum(components(theta, centres, lo, hi))
    draws = []
    for _ in range(n_boot):
        resampled = rng.poisson(model)
        res = minimize(neg_log_likelihood, theta, args=(resampled, centres, lo, hi),
                       method="Nelder-Mead", options=dict(maxiter=60000, maxfev=60000))
        _, frac, _ = window_fractions(res.x, centres, lo, hi, window)
        draws.append([frac["kaon"], frac["proton"], frac["light"], res.x[5], abs(res.x[6])])
    draws = np.array(draws)
    keys = ["kaon", "proton", "light", "mu", "sigma"]
    return {k: [float(np.percentile(draws[:, i], 16)), float(np.percentile(draws[:, i], 84))]
            for i, k in enumerate(keys)}


def plot_fit(counts, centres, theta, lo, hi, chi2_dof, bin_width, fractions,
             intervals, out_dir):
    light, proton, kaon = components(theta, centres, lo, hi)
    s = apply_style(SINGLE_COL)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL * 0.62, DOUBLE_COL * 0.62 / 1.45))

    # Shade the analysis window: the composition quoted below is the integral
    # of each component over exactly this range, i.e. the kaon beam selection.
    ax.axvspan(*WINDOW, color=COLOURS["kaon"], alpha=0.10, lw=0, zorder=0)
    ax.text(WINDOW[0] + 4, counts.max() * 0.46,
            f"kaon selection\n{WINDOW[0]:.0f}-{WINDOW[1]:.0f} MeV",
            ha="left", va="center", fontsize=6.3 * s, color="#B35A20")

    ax.errorbar(centres, counts, yerr=np.sqrt(counts), fmt="o", ms=2.2 * s,
                color="0.3", lw=0.7 * s, label="data", zorder=4)
    ax.plot(centres, light + proton + kaon, "-", color="k", lw=1.4 * s,
            label="total fit", zorder=3)
    ax.plot(centres, kaon, "--", color=COLOURS["kaon"], lw=1.1 * s,
            label=f"$K^+$  ($\\mu$={theta[5]:.0f}, $\\sigma$={abs(theta[6]):.0f})")
    ax.plot(centres, light, "--", color=COLOURS["muon"], lw=1.1 * s, label="light tail")
    ax.plot(centres, proton, "--", color=COLOURS["proton"], lw=1.1 * s, label="proton tail")
    ax.axvline(PDG_MASS["kaon"], ls=":", lw=0.8 * s, color="0.4")
    ax.text(PDG_MASS["kaon"] + 4, counts.max() * 0.04, "PDG $K^+$",
            fontsize=7 * s, color="0.4", va="bottom")
    ax.set_xlabel("Beamline mass [MeV/$c^2$]")
    ax.set_ylabel(f"Counts / {bin_width:.0f} MeV")
    ax.set_xlim(lo, hi)
    ax.set_ylim(0, counts.max() * 1.25)
    ax.legend(fontsize=6.8 * s, loc="upper left")

    # The measurement itself, on the figure rather than only in the caption.
    lines = [f"in {WINDOW[0]:.0f}-{WINDOW[1]:.0f} MeV:"]
    for name, display in [("kaon", "$K^+$ purity"), ("proton", "proton contam."),
                          ("light", "light contam.")]:
        band = intervals[name]
        lines.append(f"  {display:15s} {fractions[name]:5.1%}  "
                     f"[{band[0]:.1%}, {band[1]:.1%}]")
    ax.text(0.975, 0.955, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=6.3 * s, family="monospace",
            bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.6 * s))
    ax.set_title(f"$\\chi^2$/dof = {chi2_dof:.2f}   "
                 f"peak $\\mu$ = {theta[5]:.1f} MeV (PDG {PDG_MASS['kaon']:.1f})",
                 fontsize=8 * s, pad=3 * s)
    fig.tight_layout()
    savefig(fig, out_dir, "kaon_peak_fit")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picky-csv", default=DEFAULT_PICKY)
    ap.add_argument("--picky", type=int, choices=[0, 1], default=1,
                    help="1 = picky events only (default, and the only well-posed fit)")
    ap.add_argument("--range", type=float, nargs=2, default=[330.0, 690.0],
                    metavar=("LO", "HI"), help="fit range in MeV")
    ap.add_argument("--bin-width", type=float, default=6.0)
    ap.add_argument("--n-boot", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    lo, hi = args.range
    picky = pd.read_csv(args.picky_csv).drop_duplicates(["run", "subrun", "event"])
    mass = picky.loc[picky["p"] == args.picky, "beamline_mass"].to_numpy()
    mass = mass[np.isfinite(mass)]

    edges = np.arange(lo, hi + args.bin_width, args.bin_width)
    centres = 0.5 * (edges[1:] + edges[:-1])
    counts, _ = np.histogram(mass, edges)
    print(f"fitting {counts.sum()} events (picky={args.picky}) in [{lo:.0f},{hi:.0f}], "
          f"{len(centres)} bins of {args.bin_width:.0f} MeV")

    theta = fit_spectrum(counts, centres, lo, hi, seed=args.seed)
    model = sum(components(theta, centres, lo, hi))
    dof = len(counts) - len(theta)
    chi2_dof = float(np.sum((counts - model) ** 2 / np.clip(model, 1, None)) / dof)

    print(f"\nfit quality: chi2/dof = {chi2_dof:.2f}")
    print(f"  kaon peak   mu = {theta[5]:.1f} MeV  (PDG K+ = {PDG_MASS['kaon']:.1f})"
          f"   sigma = {abs(theta[6]):.1f} MeV")
    print(f"  tails       light {abs(theta[1]):.1f} MeV   proton {abs(theta[3]):.1f} MeV")

    yields, fractions, total = window_fractions(theta, centres, lo, hi, WINDOW)
    observed = counts[(centres > WINDOW[0]) & (centres < WINDOW[1])].sum()
    print(f"\ncomposition of the kaon window [{WINDOW[0]:.0f},{WINDOW[1]:.0f}] "
          f"(fitted total {total:.0f}, observed {observed}):")
    for name in ("kaon", "proton", "light"):
        print(f"   {name:7s} {yields[name]:8.0f}   {fractions[name]:7.1%}")

    intervals = bootstrap(theta, counts, centres, lo, hi, WINDOW, args.n_boot, args.seed + 1)
    print("\nbootstrap 68% intervals:")
    for name in ("kaon", "proton", "light"):
        print(f"   {name + ' fraction':16s} {intervals[name][0]:6.1%} - {intervals[name][1]:6.1%}")
    print(f"   {'peak mu [MeV]':16s} {intervals['mu'][0]:6.1f} - {intervals['mu'][1]:6.1f}")

    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "figs" / "beamline_mass_fit"
    plot_fit(counts, centres, theta, lo, hi, chi2_dof, args.bin_width,
             fractions, intervals, out_dir)

    with open(Path(out_dir) / "metrics.json", "w") as fh:
        json.dump({
            "picky": args.picky, "fit_range": [lo, hi], "bin_width": args.bin_width,
            "n_events_in_range": int(counts.sum()), "chi2_dof": chi2_dof,
            "peak_mu": float(theta[5]), "peak_sigma": float(abs(theta[6])),
            "pdg_kaon": PDG_MASS["kaon"],
            "tau_light": float(abs(theta[1])), "tau_proton": float(abs(theta[3])),
            "window": list(WINDOW), "window_yields": yields, "window_fractions": fractions,
            "bootstrap_68": intervals,
        }, fh, indent=2)
    print(f"  saved {Path(out_dir) / 'metrics.json'}")


if __name__ == "__main__":
    main()
