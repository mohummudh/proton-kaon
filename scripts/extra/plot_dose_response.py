#!/usr/bin/env python3
"""
scripts/extra/plot_dose_response.py

Does the latent space independently identify proton contamination in the
kaon-tagged sample? The VAE only ever sees TPC images; `beamline_mass` comes
from the upstream spectrometer, so it is an external check on any structure
the latent finds.

METHOD
    1. Drape a density over the pure proton latents (q_P) and over the
       kaon-tagged latents (q_K). Score every kaon-tagged event by
           proton-likeness = log q_P(z) - log q_K(z)
       By default q_K is the SHARPENED density from anchored_clustering.py, so
       this figure and the anchored assignment rank events by the same
       quantity and the story stays on one method. --score simple fits q_K
       once instead; it ranks kaon events almost identically (Spearman ~0.91)
       but leaves a larger residual trend in the proton control.
    2. Bin by that score and plot the median spectrometer mass per bin.
       If the score is finding protons, mass should rise with it.
    3. Run the identical procedure on the two CLEAN samples (proton, MIP) as
       negative controls.

    Densities are CROSS-FITTED for the sample they score: each pure sample is
    split in half and every event is scored by a model fit on the other half.
    Without this a sample would be scored by a density fit to itself and the
    control would be meaningless.

READING PANEL (b) -- this is the important part
    The raw correlation is NOT a clean control: the proton sample also shows
    rho ~ +0.09, because better-looking images come with better-measured
    masses (a reconstruction-quality effect). Panel (b) separates the two by
    plotting the shift of each sample's median away from ITS OWN PDG mass:
        clean samples CONVERGE on their own mass (quality effect)
        the kaon sample DIVERGES past its own mass, toward the proton
    Same correlation coefficient, opposite meaning.

PICKY
    Protons and MIPs are 100% p=1 while kaons are only ~14.5%, so the default
    run produces a picky-only figure (the apples-to-apples comparison, and the
    one whose mass axis is trustworthy) and a non-picky one for contrast.

OUTPUTS (under figs/<model_name>/dose_response/)
    dose_response_picky{1,0,all}.{png,pdf}          anchored score (default)
    dose_response_picky{1,0,all}_simple.{png,pdf}   with --score simple
    metrics.json    rho and p per sample, bin medians, PDG offsets

Usage:
    python scripts/extra/plot_dose_response.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_dose_response.py --config ... --picky 1 --bins 5
    python scripts/extra/plot_dose_response.py --config ... --score simple
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.mixture import GaussianMixture

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PDG_MASS, SINGLE_COL,
                        SPECIES, apply_style, figure_dir, load_beam_data,
                        load_config, savefig, select)

from anchored_clustering import sharpen_kaon_density

BIN_NAME = {4: "quartile", 5: "quintile", 10: "decile"}


def fit_density(X, n_components, seed=0):
    return GaussianMixture(n_components, covariance_type="full", n_init=6,
                           random_state=seed).fit(X)


def cross_fitted_scores(X, n_components, seed=0):
    """Log-density of every row under a model fit on the OTHER half of X.

    Returns (full_model, honest_self_scores). The full model is for scoring
    other samples; the self-scores avoid a sample grading its own density.
    """
    order = np.random.default_rng(seed).permutation(len(X))
    a, b = order[: len(X) // 2], order[len(X) // 2:]
    model_a, model_b = fit_density(X[a], n_components, 0), fit_density(X[b], n_components, 1)
    scores = np.empty(len(X))
    scores[b] = model_a.score_samples(X[b])
    scores[a] = model_b.score_samples(X[a])
    return fit_density(X, n_components, 2), scores


def proton_likeness(Z, df, n_proton_comp=6, n_kaon_comp=4, seed=0):
    """log q_proton(z) - log q_kaon(z), with the kaon density fit once."""
    Z_p = Z[(df["species"] == "proton").to_numpy()]
    Z_k = Z[(df["species"] == "kaon").to_numpy()]
    Z_m = Z[(df["species"] == "muon").to_numpy()]

    q_proton, self_proton = cross_fitted_scores(Z_p, n_proton_comp, seed)
    q_kaon, self_kaon = cross_fitted_scores(Z_k, n_kaon_comp, seed + 1)
    return {
        "proton": self_proton - q_kaon.score_samples(Z_p),
        "kaon": q_proton.score_samples(Z_k) - self_kaon,
        "muon": q_proton.score_samples(Z_m) - q_kaon.score_samples(Z_m),
    }


def anchored_proton_likeness(Z, df, n_proton_comp=6, n_kaon_comp=6, seed=0):
    """Same score, but with the sharpened kaon density from anchored_clustering.

    Ranks events the same way as the simple version on the kaon sample
    (Spearman ~0.91), so the signal is essentially unchanged -- but the proton
    control tightens toward null, because a kaon density that has stopped
    absorbing the contamination separates the two samples more cleanly.

    Cross-fitting is applied wherever a sample would otherwise be scored by a
    density fit on itself: the kaon sample is halved and each half scored by a
    kaon density sharpened on the other, and the proton density is halved the
    same way. MIPs need neither, since neither density was fit on them.

    WHY A TWO-WAY RATIO AND NOT THE THREE-WAY RESPONSIBILITY
        anchored_clustering.py assigns labels by argmax over all three
        components; here we need something sortable, so we read a graded score
        off the same densities instead. The obvious alternative -- the
        three-way P(proton) -- was tested and is equivalent in practice: it
        ranks kaon events at Spearman 0.93 against this score, and the
        dose-response moves from +0.186 to +0.198 (kaon) while the proton
        control loosens from +0.031 to +0.059. A wash, so the two-way ratio is
        kept: it holds the tighter control on the one sample we most need to be
        null, and "how much more proton-like than kaon-like" is far easier to
        state than a posterior responsibility.
    """
    is_proton = (df["species"] == "proton").to_numpy()
    is_kaon = (df["species"] == "kaon").to_numpy()
    Z_p, Z_k, Z_m = Z[is_proton], Z[is_kaon], Z[(df["species"] == "muon").to_numpy()]

    q_proton, self_proton = cross_fitted_scores(Z_p, n_proton_comp, seed)
    q_muon = fit_density(Z[(df["species"] == "muon").to_numpy()], n_proton_comp, seed)
    counts = np.array([len(Z_p), len(Z_k), len(Z_m)], dtype=float)
    weights = counts / counts.sum()

    full_kaon = sharpen_kaon_density(Z_k, q_proton, q_muon, weights, n_kaon_comp, seed=seed)

    order = np.random.default_rng(seed).permutation(len(Z_k))
    half_a, half_b = order[: len(Z_k) // 2], order[len(Z_k) // 2:]
    kaon_self = np.empty(len(Z_k))
    for fit_on, score_on in ((half_a, half_b), (half_b, half_a)):
        q = sharpen_kaon_density(Z_k[fit_on], q_proton, q_muon, weights, n_kaon_comp, seed=seed)
        kaon_self[score_on] = q.score_samples(Z_k[score_on])

    return {
        "proton": self_proton - full_kaon.score_samples(Z_p),
        "kaon": q_proton.score_samples(Z_k) - kaon_self,
        "muon": q_proton.score_samples(Z_m) - full_kaon.score_samples(Z_m),
    }


def binned_response(score, mass, species, n_bins, n_boot=600, seed=0):
    """Median mass per score bin, with bootstrap intervals and PDG offsets."""
    rng = np.random.default_rng(seed)
    ok = np.isfinite(mass)
    score, mass = score[ok], mass[ok]
    bin_id = pd.qcut(score, n_bins, labels=False)

    median = np.empty(n_bins)
    lo = np.empty(n_bins)
    hi = np.empty(n_bins)
    offset = np.empty(n_bins)
    off_lo = np.empty(n_bins)
    off_hi = np.empty(n_bins)
    for b in range(n_bins):
        vals = mass[bin_id == b]
        boot = np.median(rng.choice(vals, (n_boot, len(vals))), axis=1)
        median[b] = np.median(vals)
        lo[b], hi[b] = np.percentile(boot, [16, 84])
        offset[b] = abs(median[b] - PDG_MASS[species])
        off_lo[b], off_hi[b] = np.percentile(np.abs(boot - PDG_MASS[species]), [16, 84])

    rho = stats.spearmanr(score, mass)
    return {"median": median, "lo": lo, "hi": hi, "offset": offset,
            "offset_lo": off_lo, "offset_hi": off_hi, "n": int(len(mass)),
            "rho": float(rho.statistic), "pvalue": float(rho.pvalue),
            "sample_median": float(np.median(mass))}


def plot_dose_response(results, n_bins, out_dir, stem, label):
    s = apply_style(SINGLE_COL)  # each PANEL is single-column wide
    fig, (ax_abs, ax_off) = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2 / 1.25))
    x = np.arange(1, n_bins + 1)

    # (a) the kaon-tagged sample in absolute mass, against PDG K+.
    r = results["kaon"]
    ax_abs.fill_between(x, r["lo"], r["hi"], color=COLOURS["kaon"], alpha=0.20, lw=0)
    ax_abs.plot(x, r["median"], "o-", color=COLOURS["kaon"], lw=1.1 * s, ms=(3.6 if n_bins <= 5 else 2.8) * s)
    ax_abs.axhline(PDG_MASS["kaon"], ls="--", lw=0.7 * s, color="0.35")
    ax_abs.set_ylim(r["median"].min() - 10, r["median"].max() + 16)
    ax_abs.text(0.62, PDG_MASS["kaon"] + 1.5, "PDG $K^+$", va="bottom", ha="left",
                fontsize=7 * s, color="0.35")
    top = r["median"].max()
    # Position scales with the bin count so it clears the info text on the left.
    arrow_x = n_bins * 0.6
    ax_abs.annotate("", xy=(arrow_x, top + 12), xytext=(arrow_x, top - 2),
                    arrowprops=dict(arrowstyle="->", lw=0.8 * s, color="0.45"))
    ax_abs.text(arrow_x + 0.25, top + 5, "toward\nproton", fontsize=7 * s,
                color="0.45", va="center")
    ax_abs.text(0.04, 0.95, f"Kaon-tagged, {label}\n$n={r['n']}$,  $\\rho={r['rho']:.3f}$",
                transform=ax_abs.transAxes, va="top", fontsize=7.5 * s)
    ax_abs.set_ylabel("Beamline mass [MeV/$c^2$]")
    ax_abs.set_title("(a)", loc="left", fontsize=9 * s, pad=4 * s)

    # (b) distance from each sample's OWN PDG mass -- the panel that separates
    #     a contamination effect from a reconstruction-quality effect.
    for name in ["kaon", "proton", "muon"]:
        r = results[name]
        suffix = "" if name == "kaon" else " (clean)"
        ax_off.fill_between(x, r["offset_lo"], r["offset_hi"], color=COLOURS[name],
                            alpha=0.16, lw=0)
        ax_off.plot(x, r["offset"], "o-", color=COLOURS[name], lw=1.1 * s, ms=(3.6 if n_bins <= 5 else 2.8) * s,
                    label=f"{DISPLAY[name]}{suffix}")
    ax_off.set_ylabel(r"$|\,$median$\,m_{\rm beam}-m_{\rm PDG}|$  [MeV/$c^2$]")
    ax_off.set_title("(b)", loc="left", fontsize=9 * s, pad=4 * s)
    ax_off.legend(loc="upper left", fontsize=7 * s)

    for ax in (ax_abs, ax_off):
        ax.set_xlabel(f"Latent proton-likeness ({BIN_NAME.get(n_bins, 'bin')})")
        ax.set_xlim(0.5, n_bins + 0.5)
        ax.set_xticks(x if n_bins <= 6 else x[::2])
    fig.tight_layout()
    savefig(fig, out_dir, stem)


SCORERS = {"simple": proton_likeness, "anchored": anchored_proton_likeness}


def run_one(Z, df, picky, n_bins, out_dir, seed, score_kind="anchored"):
    """One picky setting: score, bin, plot, and return the metrics.

    The picky filter applies to the KAON sample only. Protons and MIPs are
    100% p=1 in this dataset, so filtering them on p=0 would empty them; they
    are the only controls available either way. The control curves still shift
    slightly between settings because the kaon density in the score changes.
    """
    label = {1: "picky", 0: "non-picky", None: "all events"}[picky]
    keep = np.ones(len(df), dtype=bool)
    if picky is not None:
        is_kaon = (df["species"] == "kaon").to_numpy()
        keep = ~is_kaon | (df["picky"] == picky).to_numpy()
    Zs, dfs = Z[keep], df.loc[keep].reset_index(drop=True)
    print(f"\n--- {label} (filter applied to the kaon sample only) ---")
    for name in SPECIES:
        print(f"    {DISPLAY[name]:7s} n={int((dfs['species'] == name).sum())}")

    scores = SCORERS[score_kind](Zs, dfs, seed=seed)
    results = {}
    for name in SPECIES:
        sel = (dfs["species"] == name).to_numpy()
        results[name] = binned_response(scores[name], dfs.loc[sel, "beamline_mass"].to_numpy(),
                                        name, n_bins, seed=seed)
        r = results[name]
        print(f"    {DISPLAY[name]:7s} rho={r['rho']:+.3f} (p={r['pvalue']:.1e})   "
              f"bin medians {np.round(r['median'], 1)}")
        print(f"            |median - PDG| {np.round(r['offset'], 1)}")

    stem = f"dose_response_picky{'all' if picky is None else picky}"
    if score_kind != "anchored":
        stem += f"_{score_kind}"
    plot_dose_response(results, n_bins, out_dir, stem, label)
    return {name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in r.items()} for name, r in results.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model YAML (must be an all-species run)")
    ap.add_argument("--features-pkl", default=None)
    ap.add_argument("--picky-csv", default=None)
    ap.add_argument("--picky", nargs="+", default=["1", "0"], choices=["1", "0", "all"],
                    help="which picky settings to produce (default: both 1 and 0)")
    ap.add_argument("--bins", type=int, default=10,
                    help="score quantile bins (default 10; the kaon picky subset is\n                          only ~1200 events, so drop to 5 if bins look noisy)")
    ap.add_argument("--score", choices=["anchored", "simple"], default="anchored",
                    help="proton-likeness score: 'anchored' uses the sharpened kaon "
                         "density shared with anchored_clustering.py (default, and what "
                         "keeps the story on one method); 'simple' fits it once")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kwargs = {k: v for k, v in [("features_path", args.features_pkl),
                                ("picky_path", args.picky_csv)] if v}
    Z, df = load_beam_data(cfg, **kwargs)
    out_dir = args.out_dir or figure_dir(cfg, "dose_response")

    payload = {}
    for setting in args.picky:
        picky = None if setting == "all" else int(setting)
        payload[setting] = run_one(Z, df, picky, args.bins, out_dir, args.seed,
                                   score_kind=args.score)

    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump({"bins": args.bins, "score": args.score, "results": payload},
                  fh, indent=2)
    print(f"\n  saved {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
