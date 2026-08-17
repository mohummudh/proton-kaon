#!/usr/bin/env python3
"""
scripts/extra/estimate_contamination.py

How much of the kaon-tagged sample is actually proton, measured from the latent
space alone. Uses only the two beam tags we trust (proton and MIP are ~100%
pure in this dataset); the kaon tag is treated as an unknown mixture.

THE FIT
    Drape a density over each trusted sample -- q_P from the pure protons, q_M
    from the pure MIPs -- then model the kaon-tagged sample as
        q_obs(z) = f_p q_P(z) + f_m q_M(z) + f_K q_K(z)
    and solve for the weights by EM, holding the three densities fixed. Each
    iteration asks every event "which source are you from?" and averages those
    probabilities to update the fractions.

WHY THE RAW ANSWER IS BIASED LOW
    q_K, the TRUE kaon density, is unknown -- there is no clean kaon sample.
    Substituting a density fit to the kaon-tagged sample itself means q_K has
    already learned to call the contaminating protons typical, so it explains
    them and starves the q_P component. It is the same circularity as asking
    how many oranges are in a bag while defining "apple" from that same bag.

THE FIX -- INJECTION RECOVERY
    Rather than argue about the size of the bias, measure it. Add a KNOWN
    number of real protons to the real kaon sample, refit, and see how much the
    estimate moves. The response is linear; its slope is the recovery
    efficiency, and dividing the un-injected estimate by that slope removes the
    bias. On run93 the slope is ~0.65, i.e. for every 100 protons genuinely
    present the raw fit finds about 65.

    Run the calibration at the real difficulty. An earlier version injected
    protons into MIPs and got slope ~0.88, but proton/MIP latent overlap is
    ~0.03 against proton/kaon ~0.29, so that badly understates the bias.

WHERE THE CORRECTION ITSELF IS WEAK -- READ BEFORE QUOTING THE NUMBER
    Dividing the intercept by the slope assumes the estimator's only error is
    multiplicative: that it recovers a fixed fraction of the protons present
    and never mistakes a true kaon for one. If it also has a false-positive
    rate, that inflates the intercept and the division over-corrects.

    Injection cannot separate the two. Adding known protons constrains the
    slope but says nothing about the false-positive rate, and measuring that
    would need a clean kaon sample -- the thing that does not exist. The
    sensitivity is real, not academic: run the same calibration on the
    anchored estimator in anchored_clustering.py, whose q_K is sharper, and it
    corrects to ~34% against this script's ~17%.

    So treat the latent estimates as a family spanning roughly 11-39% across
    reasonable density models and corrections, and lead with the beamline mass
    fit (fit_kaon_peak.py: 25.1%, 68% interval 19.8-30.0%), which needs none of
    these assumptions. The latent numbers are consistent with it, not
    independent confirmations of it.

TRANSFER TO NON-PICKY EVENTS
    The beamline mass fit (fit_kaon_peak.py) can only be done on picky events.
    Extending it to the mostly non-picky analysis sample needs an argument, and
    this script supplies it: the same EM fit is run on the picky and non-picky
    kaon subsets separately. If they agree, contamination does not depend on
    beamline reconstruction quality and the picky measurement transfers.

OUTPUTS (under figs/<model_name>/contamination/)
    contamination.{png,pdf}   (a) injection-recovery calibration
                              (b) raw vs corrected by subsample (the transfer
                                  check) (c) beamline mass of the events the
                                  fit flags, an independent validation
    metrics.json

Usage:
    python scripts/extra/estimate_contamination.py --config configs/run_0093_*.yaml
    python scripts/extra/estimate_contamination.py --config ... --n-inject 400 800 1600 2400 3200
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

from _beam_data import (COLOURS, DOUBLE_COL, PDG_MASS, SINGLE_COL, apply_style,
                        figure_dir, load_beam_data, load_config, savefig)


def fit_density(X, n_components, seed=0):
    return GaussianMixture(n_components, covariance_type="full", n_init=6,
                           random_state=seed).fit(X)


def em_weights(log_densities, max_iter=400, tol=1e-11):
    """EM over mixture weights only, with the component densities held fixed.

    log_densities is (n_events, n_components). Returns the fitted weights.
    """
    n_comp = log_densities.shape[1]
    weights = np.full(n_comp, 1.0 / n_comp)
    for _ in range(max_iter):
        log_joint = log_densities + np.log(weights)
        resp = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
        resp /= resp.sum(axis=1, keepdims=True)
        updated = resp.mean(axis=0)
        if np.abs(updated - weights).max() < tol:
            return updated, resp
        weights = updated
    return weights, resp


def fit_contamination(Z_kaon, q_proton, q_muon, n_host_comp=6, seed=0):
    """Raw EM proton fraction in a kaon-tagged bag, plus per-event responsibilities.

    The host density (standing in for the unknown true-kaon density) is fit on
    the bag itself -- the step that biases the answer low.
    """
    q_host = fit_density(Z_kaon, n_host_comp, seed)
    log_densities = np.column_stack([q_proton.score_samples(Z_kaon),
                                     q_muon.score_samples(Z_kaon),
                                     q_host.score_samples(Z_kaon)])
    weights, resp = em_weights(log_densities)
    return {"f_proton": float(weights[0]), "f_muon": float(weights[1]),
            "f_kaon": float(weights[2]), "resp_proton": resp[:, 0]}


def injection_recovery(Z_kaon, Z_proton, q_proton, q_muon, n_inject, seed=0):
    """Spike the kaon bag with known protons and measure the recovery slope.

    Returns (rows, slope, intercept). `intercept` is the fit's reading on the
    un-injected bag; intercept / slope is the bias-corrected contamination.
    """
    rng = np.random.default_rng(seed)
    rows = [{"n_injected": 0, "f_true_injected": 0.0,
             "f_fitted": fit_contamination(Z_kaon, q_proton, q_muon, seed=seed)["f_proton"]}]
    for n in n_inject:
        spike = Z_proton[rng.choice(len(Z_proton), n, replace=False)]
        bag = np.vstack([Z_kaon, spike])
        rows.append({"n_injected": int(n), "f_true_injected": n / len(bag),
                     "f_fitted": fit_contamination(bag, q_proton, q_muon, seed=seed)["f_proton"]})

    table = pd.DataFrame(rows)
    design = np.column_stack([np.ones(len(table)), table["f_true_injected"]])
    intercept, slope = np.linalg.lstsq(design, table["f_fitted"], rcond=None)[0]
    return table, float(slope), float(intercept)


def plot_contamination(table, slope, intercept, subsamples, mass_split, out_dir):
    s = apply_style(SINGLE_COL)  # each PANEL is single-column wide
    fig, (ax_cal, ax_sub, ax_mass) = plt.subplots(1, 3, figsize=(DOUBLE_COL, DOUBLE_COL / 3 / 1.0))

    # (a) the calibration: how much of an injected signal the fit recovers.
    x = table["f_true_injected"].to_numpy()
    ax_cal.plot(x, table["f_fitted"], "o", color=COLOURS["kaon"], ms=4 * s, zorder=3)
    grid = np.linspace(0, x.max() * 1.05, 50)
    ax_cal.plot(grid, intercept + slope * grid, "-", color=COLOURS["kaon"], lw=1.1 * s,
                label=f"slope = {slope:.2f}")
    ax_cal.plot(grid, intercept + grid, "--", color="0.55", lw=0.9 * s, label="perfect recovery")
    ax_cal.set_xlabel("Protons injected (fraction of bag)")
    ax_cal.set_ylabel("Fitted proton fraction")
    ax_cal.set_title("(a) calibration", loc="left", fontsize=8.5 * s, pad=4 * s)
    ax_cal.legend(loc="upper left", fontsize=6.8 * s)

    # (b) raw vs bias-corrected per subsample -- the picky transfer check.
    names = list(subsamples)
    pos = np.arange(len(names))
    raw = [subsamples[n]["raw"] for n in names]
    corrected = [subsamples[n]["corrected"] for n in names]
    ax_sub.bar(pos - 0.2, raw, width=0.38, color="0.72", label="raw EM")
    ax_sub.bar(pos + 0.2, corrected, width=0.38, color=COLOURS["kaon"], label="bias-corrected")
    for i, (r, c) in enumerate(zip(raw, corrected)):
        ax_sub.text(i - 0.2, r + 0.005, f"{r:.1%}", ha="center", fontsize=6 * s)
        ax_sub.text(i + 0.2, c + 0.005, f"{c:.1%}", ha="center", fontsize=6 * s)
    ax_sub.set_xticks(pos)
    ax_sub.set_xticklabels([f"{n}\n(n={subsamples[n]['n']})" for n in names], fontsize=6.8 * s)
    ax_sub.set_ylabel("Proton contamination")
    ax_sub.set_ylim(0, max(corrected) * 1.22)
    ax_sub.set_title("(b) does it transfer?", loc="left", fontsize=8.5 * s, pad=13 * s)
    # Bars fill the axes, so the legend goes above rather than over the labels.
    ax_sub.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2,
                  fontsize=6.5 * s, frameon=False, columnspacing=1.0, handlelength=1.3)

    # (c) independent check: the spectrometer never entered the fit, so the
    #     flagged events should sit higher in mass if they really are protons.
    flagged, rest = mass_split
    edges = np.linspace(350, 650, 26)
    for vals, colour, label in [(rest, "0.55", "low $P$(proton)"),
                                (flagged, COLOURS["proton"], "high $P$(proton)")]:
        vals = vals[np.isfinite(vals)]
        counts, _ = np.histogram(vals, edges)
        ax_mass.stairs(counts / counts.sum(), edges, color=colour, lw=1.1 * s, label=label)
    ax_mass.axvline(PDG_MASS["kaon"], ls="--", lw=0.7 * s, color="0.35")
    ax_mass.set_xlabel("Beamline mass [MeV/$c^2$]")
    ax_mass.set_ylabel("Fraction / bin")
    ax_mass.set_title("(c) external check", loc="left", fontsize=8.5 * s, pad=4 * s)
    ax_mass.set_ylim(0, ax_mass.get_ylim()[1] * 1.22)
    ax_mass.legend(loc="upper right", fontsize=6.5 * s)

    fig.tight_layout()
    savefig(fig, out_dir, "contamination")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--features-pkl", default=None)
    ap.add_argument("--picky-csv", default=None)
    ap.add_argument("--n-inject", type=int, nargs="+", default=[400, 800, 1600, 2400, 3200],
                    help="proton spike sizes for the recovery calibration")
    ap.add_argument("--flag-quantile", type=float, default=0.80,
                    help="responsibility quantile defining 'flagged' in panel (c)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kwargs = {k: v for k, v in [("features_path", args.features_pkl),
                                ("picky_path", args.picky_csv)] if v}
    Z, df = load_beam_data(cfg, **kwargs)
    out_dir = args.out_dir or figure_dir(cfg, "contamination")

    is_species = {s: (df["species"] == s).to_numpy() for s in ("proton", "kaon", "muon")}
    Z_proton, Z_kaon = Z[is_species["proton"]], Z[is_species["kaon"]]
    q_proton = fit_density(Z_proton, 6, args.seed)
    q_muon = fit_density(Z[is_species["muon"]], 6, args.seed)

    print("\ninjection recovery (spiking the real kaon bag with known protons)")
    table, slope, intercept = injection_recovery(Z_kaon, Z_proton, q_proton, q_muon,
                                                 args.n_inject, args.seed)
    print(table.assign(f_true_injected=table["f_true_injected"].round(4),
                       f_fitted=table["f_fitted"].round(4)).to_string(index=False))
    print(f"  fitted = {intercept:.4f} + {slope:.3f} * injected")
    print(f"  recovery slope {slope:.3f}  ->  for every 100 protons present the fit finds "
          f"{slope * 100:.0f}")
    print(f"  BIAS-CORRECTED contamination = {intercept:.4f} / {slope:.3f} = "
          f"{intercept / slope:.1%}")

    print("\ntransfer check -- same fit on each kaon subsample")
    subsamples = {}
    for name, mask in [("all", np.ones(len(df), bool)),
                       ("picky", (df["picky"] == 1).to_numpy()),
                       ("non-picky", (df["picky"] == 0).to_numpy())]:
        sel = is_species["kaon"] & mask
        if sel.sum() < 200:
            continue
        n_comp = 6 if sel.sum() > 3000 else 4
        raw = fit_contamination(Z[sel], q_proton, q_muon, n_host_comp=n_comp,
                                seed=args.seed)["f_proton"]
        subsamples[name] = {"n": int(sel.sum()), "raw": raw, "corrected": raw / slope}
        print(f"  {name:10s} n={sel.sum():5d}   raw {raw:.3f}   corrected {raw / slope:.1%}")
    spread = max(v["raw"] for v in subsamples.values()) - min(v["raw"] for v in subsamples.values())
    print(f"  picky vs non-picky raw spread = {spread:.4f} "
          f"-> contamination is {'independent of' if spread < 0.02 else 'sensitive to'} "
          f"beamline reconstruction quality")

    result = fit_contamination(Z_kaon, q_proton, q_muon, seed=args.seed)
    resp = result["resp_proton"]
    kaon_mass = df.loc[is_species["kaon"], "beamline_mass"].to_numpy()
    threshold = np.quantile(resp, args.flag_quantile)
    flagged, rest = kaon_mass[resp > threshold], kaon_mass[resp <= threshold]
    shift = np.nanmedian(flagged) - np.nanmedian(rest)
    pval = stats.mannwhitneyu(flagged[np.isfinite(flagged)], rest[np.isfinite(rest)]).pvalue
    print(f"\nexternal check: flagged events sit {shift:+.1f} MeV higher in spectrometer mass "
          f"(Mann-Whitney p={pval:.1e})")

    plot_contamination(table, slope, intercept, subsamples, (flagged, rest), out_dir)

    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump({
            "raw_em": {k: v for k, v in result.items() if k != "resp_proton"},
            "recovery_slope": slope, "recovery_intercept": intercept,
            "corrected_contamination": intercept / slope,
            "injection_table": table.to_dict(orient="records"),
            "subsamples": subsamples,
            "picky_transfer_raw_spread": float(spread),
            "flagged_mass_shift_mev": float(shift), "flagged_mass_pvalue": float(pval),
        }, fh, indent=2)
    print(f"  saved {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
