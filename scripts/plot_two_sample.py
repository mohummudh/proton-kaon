#!/usr/bin/env python3
"""
scripts/plot_two_sample.py

Render tables, figures and a written report from the results.json produced by
scripts/latent_two_sample.py.  Everything needed is inside the JSON, so this can
be re-run to restyle figures without recomputing anything and without the
inference drive mounted.

Usage
-----
    # re-render one run
    python scripts/plot_two_sample.py --results figs/<model>/two-sample/default/results.json

    # put two or more data configurations side by side (e.g. real vs null control)
    python scripts/plot_two_sample.py \
        --results figs/<model>/two-sample/default/results.json \
                  figs/<model>/two-sample/control/results.json \
        --compare-out figs/<model>/two-sample/_compare

Written per run:
    summary.csv        one row per comparison, the joint tests
    marginals.csv      one row per (comparison, latent dim)
    table_summary.tex  booktabs, paste-ready
    table_marginals.tex
    report.md          numbers with the interpretation attached
    ecdf_diff_*.pdf    KS figure: ECDF difference against its permutation noise floor
    ecdf_*.pdf         the two raw ECDF staircases (teaching/supplementary)
    marginal_effects.pdf
    permutation_energy.pdf
    permutation_wasserstein.pdf
    c2st.pdf
    overview.pdf       the single summary figure for the paper
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    "savefig.bbox":           "tight",
    "savefig.pad_inches":     0.02,
})

import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

BLUE   = "#0077BB"   # Paul Tol palette — colourblind-safe, matches analyse_latents
ORANGE = "#EE7733"
PURPLE = "#AA3377"
RED    = "#CC3311"
GREY   = "0.70"
SPECIES_COLOUR = {"combined": "0.35", "combined_matched": PURPLE,
                  "proton": BLUE, "kaon": ORANGE, "muon": RED}

DOUBLE_COL = 6.875   # ~175 mm
SINGLE_COL = 3.375


def _savefig(path: Path) -> None:
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close()


def md_table(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Markdown table without pulling in `tabulate` as a dependency."""
    def cell(v):
        if isinstance(v, (float, np.floating)):
            return "nan" if np.isnan(v) else format(v, floatfmt)
        if isinstance(v, (bool, np.bool_)):
            return "yes" if v else "-"
        return str(v)

    head = "| " + " | ".join(str(c) for c in df.columns) + " |"
    rule = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(cell(v) for v in row) + " |"
            for row in df.itertuples(index=False, name=None)]
    return "\n".join([head, rule, *body])


def _legend(**kw):
    base = dict(frameon=True, framealpha=0.85, edgecolor="0.75",
                handlelength=1.2, handletextpad=0.4, borderpad=0.5, labelspacing=0.35)
    base.update(kw)
    return base


# ── tables ─────────────────────────────────────────────────────────────────────

def build_summary(payload: dict) -> pd.DataFrame:
    rows = []
    for name, r in payload["comparisons"].items():
        row = {"comparison": name, "n_train": r["n_train"], "n_val": r["n_val"]}
        if "marginal" in r:
            m = r["marginal"]
            row.update({
                "max_D": m["global_max_D"],
                "max_D_p_perm": m["global_max_D_p_perm"],
                "n_holm_sig": m["n_holm_significant"],
                "max_w1_sigma": float(np.nanmax(m["w1_sigma"])),
            })
        if "energy" in r:
            e = r["energy"]
            row.update({
                "energy": e["energy"],
                "energy_norm": e["energy_normalised"],
                "energy_p": e["p_value"],
                "energy_z": e["z_score"],
                "energy_null_mean": e["null_mean"],
            })
        if "wasserstein_nd" in r:
            for k in ("W1", "W2"):
                w = r["wasserstein_nd"][k]
                row.update({
                    f"{k}_nd": w["value"],
                    f"{k}_nd_null": w["null_mean"],
                    f"{k}_nd_ratio": w["excess_over_null"],
                    f"{k}_nd_p": w["p_value"],
                })
        if "wasserstein_sliced" in r:
            s = r["wasserstein_sliced"]
            row.update({"SW1": s["value"], "SW1_null": s["null_mean"],
                        "SW1_ratio": s["excess_over_null"], "SW1_p": s["p_value"]})
        if "c2st" in r:
            c = r["c2st"]
            row["auc_null_sd"] = c["auc_null_sd"]
            for kind in ("mlp", "logreg"):
                if kind in c:
                    row[f"c2st_{kind}_auc"] = c[kind]["auc"]
                    lo, hi = c[kind].get("auc_ci95", [np.nan, np.nan])
                    row[f"c2st_{kind}_lo"], row[f"c2st_{kind}_hi"] = lo, hi
                    row[f"c2st_{kind}_draw_sd"] = c[kind].get("auc_repeat_sd", np.nan)
                    if "null_sd" in c[kind]:
                        row[f"c2st_{kind}_null_sd"] = c[kind]["null_sd"]
                        row[f"c2st_{kind}_p"] = c[kind]["p_value"]
                        row[f"c2st_{kind}_excess_sd"] = c[kind]["auc_excess_sd"]
        row["verdict"] = verdict(r)
        rows.append(row)
    return pd.DataFrame(rows)


def build_marginals(payload: dict) -> pd.DataFrame:
    rows = []
    for name, r in payload["comparisons"].items():
        if "marginal" not in r:
            continue
        m = r["marginal"]
        for d in range(m["n_dim"]):
            rows.append({
                "comparison": name,
                "dim": f"z{d}",
                "D": m["ks_D"][d],
                "p_raw": m["ks_p_raw"][d],
                "p_holm": m["ks_p_holm"][d],
                "p_perm": m["ks_p_perm"][d],
                "D_null_p95": m["ks_D_null_p95"][d],
                "w1_sigma": m["w1_sigma"][d],
                "w1_sigma_null_p95": m["w1_sigma_null_p95"][d],
                "w1_p_perm": m["w1_p_perm"][d],
                "mean_shift_sigma": m["mean_shift_sigma"][d],
                "std_ratio": m["std_ratio"][d],
                "holm_sig": m["ks_p_holm"][d] < 0.05,
            })
    return pd.DataFrame(rows)


def verdict(r: dict) -> str:
    """One-line reading of a comparison, weighting effect size over significance.

    At n in the thousands a p-value below 0.05 costs almost nothing, so the
    verdict only calls a difference real when it also clears the noise floor by
    a visible margin on an interpretable scale (AUC, or Wasserstein in sigma).
    """
    bits = []
    sig = False
    if "energy" in r:
        p = r["energy"]["p_value"]
        sig |= p < 0.05
        bits.append(f"energy p={p:.3f}")

    # effect size drives the verdict; prefer the AUC, fall back to the largest
    # marginal displacement in sigma when no C2ST was run
    auc = auc_excess = None
    for kind in ("mlp", "logreg"):
        if "c2st" in r and kind in r["c2st"]:
            c = r["c2st"][kind]
            auc = c["auc"]
            # the empirical null absorbs CV/refit/subsample instability that the
            # analytic Hanley-McNeil sd assumes away, so use it when we have it
            centre, sd, source = c2st_null(r, kind)
            auc_excess = (auc - centre) / sd if sd > 0 else 0.0
            src = "perm" if source == "permutation" else "analytic"
            bits.append(f"AUC={auc:.3f} [{kind}] ({auc_excess:+.1f} {src} sd above chance)")
            if "p_value" in c:
                sig |= c["p_value"] < 0.05
                bits.append(f"C2ST p={c['p_value']:.3f}")
            break

    wmax = None
    if "marginal" in r:
        bits.append(f"{r['marginal']['n_holm_significant']}/{r['marginal']['n_dim']} dims Holm-sig")
        wmax = float(np.nanmax(r["marginal"]["w1_sigma"]))
        bits.append(f"max shift {wmax:.3f} sigma")

    if auc_excess is not None:
        # one-sided: an AUC at or below chance is noise, never evidence of a
        # difference, so only a positive excess can escalate the verdict
        big = auc_excess >= 2
        head = ("INDISTINGUISHABLE" if not sig and not big else
                "DETECTABLE BUT NEGLIGIBLE" if not big else
                "SMALL BUT REAL" if auc < 0.55 else
                "DISTINGUISHABLE")
    elif wmax is not None:
        # no classifier available: judge on the marginal effect size alone
        head = ("INDISTINGUISHABLE" if not sig else
                "DETECTABLE BUT NEGLIGIBLE" if wmax < 0.10 else
                "SMALL BUT REAL" if wmax < 0.25 else
                "DISTINGUISHABLE")
    else:
        head = "SIGNIFICANT" if sig else "INDISTINGUISHABLE"
    return f"{head} — " + ", ".join(bits)


# ── figures ────────────────────────────────────────────────────────────────────

def fig_ecdf_difference(payload: dict, out_dir: Path) -> None:
    """The KS figure.  Plots F_train(x) - F_val(x) against a permutation band.

    At these sample sizes the two raw ECDFs are visually identical, so the
    difference curve is the only readable form.  The shaded band is the
    per-dimension permutation noise floor (95th percentile of |D| under
    reshuffled labels): a curve staying inside it is a dimension where train and
    val are indistinguishable.
    """
    for name, r in payload["comparisons"].items():
        if "marginal" not in r:
            continue
        m = r["marginal"]
        n_dim = m["n_dim"]
        ncol = 4
        nrow = int(np.ceil(n_dim / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE_COL, 1.75 * nrow),
                                 sharey=True)
        axes = np.atleast_1d(axes).ravel()
        colour = SPECIES_COLOUR.get(name, BLUE)
        for d in range(n_dim):
            ax = axes[d]
            g = np.asarray(m["ecdf_grid"][d])
            diff = np.asarray(m["ecdf_train"][d]) - np.asarray(m["ecdf_val"][d])
            band = m["ks_D_null_p95"][d]
            ax.axhspan(-band, band, color=GREY, alpha=0.45, lw=0,
                       label="permutation noise floor (95%)" if d == 0 else None)
            ax.axhline(0.0, color="0.4", lw=0.5)
            ax.plot(g, diff, color=colour, lw=1.1,
                    label=r"$F_{\mathrm{train}}-F_{\mathrm{val}}$" if d == 0 else None)
            star = "*" if m["ks_p_holm"][d] < 0.05 else ""
            ax.set_title(f"z{d}   D={m['ks_D'][d]:.3f}{star}", pad=3)
            if d >= n_dim - ncol:
                ax.set_xlabel("latent value", fontsize=9)
            if d % ncol == 0:
                ax.set_ylabel("ECDF difference")
        for ax in axes[n_dim:]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=2,
                   bbox_to_anchor=(0.5, -0.04), **_legend(frameon=False))
        fig.suptitle(f"{name}: per-dimension ECDF difference "
                     f"(* = Holm-significant at 0.05)", y=1.01)
        fig.tight_layout()
        _savefig(out_dir / f"ecdf_diff_{name}.png")


def fig_ecdf_raw(payload: dict, out_dir: Path) -> None:
    """The two ECDF staircases overlaid — what the KS statistic literally measures."""
    for name, r in payload["comparisons"].items():
        if "marginal" not in r:
            continue
        m = r["marginal"]
        n_dim = m["n_dim"]
        ncol = 4
        nrow = int(np.ceil(n_dim / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE_COL, 1.75 * nrow),
                                 sharey=True)
        axes = np.atleast_1d(axes).ravel()
        for d in range(n_dim):
            ax = axes[d]
            g = np.asarray(m["ecdf_grid"][d])
            fa = np.asarray(m["ecdf_train"][d])
            fb = np.asarray(m["ecdf_val"][d])
            ax.plot(g, fa, color=BLUE, lw=1.0, label="train" if d == 0 else None)
            ax.plot(g, fb, color=ORANGE, lw=1.0, ls="--", label="val" if d == 0 else None)
            # mark the widest vertical gap on this grid — the KS statistic
            j = int(np.argmax(np.abs(fa - fb)))
            ax.vlines(g[j], min(fa[j], fb[j]), max(fa[j], fb[j]), color=RED, lw=1.4,
                      label="max gap = D" if d == 0 else None)
            ax.set_title(f"z{d}   D={m['ks_D'][d]:.3f}", pad=3)
            if d >= n_dim - ncol:
                ax.set_xlabel("latent value", fontsize=9)
            if d % ncol == 0:
                ax.set_ylabel("ECDF")
        for ax in axes[n_dim:]:
            ax.axis("off")
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=3,
                   bbox_to_anchor=(0.5, -0.04), **_legend(frameon=False))
        fig.suptitle(f"{name}: empirical CDFs, train vs val", y=1.01)
        fig.tight_layout()
        _savefig(out_dir / f"ecdf_{name}.png")


def fig_marginal_effects(payload: dict, out_dir: Path) -> None:
    """Per-dimension Wasserstein effect size against its noise floor.

    The number that belongs next to every KS p-value: how far apart the two
    distributions are, in units of the pooled standard deviation.
    """
    comps = [(n, r) for n, r in payload["comparisons"].items() if "marginal" in r]
    if not comps:
        return
    fig, axes = plt.subplots(len(comps), 1, figsize=(DOUBLE_COL, 1.9 * len(comps)),
                             sharex=True, squeeze=False)
    for ax, (name, r) in zip(axes.ravel(), comps):
        m = r["marginal"]
        x = np.arange(m["n_dim"])
        obs = np.asarray(m["w1_sigma"])
        floor = np.asarray(m["w1_sigma_null_p95"])
        colour = SPECIES_COLOUR.get(name, BLUE)
        ax.bar(x, floor, width=0.62, color=GREY, alpha=0.55,
               label="same-distribution noise floor (95%)", zorder=1)
        ax.scatter(x, obs, s=26, color=colour, zorder=3, label="observed")
        for i in range(m["n_dim"]):
            if m["ks_p_holm"][i] < 0.05:
                ax.annotate("*", (x[i], obs[i]), textcoords="offset points",
                            xytext=(0, 5), ha="center", fontsize=11, color=RED)
        ax.set_ylabel(r"$W_1/\sigma$")
        ax.set_title(f"{name}  (n={r['n_train']} vs {r['n_val']})", pad=3, loc="left")
        ax.set_xticks(x)
        ax.set_xticklabels([f"z{i}" for i in x])
        ax.set_ylim(bottom=0)
    axes.ravel()[0].legend(**_legend(loc="upper right"))
    axes.ravel()[-1].set_xlabel("latent dimension")
    fig.suptitle("Marginal effect size vs the permutation noise floor "
                 "(* = Holm-significant KS at 0.05)", y=1.005)
    fig.tight_layout()
    _savefig(out_dir / "marginal_effects.png")


def _null_panel(ax, obs, null, p, colour, extra=None, xlabel=None, ylabel=None):
    """One permutation-null histogram with the observed value marked."""
    ax.hist(null, bins=30, color=GREY, alpha=0.75, lw=0)
    ax.axvline(obs, color=colour, lw=1.6, label="observed")
    ax.axvline(np.percentile(null, 95), color="0.35", lw=0.9, ls=":", label="null 95%")
    txt = f"p = {p:.3f}" + (f"\n{extra}" if extra else "")
    ax.annotate(txt, xy=(0.05, 0.96), xycoords="axes fraction",
                ha="left", va="top", fontsize=7)
    ax.xaxis.set_major_locator(MaxNLocator(3))
    ax.yaxis.set_major_locator(MaxNLocator(4))
    ax.tick_params(labelsize=7.5)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=8.5)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=8.5)


def fig_permutation_energy(payload: dict, out_dir: Path) -> None:
    """What a permutation test actually is, drawn: the grey histogram is the
    distribution the statistic would have had if train and val matched."""
    comps = [(n, r) for n, r in payload["comparisons"].items() if "energy" in r]
    if not comps:
        return
    fig, axes = plt.subplots(1, len(comps), figsize=(DOUBLE_COL, 2.3), squeeze=False)
    for j, (ax, (name, r)) in enumerate(zip(axes.ravel(), comps)):
        e = r["energy"]
        _null_panel(ax, e["energy"], np.asarray(e["null_samples"]), e["p_value"],
                    SPECIES_COLOUR.get(name, BLUE),
                    extra=f"z = {e['z_score']:+.1f}",
                    xlabel="energy distance",
                    ylabel="permutation replicates" if j == 0 else None)
        ax.set_title(f"{name}\n(n={e['n_a']}/{e['n_b']})", fontsize=8.5, pad=3)
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.08), **_legend(frameon=False))
    fig.suptitle("Energy distance against its permutation null", y=1.02)
    fig.tight_layout()
    _savefig(out_dir / "permutation_energy.png")


def fig_permutation_wasserstein(payload: dict, out_dir: Path) -> None:
    """Same layout as the energy figure, and the contrast is the point: here the
    null histogram sits far from zero because the 8-D empirical Wasserstein
    estimator is dominated by finite-sample bias."""
    comps = [(n, r) for n, r in payload["comparisons"].items() if "wasserstein_nd" in r]
    if not comps:
        return
    variants = [("W1 (exact, 8-D)", lambda r: r["wasserstein_nd"]["W1"]),
                ("W2 (exact, 8-D)", lambda r: r["wasserstein_nd"]["W2"])]
    if any("wasserstein_sliced" in r for _, r in comps):
        variants.append(("sliced W1 (full n)", lambda r: r.get("wasserstein_sliced")))
    fig, axes = plt.subplots(len(variants), len(comps),
                             figsize=(DOUBLE_COL, 1.85 * len(variants)), squeeze=False)
    for i, (vname, getter) in enumerate(variants):
        for j, (name, r) in enumerate(comps):
            ax = axes[i, j]
            w = getter(r)
            if w is None:
                ax.axis("off")
                continue
            _null_panel(ax, w["value"], np.asarray(w["null_samples"]), w["p_value"],
                        SPECIES_COLOUR.get(name, BLUE),
                        extra=f"obs/null = {w['excess_over_null']:.3f}",
                        xlabel="distance" if i == len(variants) - 1 else None,
                        ylabel=vname if j == 0 else None)
            if i == 0:
                ax.set_title(name, fontsize=8.5, pad=3)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2,
               bbox_to_anchor=(0.5, -0.05), **_legend(frameon=False))
    fig.suptitle("Multivariate Wasserstein against its permutation null\n"
                 "(the null sits far from zero — that is finite-sample bias, not signal)",
                 y=1.02, fontsize=10)
    fig.tight_layout()
    _savefig(out_dir / "permutation_wasserstein.png")


def c2st_null(r: dict, kind: str) -> tuple:
    """(centre, sd, source) for a C2ST result — permutation null where available.

    The fold-to-fold CI is not an honest error bar here: every fold shares one
    balanced subsample and one seed, so it misses the refit instability that
    moves the AUC most.  The permutation null does capture it.
    """
    c = r["c2st"][kind]
    if "null_sd" in c:
        # single-draw null spread: wider than the spread of the reported mean,
        # so the band it draws is the conservative one
        return c["null_mean"], c["null_sd"], "permutation"
    return 0.5, r["c2st"]["auc_null_sd"], "analytic"


def c2st_err(r: dict, kind: str) -> list:
    """x-error for a C2ST point: the null spread when we have it, else the
    across-draw spread of the estimate itself."""
    c = r["c2st"][kind]
    _, sd, source = c2st_null(r, kind)
    if source == "permutation":
        return [[1.96 * sd], [1.96 * sd]]
    lo, hi = c.get("auc_ci95", [np.nan, np.nan])
    if np.isnan(lo):
        return [[0.0], [0.0]]
    return [[c["auc"] - lo], [hi - c["auc"]]]


def fig_c2st(payload: dict, out_dir: Path) -> None:
    comps = [(n, r) for n, r in payload["comparisons"].items() if "c2st" in r]
    if not comps:
        return
    kinds = [k for k in ("mlp", "logreg") if k in comps[0][1]["c2st"]]
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 0.55 * len(comps) * len(kinds) + 1.2))
    ypos, ylabels = [], []
    y = 0.0
    source = "analytic"
    for name, r in comps:
        for kind in kinds:
            c = r["c2st"][kind]
            centre, sd, source = c2st_null(r, kind)
            # each row gets its own noise floor: the null width depends on n,
            # and one shared band drawn at the widest row would make the
            # largest-n comparisons look far less significant than they are
            ax.barh(y, 4 * sd, left=centre - 2 * sd, height=0.72,
                    color=GREY, alpha=0.5, lw=0, zorder=0)
            ax.errorbar(c["auc"], y, xerr=c2st_err(r, kind),
                        fmt="o", ms=5, capsize=2.5, lw=1.1, zorder=3,
                        color=SPECIES_COLOUR.get(name, BLUE),
                        mfc="white" if kind == "logreg" else None)
            ypos.append(y)
            ylabels.append(f"{name} ({kind})")
            y += 1
        y += 0.4
    ax.barh(np.nan, 0, color=GREY, alpha=0.5, lw=0,
            label=rf"chance $\pm 2\,\mathrm{{sd}}$ ({source} null, per row)")
    ax.axvline(0.5, color="0.3", lw=0.8, ls="--", label="chance (AUC = 0.5)")
    ax.set_yticks(ypos)
    ax.set_yticklabels(ylabels)
    ax.invert_yaxis()
    ax.set_xlabel("cross-validated AUC, train latents vs val latents")
    ax.legend(ncol=2, bbox_to_anchor=(0.5, -0.14), loc="upper center",
              **_legend(frameon=False))
    ax.set_title("Classifier two-sample test  (filled = MLP, hollow = logistic)", loc="left")
    fig.tight_layout()
    _savefig(out_dir / "c2st.png")


def fig_overview(payload: dict, out_dir: Path) -> None:
    """The one figure for the paper: effect size on the left, formal test in the
    middle, per-dimension detail on the right."""
    comps = list(payload["comparisons"].items())
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6))

    # (a) C2ST AUC
    ax = axes[0]
    names = [n for n, r in comps if "c2st" in r and "mlp" in r["c2st"]]
    if names:
        y = np.arange(len(names))
        aucs = [payload["comparisons"][n]["c2st"]["mlp"]["auc"] for n in names]
        nulls = [c2st_null(payload["comparisons"][n], "mlp") for n in names]
        band = 2 * max(sd for _, sd, _ in nulls)
        ax.axvspan(0.5 - band, 0.5 + band, color=GREY, alpha=0.45, lw=0)
        ax.axvline(0.5, color="0.3", lw=0.8, ls="--")
        for i, n in enumerate(names):
            err = c2st_err(payload["comparisons"][n], "mlp")
            ax.errorbar(aucs[i], y[i], xerr=err, fmt="o",
                        ms=5, capsize=2.5, lw=1.1, color=SPECIES_COLOUR.get(n, BLUE))
        ax.set_yticks(y)
        ax.set_yticklabels(names)
        ax.invert_yaxis()
        ax.set_xlabel("C2ST AUC")
        ax.set_title("(a) separability", loc="left", pad=3)

    # (b) energy: observed vs null spread, in null-sd units
    ax = axes[1]
    names = [n for n, r in comps if "energy" in r]
    if names:
        y = np.arange(len(names))
        z = [payload["comparisons"][n]["energy"]["z_score"] for n in names]
        ax.axvspan(-2, 2, color=GREY, alpha=0.45, lw=0)
        ax.axvline(0.0, color="0.3", lw=0.8, ls="--")
        ax.barh(y, z, height=0.5,
                color=[SPECIES_COLOUR.get(n, BLUE) for n in names])
        # leave room on both sides so the p-labels never run into the spines
        lo, hi = min(min(z), -2.5), max(max(z), 2.5)
        pad = 0.45 * (hi - lo)
        ax.set_xlim(lo - pad, hi + pad)
        for i, n in enumerate(names):
            p = payload["comparisons"][n]["energy"]["p_value"]
            ax.annotate(f"p={p:.3f}", (z[i], y[i]), textcoords="offset points",
                        xytext=(4 if z[i] >= 0 else -4, 0), fontsize=8,
                        va="center", ha="left" if z[i] >= 0 else "right")
        ax.set_yticks(y)
        ax.set_yticklabels([])
        ax.invert_yaxis()
        ax.set_xlabel("energy distance (null sd units)")
        ax.set_title("(b) joint test", loc="left", pad=3)

    # (c) per-dimension marginal effect sizes
    ax = axes[2]
    plotted, top = False, 0.0
    for n, r in comps:
        if "marginal" not in r:
            continue
        m = r["marginal"]
        x = np.arange(m["n_dim"])
        ax.plot(x, m["w1_sigma"], "o-", ms=3.5, lw=0.9,
                color=SPECIES_COLOUR.get(n, BLUE), label=n)
        ax.plot(x, m["w1_sigma_null_p95"], ls=":", lw=0.8,
                color=SPECIES_COLOUR.get(n, BLUE), alpha=0.7)
        top = max(top, np.nanmax(m["w1_sigma"]), np.nanmax(m["w1_sigma_null_p95"]))
        plotted = True
    if plotted:
        ax.set_xticks(x)
        ax.set_xticklabels([f"z{i}" for i in x], fontsize=8)
        ax.set_ylabel(r"$W_1/\sigma$")
        # headroom for the legend, so no series is hidden behind it or clipped
        ax.set_ylim(0, top * 1.45)
        ax.set_title("(c) per-dimension (dotted = noise floor)", loc="left", pad=3)
        ax.legend(ncol=2, **_legend(loc="upper center", fontsize=6.5))
    fig.tight_layout()
    _savefig(out_dir / "overview.png")


# ── report ─────────────────────────────────────────────────────────────────────

PREAMBLE = """\
# Train vs. val latent two-sample tests

**Read the effect sizes first, the p-values second.** Train and val here are a
random split of one sample, so under the null they genuinely are the same
distribution. A significant result is evidence that the *encoder* treats images
it was trained on differently from images it was not — memorisation — not
evidence of covariate shift. And at n in the thousands, statistical significance
is nearly free: the tests below have the power to flag shifts far too small to
matter physically. That is why every number is paired with a noise floor.

| level | test | what it catches |
|---|---|---|
| marginal (8 x 1-D) | KS per dim + Holm | per-axis shifts and scale changes; names the culprit dimension |
| joint (8-D) | energy distance, permutation-calibrated | correlation-structure changes the marginals are blind to |
| joint (8-D) | Wasserstein W1/W2, permutation-calibrated | geometric displacement — but see the bias warning below |
| joint, interpretable | C2ST | the same, reported as *how separable*, in AUC |

**The noise floor.** Every joint statistic is compared against a permutation
null: pool the two groups, reshuffle the labels, recompute,
repeat. Under the null the labels carry no information, so the reshuffled values
are draws from exactly the distribution the statistic would have had if the two
groups matched. That converts "energy = 0.003, is that big?" into an answer.

**Repeats.** The energy, Wasserstein and C2ST kernels all have to subsample, and
one draw is not reproducible — on this data a single draw moved the exact-
Wasserstein p-value between 0.01 and 0.92. Each is therefore run several times,
each repeat a complete permutation test on its own draw, and the p-values are
merged as twice their median, which is valid under arbitrary dependence between
repeats. It is mildly conservative, so a p-value here will not be smaller than
2/(permutations + 1); the reported statistic is the mean across draws and the
+/- next to it is the spread across them.

**`combined` vs `combined_matched`.** A balanced split makes the *training* set
an exact even mix of species and leaves the validation remainder unbalanced. The
two pools then differ in species composition, and since species are separable in
latent space by construction, a pooled train-vs-val test will fire for that
reason alone — nothing to do with train/val membership. `combined_matched`
resamples the validation pool to the training mixture first. Quote the matched
one; the raw `combined` row is retained only to show the size of the confound.

**Why the Wasserstein numbers look alarming and are not.** The empirical
multivariate Wasserstein estimator converges at n^(-1/d), so in 8-D the value
computed from a few thousand points is dominated by finite-sample bias rather
than by any real difference. Two samples drawn from the *identical* distribution
give a comfortably non-zero W. That is why the tables report the ratio of the
observed value to the permutation-null mean: only the ratio is meaningful, and a
ratio near 1.000 means "indistinguishable", not "no distance". Energy distance
does not have this problem — its estimator is unbiased under the null, which is
why it is the primary joint test here.
"""


def write_report(payload: dict, summary: pd.DataFrame, marg: pd.DataFrame,
                 out_dir: Path) -> None:
    meta = payload["meta"]
    L = [PREAMBLE, "\n## Run\n"]
    L.append("| field | value |\n|---|---|")
    for k in ("model_name", "label", "null_control", "standardize", "subsample_frac",
              "seed", "n_perm", "energy_n", "energy_repeats", "wnd_n", "wnd_perm",
              "wnd_repeats", "sliced_n_proj", "c2st_folds", "c2st_perm",
              "c2st_repeats", "git_commit", "timestamp"):
        L.append(f"| `{k}` | `{meta.get(k)}` |")
    L.append(f"\n```\n{meta.get('command', '')}\n```\n")
    if meta.get("null_control"):
        L.append("> **This is a negative control run.** Train/val labels were reshuffled, "
                 "so the null is true by construction. Every test here should come back "
                 "non-significant; use these numbers as the reference column for the real run.\n")

    L.append("\n## Verdicts\n")
    L.append("| comparison | n train | n val | verdict |\n|---|---|---|---|")
    for _, r in summary.iterrows():
        L.append(f"| **{r['comparison']}** | {r['n_train']} | {r['n_val']} | {r['verdict']} |")

    for name, r in payload["comparisons"].items():
        L.append(f"\n## {name}\n")
        if "species_fractions" in r:
            sf = r["species_fractions"]
            L.append("Species composition of the two pools:\n")
            L.append("| species | train | val |\n|---|---|---|")
            for s in sf["train"]:
                L.append(f"| {s} | {sf['train'][s]:.1%} | {sf['val'][s]:.1%} |")
            if name == "combined" and sf["max_abs_difference"] > 0.01:
                L.append(
                    f"\n> **Confounded — do not report this comparison on its own.** The two "
                    f"pools differ in species mixture by up to {sf['max_abs_difference']:.1%}, "
                    f"because the balanced split makes the training set exact thirds and leaves "
                    f"the remainder unbalanced. Species are separable in latent space by "
                    f"construction, so any difference found here is at least partly a "
                    f"composition effect and not evidence about train/val membership. "
                    f"`combined_matched` repeats it with the val pool resampled to the train "
                    f"mixture; that is the one to quote.\n")
            elif name == "combined_matched":
                taken = r.get("val_taken_per_species", {})
                L.append(f"\nVal pool resampled to the train mixture: "
                         f"{', '.join(f'{s} n={n}' for s, n in taken.items())}. "
                         f"With composition held fixed, anything left is about train/val "
                         f"membership.\n")
        if "marginal" in r:
            m = r["marginal"]
            L.append("### 1. Marginal KS + Holm, with Wasserstein effect sizes\n")
            sub = marg[marg["comparison"] == name].drop(columns=["comparison"])
            L.append(md_table(sub))
            L.append(
                f"\nGlobal (max D over all {m['n_dim']} dims, permutation-calibrated so no "
                f"multiplicity correction is needed): **D = {m['global_max_D']:.4f}, "
                f"p = {m['global_max_D_p_perm']:.4f}** "
                f"(null 95% = {m['global_max_D_null_p95']:.4f}). "
                f"{m['n_holm_significant']} of {m['n_dim']} dimensions are Holm-significant "
                f"at 0.05. Largest marginal displacement: "
                f"**{np.nanmax(m['w1_sigma']):.4f} sigma** in "
                f"z{int(np.nanargmax(m['w1_sigma']))}.\n")
        if "energy" in r:
            e = r["energy"]
            L.append("### 2. Energy distance (joint, 8-D)\n")
            L.append(
                f"- E = **{e['energy']:.6f}** on {e['n_a']} vs {e['n_b']} subsampled points\n"
                f"- permutation null: mean {e['null_mean']:.6f}, sd {e['null_sd']:.6f}, "
                f"95% {e['null_p95']:.6f} ({e['n_perm']} replicates)\n"
                f"- **p = {e['p_value']:.4f}**, z = {e['z_score']:+.2f} null sd\n"
                f"- normalised (0 = identical, 1 = fully separated): "
                f"{e['energy_normalised']:.6f}\n")
        if "wasserstein_nd" in r:
            L.append("### 3. Wasserstein (joint, 8-D)\n")
            L.append("| variant | observed | null mean | obs/null | p |\n|---|---|---|---|---|")
            for k in ("W1", "W2"):
                w = r["wasserstein_nd"][k]
                L.append(f"| {k} exact (n={r['wasserstein_nd']['n_per_side']}/side) | "
                         f"{w['value']:.4f} | {w['null_mean']:.4f} | "
                         f"{w['excess_over_null']:.4f} | {w['p_value']:.4f} |")
            if "wasserstein_sliced" in r:
                s = r["wasserstein_sliced"]
                L.append(f"| sliced W1 (full n, {s['n_proj']} projections) | "
                         f"{s['value']:.5f} | {s['null_mean']:.5f} | "
                         f"{s['excess_over_null']:.4f} | {s['p_value']:.4f} |")
            w1 = r["wasserstein_nd"]["W1"]
            L.append(f"\nThe null mean of {w1['null_mean']:.4f} is what two samples from the "
                     f"*same* distribution give at this n and dimension. The observed "
                     f"{w1['value']:.4f} is {w1['excess_over_null']:.4f} times that. Report the "
                     f"ratio, never the raw distance.\n")
        if "c2st" in r:
            c = r["c2st"]
            L.append("### 4. Classifier two-sample test\n")
            L.append(f"Balanced at {c['n_per_class']} per class, {c['n_folds']}-fold CV. "
                     f"Analytic (Hanley-McNeil) null sd at this n: {c['auc_null_sd']:.4f}.\n")
            L.append("| model | AUC | accuracy | perm null (mean +/- sd) | excess | p |"
                     "\n|---|---|---|---|---|---|")
            for kind in ("mlp", "logreg"):
                if kind not in c:
                    continue
                x = c[kind]
                if "null_sd" in x:
                    nullcol = f"{x['null_mean']:.4f} +/- {x['null_sd']:.4f}"
                    exc = f"{x['auc_excess_sd']:+.2f} sd"
                    pcol = f"{x['p_value']:.4f}"
                else:
                    nullcol = f"0.5 +/- {c['auc_null_sd']:.4f} (analytic)"
                    exc = f"{(x['auc'] - 0.5) / c['auc_null_sd']:+.2f} sd"
                    pcol = "-"
                L.append(f"| {kind} | {x['auc']:.4f} +/- {x.get('auc_repeat_sd', float('nan')):.4f} | "
                         f"{x['accuracy']:.4f} | {nullcol} | {exc} | {pcol} |")
            emp = [c[k]["null_sd"] for k in ("mlp", "logreg") if k in c and "null_sd" in c[k]]
            if emp:
                pc = next((c[k].get("p_combination") for k in ("mlp", "logreg")
                           if k in c and "p_combination" in c[k]), "")
                L.append(
                    f"\nAUC is the mean over {c.get('n_repeats', 1)} balanced subsample draws, "
                    f"+/- the spread across those draws. The empirical null sd ({max(emp):.4f}) "
                    f"is {max(emp) / c['auc_null_sd']:.1f}x the analytic Hanley-McNeil value "
                    f"({c['auc_null_sd']:.4f}): the gap is subsample and refit instability, "
                    f"which the analytic formula assumes away, so quote the empirical one. "
                    f"p-value combination across draws: {pc}.\n")
            L.append("")

    L.append(WHAT_TO_REPORT)
    (out_dir / "report.md").write_text("\n".join(L))


WHAT_TO_REPORT = """
## What to put in the paper

**One figure, one table, three sentences** is the right budget for a validation
check like this.

*Figure* — `overview.pdf`. Panel (a) C2ST AUC per comparison with the chance
band, panel (b) the energy test in null-sd units, panel (c) per-dimension
Wasserstein effect sizes against their noise floor. If you only have room for
one panel, use (a): a forest plot of AUCs sitting on 0.5 is immediately legible
and needs no statistics background from the reader.

*Table* — `table_marginals.tex` for the supplement (per-dimension D, Holm p,
W/sigma), `table_summary.tex` for the main text if a table is wanted at all.

*Prose* — state the design (random split, so the null is true by construction
and a positive result would mean encoder memorisation), give the C2ST AUC with
its permutation null as the headline effect size, and give the energy-test
p-value as the formal statement. Mention the largest per-dimension displacement
in sigma so a reader can judge physical relevance. Report the per-species rows
and `combined_matched`; if you quote a pooled number, say that the pools were
composition-matched, because an unmatched pooled test is not measuring what the
sentence around it will claim.

*Do not* lead with the raw 8-D Wasserstein value. Quote it only as a ratio to
its permutation null, and only if a reviewer asks: it is the least informative
number here and the most easily misread.

*If the tests do fire*, the useful follow-ups in order are: (1) which dimensions,
from the marginal table; (2) whether the effect survives `--subsample-frac 0.25`
at a proportionally similar size (if the AUC holds but the p-value degrades, the
effect is real but small; if the AUC collapses too, it was n-driven);
(3) `--standardize none` to check the result is not an artefact of upweighting
collapsed latent dimensions; (4) `--null-control`, which must come back clean.
"""


def write_latex(summary: pd.DataFrame, marg: pd.DataFrame, out_dir: Path) -> None:
    cols = [c for c in ["comparison", "n_train", "n_val", "max_D", "n_holm_sig",
                        "energy", "energy_p", "W1_nd_ratio", "c2st_mlp_auc",
                        "auc_null_sd"] if c in summary.columns]
    summary[cols].to_latex(out_dir / "table_summary.tex", index=False,
                           float_format="%.4f", escape=True,
                           caption="Train vs. validation latent two-sample tests.",
                           label="tab:two-sample-summary", position="htbp")
    mcols = [c for c in ["comparison", "dim", "D", "p_raw", "p_holm", "w1_sigma",
                         "w1_sigma_null_p95", "mean_shift_sigma", "std_ratio"]
             if c in marg.columns]
    marg[mcols].to_latex(out_dir / "table_marginals.tex", index=False,
                         float_format="%.4f", escape=True,
                         caption="Per-dimension marginal comparison of train and "
                                 "validation latents.",
                         label="tab:two-sample-marginals", position="htbp",
                         longtable=True)


def render_all(payload: dict, out_dir: Path) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = build_summary(payload)
    marg = build_marginals(payload)
    summary.to_csv(out_dir / "summary.csv", index=False)
    if not marg.empty:
        marg.to_csv(out_dir / "marginals.csv", index=False)
    write_latex(summary, marg, out_dir)

    fig_ecdf_difference(payload, out_dir)
    fig_ecdf_raw(payload, out_dir)
    fig_marginal_effects(payload, out_dir)
    fig_permutation_energy(payload, out_dir)
    fig_permutation_wasserstein(payload, out_dir)
    fig_c2st(payload, out_dir)
    fig_overview(payload, out_dir)

    write_report(payload, summary, marg, out_dir)

    print("\n" + "=" * 78)
    with pd.option_context("display.width", 200, "display.max_columns", 50):
        show = [c for c in ["comparison", "n_train", "n_val", "max_D", "n_holm_sig",
                            "energy_p", "energy_z", "W1_nd_ratio", "c2st_mlp_auc",
                            "c2st_mlp_null_sd", "c2st_mlp_p"] if c in summary.columns]
        print(summary[show].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("=" * 78)
    for _, r in summary.iterrows():
        print(f"  {r['comparison']:<10} {r['verdict']}")
    print(f"\nTables, figures and report.md written to {out_dir}")


# ── cross-configuration comparison ─────────────────────────────────────────────

def render_comparison(payloads: list, out_dir: Path) -> None:
    """Stack several runs (different splits, scalings, subsample fractions) into
    one table and one figure, so the configurations can be judged side by side."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for p in payloads:
        df = build_summary(p)
        df.insert(0, "run", p["meta"]["label"])
        df.insert(1, "null_control", p["meta"]["null_control"])
        df.insert(2, "standardize", p["meta"]["standardize"])
        df.insert(3, "subsample_frac", p["meta"]["subsample_frac"])
        frames.append(df)
    allsum = pd.concat(frames, ignore_index=True)
    allsum.to_csv(out_dir / "compare_summary.csv", index=False)

    comps = sorted(allsum["comparison"].unique())
    runs = list(dict.fromkeys(allsum["run"]))
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 0.42 * len(allsum) + 1.4),
                             sharey=True)
    y = np.arange(len(allsum))
    labels = [f"{r['run']} / {r['comparison']}" for _, r in allsum.iterrows()]
    marks = {run: m for run, m in zip(runs, ["o", "s", "^", "D", "v", "P", "X"] * 4)}

    ax = axes[0]
    if "c2st_mlp_auc" in allsum.columns:
        # prefer the permutation null spread; fall back to the analytic sd or
        # the across-draw CI, whichever this run actually recorded
        sd_col = ("c2st_mlp_null_sd" if "c2st_mlp_null_sd" in allsum.columns
                  else "auc_null_sd")
        band = 2 * allsum[sd_col].max()
        ax.axvspan(0.5 - band, 0.5 + band, color=GREY, alpha=0.45, lw=0)
        ax.axvline(0.5, color="0.3", lw=0.8, ls="--")
        for i, (_, r) in enumerate(allsum.iterrows()):
            e = 1.96 * r[sd_col]
            if np.isnan(e):
                e = 0.0
            ax.errorbar(r["c2st_mlp_auc"], y[i], xerr=[[e], [e]],
                        fmt=marks.get(r["run"], "o"), ms=4.5, capsize=2, lw=1.0,
                        color=SPECIES_COLOUR.get(r["comparison"], BLUE))
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("C2ST AUC")

    ax = axes[1]
    if "energy_z" in allsum.columns:
        ax.axvspan(-2, 2, color=GREY, alpha=0.45, lw=0)
        ax.axvline(0.0, color="0.3", lw=0.8, ls="--")
        ax.barh(y, allsum["energy_z"], height=0.6,
                color=[SPECIES_COLOUR.get(c, BLUE) for c in allsum["comparison"]])
    ax.set_xlabel("energy distance (null sd units)")
    fig.suptitle("Configurations side by side", y=1.01)
    fig.tight_layout()
    _savefig(out_dir / "compare_overview.png")

    lines = ["# Two-sample tests across data configurations\n",
             "Each row is one comparison from one run. A run flagged "
             "`null_control` had its train/val labels reshuffled, so it is the "
             "reference column: any pattern present there is not a finding.\n"]
    show = [c for c in ["run", "null_control", "standardize", "subsample_frac",
                        "comparison", "n_train", "n_val", "max_D", "n_holm_sig",
                        "energy_p", "energy_z", "W1_nd_ratio", "c2st_mlp_auc",
                        "auc_null_sd"] if c in allsum.columns]
    lines.append(md_table(allsum[show]))
    lines.append("\n## Verdicts\n")
    for _, r in allsum.iterrows():
        lines.append(f"- **{r['run']} / {r['comparison']}** — {r['verdict']}")
    (out_dir / "compare_report.md").write_text("\n".join(lines))
    print(f"Comparison written to {out_dir}")


def main():
    ap = argparse.ArgumentParser(description="Render two-sample latent test results.")
    ap.add_argument("--results", nargs="+", required=True,
                    help="One or more results.json files from latent_two_sample.py")
    ap.add_argument("--out", default=None,
                    help="Output dir for a single run (default: alongside results.json)")
    ap.add_argument("--compare-out", default=None,
                    help="Also write a side-by-side comparison of all --results here")
    args = ap.parse_args()

    payloads = []
    for path in args.results:
        with open(path) as f:
            payload = json.load(f)
        payloads.append(payload)
        out_dir = Path(args.out) if args.out else Path(path).parent
        print(f"\n### {payload['meta']['label']}  ({payload['meta']['model_name']})")
        render_all(payload, out_dir)

    if args.compare_out:
        render_comparison(payloads, Path(args.compare_out))
    elif len(payloads) > 1:
        print("\n(pass --compare-out DIR to also write the side-by-side comparison)")


if __name__ == "__main__":
    main()
