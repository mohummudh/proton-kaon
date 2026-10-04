#!/usr/bin/env python3
"""
scripts/extra/fit_nonpicky_composition.py

Measure the composition of the kaon beamline window (350-650 MeV) for
NON-picky events, instead of assuming it equals the picky composition that
fit_kaon_peak.py measures. Uses only picky+match.csv.

WHY THIS IS NEEDED
    The picky fraction is not flat in beamline mass: it is ~0.33 on the light
    and kaon peaks, ~0.5 on the proton peak, and ~0.07-0.12 in the valleys
    either side of the kaon peak. Non-picky masses are smeared, so the
    non-picky window is filled by migrants from the proton and pi/mu peaks,
    which are 20-40x larger than the kaon peak. Its composition cannot be
    assumed equal to the picky one.

METHOD (forward folding), in x = signed m^2 [GeV^2]
    x is the natural variable: the beamline computes x = p^2 (1/beta^2 - 1),
    so the light peak is one peak near x = 0 (in signed m it splits in two
    around 0), a momentum error multiplies x by k^2, and a TOF error adds to x.

    Stage 1  Fit the full picky spectrum with a light (pi/mu/e), kaon, proton
             and flat component. The per-species shapes are frozen as the
             true lineshapes.
    Stage 2  Fit the full non-picky spectrum as the stage-1 shapes smeared by
                 x' = s * x + a,   log s ~ double Gaussian (momentum error),
                                   a     ~ Gaussian(s) (TOF error),
             shared by all species, with free per-species yields, plus a
             broad background for mismatched reconstructions. Every shape is a
             Gaussian or an exponentially-modified Gaussian, and both families
             are closed under scaling and Gaussian convolution, so all bin and
             window integrals are analytic.
    Result   Integrate each smeared component over the window.

    The large pi/mu and proton peaks fix their own smeared tails from the
    regions they dominate, so their leakage into the window is constrained by
    events outside it; the kaon yield is what remains.

CHECKS
    closure     Smear the picky EVENTS with known kernels that lie OUTSIDE the
                fit's kernel family (t-distributed scale, one-sided TOF
                offsets, plus uniform junk), label them softly with the stage-1
                posterior, refit, and compare to the true window composition.
                Tests stage 2 (identifiability and kernel misspecification),
                not the stage-1 shapes.
    kaon_null   Refit with the kaon yield fixed to zero; 2*dNLL is the evidence
                for a smeared kaon component in the non-picky spectrum.
    kaon_shift  Refit with the kaon position free; the fitted non-picky kaon
                peak should land near the picky one. NOTE: with a per-species
                kernel the shift is nearly degenerate with the kaon's own scale
                bias, so this test is uninformative there -- read the kaon
                kernel's core bias (s0_bias_kaon) from the nominal fit instead.
    variants    Template shapes x kernels x fit ranges. The spread over
                variants that fit acceptably is the systematic.

OUTPUTS (under figs/beamline_mass_fit/ or --out-dir)
    nonpicky_fit.{png,pdf}        the nominal stage-2 fit, full range and window
    nonpicky_variants.{png,pdf}   kaon fraction per variant, picky vs non-picky
    nonpicky_metrics.json

Usage:
    python scripts/extra/fit_nonpicky_composition.py
    python scripts/extra/fit_nonpicky_composition.py --n-boot 0 --no-closure   # quick
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import erfcx, ndtr

from _beam_data import (COLOURS, DEFAULT_PICKY, DOUBLE_COL, PDG_MASS,
                        PROJECT_ROOT, SINGLE_COL, apply_style, savefig)

WINDOW_MEV = (350.0, 650.0)
WINDOW = (WINDOW_MEV[0] ** 2 / 1e6, WINDOW_MEV[1] ** 2 / 1e6)  # in x [GeV^2]
SPECIES = ("light", "kaon", "proton")
KIND = {"g": 0, "er": 1, "el": 2}  # Gaussian, right-tailed exGaussian, left-tailed

SQRT2 = np.sqrt(2.0)
GH_T, GH_W = np.polynomial.hermite.hermgauss(11)
GH_W = GH_W / np.sqrt(np.pi)


# --------------------------------------------------------------------------- shapes

def _exg_right_tail(z, sig, lam):
    """The subtracted term in the right-tailed exGaussian CDF, overflow-safe."""
    a = z / sig
    u = (lam * sig - a) / SQRT2
    t1 = 0.5 * erfcx(np.maximum(u, -25.0)) * np.exp(-0.5 * a * a)
    t2 = np.exp(np.minimum(-lam * z + 0.5 * (lam * sig) ** 2, 700.0)) * ndtr(a - lam * sig)
    return np.where(u > -25.0, t1, t2)


def cdf(kind, mu, sig, lam, x):
    """CDF of a Gaussian (kind 0) or exGaussian with a right (1) / left (2) tail.

    All parameter arrays broadcast against x. exGaussian = Gauss(mu, sig)
    convolved with an exponential of rate lam (mean tail length 1/lam).
    """
    z = x - mu
    g = ndtr(z / sig)
    right = g - _exg_right_tail(z, sig, lam)
    left = 1.0 - (ndtr(-z / sig) - _exg_right_tail(-z, sig, lam))
    return np.where(kind == 0, g, np.where(kind == 1, right, left))


class Shape:
    """A normalised mixture of Gaussian / exGaussian components, as flat arrays."""

    def __init__(self, kind, mu, sig, lam, w):
        self.kind, self.mu, self.sig, self.lam = map(np.asarray, (kind, mu, sig, lam))
        self.w = np.asarray(w) / np.sum(w)

    def smear(self, s, sw, adds):
        """Apply x' = s*x + a over scale nodes (s, sw) and additive Gaussians.

        adds is a list of (mean, width, weight). Returns a new, larger Shape.
        """
        out = []
        for a, tau, aw in adds:
            S = s[:, None]
            out.append((np.broadcast_to(self.kind, (len(s), len(self.kind))),
                        S * self.mu + a,
                        np.sqrt((S * self.sig) ** 2 + tau ** 2),
                        self.lam / S,
                        sw[:, None] * self.w * aw))
        cat = [np.concatenate([o[i].ravel() for o in out]) for i in range(5)]
        return Shape(*cat)

    def integral(self, x):
        """CDF of the mixture at points x (1-D); returns an array like x."""
        c = cdf(self.kind[:, None], self.mu[:, None], self.sig[:, None],
                self.lam[:, None], x[None, :])
        return self.w @ c

    def bins(self, edges):
        return np.diff(self.integral(edges))

    def window(self, w=WINDOW):
        return float(np.diff(self.integral(np.asarray(w)))[0])


# --------------------------------------------------------------------------- params

class Params:
    """Named parameters with initial values and bounds, for L-BFGS-B."""

    def __init__(self):
        self.names, self.init, self.bounds = [], [], []

    def add(self, name, init, lo, hi):
        self.names.append(name); self.init.append(init); self.bounds.append((lo, hi))

    def unpack(self, theta):
        return dict(zip(self.names, theta))

    def fit(self, nll, n_starts, seed, fixed=None, jitter=0.15):
        """Multi-start L-BFGS-B. `fixed` pins named parameters to values."""
        fixed = fixed or {}
        free = [i for i, n in enumerate(self.names) if n not in fixed]
        full = np.array(self.init, float)
        for n, v in fixed.items():
            full[self.names.index(n)] = v

        def f(t):
            full_t = full.copy(); full_t[free] = t
            return nll(full_t)

        lo = np.array([self.bounds[i][0] for i in free]); hi = np.array([self.bounds[i][1] for i in free])
        rng = np.random.default_rng(seed)
        best = None
        for k in range(n_starts):
            start = full[free] + (0 if k == 0 else rng.normal(0, jitter, len(free)) * (hi - lo) * 0.1)
            start = np.clip(start, lo, hi)
            r = minimize(f, start, method="L-BFGS-B", bounds=list(zip(lo, hi)),
                         options=dict(maxiter=20000, maxfun=400000))
            if best is None or r.fun < best.fun:
                best = r
        out = full.copy(); out[free] = best.x
        return out, float(best.fun)


def poisson_nll(model, counts):
    model = np.clip(model, 1e-9, None)
    return float(np.sum(model - counts * np.log(model)))


def chi2_dof(model, counts, n_par):
    """Neyman-style chi2 with bins of expected < 5 merged away (only a guide)."""
    ok = model > 5
    return float(np.sum((counts[ok] - model[ok]) ** 2 / model[ok]) / max(ok.sum() - n_par, 1))


# --------------------------------------------------------------------------- stage 1

# Template variants for the picky lineshapes. Each species is a list of
# (kind, mu-init, mu-bounds, sigma-init, tail-length-init or None).
TEMPLATES = {
    # Gaussian mixtures; the kaon's wide Gaussian may sit off-centre.
    "gauss": {
        "light": [("g", 0.015, (-0.03, 0.05), 0.016, None), ("g", 0.006, (-0.05, 0.08), 0.034, None),
                  ("g", -0.07, (-0.15, 0.15), 0.04, None)],
        "kaon": [("g", 0.242, (0.18, 0.30), 0.035, None), ("g", 0.20, (0.15, 0.35), 0.07, None)],
        "proton": [("g", 0.85, (0.75, 1.0), 0.07, None), ("g", 0.94, (0.7, 1.1), 0.13, None),
                   ("g", 0.86, (0.6, 1.2), 0.33, None)],
    },
    # Gaussian cores with exponential tails pointing INTO the kaon window,
    # the analogue of fit_kaon_peak.py's local exponential tails.
    "exptail": {
        "light": [("g", 0.015, (-0.03, 0.05), 0.016, None), ("g", 0.006, (-0.05, 0.08), 0.034, None),
                  ("er", 0.01, (-0.05, 0.08), 0.03, 0.05)],
        "kaon": [("g", 0.242, (0.18, 0.30), 0.035, None)],
        "proton": [("g", 0.85, (0.75, 1.0), 0.07, None), ("g", 0.94, (0.7, 1.1), 0.13, None),
                   ("el", 0.90, (0.7, 1.1), 0.08, 0.15)],
    },
}


def stage1_params(spec):
    P = Params()
    for s, v in zip(SPECIES + ("flat",), (1e5, 3e3, 8e4, 6e2)):
        P.add(f"logY_{s}", np.log(v), 0.0, 15.0)
    for s in SPECIES:
        for i, (kind, mu, mb, sig, tail) in enumerate(spec[s]):
            P.add(f"{s}{i}_mu", mu, *mb)
            P.add(f"{s}{i}_logsig", np.log(sig), np.log(0.005), np.log(0.6))
            if tail is not None:
                P.add(f"{s}{i}_logtail", np.log(tail), np.log(0.005), np.log(1.0))
            if i:
                P.add(f"{s}{i}_logw", np.log(0.3), -8.0, 3.0)
    return P


def stage1_shapes(spec, p):
    """Per-species normalised Shapes from a stage-1 parameter dict."""
    shapes = {}
    for s in SPECIES:
        kind, mu, sig, lam, w = [], [], [], [], []
        for i, (k, *_rest) in enumerate(spec[s]):
            kind.append(KIND[k]); mu.append(p[f"{s}{i}_mu"]); sig.append(np.exp(p[f"{s}{i}_logsig"]))
            lam.append(1.0 / np.exp(p[f"{s}{i}_logtail"]) if k != "g" else 1.0)
            w.append(np.exp(p[f"{s}{i}_logw"]) if i else 1.0)
        shapes[s] = Shape(kind, mu, sig, lam, w)
    return shapes


def stage1_expect(spec, p, edges, lo, hi):
    shapes = stage1_shapes(spec, p)
    comps = {s: np.exp(p[f"logY_{s}"]) * shapes[s].bins(edges) for s in SPECIES}
    comps["flat"] = np.exp(p["logY_flat"]) * np.diff(edges) / (hi - lo)
    return comps, shapes


def fit_stage1(spec, counts, edges, lo, hi, n_starts, seed):
    P = stage1_params(spec)
    nll = lambda t: poisson_nll(sum(stage1_expect(spec, P.unpack(t), edges, lo, hi)[0].values()), counts)
    theta, fun = P.fit(nll, n_starts, seed)
    p = P.unpack(theta)
    comps, shapes = stage1_expect(spec, p, edges, lo, hi)
    win = {s: np.exp(p[f"logY_{s}"]) * shapes[s].window() for s in SPECIES}
    win["flat"] = np.exp(p["logY_flat"]) * (WINDOW[1] - WINDOW[0]) / (hi - lo)
    return dict(p=p, nll=fun, comps=comps, shapes=shapes, window=win,
                chi2_dof=chi2_dof(sum(comps.values()), counts, len(theta)))


# --------------------------------------------------------------------------- stage 2

# Kernel variants: which smearing freedoms stage 2 has.
#   n_scale      Gaussians in log s (0 = no momentum smearing); each has its own
#                bias and width, so the mixture can be skewed.
#   tof          Gaussians in the additive (TOF) term.
#   per_species  give each species its own scale kernel, e.g. because picky and
#                non-picky momentum spectra differ by species.
KERNELS = {
    "scale2+tof2":    dict(n_scale=2, tof=2, per_species=False),
    "scale1+tof2":    dict(n_scale=1, tof=2, per_species=False),
    "scale2+tof1":    dict(n_scale=2, tof=1, per_species=False),
    "scale3+tof2":    dict(n_scale=3, tof=2, per_species=False),
    "perspecies2":    dict(n_scale=2, tof=2, per_species=True),
    "perspecies3":    dict(n_scale=3, tof=2, per_species=True),
    "tof_only":       dict(n_scale=0, tof=2, per_species=False),  # expected to fail
}
SCALE_INIT = [(0.03, 0.05), (0.03, 0.25), (-0.3, 0.3)]  # (bias, width) per scale Gaussian


def stage2_params(kern, kaon_shift=False, bg=True):
    P = Params()
    for s, v in zip(SPECIES, (2e5, 5e3, 1.3e5)):
        P.add(f"logY_{s}", np.log(v), 0.0, 15.0)
    P.add("logY_flat", np.log(2e3), 0.0, 15.0)
    if bg:
        # Mismatched reconstructions. Kept genuinely broad (sigma >= 0.4 GeV^2) so
        # it cannot impersonate a smeared kaon peak.
        P.add("logY_bg", np.log(1e4), 0.0, 15.0)
        P.add("bg_mu", 0.5, -0.5, 2.5)
        P.add("bg_logsig", np.log(0.8), np.log(0.4), np.log(5.0))
    groups = SPECIES if kern["per_species"] else ("all",)
    for g in groups:
        for j in range(kern["n_scale"]):
            b, w = SCALE_INIT[j]
            P.add(f"s{j}_bias_{g}", b, -1.0, 1.0)
            P.add(f"s{j}_logsig_{g}", np.log(w), np.log(0.003), np.log(1.5))
            if j:
                P.add(f"s{j}_logw_{g}", -1.5, -8.0, 3.0)
    P.add("a0", 0.0, -0.05, 0.05)
    P.add("a0_logtau", np.log(0.01), np.log(0.001), np.log(0.2))
    if kern["tof"] == 2:
        P.add("a1", 0.05, -0.1, 0.3)
        P.add("a1_logtau", np.log(0.04), np.log(0.003), np.log(0.4))
        P.add("a1_logitf", -2.0, -8.0, 3.0)
    if kaon_shift:
        P.add("kaon_shift", 1.0, 0.6, 1.4)
    return P


def _scale_nodes(p, g, kern):
    if kern["n_scale"] == 0:
        return np.array([1.0]), np.array([1.0])
    us, ws = [], []
    wts = np.array([1.0] + [np.exp(p[f"s{j}_logw_{g}"]) for j in range(1, kern["n_scale"])])
    wts /= wts.sum()
    for j in range(kern["n_scale"]):
        us.append(p[f"s{j}_bias_{g}"] + SQRT2 * np.exp(p[f"s{j}_logsig_{g}"]) * GH_T)
        ws.append(wts[j] * GH_W)
    return np.exp(np.concatenate(us)), np.concatenate(ws)


def smeared_shapes(shapes, p, kern):
    adds = [(p["a0"], np.exp(p["a0_logtau"]), 1.0)]
    if kern["tof"] == 2:
        f = 1.0 / (1.0 + np.exp(-p["a1_logitf"]))
        adds = [(p["a0"], np.exp(p["a0_logtau"]), 1 - f), (p["a1"], np.exp(p["a1_logtau"]), f)]
    out = {}
    for s in SPECIES:
        base = shapes[s]
        if s == "kaon" and "kaon_shift" in p:
            base = Shape(base.kind, base.mu * p["kaon_shift"], base.sig, base.lam, base.w)
        g = s if kern["per_species"] else "all"
        out[s] = base.smear(*_scale_nodes(p, g, kern), adds)
    return out


def stage2_expect(shapes, p, kern, edges, lo, hi):
    sm = smeared_shapes(shapes, p, kern)
    comps = {s: np.exp(p[f"logY_{s}"]) * sm[s].bins(edges) for s in SPECIES}
    comps["flat"] = np.exp(p["logY_flat"]) * np.diff(edges) / (hi - lo)
    if "logY_bg" in p:
        bg = Shape([0], [p["bg_mu"]], [np.exp(p["bg_logsig"])], [1.0], [1.0])
        comps["bg"] = np.exp(p["logY_bg"]) * bg.bins(edges)
    return comps, sm


def stage2_window(sm, p, lo, hi):
    win = {s: np.exp(p[f"logY_{s}"]) * sm[s].window() for s in SPECIES}
    other = np.exp(p["logY_flat"]) * (WINDOW[1] - WINDOW[0]) / (hi - lo)
    if "logY_bg" in p:
        other += np.exp(p["logY_bg"]) * Shape([0], [p["bg_mu"]], [np.exp(p["bg_logsig"])], [1.0], [1.0]).window()
    win["unassigned"] = other
    return win


def fit_stage2(shapes, counts, edges, lo, hi, kern, n_starts, seed, kaon_shift=False,
               bg=True, fixed=None):
    P = stage2_params(kern, kaon_shift=kaon_shift, bg=bg)

    def nll(t):
        return poisson_nll(sum(stage2_expect(shapes, P.unpack(t), kern, edges, lo, hi)[0].values()), counts)

    theta, fun = P.fit(nll, n_starts, seed, fixed=fixed)
    p = P.unpack(theta)
    comps, sm = stage2_expect(shapes, p, kern, edges, lo, hi)
    n_free = len(theta) - len(fixed or {})
    return dict(p=p, nll=fun, comps=comps, window=stage2_window(sm, p, lo, hi),
                chi2_dof=chi2_dof(sum(comps.values()), counts, n_free))


def fractions(win):
    tot = sum(win.values())
    return {k: float(v / tot) for k, v in win.items()}


# --------------------------------------------------------------------------- closure

CLOSURE_TRUTHS = {
    # log k ~ Student-t (heavy tails, not a Gaussian mixture), one-sided
    # exponential TOF offsets, 4% uniform junk.
    "t3+exp": dict(df=3, scale=0.06, bias=0.04, tof_frac=0.15, tof_mean=0.04, junk=0.04),
    # narrower core, fatter tails, no TOF offsets, more junk.
    "t2+junk": dict(df=2, scale=0.035, bias=0.02, tof_frac=0.0, tof_mean=0.0, junk=0.08),
}


def closure_test(x_picky, s1, spec, edges, lo, hi, kern, truth, n_starts, seed, copies=2):
    """Smear picky events with a known out-of-family kernel and refit (stage 2)."""
    rng = np.random.default_rng(seed)
    # Soft species labels from the stage-1 posterior in each event's bin.
    idx = np.clip(np.digitize(x_picky, edges) - 1, 0, len(edges) - 2)
    inside = (x_picky >= lo) & (x_picky < hi)
    tot = sum(s1["comps"].values())
    post = {s: (s1["comps"][s] / tot)[idx] * inside for s in list(SPECIES) + ["flat"]}

    xs, ws = [], {s: [] for s in post}
    for _ in range(copies):
        k = np.exp(truth["bias"] + truth["scale"] * rng.standard_t(truth["df"], len(x_picky)))
        tof = np.where(rng.random(len(x_picky)) < truth["tof_frac"],
                       rng.exponential(truth["tof_mean"] or 1e-9, len(x_picky)), 0.0)
        xs.append(k * x_picky + tof)
        for s in post:
            ws[s].append(post[s])
    xs = np.concatenate(xs)
    ws = {s: np.concatenate(v) for s, v in ws.items()}
    n_junk = int(truth["junk"] * len(xs))
    junk = rng.uniform(lo, hi, n_junk)

    in_win = (xs > WINDOW[0]) & (xs < WINDOW[1])
    true_win = {s: float(ws[s][in_win].sum()) for s in SPECIES}
    true_win["unassigned"] = float(ws["flat"][in_win].sum() + ((junk > WINDOW[0]) & (junk < WINDOW[1])).sum())
    counts, _ = np.histogram(np.concatenate([xs, junk]), edges)
    fit = fit_stage2(s1["shapes"], counts, edges, lo, hi, kern, n_starts, seed)
    return dict(true=fractions(true_win), fitted=fractions(fit["window"]), chi2_dof=fit["chi2_dof"])


# --------------------------------------------------------------------------- plots

def m_of_x(x):
    return np.sign(x) * np.sqrt(np.abs(x)) * 1e3


def plot_fit(counts, edges, fit, frac, out_dir):
    ctr = 0.5 * (edges[1:] + edges[:-1])
    s = apply_style(DOUBLE_COL / 2)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, DOUBLE_COL / 2.5))
    colours = {"light": COLOURS["muon"], "kaon": COLOURS["kaon"], "proton": COLOURS["proton"],
               "bg": "0.55", "flat": "0.75"}
    labels = {"light": r"$\pi/\mu/e$ (smeared)", "kaon": r"$K^+$ (smeared)",
              "proton": "proton (smeared)", "bg": "broad background", "flat": "flat"}
    total = sum(fit["comps"].values())
    for ax, (a, b), logy in [(axes[0], (edges[0], edges[-1]), True), (axes[1], (0.06, 0.55), False)]:
        sel = (ctr > a) & (ctr < b)
        ax.axvspan(*WINDOW, color=COLOURS["kaon"], alpha=0.10, lw=0, zorder=0)
        ax.errorbar(ctr[sel], counts[sel], yerr=np.sqrt(counts[sel]), fmt="o", ms=1.2 * s,
                    color="0.3", lw=0.5 * s, label="non-picky data", zorder=4)
        ax.plot(ctr[sel], total[sel], "-", color="k", lw=1.0 * s, label="total fit", zorder=3)
        for c, y in fit["comps"].items():
            ax.plot(ctr[sel], y[sel], "--", color=colours[c], lw=0.9 * s, label=labels[c])
        ax.set_xlim(a, b)
        ax.set_xlabel(r"signed $m^2$ [GeV$^2/c^4$]")
        if logy:
            ax.set_yscale("log"); ax.set_ylim(0.5, counts.max() * 3)
        else:
            ax.set_ylim(0, counts[sel].max() * 1.35)
        top = ax.secondary_xaxis("top", functions=(m_of_x, lambda m: np.sign(m) * (m / 1e3) ** 2))
        top.set_xlabel("signed mass [MeV]", fontsize=7 * s)
    axes[0].set_ylabel(f"Counts / {edges[1] - edges[0]:.3f} GeV$^2$")
    axes[0].legend(fontsize=5.8 * s, loc="upper right")
    axes[1].text(0.97, 0.95,
                 f"non-picky, {WINDOW_MEV[0]:.0f}-{WINDOW_MEV[1]:.0f} MeV:\n"
                 f"  $K^+$      {frac['kaon']:5.1%}\n  proton     {frac['proton']:5.1%}\n"
                 f"  $\\pi/\\mu/e$    {frac['light']:5.1%}\n  unassigned {frac['unassigned']:5.1%}",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=6 * s, family="monospace",
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.75", lw=0.6 * s))
    fig.tight_layout()
    savefig(fig, out_dir, "nonpicky_fit")


def plot_variants(rows, nominal, out_dir):
    s = apply_style(SINGLE_COL)
    fig, ax = plt.subplots(figsize=(SINGLE_COL * 1.3, 0.22 * len(rows) + 0.8))
    for i, r in enumerate(rows):
        faded = not r["accepted"]
        ax.plot(r["picky_kaon"], i, "s", color=COLOURS["proton"], ms=3.5 * s, alpha=0.35 if faded else 1)
        ax.plot(r["nonpicky_kaon"], i, "o", color=COLOURS["kaon"], ms=3.5 * s, alpha=0.35 if faded else 1)
        if r["name"] == nominal and "stat68" in r:
            ax.hlines(i, *r["stat68"], color=COLOURS["kaon"], lw=1.2 * s)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r["name"] + ("" if r["accepted"] else " (rejected)") for r in rows], fontsize=6 * s)
    ax.set_xlabel(f"$K^+$ fraction of the {WINDOW_MEV[0]:.0f}-{WINDOW_MEV[1]:.0f} MeV window")
    ax.plot([], [], "s", color=COLOURS["proton"], label="picky (stage 1)")
    ax.plot([], [], "o", color=COLOURS["kaon"], label="non-picky (stage 2)")
    ax.legend(fontsize=6 * s, loc="best")
    ax.set_xlim(0, 1); ax.invert_yaxis(); ax.grid(axis="x", lw=0.3, alpha=0.5)
    fig.tight_layout()
    savefig(fig, out_dir, "nonpicky_variants")


# --------------------------------------------------------------------------- main

def _boot_one(spec, c1, c2, edges, lo, hi, kern, seed):
    """One parametric-bootstrap replica through both stages."""
    b1 = fit_stage1(spec, c1, edges, lo, hi, 1, seed)
    b2 = fit_stage2(b1["shapes"], c2, edges, lo, hi, kern, 1, seed)
    return [fractions(b1["window"])["kaon"], fractions(b2["window"])["kaon"]]


def histogram(x, lo, hi, bw):
    edges = np.arange(lo, hi + bw / 2, bw)
    return np.histogram(x, edges)[0], edges


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--picky-csv", default=DEFAULT_PICKY)
    ap.add_argument("--range", type=float, nargs=2, default=[-0.30, 2.60], metavar=("LO", "HI"),
                    help="fit range in signed m^2 [GeV^2]")
    ap.add_argument("--bin-width", type=float, default=0.005)
    ap.add_argument("--n-starts", type=int, default=4)
    ap.add_argument("--n-boot", type=int, default=20)
    ap.add_argument("--chi2-rel", type=float, default=1.5,
                    help="variants with stage-2 chi2/dof above this multiple of the best are shown "
                         "but excluded from the systematic range")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--no-closure", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    out_dir = Path(args.out_dir) if args.out_dir else PROJECT_ROOT / "figs" / "beamline_mass_fit"

    df = pd.read_csv(args.picky_csv).drop_duplicates(["run", "subrun", "event"])
    m = df["beamline_mass"].to_numpy()
    x = np.sign(m) * m ** 2 / 1e6
    xp, xn = x[df["p"].to_numpy() == 1], x[df["p"].to_numpy() == 0]
    lo, hi = args.range
    cp, edges = histogram(xp, lo, hi, args.bin_width)
    cn, _ = histogram(xn, lo, hi, args.bin_width)
    in_w = lambda c: int(c[(edges[:-1] >= WINDOW[0]) & (edges[1:] <= WINDOW[1])].sum())
    print(f"picky {cp.sum()} / non-picky {cn.sum()} events in x=[{lo},{hi}] GeV^2; "
          f"in window: {in_w(cp)} / {in_w(cn)}")

    # ---- variant grid, in parallel: stage 1 per (template, range), then stage 2
    grid = [(t, k, (lo, hi)) for t in TEMPLATES for k in KERNELS]
    grid += [(t, k, (-0.2, 2.0)) for t in TEMPLATES for k in ("scale2+tof2", "perspecies2")]
    hists = {}
    for _, _, (a, b) in grid:
        if (a, b) not in hists:
            ca, ea = histogram(xp, a, b, args.bin_width)
            hists[(a, b)] = (ca, histogram(xn, a, b, args.bin_width)[0], ea)
    with ProcessPoolExecutor(args.workers) as pool:
        keys = sorted({(t, a, b) for t, _, (a, b) in grid})
        futs = {k: pool.submit(fit_stage1, TEMPLATES[k[0]], hists[k[1:]][0], hists[k[1:]][2],
                               k[1], k[2], args.n_starts, args.seed) for k in keys}
        stage1 = {k: f.result() for k, f in futs.items()}
        print("stage 1 done: " + ", ".join(f"{k[0]} [{k[1]},{k[2]}] chi2/dof {v['chi2_dof']:.2f} "
                                           f"K {fractions(v['window'])['kaon']:.1%}" for k, v in stage1.items()),
              flush=True)
        futs = [(t, kn, (a, b), pool.submit(fit_stage2, stage1[(t, a, b)]["shapes"], hists[(a, b)][1],
                                             hists[(a, b)][2], a, b, KERNELS[kn], args.n_starts, args.seed))
                for t, kn, (a, b) in grid]
        rows, fits = [], {}
        for tname, kname, (a, b), f in futs:
            s1, s2 = stage1[(tname, a, b)], f.result()
            name = f"{tname} / {kname}" + ("" if (a, b) == (lo, hi) else f" / x in [{a},{b}]")
            fp, fn = fractions(s1["window"]), fractions(s2["window"])
            rows.append(dict(name=name, template=tname, kernel=kname, range=[a, b],
                             picky_chi2_dof=s1["chi2_dof"], chi2_dof=s2["chi2_dof"], nll=s2["nll"],
                             picky_kaon=fp["kaon"], nonpicky_kaon=fn["kaon"],
                             picky_fractions=fp, nonpicky_fractions=fn,
                             nonpicky_kaon_in_window=float(s2["window"]["kaon"]),
                             # K among events the fit assigns to a species, i.e. treating the
                             # broad background as removable: an upper-side reading.
                             nonpicky_kaon_of_assigned=fn["kaon"] / (1 - fn["unassigned"])))
            fits[name] = (s1, s2, hists[(a, b)][1], hists[(a, b)][2], a, b)
            print(f"  {name:40s} picky chi2/dof {s1['chi2_dof']:.2f} K {fp['kaon']:.1%} | "
                  f"non-picky chi2/dof {s2['chi2_dof']:.2f} K {fn['kaon']:.1%} p {fn['proton']:.1%} "
                  f"L {fn['light']:.1%} other {fn['unassigned']:.1%}", flush=True)

        # With ~4e5 events any residual shape mismatch is significant, so an absolute
        # chi2 cut rejects everything; accept variants within a factor of the best.
        best_chi2 = min(r["chi2_dof"] for r in rows)
        for r in rows:
            r["accepted"] = bool(r["chi2_dof"] <= args.chi2_rel * best_chi2)
        # Nominal: the best-fitting full-range variant. Per-species kernels win by a
        # wide margin (chi2/dof ~1.3 vs >=2.4 shared): picky and non-picky momentum
        # spectra differ by species, so one shared smearing cannot describe all three.
        cands = [r for r in rows if r["range"] == [lo, hi]]
        nominal = min(cands, key=lambda r: r["chi2_dof"])
        s1, s2, na, ea, a, b = fits[nominal["name"]]
        tname, kname = nominal["template"], nominal["kernel"]
        kern = KERNELS[kname]
        print(f"\nnominal: {nominal['name']}  (best chi2/dof {best_chi2:.2f}; "
              f"accepting <= {args.chi2_rel} x best)", flush=True)

        # ---- kaon evidence, kaon position, bootstrap and closure, all in parallel
        null_f = pool.submit(fit_stage2, s1["shapes"], na, ea, a, b, kern, args.n_starts, args.seed,
                             fixed={"logY_kaon": 0.0})
        shift_f = pool.submit(fit_stage2, s1["shapes"], na, ea, a, b, kern, args.n_starts, args.seed,
                              kaon_shift=True)
        rng = np.random.default_rng(args.seed + 1)
        mod1 = sum(s1["comps"].values()); mod2 = sum(s2["comps"].values())
        boot_f = [pool.submit(_boot_one, TEMPLATES[tname], rng.poisson(mod1), rng.poisson(mod2),
                              ea, a, b, kern, args.seed) for _ in range(args.n_boot)]
        clos_f = {} if args.no_closure else {
            c: pool.submit(closure_test, xp, s1, TEMPLATES[tname], ea, a, b, kern, truth,
                           args.n_starts, args.seed) for c, truth in CLOSURE_TRUTHS.items()}

        null, shift = null_f.result(), shift_f.result()
        d2nll = 2 * (null["nll"] - s2["nll"])
        ks = shift["p"]["kaon_shift"]
        kcore = s1["shapes"]["kaon"].mu[0]
        k_mass = float(np.sqrt(ks * kcore * np.exp(shift["p"].get("s0_bias_all", shift["p"].get("s0_bias_kaon", 0.0)))) * 1e3)
        print(f"  kaon null: 2*dNLL = {d2nll:.1f}   free kaon position: shift {ks:.3f} -> "
              f"smeared peak ~{k_mass:.0f} MeV (picky core {np.sqrt(kcore) * 1e3:.0f}, "
              f"PDG {PDG_MASS['kaon']:.0f}); K fraction with free position "
              f"{fractions(shift['window'])['kaon']:.1%}", flush=True)
        boot = np.array([f.result() for f in boot_f])
        if len(boot):
            nominal["stat68"] = [float(np.percentile(boot[:, 1], 16)), float(np.percentile(boot[:, 1], 84))]
            nominal["picky_stat68"] = [float(np.percentile(boot[:, 0], 16)), float(np.percentile(boot[:, 0], 84))]
        closure = {c: f.result() for c, f in clos_f.items()}
        for c, r in closure.items():
            print(f"  closure {c:8s}: true K {r['true']['kaon']:.1%} fitted K {r['fitted']['kaon']:.1%} "
                  f"(chi2/dof {r['chi2_dof']:.2f}); true p {r['true']['proton']:.1%} "
                  f"fitted {r['fitted']['proton']:.1%}")

    # ---- summary
    acc = [r for r in rows if r["accepted"]]
    syst = [min(r["nonpicky_kaon"] for r in acc), max(r["nonpicky_kaon"] for r in acc)] if acc else None
    syst_p = [min(r["picky_kaon"] for r in acc), max(r["picky_kaon"] for r in acc)] if acc else None
    print(f"\nRESULT  non-picky K+ fraction of the {WINDOW_MEV[0]:.0f}-{WINDOW_MEV[1]:.0f} MeV window: "
          f"{nominal['nonpicky_kaon']:.1%} (nominal)")
    if "stat68" in nominal:
        print(f"        stat 68%: {nominal['stat68'][0]:.1%} - {nominal['stat68'][1]:.1%}")
    if syst:
        print(f"        range over {len(acc)} accepted variants: {syst[0]:.1%} - {syst[1]:.1%}")
        print(f"        picky, same variants: {syst_p[0]:.1%} - {syst_p[1]:.1%}")

    plot_fit(na, ea, s2, fractions(s2["window"]), out_dir)
    plot_variants(rows, nominal["name"], out_dir)
    with open(out_dir / "nonpicky_metrics.json", "w") as fh:
        json.dump({
            "window_mev": list(WINDOW_MEV), "window_x_gev2": list(WINDOW),
            "n_picky_in_window": in_w(cp), "n_nonpicky_in_window": in_w(cn),
            "nominal": nominal, "variant_range_nonpicky_kaon": syst, "variant_range_picky_kaon": syst_p,
            "kaon_null_2dnll": d2nll,
            "kaon_free_shift": {"shift": ks, "smeared_peak_mev": k_mass,
                                "kaon_fraction": fractions(shift["window"])["kaon"]},
            "nominal_params": {k: float(v) for k, v in s2["p"].items()},
            "closure": closure, "variants": rows,
        }, fh, indent=2)
    print(f"  saved {out_dir / 'nonpicky_metrics.json'}")


if __name__ == "__main__":
    main()
