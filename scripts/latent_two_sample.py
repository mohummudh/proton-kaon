#!/usr/bin/env python3
"""
scripts/latent_two_sample.py

Two-sample tests on VAE latents: is the distribution of train latents the same
as the distribution of val latents?

Because train/val here is a *random split of one sample*, the null hypothesis is
true by construction unless the encoder treats images it was trained on
differently from images it was not.  A significant result is therefore evidence
of memorisation/overfitting in the encoder, not of covariate shift.  At n in the
thousands statistical significance is nearly free, so every test below is
reported with an effect size and a same-distribution noise floor next to it.

Four tests, at two levels:

  marginal (8 x 1-D)   KS per latent dim + Holm correction; 1-D Wasserstein in
                       units of sigma as the effect size.  Interpretable: names
                       the culprit dimension.  Blind to correlation changes.
  joint (8-D)          energy distance (Szekely-Rizzo), permutation-calibrated.
                       Zero iff the distributions match, unbiased under the null.
  joint (8-D)          Wasserstein W1/W2, exact via optimal assignment, plus a
                       sliced variant.  A distance, not a test — reported with a
                       permutation null because the empirical estimator in 8-D
                       is dominated by finite-sample bias (see docs in report).
  joint, interpretable C2ST: cross-validated AUC of a classifier trained to tell
                       train latents from val latents.  0.5 => indistinguishable.

Every joint statistic is calibrated by a permutation null: pool the two groups,
reshuffle the group labels, recompute.  Under the null the labels are arbitrary,
so the reshuffled values are draws from exactly the distribution the statistic
would have had if the two groups matched.  That gives an exact p-value and, more
usefully, a noise floor to compare the observed number against.

Comparisons: `combined` pools all species, `combined_matched` does the same after
resampling the val pool to the training species mixture (a balanced split leaves
the two pools with different mixtures, and species are separable in latent space
by construction, so the unmatched version is confounded), plus `proton`, `kaon`
and `muon` individually.

Usage
-----
    # everything, all five comparisons, on the run93 model  (~20 min)
    python scripts/latent_two_sample.py --config configs/run_0093_*.yaml

    # quick pass: skips the C2ST permutation null, which dominates runtime
    python scripts/latent_two_sample.py --config configs/run_0093_*.yaml \
        --c2st-perm 0 --label quick

    # negative control: relabel train/val at random, so the answer is known-null
    python scripts/latent_two_sample.py --config configs/run_0093_*.yaml \
        --null-control --label control

    # robustness: latent space in its native (unstandardised) metric
    python scripts/latent_two_sample.py --config configs/run_0093_*.yaml \
        --standardize none --label rawscale

    # is significance just being bought by n?  run at 25% of the data
    python scripts/latent_two_sample.py --config configs/run_0093_*.yaml \
        --subsample-frac 0.25 --label quarter

Outputs (all under figs/<model_name>/two-sample/<label>/):
    results.json     full nested results incl. permutation null summaries
    summary.csv      one row per comparison — the joint tests
    marginals.csv    one row per (comparison, latent dim) — the KS/Wasserstein table
Figures and report.md are rendered by scripts/plot_two_sample.py, which this
script calls at the end unless --no-plots.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.train.naming import model_name as build_model_name  # noqa: E402

COMPARISONS = ["combined", "combined_matched", "proton", "kaon", "muon"]
TESTS = ["ks", "energy", "wasserstein", "c2st"]
ECDF_GRID = 512   # points at which the stored ECDF pair is sampled, for replotting


# ── multiple-comparison correction ─────────────────────────────────────────────

def holm(pvals: np.ndarray) -> np.ndarray:
    """Holm step-down adjusted p-values, controlling the family-wise error rate.

    Run 8 tests at alpha=0.05 with nothing wrong and the chance of at least one
    false alarm is 1 - 0.95^8 ~ 34%.  Holm sorts the p-values ascending and
    compares the k-th smallest against alpha/(m-k+1), which is uniformly more
    powerful than Bonferroni's alpha/m for all of them and needs no assumptions.
    Returned values are directly comparable to alpha.
    """
    p = np.asarray(pvals, dtype=float)
    m = p.size
    order = np.argsort(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * p[i])   # enforce monotonicity
        adj[i] = min(running, 1.0)
    return adj


# ── marginal statistics (1-D, one latent dimension at a time) ──────────────────

def ks_and_w1_from_sorted(labels_sorted: np.ndarray, values_sorted: np.ndarray,
                          n_a: int, n_b: int, step_mask: np.ndarray) -> tuple:
    """Both marginal statistics in one O(N) pass over a pre-sorted axis.

    `values_sorted` is one latent dimension of the pooled sample, sorted
    ascending; `labels_sorted` is the boolean group-A membership in that same
    order.  Cumulative counts of each label give the two ECDFs directly:

        D  = sup_x |F_a(x) - F_b(x)|         <- Kolmogorov-Smirnov: largest gap
        W1 = integral |F_a(x) - F_b(x)| dx   <- Wasserstein: area between them

    Same two staircases, sup-norm versus L1-norm.  Sorting once and permuting
    only the labels is what makes the permutation null affordable.
    `step_mask` marks the last element of each run of tied values, so D is the
    sup over distinct support points rather than mid-tie.
    """
    ca = np.cumsum(labels_sorted) / n_a
    cb = np.cumsum(~labels_sorted) / n_b
    diff = np.abs(ca - cb)
    d_stat = diff[step_mask].max()
    w1 = float(np.dot(diff[:-1], np.diff(values_sorted)))
    return float(d_stat), w1


def marginal_tests(A: np.ndarray, B: np.ndarray, n_perm: int, rng) -> dict:
    """Per-dimension KS + Wasserstein, with a permutation null for both.

    The permutation shuffles *rows*, not each dimension independently, so the
    null for max-D across dimensions correctly accounts for the correlation
    between latent dims — a global test that needs no multiplicity correction,
    reported alongside the per-dim Holm-adjusted p-values.
    """
    n_a, n_b, n_dim = len(A), len(B), A.shape[1]
    Z = np.vstack([A, B])
    labels = np.zeros(n_a + n_b, dtype=bool)
    labels[:n_a] = True

    # pooled sd per dim: permutation-invariant, so dividing by it keeps the
    # permutation test exact while making the effect size scale-free.
    sigma = Z.std(axis=0, ddof=1)
    sigma[sigma == 0] = np.nan

    orders, sorted_vals, step_masks = [], [], []
    for d in range(n_dim):
        o = np.argsort(Z[:, d], kind="stable")
        v = Z[o, d]
        orders.append(o)
        sorted_vals.append(v)
        step_masks.append(np.append(np.diff(v) != 0, True))

    obs_d = np.empty(n_dim)
    obs_w = np.empty(n_dim)
    obs_p = np.empty(n_dim)
    for d in range(n_dim):
        obs_d[d], obs_w[d] = ks_and_w1_from_sorted(
            labels[orders[d]], sorted_vals[d], n_a, n_b, step_masks[d])
        # scipy for the reported p-value: the standard exact/asymptotic KS
        # p-value a reviewer expects, rather than one of our own construction.
        obs_p[d] = ks_2samp(A[:, d], B[:, d]).pvalue

    null_d = np.empty((n_perm, n_dim))
    null_w = np.empty((n_perm, n_dim))
    for k in range(n_perm):
        perm = rng.permutation(labels)
        for d in range(n_dim):
            null_d[k, d], null_w[k, d] = ks_and_w1_from_sorted(
                perm[orders[d]], sorted_vals[d], n_a, n_b, step_masks[d])

    w_sigma = obs_w / sigma
    null_w_sigma = null_w / sigma
    # per-dim permutation p-values (upper tail; +1 counts the observed value,
    # which is what makes the p-value exactly valid rather than anti-conservative)
    p_perm_d = (1 + (null_d >= obs_d).sum(axis=0)) / (1 + n_perm)
    p_perm_w = (1 + (null_w >= obs_w).sum(axis=0)) / (1 + n_perm)

    obs_max_d = float(obs_d.max())
    null_max_d = null_d.max(axis=1)
    p_max_d = float((1 + (null_max_d >= obs_max_d).sum()) / (1 + n_perm))

    mean_shift = (B.mean(axis=0) - A.mean(axis=0)) / sigma
    std_ratio = B.std(axis=0, ddof=1) / A.std(axis=0, ddof=1)

    # Compact ECDF pair per dimension, so the KS figure can be redrawn from
    # results.json alone — no need to re-mount the inference drive.
    grid = np.empty((n_dim, ECDF_GRID))
    ecdf_a = np.empty((n_dim, ECDF_GRID))
    ecdf_b = np.empty((n_dim, ECDF_GRID))
    for d in range(n_dim):
        lo, hi = np.percentile(Z[:, d], [0.2, 99.8])
        grid[d] = np.linspace(lo, hi, ECDF_GRID)
        ecdf_a[d] = np.searchsorted(np.sort(A[:, d]), grid[d], side="right") / n_a
        ecdf_b[d] = np.searchsorted(np.sort(B[:, d]), grid[d], side="right") / n_b

    return {
        "n_a": int(n_a), "n_b": int(n_b), "n_dim": int(n_dim), "n_perm": int(n_perm),
        "pooled_sigma": sigma.tolist(),
        "ks_D": obs_d.tolist(),
        "ks_p_raw": obs_p.tolist(),
        "ks_p_holm": holm(obs_p).tolist(),
        "ks_p_perm": p_perm_d.tolist(),
        "ks_D_null_p95": np.percentile(null_d, 95, axis=0).tolist(),
        "w1_sigma": w_sigma.tolist(),
        "w1_sigma_null_mean": null_w_sigma.mean(axis=0).tolist(),
        "w1_sigma_null_p95": np.percentile(null_w_sigma, 95, axis=0).tolist(),
        "w1_p_perm": p_perm_w.tolist(),
        "mean_shift_sigma": mean_shift.tolist(),
        "std_ratio": std_ratio.tolist(),
        "global_max_D": obs_max_d,
        "global_max_D_p_perm": p_max_d,
        "global_max_D_null_p95": float(np.percentile(null_max_d, 95)),
        "n_holm_significant": int((holm(obs_p) < 0.05).sum()),
        "ecdf_grid": grid.tolist(),
        "ecdf_train": ecdf_a.tolist(),
        "ecdf_val": ecdf_b.tolist(),
    }


# ── joint test 1: energy distance ──────────────────────────────────────────────

def combine_repeats(values, p_values, nulls) -> dict:
    """Aggregate several independently-valid permutation tests into one result.

    Every test here has to subsample (an O(n^2) or O(n^3) kernel over thousands
    of points), and a single subsample draw is not reproducible: on this data the
    exact-Wasserstein p-value moves between 0.01 and 0.92 from draw to draw.  So
    each repeat is run as a *complete* permutation test on its own draw — own
    subsample, own null, own exactly-valid p-value — and the repeats are then
    combined.

    Combining needs care.  The repeats are not independent: they resample the
    same underlying two groups, so pooling their nulls and treating the mean as
    if it had sd/sqrt(R) is anti-conservative (measured at 11.5% false positives
    against a nominal 5%).  Twice the median p-value, however, is a valid
    p-value under *arbitrary* dependence (Vovk and Wang's median merging rule),
    which is exactly the guarantee needed here.  It costs a little power and
    assumes nothing.
    """
    flat = np.concatenate([np.asarray(n).ravel() for n in nulls])
    value = float(np.mean(values))
    sd = float(flat.std(ddof=1))
    return {
        "value": value,
        "value_repeats": [float(v) for v in values],
        "value_repeat_sd": float(np.std(values, ddof=1)) if len(values) > 1 else np.nan,
        "p_values": [float(p) for p in p_values],
        "p_value": (float(min(1.0, 2.0 * np.median(p_values))) if len(p_values) > 1
                    else float(p_values[0])),
        "p_combination": "2 x median (valid under arbitrary dependence)" if len(p_values) > 1
                         else "single permutation test",
        "null_mean": float(flat.mean()),
        "null_sd": sd,
        "null_p95": float(np.percentile(flat, 95)),
        "z_score": float((value - flat.mean()) / sd) if sd > 0 else np.nan,
        "null_samples": flat[:2000].tolist(),
    }


def _energy_batch(D: np.ndarray, rowsum: np.ndarray, total: float,
                  X: np.ndarray, n: int, m: int) -> tuple:
    """Energy distance for many group assignments at once.

        E = 2 E||X - Y|| - E||X - X'|| - E||Y - Y'||

    i.e. average cross-group distance (doubled) minus the two average
    within-group distances.  If the clouds sit on top of each other a cross pair
    looks like a within pair, the terms cancel, and E -> 0.  Szekely-Rizzo:
    E >= 0 with equality iff the distributions are identical, in any dimension —
    so unlike the marginals it has no blind spot.

    `X` holds one 0/1 group-A indicator column per assignment.  Every group sum
    needed is a quadratic form in those indicators, so the whole permutation null
    collapses to a single matrix product: with S the total of all pairwise
    distances, S_AA = x'Dx and S_BB = (1-x)'D(1-x) give S_AB = (S - S_AA - S_BB)/2
    for free.  That is what makes a large subsample affordable here, and sample
    size is the thing this test most needs.  The diagonal of D is zero, so the
    within-group sums are already the unbiased off-diagonal (U-statistic) form.
    """
    DX = D @ X                                        # (N, K)
    s_aa = np.einsum("ik,ik->k", X, DX)
    s_bb = np.einsum("ik,ik->k", 1.0 - X, rowsum[:, None] - DX)
    s_ab = (total - s_aa - s_bb) / 2.0
    energy = (2.0 * s_ab / (n * m)
              - s_aa / (n * (n - 1))
              - s_bb / (m * (m - 1)))
    return energy, s_ab / (n * m)


def energy_test(A: np.ndarray, B: np.ndarray, n_max: int, n_perm: int,
                n_repeats: int, rng, batch: int = 256) -> dict:
    """Energy test, averaged over independent subsample draws.

    The statistic needs an O(n^2) distance matrix, so it is subsampled — and a
    single draw is noisy enough that the p-value from one draw does not reproduce
    on the next.  The reported value is therefore the mean over `n_repeats`
    draws, each with its own permutation null, combined into the null of that
    mean.
    """
    na = min(len(A), n_max)
    nb = min(len(B), n_max)
    per_obs, per_cross, per_p, nulls = [], [], [], []
    for _ in range(n_repeats):
        Z = np.vstack([A[rng.choice(len(A), na, replace=False)],
                       B[rng.choice(len(B), nb, replace=False)]])
        N = na + nb
        D = cdist(Z, Z).astype(np.float32)
        rowsum = D.sum(axis=1, dtype=np.float64).astype(np.float32)
        total = float(rowsum.sum(dtype=np.float64))

        x0 = np.zeros((N, 1), dtype=np.float32)
        x0[:na] = 1.0
        e_obs, cross = _energy_batch(D, rowsum, total, x0, na, nb)
        per_obs.append(float(e_obs[0]))
        per_cross.append(float(cross[0]))

        null = np.empty(n_perm)
        done = 0
        while done < n_perm:
            k = min(batch, n_perm - done)
            X = np.zeros((N, k), dtype=np.float32)
            for j in range(k):
                X[rng.permutation(N)[:na], j] = 1.0
            null[done:done + k] = _energy_batch(D, rowsum, total, X, na, nb)[0]
            done += k
        nulls.append(null)
        per_p.append((1 + (null >= per_obs[-1]).sum()) / (1 + n_perm))
        del D

    out = {"n_a": int(na), "n_b": int(nb), "n_perm": int(n_perm),
           "n_repeats": int(n_repeats)}
    out.update(combine_repeats(per_obs, per_p, nulls))
    cross = float(np.mean(per_cross))
    out["energy"] = out["value"]
    out["energy_repeat_sd"] = out["value_repeat_sd"]
    # unit-free: 0 when identical, ->1 when the clouds are fully separated
    out["energy_normalised"] = (float(out["value"] / (2.0 * cross))
                                if cross > 0 else np.nan)
    return out


# ── joint test 2: Wasserstein in 8-D ───────────────────────────────────────────

def _w_exact_from_matrix(D: np.ndarray, ia: np.ndarray, ib: np.ndarray) -> tuple:
    """Exact W1 and W2 between two equal-size empirical clouds.

    With uniform weights and equal sample sizes the optimal transport plan is a
    permutation (the vertices of the Birkhoff polytope are permutation
    matrices), so exact optimal transport reduces to a linear assignment
    problem — no LP, no entropic regularisation, no tuning.
    """
    C = D[np.ix_(ia, ib)]
    r, c = linear_sum_assignment(C)
    w1 = float(C[r, c].mean())
    C2 = C ** 2
    r2, c2 = linear_sum_assignment(C2)
    w2 = float(np.sqrt(C2[r2, c2].mean()))
    return w1, w2


def wasserstein_nd_test(A: np.ndarray, B: np.ndarray, n_max: int, n_perm: int,
                        n_repeats: int, rng) -> dict:
    """Exact 8-D Wasserstein, averaged over independent subsample draws.

    The empirical estimator converges at n^(-1/d), so in 8-D what you compute is
    dominated by finite-sample bias, not by real distribution shift: two samples
    drawn from the *identical* distribution give a comfortably non-zero W.  The
    permutation null makes that concrete — expect null_mean to be large and the
    observed value to sit right on top of it.  Report the ratio, never the raw
    number on its own.

    The assignment solve is O(n^3), so this is the most heavily subsampled test
    here, and a single draw swings enough to move the p-value from 0.01 to 0.92.
    Hence the repeat-and-average treatment, same as the energy test.
    """
    m = min(len(A), len(B), n_max)   # equal sizes required for the assignment form
    obs1, obs2, nulls1, nulls2, p1, p2 = [], [], [], [], [], []
    for _ in range(n_repeats):
        Z = np.vstack([A[rng.choice(len(A), m, replace=False)],
                       B[rng.choice(len(B), m, replace=False)]])
        D = cdist(Z, Z).astype(np.float64)
        idx = np.arange(2 * m)
        w1, w2 = _w_exact_from_matrix(D, idx[:m], idx[m:])
        obs1.append(w1)
        obs2.append(w2)

        n1 = np.empty(n_perm)
        n2 = np.empty(n_perm)
        for k in range(n_perm):
            p = rng.permutation(2 * m)
            n1[k], n2[k] = _w_exact_from_matrix(D, p[:m], p[m:])
        nulls1.append(n1)
        nulls2.append(n2)
        p1.append((1 + (n1 >= w1).sum()) / (1 + n_perm))
        p2.append((1 + (n2 >= w2).sum()) / (1 + n_perm))
        del D

    def _pack(obs_list, p_list, nulls):
        res = combine_repeats(obs_list, p_list, nulls)
        res["excess_over_null"] = (float(res["value"] / res["null_mean"])
                                   if res["null_mean"] > 0 else np.nan)
        return res

    return {"n_per_side": int(m), "n_perm": int(n_perm), "n_repeats": int(n_repeats),
            "W1": _pack(obs1, p1, nulls1), "W2": _pack(obs2, p2, nulls2)}


def sliced_wasserstein_test(A: np.ndarray, B: np.ndarray, n_proj: int,
                            n_perm: int, rng) -> dict:
    """Sliced W1: average 1-D Wasserstein over random directions.

    Cheap enough to run on the *full* sample rather than a subsample, so it
    complements the exact version, which has to be subsampled.  Projections are
    drawn once and reused across permutations so the null reflects label noise
    only, not projection noise.
    """
    n_a, n_b = len(A), len(B)
    d = A.shape[1]
    V = rng.normal(size=(d, n_proj))
    V /= np.linalg.norm(V, axis=0, keepdims=True)
    P = np.vstack([A, B]) @ V                      # (N, n_proj)

    labels = np.zeros(n_a + n_b, dtype=bool)
    labels[:n_a] = True
    orders, sorted_vals, step_masks = [], [], []
    for j in range(n_proj):
        o = np.argsort(P[:, j], kind="stable")
        v = P[o, j]
        orders.append(o)
        sorted_vals.append(v)
        step_masks.append(np.append(np.diff(v) != 0, True))

    def _sw(lab):
        return float(np.mean([
            ks_and_w1_from_sorted(lab[orders[j]], sorted_vals[j], n_a, n_b, step_masks[j])[1]
            for j in range(n_proj)
        ]))

    obs = _sw(labels)
    null = np.array([_sw(rng.permutation(labels)) for _ in range(n_perm)])
    sd = float(null.std(ddof=1))
    return {
        "n_a": int(n_a), "n_b": int(n_b), "n_proj": int(n_proj), "n_perm": int(n_perm),
        "value": float(obs),
        "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm)),
        "null_mean": float(null.mean()),
        "null_sd": sd,
        "null_p95": float(np.percentile(null, 95)),
        "excess_over_null": float(obs / null.mean()) if null.mean() > 0 else np.nan,
        "z_score": float((obs - null.mean()) / sd) if sd > 0 else np.nan,
        "null_samples": null.tolist(),
    }


# ── joint test 3: classifier two-sample test ───────────────────────────────────

def make_c2st_classifier(kind: str, seed: int) -> Pipeline:
    """Same shape as analyse_latents.make_mlp_pipeline, classifier head."""
    if kind == "logreg":
        return Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
        ])
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", MLPClassifier(
            hidden_layer_sizes=(16, 16), activation="relu", solver="adam",
            max_iter=500, random_state=seed, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10, tol=1e-3,
        )),
    ])


def auc_null_sd(n1: int, n2: int) -> float:
    """Hanley-McNeil sd of the AUC under H0 (true AUC = 0.5).

    Var = (A(1-A) + (n1-1)(Q1 - A^2) + (n2-1)(Q2 - A^2)) / (n1 n2), which at
    A = 0.5 (so Q1 = Q2 = 1/3) collapses to (1/4 + (n1+n2-2)/12) / (n1 n2).
    Cross-validated AUC carries extra variance from model refitting, so treat
    this as a lower bound on the noise floor and cross-check against the
    fold-to-fold spread reported alongside it.
    """
    var = (0.25 + (n1 + n2 - 2) / 12.0) / (n1 * n2)
    return float(np.sqrt(var))


def c2st(A: np.ndarray, B: np.ndarray, kinds, n_folds: int, n_perm: int,
         n_repeats: int, rng, seed: int) -> dict:
    """Classifier two-sample test.

    If the two samples come from the same distribution then nothing can tell
    them apart, so a properly cross-validated classifier cannot beat chance.
    The held-out AUC *is* the test statistic, and unlike a p-value it says how
    separable the groups are, on the same scale as every other AUC in the
    analysis.  Classes are balanced by subsampling so 0.5 is a clean baseline
    and accuracy stays interpretable.

    Both the estimate and the null redraw that balanced subsample.  This matters:
    holding one subsample fixed and permuting only the labels gives a null that
    is far too narrow, because which points get drawn (and in which order, which
    moves the CV folds and the MLP's internal validation split) shifts the AUC
    by more than label noise does.  The observed value is therefore the mean over
    `n_repeats` draws, and each null replicate draws n points for each group from
    the pooled sample — a procedure distributionally identical to the observed
    one when the two distributions match, so it calibrates every source of
    variability the estimate is exposed to.
    """
    n = min(len(A), len(B))
    Z = np.vstack([A, B])
    y = np.concatenate([np.zeros(n), np.ones(n)])

    out = {"n_per_class": int(n), "n_folds": int(n_folds), "n_repeats": int(n_repeats),
           "auc_null_sd": auc_null_sd(n, n),
           "acc_null_sd": float(np.sqrt(0.25 / (2 * n)))}

    for kind in kinds:
        pipe = make_c2st_classifier(kind, seed)
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

        def one(a, b):
            X = np.vstack([a, b])
            proba = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba")[:, 1]
            return float(roc_auc_score(y, proba)), float(accuracy_score(y, proba > 0.5))

        # Each repeat is a self-contained test: its own balanced draw, its own
        # null built by drawing both groups from the pooled sample (which
        # mirrors the observed procedure exactly when the distributions match),
        # and so its own exactly-valid one-sided p-value.
        # n_perm is per repeat, not a budget split across them: the combined
        # p-value cannot go below 2/(n_perm+1), so splitting would floor it
        # (99 permutations over 5 repeats can never report below p=0.10).
        aucs, accs, per_p, nulls = [], [], [], []
        for _ in range(n_repeats):
            a_auc, a_acc = one(A[rng.choice(len(A), n, replace=False)],
                               B[rng.choice(len(B), n, replace=False)])
            aucs.append(a_auc)
            accs.append(a_acc)
            if n_perm > 0:
                null = np.empty(n_perm)
                for k in range(n_perm):
                    idx = rng.choice(len(Z), 2 * n, replace=False)
                    null[k] = one(Z[idx[:n]], Z[idx[n:]])[0]
                nulls.append(null)
                # one-sided: only an above-chance AUC is evidence of separability
                per_p.append((1 + (null >= a_auc).sum()) / (1 + n_perm))

        res = {"accuracy": float(np.mean(accs))}
        if n_perm > 0:
            res.update(combine_repeats(aucs, per_p, nulls))
            res["auc"] = res["value"]
            res["auc_repeats"] = res["value_repeats"]
            res["auc_repeat_sd"] = res["value_repeat_sd"]
            res["auc_excess_sd"] = res["z_score"]
            res["p_floor"] = float(min(1.0, 2.0 / (n_perm + 1)))
        else:
            auc = float(np.mean(aucs))
            sd = float(np.std(aucs, ddof=1)) if n_repeats > 1 else np.nan
            sem = sd / np.sqrt(n_repeats) if n_repeats > 1 else np.nan
            res.update({
                "auc": auc, "auc_repeats": [float(v) for v in aucs],
                "auc_repeat_sd": sd,
                "auc_ci95": [float(auc - 1.96 * sem), float(auc + 1.96 * sem)]
                            if n_repeats > 1 else [np.nan, np.nan],
            })
        out[kind] = res
    return out


# ── data loading ───────────────────────────────────────────────────────────────

def match_species_composition(per_species: dict, rng) -> tuple:
    """Subsample the val pool so its species mix matches the train pool's.

    A balanced split (data.tag=bal9419) makes the *training* set exact thirds and
    leaves the remainder unbalanced, so a naive pooled train-vs-val comparison is
    confounded: species are separable in latent space by construction, and two
    pools with different species mixtures differ for that reason alone, with
    nothing to do with train/val membership.  Matching the mixture first is what
    isolates the question actually being asked.
    """
    n_train = sum(len(v[0]) for v in per_species.values())
    frac = {s: len(v[0]) / n_train for s, v in per_species.items()}
    # largest val pool that can hit those fractions without sampling with replacement
    n_val = min(len(v[1]) / frac[s] for s, v in per_species.items())
    val_parts, taken = [], {}
    for s, (_, val) in per_species.items():
        k = min(len(val), int(round(frac[s] * n_val)))
        val_parts.append(val[rng.choice(len(val), k, replace=False)])
        taken[s] = k
    train = np.vstack([v[0] for v in per_species.values()])
    return train, np.vstack(val_parts), frac, taken


def species_fractions(per_species: dict) -> dict:
    """Per-species share of the train and val pools — recorded so a composition
    confound in the 'combined' comparison is visible in the results, not hidden."""
    n_tr = sum(len(v[0]) for v in per_species.values())
    n_va = sum(len(v[1]) for v in per_species.values())
    return {
        "train": {s: len(v[0]) / n_tr for s, v in per_species.items()},
        "val": {s: len(v[1]) / n_va for s, v in per_species.items()},
        "max_abs_difference": max(abs(len(v[0]) / n_tr - len(v[1]) / n_va)
                                  for v in per_species.values()),
    }


def load_groups(cfg: dict, model_name: str, comparisons, rng) -> dict:
    """Returns {comparison_name: (train_latents, val_latents)}.

    Only train.npz/val.npz are already split; kaon.npz and muon.npz hold *all*
    latents for that species and are split with the within-species indices in
    species_split.npz, which is the split actually used in training.
    """
    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name
    if not inf_dir.exists():
        raise FileNotFoundError(f"No inference directory at {inf_dir} — run run_inference.py first")

    p_train = np.load(inf_dir / "train.npz")["latents"]
    p_val = np.load(inf_dir / "val.npz")["latents"]

    all_species = cfg["data"].get("proton") == "all"
    ss = np.load(inf_dir / "species_split.npz") if all_species else None

    per_species = {"proton": (p_train, p_val)}
    if ss is not None:
        for name, fname, key in (("kaon", "kaon.npz", "k"), ("muon", "muon.npz", "m")):
            path = inf_dir / fname
            if not path.exists():
                print(f"  ! {fname} missing — skipping {name}")
                continue
            lat = np.load(path)["latents"]
            tr, va = ss[f"{key}_train_idx"], ss[f"{key}_val_idx"]
            if lat.shape[0] != len(tr) + len(va):
                raise ValueError(
                    f"{fname} has {lat.shape[0]} latents but species_split expects "
                    f"{len(tr) + len(va)} — latents and split are out of sync")
            per_species[name] = (lat[tr], lat[va])

    groups, extras = {}, {}
    for name in comparisons:
        if name in ("combined", "combined_matched"):
            if len(per_species) < 2:
                print(f"  ! '{name}' needs more than one species — skipping")
                continue
            frac = species_fractions(per_species)
            if name == "combined":
                groups["combined"] = (
                    np.vstack([v[0] for v in per_species.values()]),
                    np.vstack([v[1] for v in per_species.values()]),
                )
                extras["combined"] = {"species_fractions": frac}
                if frac["max_abs_difference"] > 0.01:
                    print(
                        f"  ! CONFOUND: train and val pools have different species mixtures "
                        f"(max difference {frac['max_abs_difference']:.1%}).\n"
                        f"    train {({s: f'{v:.1%}' for s, v in frac['train'].items()})}\n"
                        f"    val   {({s: f'{v:.1%}' for s, v in frac['val'].items()})}\n"
                        f"    Species are separable in latent space, so 'combined' will show a "
                        f"difference for that reason alone. Use combined_matched.")
            else:
                tr, va, f_used, taken = match_species_composition(per_species, rng)
                groups["combined_matched"] = (tr, va)
                extras["combined_matched"] = {
                    "species_fractions": frac,
                    "matched_to_train_fractions": f_used,
                    "val_taken_per_species": taken,
                }
                print(f"  combined_matched: val resampled to the train mixture "
                      f"({', '.join(f'{s} {n}' for s, n in taken.items())})")
        elif name in per_species:
            groups[name] = per_species[name]
        else:
            print(f"  ! no latents for '{name}' — skipping")
    return groups, extras


def apply_standardize(A: np.ndarray, B: np.ndarray, mode: str) -> tuple:
    """Put the two clouds on a common scale before any Euclidean-distance test.

    Latent dims of a beta-VAE have very different variances (collapsed dims sit
    at the prior), so an unstandardised Euclidean distance is dominated by the
    highest-variance dims.  'pooled' z-scores using the *combined* sample, which
    is a function of the pooled data alone and therefore invariant under label
    permutation — the permutation test stays exact.  'none' keeps the native
    N(0, I) prior metric and is worth reporting as a robustness check.
    Marginal (KS/1-D Wasserstein) results are unaffected by this choice; they
    divide by the pooled sd of each dimension regardless.
    """
    if mode == "none":
        return A, B
    ref = np.vstack([A, B]) if mode == "pooled" else A
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0, ddof=1)
    sd[sd == 0] = 1.0
    return (A - mu) / sd, (B - mu) / sd


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              cwd=PROJECT_ROOT, capture_output=True, text=True,
                              check=True).stdout.strip()
    except Exception:
        return "unknown"


# ── driver ─────────────────────────────────────────────────────────────────────

def analyse_pair(name: str, A: np.ndarray, B: np.ndarray, args, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    res = {"comparison": name, "n_train": int(len(A)), "n_val": int(len(B)),
           "n_dim": int(A.shape[1])}
    A_s, B_s = apply_standardize(A, B, args.standardize)

    if "ks" in args.tests:
        t = time.time()
        res["marginal"] = marginal_tests(A, B, args.n_perm, rng)
        print(f"    marginals      {time.time() - t:5.1f}s  "
              f"max D={res['marginal']['global_max_D']:.4f} "
              f"(perm p={res['marginal']['global_max_D_p_perm']:.4f}), "
              f"{res['marginal']['n_holm_significant']}/{A.shape[1]} dims Holm-significant")

    if "energy" in args.tests:
        t = time.time()
        res["energy"] = energy_test(A_s, B_s, args.energy_n, args.n_perm,
                                    args.energy_repeats, rng)
        e = res["energy"]
        print(f"    energy         {time.time() - t:5.1f}s  "
              f"E={e['energy']:.5f} +/-{e['energy_repeat_sd']:.5f} over draws  "
              f"p={e['p_value']:.4f}  z={e['z_score']:+.2f}")

    if "wasserstein" in args.tests:
        t = time.time()
        res["wasserstein_nd"] = wasserstein_nd_test(A_s, B_s, args.wnd_n, args.wnd_perm,
                                                    args.wnd_repeats, rng)
        w = res["wasserstein_nd"]["W1"]
        print(f"    W (exact 8-D)  {time.time() - t:5.1f}s  "
              f"W1={w['value']:.4f} vs null {w['null_mean']:.4f} "
              f"(ratio {w['excess_over_null']:.3f})  p={w['p_value']:.4f}")
        if args.sliced_n_proj > 0:
            t = time.time()
            res["wasserstein_sliced"] = sliced_wasserstein_test(
                A_s, B_s, args.sliced_n_proj, args.wnd_perm, rng)
            s = res["wasserstein_sliced"]
            print(f"    W (sliced)     {time.time() - t:5.1f}s  "
                  f"SW1={s['value']:.5f} vs null {s['null_mean']:.5f}  p={s['p_value']:.4f}")

    if "c2st" in args.tests:
        t = time.time()
        res["c2st"] = c2st(A_s, B_s, args.c2st_models, args.c2st_folds,
                           args.c2st_perm, args.c2st_repeats, rng, args.seed)
        for k in args.c2st_models:
            c = res["c2st"][k]
            if "null_sd" in c:
                tail = (f"+/-{c['auc_repeat_sd']:.4f} over draws | "
                        f"null {c['null_mean']:.4f} +/- {c['null_sd']:.4f}  "
                        f"p={c['p_value']:.4f}  ({c['auc_excess_sd']:+.1f} sd)")
            else:
                tail = f"analytic null sd={res['c2st']['auc_null_sd']:.4f}"
            print(f"    C2ST {k:<7}   {time.time() - t:5.1f}s  "
                  f"AUC={c['auc']:.4f}  {tail}")
    return res


def main():
    ap = argparse.ArgumentParser(
        description="Two-sample tests comparing train and val VAE latents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--config", required=True, help="Path to model YAML config")
    ap.add_argument("--comparisons", nargs="+", choices=COMPARISONS, default=COMPARISONS,
                    help="'combined' pools all species; the rest are per-species")
    ap.add_argument("--tests", nargs="+", choices=TESTS, default=TESTS)
    ap.add_argument("--label", default="default",
                    help="Names the output subdirectory — use it to keep data configurations apart")
    ap.add_argument("--standardize", choices=["pooled", "train", "none"], default="pooled",
                    help="Scaling applied before the multivariate (Euclidean) tests")
    ap.add_argument("--null-control", action="store_true",
                    help="Negative control: reshuffle the train/val labels, preserving group "
                         "sizes, so the null is true by construction. Everything should come "
                         "back non-significant; if it does not, the pipeline is broken.")
    ap.add_argument("--subsample-frac", type=float, default=1.0,
                    help="Run on a random fraction of each group — shows how much of any "
                         "significance is bought by sample size alone")
    ap.add_argument("--n-perm", type=int, default=999,
                    help="Permutations for the marginal and energy nulls")
    ap.add_argument("--energy-n", type=int, default=5000,
                    help="Max points per side for the energy test. Memory is the binding "
                         "constraint: the pooled distance matrix is (2n)^2 float32, so 5000 "
                         "per side is ~400 MB.")
    ap.add_argument("--energy-repeats", type=int, default=3,
                    help="Independent subsample draws averaged into the energy statistic. "
                         "One draw is not reproducible run to run.")
    ap.add_argument("--wnd-n", type=int, default=1000,
                    help="Points per side for exact 8-D Wasserstein (O(n^3) assignment)")
    ap.add_argument("--wnd-perm", type=int, default=149,
                    help="Permutations per repeat for the Wasserstein nulls (each needs an "
                         "assignment solve)")
    ap.add_argument("--wnd-repeats", type=int, default=3,
                    help="Independent subsample draws averaged into the Wasserstein "
                         "statistics. A single draw can swing the p-value from 0.01 to 0.9.")
    ap.add_argument("--sliced-n-proj", type=int, default=100,
                    help="Random projections for sliced Wasserstein; 0 disables it")
    ap.add_argument("--c2st-models", nargs="+", choices=["mlp", "logreg"],
                    default=["mlp", "logreg"])
    ap.add_argument("--c2st-folds", type=int, default=5)
    ap.add_argument("--c2st-repeats", type=int, default=3,
                    help="Balanced subsample draws averaged into the reported AUC. A single "
                         "draw is noticeably unstable — which points are drawn also sets the "
                         "CV folds and the MLP's internal validation split.")
    ap.add_argument("--c2st-perm", type=int, default=99,
                    help="Null replicates per C2ST repeat (total cost is this times "
                         "--c2st-repeats). Each replicate is a full CV, so "
                         "this is the slow part; set 0 for a quick pass, which falls back to "
                         "the analytic Hanley-McNeil null. The analytic null assumes a fixed "
                         "scoring rule and so understates the spread of a cross-validated AUC, "
                         "which is refit on every fold — prefer the empirical null when "
                         "quoting a result.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="Override the output directory")
    ap.add_argument("--no-plots", action="store_true",
                    help="Skip rendering; run scripts/plot_two_sample.py later")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = build_model_name(cfg)
    print(f"Model:  {model_name}")

    out_dir = Path(args.out) if args.out else (
        PROJECT_ROOT / "figs" / model_name / "two-sample" / args.label)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    rng = np.random.default_rng(args.seed)
    groups, extras = load_groups(cfg, model_name, args.comparisons, rng)

    if args.subsample_frac < 1.0:
        f = args.subsample_frac
        groups = {k: (v[0][rng.choice(len(v[0]), max(2, int(f * len(v[0]))), replace=False)],
                      v[1][rng.choice(len(v[1]), max(2, int(f * len(v[1]))), replace=False)])
                  for k, v in groups.items()}
        print(f"Subsampled to {f:.0%} of each group")

    if args.null_control:
        # Pool and re-split at the same group sizes. The two groups are now
        # genuinely the same distribution, so this run is the reference column:
        # anything the real run shows that this one also shows is not a finding.
        relabelled = {}
        for k, (A, B) in groups.items():
            Z = np.vstack([A, B])
            p = rng.permutation(len(Z))
            relabelled[k] = (Z[p[:len(A)]], Z[p[len(A):]])
        groups = relabelled
        print("NULL CONTROL: train/val labels reshuffled — the null is true by construction")

    results = {}
    for i, (name, (A, B)) in enumerate(groups.items()):
        print(f"\n=== {name}  (train n={len(A)}, val n={len(B)}, dim={A.shape[1]}) ===")
        # a per-comparison seed keeps each comparison reproducible on its own
        results[name] = analyse_pair(name, A, B, args, args.seed + 1000 * i)
        results[name].update(extras.get(name, {}))

    payload = {
        "meta": {
            "model_name": model_name,
            "config": str(Path(args.config).resolve()),
            "label": args.label,
            "null_control": bool(args.null_control),
            "standardize": args.standardize,
            "subsample_frac": args.subsample_frac,
            "seed": args.seed,
            "tests": args.tests,
            "n_perm": args.n_perm,
            "energy_n": args.energy_n,
            "energy_repeats": args.energy_repeats,
            "wnd_n": args.wnd_n,
            "wnd_perm": args.wnd_perm,
            "wnd_repeats": args.wnd_repeats,
            "sliced_n_proj": args.sliced_n_proj,
            "c2st_models": args.c2st_models,
            "c2st_folds": args.c2st_folds,
            "c2st_perm": args.c2st_perm,
            "c2st_repeats": args.c2st_repeats,
            "git_commit": git_commit(),
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "command": " ".join(sys.argv),
        },
        "comparisons": results,
    }
    json_path = out_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {json_path}")

    if not args.no_plots:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from plot_two_sample import render_all   # noqa: E402
        render_all(payload, out_dir)
    else:
        print(f"Render with:\n  python scripts/plot_two_sample.py --results {json_path}")


if __name__ == "__main__":
    main()
