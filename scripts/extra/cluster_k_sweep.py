#!/usr/bin/env python3
"""
scripts/extra/cluster_k_sweep.py

Measurement layer for the unsupervised k sweep: how the latent space partitions
as the component count grows, and how well those partitions serve two concrete
tasks. Shared by plot_kscan_composition.py and plot_kscan_embeddings.py so every
figure and number in this thread comes from one fit per (k, seed).

NO LABEL ENTERS ANY FIT. Mixtures are fitted on the raw 8D latents; beam tags,
anchored assignments and spectrometer mass are read back afterwards only to
score and describe. Same protocol as cluster_latents.py, so k=3 reproduces the
ARI 0.367 / purity 0.734 already in the paper.

k IS SPECIFIED, NOT DISCOVERED. BIC falls monotonically across the whole range
tested (954,680 at k=3 to 832,357 at k=30 and still decreasing), so no internal
criterion selects any k. This sweep describes what the space looks like carved
into k parts; it is not a search for the true number.

MODES
  scan    per-cluster composition, plus per-k trend statistics. The trend table
          is what makes a long sweep readable: max cluster purity, how many
          clusters clear 80/90%, what fraction of events land in a >=85% pure
          cluster, and the smallest cluster. That last one is the fraud check --
          purity that rises only because clusters shrink is not a finding.

  tasks   two downstream tasks per k.
            p/MIP   held-out binary accuracy. Cluster calls are decided on half
                    the events and scored on the other half, so a cluster that
                    is pure only because it is tiny earns nothing. The naive
                    same-events version is NOT monotone in k here, because GMM
                    refits from scratch at each k rather than refining.
            decon   agreement with the anchored assignment on the kaon->proton
                    call, over kaon-tagged events only. Reported as MCC and F1:
                    the target is 20.6% positive, so plain accuracy would rank
                    a do-nothing partition near 80%.
            mass    median spectrometer mass of kaon-tagged events in
                    proton-majority clusters minus the same in kaon-majority
                    clusters. EXTERNAL to both the model and the anchored
                    target, so it is the column that says whether agreement with
                    anchored is real or two methods sharing a bias.

  seeds   replicate `tasks` over several seeds. NOT OPTIONAL AT HIGH k. Single
          fits at k~38 gave MCC differences of 0.01-0.02 between neighbouring k,
          but the between-seed sd is ~0.035 -- larger than the between-k spread,
          so the band's internal ordering is noise. Worse, mean pairwise ARI
          between seeds at the same k is only 0.62-0.66, meaning the partitions
          themselves differ run to run and no individual cluster is citable.
          At k=3 the same check returns sd ~0.000. Run this before quoting any
          specific k or cluster.

CAVEAT ON THE ANCHORED TARGET
    anchored_clustering is not ground truth. It uses the proton and MIP beam
    tags and is derived from the same latent space, so `decon` agreement is
    inflated by shared method. The mass column is the independent check.

CAVEAT ON MASS
    The kaon TAG is itself a spectrometer mass selection (the tagged sample
    spans 348.9-648.2 MeV with hard edges). Mass can only reveal shifts WITHIN
    that window; a contaminating proton that survived the tag is one whose mass
    was mis-reconstructed into it. A positive split means "these sit at the
    heavy edge", never "these are protons".

OUTPUTS (under figs/<model_name>/kscan/)
    kscan_clusters.csv   one row per (k, cluster)
    kscan_summary.csv    one row per k, trend statistics
    kscan_tasks.csv      one row per k, the two tasks plus the mass split
    kscan_seeds.csv      one row per (k, seed)

Usage:
    python scripts/extra/cluster_k_sweep.py --config configs/run_0093_*.yaml --mode scan
    python scripts/extra/cluster_k_sweep.py --config ... --mode tasks --kmax 50
    python scripts/extra/cluster_k_sweep.py --config ... --mode seeds --kmin 36 --kmax 41
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, f1_score, matthews_corrcoef,
                             normalized_mutual_info_score)
from sklearn.mixture import GaussianMixture

from _beam_data import (SPECIES, build_model_name, figure_dir, load_beam_data,
                        load_config)

PROTON, KAON, MIP = 0, 1, 2


# ---------------------------------------------------------------- shared bits
def labels_for(Z, k, seed=0, cache_dir=None, recompute=False, n_init=20):
    """Cluster labels for one (k, seed), cached next to the inference files.

    Cached because a full sweep is ~50 mixture fits and every figure in this
    thread needs the same labels; refitting per figure would also risk two
    figures showing different partitions of the same k.
    """
    if cache_dir is not None:
        path = Path(cache_dir) / f"gmm_labels_k{k}_seed{seed}.npy"
        if path.exists() and not recompute:
            lab = np.load(path)
            if len(lab) == len(Z):
                return lab
    lab = GaussianMixture(k, covariance_type="full", n_init=n_init,
                          random_state=seed).fit_predict(Z)
    if cache_dir is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, lab)
    return lab


def composition(lab, species, k, sort="gradient"):
    """Per-cluster species fractions in a reading order.

    sort="gradient" (default) sorts by DESCENDING (proton - MIP), the single key
    that makes both orderings hold at once: raising a cluster's proton fraction
    moves it left, raising its MIP fraction moves it right. Proton therefore
    falls and MIP rises across the axis together, and kaon-dominated clusters --
    which are low in both -- land in the middle.

    No key can make both hold EXACTLY, since proton and MIP are not perfectly
    anti-correlated: kaon absorbs the remainder. This is the ordering that
    minimises the tension between them.

    sort="proton-then-mip" is the plain lexicographic rule (descending proton,
    ascending MIP as tie-break). It is the literal reading but a worse picture:
    proton fractions are all distinct on real data, so the MIP key never fires
    and kaon-dominated clusters end up scattered among the MIP-dominated ones.
    """
    rows = []
    for c in range(k):
        m = lab == c
        if not m.any():
            continue
        rows.append({"cluster": c, "n": int(m.sum()),
                     **{s: float((species[m] == s).mean()) for s in SPECIES}})

    if sort == "proton-then-mip":
        rows.sort(key=lambda r: (-r["proton"], r["muon"]))
    else:
        rows.sort(key=lambda r: -(r["proton"] - r["muon"]))
    return rows


def _truth(df):
    return pd.Categorical(df["species"], categories=SPECIES).codes


# ---------------------------------------------------------------- mode: scan
def scan(Z, df, ks, seed, cache_dir):
    truth = _truth(df)
    N = len(Z)
    clusters, summary = [], []
    for k in ks:
        lab = labels_for(Z, k, seed, cache_dir)
        purity_c, sizes = [], []
        for c in range(k):
            m = lab == c
            cnt = np.bincount(truth[m], minlength=3)
            sizes.append(int(m.sum()))
            purity_c.append(cnt.max() / max(m.sum(), 1))
            clusters.append({
                "k": k, "cluster": c, "n": int(m.sum()),
                **{f"n_{s}": int(cnt[i]) for i, s in enumerate(SPECIES)},
                **{f"frac_{s}": float(cnt[i] / max(m.sum(), 1))
                   for i, s in enumerate(SPECIES)},
                "majority": SPECIES[int(cnt.argmax())],
                "cluster_purity": float(cnt.max() / max(m.sum(), 1)),
                "median_mean_adc": float(df.loc[m, "mean_adc"].median()),
                "median_solidity": float(df.loc[m, "solidity"].median()),
                "median_height": float(df.loc[m, "height"].median()),
            })
        purity_c, sizes = np.array(purity_c), np.array(sizes)
        best = {f"best_{s}": float(max(
            np.bincount(truth[lab == c], minlength=3)[i] / max((lab == c).sum(), 1)
            for c in range(k))) for i, s in enumerate(SPECIES)}
        summary.append({
            "k": k, "ari": adjusted_rand_score(truth, lab),
            "nmi": normalized_mutual_info_score(truth, lab),
            "purity": float(sum(np.bincount(truth[lab == c], minlength=3).max()
                                for c in range(k)) / N),
            "max_cluster_purity": float(purity_c.max()),
            "n_clusters_80pct": int((purity_c >= 0.80).sum()),
            "n_clusters_90pct": int((purity_c >= 0.90).sum()),
            "frac_in_85pct_clusters": float(sizes[purity_c >= 0.85].sum() / N),
            "smallest_cluster": int(sizes.min()),
            "n_below_100": int((sizes < 100).sum()), **best})
        s = summary[-1]
        print(f"  k={k:>3} ARI {s['ari']:.3f}  purity {s['purity']:.3f}  "
              f"best k/p/m {s['best_kaon']:.3f}/{s['best_proton']:.3f}/"
              f"{s['best_muon']:.3f}  min n {s['smallest_cluster']:>5}", flush=True)
    return pd.DataFrame(clusters), pd.DataFrame(summary)


# --------------------------------------------------------------- mode: tasks
def _task_row(lab, k, truth, anch, mass, half):
    """The two tasks plus the external mass split, for one fitted partition."""
    maj = np.array([np.bincount(truth[lab == c], minlength=3).argmax()
                    for c in range(k)])
    pm, isP = truth != KAON, truth == PROTON

    ok = n = 0                                   # held-out proton/MIP
    for c in range(k):
        m = lab == c
        aP = int((m & half & pm & isP).sum()); aM = int((m & half & pm & ~isP).sum())
        bP = int((m & ~half & pm & isP).sum()); bM = int((m & ~half & pm & ~isP).sum())
        ok += bP if aP >= aM else bM
        n += bP + bM

    kt = truth == KAON                           # agreement with anchored
    target = anch[kt] == PROTON
    pred = maj[lab[kt]] == PROTON

    inP = np.isin(lab, np.where(maj == PROTON)[0])
    inK = np.isin(lab, np.where(maj == KAON)[0])
    a, b = mass[kt & inP], mass[kt & inK]
    split = (float(np.median(a) - np.median(b))
             if len(a) > 20 and len(b) > 20 else np.nan)
    sizes = np.bincount(lab, minlength=k)
    return {"k": k, "pmip_heldout": ok / n,
            "mcc": matthews_corrcoef(target, pred), "f1": f1_score(target, pred),
            "precision": float((target & pred).sum() / max(pred.sum(), 1)),
            "recall": float((target & pred).sum() / max(target.sum(), 1)),
            "mass_split": split, "n_flagged": int(pred.sum()),
            "smallest_cluster": int(sizes.min()),
            "n_below_100": int((sizes < 100).sum())}


def tasks(Z, df, ks, seed, cache_dir, seeds=None):
    from anchored_clustering import anchored_fit
    truth = _truth(df)
    mass = df["beamline_mass"].to_numpy(float)
    anch, _, _ = anchored_fit(Z, df["species"])
    # deterministic split, so rows from separate runs stay comparable
    half = np.random.default_rng(0).random(len(Z)) < 0.5
    rows = []
    for k in ks:
        for s in (seeds if seeds is not None else [seed]):
            lab = labels_for(Z, k, s, cache_dir)
            r = _task_row(lab, k, truth, anch, mass, half)
            r["seed"] = s
            rows.append(r)
            print(f"  k={k:>3} seed{s}  p/MIP {r['pmip_heldout']:.4f}  "
                  f"MCC {r['mcc']:.3f}  prec {r['precision']:.3f}  "
                  f"mass {r['mass_split']:+.1f}  min n {r['smallest_cluster']:>5}",
                  flush=True)
    return pd.DataFrame(rows)


def report_seeds(r, Z, ks, cache_dir):
    """Is the band's internal ordering resolvable, and are the partitions stable?"""
    print("\n=== mean +- sd over seeds ===", flush=True)
    for k in ks:
        x = r[r.k == k]
        print(f"  k={k}: p/MIP {x.pmip_heldout.mean():.4f}+-{x.pmip_heldout.std():.4f}"
              f"   MCC {x.mcc.mean():.3f}+-{x.mcc.std():.3f}"
              f"   mass {x.mass_split.mean():+.1f}+-{x.mass_split.std():.1f}", flush=True)
    print("\n=== between-k spread vs within-k (seed) spread ===", flush=True)
    print("    ratio < 1 means neighbouring k are NOT distinguishable", flush=True)
    for col in ("pmip_heldout", "mcc", "mass_split"):
        bk = r.groupby("k")[col].mean().std()
        wk = r.groupby("k")[col].std().mean()
        print(f"  {col:<14} between-k {bk:.4f}   within-k {wk:.4f}   "
              f"ratio {bk/wk:.2f}", flush=True)
    print("\n=== partition stability: pairwise ARI between seeds at fixed k ===",
          flush=True)
    print("    low values mean individual clusters are not reproducible", flush=True)
    for k in ks:
        ss = sorted(r[r.k == k].seed.unique())
        labs = {s: labels_for(Z, k, s, cache_dir) for s in ss}
        a = [adjusted_rand_score(labs[i], labs[j])
             for ii, i in enumerate(ss) for j in ss[ii + 1:]]
        if a:
            print(f"  k={k}: mean {np.mean(a):.3f}  (min {np.min(a):.3f}, "
                  f"max {np.max(a):.3f})", flush=True)


# ----------------------------------------------------------------------- CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="model YAML (all-species run)")
    ap.add_argument("--mode", choices=["scan", "tasks", "seeds"], default="scan")
    ap.add_argument("--kmin", type=int, default=3)
    ap.add_argument("--kmax", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-seeds", type=int, default=5,
                    help="seeds mode: how many seeds per k")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    cache_dir = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else figure_dir(cfg, "kscan")
    out_dir.mkdir(parents=True, exist_ok=True)
    ks = list(range(args.kmin, args.kmax + 1))
    print(f"{args.mode}: k={args.kmin}..{args.kmax} on {len(Z)} events "
          f"x {Z.shape[1]}D -> {out_dir}", flush=True)

    if args.mode == "scan":
        clusters, summary = scan(Z, df, ks, args.seed, cache_dir)
        clusters.to_csv(out_dir / "kscan_clusters.csv", index=False)
        summary.to_csv(out_dir / "kscan_summary.csv", index=False)
        print(f"\nwrote kscan_clusters.csv ({len(clusters)} rows), "
              f"kscan_summary.csv ({len(summary)} rows)")
        print("\nNote: BIC falls throughout this range, so no k here is selected "
              "by the data.")
    elif args.mode == "tasks":
        r = tasks(Z, df, ks, args.seed, cache_dir)
        r.to_csv(out_dir / "kscan_tasks.csv", index=False)
        print(f"\nwrote kscan_tasks.csv ({len(r)} rows)")
        print("Single seed per k: do NOT pick a k from this without --mode seeds.")
    else:
        r = tasks(Z, df, ks, args.seed, cache_dir, seeds=list(range(args.n_seeds)))
        r.to_csv(out_dir / "kscan_seeds.csv", index=False)
        report_seeds(r, Z, ks, cache_dir)
        print(f"\nwrote kscan_seeds.csv ({len(r)} rows)")


if __name__ == "__main__":
    main()
