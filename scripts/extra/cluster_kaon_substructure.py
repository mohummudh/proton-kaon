#!/usr/bin/env python3
"""
scripts/extra/cluster_kaon_substructure.py

Sub-structure WITHIN the kaon-tagged sample: what its latents look like on their
own, what clustering them finds, and whether any of the resulting populations are
recognisably junk rather than physics.

THE HEADLINE, MEASURED FIRST: IT IS A CONTINUUM, NOT CLUMPS
    Do not read the sub-clusters as discovered populations. On the kaon-only
    latents the BIC falls monotonically out to k=10 with no elbow, and the
    silhouette is 0.03-0.10 at every k -- an order of magnitude below what separated
    structure gives. So the kaon sample is one connected, heterogeneous cloud, and
    any partition of it is a STRATIFICATION the analyst imposed, not a set of
    natural boundaries the data volunteered.

    That is worth stating plainly because the partition is still useful. Cutting a
    continuum at arbitrary places is a legitimate way to characterise what varies
    along it, and here the strata line up with interpretable physics. What is not
    legitimate is calling them clusters and counting them.

WHAT THE STRATA ACTUALLY TRACK
    Reconstruction error is the sharpest axis: at k=5 the per-cluster medians span
    0.048 to 0.820, a factor of 17, while the whole sample's interquartile range is
    much narrower. So the dominant thing the latent space encodes about a kaon-tagged
    event is HOW HARD THE VAE FOUND IT, which is a statement about the model as much
    as about the event. Morphology follows it: the well-reconstructed stratum is
    short, bright, high-solidity and single-peaked, and the badly-reconstructed one
    is long, faint, low-solidity and multi-peaked.

RECOGNISING JUNK
    Three explicit rules, applied and reported rather than eyeballed:

      small      n_pixels below --min-pixels. A near-empty image: the cluster
                 finder found something, but there is little of it.
      hard       recon_error above the --junk-re-quantile quantile. The VAE could
                 not model the event, so its latent position is least trustworthy
                 precisely here.
      far        ||z|| above the same quantile. Latent-space outliers, which the
                 anchored densities extrapolate over rather than interpolate.

    These are diagnostics, not vetoes. A short bright track is exactly what a
    stopping proton looks like, and a high recon error is what a genuine multi-prong
    kaon decay looks like, so flagged events are not automatically bad -- the flags
    say where to LOOK, and the per-cluster image grids are how you look.

OUTPUTS (under figs/<model_name>/kaon-substructure/)
    kaon_substructure.{png,pdf}      k-scan, UMAP by stratum, profile heatmap,
                                     recon-error distributions
    kaon_stratum_NN.{png,pdf}        16 example events per stratum
    kaon_substructure.csv            per-event stratum, flags and features
    metrics.json                     k-scan, per-stratum profiles, flag counts

Usage:
    python scripts/extra/cluster_kaon_substructure.py --config configs/run_0093_*.yaml
    python scripts/extra/cluster_kaon_substructure.py --config ... --k 6 --scan-k 2 12
    python scripts/extra/cluster_kaon_substructure.py --config ... --no-images
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from _beam_data import (DOUBLE_COL, SINGLE_COL, apply_style, figure_dir,
                        load_beam_data, load_config, load_embedding, savefig)
from plot_anchored_proton_kaons import IMAGE_256, PLANE_INDEX, group_windows, plot_grid

# Features profiled per stratum. Chosen to span calorimetry, topology, size and
# model difficulty rather than to be exhaustive — 22 rows of heatmap is unreadable.
PROFILE = ["recon_error", "mean_adc", "solidity", "fill_fraction", "n_local_maxima",
           "height", "n_pixels", "bragg_peak_ratio", "beamline_mass"]
LABELS = {"recon_error": "Reconstruction error", "mean_adc": "Calorimetry proxy",
          "solidity": "Topology proxy", "fill_fraction": "Fill fraction",
          "n_local_maxima": "Local maxima", "height": "Track length",
          "n_pixels": "Signal pixels", "bragg_peak_ratio": "Bragg peak ratio",
          "beamline_mass": "Beamline mass"}


def scan_k(Zk, lo, hi, seed=0):
    """BIC and silhouette across k, to show whether any k is actually preferred."""
    rows = []
    for k in range(lo, hi + 1):
        g = GaussianMixture(k, covariance_type="full", n_init=10, random_state=seed).fit(Zk)
        lab = g.predict(Zk)
        rows.append({"k": k, "bic": float(g.bic(Zk)), "aic": float(g.aic(Zk)),
                     "silhouette": float(silhouette_score(Zk, lab, sample_size=4000,
                                                          random_state=seed))})
    return pd.DataFrame(rows)


def flag_junk(dk, Zk, min_pixels, q):
    """Explicit, reported junk heuristics. See the module docstring on why these are
    diagnostics rather than vetoes."""
    norm = np.linalg.norm(Zk, axis=1)
    re_cut = float(dk["recon_error"].quantile(q))
    nm_cut = float(np.quantile(norm, q))
    flags = pd.DataFrame({
        "small": dk["n_pixels"].to_numpy() < min_pixels,
        "hard": dk["recon_error"].to_numpy() > re_cut,
        "far": norm > nm_cut,
    })
    flags["any_flag"] = flags.any(axis=1)
    return flags, {"min_pixels": min_pixels, "recon_error_cut": re_cut,
                   "latent_norm_cut": nm_cut, "quantile": q}


def plot_overview(scan, emb, dk, labels, k, out_dir):
    s = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, DOUBLE_COL * 0.78))
    cmap = plt.get_cmap("tab10")

    # (a) is any k preferred? BIC on the left axis, silhouette on the right.
    ax = axes[0, 0]
    ax.plot(scan["k"], scan["bic"], "o-", color="#0077BB", ms=4, lw=1.1, label="BIC")
    ax.set_ylabel("BIC", color="#0077BB")
    ax.tick_params(axis="y", colors="#0077BB")
    ax2 = ax.twinx()
    ax2.plot(scan["k"], scan["silhouette"], "s--", color="#EE7733", ms=3.6, lw=1.1,
             label="silhouette")
    ax2.set_ylabel("Silhouette", color="#EE7733")
    ax2.tick_params(axis="y", colors="#EE7733")
    ax2.spines[["top"]].set_visible(False)
    ax.axvline(k, color="0.6", lw=0.8, ls=":", zorder=0)
    ax.set_xlabel("Number of components $k$")
    ax.set_title("(a) no $k$ is preferred", loc="left", fontsize=9 * s, pad=3)

    # (b) where the strata sit in the shared projection
    ax = axes[0, 1]
    if emb is not None:
        for c in range(k):
            m = labels == c
            ax.scatter(emb[m, 0], emb[m, 1], s=1.2, alpha=0.45, lw=0,
                       color=cmap(c), label=f"{c} (n={m.sum()})", rasterized=True)
        leg = ax.legend(markerscale=6, fontsize=6 * s, handletextpad=0.25,
                        borderpad=0.3, loc="best", frameon=True, framealpha=0.85)
        for h in leg.legend_handles:
            h.set_alpha(1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    else:
        ax.axis("off")
    ax.set_title("(b) strata in the shared projection", loc="left", fontsize=9 * s, pad=3)

    # (c) standardised median profile: what each stratum is, in one glance. Medians
    # are z-scored across strata per feature, so the colour says "high or low FOR
    # THIS FEATURE" rather than being dominated by whichever feature has big units.
    ax = axes[1, 0]
    feats = [f for f in PROFILE if f in dk.columns]
    med = dk.groupby(labels)[feats].median()
    zs = ((med - med.mean()) / med.std().replace(0, np.nan)).T
    im = ax.imshow(zs.to_numpy(), cmap="RdBu_r", vmin=-2, vmax=2, aspect="auto")
    ax.set_xticks(range(k)); ax.set_xticklabels(range(k))
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels([LABELS.get(f, f) for f in feats], fontsize=6.4 * s)
    ax.set_xlabel("Stratum")
    ax.set_title("(c) median profile, $z$ across strata", loc="left", fontsize=9 * s, pad=3)
    cb = fig.colorbar(im, ax=ax, pad=0.02, fraction=0.046)
    cb.ax.tick_params(labelsize=6 * s)

    # (d) reconstruction error is the sharpest axis, so it gets its own panel
    ax = axes[1, 1]
    for c in range(k):
        v = dk.loc[labels == c, "recon_error"].to_numpy()
        ax.hist(v, bins=np.linspace(0, dk["recon_error"].quantile(0.995), 40),
                histtype="step", lw=1.1, color=cmap(c), label=f"{c}")
    ax.set_xlabel("Reconstruction error")
    ax.set_ylabel("Events")
    ax.set_yscale("log")
    ax.legend(fontsize=6.4 * s, ncol=2, frameon=True, framealpha=0.85)
    ax.set_title("(d) model difficulty by stratum", loc="left", fontsize=9 * s, pad=3)

    for a in axes.ravel():
        a.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig(fig, out_dir, "kaon_substructure")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=5,
                    help="components for the reported stratification (default 5). "
                         "Specified, not discovered — see the docstring.")
    ap.add_argument("--scan-k", type=int, nargs=2, default=[2, 10], metavar=("LO", "HI"))
    ap.add_argument("--n", type=int, default=16, help="example events per stratum")
    ap.add_argument("--min-pixels", type=int, default=1000,
                    help="n_pixels below this flags an event 'small' (default 1000, "
                         "about the 2nd percentile of the kaon-tagged sample)")
    ap.add_argument("--junk-re-quantile", type=float, default=0.99,
                    help="quantile of recon_error and of ||z|| above which events are "
                         "flagged 'hard' and 'far' (default 0.99)")
    ap.add_argument("--plane", choices=list(PLANE_INDEX), default="collection")
    ap.add_argument("--scale", choices=["log1p", "linear"], default="linear")
    ap.add_argument("--no-images", action="store_true", help="skip the example grids")
    ap.add_argument("--no-umap", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(figure_dir(cfg, "kaon-substructure")))
    out_dir.mkdir(parents=True, exist_ok=True)

    is_kaon = (df["species"] == "kaon").to_numpy()
    Zk = Z[is_kaon]
    dk = df[is_kaon].reset_index(drop=True)
    print(f"kaon-tagged sample: {len(Zk)} events, {Zk.shape[1]}D latent")

    scan = scan_k(Zk, *args.scan_k, seed=args.seed)
    print("\nk-scan — BIC falling with no elbow and a low silhouette at every k means "
          "a continuum, not clusters:")
    print(scan.round(3).to_string(index=False))

    gmm = GaussianMixture(args.k, covariance_type="full", n_init=20,
                          random_state=args.seed).fit(Zk)
    labels = gmm.predict(Zk)
    dk["stratum"] = labels

    flags, cuts = flag_junk(dk, Zk, args.min_pixels, args.junk_re_quantile)
    dk = pd.concat([dk, flags], axis=1)
    print(f"\njunk flags (cuts: n_pixels<{cuts['min_pixels']}, "
          f"recon_error>{cuts['recon_error_cut']:.3f}, "
          f"||z||>{cuts['latent_norm_cut']:.2f}):")
    print(f"  small {int(flags['small'].sum()):5d}   hard {int(flags['hard'].sum()):5d}   "
          f"far {int(flags['far'].sum()):5d}   any {int(flags['any_flag'].sum()):5d} "
          f"({flags['any_flag'].mean():.1%})")

    feats = [f for f in PROFILE if f in dk.columns]
    prof = dk.groupby("stratum")[feats].median()
    prof.insert(0, "n", dk.groupby("stratum").size())
    for f in ("small", "hard", "far"):
        prof[f"{f}_frac"] = dk.groupby("stratum")[f].mean().round(3)
    print(f"\nper-stratum medians at k={args.k}:")
    print(prof.round(3).to_string())

    emb = None
    if not args.no_umap:
        emb = load_embedding(cfg, Z)[is_kaon]
    plot_overview(scan, emb, dk, labels, args.k, out_dir)

    if not args.no_images:
        print(f"\nloading {args.plane} plane from {IMAGE_256.name} (mmap'd)")
        k_images = torch.load(IMAGE_256, map_location="cpu", weights_only=False,
                              mmap=True)["k"]
        plane = PLANE_INDEX[args.plane]
        # Evenly spaced by recon error within the stratum, so the examples span what
        # the stratum contains instead of clustering at one end of it.
        picks, imgs = {}, {}
        for c in range(args.k):
            rows = np.flatnonzero(labels == c)
            rows = rows[np.argsort(dk["recon_error"].to_numpy()[rows])]
            take = np.linspace(0, len(rows) - 1, min(args.n, len(rows))).astype(int)
            picks[c] = rows[take]
            imgs[c] = [k_images[int(i)][plane].numpy() for i in picks[c]]
        windows = group_windows(imgs)
        transform = np.log1p if args.scale == "log1p" else (lambda a: a)
        cbar = (r"$\log(1 + \mathrm{ADC})$, " if args.scale == "log1p"
                else "ADC counts, ") + f"{args.plane} plane"
        allpx = np.concatenate([transform(i).ravel() for g in imgs.values() for i in g])
        vmax = float(np.percentile(allpx, 99.9))
        for c in range(args.k):
            lab = [f"RE = {dk['recon_error'].iloc[i]:.2f}\n"
                   f"$S$ = {dk['solidity'].iloc[i]:.2f}" for i in picks[c]]
            title = (f"Kaon stratum {c} (n={int((labels == c).sum())}, "
                     f"median RE={prof.loc[c, 'recon_error']:.3f}, "
                     f"$S$={prof.loc[c, 'solidity']:.2f})")
            plot_grid(imgs[c], lab, windows[c], vmax, title, out_dir,
                      f"kaon_stratum_{c:02d}" + ("" if args.scale == "log1p" else "_adc"),
                      cbar_label=cbar, transform=transform)

    dk.to_csv(out_dir / "kaon_substructure.csv", index=False)
    with open(out_dir / "metrics.json", "w") as fh:
        json.dump({"n_events": int(len(Zk)), "k": args.k,
                   "k_scan": scan.to_dict(orient="records"),
                   "junk_cuts": cuts,
                   "flag_counts": {c: int(flags[c].sum())
                                   for c in ("small", "hard", "far", "any_flag")},
                   "profiles": prof.round(4).to_dict()}, fh, indent=2)
    print(f"\n  saved {out_dir}/kaon_substructure.csv and metrics.json")


if __name__ == "__main__":
    main()
