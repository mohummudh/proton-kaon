#!/usr/bin/env python3
"""
scripts/extra/plot_latent_territories.py

Where do kaon-tagged events sit relative to the two clean samples, and what do
the ones in "foreign" territory actually look like?

TERRITORY
    For every event, take its 50 nearest neighbours in the 8D latent (samples
    subsampled to equal size first, so neighbour counts are not just a
    reflection of sample sizes) and record which beam tag dominates. A
    kaon-tagged event whose neighbourhood is mostly protons is in "proton
    territory"; mostly MIPs, "MIP territory"; otherwise it is bulk.

    This is deliberately model-free -- no mixture fit, no density estimate --
    so it reads the same structure the UMAP shows by eye.

WHAT THE PANELS ARE FOR
    Proton territory and MIP territory look superficially alike in the UMAP,
    but the physics says they are different things:

      proton territory  ~22% of kaon-tagged events. Their spectrometer mass is
                        pushed ~100 MeV toward the proton side of the window,
                        and every proxy shifts proton-ward. Consistent with
                        genuine proton contamination.
      MIP territory     ~10% of kaon-tagged events. Their mass is statistically
                        indistinguishable from the kaon bulk (~+10 MeV), so
                        they are NOT pion contamination. They are longer than
                        typical kaons with more local maxima -- consistent with
                        kaons that decayed in flight, where the imaged last 50
                        wires are the daughter muon.

    So latent proximity to a cluster is not the same as being that species.
    The mass panel is what separates the two readings, which is why it belongs
    in the same figure as the proxies.

OUTPUTS (under figs/<model_name>/territories/)
    territories.{png,pdf}   UMAP (all species | kaon by territory) plus
                            beamline mass, both proxies, and reconstruction
                            error broken down by territory
    metrics.json            counts and per-territory medians

Usage:
    python scripts/extra/plot_latent_territories.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_latent_territories.py --config ... --n-neighbours 100
"""

import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, PDG_MASS, PROXY_LABELS,
                        SINGLE_COL, SPECIES, apply_style, figure_dir,
                        load_beam_data, load_config, load_embedding, savefig)

BULK = "bulk"
TERRITORY_COLOUR = {BULK: "0.65", "proton": COLOURS["proton"], "muon": COLOURS["muon"]}
TERRITORY_LABEL = {BULK: "kaon: bulk", "proton": "in proton territory",
                   "muon": "in MIP territory"}


def neighbour_fractions(Z, species, n_neighbours=50, seed=0):
    """Fraction of each event's nearest neighbours carrying each beam tag.

    Samples are subsampled to a common size before building the neighbour
    graph, otherwise a larger sample wins neighbours simply by being larger.
    Events outside the subsample are still scored, against the balanced graph.
    """
    rng = np.random.default_rng(seed)
    counts = species.value_counts()
    n_each = int(counts.min())
    reference = np.concatenate([
        rng.choice(np.flatnonzero((species == s).to_numpy()), n_each, replace=False)
        for s in SPECIES
    ])

    scaler = StandardScaler().fit(Z[reference])
    graph = NearestNeighbors(n_neighbors=n_neighbours).fit(scaler.transform(Z[reference]))
    _, idx = graph.kneighbors(scaler.transform(Z))
    labels = pd.Categorical(species.to_numpy()[reference], categories=SPECIES).codes[idx]
    return {s: (labels == i).mean(axis=1) for i, s in enumerate(SPECIES)}


def assign_territory(fractions, threshold=0.5):
    """Label each event by which foreign sample dominates its neighbourhood."""
    territory = np.full(len(fractions["proton"]), BULK, dtype=object)
    territory[fractions["proton"] > threshold] = "proton"
    territory[fractions["muon"] > threshold] = "muon"
    return territory


def _scatter_umap(ax, embedding, groups, order, colours, labels, scale, sizes):
    for key in order:
        sel = groups == key
        ax.scatter(embedding[sel, 0], embedding[sel, 1], s=sizes[key], c=colours[key],
                   alpha=0.30 if sizes[key] < 1.0 else 0.55, lw=0,
                   label=f"{labels[key]}" if labels else None, rasterized=True)
    legend = ax.legend(markerscale=9, handletextpad=0.2, borderpad=0.3,
                       loc="lower left", fontsize=6.2 * scale)
    for handle in legend.legend_handles:
        handle.set_alpha(1)
    ax.set_xticks([]); ax.set_yticks([])


def _hist_by_territory(ax, df, column, territory, edges, scale, xlabel, logx=False):
    for key in [BULK, "muon", "proton"]:
        vals = df.loc[territory == key, column].to_numpy()
        vals = vals[np.isfinite(vals)]
        if len(vals) < 10:
            continue
        counts, _ = np.histogram(vals, edges)
        ax.stairs(counts / counts.sum(), edges, color=TERRITORY_COLOUR[key],
                  lw=1.1 * scale, label=TERRITORY_LABEL[key])
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Fraction / bin")
    if logx:
        ax.set_xscale("log")


def plot_territories(embedding, df, territory, out_dir):
    s = apply_style(SINGLE_COL)  # each PANEL is single-column wide
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE_COL, DOUBLE_COL * 0.66))
    is_kaon = (df["species"] == "kaon").to_numpy()

    _scatter_umap(axes[0, 0], embedding, df["species"].to_numpy(),
                  ["proton", "muon", "kaon"], COLOURS, DISPLAY, s,
                  {k: 0.5 for k in SPECIES})
    axes[0, 0].set_title("(a) All species", loc="left", fontsize=8.5 * s, pad=3 * s)
    axes[0, 0].set_ylabel("UMAP 2")

    counts = pd.Series(territory[is_kaon]).value_counts()
    labels = {k: f"{TERRITORY_LABEL[k]} ({counts.get(k, 0)})" for k in TERRITORY_COLOUR}
    _scatter_umap(axes[0, 1], embedding[is_kaon], territory[is_kaon],
                  [BULK, "proton", "muon"], TERRITORY_COLOUR, labels, s,
                  {BULK: 0.5, "proton": 1.4, "muon": 1.4})
    axes[0, 1].set_title("(b) Kaon-tagged only", loc="left", fontsize=8.5 * s, pad=3 * s)
    for ax in axes[0, :2]:
        ax.set_xlabel("UMAP 1")

    kaon_df = df.loc[is_kaon].reset_index(drop=True)
    kaon_territory = territory[is_kaon]

    _hist_by_territory(axes[0, 2], kaon_df, "beamline_mass", kaon_territory,
                       np.linspace(350, 650, 26), s, "Beamline mass [MeV/$c^2$]")
    axes[0, 2].axvline(PDG_MASS["kaon"], ls="--", lw=0.7 * s, color="0.35")
    axes[0, 2].set_title("(c) Spectrometer mass", loc="left", fontsize=8.5 * s, pad=3 * s)
    axes[0, 2].legend(loc="upper right", fontsize=6.2 * s)

    for ax, letter, column in zip(axes[1, :2], "de", PROXY_LABELS):
        vals = kaon_df[column].to_numpy()
        upper = np.nanpercentile(vals, 99)
        _hist_by_territory(ax, kaon_df, column, kaon_territory,
                           np.linspace(np.nanmin(vals), upper, 36), s,
                           PROXY_LABELS[column])
        ax.set_title(f"({letter}) {PROXY_LABELS[column]}", loc="left",
                     fontsize=8.5 * s, pad=3 * s)
    axes[1, 0].legend(loc="upper right", fontsize=6.2 * s)

    upper = np.nanpercentile(kaon_df["recon_error"], 99)
    _hist_by_territory(axes[1, 2], kaon_df, "recon_error", kaon_territory,
                       np.linspace(0, upper, 36), s, "Reconstruction error")
    axes[1, 2].set_title("(f) Reconstruction error", loc="left", fontsize=8.5 * s, pad=3 * s)

    fig.tight_layout()
    savefig(fig, out_dir, "territories")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--features-pkl", default=None)
    ap.add_argument("--picky-csv", default=None)
    ap.add_argument("--n-neighbours", type=int, default=50)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="neighbour fraction above which an event is in that territory")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    kwargs = {k: v for k, v in [("features_path", args.features_pkl),
                                ("picky_path", args.picky_csv)] if v}
    Z, df = load_beam_data(cfg, **kwargs)
    embedding = load_embedding(cfg, Z)
    out_dir = args.out_dir or figure_dir(cfg, "territories")

    fractions = neighbour_fractions(Z, df["species"], args.n_neighbours, args.seed)
    territory = assign_territory(fractions, args.threshold)

    print(f"\nneighbourhood composition ({args.n_neighbours} nearest neighbours, balanced):")
    table = pd.DataFrame({DISPLAY[s]: [fractions[s][(df["species"] == r).to_numpy()].mean()
                                       for r in SPECIES] for s in SPECIES},
                         index=[f"{DISPLAY[r]}-tagged" for r in SPECIES])
    print(table.round(3).to_string())

    is_kaon = (df["species"] == "kaon").to_numpy()
    counts = pd.Series(territory[is_kaon]).value_counts()
    print(f"\nkaon-tagged events by territory (of {is_kaon.sum()}):")
    for key in [BULK, "proton", "muon"]:
        n = int(counts.get(key, 0))
        print(f"  {TERRITORY_LABEL[key]:24s} {n:5d}  ({n / is_kaon.sum():5.1%})")
    for other in ("proton", "muon"):
        sel = (df["species"] == other).to_numpy()
        n = int((territory[sel] == other).sum())
        print(f"  (for scale, {DISPLAY[other]}-tagged in its own territory: {n / sel.sum():.1%})")

    kaon_df = df.loc[is_kaon].reset_index(drop=True)
    kaon_df["territory"] = territory[is_kaon]
    columns = list(PROXY_LABELS) + ["beamline_mass", "height", "median_adc",
                                    "n_local_maxima", "recon_error"]
    medians = kaon_df.groupby("territory")[columns].median()
    print("\nper-territory medians (kaon-tagged only):")
    print(medians.round(3).to_string())
    print("\nclean-sample medians for reference:")
    print(df[df["species"] != "kaon"].groupby("species")[columns].median().round(3).to_string())

    bulk_mass = kaon_df.loc[kaon_df["territory"] == BULK, "beamline_mass"].dropna()
    print("\nmass shift vs the kaon bulk -- the panel that separates the two readings:")
    shifts = {}
    for key in ("proton", "muon"):
        vals = kaon_df.loc[kaon_df["territory"] == key, "beamline_mass"].dropna()
        shift = float(vals.median() - bulk_mass.median())
        pval = float(stats.mannwhitneyu(vals, bulk_mass).pvalue)
        shifts[key] = {"shift_mev": shift, "pvalue": pval, "n": int(len(vals))}
        print(f"  {TERRITORY_LABEL[key]:24s} {shift:+7.1f} MeV   (p={pval:.1e})")

    plot_territories(embedding, df, territory, out_dir)

    with open(f"{out_dir}/metrics.json", "w") as fh:
        json.dump({
            "n_neighbours": args.n_neighbours, "threshold": args.threshold,
            "neighbour_composition": table.round(4).to_dict(),
            "kaon_territory_counts": {k: int(counts.get(k, 0)) for k in TERRITORY_COLOUR},
            "territory_medians": medians.round(4).to_dict(),
            "mass_shift_vs_bulk": shifts,
        }, fh, indent=2)
    print(f"  saved {out_dir}/metrics.json")


if __name__ == "__main__":
    main()
