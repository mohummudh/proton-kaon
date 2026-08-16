#!/usr/bin/env python3
"""
scripts/extra/_beam_data.py

Shared loading, alignment and plot styling for the beam-composition scripts:

    cluster_latents.py     unsupervised GMM clustering of the latent space
    plot_dose_response.py  latent proton-likeness vs spectrometer mass
    fit_kaon_peak.py       three-component fit to the beamline mass spectrum

Not a standalone script -- imported by the three above so the (fragile)
latent<->feature alignment lives in exactly one place.

ALIGNMENT
    Latents are stored per species and concatenated as
        Z = vstack([train.npz, val.npz, kaon.npz, muon.npz])
    where train/val are PROTONS ONLY, and kaon/muon each hold that species'
    train+val together. Features are matched positionally, so the proton block
    must be reordered by species_split.npz's p_train_idx / p_val_idx before it
    lines up. Getting this wrong mispairs rows *without* raising a shape error,
    which is why it is centralised here and asserted on every load.

BEAMLINE COLUMNS (docs/picky+match.csv)
    beamline_mass  spectrometer mass in MeV/c^2 -- the species tags ARE mass
                   windows on this variable, so it cannot validate the tags,
                   only the contamination within a window.
    p              the "picky" beamline-reconstruction quality flag, NOT
                   momentum. Protons and MIPs in this dataset are 100% p=1;
                   kaons are only ~14.5% p=1. Any proton-vs-kaon comparison is
                   therefore quality-mismatched unless restricted to p=1.
    m              a second beamline flag; unused here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from src.train.naming import model_name as build_model_name

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_FEATURES = "/Volumes/easystore/proton-kaon/features/features.pkl"
DEFAULT_PICKY = "/Volumes/easystore/proton-kaon/docs/picky+match.csv"

# Paul Tol palette, matching plot_umap_all.py / plot_proxy_hists.py.
COLOURS = {"proton": "#0077BB", "kaon": "#EE7733", "muon": "#AA3377"}
DISPLAY = {"proton": "Proton", "kaon": "Kaon", "muon": "MIPs"}
SPECIES = ["proton", "kaon", "muon"]

PDG_MASS = {"proton": 938.272, "kaon": 493.677, "muon": 139.570}  # muon slot = pi+

# The two interpretable proxies used throughout the paper.
PROXY_LABELS = {"mean_adc": "Calorimetry Proxy", "solidity": "Topology Proxy"}

SINGLE_COL = 3.375  # inches, ~86 mm
DOUBLE_COL = 6.875  # inches, ~175 mm
DPI = 300


def apply_style(fig_w=SINGLE_COL):
    """Publication rcParams scaled to figure width; returns the scale factor.

    Text is specified at 9 pt on a SINGLE_COL figure, so a wider figure scales
    everything by fig_w / SINGLE_COL to keep the same relative look. For a
    multi-panel figure, pass the width of ONE PANEL, not the whole figure.
    """
    s = fig_w / SINGLE_COL
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9 * s,
        "axes.labelsize": 9 * s,
        "xtick.labelsize": 8 * s,
        "ytick.labelsize": 8 * s,
        "legend.fontsize": 8 * s,
        "axes.linewidth": 0.6 * s,
        "xtick.major.width": 0.6 * s,
        "ytick.major.width": 0.6 * s,
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })
    return s


def savefig(fig, out_dir, stem):
    """Write <stem>.png and <stem>.pdf under out_dir; return the .png path."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{stem}.png"
    fig.savefig(png)
    fig.savefig(out_dir / f"{stem}.pdf")
    plt.close(fig)
    print(f"  saved {png}")
    return png


def load_config(path):
    with open(path) as fh:
        return yaml.safe_load(fh)


def load_beam_data(cfg, features_path=DEFAULT_FEATURES, picky_path=DEFAULT_PICKY):
    """Load latents, physics features and beamline columns, all row-aligned.

    Returns (Z, df) where Z is (N, latent_dim) and df has N rows carrying the
    22 physics features plus `species`, `beamline_mass`, `picky`, and
    `recon_error`. Row i of Z and row i of df are the same event.

    Requires an all-species model (data.proton == "all"); the beam-composition
    analyses are only defined when protons, kaons and MIPs share a latent space.
    """
    if cfg["data"].get("proton") != "all":
        raise ValueError(
            "load_beam_data requires an all-species model (data.proton: all); "
            f"this config has data.proton = {cfg['data'].get('proton')!r}."
        )

    name = build_model_name(cfg)
    inference_dir = Path(cfg["output"]["inference_dir"]) / name

    npz = {k: np.load(inference_dir / f"{k}.npz") for k in ("train", "val", "kaon", "muon")}
    Z = np.vstack([npz[k]["latents"] for k in ("train", "val", "kaon", "muon")]).astype(float)
    recon_error = np.concatenate([npz[k]["re"] for k in ("train", "val", "kaon", "muon")])
    n_proton = len(npz["train"]["latents"]) + len(npz["val"]["latents"])
    n_kaon, n_muon = len(npz["kaon"]["latents"]), len(npz["muon"]["latents"])

    # Reassemble proton features into the stored train-then-val latent order.
    split = np.load(inference_dir / "species_split.npz")
    features = pd.read_pickle(features_path)
    by_species = {s: features[features["particle_type"] == s].reset_index(drop=True)
                  for s in SPECIES}
    proton_order = np.concatenate([split["p_train_idx"], split["p_val_idx"]])
    df = pd.concat(
        [by_species["proton"].iloc[proton_order],
         by_species["kaon"],
         by_species["muon"]],
        ignore_index=True,
    )

    if len(df) != len(Z):
        raise ValueError(
            f"features ({len(df)}) and latents ({len(Z)}) are not row-aligned. "
            f"Expected {n_proton} proton + {n_kaon} kaon + {n_muon} muon rows."
        )

    df["species"] = np.repeat(SPECIES, [n_proton, n_kaon, n_muon])
    df["recon_error"] = recon_error

    picky = pd.read_csv(picky_path).drop_duplicates(["run", "subrun", "event"])
    df = df.merge(
        picky[["run", "subrun", "event", "beamline_mass", "p"]],
        on=["run", "subrun", "event"], how="left",
    ).rename(columns={"p": "picky"})

    if len(df) != len(Z):
        raise ValueError("beamline merge changed the row count -- duplicate event keys?")

    print(f"  loaded {len(Z)} events x {Z.shape[1]}D latent from {inference_dir.name}")
    for s in SPECIES:
        sel = df["species"] == s
        n_picky = int((df.loc[sel, "picky"] == 1).sum())
        print(f"    {DISPLAY[s]:7s} {int(sel.sum()):6d}   picky (p=1): {n_picky:6d} "
              f"({n_picky / max(int(sel.sum()), 1):5.1%})")
    return Z, df


def select(Z, df, picky=None, species=None):
    """Subset both arrays together.

    picky   1 keeps only p=1, 0 keeps only p=0, None keeps everything.
    species a species name or list of names; None keeps everything.
    """
    mask = np.ones(len(df), dtype=bool)
    if picky is not None:
        mask &= (df["picky"] == picky).to_numpy()
    if species is not None:
        wanted = [species] if isinstance(species, str) else list(species)
        mask &= df["species"].isin(wanted).to_numpy()
    return Z[mask], df.loc[mask].reset_index(drop=True)


def figure_dir(cfg, subdir):
    """figs/<model_name>/<subdir>/, created if needed."""
    out = PROJECT_ROOT / "figs" / build_model_name(cfg) / subdir
    out.mkdir(parents=True, exist_ok=True)
    return out


def load_embedding(cfg, Z):
    """2D UMAP of the latent space, row-aligned with Z.

    Reuses the reducer cached by plot_umap_all.py so every UMAP figure in the
    project shares one projection; fits and caches one if it is missing. Pass
    the FULL Z (before any picky/species filtering) so the stored embedding can
    be used directly, then subset the result alongside the dataframe.
    """
    import pickle

    reducer_path = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg) / "reducer.pkl"
    if reducer_path.exists():
        with open(reducer_path, "rb") as fh:
            reducer = pickle.load(fh)
        print(f"  loaded UMAP reducer from {reducer_path.name}")
    else:
        import umap
        print("  fitting a new UMAP reducer (no cached reducer.pkl)...")
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42).fit(Z)
        with open(reducer_path, "wb") as fh:
            pickle.dump(reducer, fh)

    stored = getattr(reducer, "embedding_", None)
    if stored is not None and len(stored) == len(Z):
        return stored
    return reducer.transform(Z)
