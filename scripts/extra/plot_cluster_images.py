#!/usr/bin/env python3
"""
scripts/extra/plot_cluster_images.py

Raw event displays for the members of one mixture cluster.

WHY THIS EXISTS
    The k sweep says which clusters are pure and which are mixed, but a mixed
    cluster is only interesting once you can see what is in it. Cluster 34 at
    k=37, for instance, is 33% proton / 29% kaon / 38% MIP by beam tag and is
    where the beam and anchored references disagree most -- a table cannot say
    whether that is a genuine overlap population or a failure of the fit.

    --source raw256 (default) draws the 256x256 RAW images, where the tracks are
    legible. --source model draws exactly what the VAE was fed: the 48x48
    two-plane tensor from data.path with data.transform applied, built through
    the same prepare_images call training uses. Nothing is cropped in that mode
    -- the frame IS the model's field of view, and cropping it would show
    something the model never saw.

    The model reads both wire planes as two channels; --plane picks which one is
    displayed.

ROW MAPPING IS THE FIDDLY PART
    load_beam_data stacks train.npz + val.npz + kaon.npz + muon.npz, so its
    proton block is [p_train_idx, p_val_idx] -- a SCATTERED subset of the
    natural proton order the image tensor uses. Kaons and MIPs are contiguous
    and in natural order, so they need only an offset. Getting this wrong shows
    the wrong events with no error, so the mapping is asserted against the
    species column before anything is drawn.

    MIP images live in a separate file (muon_256x256_raw.pt); pk_256x256 holds
    only protons and kaons.

SAMPLING
    Events are drawn stratified by beam tag in proportion to the cluster's own
    composition, so a mixed cluster shows a mixed grid rather than whichever
    species happens to sort first. Panels are labelled with the beam tag.

OUTPUTS (under figs/<model_name>/kscan/cluster_images/)
    cluster_k<k>_c<cluster>_<scale>.{png,pdf}

Usage:
    python scripts/extra/plot_cluster_images.py --config configs/run_0093_*.yaml \\
        --k 37 --cluster 34
    python scripts/extra/plot_cluster_images.py --config ... --k 37 --cluster 34 \\
        --scale linear --n 24
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _beam_data import (COLOURS, DISPLAY, DOUBLE_COL, SINGLE_COL, SPECIES,
                        apply_style, build_model_name, figure_dir,
                        load_beam_data, load_config, savefig)
from cluster_k_sweep import labels_for

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.transforms import prepare_images  # noqa: E402

PK_IMAGES = Path("/Volumes/easystore/proton-kaon/images/pk_256x256_raw_10-179wires.pt")
MIP_IMAGES = Path("/Volumes/easystore/proton-kaon/images/muon_256x256_raw.pt")
PLANE_INDEX = {"collection": 0, "induction": 1}


def row_maps(cfg, df):
    """Latent-row index -> (species, index within that species' image tensor)."""
    inf = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    ss = np.load(inf / "species_split.npz")
    n_p = int((df["species"] == "proton").sum())
    n_k = int((df["species"] == "kaon").sum())
    proton_rows = np.concatenate([ss["p_train_idx"], ss["p_val_idx"]])
    assert len(proton_rows) == n_p, (len(proton_rows), n_p)

    def to_image_index(row):
        if row < n_p:
            return "proton", int(proton_rows[row])
        if row < n_p + n_k:
            return "kaon", int(row - n_p)
        return "muon", int(row - n_p - n_k)

    # the mapping must reproduce the species column, or it is silently wrong
    for probe in (0, n_p - 1, n_p, n_p + n_k - 1, n_p + n_k, len(df) - 1):
        assert to_image_index(probe)[0] == df["species"].iloc[probe], probe
    return to_image_index


def sample_stratified(rows, species, n, seed=0):
    """Draw n events in proportion to the cluster's own species mix."""
    rng = np.random.default_rng(seed)
    picks = []
    present = [s for s in SPECIES if (species[rows] == s).any()]
    for i, s in enumerate(present):
        pool = rows[species[rows] == s]
        # largest-remainder so the counts sum to n
        want = int(round(n * len(pool) / len(rows)))
        want = min(max(want, 1), len(pool))
        picks.append(rng.choice(pool, want, replace=False))
    out = np.concatenate(picks)
    if len(out) > n:
        out = rng.choice(out, n, replace=False)
    return np.sort(out)


def window(imgs, pad=6, pct=(2, 98)):
    """One crop window shared by the whole grid.

    Not a per-image bounding box: that silently rescales each event, and track
    length is exactly what distinguishes these populations.
    """
    rows, cols = [], []
    for im in imgs:
        nz = np.argwhere(im > 0.01)
        if len(nz):
            rows.append((nz[:, 0].min(), nz[:, 0].max()))
            cols.append((nz[:, 1].min(), nz[:, 1].max()))
    r0 = max(0, min(r[0] for r in rows) - pad)
    r1 = min(imgs[0].shape[0], max(r[1] for r in rows) + pad)
    c0 = max(0, min(c[0] for c in cols) - pad)
    c1 = min(imgs[0].shape[1], max(c[1] for c in cols) + pad)
    return slice(int(r0), int(r1)), slice(int(c0), int(c1))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--cluster", type=int, required=True)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--ncol", type=int, default=4)
    ap.add_argument("--plane", choices=list(PLANE_INDEX), default="collection")
    ap.add_argument("--source", choices=["raw256", "model"], default="raw256",
                    help="raw256: 256x256 raw displays. model: the 48x48 "
                         "transformed tensor the VAE was actually fed.")
    ap.add_argument("--no-transform", action="store_true",
                    help="--source model only: skip data.transform, showing the "
                         "48x48 in raw ADC. Same geometry the model sees, before "
                         "log1p compresses the dynamic range.")
    ap.add_argument("--scale", choices=["log1p", "linear"], default="log1p")
    ap.add_argument("--vmax-pct", type=float, default=99.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    species = df["species"].to_numpy()
    cache = Path(cfg["output"]["inference_dir"]) / build_model_name(cfg)
    lab = labels_for(Z, args.k, args.seed, cache)
    rows = np.flatnonzero(lab == args.cluster)
    if len(rows) == 0:
        raise SystemExit(f"cluster {args.cluster} is empty at k={args.k}")
    comp = {s: float((species[rows] == s).mean()) for s in SPECIES}
    print(f"k={args.k} cluster {args.cluster}: {len(rows)} events  "
          + "  ".join(f"{DISPLAY[s]} {comp[s]:.1%}" for s in SPECIES), flush=True)

    picks = sample_stratified(rows, species, args.n, args.seed)
    to_image = row_maps(cfg, df)
    plane = PLANE_INDEX[args.plane]

    need = {to_image(int(r))[0] for r in picks}
    store = {}
    if args.source == "model":
        tx = "none" if args.no_transform else cfg["data"].get("transform", "none")
        hw = tuple(cfg["model"]["input_hw"])
        print(f"  loading the model's own input: {Path(cfg['data']['path']).name}, "
              f"transform={tx}, {hw[0]}x{hw[1]}", flush=True)
        raw = torch.load(cfg["data"]["path"], map_location="cpu")
        for sp, key in (("proton", "p"), ("kaon", "k"), ("muon", "m")):
            if sp in need:
                store[sp] = prepare_images(raw[key], tx, hw)
    else:
        print(f"  loading {args.plane} plane (mmap'd) for: {', '.join(sorted(need))}",
              flush=True)
        pk = torch.load(PK_IMAGES, map_location="cpu", weights_only=False, mmap=True)
        store["proton"], store["kaon"] = pk["p"], pk["k"]
        if "muon" in need:
            m = torch.load(MIP_IMAGES, map_location="cpu", weights_only=False,
                           mmap=True)
            store["muon"] = m["m"] if isinstance(m, dict) and "m" in m else m

    imgs, tags = [], []
    for r in picks:
        sp, idx = to_image(int(r))
        imgs.append(store[sp][idx][plane].numpy().astype(float))
        tags.append(sp)

    if args.source == "model":
        # already transformed by prepare_images, and the 48x48 frame is the
        # model's whole field of view, so it is shown uncropped and unscaled
        shown = imgs
        win_r = win_c = slice(None)
    else:
        tf = np.log1p if args.scale == "log1p" else (lambda a: a)
        shown = [tf(im) for im in imgs]
        win_r, win_c = window(imgs)
    vmax = float(np.percentile(np.concatenate([im[win_r, win_c].ravel()
                                               for im in shown]), args.vmax_pct))

    nrow = int(np.ceil(len(shown) / args.ncol))
    scale = apply_style(SINGLE_COL)
    fig, axes = plt.subplots(nrow, args.ncol,
                             figsize=(DOUBLE_COL, DOUBLE_COL / args.ncol * nrow * 1.12),
                             squeeze=False)
    for i, ax in enumerate(axes.ravel()):
        if i >= len(shown):
            ax.axis("off")
            continue
        ax.imshow(shown[i][win_r, win_c], cmap="magma", vmin=0, vmax=vmax,
                  aspect="auto", interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(DISPLAY[tags[i]], fontsize=6.5 * scale, pad=1.5,
                     color=COLOURS[tags[i]])
        for sp_ in ax.spines.values():
            sp_.set_edgecolor(COLOURS[tags[i]])
            sp_.set_linewidth(0.9 * scale)
    view = ("model geometry, 48$\\times$48, raw ADC"
            if args.source == "model" and args.no_transform
            else "model input, 48$\\times$48, log1p" if args.source == "model"
            else f"256$\\times$256 raw, {args.scale}")
    fig.suptitle(f"$k$ = {args.k}, cluster {args.cluster} — {len(rows)} events  "
                 + "/ ".join(f"{DISPLAY[s]} {comp[s]:.0%}" for s in SPECIES)
                 + f"   [{view}, {args.plane}]",
                 fontsize=8.5 * scale, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    out_dir = (Path(args.out_dir) if args.out_dir
               else Path(figure_dir(cfg, "kscan")) / "cluster_images")
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = (("model_rawadc" if args.no_transform else "model")
           if args.source == "model" else args.scale)
    savefig(fig, out_dir, f"cluster_k{args.k}_c{args.cluster}_{tag}")


if __name__ == "__main__":
    main()
