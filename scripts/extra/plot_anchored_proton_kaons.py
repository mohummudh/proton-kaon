#!/usr/bin/env python3
"""
scripts/extra/plot_anchored_proton_kaons.py

The kaon-tagged events that anchored clustering assigns to the PROTON component,
shown as detector images. Two grids: the most proton-like, and the least
proton-like that were still assigned to proton.

WHAT THIS IS FOR
    anchored_clustering.py estimates how much of the kaon-tagged sample is really
    proton, but it reports that as a number. This shows the events themselves, which
    is the check a referee will actually want: if the events the method calls
    contaminating protons look like protons to the eye — one clean track, a sharp
    Bragg peak at the end, no kink or decay products — the estimate is believable in
    a way no ARI can make it.

    The second grid is the more informative one. The events sitting just inside the
    decision boundary are where the method is weakest, so showing them is what
    stops this from being a curated best-case gallery. If those look like plausible
    kaons rather than protons, the contamination estimate is biased high, and the
    figure says so.

RANKING
    Events are ranked by the decision MARGIN, log q_proton - max(log q_kaon,
    log q_muon), each weighted by its mixture prior. That is exactly the quantity
    whose sign decides the assignment, so "most proton-like" is the largest margin
    and "least proton-like but still assigned" is the smallest positive one. The
    posterior probability is printed on each panel as the readable version.

IMAGES
    The 256x256 raw tensor, not the 48x48 the VAE saw. The VAE's input is a
    bilinear downsample of this, so the 256 version is the same event at the
    resolution the detector actually recorded, and structure the network had to work
    without -- individual hits, the fine shape of a kink -- is visible.

    Kaon rows are verified row-identical between pkm_48x48 (what the model trained
    on) and pk_256x256, so index k in one is index k in the other. Loaded mmap'd:
    the file is 9.8 GB and only 32 rows are ever touched.

    All panels share one colour scale and one crop window, so cell-to-cell
    comparisons are meaningful. Per-image autoscaling would make every event look
    equally bright and destroy the comparison the figure exists for.

Usage:
    python scripts/extra/plot_anchored_proton_kaons.py --config configs/run_0093_*.yaml
    python scripts/extra/plot_anchored_proton_kaons.py --config ... --plane induction
    python scripts/extra/plot_anchored_proton_kaons.py --config ... --n 16 --picky 1
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from _beam_data import (DOUBLE_COL, SINGLE_COL, SPECIES, apply_style, figure_dir,
                        load_beam_data, load_config, savefig)
from anchored_clustering import fit_density, sharpen_kaon_density

IMAGE_256 = Path("/Volumes/easystore/proton-kaon/images/pk_256x256_raw_10-179wires.pt")
PLANE_INDEX = {"collection": 0, "induction": 1}


def anchored_log_joint(Z, species, n_anchor_comp=6, n_free_comp=6, n_iter=6, seed=0):
    """Per-event weighted log densities under the three anchored components.

    Mirrors anchored_clustering.anchored_fit but returns the full log_joint rather
    than only its argmax, because the margin is the thing being ranked here. The
    densities themselves come from the shared sharpen_kaon_density, so the partition
    is identical to the one anchored_clustering.py reports.
    """
    is_kaon = (species == "kaon").to_numpy()
    q_proton = fit_density(Z[(species == "proton").to_numpy()], n_anchor_comp, seed)
    q_muon = fit_density(Z[(species == "muon").to_numpy()], n_anchor_comp, seed)
    counts = np.array([(species == s).sum() for s in SPECIES], dtype=float)
    weights = counts / counts.sum()
    q_kaon = sharpen_kaon_density(Z[is_kaon], q_proton, q_muon, weights,
                                  n_free_comp, n_iter, seed)
    return np.column_stack([q_proton.score_samples(Z),
                            q_kaon.score_samples(Z),
                            q_muon.score_samples(Z)]) + np.log(weights)


def _boxes(images):
    out = []
    for img in images:
        nz = np.argwhere(img > 0.01)
        if len(nz):
            out.append((nz[:, 0].min(), nz[:, 0].max(), nz[:, 1].min(), nz[:, 1].max()))
    return np.array(out)


def group_windows(groups, pad=4, col_pct=(2, 98)):
    """Crop windows: drift axis shared across all grids, wire axis per grid.

    A per-image bounding box is not an option — it silently rescales each event, so a
    short track and a long one would be drawn the same size, and track LENGTH is
    exactly what separates a proton-like event from a kaon with decay products.

    But one window shared across all three grids does not work either, because the
    groups genuinely differ in size: the confidently-proton events reach row 70 and
    span ~32 wires, while every confidently-kaon event reaches row 252 and spans
    ~160. Forcing one window would draw the proton-like tracks as specks in a mostly
    empty frame.

    The split resolves it. The DRIFT axis is shared and kept full, so the
    short-versus-long comparison across grids survives and is read directly off the
    fraction of the panel a track fills. The WIRE axis is per grid and percentile-
    cropped, since it only controls how fat a track looks, not its length. Clipping
    is reported per grid.
    """
    all_boxes = np.vstack([_boxes(imgs) for imgs in groups.values()])
    n_r, n_c = next(iter(groups.values()))[0].shape
    r0 = max(0, all_boxes[:, 0].min() - pad)
    r1 = min(n_r, all_boxes[:, 1].max() + pad + 1)

    windows = {}
    for key, imgs in groups.items():
        b = _boxes(imgs)
        c0 = max(0, int(np.percentile(b[:, 2], col_pct[0])) - pad)
        c1 = min(n_c, int(np.percentile(b[:, 3], col_pct[1])) + pad + 1)
        clipped = int(((b[:, 2] < c0) | (b[:, 3] >= c1)).sum())
        note = f", {clipped} clipped in wire" if clipped else ""
        # str(key): callers key groups by name or by integer stratum, so do not
        # assume either.
        print(f"  {str(key):9s} window rows {r0}-{r1} (shared), wires {c0}-{c1}{note}")
        windows[key] = (r0, r1, c0, c1)
    return windows


def plot_grid(images, labels, window, vmax, title, out_dir, stem, ncol=4,
              cbar_label=r"$\log(1 + \mathrm{ADC})$, collection plane",
              transform=np.log1p):
    """Square grid of event images, one shared colour scale and colourbar.

    Labels go ABOVE each panel rather than inside it. There is no corner that is
    reliably empty across both grids -- tracks start at the left edge and run right,
    the diagonal ones reach the bottom-left, and others end bottom-right -- so any
    in-axes position overlaps the data for some event.
    """
    s = apply_style(SINGLE_COL)
    n = len(images)
    nrow = int(np.ceil(n / ncol))
    r0, r1, c0, c1 = window
    # Transposed on display so the long (drift) axis runs horizontally, which is
    # how the event displays elsewhere in the repo are oriented.
    cell_aspect = (r1 - r0) / (c1 - c0)
    cell_w = DOUBLE_COL / ncol
    # Floor the cell height. Deriving it from the aspect ratio alone collapses the
    # panels when the wire window is narrow -- the first kaon-likeness decile spans
    # only ~45 wires against 256 drift ticks, which left rows too short for their own
    # titles to fit and made them overlap the images.
    cell_h = max(cell_w / cell_aspect, 0.50)
    title_h, cbar_h = 0.20, 0.62                      # inches reserved
    fig_h = nrow * (cell_h + title_h) + cbar_h
    fig, axes = plt.subplots(nrow, ncol, figsize=(DOUBLE_COL, fig_h))
    axes = np.atleast_1d(axes).ravel()
    im = None
    for ax, img, lab in zip(axes, images, labels):
        im = ax.imshow(transform(img[r0:r1, c0:c1]).T, cmap="viridis", origin="lower",
                       vmin=0, vmax=vmax, interpolation="nearest", aspect="auto",
                       rasterized=True)
        ax.set_title(lab, fontsize=6.2 * s, pad=2.2 * s, loc="left", color="0.25")
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("0.8"); sp.set_linewidth(0.5 * s)
    for ax in axes[n:]:
        ax.axis("off")
    # Absolute reservations rather than fractions: these figures range from ~3 to
    # ~10 inches tall, and a fixed fraction would leave the short ones with a
    # suptitle sitting on the first row of panels.
    fig.suptitle(title, fontsize=9 * s, y=1 - 0.06 / fig_h)
    fig.tight_layout(rect=(0, cbar_h / fig_h, 1, 1 - 0.30 / fig_h))
    if im is not None:
        cax = fig.add_axes([0.25, 0.28 / fig_h, 0.5, 0.16 / fig_h])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.set_label(cbar_label, fontsize=7 * s)
        cb.ax.tick_params(labelsize=6 * s)
    savefig(fig, out_dir, stem)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="all-species model YAML")
    ap.add_argument("--n", type=int, default=16, help="events per grid (default 16)")
    ap.add_argument("--mode", choices=["extremes", "deciles"], default="extremes",
                    help="'extremes' (default): most proton-like, boundary, and most "
                         "kaon-like. 'deciles': every kaon-tagged event ranked by "
                         "kaon-likeness and split into equal-count bins, one grid each, "
                         "which walks the whole axis instead of only its ends.")
    ap.add_argument("--deciles", type=int, default=10,
                    help="number of equal-count bins in --mode deciles (default 10)")
    ap.add_argument("--plane", choices=list(PLANE_INDEX), default="collection")
    ap.add_argument("--scale", choices=["log1p", "linear"], default="log1p",
                    help="colour scale. log1p (default) keeps the track body visible "
                         "under the bright Bragg peak; linear shows raw ADC counts. "
                         "Written to separate files, so both can coexist.")
    ap.add_argument("--vmax-pct", type=float, default=99.9,
                    help="percentile of pooled pixel values used as the shared vmax "
                         "(default 99.9; lower it on a linear scale to bring the "
                         "track body up out of the floor)")
    ap.add_argument("--images", type=Path, default=IMAGE_256)
    ap.add_argument("--picky", type=int, choices=[0, 1], default=None,
                    help="restrict to picky (1) or non-picky (0) kaon-tagged events")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)
    Z, df = load_beam_data(cfg)
    out_dir = Path(args.out_dir) if args.out_dir else Path(figure_dir(cfg, "anchored"))

    log_joint = anchored_log_joint(Z, df["species"], seed=args.seed)
    assigned = log_joint.argmax(axis=1)
    # Margin over the runner-up: the quantity whose sign decides the assignment.
    margin = log_joint[:, 0] - np.max(log_joint[:, 1:], axis=1)
    post = np.exp(log_joint - log_joint.max(axis=1, keepdims=True))
    post /= post.sum(axis=1, keepdims=True)

    is_kaon = (df["species"] == "kaon").to_numpy()
    n_proton = int((df["species"] == "proton").sum())
    keep = is_kaon.copy()
    if args.picky is not None:
        keep &= (df["picky"] == args.picky).to_numpy()

    as_proton = keep & (assigned == 0)
    as_kaon = keep & (assigned == 1)
    n_kaon_tagged = int(keep.sum())
    print(f"kaon-tagged events: {n_kaon_tagged}")
    print(f"  assigned to proton: {int(as_proton.sum())} "
          f"({as_proton.sum() / n_kaon_tagged:.1%})")
    print(f"  assigned to kaon:   {int(as_kaon.sum())} "
          f"({as_kaon.sum() / n_kaon_tagged:.1%})")
    for label, m in (("proton", as_proton), ("kaon", as_kaon)):
        if m.sum() < args.n:
            print(f"  WARNING: only {int(m.sum())} events assigned to {label}, "
                  f"fewer than the {args.n} a grid needs")

    # Margin for the kaon component, the mirror of `margin` for proton.
    kaon_margin = log_joint[:, 1] - np.max(log_joint[:, [0, 2]], axis=1)

    if args.mode == "extremes":
        p_rows = np.flatnonzero(as_proton)
        p_order = p_rows[np.argsort(margin[p_rows])[::-1]]     # most proton-like first
        k_rows = np.flatnonzero(as_kaon)
        k_order = k_rows[np.argsort(kaon_margin[k_rows])[::-1]]  # most kaon-like first
        picks = {"most": p_order[:args.n],
                 "least": p_order[::-1][:args.n],
                 "kaonlike": k_order[:args.n]}
        titles = {
            "most": f"Kaon-tagged events assigned to proton — {args.n} most proton-like",
            "least": f"Kaon-tagged events assigned to proton — {args.n} least "
                     f"proton-like (still assigned)",
            "kaonlike": f"Kaon-tagged events assigned to kaon — {args.n} most kaon-like",
        }
        stems = {"most": "anchored_proton_kaons_most",
                 "least": "anchored_proton_kaons_least",
                 "kaonlike": "anchored_kaonlike_most"}
        # Posterior of the component the event was assigned to, so the number means
        # the same thing on every grid: how sure the fit is of this event.
        post_col = {"most": 0, "least": 0, "kaonlike": 1}
    else:
        # Every kaon-tagged event ranked by kaon-likeness, split into equal-count
        # deciles. Ranking on the whole sample rather than on the kaon-assigned
        # subset is the point: decile 1 is the proton-like end and decile 10 the
        # dense-kaon end, so the series walks the axis the fit orders events along
        # instead of only showing its two extremes.
        rows = np.flatnonzero(keep)
        order = rows[np.argsort(kaon_margin[rows])]            # least kaon-like first
        bounds = np.linspace(0, len(order), args.deciles + 1).astype(int)
        picks, titles, stems, post_col = {}, {}, {}, {}
        for d in range(args.deciles):
            block = order[bounds[d]:bounds[d + 1]]
            # Evenly spaced by rank within the decile, so the 16 span it rather than
            # clustering at one edge, and no seed is involved.
            take = np.linspace(0, len(block) - 1, min(args.n, len(block))).astype(int)
            key = f"d{d + 1:02d}"
            picks[key] = block[take]
            med = float(np.median(post[block, 1]))
            frac_k = float((assigned[block] == 1).mean())
            titles[key] = (f"Kaon-likeness decile {d + 1} of {args.deciles} "
                           f"(n={len(block)}, median $p_\\mathrm{{kaon}}$={med:.3f}, "
                           f"{frac_k:.0%} assigned kaon)")
            stems[key] = f"anchored_kaonlike_decile{d + 1:02d}"
            post_col[key] = 1

    print(f"\nloading {args.plane} plane from {args.images.name} (mmap'd)")
    k_images = torch.load(args.images, map_location="cpu", weights_only=False,
                          mmap=True)["k"]
    plane = PLANE_INDEX[args.plane]

    imgs = {}
    for key, idx in picks.items():
        within = idx - n_proton                        # row -> within-kaon index
        imgs[key] = [k_images[int(i)][plane].numpy() for i in within]

    windows = group_windows(imgs)
    # log1p compresses the ~1500-count Bragg peaks so the dimmer track body stays
    # visible; linear shows the ADC counts as recorded, at the cost of the peaks
    # dominating. Either way the scale is shared across all panels and both grids.
    transform = np.log1p if args.scale == "log1p" else (lambda a: a)
    cbar_label = (r"$\log(1 + \mathrm{ADC})$, " if args.scale == "log1p"
                  else "ADC counts, ") + f"{args.plane} plane"
    all_px = np.concatenate([transform(im).ravel()
                             for group in imgs.values() for im in group])
    vmax = float(np.percentile(all_px, args.vmax_pct))
    print(f"  {args.scale} scale, shared vmax {vmax:.2f} "
          f"(p{args.vmax_pct} of pixels; max {all_px.max():.1f})")

    records = []
    for key, idx in picks.items():
        # Posterior rather than margin: at the boundary the margin rounds to 0.0 for
        # every panel and says nothing, whereas p reads 1.000 against ~0.50 and
        # carries the contrast directly. The margin is in the CSV for ranking.
        # Solidity is on each panel because it is the paper's topology proxy and is
        # what a reader will check the images against: protons sit near 0.67, kaons
        # near 0.40.
        pc = post_col[key]
        labels = [f"$p$ = {post[i, pc]:.3f}\n$S$ = {df['solidity'].iloc[i]:.2f}"
                  for i in idx]
        suffix = "" if args.scale == "log1p" else "_adc"
        plot_grid(imgs[key], labels, windows[key], vmax, titles[key], out_dir,
                  f"{stems[key]}{suffix}", cbar_label=cbar_label, transform=transform)
        for rank, i in enumerate(idx):
            records.append({
                "group": key, "rank": rank, "kaon_index": int(i - n_proton),
                "p_proton": float(post[i, 0]), "p_kaon": float(post[i, 1]),
                "p_muon": float(post[i, 2]),
                "proton_margin": float(margin[i]),
                "kaon_margin": float(kaon_margin[i]),
                "assigned": SPECIES[int(assigned[i])],
                "mean_adc": float(df["mean_adc"].iloc[i]),
                "solidity": float(df["solidity"].iloc[i]),
                "n_local_maxima": float(df["n_local_maxima"].iloc[i]),
                "beamline_mass": float(df["beamline_mass"].iloc[i]),
                "picky": int(df["picky"].iloc[i]),
            })
    out = pd.DataFrame(records)
    out.to_csv(out_dir / "anchored_proton_kaons.csv", index=False)
    print(f"\n{out.groupby('group')[['p_proton', 'mean_adc', 'solidity', 'n_local_maxima', 'beamline_mass']].describe().T.round(3).to_string()}")
    print(f"  saved {out_dir}/anchored_proton_kaons.csv")


if __name__ == "__main__":
    main()
