#!/usr/bin/env python3
"""
scripts/inspect_solidity.py

Visualises what solidity measures on real track images.

For each particle type (proton, kaon, muon) it shows a grid of representative
tracks (near mean H and W). Each panel has three layers:
  - signal pixels (viridis, the numerator)
  - "gap" pixels inside the hull but with no signal (red tint, the "wasted" space)
  - the convex hull boundary (white outline, the denominator boundary)
Solidity = signal pixels / hull pixels.

Images are placed on a canvas using the same horizontal alignment as
image_making.py (peak column of first wire row → canvas centre).

Usage:
    python scripts/inspect_solidity.py
    python scripts/inspect_solidity.py --n-tracks 12 --seed 7
    python scripts/inspect_solidity.py --no-muons
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import torch
from skimage.measure import label, regionprops
from skimage.morphology import convex_hull_image

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.cuts import image_cuts
from src.images import pad_image_batch_gpu

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")

COL_PKL      = Path("/Volumes/easystore/proton-kaon/clusters/col.pkl")
IND_PKL      = Path("/Volumes/easystore/proton-kaon/clusters/ind.pkl")
MUON_COL_PKL = Path("/Volumes/easystore/proton-kaon/clusters/muon_col.pkl")
OUT_DIR      = PROJECT_ROOT / "figs" / "inspect_solidity"

COLOURS = {
    "proton": "#4C78A8",
    "kaon":   "#F58518",
    "muon":   "#9467BD",
}


# ── image placement — uses pad_image_batch_gpu from image_making pipeline ────

CANVAS_W = 700   # fixed width — matches pad_image_batch_gpu centre heuristic
CANVAS_H = 200    # fixed height — accommodates muons (~194 wires) and all shorter tracks
DPI      = 150

PANEL_SIZE_IN = 3.0   # square panel — same width and height for every plot
PANEL_W_IN    = PANEL_SIZE_IN
PANEL_H_IN    = PANEL_SIZE_IN


def _pad_batch(imgs):
    """
    Pad a list of images onto the shared (CANVAS_W × CANVAS_H) canvas,
    exactly as image_making.py does — no row cutting.
    Returns (padded_arrays, x0_list).
    """
    padded_list = pad_image_batch_gpu(imgs, device=DEVICE, batch_size=64,
                                       cut_rows=0, target_wh=(CANVAS_W, CANVAS_H))
    x0_list = [max(0, min(CANVAS_W // 2 - int(np.argmax(img[0])), CANVAS_W - img.shape[1]))
               for img in imgs]
    return [np.array(p) for p in padded_list], x0_list


# ── solidity ─────────────────────────────────────────────────────────────────

def compute_solidity(img, threshold=0):
    binary  = img > threshold
    labeled = label(binary)
    if labeled.max() == 0:
        return np.nan, None, None
    regions    = regionprops(labeled)
    main       = max(regions, key=lambda r: r.area)
    signal_mask = labeled == main.label
    hull_mask   = convex_hull_image(signal_mask)
    return main.solidity, signal_mask, hull_mask


# ── representative track selection ───────────────────────────────────────────

def select_representative(rows, n, seed, tol=0.75):
    """
    Return n tracks whose (H, W) are within tol*std of the mean.
    Falls back to the full set if fewer than n pass the filter.
    """
    shapes = [np.array(r["image_intensity"]).shape for _, r in rows.iterrows()]
    hs = np.array([s[0] for s in shapes])
    ws = np.array([s[1] for s in shapes])
    mask = (
        (np.abs(hs - hs.mean()) < tol * hs.std()) &
        (np.abs(ws - ws.mean()) < tol * ws.std())
    )
    filtered = rows[mask] if mask.sum() >= n else rows
    sample = filtered.sample(min(n, len(filtered)), random_state=seed).reset_index(drop=True)

    # sort high → low solidity
    solidities = [compute_solidity(np.array(r["image_intensity"]))[0] for _, r in sample.iterrows()]
    order = np.argsort(solidities)[::-1]
    return sample.iloc[order].reset_index(drop=True)


# ── single-panel renderer ─────────────────────────────────────────────────────

# ── per-particle grid ─────────────────────────────────────────────────────────

def plot_grid(sample, particle, out_dir):
    imgs   = [np.array(r["image_intensity"]) for _, r in sample.iterrows()]
    padded_list, x0_list = _pad_batch(imgs)

    ncols  = 4
    nrows  = int(np.ceil(len(imgs) / ncols))
    colour = COLOURS[particle]

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * PANEL_W_IN, nrows * PANEL_H_IN + 0.6),
                             constrained_layout=True)
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, (img, padded, x0) in enumerate(zip(imgs, padded_list, x0_list)):
        ax  = axes[idx // ncols, idx % ncols]
        sol = compute_solidity(img)[0]
        _render_panel(ax, img, padded, x0)
        ax.set_title(f"solidity = {sol:.3f}" if not np.isnan(sol) else "solidity = nan",
                     fontsize=8)

    for idx in range(len(sample), nrows * ncols):
        axes[idx // ncols, idx % ncols].set_visible(False)

    signal_patch = mpatches.Patch(facecolor=plt.cm.inferno(0.6),
                                  label="signal pixels — coloured by ADC (numerator)")
    gap_patch    = mpatches.Patch(facecolor=(0.85, 0.15, 0.15, 0.5),
                                  label="gap inside outline (empty space)")
    hull_patch   = mpatches.Patch(facecolor="none", edgecolor="white", linewidth=1.2,
                                  label="tightest gap-free outline (denominator boundary)")
    fig.legend(handles=[signal_patch, gap_patch, hull_patch],
               loc="lower center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.03), framealpha=0.9)

    fig.suptitle(
        f"{particle.capitalize()}s — solidity: signal pixels ÷ pixels inside tightest enclosing outline\n"
        f"tracks near mean size  |  sorted high → low solidity",
        fontsize=10, fontweight="bold", color=colour,
    )
    path = out_dir / f"solidity_{particle}.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


# ── solidity histogram ────────────────────────────────────────────────────────

def plot_histogram(data_by_particle, out_dir):
    fig, ax = plt.subplots(figsize=(8, 4))
    for particle, vals in data_by_particle.items():
        finite = vals[np.isfinite(vals)]
        ax.hist(finite, bins=60, range=(0, 1), density=True,
                color=COLOURS[particle], alpha=0.50,
                label=f"{particle} (n={len(finite)})", histtype="stepfilled")
        ax.hist(finite, bins=60, range=(0, 1), density=True,
                color=COLOURS[particle], alpha=0.9,
                histtype="step", linewidth=1.4)

    ax.set_xlabel("Solidity  (signal pixels / pixels inside enclosing outline)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Solidity distribution per particle type", fontsize=12, fontweight="bold")
    ax.legend(fontsize=10, framealpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)

    path = out_dir / "histogram.png"
    plt.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"  saved {path}")


def compute_solidities(rows):
    return np.array([
        compute_solidity(np.array(r["image_intensity"]))[0]
        for _, r in rows.iterrows()
    ])


def pick_in_solidity_range(rows, sol_min, sol_max):
    """
    Among tracks with solidity in [sol_min, sol_max], return the one
    whose (H, W) is closest to that particle type's mean.
    """
    shapes     = [np.array(r["image_intensity"]).shape for _, r in rows.iterrows()]
    hs         = np.array([s[0] for s in shapes], dtype=float)
    ws         = np.array([s[1] for s in shapes], dtype=float)
    solidities = np.array([compute_solidity(np.array(r["image_intensity"]))[0]
                           for _, r in rows.iterrows()])

    mask = (solidities >= sol_min) & (solidities <= sol_max) & np.isfinite(solidities)
    if mask.sum() == 0:
        raise ValueError(f"No tracks found with solidity in [{sol_min}, {sol_max}]")

    cand_idx = np.where(mask)[0]
    dist = (((hs[cand_idx] - hs.mean()) / (hs.std() + 1e-9)) ** 2 +
            ((ws[cand_idx] - ws.mean()) / (ws.std() + 1e-9)) ** 2) ** 0.5
    best = cand_idx[int(np.argmin(dist))]
    return rows.iloc[best]


def _render_panel(ax, img, padded, x0):
    """
    Draw one track onto ax.
    Base layer: inferno imshow on the padded+cropped image (dark background, ADC coloured).
    Second layer: semi-transparent red for gap pixels inside the hull.
    Third layer: white contour for the hull boundary.
    """
    sol, signal_mask, hull_mask = compute_solidity(img)
    if signal_mask is None:
        ax.axis("off")
        return

    h, w = img.shape

    sig_full  = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    hull_full = np.zeros((CANVAS_H, CANVAS_W), dtype=bool)
    sig_full[0:h,  x0:x0+w] = signal_mask
    hull_full[0:h, x0:x0+w] = hull_mask
    gap_full = hull_full & ~sig_full

    # layer 1: ADC image on full fixed canvas
    ax.imshow(padded, cmap="inferno", origin="lower",
              interpolation="nearest", aspect="auto")

    # layer 2: red gap overlay
    gap_rgba = np.zeros((CANVAS_H, CANVAS_W, 4))
    gap_rgba[gap_full] = [0.85, 0.15, 0.15, 0.5]
    ax.imshow(gap_rgba, origin="lower", interpolation="nearest", aspect="auto")

    # layer 3: hull boundary
    ax.contour(hull_full.astype(float), levels=[0.5],
               colors="white", linewidths=1.0, alpha=0.9)
    ax.axis("off")


def plot_trio_plain(particles, out_dir, sol_ranges=None):
    """
    Same 3 selected tracks as plot_trio, but plain — just the image itself,
    viridis colormap, no solidity overlays (no gap tint, no hull outline, no legend).
    """
    if sol_ranges is None:
        sol_ranges = {"proton": (0.65, 0.75), "kaon": (0.20, 0.30), "muon": (0.45, 0.55)}

    names, imgs = [], []
    for particle, rows in particles.items():
        lo, hi = sol_ranges.get(particle, (0.0, 1.0))
        row = pick_in_solidity_range(rows, lo, hi)
        imgs.append(np.array(row["image_intensity"]))
        names.append(particle)

    padded_list, x0_list = _pad_batch(imgs)

    DOUBLE_COL = 6.875
    _rc = {
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          12,
        "axes.labelsize":     12,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    }

    with matplotlib.rc_context(_rc):
        fig, axes = plt.subplots(1, len(names), figsize=(DOUBLE_COL, 3.1))
        fig.subplots_adjust(left=0.01, right=0.99, top=0.88, bottom=0.05, wspace=0.04)

        if len(names) == 1:
            axes = [axes]

        for ax, particle, padded in zip(axes, names, padded_list):
            ax.imshow(padded, cmap="viridis", origin="lower",
                      interpolation="nearest", aspect="auto")
            ax.set_title(particle.capitalize(), fontsize=13, fontweight="bold",
                         color=COLOURS[particle], pad=5)
            ax.axis("off")

        path = out_dir / "solidity_trio_plain.png"
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close()
        print(f"  saved {path}")


def plot_trio(particles, out_dir,
              sol_ranges=None):
    """
    One panel per particle type, picking a track in the requested solidity range.
    Default ranges chosen to contrast: proton ≈ compact, kaon ≈ fragmented.
    """
    if sol_ranges is None:
        sol_ranges = {"proton": (0.65, 0.75), "kaon": (0.20, 0.30), "muon": (0.45, 0.55)}

    names, imgs, sols = [], [], []
    for particle, rows in particles.items():
        lo, hi = sol_ranges.get(particle, (0.0, 1.0))
        row = pick_in_solidity_range(rows, lo, hi)
        img = np.array(row["image_intensity"])
        names.append(particle)
        imgs.append(img)
        sols.append(compute_solidity(img)[0])
        print(f"  {particle:6s}: H={img.shape[0]}  W={img.shape[1]}  solidity={sols[-1]:.3f}")

    # pad all three on the identical fixed canvas
    padded_list, x0_list = _pad_batch(imgs)

    # Publication-quality styling — double-column width, Times New Roman
    DOUBLE_COL = 6.875   # ~175 mm, spans both journal columns
    _rc = {
        "font.family":        "serif",
        "font.serif":         ["Times New Roman", "DejaVu Serif"],
        "font.size":          12,
        "axes.labelsize":     12,
        "legend.fontsize":    10,
        "legend.title_fontsize": 10,
        "figure.dpi":         300,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.02,
    }

    with matplotlib.rc_context(_rc):
        fig, axes = plt.subplots(
            1, len(names),
            figsize=(DOUBLE_COL, 3.1),
        )
        # Reserve space: top for two-line subtitles, bottom for legend row
        fig.subplots_adjust(left=0.01, right=0.99,
                            top=0.84, bottom=0.17,
                            wspace=0.04)

        if len(names) == 1:
            axes = [axes]

        for ax, particle, img, padded, x0, sol in zip(
                axes, names, imgs, padded_list, x0_list, sols):
            _render_panel(ax, img, padded, x0)
            ax.set_title(
                f"{particle.capitalize()}\nSolidity = {sol:.3f}",
                fontsize=13, fontweight="bold", color=COLOURS[particle],
                pad=5,
            )

        signal_patch = mpatches.Patch(facecolor=plt.cm.inferno(0.6),
                                      label="Signal pixels (ADC, numerator)")
        gap_patch    = mpatches.Patch(facecolor=(0.85, 0.15, 0.15, 0.5),
                                      label="Gap pixels inside convex hull")
        hull_patch   = mpatches.Patch(facecolor="none", edgecolor="white", linewidth=1.5,
                                      label="Convex hull boundary (denominator)")
        fig.legend(handles=[signal_patch, gap_patch, hull_patch],
                   loc="lower center", ncol=3, fontsize=10,
                   bbox_to_anchor=(0.5, 0.01),
                   framealpha=0.92, edgecolor="0.75",
                   handlelength=1.2, handletextpad=0.5,
                   borderpad=0.6, labelspacing=0.3)

        path = out_dir / "solidity_trio.png"
        fig.savefig(path, dpi=300)
        fig.savefig(path.with_suffix(".pdf"))
        plt.close()
        print(f"  saved {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tracks", type=int, default=16,
                        help="Number of example tracks per particle (default 16)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-muons", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading col.pkl…")
    col = pd.read_pickle(COL_PKL)
    ind = pd.read_pickle(IND_PKL)
    col, ind = image_cuts(col, ind, lower=10)
    print(f"  {len(col)} rows after cuts")

    protons = col[col["particle_type"] == "proton"]
    kaons   = col[col["particle_type"] == "kaon"]
    print(f"  protons: {len(protons)}  kaons: {len(kaons)}")

    particles = {"proton": protons, "kaon": kaons}

    if not args.no_muons and MUON_COL_PKL.exists():
        muons = pd.read_pickle(MUON_COL_PKL)
        print(f"  muons:   {len(muons)}")
        particles["muon"] = muons
    elif not args.no_muons:
        print(f"  muon pkl not found at {MUON_COL_PKL}, skipping")

    print("\nSelecting representative tracks and plotting grids…")
    for particle, rows in particles.items():
        sample = select_representative(rows, args.n_tracks, args.seed)
        shapes = [np.array(r["image_intensity"]).shape for _, r in sample.iterrows()]
        hs = [s[0] for s in shapes]
        ws = [s[1] for s in shapes]
        print(f"  {particle:6s}: {len(sample)} tracks  "
              f"H=[{min(hs)},{max(hs)}]  W=[{min(ws)},{max(ws)}]")
        plot_grid(sample, particle, OUT_DIR)

    print("\nPlotting trio (one representative track per particle)…")
    trio_sol_ranges = {
        "proton": (0.65, 0.75),
        "kaon":   (0.20, 0.30),
        "muon":   (0.45, 0.55),
    }
    plot_trio(particles, OUT_DIR, sol_ranges=trio_sol_ranges)

    print("\nPlotting plain trio (no solidity overlay)…")
    plot_trio_plain(particles, OUT_DIR, sol_ranges=trio_sol_ranges)

    print("\nComputing solidity distributions…")
    solidities_by_particle = {}
    for particle, rows in particles.items():
        vals = compute_solidities(rows)
        solidities_by_particle[particle] = vals
        finite = vals[np.isfinite(vals)]
        print(f"  {particle:6s}  mean={finite.mean():.3f}  "
              f"median={np.median(finite):.3f}  "
              f"frac<0.8={100*(finite<0.8).mean():.1f}%")

    plot_histogram(solidities_by_particle, OUT_DIR)
    print(f"\nDone. All plots saved to {OUT_DIR}")


if __name__ == "__main__":
    main()
