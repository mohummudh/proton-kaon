#!/usr/bin/env python3
"""
scripts/extra/animate_latent_tour.py

Animated "tour" of the VAE latent space. Visits N example tracks one at a
time: a marker travels across the UMAP projection to each example's landing
spot while its collection/induction-plane images crossfade into view.

Takes its data loading from umap_explorer.py (latents, UMAP reducer, raw
image lookup) and its per-track rendering style from inspect_solidity.py
(one clean image per example, publication-ish colours), but the output is a
saved animation (GIF, and MP4 if ffmpeg is available) rather than an
interactive app or a static grid.

Usage:
    python scripts/extra/animate_latent_tour.py --config configs/your_config.yaml
    python scripts/extra/animate_latent_tour.py --config configs/your_config.yaml --muon
    python scripts/extra/animate_latent_tour.py --config configs/your_config.yaml --n-examples 5 --seed 7
    python scripts/extra/animate_latent_tour.py --config configs/your_config.yaml --image-size 256
    python scripts/extra/animate_latent_tour.py --config configs/your_config.yaml --muon \
        --indices train:840 val:810 kaon:5385 muon:3934
"""

import argparse
import pickle
import re
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter
import numpy as np
import torch
import yaml

try:
    import umap  # noqa: F401  (needed to unpickle reducer.pkl)
except ImportError:
    print("Error: umap-learn is not installed.")
    sys.exit(1)

# ── species styling — shared hex codes with plot_umap_all.py / umap_explorer.py ──
COLORS = {
    "Proton (Train)": "#0077BB",
    "Proton (Val)":   "#CC0000",
    "Kaon":           "#EE7733",
    "MIPs":           "#AA3377",
}
# priority order examples are drawn from when --n-examples <= number of
# available categories; extra examples cycle back through this list.
CATEGORY_ORDER = ["Proton (Train)", "Proton (Val)", "Kaon", "MIPs"]
MAX_BG_POINTS = 3000  # subsample large background clouds for animation speed


def build_model_name(cfg: dict) -> str:
    """Same naming scheme as plot_umap_all.py / umap_explorer.py — must stay
    in sync so all three scripts resolve to the same inference directory."""
    species_tag = "_speciesall" if cfg["data"].get("proton") == "all" else ""
    return (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
        f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
        f"_tx{cfg['data'].get('transform', 'none')}{species_tag}"
    )


def resolve_proton_split(inf_dir: Path, splits_dir: Path):
    """
    Map train.npz / val.npz row -> index into the full proton image tensor.

    "speciesall" runs use a different, larger train/val partition
    (species_split.npz) than single-species runs (split_p.npz) — the array
    lengths only match one of the two, so pick whichever one actually agrees
    with the latent counts.
    """
    species_split_path = inf_dir / "species_split.npz"
    if species_split_path.exists():
        ss = np.load(species_split_path)
        return ss["p_train_idx"], ss["p_val_idx"]
    split = np.load(splits_dir / "split_p.npz")
    return split["train_idx"], split["val_idx"]


def ease_in_out(t: np.ndarray) -> np.ndarray:
    """Smoothstep easing for marker motion and image crossfades."""
    return t * t * (3 - 2 * t)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--muon", action="store_true", help="Include muon examples")
    parser.add_argument("--image-size", type=int, default=256, choices=[48, 256],
                        help="Pull display images from the NxN image tensor "
                             "(default 256 — sharper for display; the VAE itself trains on 48x48). "
                             "Needs pk_256x256_*.pt / muon_256x256_raw.pt on disk; "
                             "MIPs are skipped for a run if their file at that size is missing.")
    parser.add_argument("--n-examples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--indices", nargs="+", default=None, metavar="DATASET:IDX",
                        help="Manually pick tour stops instead of random sampling, e.g. "
                             "--indices train:840 val:810 kaon:5385 muon:3934 "
                             "(order on the command line is the tour order; overrides "
                             "--n-examples and --seed)")
    parser.add_argument("--hold-frames", type=int, default=26,
                        help="Frames to hold on each landed example")
    parser.add_argument("--transition-frames", type=int, default=20,
                        help="Frames to travel/crossfade between examples")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--out-dir", default=None,
                        help="Output directory (default: figs/<model_name>/latent_tour)")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_name = build_model_name(cfg)
    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name
    splits_dir = Path(cfg["output"]["splits_dir"])

    # ── latents + UMAP ────────────────────────────────────────────────────────
    print(f"Loading latents from {inf_dir} ...")
    train_latents = np.load(inf_dir / "train.npz")["latents"]
    val_latents   = np.load(inf_dir / "val.npz")["latents"]
    kaon_latents  = np.load(inf_dir / "kaon.npz")["latents"]
    train_re = np.load(inf_dir / "train.npz")["re"]
    val_re   = np.load(inf_dir / "val.npz")["re"]
    kaon_re  = np.load(inf_dir / "kaon.npz")["re"]

    muon_latents = muon_re = None
    if args.muon and (inf_dir / "muon.npz").exists():
        muon_npz = np.load(inf_dir / "muon.npz")
        muon_latents, muon_re = muon_npz["latents"], muon_npz["re"]
        print(f"  muon latents: {len(muon_latents)}")
    elif args.muon:
        print("  --muon requested but muon.npz not found, skipping")

    with open(inf_dir / "reducer.pkl", "rb") as f:
        reducer = pickle.load(f)

    print("Transforming to UMAP...")
    train_umap = reducer.transform(train_latents)
    val_umap   = reducer.transform(val_latents)
    kaon_umap  = reducer.transform(kaon_latents)
    muon_umap  = reducer.transform(muon_latents) if muon_latents is not None else None

    # ── images ────────────────────────────────────────────────────────────────
    # Images are loaded mmap'd (lazy, page-in-on-access) rather than materialised
    # with .numpy() — the 256x256 pk tensor is ~9.8 GB on disk and we only ever
    # pull out a handful of individual rows.
    print(f"Loading images ({args.image_size}x{args.image_size})...")
    pk_path  = Path(cfg["data"]["path"])
    raw_path = Path(str(pk_path).replace("log1p", "raw").replace("_log1p", "_raw"))
    if not raw_path.exists():
        raw_path = pk_path
    sized_path = Path(re.sub(r"\d+x\d+", f"{args.image_size}x{args.image_size}", str(raw_path)))
    if sized_path.exists():
        raw_path = sized_path
    elif args.image_size != 48:
        print(f"  {sized_path.name} not found, falling back to {raw_path.name}")
    pk_data  = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)
    p_images = pk_data["p"]
    k_images = pk_data["k"]

    train_idx, val_idx = resolve_proton_split(inf_dir, splits_dir)

    m_images = None
    if muon_latents is not None:
        muon_img_path = Path(
            f"/Volumes/easystore/proton-kaon/images/muon_{args.image_size}x{args.image_size}_raw.pt")
        if muon_img_path.exists():
            m_images = torch.load(muon_img_path, map_location="cpu", weights_only=False,
                                  mmap=True)["m"]
        else:
            print(f"  {muon_img_path.name} not found — generate it to include MIPs at "
                  f"{args.image_size}x{args.image_size}. Skipping MIPs for this run.")
            muon_latents = muon_umap = None

    def get_image(dataset, local_idx):
        if dataset == "train":
            img = p_images[train_idx[local_idx]].numpy()
        elif dataset == "val":
            img = p_images[val_idx[local_idx]].numpy()
        elif dataset == "kaon":
            img = k_images[local_idx].numpy()
        elif dataset == "muon":
            img = m_images[local_idx].numpy()
        else:
            raise ValueError(dataset)
        return img[0], img[1]  # collection, induction

    # ── category table ───────────────────────────────────────────────────────
    categories = {
        "Proton (Train)": dict(dataset="train", umap=train_umap, re=train_re),
        "Proton (Val)":   dict(dataset="val",   umap=val_umap,   re=val_re),
        "Kaon":           dict(dataset="kaon",  umap=kaon_umap,  re=kaon_re),
    }
    if muon_umap is not None:
        categories["MIPs"] = dict(dataset="muon", umap=muon_umap, re=muon_re)

    dataset_to_category = {info["dataset"]: cat for cat, info in categories.items()}
    available = [c for c in CATEGORY_ORDER if c in categories]

    def make_stop(cat, local_idx):
        info = categories[cat]
        col_img, ind_img = get_image(info["dataset"], local_idx)
        dup = sum(1 for s in stops if s["category"] == cat)
        label = cat if dup == 0 else f"{cat} #{dup + 1}"
        return dict(
            category=cat, label=label, color=COLORS[cat],
            dataset=info["dataset"], local_idx=local_idx,
            xy=np.array(info["umap"][local_idx], dtype=float),
            col_img=col_img.astype(np.float64), ind_img=ind_img.astype(np.float64),
            re=float(info["re"][local_idx]),
        )

    stops = []
    if args.indices:
        # ── manual selection — exact order as given on the command line ──────
        for spec in args.indices:
            dataset, _, idx_str = spec.partition(":")
            if dataset not in dataset_to_category:
                raise ValueError(
                    f"--indices entry {spec!r}: dataset must be one of "
                    f"{sorted(dataset_to_category)} (is '{dataset}' loaded? "
                    f"e.g. pass --muon for the muon dataset)")
            cat = dataset_to_category[dataset]
            local_idx = int(idx_str)
            n_pts = len(categories[cat]["umap"])
            if not (0 <= local_idx < n_pts):
                raise ValueError(
                    f"--indices entry {spec!r}: index {local_idx} out of range "
                    f"[0, {n_pts}) for dataset '{dataset}'")
            stops.append(make_stop(cat, local_idx))
    else:
        # ── pick n_examples, cycling through categories in priority order ────
        used_idx = {c: set() for c in available}
        i = 0
        while len(stops) < args.n_examples:
            cat = available[i % len(available)]
            n_pts = len(categories[cat]["umap"])
            remaining = [j for j in range(n_pts) if j not in used_idx[cat]]
            if not remaining:
                i += 1
                continue
            local_idx = int(rng.choice(remaining))
            used_idx[cat].add(local_idx)
            stops.append(make_stop(cat, local_idx))
            i += 1

    print("\nTour stops:")
    for n, s in enumerate(stops):
        print(f"  {n+1}. {s['label']:16s} idx={s['local_idx']:<6d} "
              f"UMAP=({s['xy'][0]:.2f}, {s['xy'][1]:.2f})  re={s['re']:.3f}")

    # ── figure ────────────────────────────────────────────────────────────────
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.linewidth": 0.6,
    })
    fig = plt.figure(figsize=(11, 5.2))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1], wspace=0.28, hspace=0.32,
                          left=0.06, right=0.97, top=0.88, bottom=0.08)
    ax_umap = fig.add_subplot(gs[:, 0])
    ax_col  = fig.add_subplot(gs[0, 1])
    ax_ind  = fig.add_subplot(gs[1, 1])

    def subsample(arr):
        if len(arr) <= MAX_BG_POINTS:
            return arr
        idx = rng.choice(len(arr), size=MAX_BG_POINTS, replace=False)
        return arr[idx]

    for cat in available:
        pts = subsample(categories[cat]["umap"])
        ax_umap.scatter(pts[:, 0], pts[:, 1], s=6, alpha=0.5, linewidths=0,
                        color=COLORS[cat], label=cat)
    leg = ax_umap.legend(loc="best", framealpha=0.85, edgecolor="0.75",
                         handlelength=1.0, handletextpad=0.4, fontsize=8)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax_umap.set_xlabel("UMAP 1")
    ax_umap.set_ylabel("UMAP 2")

    # fix axis limits so the view doesn't rescale during the animation
    all_xy = np.vstack([categories[c]["umap"] for c in available])
    pad_x = 0.08 * (all_xy[:, 0].max() - all_xy[:, 0].min())
    pad_y = 0.08 * (all_xy[:, 1].max() - all_xy[:, 1].min())
    ax_umap.set_xlim(all_xy[:, 0].min() - pad_x, all_xy[:, 0].max() + pad_x)
    ax_umap.set_ylim(all_xy[:, 1].min() - pad_y, all_xy[:, 1].max() + pad_y)

    trail_line, = ax_umap.plot([], [], "-", color="0.3", lw=1.2, alpha=0.7, zorder=4)
    visited_scatter = ax_umap.scatter([], [], s=40, edgecolors="black",
                                      linewidths=0.8, zorder=5)
    marker = ax_umap.scatter([], [], s=140, edgecolors="black", linewidths=1.4,
                             marker="*", zorder=6)

    img_h, img_w = stops[0]["col_img"].shape

    def _image_panel(ax, title):
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        im = ax.imshow(np.zeros((img_h, img_w)), cmap="viridis", origin="lower",
                       vmin=0, vmax=1, interpolation="nearest", aspect="auto")
        return im

    im_col = _image_panel(ax_col, "Collection Plane")
    im_ind = _image_panel(ax_ind, "Induction Plane")

    title_text = fig.text(0.06, 0.94, "", fontsize=16, fontweight="bold", va="top")

    # ── precompute per-frame state ───────────────────────────────────────────
    blank = np.zeros((img_h, img_w), dtype=np.float64)
    frames = []
    for i, cur in enumerate(stops):
        prev_xy = stops[i - 1]["xy"] if i > 0 else cur["xy"]
        prev_col = stops[i - 1]["col_img"] if i > 0 else blank
        prev_ind = stops[i - 1]["ind_img"] if i > 0 else blank

        n_t = args.transition_frames
        for f in range(n_t):
            t = ease_in_out(np.array(f / max(n_t - 1, 1)))
            frames.append(dict(
                pos=prev_xy + (cur["xy"] - prev_xy) * t,
                col_img=prev_col * (1 - t) + cur["col_img"] * t,
                ind_img=prev_ind * (1 - t) + cur["ind_img"] * t,
                visited_upto=i - 1, stop_idx=i,
                label=cur["label"], color=cur["color"],
            ))

        n_h = args.hold_frames
        for f in range(n_h):
            pulse = 0.5 + 0.5 * np.sin(2 * np.pi * 2 * f / max(n_h, 1))
            frames.append(dict(
                pos=cur["xy"], col_img=cur["col_img"], ind_img=cur["ind_img"],
                visited_upto=i, stop_idx=i,
                label=cur["label"], color=cur["color"],
                pulse=pulse,
            ))

    def update(i):
        st = frames[i]
        marker.set_offsets([st["pos"]])
        marker.set_facecolor(st["color"])
        base_size = 140
        marker.set_sizes([base_size * (1 + 0.35 * st.get("pulse", 0.0))])

        visited = stops[: st["visited_upto"] + 1]
        path_xy = [s["xy"] for s in visited] + [st["pos"]]
        trail_line.set_data([p[0] for p in path_xy], [p[1] for p in path_xy])
        if visited:
            visited_scatter.set_offsets(np.array([s["xy"] for s in visited]))
            visited_scatter.set_facecolor([s["color"] for s in visited])
        else:
            visited_scatter.set_offsets(np.empty((0, 2)))

        im_col.set_data(st["col_img"])
        im_col.set_clim(0, max(float(st["col_img"].max()), 1.0))
        im_ind.set_data(st["ind_img"])
        im_ind.set_clim(0, max(float(st["ind_img"].max()), 1.0))

        title_text.set_text(st["label"])
        title_text.set_color(st["color"])
        return marker, trail_line, visited_scatter, im_col, im_ind, title_text

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / args.fps,
                         blit=False)

    out_dir = Path(args.out_dir) if args.out_dir else Path("figs") / model_name / "latent_tour"
    out_dir.mkdir(parents=True, exist_ok=True)

    gif_path = out_dir / "latent_tour.gif"
    print(f"\nRendering {len(frames)} frames -> {gif_path} ...")
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    print(f"  saved {gif_path}")

    if shutil.which("ffmpeg"):
        mp4_path = out_dir / "latent_tour.mp4"
        print(f"Rendering {len(frames)} frames -> {mp4_path} ...")
        anim.save(mp4_path, writer=FFMpegWriter(fps=args.fps, bitrate=4000))
        print(f"  saved {mp4_path}")
    else:
        print("ffmpeg not found on PATH — skipping .mp4 (GIF still produced).")

    plt.close(fig)
    print("\nDone.")


if __name__ == "__main__":
    main()
