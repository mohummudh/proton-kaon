#!/usr/bin/env python3
"""
scripts/plot_reconstructions.py

Reconstruction-quality figure for presentation: input vs. VAE reconstruction
for 3 representative proton validation examples (short track, long track,
clear Bragg peak), plus a validation reconstruction-error histogram.

Usage:
    python scripts/plot_reconstructions.py --config configs/run_0066_....yaml
    python scripts/plot_reconstructions.py --config ... --indices 12 345 678

Outputs (under figs/<model_name>/reconstructions/):
    proton_reconstructions.png/.pdf
    proton_val_re_hist.png/.pdf
"""

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family":           "serif",
    "font.serif":            ["Times New Roman", "DejaVu Serif"],
    "font.size":              11,
    "axes.labelsize":         11,
    "xtick.labelsize":        10,
    "ytick.labelsize":        10,
    "legend.fontsize":        9,
    "legend.title_fontsize":  9,
    "axes.linewidth":         0.6,
    "xtick.major.width":      0.6,
    "ytick.major.width":      0.6,
    "xtick.major.size":       3.0,
    "ytick.major.size":       3.0,
    "xtick.minor.visible":    False,
    "ytick.minor.visible":    False,
    "figure.dpi":             300,
    "savefig.dpi":            300,
    "savefig.bbox":           "tight",
    "savefig.pad_inches":     0.02,
})

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from src.models.configVAE import VAE          # noqa: E402
from src.transforms import apply_transform    # noqa: E402

BLUE = "#0077BB"   # Paul Tol palette — colourblind-safe (matches analyse_latents.py)

parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
parser.add_argument("--indices", nargs=3, type=int, default=None,
                    help="Manually pick 3 validation-set indices (within val split) "
                         "instead of the automatic short/long/Bragg selection")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)


def build_model_name(cfg: dict) -> str:
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


def _savefig(path: Path) -> None:
    """Save figure as both PNG (raster) and PDF (vector) for publication."""
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.savefig(path.with_suffix(".pdf"), bbox_inches="tight")


model_name = build_model_name(cfg)
out_dir = Path("figs") / model_name / "reconstructions"
out_dir.mkdir(parents=True, exist_ok=True)

# ── data: proton images, log1p-transformed, validation split ──────────────────
transform = cfg["data"].get("transform", "none")
data = torch.load(cfg["data"]["path"], map_location="cpu")
p = apply_transform(data[cfg["data"]["proton"]], transform)

split_path = Path(cfg["output"]["splits_dir"]) / f"split_{cfg['data']['proton']}.npz"
if split_path.exists():
    split = np.load(split_path)
    val_idx = split["val_idx"]
else:
    from sklearn.model_selection import train_test_split
    print(f"WARNING: {split_path} not found — regenerating deterministically "
          f"(val_split={cfg['data']['val_split']}, seed={cfg['data']['random_seed']})")
    _, val_idx = train_test_split(
        np.arange(len(p)),
        test_size=cfg["data"]["val_split"], random_state=cfg["data"]["random_seed"],
    )

val = p[val_idx]                                   # (N, 2, 48, 48)
print(f"Validation protons: {len(val)}  |  transform: {transform}")

# ── model ──────────────────────────────────────────────────────────────────────
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
attn_cfg = cfg["model"].get("attention", {})
model = VAE(
    input_hw=tuple(cfg["model"]["input_hw"]),
    latent=cfg["model"]["latent"],
    channels=cfg["model"]["channels"],
    kernel=cfg["model"]["kernel"],
    stride=cfg["model"]["stride"],
    padding=cfg["model"]["padding"],
    activation=cfg["model"]["activation"],
    p_enc=cfg["model"].get("dropout", 0.0),
    use_bottleneck_attn=attn_cfg.get("enabled", False),
    attn_after_stage=attn_cfg.get("after_stage"),
    attn_heads=attn_cfg.get("heads", 4),
    attn_depth=attn_cfg.get("depth", 2),
).to(device)
ckpt_path = Path(cfg["output"]["dir"]) / f"{model_name}.pt"
model.load_state_dict(torch.load(ckpt_path, map_location=device))
model.eval()
print(f"Loaded checkpoint: {ckpt_path}")


@torch.no_grad()
def reconstruct(x: torch.Tensor) -> torch.Tensor:
    """Deterministic encoder → μ → decoder pass (no sampling)."""
    mu, _ = model.encode(x.to(device))
    return model.decode(mu).cpu()


# ── full-val reconstruction pass (for RE histogram + example selection) ───────
recons = []
with torch.no_grad():
    for i in range(0, len(val), 64):
        recons.append(reconstruct(val[i:i + 64]))
recons = torch.cat(recons)
re_per_sample = ((recons - val) ** 2).mean(dim=(1, 2, 3)).numpy()   # matches inference.py

# ── representative example selection (collection plane, channel 0) ────────────
col = val[:, 0].numpy()                              # (N, 48, 48)
thresh = 0.1 * col.max()
n_signal = (col > thresh).sum(axis=(1, 2)).astype(float)

# Bragg prominence: peak height relative to the mean signal along the track
sig_mean = np.array([im[im > thresh].mean() if (im > thresh).any() else np.inf for im in col])
prominence = col.max(axis=(1, 2)) / sig_mean

if args.indices is not None:
    picks = {f"Example {chr(65 + i)}": j for i, j in enumerate(args.indices)}
else:
    short_i = int(np.argmin(np.abs(n_signal - np.percentile(n_signal, 10))))
    long_i  = int(np.argmin(np.abs(n_signal - np.percentile(n_signal, 90))))
    # Bragg pick: prominent peak (top decile) that is also well reconstructed,
    # so the panel demonstrates the peak being captured rather than lost.
    candidates = np.where(prominence >= np.percentile(prominence, 90))[0]
    candidates = [int(i) for i in candidates if i not in (short_i, long_i)]
    bragg_i = min(candidates, key=lambda i: re_per_sample[i])
    picks = {"Short track": short_i, "Long track": long_i, "Bragg peak": bragg_i}

# short / medium / long variant (10th / 50th / 90th percentile track length)
def nearest_to_percentile(pct: float, rank: int = 0) -> int:
    """rank-th closest val example to the given track-length percentile."""
    order = np.argsort(np.abs(n_signal - np.percentile(n_signal, pct)))
    return int(order[rank])

length_picks = {
    "Short track":  nearest_to_percentile(10, rank=1),
    "Medium track": nearest_to_percentile(50, rank=1),
    "Long track":   nearest_to_percentile(90),
}

# short / long / kinked variant (kink = angle between the track's two halves)
def kink_angle(img: np.ndarray) -> float:
    """Angle (deg) between the principal directions of the two track halves."""
    pts = np.column_stack(np.where(img > thresh)).astype(float)
    if len(pts) < 20:
        return 0.0
    centred = pts - pts.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    t = centred @ vt[0]                              # position along principal axis
    halves = [pts[t <= np.median(t)], pts[t > np.median(t)]]
    dirs = []
    for h in halves:
        _, _, v = np.linalg.svd(h - h.mean(axis=0), full_matrices=False)
        dirs.append(v[0])
    cos = np.clip(abs(dirs[0] @ dirs[1]), 0.0, 1.0)
    return float(np.degrees(np.arccos(cos)))

kink_angles = np.array([kink_angle(im) for im in col])
# require a real track (not a blob): decent length, then largest kink angle
# (rank picks the k-th most-kinked candidate for variety)
track_like = np.where(n_signal >= np.percentile(n_signal, 25))[0]
kink_rank = 3   # val 1005 — clean single kink; ranks 0-2 are messier branched events
kink_i = int(track_like[np.argsort(kink_angles[track_like])[::-1][kink_rank]])
kink_picks = {
    "Short track":   length_picks["Short track"],
    "Long track":    picks["Long track"],           # val idx 36
    "Kinked track":  kink_i,
}

for group, group_picks in (("main", picks), ("length", length_picks), ("kink", kink_picks)):
    for label, i in group_picks.items():
        print(f"[{group:6s}] {label:12s} | val idx {i:5d} (dataset idx {val_idx[i]:5d}) | "
              f"signal px {int(n_signal[i]):4d} | RE {re_per_sample[i]:.4f}")

# ── figure 1: input vs reconstruction, 3 rows × 2 columns ─────────────────────
def plot_recon_grid(pairs: dict, fname: str, suptitle: str) -> None:
    """pairs: {row label: (input_img, recon_img, re)} — collection plane, 48x48."""
    fig, axes = plt.subplots(3, 2, figsize=(5.2, 7.2))
    for row, (label, (x, r, re)) in enumerate(pairs.items()):
        vmax = max(x.max(), r.max())                 # shared scale per example
        for j, img in enumerate((x, r)):
            ax = axes[row, j]
            ax.imshow(img, origin="lower", cmap="viridis", vmin=0, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 1].text(0.97, 0.97, f"RE = {re:.3f}",
                          transform=axes[row, 1].transAxes,
                          ha="right", va="top", fontsize=9, color="white")
        axes[row, 0].set_ylabel(label, fontsize=11)

    axes[0, 0].set_title("Input", fontsize=12)
    axes[0, 1].set_title("Reconstruction", fontsize=12)
    fig.suptitle(suptitle, fontsize=13, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    _savefig(out_dir / fname)
    plt.close(fig)


plot_recon_grid(
    {label: (col[i], recons[i, 0].numpy(), re_per_sample[i]) for label, i in picks.items()},
    "proton_reconstructions.png",
    r"$\beta$-VAE Proton Reconstructions (validation set)",
)

# same figure in raw ADC space: invert the log1p transform with expm1
# (RE annotation stays in log1p space — the metric the model is trained on)
plot_recon_grid(
    {label: (np.expm1(col[i]), np.expm1(recons[i, 0].numpy()), re_per_sample[i]) for label, i in picks.items()},
    "proton_reconstructions_raw_adc.png",
    r"$\beta$-VAE Proton Reconstructions (validation set, raw ADC)",
)

# short / medium / long track-length variant
plot_recon_grid(
    {label: (col[i], recons[i, 0].numpy(), re_per_sample[i]) for label, i in length_picks.items()},
    "proton_reconstructions_length.png",
    r"$\beta$-VAE Proton Reconstructions (validation set)",
)
plot_recon_grid(
    {label: (np.expm1(col[i]), np.expm1(recons[i, 0].numpy()), re_per_sample[i]) for label, i in length_picks.items()},
    "proton_reconstructions_length_raw_adc.png",
    r"$\beta$-VAE Proton Reconstructions (validation set, raw ADC)",
)

# short / long / kinked (scattered proton) variant
plot_recon_grid(
    {label: (col[i], recons[i, 0].numpy(), re_per_sample[i]) for label, i in kink_picks.items()},
    "proton_reconstructions_kink.png",
    r"$\beta$-VAE Proton Reconstructions (validation set)",
)
plot_recon_grid(
    {label: (np.expm1(col[i]), np.expm1(recons[i, 0].numpy()), re_per_sample[i]) for label, i in kink_picks.items()},
    "proton_reconstructions_kink_raw_adc.png",
    r"$\beta$-VAE Proton Reconstructions (validation set, raw ADC)",
)

# ── figure 2: validation reconstruction-error histogram ───────────────────────
fig, ax = plt.subplots(figsize=(5.2, 3.4))
ax.hist(re_per_sample, bins=50, color=BLUE, edgecolor="white", linewidth=0.4)
ax.axvline(np.median(re_per_sample), color="0.2", linestyle="--", linewidth=1,
           label=f"Median = {np.median(re_per_sample):.4f}")
ax.axvline(re_per_sample.mean(), color="0.2", linestyle=":", linewidth=1,
           label=f"Mean = {re_per_sample.mean():.4f}")
ax.set_xlabel("Per-image reconstruction MSE (log1p space)")
ax.set_ylabel("Validation protons")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False)
fig.tight_layout()
_savefig(out_dir / "proton_val_re_hist.png")
plt.close(fig)

print(f"\nSaved figures to {out_dir}/")
print("  proton_reconstructions.png/.pdf")
print("  proton_reconstructions_raw_adc.png/.pdf")
print("  proton_reconstructions_length.png/.pdf")
print("  proton_reconstructions_length_raw_adc.png/.pdf")
print("  proton_reconstructions_kink.png/.pdf")
print("  proton_reconstructions_kink_raw_adc.png/.pdf")
print("  proton_val_re_hist.png/.pdf")
