import argparse
import pickle
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import umap
except ImportError:
    print("Error: umap-learn is not installed.")
    exit(1)

# ── Global matplotlib settings for publication ────────────────────────────────
plt.rcParams.update({
    # Font
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif"],
    # "text.usetex":        True,           # drop this line if no LaTeX install
    "font.size":          9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "legend.title_fontsize": 8,
    # Lines / ticks
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   3.0,
    "ytick.major.size":   3.0,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    # Output
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.02,
})

# Paul Tol palette — colourblind-safe and greyscale-distinguishable
COLORS = {
    "Proton (Train)": "#0077BB",
    "Proton (Val)":   "#CC0000",
    "Kaon":           "#EE7733",
    "Muon":           "#AA3377",
    "CSDA-Kaon":      "#CC3311",
}

# Figure widths matching typical journal column widths (inches)
SINGLE_COL = 3.375   # ~86 mm  — single column
DOUBLE_COL = 6.875   # ~175 mm — double / full width


def build_model_name(cfg: dict) -> str:
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
        f"_tx{cfg['data'].get('transform', 'none')}"
    )


def style_legend(leg, marker_size: float = 30):
    """Ensure legend markers are fully opaque and consistently sized."""
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([marker_size])


def make_legend_kwargs(loc: str = "best") -> dict:
    return dict(
        frameon=True,
        framealpha=0.85,
        edgecolor="0.75",
        loc=loc,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.5,
        labelspacing=0.35,
    )


def save(fig, path: Path):
    """Save as both PDF (paper) and PNG (quick preview)."""
    pdf_path = path.with_suffix(".pdf")
    png_path = path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    print(f"  Saved {pdf_path.name}  +  {png_path.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    parser.add_argument("--no-csda", action="store_true",
                        help="Skip loading and plotting CSDA kaons")
    parser.add_argument("--double-col", action="store_true",
                        help="Use double-column figure width instead of single-column")
    parser.add_argument("--from-cache", action="store_true",
                        help="Skip UMAP transform; load pre-computed embeddings from cache_umap.npz")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = build_model_name(cfg)
    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name
    out_dir = Path("figs") / model_name / "latents-features"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_w = DOUBLE_COL if args.double_col else SINGLE_COL
    # Height: golden-ratio-ish for scatter plots
    fig_h = round(fig_w / 1.25, 3)

    # ── Load latents ──────────────────────────────────────────────────────────
    try:
        train_latents = np.load(inf_dir / "train.npz")["latents"]
        val_latents   = np.load(inf_dir / "val.npz")["latents"]
        kaon_latents  = np.load(inf_dir / "kaon.npz")["latents"]
        print(
            f"Loaded {len(train_latents)} protons (train), "
            f"{len(val_latents)} protons (val), "
            f"{len(kaon_latents)} kaon candidates."
        )
    except FileNotFoundError as e:
        print(f"Error: Could not find inference files in {inf_dir}\n{e}")
        return

    muon_latents = None
    muon_path = inf_dir / "muon.npz"
    if muon_path.exists():
        muon_latents = np.load(muon_path)["latents"]
        print(f"Loaded {len(muon_latents)} muons.")

    csda_kaon_latents = None
    csda_kaon_path = inf_dir / "csda_kaon.npz"
    if not args.no_csda and csda_kaon_path.exists():
        csda_kaon_latents = np.load(csda_kaon_path)["latents"]
        print(f"Loaded {len(csda_kaon_latents)} csda-kaons.")

    # ── UMAP reducer ─────────────────────────────────────────────────────────
    reducer_path = inf_dir / "reducer.pkl"
    if reducer_path.exists():
        with open(reducer_path, "rb") as f:
            reducer = pickle.load(f)
        print(f"Loaded existing UMAP reducer from {reducer_path}")
    else:
        print("Training new UMAP reducer (this may take a moment)...")
        all_latents_list = [train_latents, val_latents, kaon_latents]
        if muon_latents is not None:
            all_latents_list.append(muon_latents)
        if csda_kaon_latents is not None:
            all_latents_list.append(csda_kaon_latents)
        all_latents = np.vstack(all_latents_list)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        reducer.fit(all_latents)
        with open(reducer_path, "wb") as f:
            pickle.dump(reducer, f)
        print("Fitted and saved new UMAP reducer.")

    embed_cache = out_dir / "cache_umap.npz"
    if args.from_cache and embed_cache.exists():
        print(f"Loading UMAP embeddings from cache ({embed_cache.name})...")
        _emb = np.load(embed_cache)
        train_umap     = _emb["train"]
        val_umap       = _emb["val"]
        kaon_umap      = _emb["kaon"]
        muon_umap      = _emb["muon"]      if "muon"      in _emb else None
        csda_kaon_umap = _emb["csda_kaon"] if "csda_kaon" in _emb else None
    else:
        print("Transforming latents...")
        train_umap = reducer.transform(train_latents)
        val_umap   = reducer.transform(val_latents)
        kaon_umap  = reducer.transform(kaon_latents)
        muon_umap      = reducer.transform(muon_latents)      if muon_latents is not None      else None
        csda_kaon_umap = reducer.transform(csda_kaon_latents) if csda_kaon_latents is not None else None
        _save_dict = dict(train=train_umap, val=val_umap, kaon=kaon_umap)
        if muon_umap      is not None: _save_dict["muon"]      = muon_umap
        if csda_kaon_umap is not None: _save_dict["csda_kaon"] = csda_kaon_umap
        np.savez(embed_cache, **_save_dict)
        print(f"Saved UMAP embeddings → {embed_cache.name}")

    # Scatter kwargs shared across UMAP plots
    sc_main  = dict(s=3, alpha=0.5, linewidths=0)
    sc_csda  = dict(s=10, alpha=1.0, linewidths=0.3, edgecolors="white")

    print("Plotting...")

    # ── Plot 1: proton train vs val ───────────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(fig_w, fig_h))
    ax1.scatter(train_umap[:, 0], train_umap[:, 1],
                c=COLORS["Proton (Train)"], label="Proton (train)", **sc_main)
    ax1.scatter(val_umap[:, 0], val_umap[:, 1],
                c=COLORS["Proton (Val)"], label="Proton (val)", **sc_main)
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    style_legend(ax1.legend(**make_legend_kwargs()))
    sns.despine(ax=ax1)
    fig1.tight_layout()
    save(fig1, out_dir / "umap_proton_train_val")
    plt.close(fig1)

    # ── Plot 2: all species ───────────────────────────────────────────────────
    proton_umap = np.vstack([train_umap, val_umap])
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    ax2.scatter(proton_umap[:, 0], proton_umap[:, 1],
                c=COLORS["Proton (Train)"], label="Proton", **sc_main)
    ax2.scatter(kaon_umap[:, 0], kaon_umap[:, 1],
                c=COLORS["Kaon"], label="Kaon", **sc_main)
    if muon_umap is not None:
        ax2.scatter(muon_umap[:, 0], muon_umap[:, 1],
                    c=COLORS["Muon"], label="Muon", **sc_main)
    if csda_kaon_umap is not None:
        ax2.scatter(csda_kaon_umap[:, 0], csda_kaon_umap[:, 1],
                    c=COLORS["CSDA-Kaon"], label="CSDA-Kaon", **sc_csda)
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    style_legend(ax2.legend(**make_legend_kwargs()))
    sns.despine(ax=ax2)
    fig2.tight_layout()
    save(fig2, out_dir / "umap_all_species")
    plt.close(fig2)

    # ── Plot 3: z4 vs z7 — all species ───────────────────────────────────────
    proton_latents = np.vstack([train_latents, val_latents])
    fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
    ax3.scatter(proton_latents[:, 4], proton_latents[:, 7],
                c=COLORS["Proton (Train)"], label="Proton", **sc_main)
    ax3.scatter(kaon_latents[:, 4], kaon_latents[:, 7],
                c=COLORS["Kaon"], label="Kaon", **sc_main)
    if muon_latents is not None:
        ax3.scatter(muon_latents[:, 4], muon_latents[:, 7],
                    c=COLORS["Muon"], label="Muon", **sc_main)
    if csda_kaon_latents is not None:
        ax3.scatter(csda_kaon_latents[:, 4], csda_kaon_latents[:, 7],
                    c=COLORS["CSDA-Kaon"], label="CSDA-Kaon", **sc_csda)
    ax3.set_xlabel(r"$z_4$")
    ax3.set_ylabel(r"$z_7$")
    style_legend(ax3.legend(**make_legend_kwargs()))
    sns.despine(ax=ax3)
    fig3.tight_layout()
    save(fig3, out_dir / "z4_vs_z7_all_species")
    plt.close(fig3)

    # ── Plot 4: z4 vs z7 — train vs val protons ──────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
    ax4.scatter(train_latents[:, 4], train_latents[:, 7],
                c=COLORS["Proton (Train)"], label="Proton (train)", **sc_main)
    ax4.scatter(val_latents[:, 4], val_latents[:, 7],
                c=COLORS["Proton (Val)"], label="Proton (val)", **sc_main)
    ax4.set_xlabel(r"$z_4$")
    ax4.set_ylabel(r"$z_7$")
    style_legend(ax4.legend(**make_legend_kwargs()))
    sns.despine(ax=ax4)
    fig4.tight_layout()
    save(fig4, out_dir / "z4_vs_z7_proton_train_val")
    plt.close(fig4)

    print("Done.")


if __name__ == "__main__":
    main()