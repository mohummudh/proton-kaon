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


def resolve_dims(args_dims, n_latent: int, out_dir: Path) -> list:
    """Pick the two latent dims for the z-scatter plots: CLI override, else the
    top-2 single-dim AUCs from the logistic probe cache, else 4/7 (paper model)."""
    if args_dims is not None:
        return args_dims
    cache = out_dir / "cache_logistic.pkl"
    if cache.exists():
        try:
            with open(cache, "rb") as f:
                results = pickle.load(f)["results"]
            singles = [(int(l[1:]), v["AUC"]) for l, v in results.items()
                       if l.startswith("z") and l[1:].isdigit()]
            if len(singles) >= 2:
                top = sorted(singles, key=lambda t: t[1], reverse=True)[:2]
                dims = sorted([top[0][0], top[1][0]])
                print(f"Auto-selected dims from logistic probe: z{dims[0]}, z{dims[1]} "
                      f"(AUC {top[0][1]:.3f}, {top[1][1]:.3f})")
                return dims
        except Exception as e:
            print(f"Could not auto-select dims from {cache.name}: {e}")
    return [4, 7] if n_latent > 7 else [0, max(0, min(1, n_latent - 1))]


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
    parser.add_argument("--dims", nargs=2, type=int, default=None, metavar=("ZA", "ZB"),
                        help="Latent dimensions for the direct z-scatter plots "
                             "(default: top-2 discriminating dims from the logistic probe cache, "
                             "falling back to 4 7)")
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
                    c=COLORS["Muon"], label="MIPs", **sc_main)
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

    # ── Plots 3 & 4: direct latent-dimension scatters ─────────────────────────
    n_latent = train_latents.shape[1]
    za, zb = resolve_dims(args.dims, n_latent, out_dir)
    if za >= n_latent or zb >= n_latent:
        print(f"Skipping z{za}/z{zb} scatter plots: model has only {n_latent} latent dims "
              f"(pick others with --dims).")
    else:
        # ── Plot 3: za vs zb — all species ────────────────────────────────────
        proton_latents = np.vstack([train_latents, val_latents])
        fig3, ax3 = plt.subplots(figsize=(fig_w, fig_h))
        ax3.scatter(proton_latents[:, za], proton_latents[:, zb],
                    c=COLORS["Proton (Train)"], label="Proton", **sc_main)
        ax3.scatter(kaon_latents[:, za], kaon_latents[:, zb],
                    c=COLORS["Kaon"], label="Kaon", **sc_main)
        if muon_latents is not None:
            ax3.scatter(muon_latents[:, za], muon_latents[:, zb],
                        c=COLORS["Muon"], label="Muon", **sc_main)
        if csda_kaon_latents is not None:
            ax3.scatter(csda_kaon_latents[:, za], csda_kaon_latents[:, zb],
                        c=COLORS["CSDA-Kaon"], label="CSDA-Kaon", **sc_csda)
        ax3.set_xlabel(rf"$z_{{{za}}}$")
        ax3.set_ylabel(rf"$z_{{{zb}}}$")
        style_legend(ax3.legend(**make_legend_kwargs()))
        sns.despine(ax=ax3)
        fig3.tight_layout()
        save(fig3, out_dir / f"z{za}_vs_z{zb}_all_species")
        plt.close(fig3)

        # ── Plot 4: za vs zb — train vs val protons ──────────────────────────
        fig4, ax4 = plt.subplots(figsize=(fig_w, fig_h))
        ax4.scatter(train_latents[:, za], train_latents[:, zb],
                    c=COLORS["Proton (Train)"], label="Proton (train)", **sc_main)
        ax4.scatter(val_latents[:, za], val_latents[:, zb],
                    c=COLORS["Proton (Val)"], label="Proton (val)", **sc_main)
        ax4.set_xlabel(rf"$z_{{{za}}}$")
        ax4.set_ylabel(rf"$z_{{{zb}}}$")
        style_legend(ax4.legend(**make_legend_kwargs()))
        sns.despine(ax=ax4)
        fig4.tight_layout()
        save(fig4, out_dir / f"z{za}_vs_z{zb}_proton_train_val")
        plt.close(fig4)

    # ── All-species extras: per-species train/val UMAP + recon errors ─────────
    ss_path = inf_dir / "species_split.npz"
    if ss_path.exists() and muon_umap is not None:
        ss = np.load(ss_path)
        species_umaps = [
            ("Proton", train_umap, val_umap, COLORS["Proton (Train)"]),
            ("Kaon",  kaon_umap[ss["k_train_idx"]], kaon_umap[ss["k_val_idx"]], COLORS["Kaon"]),
            ("Muon",  muon_umap[ss["m_train_idx"]], muon_umap[ss["m_val_idx"]], COLORS["Muon"]),
        ]

        # Plot 5: does each species' val set live where its train set does?
        fig5, axes5 = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.6),
                                   sharex=True, sharey=True)
        for ax, (sp_name, tr_emb, va_emb, colour) in zip(axes5, species_umaps):
            ax.scatter(tr_emb[:, 0], tr_emb[:, 1], c="0.7",
                       label=f"train (n={len(tr_emb)})", **sc_main)
            ax.scatter(va_emb[:, 0], va_emb[:, 1], c=colour,
                       label=f"val (n={len(va_emb)})", **sc_main)
            ax.set_title(sp_name, fontsize=9)
            ax.set_xlabel("UMAP 1")
            style_legend(ax.legend(**make_legend_kwargs()), marker_size=15)
            sns.despine(ax=ax)
        axes5[0].set_ylabel("UMAP 2")
        fig5.tight_layout()
        save(fig5, out_dir / "umap_train_val_by_species")
        plt.close(fig5)

        # Plot 6: reconstruction-error distributions per species, train vs val
        kaon_re = np.load(inf_dir / "kaon.npz")["re"]
        muon_re = np.load(inf_dir / "muon.npz")["re"]
        species_res = [
            ("Proton", np.load(inf_dir / "train.npz")["re"],
                       np.load(inf_dir / "val.npz")["re"],   COLORS["Proton (Train)"]),
            ("Kaon",  kaon_re[ss["k_train_idx"]], kaon_re[ss["k_val_idx"]], COLORS["Kaon"]),
            ("Muon",  muon_re[ss["m_train_idx"]], muon_re[ss["m_val_idx"]], COLORS["Muon"]),
        ]
        re_max = np.percentile(np.concatenate([np.concatenate([tr, va])
                                               for _, tr, va, _ in species_res]), 99)
        bins = np.linspace(0, re_max, 41)

        fig6, axes6 = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.4), sharey=True)
        for ax, (sp_name, tr_re, va_re, colour) in zip(axes6, species_res):
            ax.hist(tr_re, bins=bins, density=True, color="0.7", alpha=0.7,
                    histtype="stepfilled", label=f"train (n={len(tr_re)})")
            ax.hist(va_re, bins=bins, density=True, color=colour, alpha=0.55,
                    histtype="stepfilled", label=f"val (n={len(va_re)})")
            ax.hist(va_re, bins=bins, density=True, color=colour,
                    histtype="step", linewidth=1.2)
            ax.set_title(sp_name, fontsize=9)
            ax.set_xlabel("Reconstruction error")
            leg = ax.legend(**make_legend_kwargs())
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)
            sns.despine(ax=ax)
        axes6[0].set_ylabel("Density")
        fig6.tight_layout()
        save(fig6, out_dir / "recon_error_by_species")
        plt.close(fig6)

    print("Done.")


if __name__ == "__main__":
    main()