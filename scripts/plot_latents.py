import argparse
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to model YAML config")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = build_model_name(cfg)
    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name
    out_dir = Path("figs") / "latents-features" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load latents
    try:
        val_latents = np.load(inf_dir / "val.npz")["latents"]
        kaon_latents = np.load(inf_dir / "kaon.npz")["latents"]
        print(f"Loaded {len(val_latents)} protons (val) and {len(kaon_latents)} kaon candidates.")
    except FileNotFoundError as e:
        print(f"Error: Could not find inference files in {inf_dir}")
        print(e)
        return

    muon_path = inf_dir / "muon.npz"
    muon_latents = None
    if muon_path.exists():
        muon_latents = np.load(muon_path)["latents"]
        print(f"Loaded {len(muon_latents)} muons.")

    n_dims = val_latents.shape[1]
    dim_names = [f"z{i}" for i in range(n_dims)]
    colors = {"Proton": "#4C78A8", "Kaon": "#F58518", "Muon": "#76B7B2"}

    # Prepare DataFrame for seaborn
    # We want Kaons at the bottom, then Protons, then Muons.
    # Seaborn/Matplotlib z-order: items EARLIER in the list are plotted FIRST (at the bottom).
    species_order = ["Kaon", "Proton", "Muon"] if muon_latents is not None else ["Kaon", "Proton"]
    
    dfs = []
    dfs.append(pd.DataFrame(kaon_latents, columns=dim_names).assign(Species="Kaon"))
    dfs.append(pd.DataFrame(val_latents, columns=dim_names).assign(Species="Proton"))
    if muon_latents is not None:
        dfs.append(pd.DataFrame(muon_latents, columns=dim_names).assign(Species="Muon"))
    
    df = pd.concat(dfs, ignore_index=True)
    df["Species"] = pd.Categorical(df["Species"], categories=species_order, ordered=True)

    # 1. ── 1D Histograms ───────────────────────────────────────────────────────
    fig_1d, axes_1d = plt.subplots(1, n_dims, figsize=(n_dims * 4, 4))
    if n_dims == 1: axes_1d = [axes_1d]

    for i in range(n_dims):
        ax = axes_1d[i]
        sns.histplot(data=df, x=f"z{i}", hue="Species", hue_order=species_order, palette=colors,
                     element="step", kde=True, stat="density", alpha=0.4, ax=ax,
                     legend=(i == n_dims - 1))
        ax.set_title(f"Latent z{i}", fontweight="bold")
        if i == n_dims - 1:
            sns.move_legend(ax, "upper right", frameon=True, framealpha=0.9)

    plt.tight_layout()
    fig_1d.savefig(out_dir / "latent_1d_histograms.png", dpi=150)
    print(f"Saved 1D histograms to {out_dir / 'latent_1d_histograms.png'}")

    # 2. ── 2D Pair Grid (Corner Plot) ──────────────────────────────────────────
    print("Generating 2D pair grid (this may take a minute for many dimensions)...")
    
    # Customizing the PairGrid for a "Corner Plot" style
    g = sns.PairGrid(df, hue="Species", hue_order=species_order, palette=colors, corner=True, diag_sharey=False)
    
    # KDEs on the diagonal
    g.map_diag(sns.kdeplot, fill=True, alpha=0.3)
    
    # 2D KDEs or Scatter on the lower triangle
    g.map_lower(sns.scatterplot, s=8, alpha=0.15, linewidth=0)
    g.map_lower(sns.kdeplot, levels=4, alpha=0.6, linewidths=1.2)

    g.add_legend(title="Particle Species", adjust_subtitles=True)
    for lh in g._legend.legend_handles:
        lh.set_alpha(1.0) # Full opacity in legend
        lh.set_linewidth(4) # Thicker lines in legend for visibility
        
    g.fig.suptitle(f"Latent Space Pairwise Distributions\nModel: {model_name}", y=1.05, fontsize=14, fontweight="bold")
    
    g.savefig(out_dir / "latent_2d_grid.png", dpi=150, bbox_inches="tight")
    print(f"Saved 2D grid to {out_dir / 'latent_2d_grid.png'}")

if __name__ == "__main__":
    main()
