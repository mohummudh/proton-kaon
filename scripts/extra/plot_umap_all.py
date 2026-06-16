import argparse
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
    out_dir = Path("figs") / model_name / "latents-features"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Load latents
    try:
        train_latents = np.load(inf_dir / "train.npz")["latents"]
        val_latents = np.load(inf_dir / "val.npz")["latents"]
        kaon_latents = np.load(inf_dir / "kaon.npz")["latents"]
        print(f"Loaded {len(train_latents)} protons (train), {len(val_latents)} protons (val), and {len(kaon_latents)} kaon candidates.")
    except FileNotFoundError as e:
        print(f"Error: Could not find inference files in {inf_dir}")
        print(e)
        return

    muon_path = inf_dir / "muon.npz"
    muon_latents = None
    if muon_path.exists():
        muon_latents = np.load(muon_path)["latents"]
        print(f"Loaded {len(muon_latents)} muons.")

    csda_kaon_path = inf_dir / "csda_kaon.npz"
    csda_kaon_latents = None
    if csda_kaon_path.exists():
        csda_kaon_latents = np.load(csda_kaon_path)["latents"]
        print(f"Loaded {len(csda_kaon_latents)} csda-kaons.")

    # UMAP Reducer
    reducer_path = inf_dir / 'reducer.pkl'
    if reducer_path.exists():
        import pickle
        with open(reducer_path, 'rb') as f:
            reducer = pickle.load(f)
        print(f"Loaded existing UMAP reducer from {reducer_path}")
    else:
        print("Training new UMAP reducer (this will take a moment)...")
        all_latents_list = [train_latents, val_latents, kaon_latents]
        if muon_latents is not None:
            all_latents_list.append(muon_latents)
        if csda_kaon_latents is not None:
            all_latents_list.append(csda_kaon_latents)
        all_latents = np.vstack(all_latents_list)
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=42)
        reducer.fit(all_latents)
        import pickle
        with open(reducer_path, 'wb') as f:
            pickle.dump(reducer, f)
        print("Fitted and saved new UMAP reducer.")

    print("Transforming latents...")
    train_umap = reducer.transform(train_latents)
    val_umap = reducer.transform(val_latents)
    kaon_umap = reducer.transform(kaon_latents)
    muon_umap = reducer.transform(muon_latents) if muon_latents is not None else None
    csda_kaon_umap = reducer.transform(csda_kaon_latents) if csda_kaon_latents is not None else None

    # Plotting
    print("Plotting...")

    colors = {
        "Proton (Train)": "#4C78A8",
        "Proton (Val)":   "#FB0019",
        "Kaon":           "#F58518",
        "Muon":           "#76B7B2",
        "CSDA-Kaon":      "#D62728",
    }

    def _style_legend(leg):
        for lh in leg.legend_handles:
            lh.set_alpha(1.0)
            lh.set_sizes([50])

    # ── Plot 1: training proton vs val proton only ────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1.scatter(train_umap[:, 0], train_umap[:, 1], s=12, alpha=0.7, c=colors["Proton (Train)"], label="Proton (Train)", linewidths=0)
    ax1.scatter(val_umap[:, 0],   val_umap[:, 1],   s=12, alpha=0.7, c=colors["Proton (Val)"],   label="Proton (Val)",   linewidths=0)
    ax1.set_title("UMAP Projection — Proton Train vs Val", fontsize=14, fontweight="bold")
    ax1.set_xlabel("UMAP 1")
    ax1.set_ylabel("UMAP 2")
    _style_legend(ax1.legend(frameon=True, framealpha=0.9, title="Particle Species"))
    sns.despine(ax=ax1)
    fig1.tight_layout()
    out1 = out_dir / "umap_proton_train_val.png"
    fig1.savefig(out1, dpi=150, bbox_inches="tight")
    print(f"Saved train/val proton plot to {out1}")

    # ── Plot 2: all protons + kaons + muons ───────────────────────────────────
    proton_umap = np.vstack([train_umap, val_umap])
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2.scatter(proton_umap[:, 0], proton_umap[:, 1], s=12, alpha=0.7, c=colors["Proton (Train)"], label="Proton", linewidths=0)
    ax2.scatter(kaon_umap[:, 0],   kaon_umap[:, 1],   s=12, alpha=0.7, c=colors["Kaon"],           label="Kaon",   linewidths=0)
    if muon_umap is not None:
        ax2.scatter(muon_umap[:, 0], muon_umap[:, 1], s=12, alpha=0.7, c=colors["Muon"], label="Muon", linewidths=0)
    if csda_kaon_umap is not None:
        ax2.scatter(csda_kaon_umap[:, 0], csda_kaon_umap[:, 1], s=24, alpha=0.9, c=colors["CSDA-Kaon"], label="CSDA-Kaon", linewidths=0)
    ax2.set_title("UMAP Projection of Latent Space", fontsize=14, fontweight="bold")
    ax2.set_xlabel("UMAP 1")
    ax2.set_ylabel("UMAP 2")
    _style_legend(ax2.legend(frameon=True, framealpha=0.9, title="Particle Species"))
    sns.despine(ax=ax2)
    fig2.tight_layout()
    out2 = out_dir / "umap_all_species.png"
    fig2.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved all-species UMAP plot to {out2}")

    # ── Plot 3: z4 vs z7 — protons + kaons + muons ───────────────────────────
    proton_latents = np.vstack([train_latents, val_latents])
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    ax3.scatter(proton_latents[:, 4], proton_latents[:, 7], s=12, alpha=0.5, c=colors["Proton (Train)"], label="Proton",  linewidths=0)
    ax3.scatter(kaon_latents[:, 4],   kaon_latents[:, 7],   s=12, alpha=0.7, c=colors["Kaon"],           label="Kaon",    linewidths=0)
    if muon_latents is not None:
        ax3.scatter(muon_latents[:, 4], muon_latents[:, 7], s=12, alpha=0.7, c=colors["Muon"],           label="Muon",    linewidths=0)
    if csda_kaon_latents is not None:
        ax3.scatter(csda_kaon_latents[:, 4], csda_kaon_latents[:, 7], s=24, alpha=0.9, c=colors["CSDA-Kaon"], label="CSDA-Kaon", linewidths=0)
    ax3.set_title("Latent Space — z4 vs z7 (all species)", fontsize=14, fontweight="bold")
    ax3.set_xlabel("z4")
    ax3.set_ylabel("z7")
    _style_legend(ax3.legend(frameon=True, framealpha=0.9, title="Particle Species"))
    sns.despine(ax=ax3)
    fig3.tight_layout()
    out3 = out_dir / "z4_vs_z7_all_species.png"
    fig3.savefig(out3, dpi=150, bbox_inches="tight")
    print(f"Saved z4 vs z7 (all species) to {out3}")

    # ── Plot 4: z4 vs z7 — training vs val protons ────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    ax4.scatter(train_latents[:, 4], train_latents[:, 7], s=12, alpha=0.7, c=colors["Proton (Train)"], label="Proton (Train)", linewidths=0)
    ax4.scatter(val_latents[:, 4],   val_latents[:, 7],   s=12, alpha=0.7, c=colors["Proton (Val)"],   label="Proton (Val)",   linewidths=0)
    ax4.set_title("Latent Space — z4 vs z7 (train vs val protons)", fontsize=14, fontweight="bold")
    ax4.set_xlabel("z4")
    ax4.set_ylabel("z7")
    _style_legend(ax4.legend(frameon=True, framealpha=0.9, title="Particle Species"))
    sns.despine(ax=ax4)
    fig4.tight_layout()
    out4 = out_dir / "z4_vs_z7_proton_train_val.png"
    fig4.savefig(out4, dpi=150, bbox_inches="tight")
    print(f"Saved z4 vs z7 (train/val protons) to {out4}")
    

if __name__ == "__main__":
    main()
