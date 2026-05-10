import argparse
import yaml
import numpy as np
import pandas as pd
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
    parser.add_argument("--features", default=None, help="Path to features.pkl (optional)")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model_name = build_model_name(cfg)
    inf_dir = Path(cfg["output"]["inference_dir"]) / model_name
    out_dir = Path("figs") / "latents-features" / model_name
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

    # Load Features for log-likelihood filtering
    features_path = (
        args.features
        or cfg.get("data", {}).get("features_path")
        or "/Volumes/easystore/proton-kaon/features/features.pkl"
    )
    features_file = Path(features_path)
    split_path = Path(cfg["output"]["splits_dir"]) / "split_p.npz"
    
    has_features = False
    if features_file.exists() and split_path.exists():
        print("Loading features to filter by log-likelihoods...")
        features = pd.read_pickle(features_file)
        index = np.load(split_path)
        
        all_proton = features[features["particle_type"] == "proton"]
        all_kaon = features[features["particle_type"] == "kaon"]
        
        train_features = all_proton.iloc[index["train_idx"]]
        val_features = all_proton.iloc[index["val_idx"]]
        kaon_features = all_kaon
        
        train_mask = train_features[['log_likelihood_kaon', 'log_likelihood_proton']].notna().all(axis=1).values
        val_mask = val_features[['log_likelihood_kaon', 'log_likelihood_proton']].notna().all(axis=1).values
        kaon_mask = kaon_features[['log_likelihood_kaon', 'log_likelihood_proton']].notna().all(axis=1).values
        
        train_umap_ll = train_umap[train_mask]
        val_umap_ll = val_umap[val_mask]
        kaon_umap_ll = kaon_umap[kaon_mask]
        
        print(f"Filtered to {len(train_umap_ll)} protons (train), {len(val_umap_ll)} protons (val), and {len(kaon_umap_ll)} kaon candidates.")
        has_features = True
    else:
        print("Features or split file not found. Skipping LL-filtered plot.")

    # Plotting
    print("Plotting...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Colors
    colors = {
        "Proton (Train)": "#4C78A8", # Dark blue
        "Kaon": "#F58518",           # Orange
        "Proton (Val)": "#FB0019",   # Light red
        "Muon": "#76B7B2"            # Teal
    }
    
    # Layering order (bottom to top): Train Protons -> Kaons -> Val Protons -> Muons
    ax.scatter(train_umap[:, 0], train_umap[:, 1], s=12, alpha=0.7, c=colors["Proton (Train)"], label="Proton (Train)", linewidths=0)
    ax.scatter(val_umap[:, 0], val_umap[:, 1], s=12, alpha=0.7, c=colors["Proton (Val)"], label="Proton (Val)", linewidths=0)
    ax.scatter(kaon_umap[:, 0], kaon_umap[:, 1], s=12, alpha=0.7, c=colors["Kaon"], label="Kaon", linewidths=0)
    
    if muon_umap is not None:
        ax.scatter(muon_umap[:, 0], muon_umap[:, 1], s=12, alpha=0.7, c=colors["Muon"], label="Muon", linewidths=0)
        
    ax.set_title(f"UMAP Projection of Latent Space\nModel: {model_name}", fontsize=14, fontweight="bold")
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    
    # Legend improvements
    leg = ax.legend(frameon=True, framealpha=0.9, title="Particle Species")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
        lh.set_sizes([50]) # Set a fixed, clean size for the legend markers
        
    sns.despine()
    plt.tight_layout()
    
    out_path = out_dir / "umap_all_species.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved unified UMAP plot to {out_path}")
    
    if has_features:
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        
        ax2.scatter(train_umap_ll[:, 0], train_umap_ll[:, 1], s=12, alpha=0.7, c=colors["Proton (Train)"], label="Proton (Train)", linewidths=0)
        ax2.scatter(val_umap_ll[:, 0], val_umap_ll[:, 1], s=12, alpha=0.7, c=colors["Proton (Val)"], label="Proton (Val)", linewidths=0)
        ax2.scatter(kaon_umap_ll[:, 0], kaon_umap_ll[:, 1], s=12, alpha=0.7, c=colors["Kaon"], label="Kaon", linewidths=0)
        
        ax2.set_title(f"UMAP Projection (Events with Likelihoods)\nModel: {model_name}", fontsize=14, fontweight="bold")
        ax2.set_xlabel("UMAP 1")
        ax2.set_ylabel("UMAP 2")
        
        leg2 = ax2.legend(frameon=True, framealpha=0.9, title="Particle Species")
        for lh in leg2.legend_handles:
            lh.set_alpha(1.0)
            lh.set_sizes([50])
            
        sns.despine(ax=ax2)
        fig2.tight_layout()
        
        out_path2 = out_dir / "umap_ll_only.png"
        fig2.savefig(out_path2, dpi=150, bbox_inches="tight")
        print(f"Saved LL-filtered UMAP plot to {out_path2}")

if __name__ == "__main__":
    main()
