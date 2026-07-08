import yaml
import argparse
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Subset, DataLoader

from src.models.configVAE import VAE
from src.inference.inference import inference
from src.transforms import apply_transform

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--include-muons", action="store_true", help="Also run inference on muon images (>=180 wires)")
parser.add_argument("--muon-image-path", default="/Volumes/easystore/proton-kaon/images/muon_48x48_raw.pt", help="Path to muon image file (must align with muon_col.pkl rows in features.pkl)")
parser.add_argument("--csda-kaon-path", default=None, help="Path to csda-kaon image file (csv_kaon_48x48_raw.pt)")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

all_species = cfg["data"].get("proton") == "all"

save_dir = Path(cfg["output"]["dir"])
save_dir.mkdir(parents=True, exist_ok=True)
species_tag = "_speciesall" if all_species else ""
name = (
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
    f"_tx{cfg['data'].get('transform', 'none')}{species_tag}.pt"
)
save_path = save_dir / name

# LOADING TRAINING DATA + VALIDATION DATA
splits_dir = Path(cfg["output"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)
proton_key = "p" if all_species else cfg["data"]["proton"]
kaon_key = "k" if all_species else cfg["data"]["kaon"]

transform = cfg["data"].get("transform", "none")

data = torch.load(cfg["data"]["path"], map_location="cpu")
p = apply_transform(data[proton_key], transform)
kaons = apply_transform(data[kaon_key], transform)

if all_species:
    # Train/val membership follows the all-species split over the concatenated
    # [p, k, m] tensor used in training (split_all.npz). Derive per-species
    # within-tensor indices so every species has a train/val separation.
    muons_all = apply_transform(data["m"], transform)
    n_p, n_k, n_m = len(p), len(kaons), len(muons_all)
    split_path = splits_dir / "split_all.npz"
    if split_path.exists():
        split = np.load(split_path)
        all_train_idx, all_val_idx = split["train_idx"], split["val_idx"]
    else:
        # regenerate deterministically with the same call as run_training.py
        from sklearn.model_selection import train_test_split
        print(f"WARNING: {split_path} not found — regenerating deterministically "
              f"(val_split={cfg['data']['val_split']}, seed={cfg['data']['random_seed']})")
        all_train_idx, all_val_idx = train_test_split(
            np.arange(n_p + n_k + n_m),
            test_size=cfg["data"]["val_split"], random_state=cfg["data"]["random_seed"],
        )
        np.savez(split_path, train_idx=all_train_idx, val_idx=all_val_idx)

    def _species_idx(idx, lo, hi):
        return np.sort(idx[(idx >= lo) & (idx < hi)] - lo)

    train_idx   = _species_idx(all_train_idx, 0, n_p)  # protons
    val_idx     = _species_idx(all_val_idx,   0, n_p)
    k_train_idx = _species_idx(all_train_idx, n_p, n_p + n_k)
    k_val_idx   = _species_idx(all_val_idx,   n_p, n_p + n_k)
    m_train_idx = _species_idx(all_train_idx, n_p + n_k, n_p + n_k + n_m)
    m_val_idx   = _species_idx(all_val_idx,   n_p + n_k, n_p + n_k + n_m)
    print(f"All-species split | p: {len(train_idx)}/{len(val_idx)}  "
          f"k: {len(k_train_idx)}/{len(k_val_idx)}  m: {len(m_train_idx)}/{len(m_val_idx)} (train/val)")
else:
    split = np.load(splits_dir / f"split_{proton_key}.npz")
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

train_subset = Subset(p, train_idx)
train_loader = DataLoader(train_subset, batch_size=cfg["train"]["batch_size"], shuffle=False)

val_subset = Subset(p, val_idx)
val_loader = DataLoader(val_subset, batch_size=cfg["train"]["batch_size"], shuffle=False)

# LOAD MODEL
model = VAE(
    input_hw=tuple(cfg["model"]["input_hw"]),
    latent=cfg["model"]["latent"],
    channels=cfg["model"]["channels"],
    kernel=cfg["model"]["kernel"],
    stride=cfg["model"]["stride"],
    padding=cfg["model"]["padding"],
    activation=cfg["model"]["activation"],
    p_enc=cfg["model"].get("dropout", 0.0),
).to(device)
model.load_state_dict(torch.load(save_path, map_location=device))

# RUN INFERENCE
train_latents, train_recon, train_re = inference(model, train_subset)
val_latents, val_recon, val_re = inference(model, val_subset)
kaon_latents, kaon_recon, kaon_re = inference(model, kaons)

inference_dir = Path(cfg["output"]["inference_dir"]) / name.replace(".pt", "")
inference_dir.mkdir(parents=True, exist_ok=True)

np.savez(inference_dir / "train.npz",
    latents=train_latents, recon=train_recon, re=train_re)

np.savez(inference_dir / "val.npz",
    latents=val_latents, recon=val_recon, re=val_re)

np.savez(inference_dir / "kaon.npz",
    latents=kaon_latents, recon=kaon_recon, re=kaon_re)

if all_species:
    # per-species train/val indices for downstream analysis
    np.savez(inference_dir / "species_split.npz",
        p_train_idx=train_idx, p_val_idx=val_idx,
        k_train_idx=k_train_idx, k_val_idx=k_val_idx,
        m_train_idx=m_train_idx, m_val_idx=m_val_idx)

    # muons come from the training data file itself (aligned with muon_col.pkl)
    muon_latents, muon_recon, muon_re = inference(model, muons_all)
    np.savez(inference_dir / "muon.npz",
        latents=muon_latents, recon=muon_recon, re=muon_re)
    print(f"Saved muon inference from training data file: {len(muon_latents)} images")

# RUN INFERENCE ON MUONS (if requested; all-species mode already saved them above)
if args.include_muons and not all_species:
    muon_data = torch.load(args.muon_image_path, map_location="cpu")
    muons = apply_transform(muon_data["m"], transform)
    muon_latents, muon_recon, muon_re = inference(model, muons)

    np.savez(inference_dir / "muon.npz",
        latents=muon_latents, recon=muon_recon, re=muon_re)
    print(f"Saved muon inference: {len(muon_latents)} images")

if args.csda_kaon_path:
    csda_data = torch.load(args.csda_kaon_path, map_location="cpu")
    csda_kaons = apply_transform(csda_data["k"], transform)
    csda_latents, csda_recon, csda_re = inference(model, csda_kaons)
    np.savez(inference_dir / "csda_kaon.npz",
        latents=csda_latents, recon=csda_recon, re=csda_re)
    print(f"Saved csda-kaon inference: {len(csda_latents)} images")

