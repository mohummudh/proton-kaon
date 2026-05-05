import yaml
import argparse
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Subset, DataLoader

from src.models.configVAE import VAE
from src.inference.inference import inference

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

save_dir = Path(cfg["output"]["dir"])
save_dir.mkdir(parents=True, exist_ok=True)
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
    f"_pad{cfg['model']['padding']}.pt"
)
save_path = save_dir / name

# LOADING TRAINING DATA + VALIDATION DATA
splits_dir = Path(cfg["output"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)
split_path = splits_dir / f"split_{cfg['data']['proton']}.npz"

split = np.load(split_path)
train_idx = split["train_idx"]
val_idx = split["val_idx"]

data = torch.load(cfg["data"]["path"], map_location="cpu")
p = data[cfg["data"]["proton"]]

train_subset = Subset(p, train_idx)
train_loader = DataLoader(train_subset, batch_size=cfg["train"]["batch_size"], shuffle=False)

val_subset = Subset(p, val_idx)
val_loader = DataLoader(val_subset, batch_size=cfg["train"]["batch_size"], shuffle=False)

# KAON DATA
kaons = data[cfg["data"]["kaon"]]

# LOAD MODEL
model = VAE(input_hw=tuple(cfg["model"]["input_hw"]), latent=cfg["model"]["latent"]).to(device)
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

