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
parser.add_argument("--muon-image-path", default="/Volumes/easystore/proton-kaon/images/muon_48x48_raw_180+wires.pt", help="Path to muon image file")
parser.add_argument("--csda-kaon-path", default=None, help="Path to csda-kaon image file (csv_kaon_48x48_raw.pt)")
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
    f"_pad{cfg['model']['padding']}"
    f"_hw{'x'.join(str(d) for d in cfg['model']['input_hw'])}"
    f"_tx{cfg['data'].get('transform', 'none')}.pt"
)
save_path = save_dir / name

# LOADING TRAINING DATA + VALIDATION DATA
splits_dir = Path(cfg["output"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)
split_path = splits_dir / f"split_{cfg['data']['proton']}.npz"

split = np.load(split_path)
train_idx = split["train_idx"]
val_idx = split["val_idx"]

transform = cfg["data"].get("transform", "none")

data = torch.load(cfg["data"]["path"], map_location="cpu")
p = apply_transform(data[cfg["data"]["proton"]], transform)
kaons = apply_transform(data[cfg["data"]["kaon"]], transform)

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

# RUN INFERENCE ON MUONS (if requested)
if args.include_muons:
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

