import yaml
import argparse

import torch
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Subset, DataLoader
from pathlib import Path

from src.models.configVAE import VAE
from src.losses.vae import vae_loss
from src.train.train import train
from src.train.plot import plot_training
from src.train.logger import save_run_log
from src.transforms import apply_transform

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--latent", type=int)
parser.add_argument("--beta", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--proton", type=str)
parser.add_argument("--channels", nargs='+', type=int)
parser.add_argument("--kernel", type=int)
parser.add_argument("--activation", type=str)
parser.add_argument("--transform", type=str)
parser.add_argument("--all", action="store_true", dest="all_species",
                    help="Train on all three species (proton + kaon + muon) concatenated.")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.latent:      cfg["model"]["latent"]    = args.latent
if args.beta:        cfg["train"]["beta"]       = args.beta
if args.lr:          cfg["optimizer"]["lr"]     = args.lr
if args.epochs:      cfg["train"]["epochs"]     = args.epochs
if args.batch_size:  cfg["train"]["batch_size"] = args.batch_size
if args.proton:      cfg["data"]["proton"]      = args.proton
if args.channels:    cfg["model"]["channels"]   = args.channels
if args.kernel:      cfg["model"]["kernel"]     = args.kernel
if args.activation:  cfg["model"]["activation"] = args.activation
if args.transform:   cfg["data"]["transform"]   = args.transform
if args.all_species: cfg["data"]["proton"]      = "all"

transform = cfg["data"].get("transform", "none")
target_hw = tuple(cfg["model"]["input_hw"])

def _load_and_prep(tensor):
    tensor = apply_transform(tensor, transform)
    if tensor.shape[-2:] != target_hw:
        tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return tensor

data = torch.load(cfg["data"]["path"], map_location="cpu")

if cfg["data"]["proton"] == "all":
    p_raw = _load_and_prep(data["p"])
    k_raw = _load_and_prep(data["k"])
    m_raw = _load_and_prep(data["m"])
    p = torch.cat([p_raw, k_raw, m_raw], dim=0)
    print(f"All-species mode | p={len(p_raw)} k={len(k_raw)} m={len(m_raw)} total={len(p)}")
    print(f"Transform applied: {transform}  |  range [{p.min():.4f}, {p.max():.4f}]")
else:
    p = _load_and_prep(data[cfg["data"]["proton"]])
    print(f"Transform applied: {transform}  |  p range [{p.min():.4f}, {p.max():.4f}]")
    if data[cfg["data"]["proton"]].shape[-2:] != target_hw:
        print(f"Resized to {target_hw}")

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

EPOCHS = cfg["train"]["epochs"]
BATCH_SIZE = cfg["train"]["batch_size"]
BETA = cfg["train"]["beta"]
LATENT = cfg["model"]["latent"]

splits_dir = Path(cfg["output"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)
split_path = splits_dir / f"split_{cfg['data']['proton']}.npz"

if split_path.exists():
    split = np.load(split_path)
    train_idx, val_idx = split["train_idx"], split["val_idx"]
else:
    all_indices = np.arange(len(p))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=cfg["data"]["val_split"], random_state=cfg["data"]["random_seed"]
    )
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx)

train_subset = Subset(p, train_idx)
val_subset = Subset(p, val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

model = VAE(input_hw=tuple(cfg["model"]["input_hw"]),
            latent=LATENT,
            channels=cfg["model"]["channels"],
            kernel=cfg["model"]["kernel"],
            stride=cfg["model"]["stride"],
            padding=cfg["model"]["padding"],
            activation=cfg["model"]["activation"],
            p_enc=cfg["model"]["dropout"]).to(device)

optim = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])

model, train_losses, train_recon, train_kl, val_losses, val_recon, val_kl = train(
    device, train_loader, val_loader, model, optim, vae_loss,
    epochs=EPOCHS, beta=BETA,
    patience=cfg["train"].get("patience", 20),
    min_delta=cfg["train"].get("min_delta", 1e-4),
)

save_dir = Path(cfg["output"]["dir"])
save_dir.mkdir(parents=True, exist_ok=True)
_species_tag = "_speciesall" if cfg["data"]["proton"] == "all" else ""
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
    f"_tx{cfg['data'].get('transform', 'none')}{_species_tag}.pt"
)
save_path = save_dir / name

# Save model
torch.save(model.state_dict(), save_path)
print(f"\nModel saved as {save_path}")

plot_training(
    train_losses, train_recon, train_kl,
    val_losses,   val_recon,   val_kl,
    save_path=save_dir / name.replace(".pt", "_curves.png")
)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)

save_run_log(
    cfg, device, train_subset, val_subset,
    train_losses, train_recon, train_kl,
    val_losses, val_recon, val_kl,
    save_path=log_dir / name.replace(".pt", ".json")
)