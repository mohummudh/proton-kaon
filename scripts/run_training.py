import yaml
import argparse

import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from pathlib import Path

from src.models.vae import VAE
from src.losses.vae import vae_loss
from src.train.train import train

from sklearn.model_selection import train_test_split

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--latent", type=int)
parser.add_argument("--beta", type=float)
parser.add_argument("--lr", type=float)
parser.add_argument("--epochs", type=int)
parser.add_argument("--batch_size", type=int)
parser.add_argument("--particle", type=str)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.latent:     cfg["model"]["latent"]    = args.latent
if args.beta:       cfg["train"]["beta"]       = args.beta
if args.lr:         cfg["optimizer"]["lr"]     = args.lr
if args.epochs:     cfg["train"]["epochs"]     = args.epochs
if args.batch_size: cfg["train"]["batch_size"] = args.batch_size
if args.particle:   cfg["data"]["particle"]    = args.particle

out = cfg["data"]["path"]
data = torch.load(out, map_location="cpu")

p = data[cfg["data"]["particle"]]

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training device:", device)

EPOCHS = cfg["train"]["epochs"]
BATCH_SIZE = cfg["train"]["batch_size"]
BETA = cfg["train"]["beta"]
LATENT = cfg["train"]["latent"]

all_indices = np.arange(len(p))
train_idx, val_idx = train_test_split(all_indices, test_size=cfg["data"]["val_split"], random_state=cfg["data"]["random_seed"])

train_subset = Subset(p, train_idx)
val_subset = Subset(p, val_idx)

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

model = VAE(input_hw=(48, 48), latent=LATENT).to(device)
optim = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])

model, _, _, _, _, _, _ = train(device, train_loader, val_loader, model, optim, vae_loss, epochs=EPOCHS, beta=BETA)

save_dir = Path(cfg["output"]["dir"])
save_dir.mkdir(parents=True, exist_ok=True)
name = (
    f"model_{cfg['model']['type']}"
    f"_latent{cfg['model']['latent']}"
    f"_beta{cfg['train']['beta']}"
    f"_lr{cfg['optimizer']['lr']}"
    f"_epoch{cfg['train']['epochs']}.pt"
)
save_path = save_dir / name

# Save model
torch.save(model.state_dict(), save_path)
print(f"\nModel saved as {save_path}")