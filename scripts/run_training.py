import yaml
import argparse
import random

import torch
import numpy as np

from torch.utils.data import Subset, DataLoader
from pathlib import Path

from src.device import pick_device
from src.models.build import build_vae
from src.losses.vae import vae_loss
from src.train.train import train
from src.train.plot import plot_training
from src.train.logger import save_run_log
from src.train.naming import model_filename, split_filename
from src.transforms import prepare_images

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
parser.add_argument("--seed", type=int,
                    help="Seed weight init, dropout and shuffle order, making the run "
                         "reproducible and adding _seed<N> to the model name. Omit to "
                         "keep the historical unseeded behaviour.")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

if args.seed is not None: cfg["train"]["seed"] = args.seed
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

# ── seeding ───────────────────────────────────────────────────────────────────
# `train.seed` is optional, and its absence is a supported state rather than an
# oversight: every model trained before this existed was unseeded, and defaulting
# it to some value would silently change what those configs produce. When it is
# absent nothing is seeded and the model name carries no seed tag, so old runs
# reproduce the same pipeline they always did.
#
# Note `data.random_seed` is a different knob: it only feeds the sklearn
# train_test_split fallback below, which a tagged split never reaches. It has
# never controlled anything about the model.
#
# cudnn.deterministic pins the convolution algorithms; without it cuDNN is free to
# pick a different (non-deterministic) kernel per run and the seed alone will not
# reproduce the weights. This costs some throughput, which is why it is only set
# on the seeded path. Reproducibility is per hardware and library version, not
# bitwise across different GPUs.
seed = cfg["train"].get("seed")
if seed is not None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Seeded run: train.seed={seed} (weight init, dropout, shuffle order)")
else:
    print("Unseeded run (no train.seed) — weight init and shuffle order vary per run")

transform = cfg["data"].get("transform", "none")
target_hw = tuple(cfg["model"]["input_hw"])

def _load_and_prep(tensor):
    return prepare_images(tensor, transform, target_hw)

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

device = pick_device()
print("Training device:", device)

EPOCHS = cfg["train"]["epochs"]
BATCH_SIZE = cfg["train"]["batch_size"]
BETA = cfg["train"]["beta"]

splits_dir = Path(cfg["output"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)
split_path = splits_dir / split_filename(cfg)

if split_path.exists():
    split = np.load(split_path)
    train_idx, val_idx = split["train_idx"], split["val_idx"]
elif cfg["data"].get("tag"):
    # A tagged variant defines its own split; regenerating a random one here
    # would silently train on the wrong sample.
    raise FileNotFoundError(
        f"data.tag={cfg['data']['tag']!r} requires a prepared split at {split_path}"
    )
else:
    all_indices = np.arange(len(p))
    train_idx, val_idx = train_test_split(
        all_indices, test_size=cfg["data"]["val_split"], random_state=cfg["data"]["random_seed"]
    )
    np.savez(split_path, train_idx=train_idx, val_idx=val_idx)

print(f"Split: {split_path.name} | train={len(train_idx)} val={len(val_idx)}")

train_subset = Subset(p, train_idx)
val_subset = Subset(p, val_idx)

# The shuffle order is part of the run: without its own seeded generator the
# loader draws from global RNG state, which torch.manual_seed above does fix, but
# only until anything else consumes from it. An explicit generator keeps the epoch
# order reproducible regardless of what else touches the global stream.
loader_gen = torch.Generator().manual_seed(seed) if seed is not None else None
train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True,
                          generator=loader_gen)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)

model = build_vae(cfg, device)

optim = torch.optim.Adam(model.parameters(), lr=cfg["optimizer"]["lr"], weight_decay=cfg["optimizer"]["weight_decay"])

model, train_losses, train_recon, train_kl, val_losses, val_recon, val_kl = train(
    device, train_loader, val_loader, model, optim, vae_loss,
    epochs=EPOCHS, beta=BETA,
    patience=cfg["train"].get("patience", 20),
    min_delta=cfg["train"].get("min_delta", 1e-4),
)

save_dir = Path(cfg["output"]["dir"])
save_dir.mkdir(parents=True, exist_ok=True)
name = model_filename(cfg)
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