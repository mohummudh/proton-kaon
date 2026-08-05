import yaml
import argparse
import torch
import numpy as np

from pathlib import Path
from torch.utils.data import Subset

from src.device import pick_device
from src.models.build import build_vae
from src.inference.inference import inference
from src.train.naming import model_filename, model_name, split_filename
from src.transforms import prepare_images

parser = argparse.ArgumentParser()
parser.add_argument("--config", default="configs/default.yaml")
parser.add_argument("--include-muons", action="store_true", help="Also run inference on muon images (>=180 wires)")
parser.add_argument("--muon-image-path", default="/Volumes/easystore/proton-kaon/images/muon_48x48_raw.pt", help="Path to muon image file (must align with muon_col.pkl rows in features.pkl)")
parser.add_argument("--csda-kaon-path", default=None, help="Path to csda-kaon image file (csv_kaon_48x48_raw.pt)")
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

device = pick_device()

all_species = cfg["data"].get("proton") == "all"

save_dir = Path(cfg["output"]["dir"])
save_dir.mkdir(parents=True, exist_ok=True)
name = model_filename(cfg)
save_path = save_dir / name

# LOADING TRAINING DATA + VALIDATION DATA
splits_dir = Path(cfg["output"]["splits_dir"])
splits_dir.mkdir(parents=True, exist_ok=True)
proton_key = "p" if all_species else cfg["data"]["proton"]
kaon_key = "k" if all_species else cfg["data"]["kaon"]

transform = cfg["data"].get("transform", "none")
target_hw = tuple(cfg["model"]["input_hw"])

data = torch.load(cfg["data"]["path"], map_location="cpu")
p = prepare_images(data[proton_key], transform, target_hw)
kaons = prepare_images(data[kaon_key], transform, target_hw)

if all_species:
    # Train/val membership follows the all-species split over the concatenated
    # [p, k, m] tensor used in training. Derive per-species within-tensor
    # indices so every species has a train/val separation.
    muons_all = prepare_images(data["m"], transform, target_hw)
    n_p, n_k, n_m = len(p), len(kaons), len(muons_all)
    split_path = splits_dir / split_filename(cfg)
    if split_path.exists():
        split = np.load(split_path)
        all_train_idx, all_val_idx = split["train_idx"], split["val_idx"]
    elif cfg["data"].get("tag"):
        raise FileNotFoundError(
            f"data.tag={cfg['data']['tag']!r} requires the split used in training at {split_path}"
        )
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
    split = np.load(splits_dir / split_filename(cfg))
    train_idx = split["train_idx"]
    val_idx = split["val_idx"]

train_subset = Subset(p, train_idx)
val_subset = Subset(p, val_idx)

# LOAD MODEL
model = build_vae(cfg, device)
model.load_state_dict(torch.load(save_path, map_location=device))

# RUN INFERENCE
train_latents, train_recon, train_re = inference(model, train_subset, device)
val_latents, val_recon, val_re = inference(model, val_subset, device)
kaon_latents, kaon_recon, kaon_re = inference(model, kaons, device)

inference_dir = Path(cfg["output"]["inference_dir"]) / model_name(cfg)
inference_dir.mkdir(parents=True, exist_ok=True)

# Latents are being rewritten — derived caches (UMAP reducer/embeddings and
# analysis caches) computed from the previous latents are now stale.
stale_caches = [inference_dir / "reducer.pkl"]
_figs_lf = Path("figs") / model_name(cfg) / "latents-features"
if _figs_lf.exists():
    stale_caches += sorted(_figs_lf.glob("cache_*"))
for _cache in stale_caches:
    if _cache.exists():
        _cache.unlink()
        print(f"Removed stale cache: {_cache}")

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
    muon_latents, muon_recon, muon_re = inference(model, muons_all, device)
    np.savez(inference_dir / "muon.npz",
        latents=muon_latents, recon=muon_recon, re=muon_re)
    print(f"Saved muon inference from training data file: {len(muon_latents)} images")

# RUN INFERENCE ON MUONS (if requested; all-species mode already saved them above)
if args.include_muons and not all_species:
    muon_data = torch.load(args.muon_image_path, map_location="cpu")
    muons = prepare_images(muon_data["m"], transform, target_hw)
    muon_latents, muon_recon, muon_re = inference(model, muons, device)

    np.savez(inference_dir / "muon.npz",
        latents=muon_latents, recon=muon_recon, re=muon_re)
    print(f"Saved muon inference: {len(muon_latents)} images")

if args.csda_kaon_path:
    csda_data = torch.load(args.csda_kaon_path, map_location="cpu")
    csda_kaons = prepare_images(csda_data["k"], transform, target_hw)
    csda_latents, csda_recon, csda_re = inference(model, csda_kaons, device)
    np.savez(inference_dir / "csda_kaon.npz",
        latents=csda_latents, recon=csda_recon, re=csda_re)
    print(f"Saved csda-kaon inference: {len(csda_latents)} images")

