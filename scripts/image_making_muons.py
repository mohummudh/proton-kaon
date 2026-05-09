"""
Extract muon (>179 wire) clusters directly from ROOT files and generate 48x48 images.

Muons are identified as high-energy clusters not captured in col.pkl/ind.pkl
(which filter for 10 < height < 179).

Usage:
    python scripts/image_making_muons.py
"""

import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd

from pathlib import Path

from src.images import cut_start, pad_image_batch_gpu
from src.open_root import open_root
from src.clustering import extract_clusters

LOG_PATH = Path("logs/image_making_muons.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)

# ── Extract clusters from ROOT files (no height filtering) ──
PROTONS_ROOT = "/Volumes/easystore/p_1track_protons_600_1600.root"
KAONS_ROOT = "/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root"

logger.info("Opening ROOT files to extract all clusters...")
p_df = open_root(PROTONS_ROOT, tree_name="ana/raw")
k_df = open_root(KAONS_ROOT, tree_name="ana/raw;352")

logger.info("Found %d proton events, %d kaon events", len(p_df), len(k_df))

# Extract ALL clusters (no height filtering applied here)
logger.info("Extracting clusters from proton ROOT...")
p_clusters = extract_clusters(p_df, particle_type="proton", threshold=15, tree_name="ana/raw")

logger.info("Extracting clusters from kaon ROOT...")
k_clusters = extract_clusters(k_df, particle_type="kaon", threshold=15, tree_name="ana/raw;352")

logger.info("Extracted %d proton clusters, %d kaon clusters", len(p_clusters), len(k_clusters))

# ── Filter for >179 wires only ──
muon_clusters = pd.concat([p_clusters, k_clusters], ignore_index=True)
muon_clusters = muon_clusters[
    (muon_clusters['height'] >= 180) &
    (muon_clusters['width'] < 1500)
].reset_index(drop=True)

logger.info("After height>=180 + width<1500: %d clusters", len(muon_clusters))

# ── Apply quality cuts (bbox + column_maxes) ──
from src.cuts import cluster_cuts

muon_clusters = cluster_cuts(muon_clusters, lower=179, upper=10000)
logger.info("After quality cuts: %d clusters", len(muon_clusters))

# ── Match collection/induction planes ──
from src.matching import matching

muon_col, muon_ind = matching(muon_clusters)
logger.info(
    "After matching collection/induction: %d collection, %d induction",
    len(muon_col), len(muon_ind)
)

# ── Save cluster pickles (like col.pkl/ind.pkl) ──
muon_col = muon_col.copy()
muon_ind = muon_ind.copy()
muon_col['particle_type'] = 'muon'
muon_ind['particle_type'] = 'muon'
muon_col.to_pickle('/Volumes/easystore/proton-kaon/clusters/muon_col.pkl')
muon_ind.to_pickle('/Volumes/easystore/proton-kaon/clusters/muon_ind.pkl')
logger.info("Saved muon cluster pickles with particle_type='muon'")

# ── Setup GPU ──
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
logger.info("Using device: %s", device)

# ── Extract image intensities ──
muon_c_list = muon_col['image_intensity'].tolist()
muon_i_list = muon_ind['image_intensity'].tolist()

logger.info("Processing %d muon collection and %d muon induction images", len(muon_c_list), len(muon_i_list))

# ── GPU batch padding (same parameters as proton/kaon) ──
muon_c = np.array(pad_image_batch_gpu(muon_c_list, device=device, batch_size=64, cut_rows=50))
muon_i = np.array(pad_image_batch_gpu(muon_i_list, device=device, batch_size=64, cut_rows=50))

logger.info("Padded shapes: muon_c=%s, muon_i=%s", muon_c.shape, muon_i.shape)

# ── Downsample to 48x48 on GPU ──
muon_c_tensor = torch.from_numpy(muon_c).float().to(device)
muon_i_tensor = torch.from_numpy(muon_i).float().to(device)

muon_c_d = F.interpolate(muon_c_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
muon_i_d = F.interpolate(muon_i_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()

logger.info("Downsampled shapes: muon_c_d=%s, muon_i_d=%s", muon_c_d.shape, muon_i_d.shape)

# ── Stack collection and induction as 2-channel images ──
muon = np.stack([muon_c_d, muon_i_d], axis=1)  # shape: (N, 2, 48, 48)
muon = torch.from_numpy(muon).float()

logger.info("Final stacked shape: %s", muon.shape)

# ── Save raw ADC values (no transform) ──
out_raw = f"/Volumes/easystore/proton-kaon/images/muon_48x48_raw_180+wires.pt"
torch.save({"m": muon.cpu()}, out_raw)
logger.info("Saved raw muon images to %s", out_raw)

# ── Save log1p version ──
muon_log = torch.log1p(muon)
out_log = f"/Volumes/easystore/proton-kaon/images/muon_48x48_log1p_180+wires.pt"
torch.save({"m": muon_log.cpu()}, out_log)
logger.info("Saved log1p muon images to %s", out_log)

# ── To load: ──
# data = torch.load("/Volumes/easystore/proton-kaon/images/muon_48x48_raw_180+wires.pt", map_location="cpu")
# m = data["m"]  # shape: (N, 2, 48, 48)
