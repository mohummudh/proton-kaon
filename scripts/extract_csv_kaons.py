"""
Extract specific kaon events directly from the raw ROOT file based on a CSV list.
No length cuts and no bounding box cuts are applied to maximize recovery.
"""
import logging
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path

from src.images import pad_image_batch_gpu
from src.open_root import open_root
from src.clustering import extract_clusters
from src.matching import matching

LOG_PATH = Path("logs/extract_csv_kaons.log")
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

KAONS_ROOT = "/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root"
CSV_PATH = "/Volumes/easystore/proton-kaon/docs/kaon_df_plane_1_thr_DAQ.csv"

def main():
    logger.info("Loading target events from CSV...")
    target_df = pd.read_csv(CSV_PATH)
    target_keys = set(zip(target_df['run'], target_df['subrun'], target_df['event']))
    logger.info("Found %d target events in CSV.", len(target_keys))

    logger.info("Opening raw KAON ROOT file...")
    # Open root without any predefined filtering
    k_df = open_root(KAONS_ROOT, tree_name="ana/raw;352")
    
    logger.info("Filtering ROOT DataFrame to only include target events...")
    # Filter the giant ROOT dataframe to only the events we care about to speed up clustering
    mask = k_df.apply(lambda row: (row['run'], row['subrun'], row['event']) in target_keys, axis=1)
    k_df_filtered = k_df[mask].reset_index(drop=True)
    logger.info("ROOT events matching CSV: %d", len(k_df_filtered))

    logger.info("Clustering target events (NO CUTS)...")
    # Using threshold=15 to match standard processing, but no width/height limits
    raw_clusters = extract_clusters(k_df_filtered, particle_type="kaon", threshold=15, tree_name="ana/raw;352")
    logger.info("Generated %d raw clusters from the target events.", len(raw_clusters))

    if len(raw_clusters) == 0:
        logger.error("No clusters generated! Exiting.")
        return

    # We do NOT apply cluster_cuts() or image_cuts() here.
    # We want everything.

    # Save per-plane clusters BEFORE matching so the labeler can show all of them
    from src.matching import _plane_masks
    _ind_mask, _col_mask = _plane_masks(raw_clusters)
    raw_col = raw_clusters[_col_mask].copy().reset_index(drop=True)
    raw_ind = raw_clusters[_ind_mask].copy().reset_index(drop=True)
    raw_col.to_pickle('/Volumes/easystore/proton-kaon/clusters/csv_kaon_col_unmatched.pkl')
    raw_ind.to_pickle('/Volumes/easystore/proton-kaon/clusters/csv_kaon_ind_unmatched.pkl')
    logger.info("Saved unmatched per-plane clusters: %d col, %d ind", len(raw_col), len(raw_ind))

    logger.info("Matching Collection and Induction planes...")
    col, ind = matching(raw_clusters)
    logger.info("Successfully matched %d tracks across both planes.", len(col))

    col = col.copy()
    ind = ind.copy()
    col['particle_type'] = 'kaon'
    ind['particle_type'] = 'kaon'

    # Filter out extreme widths that break the image padding, matching pipeline limits
    # Must drop synchronously so col and ind remain identical length
    valid_mask = (col['width'] < 1500) & (ind['width'] < 1500)
    col = col[valid_mask].reset_index(drop=True)
    ind = ind[valid_mask].reset_index(drop=True)
    logger.info("After width filtering (<1500), retained %d tracks (all clusters, multiple per event possible).", len(col))

    out_col = '/Volumes/easystore/proton-kaon/clusters/csv_kaon_col.pkl'
    out_ind = '/Volumes/easystore/proton-kaon/clusters/csv_kaon_ind.pkl'
    col.to_pickle(out_col)
    ind.to_pickle(out_ind)
    logger.info("Saved cluster DataFrames to %s", out_col)

    # ── Setup GPU ──
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    logger.info("Using device: %s for image padding and downsampling.", device)

    # ── Extract image intensities ──
    c_list = col['image_intensity'].tolist()
    i_list = ind['image_intensity'].tolist()

    # ── GPU batch padding ──
    c_padded = np.array(pad_image_batch_gpu(c_list, device=device, batch_size=64, cut_rows=50))
    i_padded = np.array(pad_image_batch_gpu(i_list, device=device, batch_size=64, cut_rows=50))

    # ── Downsample to 48x48 on GPU ──
    c_tensor = torch.from_numpy(c_padded).float().to(device)
    i_tensor = torch.from_numpy(i_padded).float().to(device)

    c_down = F.interpolate(c_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
    i_down = F.interpolate(i_tensor.unsqueeze(1), size=(48, 48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()

    # ── Stack collection and induction as 2-channel images ──
    images = np.stack([c_down, i_down], axis=1)  # shape: (N, 2, 48, 48)
    images_tensor = torch.from_numpy(images).float()

    logger.info("Final image tensor shape: %s", images_tensor.shape)

    # ── Save raw ADC values (no transform) ──
    out_img = "/Volumes/easystore/proton-kaon/images/csv_kaon_48x48_raw.pt"
    torch.save({"k": images_tensor.cpu()}, out_img)
    logger.info("Saved raw 48x48 images to %s", out_img)

if __name__ == "__main__":
    main()
