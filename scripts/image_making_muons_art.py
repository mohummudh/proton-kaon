"""
Create 48x48 images from muon ROOT files in art framework format.

Input:  /Volumes/easystore/proton-kaon/raw/Muons_50_300/muon_p1.root
        /Volumes/easystore/proton-kaon/raw/Muons_50_300/muon_p2.root
Output: /Volumes/easystore/proton-kaon/images/muon_48x48_raw.pt
        /Volumes/easystore/proton-kaon/images/muon_48x48_log1p.pt

Usage:
    uv run python scripts/image_making_muons_art.py
    uv run python scripts/image_making_muons_art.py --max-events 5000
"""

import argparse
import logging
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import uproot
from skimage.measure import label, regionprops
from tqdm import tqdm

from src.images import pad_image_batch_gpu

ROOT_FILES = [
    "/Volumes/easystore/proton-kaon/raw/Muons_50_300/muon_p1.root",
    "/Volumes/easystore/proton-kaon/raw/Muons_50_300/muon_p2.root",
]
OUTPUT_RAW = "/Volumes/easystore/proton-kaon/images/muon_48x48_raw.pt"
OUTPUT_LOG = "/Volumes/easystore/proton-kaon/images/muon_48x48_log1p.pt"
OUTPUT_PKL = "/Volumes/easystore/proton-kaon/clusters/muon_art_col.pkl"

WIRES_PER_PLANE = 240
TICKS_PER_EVENT = 3072
COL_THRESHOLD   = 15
IND_THRESHOLD   = 7
MIN_HEIGHT       = 20
MAX_WIDTH        = 1500
IO_BATCH_SIZE   = 500

ART_CH  = ("raw::RawDigits_daq__EventBuilderNoMerge."
            "/raw::RawDigits_daq__EventBuilderNoMerge.obj"
            "/raw::RawDigits_daq__EventBuilderNoMerge.obj.fChannel")
ART_ADC = ("raw::RawDigits_daq__EventBuilderNoMerge."
            "/raw::RawDigits_daq__EventBuilderNoMerge.obj"
            "/raw::RawDigits_daq__EventBuilderNoMerge.obj.fADC")

LOG_PATH = Path("logs/image_making_muons_art.log")
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


# Bounding box cuts — same values as cluster_cuts() in src/cuts.py
BBOX = {
    "collection": dict(min_row_lo=12, min_row_hi=37,  max_col_lo=789, max_col_hi=1927),
    "induction":  dict(min_row_lo=11, min_row_hi=35,  max_col_lo=786, max_col_hi=1794),
}


def largest_cluster(matrix: np.ndarray, threshold: float, plane: str) -> np.ndarray | None:
    labeled, n = label(matrix > threshold, return_num=True)
    if n == 0:
        return None
    regions = regionprops(labeled, intensity_image=matrix)
    bb = BBOX[plane]

    candidates = []
    for r in regions:
        min_row, min_col, max_row, max_col = r.bbox
        h = max_row - min_row
        w = max_col - min_col
        if (h >= MIN_HEIGHT and w < MAX_WIDTH
                and bb["min_row_lo"] < min_row < bb["min_row_hi"]
                and bb["max_col_lo"] < max_col < bb["max_col_hi"]):
            candidates.append(r)

    if not candidates:
        return None
    return max(candidates, key=lambda r: r.area).image_intensity


def process_event(ch_ak, adc_ak) -> tuple[np.ndarray | None, np.ndarray | None]:
    channels = np.asarray(ak.to_list(ch_ak), dtype=np.int32)
    adc_2d   = np.array([ak.to_list(row) for row in adc_ak], dtype=np.float32)

    sort_idx = np.argsort(channels)
    channels = channels[sort_idx]
    adc_2d   = adc_2d[sort_idx]

    col_matrix = np.zeros((WIRES_PER_PLANE, TICKS_PER_EVENT), dtype=np.float32)
    ind_matrix = np.zeros((WIRES_PER_PLANE, TICKS_PER_EVENT), dtype=np.float32)

    col_sel = channels >= WIRES_PER_PLANE
    ind_sel = ~col_sel

    col_matrix[channels[col_sel] - WIRES_PER_PLANE] = np.clip(adc_2d[col_sel], 0, None)
    ind_matrix[channels[ind_sel]]                    = np.clip(adc_2d[ind_sel], 0, None)

    return (
        largest_cluster(col_matrix, COL_THRESHOLD, "collection"),
        largest_cluster(ind_matrix, IND_THRESHOLD, "induction"),
    )


def process_file(path: str, max_events: int | None) -> tuple[list, list]:
    col_images, ind_images = [], []

    tree    = uproot.open(path)["Events"]
    n_total = min(tree.num_entries, max_events) if max_events else tree.num_entries
    logger.info("Processing %s  (%d events)", path, n_total)

    for start in tqdm(range(0, n_total, IO_BATCH_SIZE), desc=Path(path).name):
        stop      = min(start + IO_BATCH_SIZE, n_total)
        ch_batch  = tree[ART_CH ].array(library="ak", entry_start=start, entry_stop=stop)
        adc_batch = tree[ART_ADC].array(library="ak", entry_start=start, entry_stop=stop)

        for i in range(len(ch_batch)):
            col_img, ind_img = process_event(ch_batch[i], adc_batch[i])
            if col_img is not None and ind_img is not None:
                col_images.append(col_img)
                ind_images.append(ind_img)

    logger.info("  kept %d paired clusters", len(col_images))
    return col_images, ind_images


def build_tensor(col_images: list, ind_images: list, device: torch.device) -> torch.Tensor:
    logger.info("Padding %d image pairs...", len(col_images))
    col_padded = np.array(pad_image_batch_gpu(col_images, device=device, batch_size=64, cut_rows=50))
    ind_padded = np.array(pad_image_batch_gpu(ind_images, device=device, batch_size=64, cut_rows=50))
    logger.info("Padded shapes: col=%s  ind=%s", col_padded.shape, ind_padded.shape)

    col_t = torch.from_numpy(col_padded).float().to(device)
    ind_t = torch.from_numpy(ind_padded).float().to(device)

    col_d = F.interpolate(col_t.unsqueeze(1), size=(48, 48), mode="bilinear", align_corners=False).squeeze(1)
    ind_d = F.interpolate(ind_t.unsqueeze(1), size=(48, 48), mode="bilinear", align_corners=False).squeeze(1)

    return torch.stack([col_d, ind_d], dim=1).cpu()  # (N, 2, 48, 48)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events per file (default: all)")
    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else
        "mps"  if torch.backends.mps.is_available() else
        "cpu"
    )
    logger.info("Device: %s", device)

    all_col, all_ind = [], []
    for path in ROOT_FILES:
        col, ind = process_file(path, args.max_events)
        all_col.extend(col)
        all_ind.extend(ind)

    logger.info("Total paired clusters: %d", len(all_col))

    muon = build_tensor(all_col, all_ind, device)
    logger.info("Final tensor shape: %s", muon.shape)

    torch.save({"m": muon}, OUTPUT_RAW)
    logger.info("Saved raw    → %s", OUTPUT_RAW)

    torch.save({"m": torch.log1p(muon)}, OUTPUT_LOG)
    logger.info("Saved log1p  → %s", OUTPUT_LOG)

    # Save cluster pkl so compute_features.py can compute physics features.
    # Rows are index-aligned with the image tensor: pkl row i ↔ muon[i].
    pkl_df = pd.DataFrame([
        {"image_intensity": img, "column_maxes": img.max(axis=1), "particle_type": "muon"}
        for img in all_col
    ])
    pkl_df.to_pickle(OUTPUT_PKL)
    logger.info("Saved cluster pkl → %s  (%d rows)", OUTPUT_PKL, len(pkl_df))


if __name__ == "__main__":
    main()
