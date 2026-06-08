"""
Extract muon clusters from art framework ROOT files.

Reads raw::RawDigits from LArIAT art ROOT files, finds all connected regions
in both planes, then applies cluster_cuts() and matching() identically to the
proton/kaon pipeline. Saves matched cluster DataFrames for use by image_making.py.

Output:
    /Volumes/easystore/proton-kaon/clusters/muon_col.pkl
    /Volumes/easystore/proton-kaon/clusters/muon_ind.pkl

Usage:
    uv run python scripts/image_making_muons_art.py
    uv run python scripts/image_making_muons_art.py --max-events 2000
"""

import argparse
import logging
import sys
from pathlib import Path

import awkward as ak
import numpy as np
import pandas as pd
import uproot
from skimage.measure import label, regionprops
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cuts import cluster_cuts
from src.matching import matching

ROOT_FILES = [
    "/Volumes/easystore/proton-kaon/raw/Muons_50_300/muon_p1.root",
    "/Volumes/easystore/proton-kaon/raw/Muons_50_300/muon_p2.root",
]
OUTPUT_COL_PKL = "/Volumes/easystore/proton-kaon/clusters/muon_col.pkl"
OUTPUT_IND_PKL = "/Volumes/easystore/proton-kaon/clusters/muon_ind.pkl"

WIRES_PER_PLANE = 240
TICKS_PER_EVENT = 3072
COL_THRESHOLD   = 15
IND_THRESHOLD   = 7
IO_BATCH_SIZE   = 500

# Height cut: only near-through-going muons (same as before, expressed as lower bound for cluster_cuts)
MIN_HEIGHT_LOWER = 175  # cluster_cuts uses height > lower, so height >= 176

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


def build_plane_matrices(ch_ak, adc_ak):
    channels = np.asarray(ak.to_list(ch_ak), dtype=np.int32)
    adc_2d   = np.array([ak.to_list(row) for row in adc_ak], dtype=np.float32)

    sort_idx = np.argsort(channels)
    channels = channels[sort_idx]
    adc_2d   = adc_2d[sort_idx]

    col_matrix = np.zeros((WIRES_PER_PLANE, TICKS_PER_EVENT), dtype=np.float32)
    ind_matrix = np.zeros((WIRES_PER_PLANE, TICKS_PER_EVENT), dtype=np.float32)

    col_sel = channels >= WIRES_PER_PLANE
    col_matrix[channels[col_sel] - WIRES_PER_PLANE] = np.clip(adc_2d[col_sel], 0, None)
    ind_matrix[channels[~col_sel]]                   = np.clip(adc_2d[~col_sel], 0, None)

    return col_matrix, ind_matrix


def regions_to_rows(plane_matrix, threshold, plane_name, event_id):
    labeled, n = label(plane_matrix > threshold, return_num=True)
    if n == 0:
        return []
    rows = []
    for j, r in enumerate(regionprops(labeled, intensity_image=plane_matrix)):
        min_row, min_col, max_row, max_col = r.bbox
        rows.append({
            'run':           0,
            'subrun':        0,
            'event':         event_id,
            'cluster_idx':   j,
            'plane':         plane_name,
            'bbox_min_row':  min_row,
            'bbox_min_col':  min_col,
            'bbox_max_row':  max_row,
            'bbox_max_col':  max_col,
            'height':        max_row - min_row,
            'width':         max_col - min_col,
            'image_intensity': r.image_intensity,
            'column_maxes':    r.image_intensity.max(axis=1),
            'particle_type':   'muon',
        })
    return rows


def extract_clusters(path: str, event_offset: int, max_events: int | None) -> tuple[pd.DataFrame, int]:
    tree    = uproot.open(path)["Events"]
    n_total = min(tree.num_entries, max_events) if max_events else tree.num_entries
    logger.info("Processing %s  (%d events)", path, n_total)

    rows = []
    event_id = event_offset

    for start in tqdm(range(0, n_total, IO_BATCH_SIZE), desc=Path(path).name):
        stop      = min(start + IO_BATCH_SIZE, n_total)
        ch_batch  = tree[ART_CH ].array(library="ak", entry_start=start, entry_stop=stop)
        adc_batch = tree[ART_ADC].array(library="ak", entry_start=start, entry_stop=stop)

        for i in range(len(ch_batch)):
            col_matrix, ind_matrix = build_plane_matrices(ch_batch[i], adc_batch[i])
            rows.extend(regions_to_rows(col_matrix, COL_THRESHOLD, 'collection', event_id))
            rows.extend(regions_to_rows(ind_matrix, IND_THRESHOLD, 'induction',  event_id))
            event_id += 1

    logger.info("  %d cluster rows from %s", len(rows), path)
    return pd.DataFrame(rows), event_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events per file (default: all)")
    args = parser.parse_args()

    # Collect all cluster rows from both files
    all_dfs = []
    offset  = 0
    for path in ROOT_FILES:
        df, offset = extract_clusters(path, offset, args.max_events)
        all_dfs.append(df)

    clusters_df = pd.concat(all_dfs, ignore_index=True)
    logger.info("Total raw clusters: %d", len(clusters_df))

    # Apply same cuts as proton/kaon pipeline (bbox + height + column_maxes uniqueness)
    # lower=175 selects height >= 176 (near-through-going muons only)
    clusters_df = cluster_cuts(clusters_df, lower=MIN_HEIGHT_LOWER, upper=10_000_000)
    logger.info("After cluster_cuts: %d", len(clusters_df))

    # Greedy 1-to-1 spatial matching of collection/induction clusters per event
    muon_col, muon_ind = matching(clusters_df)
    logger.info("Matched pairs: %d", len(muon_col))

    muon_col.to_pickle(OUTPUT_COL_PKL)
    muon_ind.to_pickle(OUTPUT_IND_PKL)
    logger.info("Saved muon_col.pkl → %s  (%d rows)", OUTPUT_COL_PKL, len(muon_col))
    logger.info("Saved muon_ind.pkl → %s  (%d rows)", OUTPUT_IND_PKL, len(muon_ind))


if __name__ == "__main__":
    main()
