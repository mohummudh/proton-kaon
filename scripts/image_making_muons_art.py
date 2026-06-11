"""
Extract muon clusters from RAW_muons.root and save matched cluster DataFrames.

Uses the same open_root() + extract_clusters() + cluster_cuts() + matching()
pipeline as proton/kaon. RAW_muons.root uses the rawadc1 flat-array format
handled by Event._load_flat_event(), so no custom parsing is needed.

ADC values are pedestal-subtracted (mean ≈ 0), so image_intensity is clipped
to 0 before saving to avoid NaN from log1p during image making.

Output:
    /Volumes/easystore/proton-kaon/clusters/muon_col.pkl
    /Volumes/easystore/proton-kaon/clusters/muon_ind.pkl

Usage:
    uv run python scripts/image_making_muons_art.py
    uv run python scripts/image_making_muons_art.py --max-events 5000
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.open_root import open_root
from src.clustering import extract_clusters
from src.cuts import cluster_cuts
from src.matching import matching

ROOT_FILE      = "/Volumes/easystore/proton-kaon/raw/Muons_50_300/RAW_muons.root"
TREE_NAME      = "anatree/raw"
OUTPUT_COL_PKL = "/Volumes/easystore/proton-kaon/clusters/muon_col.pkl"
OUTPUT_IND_PKL = "/Volumes/easystore/proton-kaon/clusters/muon_ind.pkl"

MIN_HEIGHT_LOWER = 175  # cluster_cuts uses height > lower, so height >= 176

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-events", type=int, default=None,
                        help="Max events to process (default: all)")
    args = parser.parse_args()

    logger.info("Opening %s", ROOT_FILE)
    muon_df = open_root(ROOT_FILE, tree_name=TREE_NAME)
    logger.info("Found %d muon events (run %d–%d)",
                len(muon_df), muon_df["run"].min(), muon_df["run"].max())

    logger.info("Extracting clusters...")
    clusters_df = extract_clusters(
        muon_df,
        particle_type="muon",
        threshold=15,
        max_events=args.max_events,
        tree_name=TREE_NAME,
    )
    logger.info("Raw clusters: %d", len(clusters_df))

    # Clip negative ADC — pedestal-subtracted data has noise below 0;
    # log1p in image_making.py would produce NaN without this.
    clusters_df["image_intensity"] = clusters_df["image_intensity"].map(
        lambda x: np.clip(x, 0, None)
    )
    clusters_df["column_maxes"] = clusters_df["image_intensity"].map(
        lambda x: x.max(axis=1)
    )

    # Same cuts as proton/kaon pipeline; lower=175 selects height >= 176
    clusters_df = cluster_cuts(clusters_df, lower=MIN_HEIGHT_LOWER, upper=10_000_000)
    logger.info("After cluster_cuts: %d", len(clusters_df))

    muon_col, muon_ind = matching(clusters_df)
    logger.info("Matched pairs: %d", len(muon_col))

    muon_col.to_pickle(OUTPUT_COL_PKL)
    muon_ind.to_pickle(OUTPUT_IND_PKL)
    logger.info("Saved → %s  (%d rows)", OUTPUT_COL_PKL, len(muon_col))
    logger.info("Saved → %s  (%d rows)", OUTPUT_IND_PKL, len(muon_ind))


if __name__ == "__main__":
    main()
