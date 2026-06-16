#!/usr/bin/env python3
"""
scripts/save_kaon_displays.py

Extract and save event displays for kaon events. Supports two modes:
1. Scan (Default): Find events excluded from col.pkl that have large clusters.
2. CSV: Plot specific events listed in a CSV file (old behaviour).

Usage:
    # New behaviour (muon-like scan)
    python scripts/save_kaon_displays.py --mode scan --height 180
    
    # Old behaviour (CSV-based)
    python scripts/save_kaon_displays.py --mode csv --csv /path/to/kaon_df.csv
"""

import argparse
import logging
import pandas as pd
import uproot
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.open_root import open_root
from src.event import Event
from src.event_display import plot_event_displays

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="Save displays for kaon events.")
    parser.add_argument("--mode", choices=["scan", "csv"], default="scan", 
                        help="Mode: 'scan' for muon-like kaons, 'csv' for specific events (old behaviour)")
    
    # Common arguments
    parser.add_argument("--kaons_root", default="/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root", help="Path to kaon ROOT file")
    parser.add_argument("--out_dir", default="/Volumes/easystore/kaondisplays", help="Directory to save images")
    parser.add_argument("--tree", default="ana/raw;352", help="Tree name in ROOT file")
    
    # Scan mode arguments
    parser.add_argument("--kaons_pkl", default="/Volumes/easystore/proton-kaon/clusters/col.pkl", help="Path to existing kaon cluster pickle")
    parser.add_argument("--height", type=int, default=180, help="Height threshold for saving displays (scan mode only)")
    
    # CSV mode arguments
    parser.add_argument("--csv", default="/Volumes/easystore/proton-kaon/docs/kaon_df_plane_1_thr_DAQ.csv", help="Path to kaon CSV (csv mode only)")
    
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    if args.mode == "scan" and args.out_dir == "/Volumes/easystore/kaondisplays":
         # Use a subfolder for scan mode to prevent cluttering the main displays folder
         out_dir = out_dir / f"180plus"
    
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Open full ROOT index to see ALL available events
    logger.info("Opening ROOT file index %s...", args.kaons_root)
    k_df = open_root(args.kaons_root, tree_name=args.tree)
    logger.info("Found %d total events in ROOT file.", len(k_df))

    # 2. Filter events based on mode
    if args.mode == "csv":
        logger.info("Mode: CSV. Loading target events from %s...", args.csv)
        csv_df = pd.read_csv(args.csv)
        target_keys = set(zip(csv_df['run'], csv_df['subrun'], csv_df['event']))
        
        mask = k_df.apply(lambda row: (row['run'], row['subrun'], row['event']) in target_keys, axis=1)
        k_df_filtered = k_df[mask].reset_index(drop=True)
        logger.info("Found %d matching events to plot from CSV.", len(k_df_filtered))
        
    else: # Scan mode
        logger.info("Mode: Scan. Identifying excluded events via %s...", args.kaons_pkl)
        existing_kaons = pd.read_pickle(args.kaons_pkl)
        existing_kaons = existing_kaons[existing_kaons['particle_type'] == 'kaon']
        included_keys = set(zip(existing_kaons['run'], existing_kaons['subrun'], existing_kaons['event']))
        
        def is_excluded(row):
            return (row['run'], row['subrun'], row['event']) not in included_keys

        excluded_mask = k_df.apply(is_excluded, axis=1)
        k_df_filtered = k_df[excluded_mask].reset_index(drop=True)
        logger.info("Identified %d excluded events to check for >%d wire clusters.", len(k_df_filtered), args.height)

    # 3. Iterate and plot
    root_file = uproot.open(args.kaons_root)
    tree = root_file[args.tree]

    saved_count = 0
    for _, row in tqdm(k_df_filtered.iterrows(), total=len(k_df_filtered)):
        try:
            event = Event(tree=tree, filepath=args.kaons_root, index=row.event_index, plot=False)
            
            should_save = True
            max_h = 0
            
            # In scan mode, only save if there's a large cluster
            if args.mode == "scan":
                clabeled, cregions = event.connectedregions(event.collection, threshold=15)
                ilabeled, iregions = event.connectedregions(event.induction, threshold=7)
                all_regions = (cregions or []) + (iregions or [])
                
                should_save = False
                for region in all_regions:
                    h = region.bbox[2] - region.bbox[0]
                    if h > max_h:
                        max_h = h
                    if h >= args.height:
                        should_save = True
                        break
            
            if should_save:
                suffix = f"_h{max_h}" if args.mode == "scan" else ""
                out_path = out_dir / f"kaon_{row.run}_{row.subrun}_{row.event}{suffix}.png"
                plot_event_displays(event, row.run, row.subrun, row.event, save_path=out_path)
                saved_count += 1

        except Exception as e:
            import matplotlib.pyplot as plt
            plt.close('all')

    logger.info("Done. Saved %d displays to %s", saved_count, out_dir)

if __name__ == "__main__":
    main()
