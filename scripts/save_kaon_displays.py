import pandas as pd
import uproot
from pathlib import Path
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.open_root import open_root
from src.event import Event
from src.event_display import plot_event_displays

KAONS_ROOT = "/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root"
CSV_PATH = "/Volumes/easystore/proton-kaon/docs/kaon_df_plane_1_thr_DAQ.csv"
OUT_DIR = Path("/Volumes/easystore/kaondisplays")

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Loading CSV...")
    csv_df = pd.read_csv(CSV_PATH)
    target_keys = set(zip(csv_df['run'], csv_df['subrun'], csv_df['event']))
    
    print("Opening ROOT file...")
    k_df = open_root(KAONS_ROOT, tree_name="ana/raw;352")
    mask = k_df.apply(lambda row: (row['run'], row['subrun'], row['event']) in target_keys, axis=1)
    k_df_filtered = k_df[mask].reset_index(drop=True)
    
    print(f"Found {len(k_df_filtered)} matching events to plot.")
    
    root_file = uproot.open(KAONS_ROOT)
    tree = root_file["ana/raw;352"]
    
    for _, row in tqdm(k_df_filtered.iterrows(), total=len(k_df_filtered)):
        try:
            event = Event(tree=tree, filepath=KAONS_ROOT, index=row.event_index, plot=False)
            
            out_path = OUT_DIR / f"kaon_{row.run}_{row.subrun}_{row.event}.png"
            plot_event_displays(event, row.run, row.subrun, row.event, save_path=out_path)
            
        except Exception as e:
            print(f"Failed to plot run {row.run} event {row.event}: {e}")
            import matplotlib.pyplot as plt
            plt.close('all')

if __name__ == "__main__":
    main()
