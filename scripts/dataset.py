# TODO: arguments, config, logging, comments, prints, plotting

import logging
import uproot
import pandas as pd

from pathlib import Path

from src.event import Event
from src.open_root import open_root
from src.clustering import extract_clusters
from src.cuts import cluster_cuts, reco_track_cuts
from src.matching import matching
from src.chi2 import parse_array, filter_arrays, build_res_index, chi2_track_alek, do_chi_squared
from src.bethe_bloch import bb_file

LOG_PATH = Path("logs/dataset.log")
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.FileHandler(LOG_PATH, mode='w', encoding="utf-8"),
        logging.StreamHandler()
    ],
)

logger = logging.getLogger(__name__)

PROTONS = "/Volumes/easystore/p_1track_protons_600_1600.root"                # picky protons + 1 track only in reco 
KAONS = "/Volumes/easystore/proton-kaon/raw/rawExtracted_350_650.root"                           # picky kaons
PROTONS_BB = '/Volumes/easystore/protons.txt'                                # Bethe-Bloch for protons
KAONS_BB = '/Volumes/easystore/kaons.txt'                                    # Bethe-Bloch for kaons

info = pd.read_csv('/Volumes/easystore/proton-deuteron/csv/picky+match.csv') # used for beamline information
trk = pd.read_csv('/Volumes/easystore/bruno/root/primary_trk_dedx_rr.csv')   # used for LArIAT reconstruction info.

# returns dataframes used to open root file + run, subrun, event to find clusters
p_df = open_root(PROTONS, tree_name="ana/raw")                              
k_df = open_root(KAONS, tree_name="ana/raw;352")

logger.info("%d proton events found", len(p_df))
logger.info("%d kaon events found", len(k_df))

# returns all clusters
p_clusters = extract_clusters(p_df, particle_type="proton", threshold=15, tree_name="ana/raw") 
k_clusters = extract_clusters(k_df, particle_type="kaon", threshold=15, tree_name="ana/raw;352")

logger.info("%d proton clusters found, %d kaon clusters found", len(p_clusters), len(k_clusters))

# removes obvious noise
p_clusters_cut = cluster_cuts(p_clusters)
k_clusters_cut = cluster_cuts(k_clusters)

logger.info("%d protons left, %d kaons left after cuts", len(p_clusters_cut), len(k_clusters_cut))

# use later for logging only
p_col = p_clusters_cut[p_clusters_cut['plane'] == 'collection']
p_ind = p_clusters_cut[p_clusters_cut['plane'] == 'induction']
k_col = k_clusters_cut[k_clusters_cut['plane'] == 'collection']
k_ind = k_clusters_cut[k_clusters_cut['plane'] == 'induction']

# matching across collection and induction plane
p_col, p_ind = matching(p_clusters_cut)
k_col, k_ind = matching(k_clusters_cut)

logger.info("%d protons left, %d kaons left after matching", len(p_col), len(k_col))
print(f'{len(p_col)} protons left, {len(k_col)} kaons left after matching')

col = pd.concat([k_col, p_col], ignore_index=True)
ind = pd.concat([k_ind, p_ind], ignore_index=True)

col = pd.merge(
    col,
    info[['run', 'subrun', 'event', 'p', 'm', 'beamline_mass']],
    on=['run', 'subrun', 'event'],
    how='left'
)
ind = pd.merge(
    ind,
    info[['run', 'subrun', 'event', 'p', 'm', 'beamline_mass']],
    on=['run', 'subrun', 'event'],
    how='left'
)

trk = reco_track_cuts(trk=trk)

col = pd.merge(
    col,
    trk[['run', 'subrun', 'event', 'trkrr', 'trkdedx']],
    on=['run', 'subrun', 'event'],
    how='inner'
)
ind = pd.merge(
    ind,
    trk[['run', 'subrun', 'event', 'trkrr', 'trkdedx']],
    on=['run', 'subrun', 'event'],
    how='inner'
)

bb_p, res_p = bb_file(PROTONS_BB)
bb_k, res_k = bb_file(KAONS_BB)

col = do_chi_squared(col, res_k, bb_k, res_p, bb_p)
ind = do_chi_squared(ind, res_k, bb_k, res_p, bb_p)

logger.info("%d total candidates", len(col))

col.to_pickle('/Volumes/easystore/proton-kaon/clusters/col.pkl')
ind.to_pickle('/Volumes/easystore/proton-kaon/clusters/ind.pkl')