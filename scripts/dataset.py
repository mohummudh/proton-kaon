# TODO: arguments, config, logging, comments, prints, plotting
# TODO: fix problem with Event in loading raw channel and adc channel across 2 name schemes

import uproot
import pandas as pd

from src.event import Event
from src.open_root import open_root
from src.clustering import extract_clusters
from src.cuts import cluster_cuts, reco_track_cuts
from src.matching import matching
from src.chi2 import parse_array, filter_arrays, build_res_index, chi2_track_alek, do_chi_squared
from src.bethe_bloch import bb_file

PROTONS = "/Volumes/easystore/p_1track_protons_600_1600.root"
KAONS = "/Volumes/easystore/RAW_Kaon_AnaTree.root"
PROTONS_BB = '/Volumes/easystore/protons.txt'
KAONS_BB = '/Volumes/easystore/kaons.txt'

info = pd.read_csv('/Volumes/easystore/proton-deuteron/csv/picky+match.csv')
trk = pd.read_csv('/Volumes/easystore/bruno/root/primary_trk_dedx_rr.csv')

p_df = open_root(PROTONS, tree_name="ana/raw")
k_df = open_root(KAONS, tree_name="anatree/raw;143")

p_clusters = extract_clusters(p_df, particle_type="proton", threshold=15, tree_name="ana/raw")
k_clusters = extract_clusters(k_df, particle_type="kaon", threshold=15, tree_name="anatree/raw;143")

p_clusters_cut = cluster_cuts(p_clusters)
k_clusters_cut = cluster_cuts(k_clusters)

# use later for logging only
p_col = p_clusters_cut[p_clusters_cut['plane'] == 'collection']
p_ind = p_clusters_cut[p_clusters_cut['plane'] == 'induction']
k_col = k_clusters_cut[k_clusters_cut['plane'] == 'collection']
k_ind = k_clusters_cut[k_clusters_cut['plane'] == 'induction']

p_col, p_ind = matching(p_clusters_cut)
k_col, k_ind = matching(k_clusters_cut)

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

col.to_pickle('/Volumes/easystore/proton-kaon/clusters/col.pkl')
ind.to_pickle('/Volumes/easystore/proton-kaon/clusters/ind.pkl')