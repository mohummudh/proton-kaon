import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import calorimetry as cal
from src.features import topology as topo
from src.features.plot import hist
from src.cuts import image_cuts

# ── paths ──────────────────────────────────────────────────────────────────
COL_PKL   = Path('/Volumes/easystore/proton-kaon/clusters/col.pkl')
IND_PKL   = Path('/Volumes/easystore/proton-kaon/clusters/ind.pkl')
FEAT_DIR  = Path('/Volumes/easystore/proton-kaon/features')
FIGS_DIR  = Path('figs/features')

FEAT_DIR.mkdir(parents=True, exist_ok=True)
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# ── load data ───────────────────────────────────────────────────────────────
print("Loading col.pkl…")
col = pd.read_pickle(COL_PKL)
ind = pd.read_pickle(IND_PKL)
col, ind = image_cuts(col, ind, lower=10)

print(f"  {len(col)} rows")

# ── feature computation ─────────────────────────────────────────────────────
IMAGE_FEATURES = {
    'total_adc':   lambda img, cm: cal.total_adc(img),
    'mean_adc':    lambda img, cm: cal.mean_adc(img),
    'median_adc':  lambda img, cm: cal.median_adc(img),
    'max_adc':     lambda img, cm: cal.max_adc(img),
    'std_adc':     lambda img, cm: cal.std_adc(img),
    'adc_entropy': lambda img, cm: cal.adc_entropy(img),
    'n_pixels':    lambda img, cm: topo.n_pixels(img),
    'solidity':    lambda img, cm: topo.solidity(img),
}

PROFILE_FEATURES = {
    'bragg_peak_height':       lambda img, cm: cal.bragg_peak_height(cm),
    'bragg_peak_position':     lambda img, cm: cal.bragg_peak_position(cm),
    'bragg_peak_ratio':        lambda img, cm: cal.bragg_peak_ratio(cm),
    'bragg_peak_to_median':    lambda img, cm: cal.bragg_peak_to_median(cm),
    'end_vs_start_ratio':      lambda img, cm: cal.end_vs_start_ratio(cm),
    'last_quartile_mean':      lambda img, cm: cal.last_quartile_mean(cm),
    'first_quartile_mean':     lambda img, cm: cal.first_quartile_mean(cm),
    'bragg_rise_slope':        lambda img, cm: cal.bragg_rise_slope(cm),
    'peak_integral_fraction':  lambda img, cm: cal.peak_integral_fraction(cm),
    'bragg_peak_width':        lambda img, cm: cal.bragg_peak_width(cm),
    'profile_cv':              lambda img, cm: cal.profile_cv(cm),
    'monotonic_rise_fraction': lambda img, cm: cal.monotonic_rise_fraction(cm),
    'relative_peak_energy':    lambda img, cm: cal.relative_peak_energy(cm),
    'profile_skewness':        lambda img, cm: topo.profile_skewness(cm),
    'profile_kurtosis':        lambda img, cm: topo.profile_kurtosis(cm),
    'n_local_maxima':          lambda img, cm: topo.n_local_maxima(cm),
}

ALL_FEATURES = {**IMAGE_FEATURES, **PROFILE_FEATURES}

print("Computing features …")
records = []
for _, row in col.iterrows():
    img = np.array(row['image_intensity'])
    cm  = np.array(row['column_maxes'])
    rec = {
        'run':                  row['run'],
        'subrun':               row['subrun'],
        'event':                row['event'],
        'particle_type':        row['particle_type'],
        'height':               row['height'],
        'chi_squared_kaon':     row['chi_squared_kaon'],
        'chi_squared_proton':   row['chi_squared_proton']
    }
    for name, fn in ALL_FEATURES.items():
        try:
            rec[name] = fn(img, cm)
        except Exception:
            rec[name] = np.nan
    records.append(rec)

feat_df = pd.DataFrame(records)
print(f"  computed {len(ALL_FEATURES)} features for {len(feat_df)} events")

# ── log-likelihoods ──────────────────────────────────────────────────────────
ll_kaon   = pd.read_csv('/Volumes/easystore/proton-kaon/docs/kaon_df_plane_1_thr_DAQ.csv')
ll_proton = pd.read_csv('/Volumes/easystore/proton-kaon/docs/proton_df_plane_1_thr_DAQ.csv')
ll = pd.concat([ll_kaon, ll_proton], ignore_index=True).drop_duplicates(subset=['run', 'subrun', 'event'])

feat_df = feat_df.merge(
    ll[['run', 'subrun', 'event', 'log_likelihood_kaon', 'log_likelihood_proton']],
    on=['run', 'subrun', 'event'],
    how='left',
)
print(f"  log-likelihoods matched: {feat_df['log_likelihood_kaon'].notna().sum()} / {len(feat_df)}")

out_path = FEAT_DIR / 'features.pkl'
feat_df.to_pickle(out_path)
print(f"Saved features → {out_path}")

# ── histograms ───────────────────────────────────────────────────────────────
# patch plt.show so hist() doesn't try to open a window
plt.show = lambda: None

feature_names = list(ALL_FEATURES.keys())
print(f"Plotting {len(feature_names)} histograms …")
for feature in feature_names:
    try:
        hist(feat_df, feature)
        fig_path = FIGS_DIR / f'{feature}.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        plt.close('all')
        print(f"  saved {fig_path}")
    except Exception as e:
        print(f"  skipped {feature}: {e}")
        plt.close('all')

print("Done.")
