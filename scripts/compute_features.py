import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml

from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import calorimetry as cal
from src.features import topology as topo
from src.features.plot import hist, plot_umap
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

feat_df.loc[feat_df['log_likelihood_proton'] < -5000, 'log_likelihood_proton'] = np.nan
feat_df.loc[feat_df['log_likelihood_kaon'] < -5000, 'log_likelihood_kaon'] = np.nan
feat_df.loc[feat_df['chi_squared_proton'] > 10, 'chi_squared_proton'] = np.nan
feat_df.loc[feat_df['chi_squared_kaon'] > 10, 'chi_squared_kaon'] = np.nan

out_path = FEAT_DIR / 'features.pkl'
feat_df.to_pickle(out_path)
print(f"Saved features → {out_path}")

# ── histograms ───────────────────────────────────────────────────────────────
# patch plt.show so hist() doesn't try to open a window
plt.show = lambda: None

feature_names = list(ALL_FEATURES.keys()) + ['height', 'chi_squared_kaon', 'chi_squared_proton', 'log_likelihood_kaon', 'log_likelihood_proton']
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

# ── umap ───────────────────────────────────────────────────────────────────
try:
    import umap

    with open("configs/default.yaml") as f:
        cfg = yaml.safe_load(f)

    name = (
        f"model_{cfg['model']['type']}"
        f"_latent{cfg['model']['latent']}"
        f"_ch{'_'.join(str(c) for c in cfg['model']['channels'])}"
        f"_beta{cfg['train']['beta']}"
        f"_lr{cfg['optimizer']['lr']}"
        f"_epoch{cfg['train']['epochs']}"
        f"_act{cfg['model']['activation']}"
        f"_kern{cfg['model']['kernel']}"
        f"_stride{cfg['model']['stride']}"
        f"_pad{cfg['model']['padding']}"
    )

    inf_dir = Path(cfg["output"]["inference_dir"]) / name
    train_latents = np.load(inf_dir / "train.npz")["latents"]
    val_latents   = np.load(inf_dir / "val.npz")["latents"]
    kaon_latents  = np.load(inf_dir / "kaon.npz")["latents"]

    idx = np.load('/Volumes/easystore/proton-kaon/training/split_p.npz')
    train_features = feat_df[feat_df['particle_type'] == 'proton'].iloc[idx['train_idx']]
    val_features   = feat_df[feat_df['particle_type'] == 'proton'].iloc[idx['val_idx']]
    kaon_features  = feat_df[feat_df['particle_type'] == 'kaon']

    all_latents = np.vstack([train_latents, val_latents, kaon_latents])

    reducer_path = inf_dir / 'reducer.pkl'
    if reducer_path.exists():
        import pickle
        with open(reducer_path, 'rb') as f:
            reducer = pickle.load(f)
        print(f"Loaded existing UMAP reducer from {reducer_path}")
    else:
        reducer = umap.UMAP(n_neighbors=30, min_dist=0.1)
        reducer.fit(all_latents)
        import pickle
        with open(reducer_path, 'wb') as f:
            pickle.dump(reducer, f)
        print(f"Saved UMAP reducer to {reducer_path}")

    train_umap = reducer.transform(train_latents)
    val_umap   = reducer.transform(val_latents)
    kaon_umap  = reducer.transform(kaon_latents)

    metadata = {'run', 'subrun', 'event', 'particle_type'}
    umap_features = [c for c in train_features.columns if c not in metadata]

    umap_dir = FIGS_DIR.parent / 'umap'
    umap_dir.mkdir(parents=True, exist_ok=True)

    print(f"Plotting {len(umap_features)} UMAP features…")
    for feature in umap_features:
        try:
            fig, axes = plot_umap(train_umap, train_features, val_umap, val_features, kaon_umap, kaon_features, feature)
            plt.savefig(umap_dir / f'{feature}.png', dpi=150, bbox_inches='tight')
            plt.close('all')
            print(f"  saved {feature}")
        except Exception as e:
            print(f"  skipped {feature}: {e}")
            plt.close('all')
    print(f"Saved UMAP plots to {umap_dir}")

except ImportError:
    print("umap not installed, skipping UMAP plots")

print("Done.")
