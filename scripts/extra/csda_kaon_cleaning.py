"""
Kaon cluster selection for csda-kaon events.

Each event in the csda-kaon CSV is a confirmed kaon event, but the clustering
algorithm picks up multiple tracks per event (secondary muons, delta rays, etc.).
This script:
  1. Loads all matched clusters from csv_kaon_col.pkl (multiple per event)
  2. Computes features for each cluster inline
  3. Scores each cluster with a 3-class RF trained on reference proton/muon/kaon
  4. For each (run, subrun, event) selects the cluster with the highest P(kaon)
  5. Saves: csv_kaon_col_clean.pkl, csv_kaon_ind_clean.pkl
  6. Regenerates images: csv_kaon_48x48_raw_clean.pt

Downstream scripts should point to the *_clean files after running this.

Usage:
    uv run python scripts/extra/csda_kaon_cleaning.py --config configs/...yaml
"""

import sys
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.features import calorimetry as cal
from src.features import topology as topo
from src.images import pad_image_batch_gpu   # same padding as extract_csv_kaons

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--config", required=True)
args = parser.parse_args()

with open(args.config) as f:
    cfg = yaml.safe_load(f)

FEAT_DIR   = Path("/Volumes/easystore/proton-kaon/features")
CLUST_DIR  = Path("/Volumes/easystore/proton-kaon/clusters")
IMG_DIR    = Path("/Volumes/easystore/proton-kaon/images")
OUT_DIR    = Path("figs") / "csda_kaon_cleaning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Features to compute ───────────────────────────────────────────────────────
ALL_FEATURES = {
    'total_adc':               lambda img, cm: cal.total_adc(img),
    'mean_adc':                lambda img, cm: cal.mean_adc(img),
    'median_adc':              lambda img, cm: cal.median_adc(img),
    'max_adc':                 lambda img, cm: cal.max_adc(img),
    'std_adc':                 lambda img, cm: cal.std_adc(img),
    'adc_entropy':             lambda img, cm: cal.adc_entropy(img),
    'n_pixels':                lambda img, cm: topo.n_pixels(img),
    'solidity':                lambda img, cm: topo.solidity(img),
    'fill_fraction':           lambda img, cm: topo.fill_fraction(img),
    'bragg_peak_height':       lambda img, cm: cal.bragg_peak_height(cm),
    'max_ADC_position':        lambda img, cm: cal.max_ADC_position(cm),
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
FEAT_COLS = list(ALL_FEATURES.keys()) + ['height']

# ── Load all csda-kaon clusters ───────────────────────────────────────────────
print("Loading all csda-kaon clusters …")
col_all = pd.read_pickle(CLUST_DIR / "csv_kaon_col.pkl")
ind_all = pd.read_pickle(CLUST_DIR / "csv_kaon_ind.pkl")
print(f"  {len(col_all)} matched clusters across "
      f"{col_all.groupby(['run','subrun','event']).ngroups} events")

events_multi = col_all.groupby(['run','subrun','event']).size()
print(f"  events with >1 cluster: {(events_multi > 1).sum()} "
      f"/ {len(events_multi)} ({100*(events_multi>1).mean():.1f}%)")

# ── Compute features for every cluster ────────────────────────────────────────
print("Computing features for all csda-kaon clusters …")
records = []
for _, row in col_all.iterrows():
    img = np.array(row['image_intensity'])
    cm  = np.array(row['column_maxes'])
    rec = {'height': row['height']}
    for name, fn in ALL_FEATURES.items():
        try:
            rec[name] = fn(img, cm)
        except Exception:
            rec[name] = np.nan
    records.append(rec)
ck_feat = pd.DataFrame(records)
print(f"  done: {len(ck_feat)} clusters")

# ── Load reference features and train classifier ──────────────────────────────
print("\nLoading reference features …")
ref_df = pd.read_pickle(FEAT_DIR / "features.pkl")
ref_df = ref_df[ref_df['particle_type'].isin(['proton', 'muon', 'kaon'])].copy()
print(f"  reference: {len(ref_df)} events")

feat_cols_avail = [c for c in FEAT_COLS if c in ref_df.columns]
ref_clean = ref_df[feat_cols_avail + ['particle_type']].dropna()
print(f"  after NaN drop: {len(ref_clean)}")

X_ref = ref_clean[feat_cols_avail].values
y_ref = ref_clean['particle_type'].values

print("Training Random Forest (proton / muon / kaon) …")
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("rf", RandomForestClassifier(
        n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1,
    )),
])
clf.fit(X_ref, y_ref)
classes = list(clf.classes_)
k_idx = classes.index("kaon")
print(f"  classes: {classes}")

# ── Score every csda-kaon cluster ─────────────────────────────────────────────
ck_feat_avail = ck_feat[feat_cols_avail].copy()
finite_mask = ck_feat_avail.notna().all(axis=1)
proba_all = np.full((len(ck_feat), 3), np.nan)
if finite_mask.sum():
    proba_all[finite_mask.values] = clf.predict_proba(ck_feat_avail[finite_mask].values)

p_kaon = proba_all[:, k_idx]
ck_feat['p_kaon'] = p_kaon

any_finite = np.isfinite(proba_all).any(axis=1)
pred_class_arr = np.full(len(proba_all), 'unknown', dtype=object)
pred_class_arr[any_finite] = np.array(clf.classes_)[np.nanargmax(proba_all[any_finite], axis=1)]
ck_feat['pred_class'] = pred_class_arr

# ── Select one cluster per event: highest P(kaon) ─────────────────────────────
col_all = col_all.copy()
col_all['_p_kaon'] = p_kaon
col_all['_row_idx'] = np.arange(len(col_all))

# For events with all-NaN scores, fall back to match_score (lowest = best geometry)
def _select_best(group):
    scores = group['_p_kaon']
    if scores.notna().any():
        return group.loc[scores.idxmax(), '_row_idx']
    else:
        return group.loc[group['match_score'].idxmin(), '_row_idx']

best_rows = col_all.groupby(['run', 'subrun', 'event']).apply(_select_best).values.astype(int)
print(f"\nSelected {len(best_rows)} clusters (one per event)")

# ── Build cleaned DataFrames ──────────────────────────────────────────────────
col_clean = col_all.iloc[best_rows].drop(columns=['_p_kaon', '_row_idx']).reset_index(drop=True)
ind_clean = ind_all.iloc[best_rows].reset_index(drop=True)

# Pred breakdown on selected clusters
sel_pred = ck_feat.iloc[best_rows]['pred_class']
print("Predicted class of selected (one-per-event) clusters:")
for c, n in sel_pred.value_counts().items():
    print(f"  {c}: {n} ({100*n/len(sel_pred):.1f}%)")

# Save cleaned cluster pkls
col_clean.to_pickle(CLUST_DIR / "csv_kaon_col_clean.pkl")
ind_clean.to_pickle(CLUST_DIR / "csv_kaon_ind_clean.pkl")
print(f"Saved cleaned cluster pkls to {CLUST_DIR}/csv_kaon_*_clean.pkl")

# ── Regenerate images for selected clusters ───────────────────────────────────
print("\nRegenerating images for selected clusters …")
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps"  if torch.backends.mps.is_available() else "cpu")
print(f"  device: {device}")

c_list = col_clean['image_intensity'].tolist()
i_list = ind_clean['image_intensity'].tolist()

c_padded = np.array(pad_image_batch_gpu(c_list, device=device, batch_size=64, cut_rows=50))
i_padded = np.array(pad_image_batch_gpu(i_list, device=device, batch_size=64, cut_rows=50))

c_tensor = torch.from_numpy(c_padded).float().to(device)
i_tensor = torch.from_numpy(i_padded).float().to(device)

c_down = F.interpolate(c_tensor.unsqueeze(1), size=(48,48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()
i_down = F.interpolate(i_tensor.unsqueeze(1), size=(48,48), mode='bilinear', align_corners=False).squeeze(1).cpu().numpy()

images = np.stack([c_down, i_down], axis=1).astype(np.float32)
out_img = IMG_DIR / "csv_kaon_48x48_raw_clean.pt"
torch.save({"k": torch.from_numpy(images)}, out_img)
print(f"Saved {len(images)} images → {out_img}")

# ── Diagnostic plots ──────────────────────────────────────────────────────────
KEY_FEATURES = ['height', 'fill_fraction', 'solidity', 'mean_adc',
                'bragg_peak_ratio', 'end_vs_start_ratio', 'profile_cv', 'monotonic_rise_fraction']
KEY_FEATURES = [f for f in KEY_FEATURES if f in ref_df.columns]
CLASS_COLORS = {'kaon': '#D62728', 'muon': '#76B7B2', 'proton': '#4C78A8', 'unknown': 'gray'}

ncols, nrows = 4, 2
fig, axes = plt.subplots(nrows, ncols, figsize=(ncols*4, nrows*3.2))
axes = axes.flatten()

for ax, feat in zip(axes, KEY_FEATURES):
    lo, hi = np.nanpercentile(ref_df[feat].dropna(), [1, 99])
    bins = np.linspace(lo, hi, 40)

    for pt, color in [('proton','#4C78A8'), ('muon','#76B7B2'), ('kaon','#F58518')]:
        vals = ref_df[ref_df['particle_type'] == pt][feat].dropna().values
        if len(vals):
            ax.hist(vals, bins=bins, density=True, alpha=0.3, color=color, label=f'ref {pt}', lw=0)

    # Selected clusters coloured by predicted class
    sel_feat = ck_feat.iloc[best_rows]
    for cls in ['kaon', 'muon', 'proton', 'unknown']:
        vals_ck = sel_feat[sel_feat['pred_class'] == cls][feat].dropna().values
        if len(vals_ck):
            ax.hist(vals_ck, bins=bins, density=True, alpha=0.85,
                    color=CLASS_COLORS[cls], label=f'selected {cls}',
                    histtype='step', lw=2)
    ax.set_title(feat, fontsize=10)
    ax.set_ylabel('Density', fontsize=8)
    sns.despine(ax=ax)

handles = (
    [plt.Line2D([0],[0], color=c, lw=8, alpha=0.3, label=f'ref {p}')
     for p, c in [('proton','#4C78A8'),('muon','#76B7B2'),('kaon','#F58518')]]
    + [plt.Line2D([0],[0], color=CLASS_COLORS[c], lw=2, label=f'selected {c}')
       for c in ['kaon','muon','proton']]
)
axes[0].legend(handles=handles, fontsize=7, loc='upper right')
fig.suptitle('Selected kaon clusters vs reference distributions', fontsize=12, fontweight='bold')
fig.tight_layout()
fig.savefig(OUT_DIR / 'feature_grid.png', dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved diagnostic plot → {OUT_DIR}/feature_grid.png")

print("\nDone.")
print(f"  Cleaned pkls: {CLUST_DIR}/csv_kaon_*_clean.pkl")
print(f"  Cleaned images: {out_img}")
print(f"  Pass these to run_inference.py with --csda-kaon-path {out_img}")
