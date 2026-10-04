#!/usr/bin/env python3
"""Compare matched log1p and raw-ADC eight-dimensional representations."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/extra')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import orthogonal_procrustes
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from _beam_data import apply_style, savefig, SINGLE_COL, DOUBLE_COL

LOG = ROOT / 'output/representation_baselines'
RAW = ROOT / 'output/representation_baselines_raw'
FIGS = RAW / 'staging_figs'
METHODS = ['pixel_pca', 'random', 'ae', 'vae', 'active_mask_ae']
DISPLAY = {'pixel_pca': 'Pixel PCA', 'random': 'Random CNN', 'ae': 'AE',
           'vae': 'VAE', 'active_mask_ae': 'Active-patch AE'}
COLORS = {'pixel_pca': '#AA3377', 'random': '#999999', 'ae': '#009988',
          'vae': '#0077BB', 'active_mask_ae': '#EE7733'}


def stem(method, seed):
    return method if method == 'pixel_pca' else f'{method}_s{seed}'


def geometry(df):
    tr = df.partition.eq('train').to_numpy()
    sample = np.random.default_rng(12).choice(np.flatnonzero(df.partition.eq('test')), 1500, replace=False)
    rows = []
    for method in METHODS:
        for seed in ([-1] if method == 'pixel_pca' else [0, 1, 2]):
            s = stem(method, seed)
            a = np.load(LOG / 'representations' / f'{s}.npy')
            b = np.load(RAW / 'representations' / f'{s}.npy')
            assert a.shape == b.shape and a.shape[1] == 8
            a = StandardScaler().fit(a[tr]).transform(a)[sample]
            b = StandardScaler().fit(b[tr]).transform(b)[sample]
            ac, bc = a - a.mean(0), b - b.mean(0)
            cka = np.linalg.norm(ac.T @ bc, 'fro') ** 2 / (
                np.linalg.norm(ac.T @ ac, 'fro') * np.linalg.norm(bc.T @ bc, 'fro'))
            ia = NearestNeighbors(n_neighbors=11).fit(a).kneighbors(a, return_distance=False)[:, 1:]
            ib = NearestNeighbors(n_neighbors=11).fit(b).kneighbors(b, return_distance=False)[:, 1:]
            overlap = np.mean([len(set(x) & set(y)) / 10 for x, y in zip(ia, ib)])
            rows.append({'method': method, 'seed': seed, 'n_test': len(sample),
                         'linear_cka': cka, 'neighbor_overlap10': overlap})
    result = pd.DataFrame(rows)
    result.to_csv(RAW / 'log_vs_raw_geometry.csv', index=False)
    return result


def performance():
    rows = []
    for scale, folder in [('log1p', LOG), ('raw', RAW)]:
        probes = pd.read_csv(folder / 'probes_all.csv')
        cluster = pd.read_csv(folder / 'clustering_all.csv')
        for method in METHODS:
            p = probes[(probes.method == method) & (probes.partition == 'test')]
            c = cluster[(cluster.method == method) & (cluster.partition == 'test') & (cluster.k == 3)]
            for target in ['mean_adc', 'median_adc', 'solidity']:
                q = p[p.target == target]
                for seed, x in q.groupby('seed'):
                    rows.append({'scale': scale, 'method': method, 'seed': seed,
                                 'metric': target + '_mean_tag_auc', 'value': x.auc.mean()})
            for seed, x in c.groupby('seed'):
                rows.append({'scale': scale, 'method': method, 'seed': seed,
                             'metric': 'k3_tag_ari', 'value': x.ari.mean()})
    result = pd.DataFrame(rows)
    result.to_csv(RAW / 'log_vs_raw_performance.csv', index=False)
    return result


def charge_displacement(df):
    """Ask whether the input-transform shift varies with endpoint charge."""
    tr = df.partition.eq('train').to_numpy()
    te = df.partition.eq('test').to_numpy()
    endpoint = np.load(LOG / 'endpoint.npy')
    charge = np.mean(endpoint[:, [1, 17]], axis=1)
    bounds = np.quantile(charge[te], np.linspace(0, 1, 11))
    rows = []
    for seed in range(3):
        s = f'vae_s{seed}'
        a = np.load(LOG / 'representations' / f'{s}.npy')
        b = np.load(RAW / 'representations' / f'{s}.npy')
        a = StandardScaler().fit(a[tr]).transform(a)
        b = StandardScaler().fit(b[tr]).transform(b)
        rotation, _ = orthogonal_procrustes(b[tr], a[tr])
        shift = np.linalg.norm(a[te] - b[te] @ rotation, axis=1)
        decile = np.clip(np.digitize(charge[te], bounds[1:-1]), 0, 9)
        for i in range(10):
            q = decile == i
            rows.append({'seed': seed, 'charge_decile': i + 1, 'n': int(q.sum()),
                         'endpoint_median_adc': float(np.median(charge[te][q])),
                         'median_aligned_latent_shift': float(np.median(shift[q]))})
    result = pd.DataFrame(rows)
    result.to_csv(RAW / 'log_vs_raw_charge_displacement.csv', index=False)
    return result


def plot(geom, perf, shift):
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.8))
    for ax, metric, label in zip(axes, ['linear_cka', 'neighbor_overlap10'],
                                 ['Linear CKA', 'Shared ten-neighbor fraction']):
        for i, method in enumerate(METHODS):
            x = geom[geom.method == method][metric]
            ax.bar(i, x.mean(), color=COLORS[method], width=.65)
            ax.scatter(np.full(len(x), i), x, color='black', s=7, zorder=3)
        ax.set_xticks(range(len(METHODS)), [DISPLAY[m] for m in METHODS], rotation=28, ha='right')
        ax.set_ylabel(label); ax.set_ylim(0, 1)
        ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); savefig(fig, FIGS, 'log_vs_raw_geometry')

    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.9))
    for ax, metric, label in zip(axes, ['solidity_mean_tag_auc', 'k3_tag_ari'],
                                 ['Solidity probe AUC', 'Three-cluster tag ARI']):
        for i, method in enumerate(METHODS):
            for j, (scale, color) in enumerate([('log1p', '#0077BB'), ('raw', '#EE7733')]):
                x = perf[(perf.method == method) & (perf.scale == scale) & (perf.metric == metric)].value
                ax.bar(i + (j - .5) * .32, x.mean(), width=.3, color=color,
                       label=scale if i == 0 else None)
                ax.scatter(np.full(len(x), i + (j - .5) * .32), x, color='black', s=6, zorder=3)
        ax.set_xticks(range(len(METHODS)), [DISPLAY[m] for m in METHODS], rotation=28, ha='right')
        ax.set_ylabel(label); ax.spines[['top', 'right']].set_visible(False)
    axes[0].set_ylim(.5, 1); axes[1].set_ylim(0, .7)
    axes[1].legend(frameon=False)
    fig.tight_layout(); savefig(fig, FIGS, 'log_vs_raw_performance')

    apply_style(SINGLE_COL)
    fig, ax = plt.subplots(figsize=(SINGLE_COL, 2.5))
    for seed, group in shift.groupby('seed'):
        ax.plot(group.endpoint_median_adc, group.median_aligned_latent_shift,
                '-o', color='#0077BB', alpha=.3, lw=.7, ms=2)
    mean = shift.groupby('charge_decile').agg({'endpoint_median_adc': 'mean',
                                                'median_aligned_latent_shift': 'mean'})
    ax.plot(mean.endpoint_median_adc, mean.median_aligned_latent_shift,
            '-o', color='#0077BB', lw=1.3, ms=3, label='Mean over seeds')
    ax.set(xlabel='Endpoint median positive ADC', ylabel='Aligned code shift (train SD units)')
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout(); savefig(fig, FIGS, 'log_vs_raw_charge_shift')


def main():
    a = pd.read_csv(LOG / 'manifest.csv')
    b = pd.read_csv(RAW / 'manifest.csv')
    assert a.equals(b), 'The input conditions do not share an identical manifest'
    plot(geometry(a), performance(), charge_displacement(a))


if __name__ == '__main__': main()
