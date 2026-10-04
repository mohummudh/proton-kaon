#!/usr/bin/env python3
"""Paper-style PDF/PNG figures; all numbers read from baseline artifacts."""
from pathlib import Path
import os
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from _beam_data import apply_style, savefig, SINGLE_COL, DOUBLE_COL, COLOURS, DISPLAY, SPECIES
from representation_baselines import OUT

FIGS = Path(os.environ.get('REPRESENTATION_BASELINES_FIGS', ROOT / 'figs/representation_baselines'))
ORDER = ['pixel_pca', 'random', 'endpoint_pca', 'endpoint', 'ae', 'vae', 'active_mask_ae', 'fulltrack_pca', 'fulltrack']
LABELS = {'pixel_pca': 'Pixel PCA (8)', 'random': 'Random CNN (8)',
          'endpoint_pca': 'Endpoint features → PCA (8)', 'endpoint': 'Endpoint features (32)',
          'ae': 'Autoencoder (8)', 'vae': 'VAE (8)', 'active_mask_ae': 'Active-patch AE (8)',
          'fulltrack_pca': 'Full-track features → PCA (8)', 'fulltrack': 'Full-track features (26)'}
SHORT = {'pixel_pca': 'Pixel PCA', 'random': 'Random CNN', 'endpoint_pca': 'Endpoint PCA',
         'endpoint': 'Endpoint features', 'ae': 'AE', 'vae': 'VAE', 'active_mask_ae': 'Active-patch AE',
         'fulltrack_pca': 'Full-track PCA', 'fulltrack': 'Full-track features'}
PALETTE = {'pixel_pca': '#AA3377', 'random': '#999999', 'endpoint_pca': '#66CCEE',
           'endpoint': '#228833', 'ae': '#009988', 'vae': '#0077BB', 'active_mask_ae': '#EE7733',
           'fulltrack_pca': '#CCBB44', 'fulltrack': '#333333'}


def collect(name):
    files = list((OUT / 'evaluation').glob(f'*/{name}.csv'))
    return pd.concat([pd.read_csv(f) for f in files if f.stat().st_size > 2], ignore_index=True) if files else pd.DataFrame()


def finish_axes(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(direction='out')


def methods(d):
    return [m for m in ORDER if m in set(d.method)]


def row_axis(ax, order, labels=True):
    ax.set_yticks(range(len(order)), [LABELS[m] for m in order] if labels else [])
    ax.set_ylim(len(order) - .5, -.5)
    if 'fulltrack_pca' in order:
        sep = order.index('fulltrack_pca') - .5
        ax.axhspan(sep, len(order) - .5, color='0.95', zorder=0)
        ax.axhline(sep, color='0.75', lw=.6)
    finish_axes(ax)


def plot_probes(d):
    d = d[d.method.isin(ORDER)]
    for part in ['test', 'run_test']:
        sub = d[d.partition == part]; order = methods(sub)
        if not len(sub): continue
        apply_style(SINGLE_COL)
        fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL + 1.25, 4.6), sharey=True)
        for col, target in enumerate(['mean_adc', 'median_adc', 'solidity']):
            ax = axes[col]
            for i, method in enumerate(order):
                for j, sp in enumerate(SPECIES):
                    a = sub[(sub.method == method) & (sub.target == target) & (sub.species == sp)]
                    if not len(a): continue
                    # Between-training-seed spread; conditional event CIs retained in tables.
                    y = i + (j - 1) * .2
                    ax.errorbar(a.auc.mean(), y, xerr=a.auc.std(ddof=1) if len(a) > 1 else 0,
                                fmt='o', ms=3.5, color=COLOURS[sp], capsize=2, lw=.8)
                    if len(a) > 1:
                        ax.scatter(a.auc, np.full(len(a), y), s=5, color=COLOURS[sp], alpha=.4)
            row_axis(ax, order, col == 0)
            ax.axvline(.5, color='.55', ls='--', lw=.65)
            ax.set_xlim(.48, 1.015); ax.set_xticks([.5, .7, .9, 1.])
            ax.set_xlabel('Held-out AUC')
            ax.set_title(['(a) Mean ADC', '(b) Median positive ADC', '(c) Solidity'][col], fontsize=9)
        axes[0].set_yticks(range(len(order)), [LABELS[m] for m in order])
        handles = [Line2D([], [], marker='o', ls='', color=COLOURS[s], label=DISPLAY[s], ms=4) for s in SPECIES]
        fig.legend(handles=handles, loc='lower center', ncol=3, frameon=False, bbox_to_anchor=(.62, -.01))
        fig.tight_layout(rect=[0, .045, 1, 1])
        savefig(fig, FIGS, f'physical_probes_{part}')


def plot_clustering(d):
    d = d[(d.k == 3) & d.method.isin(ORDER)]
    if d.empty: return
    order = methods(d); apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 4.35), sharey=True)
    for j, metric in enumerate(['ari', 'mapped_tag_agreement']):
        ax = axes[j]
        for i, method in enumerate(order):
            for p, (part, color) in enumerate([('test', '#0077BB'), ('run_test', '#EE7733')]):
                a = d[(d.method == method) & (d.partition == part)]
                ax.errorbar(a[metric].mean(), i + (p - .5) * .23, xerr=a[metric].std(),
                            fmt='o', ms=4, color=color, capsize=2, lw=.8)
        row_axis(ax, order, j == 0)
        ax.set_xlabel(['ARI against beamline tags', 'Mapped tag agreement'][j])
        ax.set_title(['(a) Partition agreement, k = 3', '(b) Development-named groups'][j], fontsize=9)
    axes[0].set_yticks(range(len(order)), [LABELS[m] for m in order])
    fig.legend(handles=[Line2D([], [], marker='o', ls='', color=c, label=l) for c, l in
                        [('#0077BB', 'Event-disjoint test'), ('#EE7733', 'Held-out runs')]],
               loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(.62, -.01))
    fig.tight_layout(rect=[0, .05, 1, 1]); savefig(fig, FIGS, 'clustering_comparison')


def plot_coverage(d):
    if d.empty: return
    use = ['endpoint', 'pixel_pca', 'ae', 'vae', 'active_mask_ae', 'fulltrack']
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.25), sharey=True)
    for ax, part, title in zip(axes, ['test', 'run_test'], ['(a) Event-disjoint test', '(b) Held-out runs']):
        for method in use:
            a = d[(d.method == method) & (d.partition == part) & (d.k == 15)]
            if a.empty: continue
            a = a.groupby('dev_purity_threshold')[['coverage', 'tag_agreement']].mean().sort_values('coverage')
            ax.plot(a.coverage, a.tag_agreement, '-o', ms=3, lw=1, color=PALETTE[method], label=SHORT[method])
        ax.set(xlabel='Retained fraction', xlim=(0, 1.02), ylim=(.45, 1.01), title=title)
        finish_axes(ax)
    axes[0].set_ylabel('Held-out tag agreement')
    axes[1].legend(frameon=False, fontsize=7, loc='lower left')
    fig.tight_layout(); savefig(fig, FIGS, 'purity_coverage_k15')


def plot_mass(d):
    if d.empty: return
    d = d[(d.k == 15) & (d.quality == 'picky') & d.method.isin(ORDER)]
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.25), sharey=True)
    for ax, part, title in zip(axes, ['test', 'run_test'], ['(a) Event-disjoint test', '(b) Held-out runs']):
        for method in ['endpoint', 'pixel_pca', 'ae', 'vae', 'active_mask_ae', 'fulltrack']:
            a = d[(d.method == method) & (d.partition == part)]
            if a.empty: continue
            a = a.groupby('fraction').mass_shift.agg(['mean', 'std'])
            ax.errorbar(a.index, a['mean'], yerr=a['std'].fillna(0), fmt='-o', ms=3,
                        color=PALETTE[method], lw=1, capsize=2, label=SHORT[method])
        ax.axhline(0, color='.5', ls='--', lw=.6)
        ax.set(xlabel='Fraction flagged within momentum bins', xticks=[.1, .2, .3, .5], title=title)
        finish_axes(ax)
    axes[0].set_ylabel('Mean mass contrast [MeV/$c^2$]')
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); savefig(fig, FIGS, 'beamline_mass_matched_coverage')


def plot_increment(d):
    if d.empty: return
    d = d[(d.quality == 'picky') & (d.reader == 'nonlinear') & d.method.isin(ORDER)]
    order = [m for m in ['pixel_pca', 'random', 'endpoint_pca', 'ae', 'vae', 'active_mask_ae', 'fulltrack_pca', 'fulltrack'] if m in set(d.method)]
    rows = []
    for method in order:
        for part in ['test', 'run_test']:
            a = d[(d.method == method) & (d.partition == part)]
            p = a.pivot(index='seed', columns='features', values='r2')
            if p.empty: continue
            diff = p.covariates_endpoint_representation - p.covariates_endpoint
            # Paired event bootstrap for the mean over training seeds.
            draws = []
            for seed in p.index:
                stem = method if seed == -1 else f'{method}_s{seed}'
                f = OUT / 'evaluation' / stem / 'mass_prediction_bootstrap.npz'
                if f.exists():
                    b = np.load(f)
                    draws.append(b[f'covariates_endpoint_representation_nonlinear_{part}_picky'] -
                                 b[f'covariates_endpoint_nonlinear_{part}_picky'])
            lo, hi = np.quantile(np.mean(draws, axis=0), [.025, .975]) if draws else (np.nan, np.nan)
            rows.append({'method': method, 'partition': part, 'delta_r2': diff.mean(), 'low': lo, 'high': hi,
                         'seed_sd': diff.std(), 'n_seeds': len(diff)})
    result = pd.DataFrame(rows); result.to_csv(OUT / 'mass_increment_summary.csv', index=False)
    apply_style(SINGLE_COL)
    fig, ax = plt.subplots(figsize=(DOUBLE_COL, 3.8))
    for i, method in enumerate(order):
        for j, (part, color) in enumerate([('test', '#0077BB'), ('run_test', '#EE7733')]):
            a = result[(result.method == method) & (result.partition == part)]
            if a.empty: continue
            a = a.iloc[0]
            ax.plot([a.low, a.high], [i + (j - .5) * .24] * 2, color=color, lw=1)
            ax.plot(a.delta_r2, i + (j - .5) * .24, 'o', color=color, ms=4)
    ax.axvline(0, color='.4', ls='--', lw=.7)
    ax.set_yticks(range(len(order)), [LABELS[m] for m in order]); ax.invert_yaxis()
    ax.set_xlabel('Additional held-out mass $R^2$ beyond momentum, quality and endpoint features')
    finish_axes(ax)
    ax.legend(handles=[Line2D([], [], marker='o', color=c, ls='-', label=l) for c, l in
                       [('#0077BB', 'Event-disjoint test'), ('#EE7733', 'Held-out runs')]], frameon=False, fontsize=7)
    fig.tight_layout(); savefig(fig, FIGS, 'incremental_beamline_information')


def plot_learning(d):
    if d.empty: return
    d = d[d.partition == 'test'].copy()
    d['base'] = d.method.str.replace(r'_n\d+$', '', regex=True)
    d['n'] = d.method.str.extract(r'_n(\d+)$')[0].fillna(9417).astype(int)
    use = ['pixel_pca', 'random', 'endpoint', 'ae', 'vae', 'active_mask_ae']
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.95))
    for j, target in enumerate(['mean_adc', 'median_adc', 'solidity']):
        ax = axes[j]
        for method in use:
            a = d[(d.base == method) & (d.target == target)].groupby(['n', 'seed']).auc.mean().reset_index()
            if a.empty: continue
            a = a.groupby('n').auc.agg(['mean', 'std'])
            ax.errorbar(a.index, a['mean'], yerr=a['std'].fillna(0), fmt='-o', ms=3, lw=1,
                        color=PALETTE[method], label=SHORT[method], capsize=2)
        ax.set_xscale('log'); ax.set_xticks([100, 1000, 9417], ['100', '1,000', '9,417'])
        ax.set_title(['(a) Mean ADC', '(b) Median positive ADC', '(c) Solidity'][j], fontsize=8.5)
        ax.set_xlabel('Pretraining images'); finish_axes(ax)
    axes[0].set_ylabel('Mean within-tag AUC')
    axes[2].legend(frameon=False, fontsize=6.5, loc='lower right')
    fig.tight_layout(); savefig(fig, FIGS, 'data_efficiency_fixed_evaluation')


def plot_training():
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.65))
    for ax, method in zip(axes, ['ae', 'vae', 'active_mask_ae']):
        for seed, color in enumerate(['#0077BB', '#EE7733', '#AA3377']):
            f = OUT / 'models' / f'{method}_s{seed}.csv'
            if not f.exists(): continue
            d = pd.read_csv(f)
            ax.plot(d.epoch, d.dev_objective, color=color, lw=1, label=f'Seed {seed}')
            best = d.dev_objective.idxmin()
            ax.plot(d.loc[best, 'epoch'], d.loc[best, 'dev_objective'], 'o', ms=3, color=color)
        ax.set(title=SHORT[method], xlabel='Epoch'); ax.set_yscale('log'); finish_axes(ax)
    axes[0].set_ylabel('Development objective')
    axes[2].legend(frameon=False, fontsize=7)
    fig.tight_layout(); savefig(fig, FIGS, 'training_convergence')


def main():
    FIGS.mkdir(parents=True, exist_ok=True)
    tables = {n: collect(n) for n in ['probes', 'clustering', 'coverage', 'mass', 'mass_prediction', 'tag_probe']}
    if (OUT / 'MAIN_ONLY.txt').exists():
        tables = {n: d[~d.method.str.contains('_n100')].copy() if not d.empty else d for n, d in tables.items()}
    for n, d in tables.items():
        if not d.empty: d.to_csv(OUT / f'{n}_all.csv', index=False)
    if not tables['probes'].empty:
        plot_probes(tables['probes'])
        if not (OUT / 'MAIN_ONLY.txt').exists(): plot_learning(tables['probes'])
    plot_clustering(tables['clustering']); plot_coverage(tables['coverage'])
    plot_mass(tables['mass']); plot_increment(tables['mass_prediction']); plot_training(); plot_reconstruction()


def plot_reconstruction():
    df = pd.read_csv(OUT / 'manifest.csv'); rows = []
    for f in (OUT / 'representations').glob('*_reconstruction.npy'):
        if (OUT / 'MAIN_ONLY.txt').exists() and '_n100' in f.stem: continue
        stem = f.stem.replace('_reconstruction', '')
        method = stem.rsplit('_s', 1)[0] if '_s' in stem else stem
        seed = int(stem.rsplit('_s', 1)[1]) if '_s' in stem else -1
        base = method.split('_n')[0]
        n = int(method.split('_n')[1]) if '_n' in method else 9417
        error = np.load(f)
        for part in ['test', 'run_test']:
            value = np.mean([error[(df.partition == part) & (df.species == sp)].mean() for sp in SPECIES])
            rows.append({'method': base, 'n_train': n, 'seed': seed, 'partition': part,
                         'weighted_reconstruction': value})
    d = pd.DataFrame(rows)
    if d.empty: return
    d.to_csv(OUT / 'reconstruction_all.csv', index=False)
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.85), sharey=True)
    use = ['pixel_pca', 'ae', 'vae', 'active_mask_ae']
    if d.n_train.nunique() == 1:
        for ax, part, title in zip(axes, ['test', 'run_test'], ['(a) Event-disjoint test', '(b) Held-out runs']):
            for i, method in enumerate(use):
                x = d[(d.method == method) & (d.partition == part)].weighted_reconstruction
                if x.empty: continue
                ax.errorbar(x.mean(), i, xerr=x.std(ddof=1) if len(x) > 1 else 0,
                            fmt='o', ms=4, color=PALETTE[method], lw=.8, capsize=2)
                ax.scatter(x, np.full(len(x), i), s=6, color=PALETTE[method], alpha=.5)
            ax.set_yticks(range(len(use)))
            ax.set_ylim(len(use) - .5, -.5)
            ax.set(xlabel='Weighted reconstruction error', title=title)
            finish_axes(ax)
        axes[0].set_yticklabels([SHORT[m] for m in use])
        axes[1].tick_params(labelleft=False)
    else:
        for ax, part, title in zip(axes, ['test', 'run_test'], ['(a) Event-disjoint test', '(b) Held-out runs']):
            for method in use:
                a = d[(d.method == method) & (d.partition == part)].groupby('n_train').weighted_reconstruction.agg(['mean', 'std'])
                if a.empty: continue
                ax.errorbar(a.index, a['mean'], yerr=a['std'].fillna(0), fmt='-o', ms=3,
                            color=PALETTE[method], label=SHORT[method], lw=1, capsize=2)
            ax.set_xscale('log'); ax.set_yscale('log'); ax.set_xticks([100, 1000, 9417], ['100', '1,000', '9,417'])
            ax.set(xlabel='Pretraining images', title=title); finish_axes(ax)
        axes[0].set_ylabel('Weighted reconstruction error')
        axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout(); savefig(fig, FIGS, 'reconstruction_fixed_evaluation')


if __name__ == '__main__': main()
