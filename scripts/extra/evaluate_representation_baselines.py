#!/usr/bin/env python3
"""Evaluate each completed representation without retraining the encoder.

Frozen protocol: train GMM on train; name clusters/probe on dev; score test/run_test.
All bootstrap intervals resample event groups, conditional on fitted models.
"""
import argparse
import json
from pathlib import Path
import sys
import warnings
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, adjusted_rand_score, r2_score, roc_auc_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from _beam_data import SPECIES
from representation_baselines import OUT, write_json, digest

TARGETS = ['mean_adc', 'median_adc', 'solidity']
KS = [3, 8, 15, 25, 37, 50]
BOOT = 250


def boot_weights(groups, n=BOOT, seed=731):
    _, inverse = np.unique(groups, return_inverse=True)
    g = inverse.max() + 1
    rng = np.random.default_rng(seed)
    return [np.bincount(rng.integers(g, size=g), minlength=g)[inverse] for _ in range(n)]


def interval(values):
    a = np.asarray(values); a = a[np.isfinite(a)]
    return [float(x) for x in np.quantile(a, [.025, .975])] if len(a) else [np.nan, np.nan]


def repr_info(stem):
    if '_s' in stem:
        method, seed = stem.rsplit('_s', 1)
        return method, int(seed)
    return stem, -1


def nonlinear():
    # Small, fixed-capacity nonlinear reader; no test-set tuning or random validation split.
    return HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=7,
                                         l2_regularization=10, min_samples_leaf=25,
                                         early_stopping=False, random_state=0)


def probes(z, df, method, seed, dest):
    rows, predictions, boots = [], {}, {}
    for sp in SPECIES:
        dev = (df.partition == 'dev') & (df.species == sp)
        for target in TARGETS:
            fitidx = np.flatnonzero(dev & np.isfinite(df[target]))
            yy = df[target].to_numpy()
            median = np.median(yy[fitidx]); binary = (yy[fitidx] > median).astype(int)
            scaler = StandardScaler().fit(z[fitidx]); x = scaler.transform(z)
            lr = LogisticRegression(C=1., max_iter=1500, class_weight='balanced').fit(x[fitidx], binary)
            ridge = Ridge(alpha=10).fit(x[fitidx], yy[fitidx])
            tree = nonlinear().fit(x[fitidx], yy[fitidx])
            for part in ['test', 'run_test']:
                idx = np.flatnonzero((df.partition == part) & (df.species == sp) & np.isfinite(yy))
                y = yy[idx]; b = (y > median).astype(int)
                p = lr.predict_proba(x[idx])[:, 1]; r = ridge.predict(x[idx]); t = tree.predict(x[idx])
                weights = boot_weights(df.event_key.to_numpy()[idx])
                ba = [roc_auc_score(b, p, sample_weight=w) for w in weights]
                lo, hi = interval(ba)
                key = f'{part}_{sp}_{target}'
                predictions[key + '_idx'] = idx; predictions[key + '_prob'] = p
                predictions[key + '_ridge'] = r; predictions[key + '_nonlinear'] = t
                boots[key] = ba
                rows.append({'method': method, 'seed': seed, 'partition': part, 'species': sp,
                             'target': target, 'n': len(idx), 'threshold_dev': median,
                             'auc': roc_auc_score(b, p), 'auc_low': lo, 'auc_high': hi,
                             'r2_linear': r2_score(y, r), 'r2_nonlinear': r2_score(y, t),
                             'rho_linear': spearmanr(y, r).statistic})
    pd.DataFrame(rows).to_csv(dest / 'probes.csv', index=False)
    np.savez_compressed(dest / 'probe_predictions.npz', **predictions)
    np.savez_compressed(dest / 'probe_bootstrap.npz', **boots)
    dev = df.partition == 'dev'; y = pd.Categorical(df.species, categories=SPECIES).codes
    lr = LogisticRegression(C=1, max_iter=1500, class_weight='balanced').fit(z[dev], y[dev])
    result = []
    for part in ['test', 'run_test']:
        m = df.partition == part
        result.append({'method': method, 'seed': seed, 'partition': part,
                       'tag_accuracy': accuracy_score(y[m], lr.predict(z[m]))})
    pd.DataFrame(result).to_csv(dest / 'tag_probe.csv', index=False)


def mass_contrast(score, df, part, fraction, picky, weights=None):
    """Within momentum strata, compare equal-fraction high scores to remaining rows.

    Bin edges fixed on dev; scores ranked without inspecting mass. Return standardized
    mean mass difference, weighted by the evaluation population's stratum proportions.
    Bootstrap weights keep the selected identities fixed (conditional uncertainty).
    """
    dev = (df.partition == 'dev') & (df.species == 'kaon')
    good = (df.partition == part) & (df.species == 'kaon')
    if picky is not None:
        dev &= df.picky == picky; good &= df.picky == picky
    devmom = df.loc[dev & np.isfinite(df.momentum), 'momentum']
    if len(devmom) < 20: return None
    bounds = np.unique(np.quantile(devmom, [0, .25, .5, .75, 1]))
    good &= np.isfinite(df.momentum) & np.isfinite(df.beamline_mass) & np.isfinite(score)
    good &= df.momentum.between(bounds[0], bounds[-1])
    idx = np.flatnonzero(good)
    if len(idx) < 30 or np.ptp(score[idx]) < 1e-10: return None
    bins = np.digitize(df.momentum.to_numpy()[idx], bounds[1:-1])
    mass = df.beamline_mass.to_numpy()[idx]
    chosen = np.zeros(len(idx), dtype=bool)
    for b in np.unique(bins):
        loc = np.flatnonzero(bins == b)
        n = max(1, int(round(fraction * len(loc))))
        chosen[loc[np.argsort(score[idx][loc], kind='stable')[-n:]]] = True
    if weights is None: weights = np.ones(len(idx))
    delta = 0.
    for b in np.unique(bins):
        a, c = (bins == b) & chosen, (bins == b) & ~chosen
        if weights[a].sum() == 0 or weights[c].sum() == 0: return None
        delta += weights[bins == b].sum() / weights.sum() * (np.average(mass[a], weights=weights[a]) - np.average(mass[c], weights=weights[c]))
    return {'shift': float(delta), 'idx': idx, 'selected': chosen, 'bins': bins,
            'n': len(idx), 'n_flagged': int(chosen.sum())}


def mass_rows(score, df, method, seed, k, gm_seed):
    rows, arrays = [], {}
    for part in ['test', 'run_test']:
        for quality in [None, 1]:
            for frac in [.1, .2, .3, .5]:
                result = mass_contrast(score, df, part, frac, quality)
                if result is None: continue
                idx, chosen, bins = result['idx'], result['selected'], result['bins']
                mass = df.beamline_mass.to_numpy()[idx]
                samples = []
                # Selection conditional bootstrap: avoids retuning at every resample.
                for w in boot_weights(df.event_key.to_numpy()[idx]):
                    delta = 0.
                    for b in np.unique(bins):
                        a, c = (bins == b) & chosen, (bins == b) & ~chosen
                        if not w[a].sum() or not w[c].sum(): delta = np.nan; break
                        delta += w[bins == b].sum() / w.sum() * (np.average(mass[a], weights=w[a]) - np.average(mass[c], weights=w[c]))
                    samples.append(delta)
                lo, hi = interval(samples)
                rows.append({'method': method, 'seed': seed, 'gmm_seed': gm_seed, 'k': k,
                             'partition': part, 'quality': 'all' if quality is None else 'picky',
                             'fraction': frac, 'n': result['n'], 'n_flagged': result['n_flagged'],
                             'mass_shift': result['shift'], 'low': lo, 'high': hi})
                arrays[f'{part}_{quality}_{frac}'] = samples
    return rows, arrays


def cluster(z, df, method, seed, dest):
    tr = df.partition == 'train'; dev = df.partition == 'dev'
    y = pd.Categorical(df.species, categories=SPECIES).codes
    rows, cover, mass, saved = [], [], [], {}
    for k in KS:
        for gs in [0, 1, 2]:
            g = GaussianMixture(k, covariance_type='full', reg_covar=1e-4,
                                n_init=5, max_iter=200, random_state=gs).fit(z[tr])
            prob = g.predict_proba(z); lab = prob.argmax(1)
            counts = np.array([np.bincount(y[dev & (lab == c)], minlength=3) for c in range(k)])
            known = counts.sum(1) > 0
            mapping = counts.argmax(1); mapping[~known] = -1
            cpurity = counts.max(1) / np.maximum(counts.sum(1), 1)
            pred = mapping[lab]
            score = prob[:, mapping == 0].sum(1)
            saved[f'k{k}_s{gs}_proton_score'] = score
            for part in ['test', 'run_test']:
                m = (df.partition == part).to_numpy()
                row = {'method': method, 'seed': seed, 'k': k, 'gmm_seed': gs,
                       'partition': part, 'n': int(m.sum()), 'ari': adjusted_rand_score(y[m], lab[m]),
                       'mapped_tag_agreement': np.mean(pred[m] == y[m]),
                       'converged': bool(g.converged_), 'iterations': int(g.n_iter_)}
                # Keep conditional event-bootstrap uncertainty for the principal k=3 fit.
                if k == 3 and gs == 0:
                    boot = [np.average(pred[m] == y[m], weights=w) for w in boot_weights(df.event_key.to_numpy()[m])]
                    row['agreement_low'], row['agreement_high'] = interval(boot)
                rows.append(row)
                for threshold in [.0, .5, .6, .7, .8, .85, .9, .95]:
                    sel = m & known[lab] & (cpurity[lab] >= threshold)
                    cover.append({'method': method, 'seed': seed, 'k': k, 'gmm_seed': gs,
                                  'partition': part, 'dev_purity_threshold': threshold,
                                  'coverage': float(sel.sum() / m.sum()),
                                  'tag_agreement': np.mean(pred[sel] == y[sel]) if sel.any() else np.nan})
            # Main mass comparison k=15; k=3,37 are fixed sensitivity checks.
            if k in [3, 15, 37] and gs == 0:
                rr, ba = mass_rows(score, df, method, seed, k, gs)
                mass.extend(rr); np.savez_compressed(dest / f'mass_bootstrap_k{k}.npz', **ba)
            print('cluster', method, seed, k, gs, flush=True)
    pd.DataFrame(rows).to_csv(dest / 'clustering.csv', index=False)
    pd.DataFrame(cover).to_csv(dest / 'coverage.csv', index=False)
    pd.DataFrame(mass).to_csv(dest / 'mass.csv', index=False)
    np.savez_compressed(dest / 'cluster_scores.npz', **saved)


def mass_prediction(z, df, method, seed, dest):
    # External-observable readout fitted on dev only; it does not feed back into the GMM.
    dev = (df.partition == 'dev') & (df.species == 'kaon')
    valid = np.isfinite(df.momentum) & np.isfinite(df.beamline_mass) & np.isfinite(df.picky)
    dev &= valid
    idx = np.flatnonzero(dev)
    cov = np.c_[np.log(np.maximum(df.momentum.to_numpy(), 1)), df.picky.to_numpy()]
    endpoint = np.load(OUT / 'representations/endpoint.npy')
    features = {'covariates': cov, 'covariates_endpoint': np.c_[cov, endpoint],
                'covariates_representation': np.c_[cov, z],
                'covariates_endpoint_representation': np.c_[cov, endpoint, z]}
    rows, arrays, boots = [], {}, {}
    for name, x in features.items():
        scaler = StandardScaler().fit(x[idx]); a = scaler.transform(x)
        for reader in ['ridge', 'nonlinear']:
            model = Ridge(alpha=10) if reader == 'ridge' else nonlinear()
            model.fit(a[idx], df.beamline_mass.to_numpy()[idx])
            for part in ['test', 'run_test']:
                for quality in ['all', 'picky']:
                    mask = (df.partition == part) & (df.species == 'kaon') & valid
                    if quality == 'picky': mask &= df.picky == 1
                    ii = np.flatnonzero(mask)
                    y = df.beamline_mass.to_numpy()[ii]; p = model.predict(a[ii])
                    ba = [r2_score(y, p, sample_weight=w) for w in boot_weights(df.event_key.to_numpy()[ii])]
                    lo, hi = interval(ba)
                    rows.append({'method': method, 'seed': seed, 'features': name, 'reader': reader,
                                 'partition': part, 'quality': quality, 'n': len(ii),
                                 'r2': r2_score(y, p), 'low': lo, 'high': hi})
                    key = f'{name}_{reader}_{part}_{quality}'
                    arrays[key] = p; boots[key] = ba
    pd.DataFrame(rows).to_csv(dest / 'mass_prediction.csv', index=False)
    np.savez_compressed(dest / 'mass_prediction_bootstrap.npz', **boots)


def evaluate(path, stages):
    df = pd.read_csv(OUT / 'manifest.csv')
    info = json.loads((OUT / 'protocol.json').read_text())
    assert digest(OUT / 'manifest.csv') == info['manifest_sha256']
    method, seed = repr_info(path.stem)
    dest = OUT / 'evaluation' / path.stem; dest.mkdir(parents=True, exist_ok=True)
    representation_hash = digest(path)
    if (dest / 'metadata.json').exists():
        previous = json.loads((dest / 'metadata.json').read_text())
        if previous['representation_sha256'] != representation_hash or previous['manifest_sha256'] != info['manifest_sha256']:
            raise RuntimeError(f'Stale evaluation cache for {path.stem}; archive it before rerunning.')
    z = np.load(path)
    assert len(z) == len(df) and np.isfinite(z).all()
    scaler = StandardScaler().fit(z[df.partition == 'train']); z = scaler.transform(z)
    if 'probes' in stages and not (dest / 'probes.csv').exists(): probes(z, df, method, seed, dest)
    if 'cluster' in stages and not (dest / 'clustering.csv').exists(): cluster(z, df, method, seed, dest)
    if 'mass_prediction' in stages and not (dest / 'mass_prediction.csv').exists(): mass_prediction(z, df, method, seed, dest)
    write_json(dest / 'metadata.json', {'representation_sha256': representation_hash, 'manifest_sha256': info['manifest_sha256'],
                                       'dimensions': z.shape[1], 'stages_requested': stages})
    print('EVALUATED', path.stem, stages, flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--representations', nargs='*')
    ap.add_argument('--stages', nargs='+', default=['probes', 'cluster', 'mass_prediction'])
    args = ap.parse_args()
    for p in sorted((OUT / 'representations').glob('*.npy')):
        if '_reconstruction' in p.stem: continue
        if args.representations and p.stem not in args.representations: continue
        stages = args.stages if '_n100' not in p.stem else [s for s in args.stages if s == 'probes']
        if stages: evaluate(p, stages)


if __name__ == '__main__': main()
