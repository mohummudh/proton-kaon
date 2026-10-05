#!/usr/bin/env python3
"""Event-held-out truth probes, image controls and interaction-pair diagnostics."""

import argparse
import json
from pathlib import Path
import sys
import warnings

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (balanced_accuracy_score, confusion_matrix, f1_score,
    mean_absolute_error, r2_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import yaml

from experiments.pilarnet_lariat.latent_truth.prepare import BASE, partition
from scripts.extra._beam_data import load_beam_data
from src.train.naming import model_name

TARGETS = {
    'incoming_ke_mev': 'log', 'momentum_mev': 'log', 'deposited_mev': 'log',
    'geom_retained_energy_mev': 'log', 'geom_retained_electrons': 'log',
    'path_cm': 'log', 'mean_dedx_mev_cm': 'log', 'endpoint_dedx_mev_cm': 'log',
    'bragg_ratio': 'log', 'linearity_3d': 'linear', 'transverse_rms_cm': 'log',
    'proton_bb_density_ratio': 'log', 'proton_bb_log_distance': 'log',
    'proton_bb_best_rr_offset_cm': 'log',
    'extent_3d_cm': 'log', 'chord_over_path': 'linear', 'n_fragments': 'log',
    'shower_fraction': 'linear', 'visible_charge_fraction': 'linear',
    'geometric_energy_fraction': 'linear', 'assigned_xz_deg': 'linear', 'assigned_yz_deg': 'linear',
    'vertex_x_cm': 'linear', 'vertex_y_cm': 'linear', 'vertex_z_cm': 'linear',
    'source_direction_x': 'linear', 'source_direction_y': 'linear', 'source_direction_z': 'linear'}
TRACK_TARGETS = {'path_cm', 'endpoint_dedx_mev_cm', 'bragg_ratio', 'chord_over_path'}
CORE = {'incoming_ke_mev', 'deposited_mev', 'geom_retained_energy_mev',
        'path_cm', 'mean_dedx_mev_cm', 'linearity_3d', 'transverse_rms_cm',
        'endpoint_dedx_mev_cm', 'bragg_ratio', 'proton_bb_density_ratio'}


def cluster_interval(y, pred, groups, metric, draws=150, seed=9105):
    _, inv = np.unique(groups, return_inverse=True)
    rng, scores = np.random.default_rng(seed), []
    n = inv.max()+1
    for _ in range(draws):
        w = np.bincount(rng.integers(n, size=n), minlength=n)[inv]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                value = metric(y, pred, sample_weight=w)
            if np.isfinite(value):
                scores.append(value)
        except ValueError:
            pass
    return np.quantile(scores, [.025, .975]).tolist() if scores else [np.nan, np.nan]


def ridge_fit(x, y, train, dev, alphas=(.1, 1, 10, 100)):
    best = None
    for alpha in alphas:
        candidate = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        candidate.fit(x[train], y[train])
        score = mean_absolute_error(y[dev], candidate.predict(x[dev]))
        if best is None or score < best[0]:
            best = score, candidate, alpha
    return best[1], best[2]


def regression(output, frame, reps):
    rng = np.random.default_rng(9105)
    rows, predictions, correlations, unsupported = [], [], [], []
    # Within-species readouts are primary; pooled targets can use species shortcuts.
    for species in ['proton', 'pion', 'muon', 'electron', 'photon', 'pooled']:
        subset = np.ones(len(frame), bool) if species == 'pooled' else frame.species.eq(species).to_numpy()
        for target, scale in TARGETS.items():
            if target in TRACK_TARGETS and species in ('electron', 'photon', 'pooled'):
                continue
            y = pd.to_numeric(frame[target], errors='coerce').to_numpy()
            valid = subset & np.isfinite(y)
            if target in ('incoming_ke_mev', 'momentum_mev'):
                valid &= frame.kinematics_reliable.to_numpy()
            if target in TRACK_TARGETS:
                valid &= frame.time_order_unique_fraction.gt(.9).to_numpy()
            if scale == 'log':
                valid &= y >= 0
            ids = {p: np.flatnonzero(valid & frame.partition.eq(p).to_numpy()) for p in ('train', 'dev', 'test')}
            if min(map(len, ids.values())) < 25 or min(np.std(y[ids[p]]) for p in ('train', 'dev', 'test')) < 1e-8:
                unsupported.append({'species': species, 'target': target,
                                    **{p: len(ii) for p, ii in ids.items()}})
                continue
            yy = np.log1p(np.maximum(y, 0)) if scale == 'log' else y
            test = ids['test']; groups = frame.iloc[test].event_key.to_numpy()
            methods = ['paper_vae', 'input_pca8', 'random_s0']
            if target in CORE:
                methods += ['proton_vae', 'vae_s0', 'vae_s1', 'vae_s2', 'ae_s0']
                if species != 'pooled':
                    methods.append('pixels')
            for method in methods:
                learners = ['ridge', 'trees'] if method in ('paper_vae', 'input_pca8') else ['ridge']
                x = reps[method]
                for learner in learners:
                    if learner == 'ridge':
                        model, alpha = ridge_fit(x, yy, ids['train'], ids['dev'])
                    else:
                        model = ExtraTreesRegressor(n_estimators=128, max_depth=10,
                            min_samples_leaf=5, random_state=9105, n_jobs=2).fit(x[ids['train']], yy[ids['train']])
                        alpha = np.nan
                    pred = model.predict(x[test])
                    r2 = r2_score(yy[test], pred)
                    low, high = cluster_interval(yy[test], pred, groups, r2_score)
                    physical = np.expm1(np.clip(pred, -20, 20)) if scale == 'log' else pred
                    rows.append({'species': species, 'target': target, 'scale': scale,
                        'representation': method, 'probe': learner, 'n_train': len(ids['train']),
                        'n_dev': len(ids['dev']), 'n_test': len(test), 'alpha': alpha,
                        'r2': r2, 'r2_low': low, 'r2_high': high,
                        'physical_r2': r2_score(y[test], physical),
                        'physical_mae': mean_absolute_error(y[test], physical),
                        'spearman': float(spearmanr(yy[test], pred).statistic) if np.std(pred) > 1e-10 else np.nan})
                    if method == 'paper_vae':
                        predictions.extend({'row': int(row), 'species': species, 'target': target,
                            'probe': learner, 'true': float(y[row]), 'predicted': float(value)}
                            for row, value in zip(test, physical))
                    if method == 'paper_vae' and learner == 'ridge':
                        for dim in range(x.shape[1]):
                            correlations.append({'species': species, 'target': target, 'dimension': dim,
                                'heldout_spearman': float(spearmanr(x[test, dim], yy[test]).statistic)})
                        if target in CORE:
                            null = yy.copy(); null[ids['train']] = rng.permutation(yy[ids['train']])
                            null_model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(x[ids['train']], null[ids['train']])
                            rows.append({'species': species, 'target': target, 'scale': scale,
                                'representation': 'paper_vae', 'probe': 'permuted_train',
                                'n_test': len(test), 'r2': r2_score(yy[test], null_model.predict(x[test]))})
            print('Truth readouts', species, target, flush=True)
    pd.DataFrame(rows).to_csv(output/'truth_readouts.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'truth_predictions.csv', index=False)
    pd.DataFrame(correlations).to_csv(output/'latent_truth_correlations.csv', index=False)
    (output/'unsupported_readouts.json').write_text(json.dumps(unsupported, indent=2))


def classifier(x, y, train, dev, nonlinear=False):
    if nonlinear:
        return ExtraTreesClassifier(n_estimators=256, min_samples_leaf=3,
            class_weight='balanced', random_state=9105, n_jobs=2).fit(x[train], y[train])
    best = None
    for c in (.01, .1, 1, 10):
        model = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=1500, class_weight='balanced'))
        model.fit(x[train], y[train])
        score = balanced_accuracy_score(y[dev], model.predict(x[dev]))
        if best is None or score > best[0]:
            best = score, model
    return best[1]


def classifications(output, frame, reps):
    ids = {p: np.flatnonzero(frame.partition.eq(p).to_numpy()) for p in ('train', 'dev', 'test')}
    results, predictions = [], []
    for task, y in [('species', frame.pid.to_numpy()), ('semantic', frame.semantic.to_numpy())]:
        support = {int(k): int(v) for k, v in pd.Series(y[ids['train']]).value_counts().items()}
        allowed = {k for k, n in support.items() if n >= 20 and (y[ids['test']] == k).sum() >= 10}
        ii = {p: a[np.isin(y[a], list(allowed))] for p, a in ids.items()}
        if len(allowed) < 2:
            continue
        for method in reps:
            for nonlinear in ([False, True] if method in ('paper_vae', 'input_pca8') else [False]):
                model = classifier(reps[method], y, ii['train'], ii['dev'], nonlinear)
                test = ii['test']; pred = model.predict(reps[method][test]); prob = model.predict_proba(reps[method][test])
                lo, hi = cluster_interval(y[test], pred, frame.iloc[test].event_key.to_numpy(), balanced_accuracy_score)
                results.append({'task': task, 'representation': method, 'probe': 'trees' if nonlinear else 'logistic',
                    'n_test': len(test), 'classes': str(sorted(allowed)),
                    'balanced_accuracy': balanced_accuracy_score(y[test], pred), 'low': lo, 'high': hi,
                    'macro_f1': f1_score(y[test], pred, average='macro'),
                    'confusion': json.dumps(confusion_matrix(y[test], pred, labels=sorted(allowed)).tolist())})
                if method in ('paper_vae', 'input_pca8') and not nonlinear:
                    predictions.extend({'task': task, 'row': int(row), 'true': int(yt), 'predicted': int(yp),
                                        'representation': method,
                                        'confidence': float(p.max())} for row, yt, yp, p in zip(test, y[test], pred, prob))
                    if task == 'species':
                        # Common retained-energy bands limit an obvious energy shortcut.
                        edges = np.unique(np.quantile(frame.iloc[ii['train']].geom_retained_energy_mev, [0, .2, .4, .6, .8, 1]))
                        band = np.digitize(frame.geom_retained_energy_mev.to_numpy(), edges[1:-1])
                        matched = []
                        rng = np.random.default_rng(9105)
                        for b in np.unique(band[test]):
                            pools = [test[(band[test] == b) & (y[test] == k)] for k in sorted(allowed)]
                            n = min(map(len, pools))
                            if n >= 3:
                                matched.extend(np.concatenate([rng.choice(pool, n, replace=False) for pool in pools]))
                        if len(matched) >= 30:
                            matched = np.asarray(matched, int)
                            mpred = model.predict(reps[method][matched])
                            results.append({'task': 'species_energy_balanced_test', 'representation': method,
                                'probe': 'logistic', 'n_test': len(matched),
                                'balanced_accuracy': balanced_accuracy_score(y[matched], mpred),
                                'macro_f1': f1_score(y[matched], mpred, average='macro'),
                                'note': 'Training-energy quantiles; equal class counts within supported test bands. Geometric retained-energy proxy, not exact waveform ancestry.'})
                        else:
                            (output/'energy_balanced_species_support.json').write_text(json.dumps({
                                'supported': False, 'matched_test_particles': len(matched),
                                'reason': 'No adequate five-species common retained-energy support; unbalanced species probes can use energy shortcuts.'}, indent=2))
                        null_y = y.copy(); null_y[ii['train']] = rng.permutation(y[ii['train']])
                        null = classifier(reps[method], null_y, ii['train'], ii['dev'])
                        results.append({'task': task, 'representation': method, 'probe': 'permuted_train',
                            'n_test': len(test), 'balanced_accuracy': balanced_accuracy_score(y[test], null.predict(reps[method][test]))})
        print('Classification', task, 'supported labels', sorted(allowed), flush=True)
    pd.DataFrame(results).to_csv(output/'classification.csv', index=False)
    pd.DataFrame(predictions).to_csv(output/'classification_predictions.csv', index=False)


def interaction_pairs(output, frame, reps):
    rng = np.random.default_rng(9105)
    rows = []
    for event, f in frame[frame.interaction_id.ge(0)].groupby('event_key'):
        indices = f.index.to_numpy()
        pairs = [(a, b) for i, a in enumerate(indices) for b in indices[i+1:]]
        if len(pairs) > 40:
            pairs = [pairs[i] for i in rng.choice(len(pairs), 40, replace=False)]
        for a, b in pairs:
            rows.append({'event_key': event, 'partition': frame.iloc[a].partition,
                'left': int(a), 'right': int(b),
                'same_interaction': int(frame.iloc[a].interaction_id == frame.iloc[b].interaction_id)})
    pairs = pd.DataFrame(rows)
    if pairs.empty:
        return
    pairs.to_csv(output/'interaction_pairs.csv', index=False)
    a, b = pairs.left.to_numpy(), pairs.right.to_numpy()
    y = pairs.same_interaction.to_numpy()
    ids = {p: np.flatnonzero(pairs.partition.eq(p).to_numpy()) for p in ('train', 'dev', 'test')}
    if any(min(np.bincount(y[ii], minlength=2)) < 15 for ii in ids.values()):
        (output/'interaction_pair_report.json').write_text(json.dumps({'supported': False,
            'reason': 'insufficient same/different interaction pairs in an event partition'}))
        return
    results = []
    for method in ['paper_vae', 'input_pca8']:
        z = reps[method]
        # Concatenate the two existing representations; no hand-built pair descriptors.
        x = np.c_[z[a], z[b]]
        model = classifier(x, y, ids['train'], ids['dev'], nonlinear=True)
        score = model.predict_proba(x[ids['test']])[:, 1]
        test = ids['test']
        low, high = cluster_interval(y[test], score, pairs.iloc[test].event_key.to_numpy(), roc_auc_score)
        results.append({'representation': method, 'n_test_pairs': len(test),
            'n_test_events': pairs.iloc[test].event_key.nunique(),
            'auc': roc_auc_score(y[test], score), 'low': low, 'high': high,
            'balanced_accuracy': balanced_accuracy_score(y[test], model.predict(x[test]))})
    pd.DataFrame(results).to_csv(output/'interaction_readout.csv', index=False)


def real_reference(output, frame, reps):
    cfg = yaml.safe_load(next((ROOT/'configs').glob('run_0093*')).read_text())
    real_z, real = load_beam_data(cfg)
    inf = Path(cfg['output']['inference_dir'])/model_name(cfg)
    ss = np.load(inf/'species_split.npz')
    np_, nk, nm = [int(real.species.eq(s).sum()) for s in ('proton', 'kaon', 'muon')]
    real_train = np.zeros(len(real), bool)
    real_train[:len(ss['p_train_idx'])] = True
    real_train[np_+ss['k_train_idx']] = True
    real_train[np_+nk+ss['m_train_idx']] = True
    scale = StandardScaler().fit(real_z[real_train])
    embedding = PCA(n_components=2).fit(scale.transform(real_z[real_train]))
    xy_sim = embedding.transform(scale.transform(reps['paper_vae']))
    xy_real = embedding.transform(scale.transform(real_z))
    np.savez(output/'real_reference.npz', real_z=real_z, real_xy=xy_real, simulation_xy=xy_sim,
        real_train=real_train, real_species=real.species.to_numpy(dtype='U10'),
        pca_variance=embedding.explained_variance_ratio_)
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight='balanced'))
    model.fit(real_z[real_train], real.species.to_numpy()[real_train])
    assignments = model.predict(reps['paper_vae'])
    allocation = pd.crosstab(frame.species, pd.Series(assignments, name='real_reference_assignment'), normalize='index')
    allocation.to_csv(output/'real_reference_assignments.csv')
    # Domain classification uses real encoder-held-out images and fixed simulation
    # event partitions. These classes are not matched in beam/TPC energy.
    domain = []
    for species in ('proton', 'muon'):
        rf = real[real.species.eq(species) & ~real_train].copy()
        rf['event_key'] = 'real:'+rf[['run', 'subrun', 'event']].astype(str).agg(':'.join, axis=1)
        rf['partition'] = rf.event_key.map(partition)
        sf = frame[frame.species.eq(species)]
        rz, sz = real_z[rf.index], reps['paper_vae'][sf.index]
        x = np.vstack([rz, sz]); y = np.r_[np.zeros(len(rz), int), np.ones(len(sz), int)]
        part = np.r_[rf.partition.to_numpy(), sf.partition.to_numpy()]
        groups = np.r_[rf.event_key.to_numpy(), sf.event_key.to_numpy()]
        ids = {p: np.flatnonzero(part == p) for p in ('train', 'dev', 'test')}
        fit = classifier(x, y, ids['train'], ids['dev'])
        test = ids['test']; score = fit.predict_proba(x[test])[:, 1]
        lo, hi = cluster_interval(y[test], score, groups[test], roc_auc_score)
        domain.append({'species': species, 'auc_real_vs_simulation': roc_auc_score(y[test], score),
            'low': lo, 'high': hi, 'n_real_test': int((y[test] == 0).sum()),
            'n_simulation_test': int((y[test] == 1).sum()),
            'note': 'Unmatched distributions; LArIAT MIP reference is not pure muon truth. No cross-domain task-transfer claim.'})
    pd.DataFrame(domain).to_csv(output/'domain_separation.csv', index=False)
    # Restrict a second proton domain check to common incoming-energy support.
    # This does not equate TPC energy: real labels are upstream beamline KE.
    momentum = pd.read_csv('/Volumes/easystore/proton-deuteron/momentum_tof.csv')
    keys = ['run', 'subrun', 'event']
    if momentum.duplicated(keys).any():
        raise ValueError('Duplicate beamline momentum event keys')
    rp = real[real.species.eq('proton') & ~real_train].copy()
    rp['latent_row'] = rp.index
    rp = rp.merge(momentum[keys+['momentum']], on=keys, how='left', validate='many_to_one')
    rp['incoming_ke_mev'] = np.hypot(rp.momentum, 938.272)-938.272
    rp['event_key'] = 'real:'+rp[keys].astype(str).agg(':'.join, axis=1)
    rp['partition'] = rp.event_key.map(partition)
    sp = frame[frame.species.eq('proton') & frame.kinematics_reliable].copy()
    x = np.vstack([real_z[rp.latent_row], reps['paper_vae'][sp.index]])
    y = np.r_[np.zeros(len(rp), int), np.ones(len(sp), int)]
    part = np.r_[rp.partition.to_numpy(), sp.partition.to_numpy()]
    energy = np.r_[rp.incoming_ke_mev.to_numpy(), sp.incoming_ke_mev.to_numpy()]
    group = np.r_[rp.event_key.to_numpy(), sp.event_key.to_numpy()]
    bands = np.floor(energy/20)
    rng, matched = np.random.default_rng(9105), {}
    for p in ('train', 'dev', 'test'):
        selected = []
        for band in np.unique(bands[np.isfinite(bands)]):
            pools = [np.flatnonzero((part == p) & (bands == band) & (y == k)) for k in (0, 1)]
            n = min(map(len, pools))
            if n >= 3:
                selected.extend(np.concatenate([rng.choice(pool, n, replace=False) for pool in pools]))
        matched[p] = np.asarray(selected, int)
    if min(map(len, matched.values())) >= 30:
        model = classifier(x, y, matched['train'], matched['dev'])
        ii = matched['test']; score = model.predict_proba(x[ii])[:, 1]
        lo, hi = cluster_interval(y[ii], score, group[ii], roc_auc_score)
        pd.DataFrame([{'auc': roc_auc_score(y[ii], score), 'low': lo, 'high': hi,
            'n_train': len(matched['train']), 'n_dev': len(matched['dev']), 'n_test': len(ii),
            'energy_bin_width_mev': 20,
            'note': 'Equal real/MC counts within 20 MeV incoming-KE bands in each event partition; upstream real beam energy differs from TPC energy. Selection, shape and response remain unmatched.'}]).to_csv(output/'energy_matched_proton_domain.csv', index=False)
    else:
        (output/'energy_matched_proton_domain_unsupported.json').write_text(json.dumps(
            {p: len(v) for p, v in matched.items()}, indent=2))


def evaluate(output):
    frame = pd.read_csv(output/'manifest.csv')
    with np.load(output/'representations.npz') as saved:
        reps = {k: saved[k] for k in ('paper_vae', 'input_pca8', 'proton_vae',
                'vae_s0', 'vae_s1', 'vae_s2', 'ae_s0', 'random_s0')}
    reps['pixels'] = np.log1p(np.load(output/'raw.npy')).reshape(len(frame), -1)
    classifications(output, frame, reps)
    interaction_pairs(output, frame, reps)
    real_reference(output, frame, reps)
    regression(output, frame, reps)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_pixels_v2')
    args = p.parse_args()
    evaluate(args.output)
