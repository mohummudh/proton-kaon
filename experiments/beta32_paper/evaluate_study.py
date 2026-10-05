#!/usr/bin/env python3
"""Measure the attached paper's claims using its original analysis functions."""
import argparse
from pathlib import Path
import sys
import json
import shutil
from types import SimpleNamespace
import itertools

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/extra'), str(HERE)]
import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from run_study import write_json, digest
from _beam_data import load_beam_data, SPECIES
from src.train.naming import model_name
from _sweep_measure import active_dims
from cluster_k_sweep import scan
from cluster_latents import fit_clusters, score_clusters
from scripts.analyse_latents import make_feature_auc_probe
from plot_proxy_auc import bootstrap_auc

def freeze_reference():
    """Freeze alignment and historical results, without changing old artifacts."""
    data = HERE / 'data'; data.mkdir(exist_ok=True)
    reference = HERE / 'reference'; reference.mkdir(exist_ok=True)
    cfg = yaml.safe_load(next((ROOT / 'configs').glob('run_0093*')).read_text())
    z, df = load_beam_data(cfg)
    inf = Path(cfg['output']['inference_dir']) / model_name(cfg)
    ss = np.load(inf / 'species_split.npz')
    order = np.r_[ss['p_train_idx'], ss['p_val_idx'], np.arange(8227)+10466,
                  np.arange(8964)+10466+8227]
    assert len(order) == len(df) == 27657 and len(np.unique(order)) == len(order)
    raw_df = df.assign(raw_tensor_row=order).sort_values('raw_tensor_row').reset_index(drop=True)
    assert np.array_equal(raw_df.raw_tensor_row, np.arange(27657))
    raw_df.to_csv(data / 'metadata.csv', index=False)
    # Keep original published inference, rather than re-encoding historical weights.
    np.savez_compressed(data / 'historical_latents.npz', mu=z, raw_order=order)
    cache = HERE / 'cache' / 'historical'; cache.mkdir(parents=True, exist_ok=True)
    for p in inf.glob('gmm_labels_k*_seed*.npy'): shutil.copy2(p, cache / p.name)
    log = Path(cfg['output']['dir']) / (model_name(cfg)+'.json')
    shutil.copy2(log, reference / 'historical_training.json')
    for src, target in [(ROOT / 'figs/split_sweep/split_sweep_runs.csv', 'training_size_beta05.csv'),
                        (ROOT / 'figs/latent_sweep/latent_sweep_runs.csv', 'latent_capacity_beta05.csv')]:
        shutil.copy2(src, reference / target)
    two = ROOT / 'figs' / model_name(cfg) / 'two-sample/paper/results.json'
    if two.exists(): shutil.copy2(two, reference / 'historical_two_sample.json')
    write_json(reference / 'sources.json', {'metadata_source': str(inf),
        'metadata_sha256': digest(data / 'metadata.csv'), 'historical_latents_sha256': digest(data / 'historical_latents.npz'),
        'historical_training_source': str(log), 'historical_training_sha256': digest(log),
        'reference_tables': {p.name: digest(p) for p in reference.glob('*.csv')}})

def load(stem):
    raw_df = pd.read_csv(HERE / 'data/metadata.csv')
    if stem == 'historical':
        a = np.load(HERE / 'data/historical_latents.npz')
        return a['mu'], raw_df.iloc[a['raw_order']].reset_index(drop=True), 'bal9419', .5
    task = next(t for t in json.loads((HERE / 'plan.json').read_text()) if t['id'] == stem)
    a = np.load(HERE / 'runs' / stem / 'latents.npz')
    s = np.load(HERE / 'splits' / f"split_all_{task['tag']}.npz")
    ptrain, pval = np.sort(s['train_idx'][s['train_idx'] < 10466]), np.sort(s['val_idx'][s['val_idx'] < 10466])
    order = np.r_[ptrain, pval, np.arange(8227)+10466, np.arange(8964)+18693]
    return a['mu'][order].astype(float), raw_df.iloc[order].reset_index(drop=True), task['tag'], task['beta']

def val_indices(df, tag):
    split = np.load(HERE / 'splits' / f'split_all_{tag}.npz')
    return {s: np.flatnonzero((df.species == s) & df.raw_tensor_row.isin(split['val_idx'])) for s in SPECIES}

def destination(stem):
    d = HERE / 'results' / stem; d.mkdir(parents=True, exist_ok=True)
    return d

def proxies(stem, intervals=False):
    z, df, tag, beta = load(stem); dest = destination(stem)
    path = dest / 'probes.csv'
    if path.exists():
        table = pd.read_csv(path)
        if not intervals or 'lo' in table.columns: return table
    probe = make_feature_auc_probe(return_scores=True)
    rows = []
    for feature in ['mean_adc', 'solidity']:
        for species, idx in val_indices(df, tag).items():
            auc, median, y, probability = probe(z[idx], df.iloc[idx], feature)
            row = {'model': stem, 'beta': beta, 'feature': feature, 'species': species,
                   'n': len(idx), 'auc': auc, 'median': median}
            if intervals:
                lo, hi, sd = bootstrap_auc(y, probability, 2000, seed=0)
                row.update(lo=lo, hi=hi, bootstrap_sd=sd)
            rows.append(row)
    table = pd.DataFrame(rows); table.to_csv(path, index=False)
    print('PROBES', stem, table[['species', 'feature', 'auc']].to_dict('records'), flush=True)
    return table

def cluster_summary(lab, df, k):
    truth = pd.Categorical(df.species, categories=SPECIES).codes
    counts = np.array([np.bincount(truth[lab == c], minlength=3) for c in range(k)])
    sizes = counts.sum(1); majority = counts.argmax(1)
    purity = counts.max(1) / np.maximum(sizes, 1)
    mapped = majority[lab]
    result = {'k': k, **score_clusters(lab, df.species, k),
        'n_clusters_80': int((purity >= .8).sum()),
        'fraction_in_80': float(sizes[purity >= .8].sum()/len(df)),
        'fraction_in_85': float(sizes[purity >= .85].sum()/len(df)),
        'smallest_cluster': int(sizes.min())}
    for i, sp in enumerate(SPECIES):
        chosen = mapped == i
        result[f'group_purity_{sp}'] = float((truth[chosen] == i).mean()) if chosen.any() else None
        result[f'best_cluster_purity_{sp}'] = float((counts[:, i]/np.maximum(sizes,1)).max())
    clean = counts[:, 1]/np.maximum(sizes,1) >= .9
    result.update(clean_kaon_clusters=int(clean.sum()), clean_kaon_count=int(counts[clean,1].sum()),
                  clean_kaon_group_purity=float(counts[clean,1].sum()/sizes[clean].sum()) if clean.any() else None)
    mass = df.beamline_mass.to_numpy(); kt = (truth == 1) & np.isfinite(mass)
    a, b = mass[kt & (mapped == 0)], mass[kt & (mapped == 1)]
    result.update(kaon_in_proton_group=int((kt & (mapped == 0)).sum()),
                  kaon_in_kaon_group=int((kt & (mapped == 1)).sum()),
                  flagged_kaon_fraction=float((kt & (mapped == 0)).sum()/(truth == 1).sum()),
                  mass_median_proton=float(np.median(a)) if len(a) else None,
                  mass_median_kaon=float(np.median(b)) if len(b) else None,
                  mass_shift=float(np.median(a)-np.median(b)) if len(a) and len(b) else None)
    correlations = []
    for i, sp in enumerate(SPECIES):
        fractions, medians = [], []
        for c in range(k):
            idx = (lab == c) & (truth == i) & np.isfinite(mass)
            if idx.sum() >= 50:
                fractions.append(counts[c,0]/sizes[c]); medians.append(np.median(mass[idx]))
        rho, p = spearmanr(fractions, medians) if len(medians) >= 3 else (np.nan,np.nan)
        correlations.append({'species': sp, 'n_clusters':len(medians), 'rho':rho, 'p_value':p})
    result['cluster_mass_correlations'] = correlations
    return result, counts, mapped

def gmm(stem, k, seed=0):
    z, df, tag, beta = load(stem); dest = destination(stem)
    cache = HERE / 'cache' / stem; cache.mkdir(parents=True, exist_ok=True)
    path = cache / f'gmm_labels_k{k}_seed{seed}.npy'
    if path.exists(): lab = np.load(path)
    else:
        lab, model = fit_clusters(z, k, seed=seed, n_init=20)
        np.save(path, lab)
        write_json(dest / f'gmm_fit_k{k}_seed{seed}.json', {'converged':model.converged_,
            'iterations':model.n_iter_, 'lower_bound':model.lower_bound_, 'n_init':20,
            'reg_covar':model.reg_covar, 'representation_rows':len(z),
            'note':'Raw posterior means; no standardization; fit on pooled train/validation as in paper.'})
    assert len(lab) == len(df)
    result, counts, mapped = cluster_summary(lab, df, k)
    write_json(dest / f'gmm_k{k}_seed{seed}.json', result)
    pd.DataFrame(counts, columns=SPECIES).assign(cluster=np.arange(k)).to_csv(dest/f'composition_k{k}_seed{seed}.csv',index=False)
    print('GMM', stem, k, seed, result['purity'], 'mass shift', result['mass_shift'], flush=True)
    return result

def contamination(stem):
    from anchored_clustering import anchored_fit
    from estimate_contamination import fit_density, injection_recovery, fit_contamination
    dest = destination(stem); path = dest / 'contamination.json'
    if path.exists(): return
    z, df, tag, beta = load(stem)
    # The historical scripts fit anchor and host densities on raw latent coordinates.
    labels, weights, _ = anchored_fit(z, df.species, seed=0)
    mass = df.beamline_mass.to_numpy(); kt = (df.species == 'kaon').to_numpy()
    vals = {s: mass[kt & (labels == i) & np.isfinite(mass)] for i,s in enumerate(SPECIES)}
    anchored = {'counts':{s:int((kt & (labels == i)).sum()) for i,s in enumerate(SPECIES)},
        'medians':{s:float(np.median(a)) if len(a) else None for s,a in vals.items()},
        'mass_shift':float(np.median(vals['proton'])-np.median(vals['kaon'])) if len(vals['proton']) and len(vals['kaon']) else None}
    np.save(HERE / 'cache' / stem / 'anchored_labels.npy', labels)
    p, m = z[df.species == 'proton'], z[df.species == 'muon']
    qp, qm = fit_density(p,6,0), fit_density(m,6,0)
    table, slope, intercept = injection_recovery(z[kt],p,qp,qm,[400,800,1600,2400,3200],0)
    table.to_csv(dest/'injection_recovery.csv',index=False)
    subsets = {}
    for name, quality in [('all',None),('picky',1),('non-picky',0)]:
        selected = kt.copy()
        if quality is not None: selected &= (df.picky == quality).to_numpy()
        n = int(selected.sum()); host = 6 if n > 3000 else 4
        raw = fit_contamination(z[selected],qp,qm,n_host_comp=host,seed=0)['f_proton']
        subsets[name] = {'n':n,'raw':raw,'corrected':raw/slope}
    write_json(path, {'anchored':anchored,'recovery_slope':slope,'recovery_intercept':intercept,
        'corrected_contamination':intercept/slope, 'subsamples':subsets,
        'note':'Tag-conditioned density calibration; not a truth-level contamination measurement.'})

def scan_metric(stem):
    dest = destination(stem); path = dest / 'scan_metrics.json'
    hist = json.loads((HERE / 'runs' / stem / 'history.json').read_text())
    pd.DataFrame(hist).to_csv(dest/'training_history.csv',index=False)
    shutil.copy2(HERE/'runs'/stem/'complete.json',dest/'training_provenance.json')
    if path.exists(): return
    z, df, tag, beta = load(stem)
    task = next(t for t in json.loads((HERE / 'plan.json').read_text()) if t['id'] == stem)
    # Reproduce the source's min_delta rule to identify the checkpoint epoch.
    best, chosen = float('inf'), None
    for row in hist:
        if row['val_loss'] < best - 1e-4: best, chosen = row['val_loss'], row
    split = np.load(HERE / 'splits' / f'split_all_{tag}.npz')
    result = {**task, 'n_train':len(split['train_idx']), 'n_val':len(split['val_idx']),
        'epochs':len(hist), 'best_epoch':chosen['epoch'], **active_dims(z),
        'val_recon':chosen['val_recon'], 'val_kl':chosen['val_kl'],
        'kl_fraction':beta*chosen['val_kl']/chosen['val_loss']}
    result.update({f"auc_{r.feature}_{r.species}":r.auc for r in proxies(stem).itertuples()})
    result.update({k:v for k,v in gmm(stem,3).items() if k in ['ari','purity']})
    a = np.load(HERE/'runs'/stem/'latents.npz')
    kl = .5 * (np.exp(a['logvar'])+a['mu']**2-1-a['logvar']).sum(1)
    result['mean_decode_val_reconstruction'] = float(a['mean_reconstruction'][split['val_idx']].mean())
    result['mean_decode_val_kl'] = float(kl[split['val_idx']].mean())
    common = np.load(HERE/'splits/split_all_pool8227_tr90.npz')['val_idx']
    if task['tag'].startswith('pool8227'):
        assert np.isin(common, split['val_idx']).all()
        result['common_val_weighted_mean_decode'] = float(a['mean_reconstruction'][common].mean())
    write_json(path,result)

def two_sample(stem):
    from scripts.latent_two_sample import analyse_pair, match_species_composition, COMPARISONS
    dest = destination(stem); path = dest / 'two_sample.json'
    if path.exists(): return
    z, df, tag, beta = load(stem); s = np.load(HERE/'splits'/f'split_all_{tag}.npz')
    per_species = {sp:(z[(df.species==sp)&df.raw_tensor_row.isin(s['train_idx'])],
                        z[(df.species==sp)&df.raw_tensor_row.isin(s['val_idx'])]) for sp in SPECIES}
    tr, va, fractions, taken = match_species_composition(per_species,np.random.default_rng(42))
    # The source's comparison ordering gives combined_matched its own RNG stream.
    position = COMPARISONS.index('combined_matched')
    args = SimpleNamespace(standardize='pooled',tests=['ks','energy','c2st'],n_perm=1999,
        energy_n=5000,energy_repeats=5,c2st_models=['mlp','logreg'],c2st_folds=5,
        c2st_perm=199,c2st_repeats=5,seed=42)
    result=analyse_pair('combined_matched',tr,va,args,42+1000*position)
    result.update(matched_to_train_fractions=fractions,val_taken_per_species=taken)
    write_json(path,result)

def stability(stem):
    dest=destination(stem); path=dest/'stability.csv'
    if path.exists(): return
    rows=[]
    for k in range(36,42):
        shifts=[]
        for seed in range(5):
            r=gmm(stem,k,seed); shifts.append(r['mass_shift'])
        pairwise=[adjusted_rand_score(np.load(HERE/'cache'/stem/f'gmm_labels_k{k}_seed{a}.npy'),
            np.load(HERE/'cache'/stem/f'gmm_labels_k{k}_seed{b}.npy')) for a,b in itertools.combinations(range(5),2)]
        rows.append({'k':k,'mean_mass_shift':np.mean(shifts),'sd_mass_shift':np.std(shifts,ddof=1),
                     'pairwise_ari_mean':np.mean(pairwise),'pairwise_ari_min':np.min(pairwise),
                     'pairwise_ari_max':np.max(pairwise)})
    pd.DataFrame(rows).to_csv(path,index=False)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage',choices=['freeze','primary','cluster_scan','stability','contamination','two_sample','scan'])
    ap.add_argument('--ids',nargs='+',default=['historical','bal9419_d8_s0_b32'])
    args=ap.parse_args()
    with threadpool_limits(limits=2):
        if args.stage=='freeze': freeze_reference(); return
        for stem in args.ids:
            if args.stage=='primary':
                proxies(stem,intervals=True)
                z,_,_,_=load(stem)
                write_json(destination(stem)/'latent_capacity.json',active_dims(z))
                for k in [3,8,12,15,37,50]: gmm(stem,k)
                if stem!='historical': scan_metric(stem)
            elif args.stage=='cluster_scan':
                for k in range(3,51): gmm(stem,k)
            elif args.stage=='scan': scan_metric(stem)
            elif args.stage=='stability': stability(stem)
            elif args.stage=='contamination': contamination(stem)
            elif args.stage=='two_sample': two_sample(stem)

if __name__=='__main__': main()
