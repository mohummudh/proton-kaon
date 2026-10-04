#!/usr/bin/env python3
"""Matched-input baseline experiment. Run from repo root; writes only new artifacts.

prepare: immutable row manifest, event/run partitions and endpoint features.
train: paired initializations, shared optimization budget, AE/VAE/masked AE.
See docs/BASELINE_PROTOCOL.md for estimands and scope.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from skimage.measure import label, regionprops
from _beam_data import load_beam_data, SPECIES
from src.models.build import build_vae
from src.train.naming import model_name

OUT = Path(os.environ.get('REPRESENTATION_BASELINES_OUT', ROOT / 'output/representation_baselines'))
INPUT_SCALE = os.environ.get('REPRESENTATION_INPUT_SCALE', 'log1p')
if INPUT_SCALE not in {'log1p', 'raw'}:
    raise ValueError(f'Unknown representation input scale: {INPUT_SCALE}')
CONFIG = next((ROOT / 'configs').glob('run_0093*'))
KEYS = ['run', 'subrun', 'event']
FEATURE_NAMES = ['mean', 'positive_median', 'positive_std', 'maximum', 'occupancy',
                 'solidity', 'x_centroid', 'y_centroid', 'x_spread', 'y_spread',
                 'profile_peak_position', 'profile_peak_mean', 'profile_cv',
                 'profile_slope', 'first_quarter_fraction', 'last_quarter_fraction']


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, default=str))
    tmp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def endpoint_features(images):
    """Fixed summaries of both supplied 48x48 planes, on raw ADC scale.

    No threshold/feature selection uses labels or held-out measurements.
    Axes are array axes, not a fitted track coordinate system.
    """
    coords = np.linspace(0, 1, 48)
    out = []
    for i, planes in enumerate(images):
        row = []
        for a in planes:
            pos = a[a > 0]
            total = a.sum()
            regions = regionprops(label(a > 0))
            sol = max(regions, key=lambda r: r.area).solidity if regions else 0.
            x, y = a.sum(axis=0), a.sum(axis=1)
            mx, my = (x * coords).sum() / max(total, 1e-12), (y * coords).sum() / max(total, 1e-12)
            profile = a.max(axis=1)  # one maximum per wire row, matching src.images.cut_start
            ps = max(profile.sum(), 1e-12)
            row.extend([a.mean(), np.median(pos) if pos.size else 0., pos.std() if pos.size else 0.,
                        a.max(), np.mean(a > 0), sol, mx, my,
                        np.sqrt((x * (coords - mx)**2).sum() / max(total, 1e-12)),
                        np.sqrt((y * (coords - my)**2).sum() / max(total, 1e-12)),
                        profile.argmax() / 47, profile.max() / max(profile.mean(), 1e-12),
                        profile.std() / max(profile.mean(), 1e-12),
                        np.polyfit(coords, profile, 1)[0], profile[:12].sum() / ps, profile[-12:].sum() / ps])
        out.append(row)
        if i % 5000 == 0:
            print('endpoint features', i, flush=True)
    return np.asarray(out, dtype=np.float32)


def prepare():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'manifest.csv').exists():
        raise RuntimeError('Manifest exists; do not silently redraw the experiment.')
    cfg = yaml.safe_load(CONFIG.read_text())
    _, df = load_beam_data(cfg)
    split = np.load(Path(cfg['output']['inference_dir']) / model_name(cfg) / 'species_split.npz')
    p_order = np.r_[split['p_train_idx'], split['p_val_idx']]
    data = torch.load(cfg['data']['path'], map_location='cpu', weights_only=False)
    raw = torch.cat([data['p'][p_order], data['k'], data['m']]).numpy()
    assert len(raw) == len(df) and raw.shape[1:] == (2, 48, 48)
    assert np.isfinite(raw).all() and raw.min() >= 0
    df['source_tensor_row'] = np.r_[p_order, np.arange(len(data['k'])), np.arange(len(data['m']))]
    df['aligned_row'] = np.arange(len(df))
    df['event_key'] = df[KEYS].astype(str).agg(':'.join, axis=1)
    df['image_sha256'] = [hashlib.sha256(a.tobytes()).hexdigest() for a in raw]
    df['source_tensor_key'] = df.species.map(dict(zip(SPECIES, ['p', 'k', 'm'])))
    conflict_events = df.groupby('event_key').species.nunique().loc[lambda x: x > 1].index
    # Conservatively exclude all copies of conflicting source tags. No inferred relabeling.
    df['partition'] = 'excluded_conflicting_tag'
    good = ~df.event_key.isin(conflict_events)
    unique = df[good].drop_duplicates('event_key')
    rng = np.random.default_rng(20260922)
    runs = np.sort(unique.run.unique())
    held_runs = rng.permutation(runs)[:max(1, round(.2 * len(runs)))]
    runheld = good & df.run.isin(held_runs)
    df.loc[runheld, 'partition'] = 'run_test'
    pool = unique[~unique.run.isin(held_runs)]
    train_keys, dev_keys, test_keys = [], [], []
    for sp in SPECIES:
        keys = rng.permutation(pool.loc[pool.species == sp, 'event_key'].to_numpy())
        sizes = df.groupby('event_key').size()
        n = int(np.searchsorted(np.cumsum([sizes[k] for k in keys]), 3139) + 1)
        train_keys.extend(keys[:n])
        remain = keys[n:]
        nd = max(1, round(.25 * len(remain)))
        dev_keys.extend(remain[:nd]); test_keys.extend(remain[nd:])
    for name, keys in [('train', train_keys), ('dev', dev_keys), ('test', test_keys)]:
        df.loc[df.event_key.isin(keys), 'partition'] = name
    assert df.groupby('event_key').partition.nunique().max() == 1
    assert df.groupby('image_sha256').partition.nunique().max() == 1
    assert not set(df.loc[df.partition == 'train', 'run']) & set(held_runs)
    assert all((df.partition == p).sum() > 0 for p in ['train', 'dev', 'test', 'run_test'])
    mom = pd.read_csv('/Volumes/easystore/proton-deuteron/momentum_tof.csv').drop_duplicates(KEYS)
    df = df.merge(mom[KEYS + ['momentum']], on=KEYS, how='left', validate='many_to_one')
    np.save(OUT / 'images_log1p.npy', np.log1p(raw))
    ef = endpoint_features(raw)
    np.save(OUT / 'endpoint.npy', ef)
    cols = list(df.columns[df.columns.get_loc('total_adc'):df.columns.get_loc('n_local_maxima') + 1])
    cols += ['height']
    np.save(OUT / 'fulltrack.npy', df[cols].to_numpy(dtype=float))
    # Feature values retained for probes; manifest has keys, images and partition together.
    df.to_csv(OUT / 'manifest.csv', index=False)
    meta = {'seed_split': 20260922, 'config': str(CONFIG), 'config_sha256': digest(CONFIG),
            'tensor': cfg['data']['path'], 'tensor_sha256': digest(cfg['data']['path']),
            'manifest_sha256': digest(OUT / 'manifest.csv'),
            'held_runs': held_runs.tolist(), 'excluded_event_keys': list(conflict_events),
            'endpoint_features': [f'{plane}_{f}' for plane in ['collection', 'induction'] for f in FEATURE_NAMES],
            'fulltrack_features': cols, 'counts': df.groupby(['partition', 'species']).size().unstack().fillna(0).to_dict('index'),
            'unique_events': df.groupby('partition').event_key.nunique().to_dict(),
            'environment': {'torch': torch.__version__, 'numpy': np.__version__, 'python': sys.version}}
    write_json(OUT / 'protocol.json', meta)
    print(json.dumps(meta['counts'], indent=2), flush=True)


def prepare_raw():
    """Reuse the frozen split and physical targets; invert the saved log1p pixels."""
    source = ROOT / 'output/representation_baselines'
    if OUT.resolve() == source.resolve():
        raise ValueError('Set REPRESENTATION_BASELINES_OUT to a separate directory for raw inputs')
    OUT.mkdir(parents=True, exist_ok=True)
    for name in ['manifest.csv', 'protocol.json', 'endpoint.npy', 'fulltrack.npy']:
        dest = OUT / name
        if not dest.exists(): shutil.copy2(source / name, dest)
    assert digest(OUT / 'manifest.csv') == json.loads((OUT / 'protocol.json').read_text())['manifest_sha256']
    original = np.load(source / 'images_log1p.npy', mmap_mode='r')
    dest = OUT / 'images_raw.npy'
    if not dest.exists():
        tmp = dest.with_suffix('.tmp.npy')
        raw = np.lib.format.open_memmap(tmp, mode='w+', dtype='float32', shape=original.shape)
        for lo in range(0, len(original), 256):
            raw[lo:lo + 256] = np.expm1(original[lo:lo + 256])
        raw.flush(); del raw
        tmp.replace(dest)
    write_json(OUT / 'input_scale.json', {'scale': 'raw_adc', 'source': str(source / 'images_log1p.npy'),
                                        'source_sha256': digest(source / 'images_log1p.npy'),
                                        'manifest_sha256': digest(OUT / 'manifest.csv'),
                                        'inversion': 'numpy.expm1 float32; no new source tensor or split'})


def mask_batch(x, generator):
    # 8x8 grid, exactly 16 masked 6x6 patches; shared spatial mask for both planes.
    r = torch.rand(len(x), 64, generator=generator)
    m = torch.zeros_like(r).scatter_(1, r.argsort(dim=1)[:, :16], 1)
    m = m.reshape(-1, 1, 8, 8).repeat_interleave(6, 2).repeat_interleave(6, 3)
    return m.to(x.device)


def active_mask_batch(x, generator):
    """Hide a random quarter of patches containing signal in either plane.

    The mask is shared across planes and contains at least one active patch per
    image. Inactive patches are never selected. Random numbers are drawn on CPU
    with the supplied seeded generator for the same behavior on CPU and MPS.
    """
    activity = (x > 0).any(dim=1, keepdim=True).float()
    active = torch.nn.functional.max_pool2d(activity, kernel_size=6, stride=6).flatten(1).cpu().bool()
    count = active.sum(1)
    if (count == 0).any():
        raise ValueError('An input image has no active patch')
    nmask = torch.clamp((count + 3) // 4, min=1)
    ranks = torch.rand(len(x), 64, generator=generator).masked_fill(~active, float('inf'))
    order = ranks.argsort(dim=1).argsort(dim=1)
    mask = (order < nmask[:, None]).reshape(-1, 1, 8, 8)
    return mask.repeat_interleave(6, 2).repeat_interleave(6, 3).to(x.device, dtype=x.dtype)


def training_rows(df, budget=None):
    if budget is None: return np.flatnonzero(df.partition == 'train')
    rng = np.random.default_rng(712)
    keys = []
    for sp in SPECIES:
        sub = df[(df.partition == 'train') & (df.species == sp)]
        order = rng.permutation(sub.event_key.unique())
        sizes = sub.groupby('event_key').size()
        n = np.searchsorted(np.cumsum([sizes[k] for k in order]), budget // 3) + 1
        keys.extend(order[:n])
    return np.flatnonzero(df.event_key.isin(keys))


def train_one(method, seed, device, epochs=200, batch=128, budget=None):
    stem = method if budget is None else f'{method}_n{budget}'
    target = OUT / 'models' / f'{stem}_s{seed}.pt'
    repr_path = OUT / 'representations' / f'{stem}_s{seed}.npy'
    if repr_path.exists():
        print('already complete', method, seed, flush=True); return
    target.parent.mkdir(exist_ok=True); repr_path.parent.mkdir(exist_ok=True)
    cfg = yaml.safe_load(CONFIG.read_text())
    torch.manual_seed(seed); np.random.seed(seed)
    torch.set_num_threads(4)
    model = build_vae(cfg, device)
    df = pd.read_csv(OUT / 'manifest.csv')
    image_path = OUT / ('images_log1p.npy' if INPUT_SCALE == 'log1p' else 'images_raw.npy')
    images = torch.from_numpy(np.load(image_path)).to(device)
    tr = training_rows(df, budget); dv = np.flatnonzero(df.partition == 'dev')
    write_json(target.with_suffix('.training_rows.json'), {'aligned_rows': tr.tolist(), 'n_images': len(tr),
                                                         'nominal_budget': budget, 'manifest_sha256': digest(OUT / 'manifest.csv')})
    opt = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=.0001)
    rng = torch.Generator().manual_seed(10000 + seed)
    records, best, stale = [], float('inf'), 0
    if target.exists() and method != 'random':
        checkpoint = torch.load(target, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        records = checkpoint['history']; best = checkpoint['best']
        # This path only extracts a completed checkpoint; interrupted runs have .resume files.
    elif method == 'random':
        # Calibrate BatchNorm running statistics using training images, without learning weights.
        model.train()
        with torch.no_grad():
            for ii in np.array_split(tr, max(1, len(tr) // batch)):
                model.encode(images[ii])
    else:
        resume = target.with_suffix('.resume.pt')
        start = 0
        if resume.exists():
            c = torch.load(resume, map_location='cpu', weights_only=False)
            model.load_state_dict(c['state_dict']); opt.load_state_dict(c['optimizer'])
            records, best, stale, start = c['history'], c['best'], c['stale'], c['epoch']
            rng.set_state(c['rng']); torch.set_rng_state(c['torch_rng'])
            if device == 'mps' and 'mps_rng' in c: torch.mps.set_rng_state(c['mps_rng'])
        bestfile = target.with_suffix('.best.pt')
        for epoch in range(start, epochs):
            t = time.time(); model.train()
            order = tr[torch.randperm(len(tr), generator=rng).numpy()]
            total = torch.zeros((), device=device)
            for lo in range(0, len(order), batch):
                xb = images[order[lo:lo + batch]]
                opt.zero_grad(set_to_none=True)
                if method in {'masked_ae', 'active_mask_ae'}:
                    mask = (active_mask_batch(xb, rng) if method == 'active_mask_ae' else mask_batch(xb, rng))
                    mu, lv = model.encode(xb * (1 - mask))
                else:
                    mu, lv = model.encode(xb)
                z = model.reparameterise(mu, lv) if method == 'vae' else mu
                re = model.decode(z)
                err = (re - xb).square() * torch.where(xb > 0, 10., 1.)
                if method in {'masked_ae', 'active_mask_ae'}: err = err * mask * 4
                loss = err.sum((1, 2, 3)).mean()
                if method == 'vae': loss = loss + .25 * (mu.square() + lv.exp() - lv - 1).sum(1).mean()
                if not torch.isfinite(loss): raise RuntimeError(f'Nonfinite loss: {method}/{seed}/{epoch}')
                loss.backward(); opt.step(); total += loss.detach() * len(xb)
            model.eval(); vl = torch.zeros((), device=device)
            vrng = torch.Generator().manual_seed(999)  # Fixed dev mask across seeds/epochs.
            with torch.no_grad():
                for lo in range(0, len(dv), batch):
                    xb = images[dv[lo:lo + batch]]
                    mask = (active_mask_batch(xb, vrng) if method == 'active_mask_ae' else
                            mask_batch(xb, vrng) if method == 'masked_ae' else None)
                    mu, lv = model.encode(xb if mask is None else xb * (1 - mask))
                    # Deterministic checkpoint criterion, not sampled reconstruction noise.
                    re = model.decode(mu)
                    err = (re - xb).square() * torch.where(xb > 0, 10., 1.)
                    if mask is not None: err = err * mask * 4
                    loss = err.sum((1, 2, 3)).mean()
                    if method == 'vae': loss += .25 * (mu.square() + lv.exp() - lv - 1).sum(1).mean()
                    vl += loss * len(xb)
            value = float(vl.cpu()) / len(dv)
            records.append({'epoch': epoch + 1, 'train_objective': float(total.cpu()) / len(tr),
                            'dev_objective': value, 'seconds': time.time() - t})
            if value < best - 1e-4:
                best, stale = value, 0
                torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, bestfile)
            else: stale += 1
            if epoch % 5 == 0 or stale >= 20 or epoch + 1 == epochs:
                print(stem, seed, records[-1], 'stale', stale, flush=True)
                pd.DataFrame(records).to_csv(target.with_suffix('.csv'), index=False)
                c = {'state_dict': model.state_dict(), 'optimizer': opt.state_dict(), 'history': records,
                     'best': best, 'stale': stale, 'epoch': epoch + 1, 'rng': rng.get_state(),
                     'torch_rng': torch.get_rng_state()}
                if device == 'mps': c['mps_rng'] = torch.mps.get_rng_state()
                torch.save(c, resume)
            if stale >= 20: break
        model.load_state_dict(torch.load(bestfile, weights_only=True))
        torch.save({'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    'history': records, 'best': best, 'method': method, 'seed': seed,
                    'manifest_sha256': digest(OUT / 'manifest.csv'), 'batch_size': batch}, target)
        resume.unlink(missing_ok=True); bestfile.unlink(missing_ok=True)
    model.eval(); zs, errors = [], []
    with torch.no_grad():
        for lo in range(0, len(images), 256):
            xb = images[lo:lo + 256]; mu, _ = model.encode(xb)
            zs.append(mu.cpu().numpy())
            if method != 'random':
                errors.append((((model.decode(mu) - xb).square() * torch.where(xb > 0, 10., 1.)).sum((1, 2, 3))).cpu().numpy())
    np.save(repr_path, np.concatenate(zs))
    if errors: np.save(repr_path.with_name(repr_path.stem + '_reconstruction.npy'), np.concatenate(errors))
    if method == 'random':
        torch.save({'state_dict': model.cpu().state_dict(), 'method': method, 'seed': seed,
                    'manifest_sha256': digest(OUT / 'manifest.csv')}, target)
    print('COMPLETE', method, seed, flush=True)
    del images, model
    if device == 'mps': torch.mps.empty_cache()


def analytic(budget=None):
    df = pd.read_csv(OUT / 'manifest.csv'); tr = df.partition == 'train'
    if budget is not None: tr = training_rows(df, budget)
    dest = OUT / 'representations'; dest.mkdir(exist_ok=True)
    suffix = '' if budget is None else f'_n{budget}'
    for stem in ['endpoint', 'fulltrack']:
        a = np.load(OUT / f'{stem}.npy')
        a = SimpleImputer(strategy='median').fit(a[tr]).transform(a)
        scaler = StandardScaler().fit(a[tr]); a = scaler.transform(a)
        np.save(dest / f'{stem}{suffix}.npy', a)
        pca = PCA(8, random_state=0).fit(a[tr]); np.save(dest / f'{stem}_pca{suffix}.npy', pca.transform(a))
    image_path = OUT / ('images_log1p.npy' if INPUT_SCALE == 'log1p' else 'images_raw.npy')
    x = np.load(image_path).reshape(len(df), -1)
    pca = PCA(8, svd_solver='randomized', random_state=0).fit(x[tr])
    np.save(dest / f'pixel_pca{suffix}.npy', pca.transform(x))
    re = []
    for lo in range(0, len(x), 512):
        a = x[lo:lo + 512]; recon = pca.inverse_transform(pca.transform(a))
        re.extend(((recon - a)**2 * np.where(a > 0, 10., 1.)).sum(1))
    np.save(dest / f'pixel_pca{suffix}_reconstruction.npy', np.array(re))
    write_json(OUT / f'analytic_metadata{suffix}.json', {'pixel_pca_variance_fraction': pca.explained_variance_ratio_.sum(),
                                               'manifest_sha256': digest(OUT / 'manifest.csv')})


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'prepare-raw', 'analytic', 'train', 'extension'])
    ap.add_argument('--methods', nargs='+', default=['random', 'ae', 'vae', 'masked_ae'])
    ap.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    ap.add_argument('--epochs', type=int, default=200)
    ap.add_argument('--train-size', type=int)
    args = ap.parse_args()
    if args.stage == 'prepare': prepare()
    elif args.stage == 'prepare-raw': prepare_raw()
    elif args.stage == 'analytic': analytic(args.train_size)
    else:
        device = 'mps' if torch.backends.mps.is_available() else 'cpu'
        print('device', device, flush=True)
        if args.stage == 'extension':
            start = time.time()
            while not (OUT / 'representations/masked_ae_s2.npy').exists():
                if time.time() - start > 7200: raise RuntimeError('Main training did not finish within two hours')
                time.sleep(10)
            budgets = [100, 1000]
        else: budgets = [args.train_size]
        for budget in budgets:
            for seed in args.seeds:
                for method in args.methods:
                    train_one(method, seed, device, args.epochs, budget=budget)


if __name__ == '__main__': main()
