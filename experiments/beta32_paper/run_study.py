#!/usr/bin/env python3
"""Resume the original paper experiment with beta=32; write only here.

The objective, batch reductions, stochastic validation, Adam settings, split
indices and stopping rule match scripts/run_training.py and src/train/train.py.
Epoch checkpoints add restart support without changing the training algorithm.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import yaml
from src.models.build import build_vae
from src.losses.vae import vae_loss

def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b''): h.update(chunk)
    return h.hexdigest()

def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, default=str) + '\n')
    tmp.replace(path)

def cpu_tree(value):
    if torch.is_tensor(value): return value.detach().cpu().clone()
    if isinstance(value, dict): return {k: cpu_tree(v) for k, v in value.items()}
    if isinstance(value, list): return [cpu_tree(v) for v in value]
    if isinstance(value, tuple): return tuple(cpu_tree(v) for v in value)
    return value

def save_checkpoint(path, value):
    tmp = path.with_suffix('.tmp.pt')
    torch.save(cpu_tree(value), tmp)
    tmp.replace(path)

def run_epoch(model, opt, loader, beta, device, updating):
    """Preserve the original batch-mean reductions and stochastic forwards."""
    model.train(updating); sums = np.zeros(3)
    with torch.set_grad_enabled(updating):
        for xb in loader:
            xb = xb.to(device)
            if updating: opt.zero_grad()
            re, mu, lv, _ = model(xb)
            loss, r, kl = vae_loss(re, xb, mu, lv, beta=beta)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite objective')
            if updating: loss.backward(); opt.step()
            sums += [loss.item(), r.item(), kl.item()]
    return sums / len(loader)

def prepare():
    """Freeze the original input, split files and implementation provenance."""
    data = HERE / 'data'; data.mkdir(exist_ok=True)
    configs = HERE / 'configs'; configs.mkdir(exist_ok=True)
    splitdir = HERE / 'splits'; splitdir.mkdir(exist_ok=True)
    original = next((ROOT / 'configs').glob('run_0093*'))
    cfg = yaml.safe_load(original.read_text())
    input_path = Path(cfg['data']['path'])
    split_source = Path(cfg['output']['splits_dir'])
    source_files = [original, ROOT / 'src/losses/vae.py', ROOT / 'src/train/train.py',
                    ROOT / 'src/models/configVAE.py', ROOT / 'scripts/run_training.py',
                    ROOT / 'scripts/extra/_sweep_measure.py',
                    ROOT / 'scripts/extra/cluster_k_sweep.py',
                    ROOT / 'scripts/analyse_latents.py']
    fingerprint = {str(p.relative_to(ROOT)): digest(p) for p in source_files}
    provenance_path = HERE / 'provenance.json'
    if provenance_path.exists():
        previous = json.loads(provenance_path.read_text())
        if previous['implementation_hashes'] != fingerprint:
            raise RuntimeError('Original implementation changed since preparation')
    dest = data / input_path.name
    if not dest.exists(): shutil.copy2(input_path, dest)
    input_hash = digest(dest)
    if input_hash != digest(input_path): raise RuntimeError('Input copy hash mismatch')
    cfg['data']['path'] = str(dest)
    cfg['output'] = {'dir': str(HERE / 'runs'), 'splits_dir': str(splitdir),
                     'inference_dir': str(HERE / 'runs')}
    tasks = []
    def add(tag, latent, seed, beta=32., family='main'):
        key = f'{tag}_d{latent}_s{seed}_b{beta:g}'
        if any(t['id'] == key for t in tasks): return
        c = copy.deepcopy(cfg)
        c['data']['tag'] = tag; c['model']['latent'] = latent
        c['train']['beta'] = beta; c['train']['seed'] = seed
        split = splitdir / f'split_all_{tag}.npz'
        if not split.exists(): shutil.copy2(split_source / split.name, split)
        split_hash = digest(split)
        if split_hash != digest(split_source / split.name): raise RuntimeError('Split copy mismatch')
        path = configs / (key + '.yaml')
        path.write_text(yaml.safe_dump(c, sort_keys=False))
        tasks.append({'id': key, 'family': family, 'config': str(path.relative_to(HERE)),
                      'tag': tag, 'latent': latent, 'seed': seed, 'beta': beta,
                      'split_sha256': split_hash})
    # Main model first; a paired fresh beta=.5 model separates beta from the
    # historical unseeded model's unknown initialization.
    add('bal9419', 8, 0)
    add('bal9419', 8, 0, .5, 'paired_control')
    for seed in [1, 2]: add('bal9419', 8, seed)
    split_grid = yaml.safe_load((ROOT / 'configs/sweep_split_pool8227_seeded.yaml').read_text())['grid']
    for tag in split_grid['data.tag']:
        for seed in split_grid['train.seed']: add(tag, 8, seed, family='training_size')
    capacity_grid = yaml.safe_load((ROOT / 'configs/sweep_latent_pool8227_tr50.yaml').read_text())['grid']
    for latent in capacity_grid['model.latent']:
        for seed in capacity_grid['train.seed']: add('pool8227_tr50', latent, seed, family='latent_capacity')
    write_json(HERE / 'plan.json', tasks)
    source_pdf = Path('/Users/user/Downloads/neurips.pdf')
    paper = HERE / 'paper_reference.pdf'
    if not paper.exists(): shutil.copy2(source_pdf, paper)
    write_json(provenance_path, {'paper_sha256': digest(paper), 'input_sha256': input_hash,
        'input_source': str(input_path), 'implementation_hashes': fingerprint,
        'original_config': str(original.relative_to(ROOT)), 'runs': len(tasks),
        'main_historical_seed': None, 'fresh_main_seeds': [0, 1, 2],
        'primary_fresh_main_seed': 0, 'paired_beta05_seed': 0,
        'note': 'Historical main model was unseeded; exact original RNG cannot be recovered. '
                'New main seeds are explicit. Paired beta=.5 seed zero is also retrained.'})
    print('PREPARED', len(tasks), 'unique runs', flush=True)

def train_task(task, device, epoch_limit=None):
    cfg = yaml.safe_load((HERE / task['config']).read_text())
    # Configs retain the original local paths for audit; these relocations make
    # the frozen folder executable on another host without changing the data.
    cfg['data']['path'] = str(HERE / 'data' / Path(cfg['data']['path']).name)
    cfg['output'] = {'dir':str(HERE/'runs'), 'splits_dir':str(HERE/'splits'),
                     'inference_dir':str(HERE/'runs')}
    dest = HERE / 'runs' / task['id']; dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'complete.json').exists(): return
    prov = json.loads((HERE / 'provenance.json').read_text())
    if digest(cfg['data']['path']) != prov['input_sha256']:
        raise RuntimeError('Frozen input tensor changed')
    splitpath = HERE / 'splits' / f"split_all_{task['tag']}.npz"
    if digest(splitpath) != task['split_sha256']: raise RuntimeError('Split changed')
    seed = task['seed']; random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    torch.set_num_threads(4)
    raw = torch.load(cfg['data']['path'], weights_only=False, map_location='cpu')
    images = torch.cat([torch.log1p(raw[k]) for k in ['p', 'k', 'm']]); del raw
    assert images.shape == (27657, 2, 48, 48) and torch.isfinite(images).all()
    split = np.load(splitpath); tr, va = split['train_idx'], split['val_idx']
    assert not np.intersect1d(tr, va).size
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(Subset(images, tr), batch_size=cfg['train']['batch_size'],
                              shuffle=True, generator=generator)
    val_loader = DataLoader(Subset(images, va), batch_size=cfg['train']['batch_size'], shuffle=False)
    model = build_vae(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg['optimizer']['lr'],
                           weight_decay=cfg['optimizer']['weight_decay'])
    history, best, best_state, stale, start = [], float('inf'), None, 0, 0
    resume = dest / 'resume.pt'
    if resume.exists():
        c = torch.load(resume, weights_only=False, map_location='cpu')
        if c['config'] != cfg: raise RuntimeError('Resume config mismatch')
        previous_device = c.get('device', 'mps' if torch.is_tensor(c.get('device_rng'))
                                else 'cuda' if 'device_rng' in c else 'cpu')
        if previous_device != device: raise RuntimeError('Resume on the original device type to preserve RNG')
        model.load_state_dict(c['state_dict']); opt.load_state_dict(c['optimizer'])
        history, best, best_state, stale, start = c['history'], c['best'], c['best_state'], c['stale'], c['epoch']
        generator.set_state(c['loader_rng']); torch.set_rng_state(c['torch_rng'])
        if device == 'mps': torch.mps.set_rng_state(c['device_rng'])
        if device == 'cuda': torch.cuda.set_rng_state_all(c['device_rng'])
    cap = cfg['train']['epochs']
    stop = min(cap, start + epoch_limit) if epoch_limit else cap
    print('START', task['id'], device, 'epoch', start, flush=True)
    for epoch in range(start, stop):
        if stale >= cfg['train']['patience']: break
        t = time.monotonic(); row = {'epoch': epoch + 1}
        for label, loader, updating in [('train', train_loader, True), ('val', val_loader, False)]:
            # Original trainer averages batches equally, including the short last batch.
            values = run_epoch(model,opt,loader,task['beta'],device,updating)
            row.update(dict(zip([label + '_loss', label + '_recon', label + '_kl'], values)))
        row['seconds'] = time.monotonic() - t; history.append(row)
        if row['val_loss'] < best - cfg['train']['min_delta']:
            best, stale, best_state = row['val_loss'], 0, cpu_tree(model.state_dict())
        else: stale += 1
        c = {'config': cfg, 'device':device, 'state_dict': model.state_dict(), 'optimizer': opt.state_dict(),
             'history': history, 'best': best, 'best_state': best_state, 'stale': stale,
             'epoch': epoch + 1, 'loader_rng': generator.get_state(), 'torch_rng': torch.get_rng_state()}
        if device == 'mps': c['device_rng'] = torch.mps.get_rng_state()
        if device == 'cuda': c['device_rng'] = torch.cuda.get_rng_state_all()
        save_checkpoint(resume, c)
        write_json(dest / 'history.json', history)
        write_json(HERE / 'status.json', {'active_run': task['id'], **row, 'stale': stale})
        print(task['id'], row, 'stale', stale, flush=True)
    if len(history) < cap and stale < cfg['train']['patience']:
        print('PAUSED at requested epoch limit', flush=True); return
    model.load_state_dict(best_state)
    save_checkpoint(dest / 'model.pt', {'state_dict': best_state, 'config': cfg,
        'input_sha256': prov['input_sha256'], 'split_sha256': task['split_sha256'],
        'best_val_loss': best, 'history': history})
    model.eval(); zs, vs, rs, sampled_rs = [], [], [], []
    # Same mean-head representation. Retain mean and sampled recon losses explicitly.
    with torch.no_grad():
        for lo in range(0, len(images), 8):
            xb = images[lo:lo+8].to(device); re, mu, lv, _ = model(xb)
            weights = torch.where(xb > .01, 10., 1.)
            zs.append(mu.cpu().numpy()); vs.append(lv.cpu().numpy())
            rs.append(((model.decode(mu)-xb).square()*weights).sum((1,2,3)).cpu().numpy())
            sampled_rs.append(((re-xb).square()*weights).sum((1,2,3)).cpu().numpy())
    np.savez_compressed(dest / 'latents.npz', mu=np.concatenate(zs), logvar=np.concatenate(vs),
                        mean_reconstruction=np.concatenate(rs), sampled_reconstruction=np.concatenate(sampled_rs))
    with np.load(dest/'latents.npz') as posterior:
        if not all(np.isfinite(posterior[k]).all() for k in posterior.files):
            raise RuntimeError('Nonfinite posterior or reconstruction output')
        assert posterior['mu'].shape == (27657,task['latent'])
    write_json(dest / 'complete.json', {'id': task['id'], 'epochs': len(history),
        'best_val_loss': best, 'device': device, 'torch_version': torch.__version__,
        'input_sha256': prov['input_sha256'], 'split_sha256': task['split_sha256'],
        'model_sha256': digest(dest / 'model.pt'), 'latents_sha256': digest(dest / 'latents.npz')})
    resume.unlink(missing_ok=True)
    print('COMPLETE', task['id'], flush=True)

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('stage', choices=['prepare', 'train'])
    ap.add_argument('--ids', nargs='+')
    ap.add_argument('--families', nargs='+')
    ap.add_argument('--epoch-limit', type=int)
    ap.add_argument('--device', choices=['mps', 'cpu', 'cuda'])
    args = ap.parse_args()
    if args.stage == 'prepare': prepare(); return
    tasks = json.loads((HERE / 'plan.json').read_text())
    device = args.device or ('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    for task in tasks:
        if args.ids and task['id'] not in args.ids: continue
        if args.families and task['family'] not in args.families: continue
        train_task(task, device, args.epoch_limit)

if __name__ == '__main__': main()
