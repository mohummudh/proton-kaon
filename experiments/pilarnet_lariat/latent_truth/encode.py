#!/usr/bin/env python3
"""Deterministic frozen encoders; no simulation training or checkpoint edits."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import torch
import yaml

from experiments.pilarnet_lariat.latent_truth.prepare import BASE, digest
from src.device import pick_device
from src.models.build import build_vae
from src.train.naming import model_filename
from src.transforms import prepare_images


def checkpoints():
    paper = next((ROOT/'configs').glob('run_0093*'))
    original = next((ROOT/'configs').glob('run_0066*'))
    cfg = yaml.safe_load(paper.read_text())
    old = yaml.safe_load(original.read_text())
    base = ROOT/'output/representation_baselines/models'
    return [('paper_vae', paper, Path(cfg['output']['dir'])/model_filename(cfg)),
            ('proton_vae', original, Path(old['output']['dir'])/model_filename(old)),
            *[(f'vae_s{s}', paper, base/f'vae_s{s}.pt') for s in range(3)],
            ('ae_s0', paper, base/'ae_s0.pt'), ('random_s0', paper, base/'random_s0.pt')]


def load_model(config, checkpoint):
    cfg = yaml.safe_load(Path(config).read_text())
    device = pick_device()
    net = build_vae(cfg, device)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    net.load_state_dict(saved.get('state_dict', saved))
    net.eval()
    return net, cfg, device


def encode_images(net, images, device, batch_size=64):
    latents, posterior, errors, reconstructions = [], [], [], []
    with torch.inference_mode():
        for start in range(0, len(images), batch_size):
            xb = images[start:start+batch_size].to(device)
            mu, logvar = net.encode(xb)
            reconstruction = net.decode(mu)
            latents.append(mu.cpu().numpy()); posterior.append(logvar.cpu().numpy())
            errors.append(((reconstruction-xb)**2).mean((1, 2, 3)).cpu().numpy())
            reconstructions.append(reconstruction.cpu().numpy())
    return tuple(np.concatenate(a) for a in (latents, posterior, errors, reconstructions))


def encode(output, reuse=None):
    torch.set_num_threads(4)
    raw = np.load(output/'raw.npy', mmap_mode='r')
    manifest = pd.read_csv(output/'manifest.csv')
    train = manifest.partition.eq('train').to_numpy()
    representations, provenance = {}, {}
    cached = np.load(reuse/'representations.npz') if reuse else None
    cached_provenance = json.loads((reuse/'encoders.json').read_text()) if reuse else {}
    if reuse and (digest(output/'manifest.csv') != digest(reuse/'manifest.csv') or
                  digest(output/'raw.npy') != digest(reuse/'raw.npy')):
        raise ValueError('Cached encodings require identical ordered particles and images')
    for name, config, checkpoint in checkpoints():
        start = time.time()
        if cached is not None:
            info = cached_provenance[name]
            if digest(checkpoint) != info['checkpoint_sha256'] or digest(config) != info['config_sha256']:
                raise ValueError(f'{name} cached checkpoint/config mismatch')
            representations[name] = cached[name]
            provenance[name] = {**info, 'reused_from': str(reuse)}
            for suffix in ('logvar', 'reconstruction_error', 'reconstruction'):
                old = reuse/f'{name}_{suffix}.npy'
                if old.exists():
                    shutil.copy2(old, output/old.name)
            print('Reused frozen encoder', name, flush=True)
            continue
        net, cfg, device = load_model(config, checkpoint)
        if cfg['data'].get('transform') != 'log1p':
            raise ValueError('This pilot expects the frozen log1p input convention')
        images = prepare_images(torch.from_numpy(np.array(raw)), cfg['data']['transform'], cfg['model']['input_hw'])
        z, logvar, error, reconstruction = encode_images(net, images, device)
        representations[name] = z
        np.save(output/f'{name}_logvar.npy', logvar)
        np.save(output/f'{name}_reconstruction_error.npy', error)
        if name == 'paper_vae':
            np.save(output/f'{name}_reconstruction.npy', reconstruction)
        provenance[name] = {'config': str(config), 'checkpoint': str(checkpoint),
            'checkpoint_sha256': digest(checkpoint), 'config_sha256': digest(config),
            'latent_dim': z.shape[1], 'device': str(device), 'frozen': True,
            'inference': 'posterior mean and decode(mean); log1p once; eval mode',
            'seconds': time.time()-start}
        print(name, z.shape, f'{time.time()-start:.1f}s', flush=True)
        del net, reconstruction
    transformed = np.log1p(np.array(raw)).reshape(len(raw), -1)
    pca = PCA(n_components=8, svd_solver='randomized', random_state=9105).fit(transformed[train])
    representations['input_pca8'] = pca.transform(transformed)
    np.savez_compressed(output/'representations.npz', **representations)
    (output/'encoders.json').write_text(json.dumps(provenance, indent=2))
    np.savez(output/'input_pca.npz', mean=pca.mean_, components=pca.components_,
             explained_variance_ratio=pca.explained_variance_ratio_)
    (output/'representation_protocol.json').write_text(json.dumps({
        'baseline': 'pixel PCA', 'n_components': 8, 'input_dimensions': transformed.shape[1],
        'input': 'flatten both 48x48 planes after the same single log1p ADC transform as the VAE',
        'fit': 'training events only; centre pixels; no per-pixel variance scaling or truth labels',
        'train_particles': int(train.sum()), 'manifest_sha256': digest(output/'manifest.csv'),
        'retained_training_variance': float(pca.explained_variance_ratio_.sum()),
        'engineered_predictors': False}, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_pixels_v2')
    p.add_argument('--reuse-encodings', type=Path)
    args = p.parse_args()
    encode(args.output, args.reuse_encodings)
