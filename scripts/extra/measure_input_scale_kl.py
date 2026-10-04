#!/usr/bin/env python3
"""Measure deterministic reconstruction and KL terms for matched VAE models."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch
import yaml
from src.models.build import build_vae


def main():
    torch.set_num_threads(4)
    df = pd.read_csv(ROOT / 'output/representation_baselines/manifest.csv')
    idx = np.random.default_rng(2104).choice(np.flatnonzero(df.partition == 'train'), 1024, replace=False)
    cfg = yaml.safe_load(next((ROOT / 'configs').glob('run_0093*')).read_text())
    rows = []
    for scale, folder, image_file in [
        ('log1p', 'representation_baselines', 'images_log1p.npy'),
        ('raw', 'representation_baselines_raw', 'images_raw.npy'),
    ]:
        base = ROOT / 'output' / folder
        x = np.load(base / image_file, mmap_mode='r')
        for seed in range(3):
            model = build_vae(cfg, 'cpu')
            checkpoint = torch.load(base / 'models' / f'vae_s{seed}.pt', map_location='cpu', weights_only=False)
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            reconstruction, kl = [], []
            with torch.no_grad():
                for chunk in np.array_split(idx, 8):
                    xb = torch.from_numpy(np.asarray(x[chunk]).copy())
                    mu, lv = model.encode(xb)
                    re = model.decode(mu)
                    reconstruction.extend(((re - xb).square() * torch.where(xb > 0, 10., 1.)).sum((1, 2, 3)).numpy())
                    kl.extend((.25 * (mu.square() + lv.exp() - lv - 1).sum(1)).numpy())
            rec, reg = np.mean(reconstruction), np.mean(kl)
            rows.append({'scale': scale, 'seed': seed, 'n_train': len(idx),
                         'reconstruction': rec, 'beta_kl': reg,
                         'kl_fraction': reg / (rec + reg)})
    out = ROOT / 'output/representation_baselines_raw/input_scale_kl_balance.csv'
    pd.DataFrame(rows).to_csv(out, index=False)
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__': main()
