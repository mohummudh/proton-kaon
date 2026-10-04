#!/usr/bin/env python3
"""Show the exact activity-aware patch mask on representative training images."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/extra')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np
import pandas as pd
import torch
from _beam_data import apply_style, savefig, SINGLE_COL, DOUBLE_COL
from representation_baselines import active_mask_batch

BASE = ROOT / 'output/representation_baselines'
STAGE = BASE / 'staging_figs'


def main():
    df = pd.read_csv(BASE / 'manifest.csv')
    images = np.load(BASE / 'images_log1p.npy', mmap_mode='r')
    chosen = []
    for species in ['proton', 'kaon', 'muon']:
        idx = np.flatnonzero((df.partition == 'train') & (df.species == species))[:250]
        counts = (images[idx] > 0).any(axis=1).reshape(len(idx), 8, 6, 8, 6).any(axis=(2, 4)).sum(axis=(1, 2))
        chosen.append(idx[np.argmin(abs(counts - np.median(counts)))])
    x = torch.from_numpy(np.asarray(images[chosen]).copy())
    mask = active_mask_batch(x, torch.Generator().manual_seed(20260925)).numpy()
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(3, 4, figsize=(DOUBLE_COL, 4.55))
    vmax = np.quantile(x.numpy(), .997)
    for i, species in enumerate(['Proton', 'Kaon window', 'MIPs']):
        for plane in range(2):
            for view in range(2):
                ax = axes[i, plane * 2 + view]
                pixels = x[i, plane].numpy() * (1 - mask[i, 0] if view else 1)
                ax.imshow(pixels, cmap='Greys_r', vmin=0, vmax=vmax, interpolation='nearest')
                if view:
                    for row, col in np.argwhere(mask[i, 0, ::6, ::6] > 0):
                        ax.add_patch(Rectangle((col * 6 - .5, row * 6 - .5), 6, 6,
                                               fill=False, edgecolor='#EE7733', linewidth=.8))
                ax.set_xticks([]); ax.set_yticks([])
                for spine in ax.spines.values(): spine.set_linewidth(.5)
                if i == 0:
                    ax.set_title(('Collection' if plane == 0 else 'Induction') +
                                 (' input' if view == 0 else ' masked'), fontsize=8)
                if plane == 0 and view == 0: ax.set_ylabel(species, fontsize=8)
    fig.tight_layout(w_pad=.4, h_pad=.4)
    savefig(fig, STAGE, 'active_patch_mask_examples')


if __name__ == '__main__': main()
