#!/usr/bin/env python3
"""Matched t-SNE views of frozen VAE8 and training-only pixel PCA8."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from sklearn.preprocessing import StandardScaler
from experiments.pilarnet_lariat.latent_truth.prepare import BASE


def run(output):
    frame = pd.read_csv(output/'manifest.csv')
    train = frame.partition.eq('train').to_numpy()
    settings = dict(n_components=2, perplexity=30, init='pca', learning_rate='auto',
                    max_iter=1500, random_state=9105, metric='euclidean', n_jobs=4)
    views, protocol = {}, {'settings': settings, 'sample': f'same {len(frame)} converted simulation particles',
        'scaling': 'each 8D representation standardized using probe-training events only',
        'label_use': 'none during PCA or t-SNE fitting; truth PID colours added afterwards',
        'use': 'joint train/dev/test visualization only; no t-SNE coordinates used for downstream probes',
        'limits': 'independent t-SNE fits have separate arbitrary coordinates; cluster areas and global distances are not comparable'}
    with np.load(output/'representations.npz') as reps:
        for name in ('paper_vae', 'input_pca8'):
            start = time.time()
            scale = StandardScaler().fit(reps[name][train])
            x = scale.transform(reps[name])
            reducer = TSNE(**settings)
            views[name] = reducer.fit_transform(x)
            protocol[name] = {'kl_divergence': float(reducer.kl_divergence_),
                'trustworthiness_15_neighbors': float(trustworthiness(x, views[name], n_neighbors=15)),
                'iterations': int(reducer.n_iter_), 'seconds': time.time()-start}
            print(name, protocol[name], flush=True)
    np.savez_compressed(output/'tsne_views.npz', **views)
    (output/'tsne_protocol.json').write_text(json.dumps(protocol, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_pixels_v2')
    args = p.parse_args()
    run(args.output)
