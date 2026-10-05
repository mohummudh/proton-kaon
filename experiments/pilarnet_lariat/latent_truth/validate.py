#!/usr/bin/env python3
"""Audit alignment, event isolation, frozen encoders and counterfactual closure."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from experiments.pilarnet_lariat.latent_truth.prepare import BASE, partition, digest


def validate(output):
    frame = pd.read_csv(output/'manifest.csv')
    assert frame.row.to_list() == list(range(len(frame))), 'Image/manifest ordering mismatch'
    assert not frame.particle_key.duplicated().any(), 'Duplicate source particles'
    assert frame.groupby('event_key').partition.nunique().max() == 1, 'Event leakage'
    assert (frame.event_key.map(partition) == frame.partition).all(), 'Partition hash mismatch'
    groups = {p: set(f.event_key) for p, f in frame.groupby('partition')}
    assert not (groups['train'] & groups['dev'] or groups['train'] & groups['test'] or groups['dev'] & groups['test'])
    raw = np.load(output/'raw.npy', mmap_mode='r')
    assert raw.shape == (len(frame), 2, 48, 48)
    assert np.isfinite(raw).all() and raw.min() >= 0
    with np.load(output/'representations.npz') as reps:
        assert reps['paper_vae'].shape == (len(frame), 8)
        assert all(len(reps[name]) == len(frame) and np.isfinite(reps[name]).all() for name in reps.files)
        assert set(reps.files) == {'paper_vae','input_pca8','proton_vae','vae_s0','vae_s1','vae_s2','ae_s0','random_s0'}, 'Unexpected predictor representation'
        with np.load(output/'input_pca.npz') as pca:
            pixels = np.log1p(np.asarray(raw)).reshape(len(frame), -1)
            assert pca['components'].shape == (8, 4608)
            assert np.allclose(pca['mean'], pixels[frame.partition.eq('train')].mean(axis=0), rtol=1e-6, atol=1e-6), 'PCA centre is not training-only'
            assert np.allclose(pca['components']@pca['components'].T, np.eye(8), atol=1e-5)
            assert np.allclose((pixels-pca['mean'])@pca['components'].T, reps['input_pca8'], rtol=1e-4, atol=1e-4), 'Pixel projection mismatch'
    with np.load(output/'tsne_views.npz') as views:
        assert set(views.files) == {'paper_vae','input_pca8'}
        assert all(views[m].shape == (len(frame),2) and np.isfinite(views[m]).all() for m in views.files)
    for name, info in json.loads((output/'encoders.json').read_text()).items():
        assert digest(info['checkpoint']) == info['checkpoint_sha256'], f'{name} weights changed'
        assert info['frozen'] and info['inference'].startswith('posterior mean')
    cf = json.loads((output/'counterfactual_protocol.json').read_text())
    assert cf['baseline_latent_max_abs_difference'] < 1e-5, 'Counterfactual baseline mismatch'
    scores = pd.read_csv(output/'truth_readouts.csv')
    assert (scores.n_test >= 25).all(), 'Unsupported readout reported'
    result = {'particles': len(frame), 'species': frame.species.value_counts().to_dict(),
        'unique_events': frame.event_key.nunique(),
        'event_partitions': {p: len(v) for p, v in groups.items()},
        'heldout_particles': int(frame.partition.eq('test').sum()),
        'readout_rows': len(scores), 'frozen_checkpoints': 7,
        'counterfactual_baseline_closure': cf['baseline_latent_max_abs_difference'],
        'pixel_pca_training_mean_and_projection': 'passed',
        'engineered_predictors': False,
        'status': 'passed'}
    (output/'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_pixels_v2')
    args = p.parse_args()
    validate(args.output)
