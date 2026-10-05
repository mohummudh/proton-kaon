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
        'status': 'passed'}
    (output/'validation.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_v1')
    args = p.parse_args()
    validate(args.output)
