#!/usr/bin/env python3
"""Change readout gain or placement of the same held-out truth particle."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/extra')]
import h5py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
import yaml

from experiments.pilarnet_lariat.latent_truth.prepare import BASE
from experiments.pilarnet_lariat.latent_truth.encode import checkpoints, load_model, encode_images
from experiments.pilarnet_lariat.latent_truth.evaluate import classifier, ridge_fit
from experiments.pilarnet_lariat.lariat_forward import (LArIATResponse, particle_groups,
    place_particle, ionization_electrons, simulate_readout, prepare_model_input)
from src.transforms import prepare_images


def run(output, per_species=25):
    frame = pd.read_csv(output/'manifest.csv')
    raw = np.load(output/'raw.npy', mmap_mode='r')
    z = np.load(output/'representations.npz')['paper_vae']
    response = LArIATResponse(**yaml.safe_load((BASE/'profile_matching/profile_response.yaml').read_text()))
    rng = np.random.default_rng(9105)
    selected = []
    for _, f in frame[frame.partition.eq('test')].groupby('species'):
        selected.extend(rng.choice(f.index, min(per_species, len(f)), replace=False))
    records, images, failures = [], [], []
    source = BASE/'full'/frame.source_file.iloc[0]
    with h5py.File(source, 'r') as h5:
        for row in selected:
            meta = frame.iloc[row]
            arrays = tuple(np.asarray(h5[k][int(meta.event_index)]).reshape(-1, w) for k, w in
                           [('point', 8), ('cluster', 6), ('cluster_extra', 5)])
            particle = next(p for p in particle_groups(*arrays) if p['group_id'] == meta.group_id)
            points = particle['points']
            points = points[np.argsort(points[:, 5], kind='stable')]
            charge = ionization_electrons(points[:, 3], points[:, 7]*response.pilarnet_dx_cm_per_unit, response)
            stable = int(hashlib.sha256(meta.particle_key.encode()).hexdigest()[:8], 16)
            waves = {}
            for angle_shift in (0., 2.):
                target = np.r_[np.tan(np.deg2rad([meta.assigned_xz_deg+angle_shift, meta.assigned_yz_deg])), 1.]
                xyz, _ = place_particle(points, particle['vertex_voxels'], response,
                    target_direction=target, allow_displaced=True)
                waves[angle_shift], _ = simulate_readout(xyz, charge, points[:, 5]-points[:, 5].min(),
                    response, seed=stable, windowed=True)
            baseline, _, _, _ = prepare_model_input(waves[0.], response)
            if not np.allclose(baseline, raw[row], rtol=1e-6, atol=1e-5):
                raise AssertionError(f'Baseline regeneration differs for row {row}')
            for name, gain, angle in [('baseline', 1., 0.), ('gain_0.8', .8, 0.),
                                      ('gain_1.2', 1.2, 0.), ('angle_plus_2deg', 1., 2.)]:
                try:
                    image, _, _, _ = prepare_model_input(waves[angle]*gain, response)
                    images.append(image)
                    records.append({'row': int(row), 'event_key': meta.event_key,
                        'species': meta.species, 'variant': name, 'gain': gain, 'angle_shift_deg': angle})
                except ValueError as error:
                    failures.append({'row': int(row), 'variant': name, 'reason': str(error)})
            if len(records) % 100 == 0:
                print('Counterfactual images', len(records), flush=True)
    torch.set_num_threads(4)
    _, config, checkpoint = checkpoints()[0]
    net, cfg, device = load_model(config, checkpoint)
    inputs = prepare_images(torch.from_numpy(np.stack(images)), cfg['data']['transform'], cfg['model']['input_hw'])
    zv, _, re, _ = encode_images(net, inputs, device)
    cf = pd.DataFrame(records)
    train = np.flatnonzero(frame.partition.eq('train'))
    dev = np.flatnonzero(frame.partition.eq('dev'))
    scale = StandardScaler().fit(z[train])
    zs, vs = scale.transform(z), scale.transform(zv)
    pid_model = classifier(z, frame.pid.to_numpy(), train, dev)
    base_pid = pid_model.predict(z)
    cf['probe_pid'] = pid_model.predict(zv)
    cf['pid_changed'] = cf.probe_pid.to_numpy() != base_pid[cf.row.to_numpy()]
    cf['latent_distance'] = np.linalg.norm(vs-zs[cf.row.to_numpy()], axis=1)
    cf['reconstruction_mse_log1p'] = re
    typical = {}
    for species, f in frame.groupby('species'):
        ii = f.index.to_numpy()
        a, b = rng.choice(ii, size=(2, 1000), replace=True)
        keep = a != b
        typical[species] = float(np.median(np.linalg.norm(zs[a[keep]]-zs[b[keep]], axis=1)))
    cf['distance_over_typical_same_species'] = cf.latent_distance/cf.species.map(typical)
    # Primary linear incoming-energy readout, trained independently of variants.
    cf['incoming_ke_predicted_mev'] = np.nan
    cf['incoming_ke_baseline_predicted_mev'] = np.nan
    for species, f in frame.groupby('species'):
        valid = frame.species.eq(species) & frame.kinematics_reliable
        ids = {p: np.flatnonzero(valid & frame.partition.eq(p)) for p in ('train', 'dev')}
        if min(map(len, ids.values())) < 25:
            continue
        energy = np.log1p(frame.incoming_ke_mev.to_numpy())
        model, _ = ridge_fit(z, energy, ids['train'], ids['dev'])
        mask = cf.species.eq(species)
        cf.loc[mask, 'incoming_ke_predicted_mev'] = np.expm1(np.clip(model.predict(zv[mask]), -20, 20))
        cf.loc[mask, 'incoming_ke_baseline_predicted_mev'] = np.expm1(np.clip(model.predict(z[cf.loc[mask, 'row']]), -20, 20))
    cf['energy_probe_fractional_shift'] = (cf.incoming_ke_predicted_mev-cf.incoming_ke_baseline_predicted_mev)/cf.incoming_ke_baseline_predicted_mev.clip(lower=1)
    cf.to_csv(output/'counterfactual_particles.csv', index=False)
    np.savez_compressed(output/'counterfactual_images.npz', raw=np.stack(images), latent=zv)
    summaries = cf.groupby(['species', 'variant']).agg(n=('row', 'size'),
        median_distance=('latent_distance', 'median'),
        median_relative_distance=('distance_over_typical_same_species', 'median'),
        pid_change_fraction=('pid_changed', 'mean'),
        median_energy_probe_fractional_shift=('energy_probe_fractional_shift', 'median'))
    summaries.to_csv(output/'counterfactual_summary.csv')
    (output/'counterfactual_protocol.json').write_text(json.dumps({'per_species': per_species,
        'selected_particles': len(selected), 'failures': failures,
        'baseline_regeneration': 'allclose raw images rtol=1e-6, atol=1e-5',
        'baseline_latent_max_abs_difference': float(np.max(np.abs(zv[cf.variant.eq('baseline')]-z[cf.loc[cf.variant.eq('baseline'), 'row']]))),
        'distance': 'Euclidean after standardization with simulation probe-training events; denominator median random distinct same-species distances',
        'gain_change': 'multiply signed waveforms before threshold, component selection and crop; unchanged truth energy',
        'angle_change': 'rigid placement xz plus 2 degrees, unchanged deposits and charge; volume/crop support may change',
        'limits': '125-particle sensitivity diagnostic, not a response calibration or an invariance guarantee'}, indent=2))
    print(summaries.to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_v1')
    p.add_argument('--per-species', type=int, default=25)
    args = p.parse_args()
    run(args.output, args.per_species)
