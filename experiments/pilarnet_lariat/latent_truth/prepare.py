#!/usr/bin/env python3
"""Balanced, event-partitioned pilot using one frozen response for every PID."""

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
import h5py
import numpy as np
import pandas as pd
import yaml

from experiments.pilarnet_lariat.apply_all import kinetic_energy
from experiments.pilarnet_lariat.lariat_forward import (
    LArIATResponse, particle_groups, place_particle, ionization_electrons,
    simulate_readout, prepare_model_input, wire_coordinates)
from experiments.pilarnet_lariat.proton_profiles import binned_profile, judge_profile

BASE = Path('/Volumes/easystore/proton-kaon/pilarnet_lariat')
SPECIES = {0: 'photon', 1: 'electron', 2: 'muon', 3: 'pion', 4: 'proton'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def partition(event_key):
    value = int(hashlib.sha256(('pilarnet-truth-v1:'+event_key).encode()).hexdigest()[:8], 16)/2**32
    return 'train' if value < .6 else 'dev' if value < .8 else 'test'


def geometry(points, voxel_cm):
    xyz = points[:, :3]*voxel_cm
    center = xyz.mean(axis=0)
    covariance = (xyz-center).T@(xyz-center)/len(xyz)
    eig, vectors = np.linalg.eigh(covariance)
    direction = vectors[:, -1]
    extent = np.ptp((xyz-center)@direction)
    return {'linearity_3d': float(eig[-1]/max(eig.sum(), 1e-12)),
            'transverse_rms_cm': float(np.sqrt(max(0, eig[:2].sum()))),
            'extent_3d_cm': float(extent),
            'chord_cm': float(np.linalg.norm(xyz[-1]-xyz[0]))}


def truth_features(particle, points, clusters, extras, response):
    indices = np.flatnonzero(clusters[:, 2] == particle['group_id'])
    counts = clusters[indices, 0]
    sem_counts = {i: int(counts[clusters[indices, 4] == i].sum()) for i in range(4)}
    interaction = np.unique(clusters[indices, 3].astype(int))
    dx = points[:, 7]*response.pilarnet_dx_cm_per_unit
    energy = points[:, 3]
    path = float(dx.sum())
    rr = path-np.cumsum(dx)+dx/2
    profile, _ = binned_profile(rr, energy/dx, dx)
    baseline = (rr > 5) & (rr < 12)
    endpoint = rr < 2
    bragg = float(np.average(energy[endpoint]/dx[endpoint], weights=dx[endpoint])/
                  np.average(energy[baseline]/dx[baseline], weights=dx[baseline])) if baseline.sum() >= 3 and endpoint.sum() >= 3 else np.nan
    g = geometry(points, response.voxel_cm)
    momenta = extras[indices, 1]*1000
    spread = float(np.ptp(momenta)/max(momenta.max(), 1e-12))
    incoming = kinetic_energy(particle['momentum']*1000, particle['mass_mev'])
    # Fragment momenta need not describe the same initial step. Do not silently
    # treat ambiguous first-fragment metadata as a unique incoming-energy truth.
    reliable = bool(spread <= .05 and incoming > 0 and energy.sum() <= 1.15*incoming)
    return {**g, 'interaction_id': int(interaction[0]) if len(interaction) == 1 else -2,
        'semantic': max(sem_counts, key=sem_counts.get),
        'semantic_purity': max(sem_counts.values())/len(points),
        'shower_fraction': sem_counts[0]/len(points), 'michel_fraction': sem_counts[2]/len(points),
        'delta_fraction': sem_counts[3]/len(points), 'n_fragments': len(indices),
        'incoming_ke_mev': incoming, 'momentum_mev': particle['momentum']*1000,
        'kinematics_reliable': reliable, 'fragment_momentum_spread': spread,
        'deposited_mev': float(energy.sum()), 'path_cm': path,
        'mean_dedx_mev_cm': float(energy.sum()/path),
        'endpoint_dedx_mev_cm': float(energy[endpoint].sum()/dx[endpoint].sum()) if endpoint.any() else np.nan,
        'bragg_ratio': bragg, 'chord_over_path': g['chord_cm']/max(path, 1e-12),
        'n_voxels': len(points),
        'time_order_unique_fraction': len(np.unique(points[:, 5]))/len(points),
        **{f'vertex_{axis}_cm': float(particle['vertex_voxels'][i]*response.voxel_cm)
           for i, axis in enumerate('xyz')},
        **({f'proton_{k}': v for k, v in judge_profile(profile).items()} if particle['pid'] == 4 else {})}


def prepare(source, output, per_species=500, max_events=1200, seed=9105):
    output.mkdir(parents=True, exist_ok=True)
    if (output/'manifest.csv').exists():
        raise RuntimeError('Pilot already exists; use another output path to change the sample')
    response_path = BASE/'profile_matching/profile_response.yaml'
    angles_path = BASE/'profile_matching/fit_reco_angles_deg.npy'
    response = LArIATResponse(**yaml.safe_load(response_path.read_text()))
    angles = np.load(angles_path)
    source_relative = source.relative_to(BASE/'full').as_posix()
    candidates, preselection = [], Counter()
    with h5py.File(source, 'r') as h5:
        for event in range(min(max_events, len(h5['cluster']))):
            clusters = np.asarray(h5['cluster'][event]).reshape(-1, 6)
            for group in np.unique(clusters[:, 2].astype(int)):
                if group < 0:
                    continue
                c = clusters[clusters[:, 2] == group]
                pids = np.unique(c[:, 5].astype(int))
                if len(pids) != 1 or pids[0] not in SPECIES:
                    continue
                pid = int(pids[0]); n = int(c[:, 0].sum())
                preselection[f'{SPECIES[pid]}:all'] += 1
                if not 20 <= n <= 10000:
                    preselection[f'{SPECIES[pid]}:voxel_cut'] += 1
                    continue
                candidates.append((event, group, pid))
        rng = np.random.default_rng(seed)
        rng.shuffle(candidates)
        records, images, rejected, counts = [], [], [], Counter()
        for event, group, pid in candidates:
            if counts[pid] >= per_species:
                continue
            event_key = f'{source_relative}:event{event}'
            particle_key = f'{event_key}:group{group}'
            stable = int(hashlib.sha256(particle_key.encode()).hexdigest()[:8], 16)
            arrays = tuple(np.asarray(h5[k][event]).reshape(-1, w) for k, w in
                           [('point', 8), ('cluster', 6), ('cluster_extra', 5)])
            try:
                particle = next(p for p in particle_groups(*arrays) if p['group_id'] == group)
                points = particle['points']
                points = points[np.argsort(points[:, 5], kind='stable')]
                if np.any(points[:, 7] <= 0):
                    raise ValueError('Nonpositive path lengths')
                xz, yz = angles[stable % len(angles)]
                target = np.r_[np.tan(np.deg2rad([xz, yz])), 1.]
                xyz, placement = place_particle(points, particle['vertex_voxels'], response,
                    target_direction=target, allow_displaced=True)
                electrons = ionization_electrons(points[:, 3], points[:, 7]*response.pilarnet_dx_cm_per_unit, response)
                wave, audit = simulate_readout(xyz, electrons, points[:, 5]-points[:, 5].min(),
                                               response, seed=stable, windowed=True)
                raw, _, _, crop = prepare_model_input(wave, response)
                if min(c['selected_signal_fraction'] for c in crop) < .5:
                    raise ValueError('Dominant component carries less than half the positive signal')
                if np.count_nonzero(raw[0].max(axis=1)) < 5:
                    raise ValueError('Less than five occupied collection rows')
                truth = truth_features(particle, points, arrays[1], arrays[2], response)
                inside = ((xyz[:, 0] >= 0) & (xyz[:, 0] <= response.main_drift_cm) &
                          (np.abs(xyz[:, 1]) <= response.height_cm/2) & (xyz[:, 2] >= 0) &
                          (xyz[:, 2] <= response.length_cm))
                wires = np.floor(wire_coordinates(xyz, response)+.5).astype(int)
                retained_energy, retained_charge = [], []
                for plane, c in enumerate(crop):
                    lo, _, hi, _ = c['bbox_wire_tick']
                    lo = max(lo, hi-50)+audit['window_origin_wire_tick'][0]
                    hi += audit['window_origin_wire_tick'][0]
                    keep = inside & (wires[:, plane] >= lo) & (wires[:, plane] < hi)
                    retained_energy.append(float(points[keep, 3].sum()))
                    retained_charge.append(float(electrons[keep].sum()))
                direction = np.asarray(placement['source_direction_estimate'])
                direction /= max(np.linalg.norm(direction), 1e-12)
                records.append({'row': len(records), 'source_file': source_relative,
                    'event_index': event, 'event_key': event_key, 'group_id': group,
                    'particle_key': particle_key, 'pid': pid, 'species': SPECIES[pid],
                    'partition': partition(event_key), **truth,
                    'assigned_xz_deg': float(xz), 'assigned_yz_deg': float(yz),
                    **{f'source_direction_{axis}': float(direction[i]) for i, axis in enumerate('xyz')},
                    'approximate_direction': 'approximate' in placement['direction_method'],
                    'visible_charge_fraction': audit['inside_volume_electrons']/max(electrons.sum(), 1e-12),
                    'geom_retained_energy_mev': float(np.mean(retained_energy)),
                    'geom_retained_electrons': float(np.mean(retained_charge)),
                    'geometric_energy_fraction': float(np.mean(retained_energy)/max(points[:, 3].sum(), 1e-12)),
                    'collection_component_fraction': crop[0]['selected_signal_fraction'],
                    'induction_component_fraction': crop[1]['selected_signal_fraction'],
                    'collection_native_wires': crop[0]['component_shape'][0],
                    'induction_native_wires': crop[1]['component_shape'][0]})
                images.append(raw); counts[pid] += 1
            except (ValueError, StopIteration) as error:
                rejected.append({'event_key': event_key, 'group_id': group, 'pid': pid, 'reason': str(error)})
            if len(records) and len(records) % 100 == 0:
                print('Accepted', len(records), {SPECIES[k]: counts[k] for k in SPECIES}, flush=True)
            if all(counts[k] >= per_species for k in SPECIES):
                break
    frame = pd.DataFrame(records)
    frame.to_csv(output/'manifest.csv', index=False)
    np.save(output/'raw.npy', np.stack(images))
    pd.DataFrame(rejected).to_csv(output/'rejected.csv', index=False)
    status_path = BASE/'full/download_status.json'
    status = json.loads(status_path.read_text())
    protocol = {'source': str(source), 'source_relative': source_relative,
        'published_sha256': status['files'][source_relative].get('sha256'),
        'source_state': status['files'][source_relative]['state'],
        'response': asdict(response), 'response_sha256': digest(response_path),
        'angle_bank_sha256': digest(angles_path), 'seed': seed, 'events_scanned': max_events,
        'requested_per_species': per_species, 'counts': frame.species.value_counts().to_dict(),
        'partitions': frame.groupby(['species', 'partition']).size().to_dict(),
        'preselection': dict(preselection), 'rejections': len(rejected),
        'truth_limits': ['no kaons', 'no parent/daughter tree or exact generator momentum direction',
          'sum(dx) is collective path for showers, not a shower length',
          'retained energy uses voxel-centre geometric support, not exact waveform ancestry',
          'incoming kinematics use first-fragment metadata only when fragment momenta agree and energy closes',
          'particles are truth-isolated; instance and voxel segmentation are not evaluated',
          'independent placement removes original vertex/direction; no original-event geometry in the encoder'],
        'selection': 'random candidates from one unused shard; 20–10000 voxels, ≥5 occupied rows, ≥50% connected positive signal',
        'response_status': 'exploratory proton-only calibration, not validated for other species',
        'full_dataset_claim': False}
    protocol['partitions'] = {f'{s}:{p}': n for (s, p), n in protocol['partitions'].items()}
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2))
    print(json.dumps({'counts': protocol['counts'], 'partitions': protocol['partitions'],
                      'rejections': len(rejected)}, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=BASE/'full/train/generic_v2_77600_v2.h5')
    p.add_argument('--output', type=Path, default=BASE/'latent_truth/pilot_pixels_v2')
    p.add_argument('--per-species', type=int, default=500)
    p.add_argument('--max-events', type=int, default=1200)
    args = p.parse_args()
    prepare(args.source, args.output, args.per_species, args.max_events)
