#!/usr/bin/env python3
"""Convert a small PILArNet-M NPZ/HDF5 sample to approximate LArIAT images."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from experiments.pilarnet_lariat.lariat_forward import (LArIATResponse, particle_groups, place_particle,
                                ionization_electrons, simulate_readout, prepare_model_input)

PID = {2: ('m', 'Muon'), 3: ('pi', 'Pion'), 4: ('p', 'Proton')}


def read_event(path, event_index):
    if path.suffix == '.npz':
        with np.load(path, allow_pickle=False) as data:
            return tuple(data[key].copy() for key in ('point', 'cluster', 'cluster_extra'))
    if path.suffix in ('.h5', '.hdf5'):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError('Install the pilarnet optional dependency to read HDF5') from exc
        with h5py.File(path, 'r') as data:
            return tuple(np.asarray(data[key][event_index]).reshape(-1, width)
                         for key, width in (('point', 8), ('cluster', 6), ('cluster_extra', 5)))
    raise ValueError('Input must be a PILArNet-M .h5 file or an extracted event .npz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', nargs='+', required=True, type=Path)
    parser.add_argument('--event-index', type=int, default=0, help='HDF5 event index; ignored for NPZ')
    parser.add_argument('--config', type=Path, default=Path(__file__).resolve().parent / 'configs/run2.yaml')
    parser.add_argument('--output', type=Path, default=Path('output/pilarnet_lariat'))
    parser.add_argument('--max-particles', type=int, default=24)
    parser.add_argument('--xz-deg', type=float, default=0, help='Beam direction angle in x-z plane')
    parser.add_argument('--yz-deg', type=float, default=0, help='Beam direction angle in y-z plane')
    parser.add_argument('--beam-jitter', action='store_true', help='MC beamlike 5/3.3 degree angle widths')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--charge-mode', choices=('energy', 'electrons'), default='energy',
                        help='energy applies Modified Box; electrons uses supplied counts without requenching')
    parser.add_argument('--min-component-fraction', type=float, default=0.8)
    args = parser.parse_args()
    if args.max_particles < 1 or args.event_index < 0 or not 0 <= args.min_component_fraction <= 1:
        parser.error('Invalid particle count, event index or component fraction')
    response = LArIATResponse(**yaml.safe_load(args.config.read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    raw_images, log_images, manifest, skipped = [], [], [], []
    arrays_by_species = {key: [] for key, _ in PID.values()}
    for path in args.input:
        arrays = read_event(path, args.event_index)
        for particle in particle_groups(*arrays):
            if particle['pid'] not in PID or len(particle['points']) < 20:
                continue
            identifier = f'{path.stem}_event{args.event_index}_group{particle["group_id"]}'
            try:
                xz, yz = args.xz_deg, args.yz_deg
                if args.beam_jitter:
                    xz += rng.normal(0, 5.0)
                    yz += rng.normal(0, 3.3)
                direction = np.array([np.tan(np.deg2rad(xz)), np.tan(np.deg2rad(yz)), 1.0])
                direction /= np.linalg.norm(direction)
                points = particle['points']
                xyz, placement = place_particle(points, particle['vertex_voxels'], response,
                                                target_direction=direction)
                if args.charge_mode == 'energy':
                    # The winning-particle voxel energy is used to avoid importing
                    # another particle's energy at an overlap into a truth-isolated group.
                    electrons = ionization_electrons(points[:, 3], points[:, 7] / 10, response)
                else:
                    electrons = points[:, 6]
                deposition_ns = points[:, 5] - points[:, 5].min()
                waveforms, readout = simulate_readout(xyz, electrons, deposition_ns,
                                                     response, seed=args.seed + len(manifest))
                raw, transformed, padded, crop = prepare_model_input(waveforms, response)
                if min(c['selected_signal_fraction'] for c in crop) < args.min_component_fraction:
                    raise ValueError('Track is fragmented by threshold; selected component is too small')
                key, name = PID[particle['pid']]
                metadata = {'id': identifier, 'source_file': str(path.resolve()),
                            'h5_event_index': args.event_index if path.suffix != '.npz' else None,
                            'group_id': particle['group_id'], 'pid': particle['pid'], 'species': name,
                            'n_voxels': len(points), 'energy_mev': float(points[:, 3].sum()),
                            'charge_mode': args.charge_mode, 'placement': placement,
                            'readout': readout, 'crop': crop}
                np.savez_compressed(args.output / f'{identifier}.npz', xyz_cm=xyz,
                                    energy_mev=points[:, 3], electrons=electrons,
                                    waveforms=waveforms, padded=padded, raw=raw, log1p=transformed)
                raw_images.append(raw)
                log_images.append(transformed)
                arrays_by_species[key].append(raw)
                manifest.append(metadata)
                print(f'{name}: {identifier}, {len(points)} voxels, {points[:, 3].sum():.1f} MeV', flush=True)
            except ValueError as exc:
                skipped.append({'id': identifier, 'reason': str(exc)})
            if len(manifest) >= args.max_particles:
                break
        if len(manifest) >= args.max_particles:
            break
    report = {'response': asdict(response), 'seed': args.seed, 'beam_jitter': args.beam_jitter,
              'charge_mode': args.charge_mode, 'min_component_fraction': args.min_component_fraction,
              'interpretation': 'Approximate truth-isolated readout; not validated detector simulation',
              'particles': manifest, 'skipped': skipped}
    (args.output / 'manifest.json').write_text(json.dumps(report, indent=2))
    if not manifest:
        raise SystemExit('No particles passed conversion; see manifest.json for reasons')
    payload = {key: torch.from_numpy(np.stack(values)) for key, values in arrays_by_species.items() if values}
    payload['all'] = torch.from_numpy(np.stack(raw_images))
    torch.save(payload, args.output / 'lariat_like_raw.pt')
    torch.save({'all': torch.from_numpy(np.stack(log_images))}, args.output / 'lariat_like_log1p.pt')
    print(f'Saved {len(manifest)} particle pairs; skipped {len(skipped)}. Output: {args.output}', flush=True)


if __name__ == '__main__':
    main()
