#!/usr/bin/env python3
"""Build an inline, step-by-step visualization of an actual PILArNet proton."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
import yaml

from experiments.pilarnet_lariat.calibrate import BASE
from experiments.pilarnet_lariat.lariat_forward import (
    LArIATResponse, ionization_electrons, place_particle, prepare_model_input,
    simulate_readout)


def heatmap(values, max_rows=140, max_cols=260, decimals=4):
    """Mean blocks for display only; exact arrays determine the final images."""
    values = np.asarray(values)
    nr, nc = values.shape
    sy, sx = max(1, int(np.ceil(nr/max_rows))), max(1, int(np.ceil(nc/max_cols)))
    padded = np.pad(values, ((0, (-nr) % sy), (0, (-nc) % sx)))
    summed = padded.reshape(padded.shape[0]//sy, sy, padded.shape[1]//sx, sx).sum(axis=(1, 3))
    counts = np.pad(np.ones_like(values), ((0, (-nr) % sy), (0, (-nc) % sx)))
    counts = counts.reshape(padded.shape[0]//sy, sy, padded.shape[1]//sx, sx).sum(axis=(1, 3))
    display = summed/counts
    encoded = display if decimals is None else np.round(display, decimals)
    return {'shape': list(display.shape), 'native_shape': [nr, nc],
            'values': encoded.ravel().tolist()}


def build(calibration, profiles, destination):
    table = pd.read_csv(profiles/'tpc_range_fitted_image_pairs.csv')
    candidates = table[(table.partition == 'validation') &
        (table.collection_component_fraction > .98) & (table.induction_component_fraction > .98)]
    # A representative clean held-out particle, never the best-looking ADC pair.
    record = candidates.sort_values('profile_log_distance').iloc[len(candidates)//2]
    real = pd.read_csv(profiles/'lariat_reco_profiles.csv').iloc[int(record.real_row)]
    source = calibration/'proton_cache'/f'{record.pilarnet_id}.npz'
    response_path = profiles/'profile_response.yaml'
    response = LArIATResponse(**yaml.safe_load(response_path.read_text()))
    with np.load(source) as cached:
        points, vertex = cached['points'], cached['vertex']
    order = np.argsort(points[:, 5], kind='stable')
    points = points[order]
    direction = np.r_[np.tan(np.deg2rad([real.xz_deg, real.yz_deg])), 1.]
    placed, placement = place_particle(points, vertex, response, target_direction=direction,
        entry_cm=(real.entry_x_cm, real.entry_y_cm, real.entry_z_cm))
    dx = points[:, 7]*response.pilarnet_dx_cm_per_unit
    charge = ionization_electrons(points[:, 3], dx, response)
    trace = {}
    waveform, audit = simulate_readout(placed, charge, points[:, 5]-points[:, 5].min(),
                                      response, windowed=True, trace=trace)
    raw, transformed, padded, crops = prepare_model_input(waveform, response)
    selected, retained = [], []
    for plane, threshold in enumerate((response.collection_threshold_adc, response.induction_threshold_adc)):
        regions = regionprops(label(waveform[plane] > threshold), intensity_image=waveform[plane])
        region = max(regions, key=lambda r: r.image_intensity.sum())
        selected.append(region.image_intensity)
        retained.append(region.image_intensity[-50:])
    # Check the live replay against the previously stored pilot's exact images.
    with np.load(profiles/'tpc_range_fitted_image_pairs.npz') as saved:
        expected = saved['pilarnet'][int(record.name)]
        observed = saved['real'][int(record.name)]
    np.testing.assert_allclose(raw, expected, rtol=3e-5, atol=2e-3)
    payload = {
        'id': record.pilarnet_id, 'partition': 'validation',
        'revision': 'f32f36bd1c17d707d0a24f0c63ec16419475c20f',
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'response_sha256': hashlib.sha256(response_path.read_bytes()).hexdigest(),
        'incoming_mev': float(record.pilarnet_incoming_ke_mev),
        'deposited_mev': float(points[:, 3].sum()), 'path_cm': float(dx.sum()),
        'count': len(points), 'electrons': float(charge.sum()),
        'after_drift_electrons': audit['after_lifetime_electrons'],
        'inside_electrons': audit['inside_volume_electrons'],
        'raw_xyz': np.round(points[:, :3], 4).tolist(),
        'placed_xyz': np.round(placed, 4).tolist(),
        'entry_xyz': placement['entry_cm'],
        'energy': np.round(points[:, 3], 4).tolist(),
        'charge': np.round(charge, 1).tolist(),
        'dx_cm': np.round(dx, 5).tolist(),
        'histograms': [heatmap(a) for a in trace['pre_loss_histograms']],
        'diffused': [heatmap(a) for a in trace['diffused']],
        'waveforms': [heatmap(a) for a in waveform],
        'selected': [heatmap(a) for a in selected],
        'retained': [heatmap(a) for a in retained],
        'padded': [heatmap(a, max_cols=376) for a in padded],
        'raw48': [heatmap(a, decimals=None) for a in raw],
        'log48': [heatmap(a, decimals=None) for a in transformed],
        'real48': [heatmap(a, decimals=None) for a in observed],
        'window_origin': audit['window_origin_wire_tick'],
        'window_shape': audit['window_shape'],
        'crops': crops, 'response': asdict(response),
        'real': {'run': int(record.run), 'event': int(record.event),
                 'beam_ke_mev': float(record.real_incoming_ke_mev),
                 'range_mev': float(record.real_range_energy_mev)},
        'display_reduction': 'Intermediate heatmaps use mean blocks; final 48×48 images are exact.'}
    template = Path(__file__).with_name('walkthrough.template.html').read_text()
    fragment = template.replace('__PROTON_DATA__', json.dumps(payload, separators=(',', ':')))
    if len(fragment.encode()) >= 1_000_000:
        raise ValueError('Visualization exceeds inline size limit')
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(fragment)
    print(json.dumps({'output': str(destination), 'bytes': len(fragment.encode()),
                      'id': payload['id'], 'voxels': len(points),
                      'incoming_mev': payload['incoming_mev'], 'deposited_mev': payload['deposited_mev'],
                      'path_cm': payload['path_cm'], 'replay_verified': True}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, default=BASE/'calibration')
    parser.add_argument('--profiles', type=Path, default=BASE/'profile_matching')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    build(args.calibration, args.profiles, args.output)
