#!/usr/bin/env python3
"""Compare energy-matched proton stopping profiles using local reconstructed hits.

Incoming beam KE stays the primary label. Range/calorimetry are separate TPC
energy estimates, not silently substituted for it. All response estimates use
fit runs only. Matching uses physical quantities, never ADC-image similarity.
"""

import argparse
from dataclasses import asdict, replace
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
import pandas as pd
from scipy.integrate import cumulative_trapezoid
from scipy.spatial import cKDTree
import uproot
import yaml

from experiments.pilarnet_lariat.calibrate import BASE, KEYS
from experiments.pilarnet_lariat.lariat_forward import (
    LArIATResponse, ionization_electrons, place_particle, simulate_readout,
    prepare_model_input)

RECO = Path('/Volumes/easystore/proton-deuteron/protons/'
            'hist_bbox_100a_RecoBBox100A_20250815T193002.root')
RANGE_TABLE = Path('/Volumes/easystore/proton-deuteron/protons.txt')
RR_EDGES = np.array([0, .5, 1, 2, 3, 5, 8, 12, 18, 25, 35, 50, 75, 100.])


def binned_profile(rr, dedx, pitch):
    """Path-weighted mean over common cm bins; no interpolation across gaps."""
    rr, dedx, pitch = map(np.asarray, (rr, dedx, pitch))
    valid = (np.isfinite(rr) & np.isfinite(dedx) & np.isfinite(pitch) &
             (rr >= 0) & (dedx > 0) & (pitch > 0))
    profile = np.full(len(RR_EDGES)-1, np.nan)
    support = np.zeros(len(profile), dtype=int)
    for i, (lo, hi) in enumerate(zip(RR_EDGES[:-1], RR_EDGES[1:])):
        mask = valid & (rr >= lo) & (rr < hi)
        support[i] = mask.sum()
        if support[i]:
            profile[i] = np.average(dedx[mask], weights=pitch[mask])
    return profile, support


@lru_cache(maxsize=4)
def range_curve(table):
    values = np.loadtxt(table)
    rr = values[-1, 0] - values[:, 0]
    order = np.argsort(rr)
    rr, dedx = rr[order], values[order, 1]
    if not np.all(np.diff(rr) > 0) or not np.all(dedx > 0):
        raise ValueError('Invalid proton residual-range table')
    return rr, cumulative_trapezoid(dedx, rr, initial=0)


def range_energy(length, table=RANGE_TABLE):
    """Diagnostic stopping-proton range energy from the project's PID table.

    Its original provenance is not recovered. Do not treat this as a precision
    upstream-material correction, or apply it to non-stopping tracks.
    """
    rr, energy = range_curve(table)
    result = np.interp(length, rr, energy)
    return np.where((np.asarray(length) >= rr[0]) & (np.asarray(length) <= rr[-1]),
                    result, np.nan)


def judge_profile(profile):
    """Judge against the existing proton Bethe-Bloch model in the same cm bins.

    A positive best residual-range offset is only a missing-tail diagnostic:
    low calorimetric calibration or non-stopping tracks can also cause it.
    It never changes a particle's geometry, dE/dx or energy label.
    """
    valid = np.isfinite(profile) & (profile > 0) & (RR_EDGES[:-1] >= .5) & (RR_EDGES[1:] <= 25)
    if valid.sum() < 4:
        return {'bb_bins': int(valid.sum()), 'bb_log_distance': np.nan,
                'bb_density_ratio': np.nan, 'bb_best_rr_offset_cm': np.nan,
                'bb_offset_log_distance': np.nan}
    shifts = np.arange(0, 20.01, .25)
    expected = np.diff(range_energy(RR_EDGES[None, :]+shifts[:, None]), axis=1)/np.diff(RR_EDGES)
    residual = np.log(np.asarray(profile)[None, valid]/expected[:, valid])
    distances = np.median(np.abs(residual), axis=1)
    best = int(np.argmin(distances))
    return {'bb_bins': int(valid.sum()), 'bb_log_distance': float(distances[0]),
            'bb_density_ratio': float(np.median(profile[valid]/expected[0, valid])),
            'bb_best_rr_offset_cm': float(shifts[best]),
            'bb_offset_log_distance': float(distances[best])}


def judge(calibration, output):
    reports = {}
    for label, manifest, profiles in (
            ('lariat', 'lariat_reco_profiles.csv', 'lariat_dedx_profiles.npy'),
            ('pilarnet', 'pilarnet_profiles.csv', 'pilarnet_dedx_profiles.npy')):
        frame = pd.read_csv(output/manifest)
        values = np.load(output/profiles)
        scores = pd.DataFrame([judge_profile(p) for p in values])
        result = pd.concat([frame, scores], axis=1)
        result.to_csv(output/f'{label}_bethe_bloch_judgement.csv', index=False)
        reports[label] = {partition: {'particles': int((result.partition == partition).sum()),
            **result[result.partition == partition][['bb_log_distance', 'bb_density_ratio',
              'bb_best_rr_offset_cm', 'bb_offset_log_distance']].median().to_dict()}
              for partition in ('fit', 'validation')}
    report = {'model': str(RANGE_TABLE),
              'sha256': hashlib.sha256(RANGE_TABLE.read_bytes()).hexdigest(),
              'model_convention': 'existing src.bethe_bloch proton table, residual range reversed',
              'judgement': 'bin-integrated dE/dx; 0.5–25 cm, at least four supported bins',
              'offset_warning': 'diagnostic only; missing endpoint, gain bias or non-stopping topology can mimic an offset',
              'results': reports}
    (output/'bethe_bloch_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def pilarnet_profiles(calibration, output):
    manifest = pd.read_csv(calibration/'pilarnet_manifest.csv')
    records, profiles, audits = [], [], []
    response = LArIATResponse()
    for record in manifest.to_dict('records'):
        with np.load(calibration/'proton_cache'/f'{record["id"]}.npz') as cached:
            points = cached['points']
        order = np.argsort(points[:, 5], kind='stable')
        p = points[order]
        dx = p[:, 7] * response.pilarnet_dx_cm_per_unit
        if np.any(dx <= 0):
            continue
        # Time orders deposits along the particle. dx sums true stored path
        # lengths; distances between voxel centres overestimate a zigzag path.
        residual = dx.sum() - np.cumsum(dx) + dx/2
        profile, _ = binned_profile(residual, p[:, 3]/dx, dx)
        supplied = points[:, 6].sum()
        q_cm = ionization_electrons(points[:, 3], points[:, 7], response).sum()
        q_mm = ionization_electrons(points[:, 3], points[:, 7]/10, response).sum()
        chord = np.linalg.norm(p[-1, :3]-p[0, :3]) * response.voxel_cm
        audits.append({'id': record['id'], 'partition': record['partition'],
                       'q_cm_over_supplied': q_cm/supplied,
                       'q_mm_over_supplied': q_mm/supplied,
                       'path_cm_over_chord': dx.sum()/max(chord, 1e-12)})
        start = np.median(p[(residual > 5) & (residual < 12), 3] /
                          dx[(residual > 5) & (residual < 12)]) if np.any((residual > 5) & (residual < 12)) else np.nan
        end = np.median(p[residual < 2, 3]/dx[residual < 2]) if np.any(residual < 2) else np.nan
        records.append({**record, 'length_cm': dx.sum(),
                        'time_order_unique_fraction': len(np.unique(p[:, 5]))/len(p),
                        'bragg_ratio': end/start,
                        'stopping_proxy': .8 < points[:, 3].sum()/record['incoming_ke_mev'] < 1.15})
        profiles.append(profile)
    frame = pd.DataFrame(records)
    frame.to_csv(output/'pilarnet_profiles.csv', index=False)
    np.save(output/'pilarnet_dedx_profiles.npy', np.asarray(profiles))
    audit = pd.DataFrame(audits)
    audit.to_csv(output/'dx_unit_audit.csv', index=False)
    fit = audit[audit.partition == 'fit']
    report = {'particles': len(audit), 'fit_particles': len(fit),
              'card_unit': 'mm', 'adopted_numeric_cm_per_dx_unit': 1.,
              'evidence': 'stored-electron and geometric-path consistency; inference contradicts card',
              'fit_quantiles': {k: fit[k].quantile([.1, .5, .9]).tolist()
                                for k in audit if k not in ('id', 'partition')}}
    (output/'dx_unit_audit.json').write_text(json.dumps(report, indent=2))
    return frame, np.asarray(profiles)


def extract_reco(source, calibration, output):
    reference = pd.read_csv(calibration/'reference_manifest.csv')
    reference['image_index'] = np.arange(len(reference))
    ambiguous = reference.duplicated(KEYS, keep=False)
    selected = reference[~ambiguous].set_index(KEYS)
    wanted = {key: row.to_dict() for key, row in selected.iterrows()}
    records, profiles, gains, widths, saved, hashes = [], [], [], [], {}, {}
    conflicts, reasons = set(), {}
    branches = KEYS + ['ntracks_reco', 'trkWCtoTPCMatch', 'trklength',
        'trkvtxx', 'trkvtxy', 'trkvtxz', 'trkendx', 'trkendy', 'trkendz',
        'trkstartdcosx', 'trkstartdcosy', 'trkstartdcosz', 'ntrkcalopts',
        'trkke', 'trkdedx', 'trkdqdx', 'trkrr', 'trkpitch', 'trkxyz',
        'efield', 'lifetime', 'hit_plane', 'hit_trkid', 'hit_charge', 'hit_rms',
        'hit_driftT', 'hit_x', 'hit_y', 'hit_z']
    tree = uproot.open(source)['anatree/anatree']
    source_entry = 0
    response = LArIATResponse()
    for arrays in tree.iterate(branches, step_size=128, library='np'):
        for event in range(len(arrays['run'])):
            entry = source_entry + event
            key = tuple(int(arrays[k][event]) for k in KEYS)
            if key not in wanted:
                continue
            if arrays['ntracks_reco'][event] != 1 or arrays['trkWCtoTPCMatch'][event][0] != 1:
                reasons['not_one_beam_matched_track'] = reasons.get('not_one_beam_matched_track', 0)+1
                continue
            n = int(arrays['ntrkcalopts'][event][0, 1])
            if not 15 <= n < 1000:
                reasons['insufficient_or_truncated_calorimetry'] = reasons.get('insufficient_or_truncated_calorimetry', 0)+1
                continue
            rr, dedx, pitch = [np.asarray(arrays[k][event][0, 1, :n], dtype=float)
                               for k in ('trkrr', 'trkdedx', 'trkpitch')]
            xyz = np.asarray(arrays['trkxyz'][event][0, 1, :n], dtype=float)
            # A few unphysical samples must not dominate a MeV integral.
            good = ((rr >= 0) & (dedx > 0) & (dedx < 100) & (pitch > .01) &
                    (pitch < 2) & np.isfinite(xyz).all(axis=1) & (xyz > -100).all(axis=1))
            if good.sum() < 15:
                continue
            rr, dedx, pitch, xyz = rr[good], dedx[good], pitch[good], xyz[good]
            signature = hashlib.sha256(np.c_[rr, dedx, pitch, xyz].tobytes()).hexdigest()
            if key in hashes:
                if hashes[key] != signature:
                    conflicts.add(key)
                continue
            hashes[key] = signature
            profile, _ = binned_profile(rr, dedx, pitch)
            start = np.array([arrays[f'trkvtx{axis}'][event][0] for axis in 'xyz'])
            end = np.array([arrays[f'trkend{axis}'][event][0] for axis in 'xyz'])
            direction = np.array([arrays[f'trkstartdcos{axis}'][event][0] for axis in 'xyz'])
            if direction[2] < 0:
                direction = -direction
            length = float(arrays['trklength'][event][0])
            coverage = float(pitch.sum()/max(length, 1e-12))
            near = dedx[rr < 2]; baseline = dedx[(rr > 5) & (rr < 12)]
            bragg = float(np.median(near)/np.median(baseline)) if len(near) >= 2 and len(baseline) >= 3 else np.nan
            contained = (2 < end[0] < 45 and abs(end[1]) < 18 and 5 < end[2] < 88)
            stopping = bool(contained and start[2] < 7 and bragg > 1.3 and .75 < coverage < 1.25
                            and rr.min() < 1 and rr.max() > .8*length)
            gain_values, width_values = [], []
            for plane in (1, 0):
                count = int(arrays['ntrkcalopts'][event][0, plane])
                if not 1 <= count < 1000:
                    gain_values.append(np.nan); width_values.append(np.nan); continue
                coordinates = np.asarray(arrays['trkxyz'][event][0, plane, :count], dtype=float)
                dd, dx = [np.asarray(arrays[k][event][0, plane, :count], dtype=float)
                          for k in ('trkdedx', 'trkpitch')]
                valid = ((dd > 0) & (dd < 100) & (dx > .01) & (dx < 2) &
                         np.isfinite(coordinates).all(axis=1) & (coordinates > -100).all(axis=1))
                if valid.sum() < 3:
                    gain_values.append(np.nan); width_values.append(np.nan); continue
                coordinates, dd, dx = coordinates[valid], dd[valid], dx[valid]
                hit_mask = (arrays['hit_plane'][event] == plane) & (arrays['hit_trkid'][event] == 0)
                hit_xyz = np.c_[tuple(arrays[f'hit_{axis}'][event][hit_mask] for axis in 'xyz')]
                if len(hit_xyz) < 3:
                    gain_values.append(np.nan); width_values.append(np.nan); continue
                distance, match = cKDTree(hit_xyz).query(coordinates)
                area = arrays['hit_charge'][event][hit_mask][match]
                sigma = arrays['hit_rms'][event][hit_mask][match]
                drift = arrays['hit_driftT'][event][hit_mask][match]*response.sample_us
                life = float(arrays['lifetime'][event])
                local_response = replace(response, field_kv_cm=float(arrays['efield'][event]))
                electrons = ionization_electrons(dd*dx, dx, local_response)
                valid_hit = ((distance < .02) & (area > 0) & (sigma > 2) & (sigma < 50) &
                             (drift >= 0) & (drift < 400) & (life > 0))
                # dE/dx already uses a charge calibration: this recovers its
                # convention, rather than independently validating that gain.
                ratio = area/electrons*np.exp(drift/max(life, 1))
                gain_values.append(float(np.median(ratio[valid_hit])) if valid_hit.sum() >= 5 else np.nan)
                width_values.append(float(np.median(sigma[valid_hit])) if valid_hit.sum() >= 5 else np.nan)
            records.append({**dict(zip(KEYS, key)), **wanted[key], 'source_entry': entry,
                'length_cm': length, 'coverage': coverage, 'calorimetric_mev': float((dedx*pitch).sum()),
                'stored_calorimetric_mev': float(arrays['trkke'][event][0, 1]),
                'range_energy_mev': float(range_energy(length)), 'bragg_ratio': bragg,
                'stopping_proxy': stopping, 'xz_deg': float(np.rad2deg(np.arctan2(direction[0], direction[2]))),
                'yz_deg': float(np.rad2deg(np.arctan2(direction[1], direction[2]))),
                'entry_x_cm': float(start[0]), 'entry_y_cm': float(start[1]), 'entry_z_cm': float(start[2]),
                'collection_area_gain': gain_values[0], 'induction_area_gain': gain_values[1],
                'collection_sigma_ticks': width_values[0], 'induction_sigma_ticks': width_values[1]})
            profiles.append(profile); gains.append(gain_values); widths.append(width_values)
            saved[key] = (rr, dedx, pitch, xyz)
        source_entry += len(arrays['run'])
        if source_entry % 1024 == 0:
            print(f'Reconstruction: {source_entry}/{tree.num_entries} entries, {len(records)} unique events', flush=True)
    keep = [i for i, r in enumerate(records) if tuple(r[k] for k in KEYS) not in conflicts]
    frame = pd.DataFrame(records).iloc[keep].reset_index(drop=True)
    profiles = np.asarray(profiles)[keep]
    frame.to_csv(output/'lariat_reco_profiles.csv', index=False)
    np.save(output/'lariat_dedx_profiles.npy', profiles)
    # Only the fit partition contributes detector response and beam directions.
    fit = frame[frame.partition == 'fit']
    area = [float(fit[f'{p}_area_gain'].median()) for p in ('collection', 'induction')]
    sigma = float(fit.collection_sigma_ticks.median())
    fitted = replace(response, collection_area_adc_ticks_per_electron=area[0],
        induction_positive_area_adc_ticks_per_electron=area[1],
        shaping_peak_us=float(1.5*sigma*response.sample_us))
    (output/'reco_response.yaml').write_text(yaml.safe_dump(asdict(fitted), sort_keys=False))
    angles = fit.loc[(fit.xz_deg.abs() < 40) & (fit.yz_deg.abs() < 40), ['xz_deg', 'yz_deg']].to_numpy()
    np.save(output/'fit_reco_angles_deg.npy', angles)
    report = {'source': str(source), 'tree': 'anatree/anatree', 'source_entries': tree.num_entries,
        'reference_rows_with_ambiguous_event_images_excluded': int(ambiguous.sum()),
        'conflicting_duplicate_reconstruction_events_excluded': len(conflicts),
        'rejected': reasons, 'unique_events': len(frame),
        'partitions': frame.partition.value_counts().to_dict(),
        'stopping_proxies': frame.groupby('partition').stopping_proxy.sum().to_dict(),
        'range_table': str(RANGE_TABLE),
        'range_table_sha256': hashlib.sha256(RANGE_TABLE.read_bytes()).hexdigest(),
        'energy_notes': 'incoming beam KE retained; range and calorimetry are separate imperfect TPC estimates',
        'response_notes': 'gain recovers existing calorimetry calibration, not independent validation; toy raw pulse',
        'response': asdict(fitted),
        'hit_dEds_excluded': 'analyser source writes it using an unmatched hit key; use track calorimetry instead'}
    (output/'reco_audit.json').write_text(json.dumps(report, indent=2))
    print(json.dumps({k: report[k] for k in ('unique_events', 'partitions', 'stopping_proxies') }), flush=True)
    return frame, profiles


def audit_endpoints(source, output):
    """Compare reconstructed hit endpoints with the full raw collection cluster."""
    geometry_path = output/'raw_cluster_geometry.csv'
    if not geometry_path.exists():
        raw = pd.read_pickle('/Volumes/easystore/proton-kaon/clusters/col.pkl')
        raw = raw[raw.particle_type == 'proton']
        raw[KEYS+['bbox_min_row', 'bbox_max_row', 'bbox_min_col', 'bbox_max_col']].to_csv(
            geometry_path, index=False)
    geometry = pd.read_csv(geometry_path).drop_duplicates(KEYS, keep=False)
    frame = pd.read_csv(output/'lariat_reco_profiles.csv')
    joined = frame.merge(geometry[KEYS+['bbox_min_row', 'bbox_max_row']], on=KEYS, validate='one_to_one')
    lookup = {int(r.source_entry): r for r in joined.itertuples()}
    tree = uproot.open(source)['anatree/anatree']
    rows, offset = [], 0
    for arrays in tree.iterate(['hit_plane', 'hit_trkid', 'hit_wire'], step_size=256, library='np'):
        for event in range(len(arrays['hit_wire'])):
            if offset+event not in lookup:
                continue
            record = lookup[offset+event]
            mask = (arrays['hit_plane'][event] == 1) & (arrays['hit_trkid'][event] == 0)
            wire = arrays['hit_wire'][event][mask]
            if not len(wire):
                continue
            rows.append({**{k: getattr(record, k) for k in KEYS},
                'reco_min_wire': int(wire.min()), 'reco_max_wire': int(wire.max()),
                'raw_min_wire': int(record.bbox_min_row), 'raw_max_wire': int(record.bbox_max_row)-1,
                'endpoint_gap_wires': int(record.bbox_max_row)-1-int(wire.max()),
                'start_gap_wires': int(wire.min())-int(record.bbox_min_row)})
        offset += len(arrays['hit_wire'])
    pd.DataFrame(rows).to_csv(output/'endpoint_audit.csv', index=False)


def match_protons(real, real_profiles, fake, fake_profiles, energy_column, physical=False):
    """Unique pairs, constrained by energy, event partition and optional physics.

    ADC images are never used to select a match. Diagnostics on selected matches
    do not establish agreement over the full unselected population.
    """
    matches = []
    for partition in ('fit', 'validation'):
        used = set()
        for si in np.flatnonzero(fake.partition.to_numpy() == partition):
            energy = float(fake.iloc[si].incoming_ke_mev)
            candidate = ((real.partition.to_numpy() == partition) &
                         (np.abs(real[energy_column].to_numpy()-energy) <= 5))
            # Compare stopping-like protons to stopping-like protons. A proton
            # that interacts and deposits only a small fraction of its KE is
            # not a valid control for a stopping track at the same initial KE.
            candidate &= real.stopping_proxy.to_numpy() & bool(fake.iloc[si].stopping_proxy)
            if physical:
                candidate &= (real.stopping_proxy.to_numpy() & bool(fake.iloc[si].stopping_proxy) &
                              (np.abs(real.length_cm.to_numpy()/fake.iloc[si].length_cm-1) <= .25))
                if 'endpoint_gap_wires' in real:
                    candidate &= real.endpoint_gap_wires.abs().to_numpy() <= 3
            options = []
            for ri in np.flatnonzero(candidate):
                if ri in used:
                    continue
                common = np.isfinite(real_profiles[ri]) & np.isfinite(fake_profiles[si])
                if common.sum() < 4:
                    continue
                shape = float(np.mean(np.abs(np.log(real_profiles[ri, common]/fake_profiles[si, common]))))
                gap = float(abs(real.iloc[ri][energy_column]-energy))
                # Incoming comparison selects closest energy, so a matching
                # procedure cannot conceal a disagreement in dE/dx or range.
                score = shape+gap/25 if physical else gap
                options.append((score, ri, gap, shape, int(common.sum())))
            if options:
                _, ri, gap, shape, support = min(options)
                used.add(ri)
                matches.append({'real_row': ri, 'pilarnet_row': si, 'partition': partition,
                    'energy_column': energy_column, 'energy_gap_mev': gap,
                    'profile_log_distance': shape, 'common_rr_bins': support,
                    'range_ratio': float(fake.iloc[si].length_cm/real.iloc[ri].length_cm)})
    return pd.DataFrame(matches)


def compare(calibration, output, max_pairs, fitted=False):
    real = pd.read_csv(output/'lariat_reco_profiles.csv')
    if (output/'endpoint_audit.csv').exists():
        endpoints = pd.read_csv(output/'endpoint_audit.csv')
        # Preserve profile-array row order explicitly when adding quality flags.
        real = real.merge(endpoints[KEYS+['endpoint_gap_wires']], on=KEYS, how='left',
                          sort=False, validate='one_to_one')
    fake = pd.read_csv(output/'pilarnet_profiles.csv')
    real_profiles = np.load(output/'lariat_dedx_profiles.npy')
    fake_profiles = np.load(output/'pilarnet_dedx_profiles.npy')
    response_path = output/('profile_response.yaml' if fitted else 'reco_response.yaml')
    response = LArIATResponse(**yaml.safe_load(response_path.read_text()))
    suffix = '_fitted' if fitted else ''
    reference = np.load(calibration/'reference_raw.npy', mmap_mode='r')
    reports = {}
    for label, energy_column, physical in (
            ('incoming', 'incoming_ke_mev', False),
            ('tpc_range', 'range_energy_mev', True),
            ('tpc_calorimetry', 'calorimetric_mev', True)):
        matches = match_protons(real, real_profiles, fake, fake_profiles, energy_column, physical)
        if matches.empty:
            reports[label] = {'matched': 0}; continue
        matches.to_csv(output/f'{label}_matches.csv', index=False)
        # A bounded pilot, including both partitions; deterministic energy order.
        chosen = pd.concat([matches[matches.partition == p].sort_values('pilarnet_row').head(max_pairs)
                            for p in ('fit', 'validation')]).reset_index(drop=True)
        real_images, fake_images, accepted, failures = [], [], [], []
        for match in chosen.to_dict('records'):
            r, s = real.iloc[match['real_row']], fake.iloc[match['pilarnet_row']]
            with np.load(calibration/'proton_cache'/f'{s.id}.npz') as cached:
                points, vertex = cached['points'], cached['vertex']
            direction = np.r_[np.tan(np.deg2rad([r.xz_deg, r.yz_deg])), 1.]
            try:
                # Conditioning on the matched real beam geometry is a diagnostic
                # control. Bulk use must instead sample the fit-only angle bank.
                xyz, _ = place_particle(points, vertex, response, target_direction=direction,
                    entry_cm=(r.entry_x_cm, r.entry_y_cm, r.entry_z_cm))
                charge = ionization_electrons(points[:, 3],
                    points[:, 7]*response.pilarnet_dx_cm_per_unit, response)
                wave, audit = simulate_readout(xyz, charge, points[:, 5]-points[:, 5].min(),
                                               response, windowed=True)
                image, _, _, crop = prepare_model_input(wave, response)
                if min(c['selected_signal_fraction'] for c in crop) < .8:
                    raise ValueError('Less than 80% of positive ADC signal in one connected track component')
                if audit['inside_volume_electrons']/charge.sum() < .95:
                    raise ValueError('More than 5% charge outside LArIAT active volume')
                observed = reference[int(r.image_index)]
                if not np.isfinite(image).all() or image.sum() <= 0:
                    raise ValueError('Invalid projected image')
                accepted.append({**match, 'pilarnet_id': s.id, 'run': int(r.run),
                    'subrun': int(r.subrun), 'event': int(r.event),
                    'real_incoming_ke_mev': float(r.incoming_ke_mev),
                    'real_matching_energy_mev': float(r[energy_column]),
                    'pilarnet_incoming_ke_mev': float(s.incoming_ke_mev),
                    'real_calorimetric_mev': float(r.calorimetric_mev),
                    'real_range_energy_mev': float(r.range_energy_mev),
                    'real_length_cm': float(r.length_cm), 'pilarnet_length_cm': float(s.length_cm),
                    'collection_component_fraction': crop[0]['selected_signal_fraction'],
                    'induction_component_fraction': crop[1]['selected_signal_fraction'],
                    **{f'{p}_peak_ratio': float(image[i].max()/max(observed[i].max(), 1e-12))
                       for i, p in enumerate(('collection', 'induction'))}})
                real_images.append(observed); fake_images.append(image)
            except ValueError as error:
                failures.append({'pilarnet_id': s.id, 'reason': str(error)})
        pairs = pd.DataFrame(accepted)
        pairs.to_csv(output/f'{label}{suffix}_image_pairs.csv', index=False)
        np.savez_compressed(output/f'{label}{suffix}_image_pairs.npz',
                            real=np.asarray(real_images), pilarnet=np.asarray(fake_images))
        reports[label] = {'matched': len(matches), 'projected': len(pairs), 'failures': failures,
            'partitions': matches.partition.value_counts().to_dict(),
            'median_energy_gap_mev': float(matches.energy_gap_mev.median()),
            'median_profile_log_distance': float(matches.profile_log_distance.median()),
            'median_range_ratio': float(matches.range_ratio.median()),
            'peak_ratios': {p: float(pairs[f'{p}_peak_ratio'].median())
                            for p in ('collection', 'induction')} if len(pairs) else {}}
        print(label, json.dumps({k: v for k, v in reports[label].items() if k != 'failures'}), flush=True)
    report = {'target': 'similar energy-conditioned stopping and ADC profiles; detector differences are allowed',
        'primary_energy': 'incoming kinetic energy', 'diagnostic_tpc_energies': ['range', 'calorimetry'],
        'energy_correction_claim': False, 'bulk_calibration_validated': False,
        'matching_warning': 'physics-selected matches are exploratory and do not validate the full population',
        'angle_warning': 'paired geometry is a diagnostic control, not independent target-domain validation',
        'comparisons': reports}
    report['response_sha256'] = hashlib.sha256(response_path.read_bytes()).hexdigest()
    (output/f'profile_comparison_report{suffix}.json').write_text(json.dumps(report, indent=2))


def profile_error(real, simulated):
    """Absolute row-maximum discrepancy, normalized by observed profile charge."""
    return np.abs(real-simulated).sum(axis=-1)/np.maximum(real.sum(axis=-1), 1e-12)


def fit_adc_profiles(output):
    """Fit two global amplitude factors; preserve all particle shapes and lengths."""
    pairs = pd.read_csv(output/'tpc_range_image_pairs.csv')
    with np.load(output/'tpc_range_image_pairs.npz') as images:
        real = images['real'].max(axis=3)
        fake = images['pilarnet'].max(axis=3)
    fit = pairs.partition.to_numpy() == 'fit'
    if fit.sum() < 20:
        raise ValueError('At least 20 fit pairs are required for the global ADC-profile correction')
    scales = np.linspace(.5, 2.5, 81)
    multipliers, losses = [], []
    for plane in range(2):
        objective = [float(np.median(profile_error(real[fit, plane], fake[fit, plane]*s))) for s in scales]
        best = int(np.argmin(objective))
        multipliers.append(float(scales[best])); losses.append(objective[best])
    prior = LArIATResponse(**yaml.safe_load((output/'reco_response.yaml').read_text()))
    fitted = replace(prior,
        collection_area_adc_ticks_per_electron=prior.collection_area_adc_ticks_per_electron*multipliers[0],
        induction_positive_area_adc_ticks_per_electron=prior.induction_positive_area_adc_ticks_per_electron*multipliers[1])
    path = output/'profile_response.yaml'
    path.write_text(yaml.safe_dump(asdict(fitted), sort_keys=False))
    report = {'energy_for_fit': 'TPC stopping-energy estimate from the existing proton range model',
        'incoming_beam_energy_retained': True, 'fit_pairs': int(fit.sum()),
        'effective_raw_projection_amplitude_multipliers': multipliers,
        'fitted_parameters': 'two global plane amplitudes; no length, energy, dE/dx or pixel warping',
        'note': 'effective raw-image response correction, not a new measurement of electronics gain',
        'fit_profile_error': losses, 'response_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'validation': 'rerun compare --fitted; linear scaling alone does not reproduce threshold/crop changes'}
    (output/'adc_fit_report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def validate_adc_profiles(calibration, output):
    """Compare held-out proton-profile discrepancies with real-to-real variation."""
    pairs = pd.read_csv(output/'tpc_range_fitted_image_pairs.csv')
    with np.load(output/'tpc_range_fitted_image_pairs.npz') as images:
        observed = images['real'].max(axis=3)
        simulated = images['pilarnet'].max(axis=3)
    real = pd.read_csv(output/'lariat_reco_profiles.csv')
    endpoints = pd.read_csv(output/'endpoint_audit.csv')
    real = real.merge(endpoints[KEYS+['endpoint_gap_wires']], on=KEYS, how='left', validate='one_to_one')
    references = np.load(calibration/'reference_raw.npy', mmap_mode='r')
    errors, intrinsic = [], []
    for i, pair in pairs.iterrows():
        if pair.partition != 'validation':
            continue
        r = real.iloc[int(pair.real_row)]
        error = profile_error(observed[i], simulated[i])
        neighbors = real[(real.partition == 'validation') & real.stopping_proxy &
            (real.endpoint_gap_wires.abs() <= 3) & (np.abs(real.range_energy_mev-r.range_energy_mev) <= 5) &
            (np.abs(real.length_cm/r.length_cm-1) < .25) & (np.abs(real.xz_deg-r.xz_deg) <= 3) &
            (np.abs(real.yz_deg-r.yz_deg) <= 5) & (real.index != int(pair.real_row))]
        if neighbors.empty:
            continue
        variations = [profile_error(observed[i], references[int(n.image_index)].max(axis=2))
                      for n in neighbors.itertuples()]
        errors.append(error); intrinsic.append(np.median(variations, axis=0))
    errors, intrinsic = np.asarray(errors), np.asarray(intrinsic)
    report = {'comparison': 'held-out ADC row maxima versus nearby real stopping-proton profiles',
        'heldout_projected_pairs': int((pairs.partition == 'validation').sum()),
        'pairs_with_real_neighbors': len(errors),
        'energy_reference': 'TPC range estimate, not equality of upstream beam KE',
        'geometry': 'matched real initial direction and vertex; a controlled diagnostic, not independent geometry validation',
        'bulk_calibration_validated': False,
        'selection_warning': 'stopping and endpoint quality cuts limit applicability; no full-population claim'}
    if len(errors):
        report.update(simulation_error_median= np.median(errors, axis=0).tolist(),
                      real_variation_median=np.median(intrinsic, axis=0).tolist(),
                      simulation_error_quantiles=np.quantile(errors, [.16, .5, .84], axis=0).tolist(),
                      relative_to_real_variation=(np.median(errors, axis=0)/np.median(intrinsic, axis=0)).tolist())
    (output/'adc_profile_validation.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['prepare', 'compare', 'judge', 'fit-adc', 'validate-adc'])
    parser.add_argument('--source', type=Path, default=RECO)
    parser.add_argument('--calibration', type=Path, default=BASE/'calibration')
    parser.add_argument('--output', type=Path, default=BASE/'profile_matching')
    parser.add_argument('--max-pairs', type=int, default=32, help='Per partition and energy definition')
    parser.add_argument('--fitted', action='store_true', help='Compare using the fit-only global ADC-profile correction')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.max_pairs < 1:
        parser.error('max-pairs must be positive')
    if args.operation == 'prepare':
        pilarnet_profiles(args.calibration, args.output)
        extract_reco(args.source, args.calibration, args.output)
        audit_endpoints(args.source, args.output)
    elif args.operation == 'compare':
        compare(args.calibration, args.output, args.max_pairs, args.fitted)
    elif args.operation == 'judge':
        judge(args.calibration, args.output)
    elif args.operation == 'fit-adc':
        fit_adc_profiles(args.output)
    else:
        validate_adc_profiles(args.calibration, args.output)


if __name__ == '__main__':
    main()
