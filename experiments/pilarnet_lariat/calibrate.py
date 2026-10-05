#!/usr/bin/env python3
"""Fit one proton response in incoming-KE bins; reserve events/runs for checks."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import wasserstein_distance
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
import torch
import yaml

from experiments.pilarnet_lariat.lariat_forward import (
    LArIATResponse, particle_groups, place_particle, ionization_electrons,
    simulate_readout, prepare_model_input)

BASE = Path('/Volumes/easystore/proton-kaon/pilarnet_lariat')
KEYS = ['run', 'subrun', 'event']
ENERGY_EDGES = np.array([100, 150, 200, 250, 300, 400, 550, 800.])
FEATURES = [f'{plane}_{name}' for plane in ('collection', 'induction') for name in
            ('log_median', 'log_q90', 'log_max', 'log_sum', 'occupancy',
             'wire_extent', 'time_spread', 'last_quarter_fraction')]


def kinetic_energy(momentum_mev, mass_mev):
    p, m = np.asarray(momentum_mev), np.asarray(mass_mev)
    if np.any(p < 0) or np.any(m < 0) or not np.isfinite(p).all() or not np.isfinite(m).all():
        raise ValueError('Mass/momentum must be nonnegative and finite')
    # Rationalized expression avoids cancellation for slow particles.
    denominator = np.hypot(p, m) + m
    return np.divide(p*p, denominator, out=np.zeros(np.broadcast_shapes(p.shape,m.shape)),
                     where=denominator>0)


def summaries(images):
    result = []
    coordinates = np.arange(48)
    for pair in images:
        row = []
        for image in pair:
            positive = image[image > 0]
            total = float(image.sum())
            time_profile, wire_profile = image.sum(axis=0), image.sum(axis=1)
            centroid = float(time_profile @ coordinates / max(total, 1e-12))
            spread = np.sqrt(time_profile @ (coordinates-centroid)**2 / max(total, 1e-12))
            active = np.flatnonzero(wire_profile > 0)
            row.extend([np.log1p(np.median(positive)) if len(positive) else 0,
                        np.log1p(np.quantile(positive, .9)) if len(positive) else 0,
                        np.log1p(image.max()), np.log1p(total),
                        float(np.mean(image > 0)),
                        float(active[-1]-active[0]+1) / 48 if len(active) else 0,
                        float(spread), float(wire_profile[36:].sum()/max(total, 1e-12))])
        result.append(row)
    return np.asarray(result)


def endpoint_angles(images, response):
    """Stereo tangent estimate from the first 20 rows of the endpoint crop.

    This is an effective endpoint-angle prior, not a measured beamline direction.
    The crop cannot distinguish initial tilt from scattering accumulated upstream.
    """
    result = []
    scale = (1502/48) / (51/48)
    factor = response.wire_pitch_cm / (response.drift_cm_us * response.sample_us)
    for pair in images:
        slopes = []
        for image in pair:
            weights = image.sum(axis=1)
            valid = np.flatnonzero(weights[:20] > 0)
            if len(valid) < 8:
                break
            centres = (image[valid] @ np.arange(48)) / weights[valid]
            slopes.append(np.polyfit(valid, centres, 1, w=np.sqrt(weights[valid]))[0] * scale)
        if len(slopes) != 2 or abs(sum(slopes)) < .5 or slopes[0]*slopes[1] <= 0:
            continue
        c, i = slopes
        dx_dz = np.sqrt(3)*c*i/(factor*(c+i))
        dy_dz = np.sqrt(3)*(i-c)/(c+i)
        angles = np.rad2deg(np.arctan([dx_dz, dy_dz]))
        if np.all(np.abs(angles) < 40):
            result.append(angles)
    if len(result) < 20:
        raise ValueError('Too few consistent stereo tangents to estimate the geometry prior')
    return np.asarray(result)


def prepare_reference(output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_pickle('/Volumes/easystore/proton-kaon/features/features.pkl')
    metadata = metadata[metadata.particle_type == 'proton'].reset_index(drop=True)
    momentum = pd.read_csv('/Volumes/easystore/proton-deuteron/momentum_tof.csv')
    if momentum.duplicated(KEYS).any():
        raise ValueError('Beamline momentum keys are not unique')
    metadata = metadata.merge(momentum[KEYS+['momentum']], on=KEYS, how='left', validate='many_to_one')
    images = torch.load('/Volumes/easystore/proton-kaon/images/pkm_48x48_raw_10-179wires.pt',
                        weights_only=True, map_location='cpu')['p'].numpy()
    if len(images) != len(metadata) or images.shape[1:] != (2, 48, 48):
        raise ValueError('Proton tensor and metadata do not have matching shapes')
    if not np.isfinite(images).all() or np.any(images < 0):
        raise ValueError('Invalid raw reference images')
    metadata['incoming_ke_mev'] = kinetic_energy(metadata.momentum.to_numpy(), 938.272)
    metadata['image_sha256'] = [hashlib.sha256(image.tobytes()).hexdigest() for image in images]
    # Keep duplicate images in a single partition, even if metadata gives another run.
    rng = np.random.default_rng(20261005)
    runs = np.sort(metadata.run.unique())
    held_runs = rng.permutation(runs)[:max(1, round(.2*len(runs)))]
    held_hashes = set(metadata.loc[metadata.run.isin(held_runs), 'image_sha256'])
    metadata['partition'] = np.where(metadata.run.isin(held_runs) |
                                     metadata.image_sha256.isin(held_hashes), 'validation', 'fit')
    assert metadata.groupby(KEYS).partition.nunique().max() == 1
    assert metadata.groupby('image_sha256').partition.nunique().max() == 1
    assert not metadata.momentum.isna().any()
    np.save(output/'reference_raw.npy', images)
    np.save(output/'reference_features.npy', summaries(images))
    metadata[KEYS+['height','momentum','incoming_ke_mev','image_sha256','partition']].to_csv(
        output/'reference_manifest.csv', index=False)
    response = LArIATResponse()
    angles = endpoint_angles(images[metadata.partition == 'fit'], response)
    np.save(output/'fit_endpoint_angles_deg.npy', angles)
    report = {'energy_definition': 'incoming kinetic energy from beamline momentum; no upstream-loss correction',
              'mass_mev': 938.272, 'momentum_unit': 'MeV/c', 'features': FEATURES,
              'counts': metadata.partition.value_counts().to_dict(), 'held_runs': held_runs.tolist(),
              'energy_quantiles_mev': np.quantile(metadata.incoming_ke_mev,[0,.1,.5,.9,1]).tolist(),
              'angle_prior': 'effective stereo endpoint tangent; does not recover true entrance angle',
              'angle_count': len(angles), 'angle_median_deg': np.median(angles,axis=0).tolist(),
              'angle_width_deg': np.std(angles,axis=0).tolist(),
              'alignment': 'proton tensor order matched to existing feature-making proton order; equal row count'}
    (output/'reference_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def cache_particles(source, output, max_particles, momentum_scale):
    cache = output/'proton_cache'; cache.mkdir(parents=True,exist_ok=True)
    records, counts = [], {}
    with h5py.File(source,'r') as data:
        for event in range(len(data['cluster'])):
            arrays = tuple(np.asarray(data[k][event]).reshape(-1,w)
                           for k,w in [('point',8),('cluster',6),('cluster_extra',5)])
            for particle in particle_groups(*arrays):
                if particle['pid'] != 4 or len(particle['points']) < 20:
                    continue
                try:
                    place_particle(particle['points'],particle['vertex_voxels'],LArIATResponse())
                    energy = float(kinetic_energy(particle['momentum']*momentum_scale,particle['mass_mev']))
                    if not ENERGY_EDGES[0] <= energy < ENERGY_EDGES[-1]:
                        continue
                    points = particle['points']
                    if np.any(points[:,7] <= 0):
                        raise ValueError('Nonpositive voxel path length')
                    identifier = f'{source.stem}_event{event}_group{particle["group_id"]}'
                    # Every group of this event has the same deterministic partition.
                    partition = 'validation' if event % 5 == 0 else 'fit'
                    np.savez_compressed(cache/f'{identifier}.npz',points=points,vertex=particle['vertex_voxels'])
                    records.append({'id':identifier,'event_index':event,'group_id':particle['group_id'],
                                    'incoming_ke_mev':energy,'momentum':particle['momentum'],
                                    'mass_mev':particle['mass_mev'],'partition':partition,
                                    'deposited_mev':float(points[:,3].sum())})
                except ValueError as error:
                    counts[str(error)] = counts.get(str(error),0)+1
            if len(records) >= max_particles:
                break
    if len(records) < 100:
        raise ValueError(f'Only {len(records)} usable protons; need at least 100')
    pd.DataFrame(records).to_csv(output/'pilarnet_manifest.csv',index=False)
    (output/'cache_report.json').write_text(json.dumps({'source':str(source),'scanned_events':event+1,
        'particles':len(records),'momentum_to_mev_scale':momentum_scale,
        'momentum_unit_status':'explicit GeV/c -> MeV/c convention, checked against deposited energy',
        'deposition_exceeds_ke_count':sum(r['deposited_mev']>1.05*r['incoming_ke_mev'] for r in records),
        'rejected':counts},indent=2))
    return records


def generate(particles, angles, response, output, save=False):
    images, accepted, failures = [], [], {}
    for record in particles:
        with np.load(output/'proton_cache'/f'{record["id"]}.npz') as data:
            points, vertex = data['points'], data['vertex']
        stable = int(hashlib.sha256(record['id'].encode()).hexdigest()[:8],16)
        xz,yz = angles[stable % len(angles)]
        direction = np.r_[np.tan(np.deg2rad([xz,yz])),1.]
        direction /= np.linalg.norm(direction)
        try:
            xyz,_ = place_particle(points,vertex,response,target_direction=direction)
            electrons = ionization_electrons(points[:,3],points[:,7]/10,response)
            wave,audit = simulate_readout(xyz,electrons,points[:,5]-points[:,5].min(),response,
                                          windowed=True,seed=stable)
            raw,_,_,crop = prepare_model_input(wave,response)
            if any(not 10 < c['component_shape'][0] < 179 or c['component_shape'][1] >= 1500 for c in crop):
                raise ValueError('Same image-size cuts as the proton reference')
            if min(c['selected_signal_fraction'] for c in crop) < .8:
                raise ValueError('Fragmented positive track')
            images.append(raw); accepted.append(record)
        except ValueError as error:
            failures[str(error)] = failures.get(str(error),0)+1
    if not images:
        return np.empty((0,2,48,48),np.float32),[],failures
    return np.stack(images),accepted,failures


def conditional_distance(reference, reference_energy, simulated, simulated_energy):
    """Quantile distance within fixed kinetic-energy bins; no energy resampling fit."""
    terms, details = [], []
    for lo,hi in zip(ENERGY_EDGES[:-1],ENERGY_EDGES[1:]):
        real = reference[(reference_energy>=lo)&(reference_energy<hi)]
        fake = simulated[(simulated_energy>=lo)&(simulated_energy<hi)]
        if len(real)<20 or len(fake)<5:
            details.append({'lo':lo,'hi':hi,'real':len(real),'simulated':len(fake),'supported':False})
            continue
        scale = np.maximum(np.quantile(real,.75,axis=0)-np.quantile(real,.25,axis=0),.05)
        distances = np.array([wasserstein_distance(real[:,i],fake[:,i])/scale[i] for i in range(real.shape[1])])
        terms.append(distances.mean())
        details.append({'lo':lo,'hi':hi,'real':len(real),'simulated':len(fake),'supported':True,
                        'mean_standardized_wasserstein':float(distances.mean()),
                        'feature_distances':dict(zip(FEATURES,distances.tolist()))})
    return (float(np.mean(terms)) if terms else 1e6), details


def fit_response(source, output, max_particles, momentum_scale, max_evaluations, resume_fit=False):
    reference = pd.read_csv(output/'reference_manifest.csv')
    features = np.load(output/'reference_features.npy')
    angles = np.load(output/'fit_endpoint_angles_deg.npy')
    manifest_path = output/'pilarnet_manifest.csv'
    particles = pd.read_csv(manifest_path).to_dict('records') if manifest_path.exists() else cache_particles(
        source,output,max_particles,momentum_scale)
    fit_particles = [p for p in particles if p['partition']=='fit']
    real_fit = reference.partition == 'fit'
    base = LArIATResponse()
    history = json.loads((output/'fit_history.json').read_text()) if resume_fit else []

    def response_from(parameters):
        return replace(base,collection_response_scale=float(base.collection_response_scale*np.exp(parameters[0])),
                       induction_response_scale=float(base.induction_response_scale*np.exp(parameters[1])),
                       shaping_peak_us=float(parameters[2]))

    def objective(parameters):
        images, accepted, failed = generate(fit_particles,angles,response_from(parameters),output)
        energy = np.array([p['incoming_ke_mev'] for p in accepted])
        metric,bins = conditional_distance(features[real_fit],reference.incoming_ke_mev.to_numpy()[real_fit],
                                           summaries(images),energy)
        # Guard against fitting by making inconvenient particles disappear.
        acceptance_penalty = 2*(1-len(accepted)/len(fit_particles))
        score = metric+acceptance_penalty
        history.append({'parameters':np.asarray(parameters).tolist(),'distance':metric,'objective':score,
                        'accepted':len(accepted),'attempted':len(fit_particles),'failed':failed})
        (output/'fit_history.json').write_text(json.dumps(history,indent=2))
        print(f'Fit trial {len(history)}: distance={metric:.3f}, accepted={len(accepted)}/{len(fit_particles)}',flush=True)
        return score

    if not resume_fit:
        initial = np.array([0.,0.,3.])
        objective(initial)
        minimize(objective,np.array([np.log(2),np.log(2),6.]),method='Powell',
                 bounds=[(np.log(.25),np.log(12)),(np.log(.25),np.log(12)),(1.,15.)],
                 options={'maxfev':max_evaluations,'xtol':.05,'ftol':.02})
    best = min(history,key=lambda r:r['objective'])
    frozen = response_from(best['parameters'])
    config = output/'fitted_response.yaml'
    config.write_text(yaml.safe_dump(asdict(frozen),sort_keys=False))
    digest = hashlib.sha256(config.read_bytes()).hexdigest()
    validation = {}
    classifier_data = {}
    for partition in ('fit','validation'):
        rows = [p for p in particles if p['partition']==partition]
        images,accepted,failed = generate(rows,angles,frozen,output)
        np.save(output/f'{partition}_pilarnet_raw.npy',images)
        pd.DataFrame(accepted).to_csv(output/f'{partition}_accepted.csv',index=False)
        real_mask = reference.partition == partition
        distance,bins = conditional_distance(features[real_mask],reference.incoming_ke_mev.to_numpy()[real_mask],
                                             summaries(images),np.array([p['incoming_ke_mev'] for p in accepted]))
        validation[partition] = {'distance':distance,'bins':bins,'attempted':len(rows),'accepted':len(accepted),'failed':failed}
        # Match incoming-KE bin populations before asking a domain classifier.
        matched_real,matched_fake = [],[]
        sim_features = summaries(images)
        sim_energy = np.array([p['incoming_ke_mev'] for p in accepted])
        real_energy = reference.incoming_ke_mev.to_numpy()[real_mask]
        real_features = features[real_mask]
        rng = np.random.default_rng(42)
        for lo,hi in zip(ENERGY_EDGES[:-1],ENERGY_EDGES[1:]):
            ri=np.flatnonzero((real_energy>=lo)&(real_energy<hi));si=np.flatnonzero((sim_energy>=lo)&(sim_energy<hi))
            n=min(len(ri),len(si))
            if n:
                matched_real.extend(real_features[rng.choice(ri,n,replace=False)])
                matched_fake.extend(sim_features[rng.choice(si,n,replace=False)])
        classifier_data[partition]=(np.r_[matched_real,matched_fake],np.r_[np.zeros(len(matched_real)),np.ones(len(matched_fake))])
    if all(len(y)>=20 for _,y in classifier_data.values()):
        model=HistGradientBoostingClassifier(max_iter=80,max_leaf_nodes=8,min_samples_leaf=10,random_state=42)
        model.fit(*classifier_data['fit'])
        x,y=classifier_data['validation']
        auc=float(roc_auc_score(y,model.predict_proba(x)[:,1]))
    else:
        auc=None
    report={'status':'pilot fit complete; held-out discrepancies must be inspected',
            'energy_definition':'incoming kinetic energy; beamline momentum vs simulated generator momentum',
            'upstream_loss_correction':'not available; this difference can affect conditional track lengths',
            'fitted_parameters':['effective collection amplitude','effective induction amplitude','shaping peak'],
            'angle_prior':'empirical stereo endpoint tangents from fit-only LArIAT images',
            'response_sha256':digest,'best_trial':best,'baseline_trial':history[0],
            'partitions':validation,'heldout_domain_classifier_auc':auc,
            'auc_interpretation':'0.5 is chance; high AUC means a measurable remaining domain difference',
            'identical_images_claim':False,'bulk_rule':'freeze this response and fit-only angle bank; no per-species refitting'}
    (output/'calibration_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'baseline':history[0]['distance'],'fit':best['distance'],'validation':validation['validation']['distance'],'auc':auc}),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation',choices=['prepare','fit'])
    parser.add_argument('--output',type=Path,default=BASE/'calibration')
    parser.add_argument('--input',type=Path,default=BASE/'full/train/generic_v2_51800_v2.h5')
    parser.add_argument('--max-particles',type=int,default=256)
    parser.add_argument('--momentum-scale',type=float,default=1000.,help='PILArNet p is treated as GeV/c; mass is MeV/c²')
    parser.add_argument('--max-evaluations',type=int,default=48)
    parser.add_argument('--wait-for-input',action='store_true',help='Wait for a verified download to be renamed into place')
    parser.add_argument('--resume-fit',action='store_true',help='Reuse completed fit history and rerun export/validation only')
    args=parser.parse_args()
    if args.operation=='prepare':prepare_reference(args.output)
    else:
        if args.wait_for_input:
            while not args.input.exists():
                time.sleep(15)
        fit_response(args.input,args.output,args.max_particles,args.momentum_scale,args.max_evaluations,args.resume_fit)


if __name__=='__main__':main()
