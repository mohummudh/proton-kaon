#!/usr/bin/env python3
"""Stream every verified PILArNet file through one frozen calibrated response."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys
import threading
import time

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import h5py
import numpy as np
import yaml

from experiments.pilarnet_lariat.lariat_forward import (
    LArIATResponse, particle_groups, place_particle, ionization_electrons,
    simulate_readout, prepare_model_input)


def kinetic_energy(momentum_mev,mass_mev):
    if not np.isfinite([momentum_mev,mass_mev]).all() or momentum_mev<0 or mass_mev<0:
        raise ValueError('Invalid mass/momentum metadata')
    denominator=np.hypot(momentum_mev,mass_mev)+mass_mev
    return momentum_mev**2/denominator if denominator>0 else 0.


def convert_file(source,relative,output,response,angles,response_sha,events_per_part,max_events=None,
                 angle_sha='',supported_energy_ranges=((100,800),)):
    destination=output/Path(relative).parent/Path(relative).stem
    destination.mkdir(parents=True,exist_ok=True)
    counts={'events':0,'accepted':0,'rejected':0,'LED_groups':0,'by_pid':{},'reasons':{}}
    with h5py.File(source,'r') as data:
        events=min(len(data['point']),max_events) if max_events is not None else len(data['point'])
        for start in range(0,events,events_per_part):
            stop=min(events,start+events_per_part)
            final=destination/f'events_{start:07d}_{stop:07d}.h5'
            if final.exists():
                with h5py.File(final,'r') as completed:
                    if completed.attrs['response_sha256'] != response_sha:
                        raise RuntimeError('Existing output uses a different response; choose a new output folder')
                    if completed.attrs['angle_bank_sha256'] != angle_sha:
                        raise RuntimeError('Existing output uses a different angle bank')
                    part_counts=json.loads(completed.attrs['counts'])
            else:
                temporary=Path(str(final)+'.partial')
                part_counts={'events':stop-start,'accepted':0,'rejected':0,'LED_groups':0,'by_pid':{},'reasons':{}}
                with h5py.File(temporary,'w') as converted:
                    converted.attrs.update(source_file=relative,response_sha256=response_sha,angle_bank_sha256=angle_sha,
                        response_json=json.dumps(response.__dict__),
                        energy_definition='generator incoming kinetic energy; momentum treated as GeV/c, mass as MeV/c²',
                        axes='collection/induction, wire-axis pixel, time-axis pixel',
                        placement='fit-only endpoint-angle prior; displaced/short direction estimates are flagged',
                        interpretation='approximate transfer; inspect response provenance and held-out calibration report')
                    images=converted.create_dataset('raw',shape=(0,2,48,48),maxshape=(None,2,48,48),
                        chunks=(64,2,48,48),dtype='f4',compression='gzip',compression_opts=4,shuffle=True)
                    fields={'event_index':'u4','group_id':'i4','pid':'u1','incoming_ke_mev':'f4',
                            'deposited_mev':'f4','retained_volume_charge_fraction':'f4',
                            'collection_component_fraction':'f4','induction_component_fraction':'f4',
                            'approximate_direction':'u1','outside_proton_energy_range':'u1'}
                    metadata={name:converted.create_dataset(name,shape=(0,),maxshape=(None,),
                              chunks=(1024,),dtype=dtype,compression='gzip') for name,dtype in fields.items()}
                    rejected_type=np.dtype([('event_index','u4'),('group_id','i4'),('pid','i2'),('reason','S256')])
                    rejected=converted.create_dataset('rejected',shape=(0,),maxshape=(None,),chunks=(1024,),
                                                      dtype=rejected_type,compression='gzip')
                    buffer_images,buffer_records,buffer_rejected=[],[],[]

                    def flush():
                        if buffer_images:
                            first=len(images);last=first+len(buffer_images)
                            images.resize(last,axis=0);images[first:last]=np.stack(buffer_images)
                            for name,dataset in metadata.items():
                                dataset.resize(last,axis=0);dataset[first:last]=[r[name] for r in buffer_records]
                            buffer_images.clear();buffer_records.clear()
                        if buffer_rejected:
                            first=len(rejected);last=first+len(buffer_rejected)
                            rejected.resize(last,axis=0);rejected[first:last]=np.array(buffer_rejected,dtype=rejected_type)
                            buffer_rejected.clear()

                    for event in range(start,stop):
                        arrays=tuple(np.asarray(data[k][event]).reshape(-1,w)
                                     for k,w in [('point',8),('cluster',6),('cluster_extra',5)])
                        part_counts['LED_groups'] += len(np.unique(arrays[1][arrays[1][:,2]<0,2]))
                        for particle in particle_groups(*arrays):
                            identifier=f'{relative}:event{event}:group{particle["group_id"]}'
                            stable=int(hashlib.sha256(identifier.encode()).hexdigest()[:8],16)
                            try:
                                if not 0<=particle['pid']<=5:
                                    raise ValueError('Unsupported PID')
                                points=particle['points']
                                xz,yz=angles[stable%len(angles)]
                                direction=np.r_[np.tan(np.deg2rad([xz,yz])),1.]
                                direction/=np.linalg.norm(direction)
                                xyz,placement=place_particle(points,particle['vertex_voxels'],response,
                                    target_direction=direction,allow_displaced=True)
                                charge=ionization_electrons(points[:,3],points[:,7]/10,response)
                                wave,audit=simulate_readout(xyz,charge,points[:,5]-points[:,5].min(),response,
                                                            seed=stable,windowed=True)
                                raw,_,_,crop=prepare_model_input(wave,response)
                                energy=float(kinetic_energy(particle['momentum']*1000,particle['mass_mev']))
                                record={'event_index':event,'group_id':particle['group_id'],'pid':particle['pid'],
                                    'incoming_ke_mev':energy,'deposited_mev':float(points[:,3].sum()),
                                    'retained_volume_charge_fraction':audit['inside_volume_electrons']/max(audit['input_electrons'],1e-12),
                                    'collection_component_fraction':crop[0]['selected_signal_fraction'],
                                    'induction_component_fraction':crop[1]['selected_signal_fraction'],
                                    'approximate_direction':int('approximate' in placement['direction_method']),
                                    'outside_proton_energy_range':int(not any(lo<=energy<hi for lo,hi in supported_energy_ranges))}
                                buffer_images.append(raw);buffer_records.append(record)
                                part_counts['accepted']+=1
                                pid=str(particle['pid']);part_counts['by_pid'][pid]=part_counts['by_pid'].get(pid,0)+1
                            except ValueError as error:
                                reason=str(error)
                                part_counts['rejected']+=1
                                part_counts['reasons'][reason]=part_counts['reasons'].get(reason,0)+1
                                buffer_rejected.append((event,particle['group_id'],particle['pid'],reason.encode()[:256]))
                            if len(buffer_images)>=256 or len(buffer_rejected)>=1024:
                                flush()
                    flush()
                    converted.attrs['counts']=json.dumps(part_counts)
                    converted.attrs['complete']=True
                os.replace(temporary,final)
            for key in ('events','accepted','rejected','LED_groups'):
                counts[key]+=part_counts[key]
            for key in ('by_pid','reasons'):
                for label,n in part_counts[key].items():counts[key][label]=counts[key].get(label,0)+n
            report=destination/'progress.json';tmp=Path(str(report)+'.tmp')
            tmp.write_text(json.dumps(counts,indent=2));os.replace(tmp,report)
            print(f'{relative}: {counts["events"]}/{events} events, {counts["accepted"]} images',flush=True)
    return counts


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    base=Path('/Volumes/easystore/proton-kaon/pilarnet_lariat')
    parser.add_argument('--input',type=Path,default=base/'full')
    parser.add_argument('--calibration',type=Path,default=base/'calibration')
    parser.add_argument('--output',type=Path,default=base/'converted')
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--events-per-part',type=int,default=5000)
    parser.add_argument('--max-events',type=int,help='Smoke-test limit per input file')
    parser.add_argument('--max-files',type=int,help='Smoke-test input-file limit')
    parser.add_argument('--wait',action='store_true',help='Wait for calibration and remaining verified downloads')
    args=parser.parse_args()
    if not 1<=args.workers<=4 or args.events_per_part<1:parser.error('Invalid workers / events per part')
    report=args.calibration/'calibration_report.json'
    if args.wait:
        while not report.exists():time.sleep(15)
    calibration=json.loads(report.read_text())
    response_path=args.calibration/'fitted_response.yaml'
    response_sha=hashlib.sha256(response_path.read_bytes()).hexdigest()
    if response_sha!=calibration['response_sha256']:raise RuntimeError('Frozen response checksum mismatch')
    response=LArIATResponse(**yaml.safe_load(response_path.read_text()))
    angles=np.load(args.calibration/'fit_endpoint_angles_deg.npy')
    angle_sha=hashlib.sha256((args.calibration/'fit_endpoint_angles_deg.npy').read_bytes()).hexdigest()
    args.output.mkdir(parents=True,exist_ok=True)
    state={'response_sha256':response_sha,'angle_bank_sha256':angle_sha,
           'state':'running','files':{},'warning':'pilot response; validation discrepancies are in calibration_report.json'}
    inventory=json.loads((args.input/'remote_inventory.json').read_text())
    relative=[r['path'] for r in inventory if r['path'].endswith('.h5')]
    relative.sort(key=lambda f:(0 if f=='train/generic_v2_51800_v2.h5' else 1,f))
    if args.max_files:relative=relative[:args.max_files]
    futures={}
    supported_ranges=[(b['lo'],b['hi']) for b in calibration['partitions']['validation']['bins'] if b['supported']]
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        while len(futures)<len(relative) or any(not f.done() for f in futures.values()):
            download=json.loads((args.input/'download_status.json').read_text())
            for rel in relative:
                if rel not in futures and download['files'][rel]['state']=='verified':
                    state['files'][rel]={'state':'running'}
                    futures[rel]=executor.submit(convert_file,args.input/rel,rel,args.output,response,angles,
                        response_sha,args.events_per_part,args.max_events,angle_sha,supported_ranges)
            for rel,future in futures.items():
                if future.done() and state['files'][rel]['state']=='running':
                    try:state['files'][rel]={'state':'complete','counts':future.result()}
                    except Exception as error:state['files'][rel]={'state':'failed','error':str(error)}
            tmp=args.output/'conversion_status.json.tmp';tmp.write_text(json.dumps(state,indent=2))
            os.replace(tmp,args.output/'conversion_status.json')
            if download['state']=='failed' and len(futures)<len(relative):
                state['state']='waiting_for_failed_download';break
            if not args.wait and len(futures)<len(relative):
                state['state']='remaining_files_not_downloaded';break
            if len(futures)==len(relative) and all(f.done() for f in futures.values()):break
            time.sleep(15)
    if len(futures)==len(relative) and all(state['files'][rel]['state']=='complete' for rel in relative):
        state['state']='complete'
    elif state['state']=='running':state['state']='failed'
    (args.output/'conversion_status.json').write_text(json.dumps(state,indent=2))
    print(f'Conversion state: {state["state"]}',flush=True)


if __name__=='__main__':main()
