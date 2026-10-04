#!/usr/bin/env python3
"""Selection-matched short-light pilot from the dedicated RAW_muons.root source.

Chunked extraction avoids materializing all connected components. Writes only to
output/latent_limit; existing MIP reference and raw data are untouched.
"""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from src.open_root import open_root
from src.clustering import extract_clusters
from src.cuts import cluster_cuts,image_cuts
from src.matching import matching
from src.images import pad_image_batch_gpu
from latent_limit_experiments import OUT,BASE,model,data,indices
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from _beam_data import apply_style,SINGLE_COL
import matplotlib.pyplot as plt

SOURCE='/Volumes/easystore/proton-kaon/raw/Muons_50_300/RAW_muons.root'


def build(batch=2000,max_events=None):
    source=open_root(SOURCE,tree_name='anatree/raw')
    if max_events:source=source.head(max_events)
    folder=OUT/'short_light_batches';folder.mkdir(parents=True,exist_ok=True)
    for lo in range(0,len(source),batch):
        dest=folder/f'batch_{lo:06d}.pkl'
        if dest.exists():continue
        sub=source.iloc[lo:lo+batch]
        clusters=extract_clusters(sub,particle_type='muon',threshold=15,tree_name='anatree/raw')
        if clusters.empty:
            pd.DataFrame().to_pickle(dest);continue
        clusters['image_intensity']=clusters.image_intensity.map(lambda a:np.clip(a,0,None))
        clusters['column_maxes']=clusters.image_intensity.map(lambda a:a.max(axis=1))
        cut=cluster_cuts(clusters,lower=10,upper=179)
        del clusters
        if cut.empty:
            pd.DataFrame().to_pickle(dest);continue
        col,ind=matching(cut);del cut
        if col.empty:
            pd.DataFrame().to_pickle(dest);continue
        col,ind=image_cuts(col,ind,lower=10,upper=179,width=1500)
        if len(col)!=len(ind):raise RuntimeError('plane mismatch')
        if len(col):
            keep=['run','subrun','event','cluster_idx','height','width','image_intensity']
            c=col[keep].reset_index(drop=True);i=ind[keep].reset_index(drop=True)
            out=c.rename(columns={x:x+'_col' for x in ['cluster_idx','height','width','image_intensity']})
            out=out.assign(cluster_idx_ind=i.cluster_idx,height_ind=i.height,width_ind=i.width,image_intensity_ind=i.image_intensity)
        else:out=pd.DataFrame()
        out.to_pickle(dest)
        print('batch',lo,len(sub),'matched short',len(out),flush=True)
    files=sorted(folder.glob('batch_*.pkl'))
    table=pd.concat([pd.read_pickle(f) for f in files if len(pd.read_pickle(f))],ignore_index=True)
    table.to_pickle(OUT/'short_light_pairs.pkl')
    print('all short pairs',len(table),'unique events',len(table.drop_duplicates(['run','subrun','event'])),flush=True)


def encode():
    table=pd.read_pickle(OUT/'short_light_pairs.pkl')
    d,zz=data()
    # The long-MIP reference came from the same ROOT source. Exclude any
    # overlapping event before embedding so a different cluster from one event
    # cannot become an artificially close cross-selection neighbour.
    key=['run','subrun','event']
    baseline_keys=pd.MultiIndex.from_frame(d[key].astype('int64'))
    source_keys=pd.MultiIndex.from_frame(table[key].astype('int64'))
    overlap=source_keys.isin(baseline_keys)
    print('excluded baseline-overlap short pairs',int(overlap.sum()),flush=True)
    table=table.loc[~overlap].reset_index(drop=True)
    table.to_pickle(OUT/'short_light_pairs_disjoint.pkl')
    table.drop(columns=['image_intensity_col','image_intensity_ind']).to_csv(
        OUT/'short_light_metadata.csv',index=False)
    net,device=model();c=table.image_intensity_col.tolist();i=table.image_intensity_ind.tolist()
    zs=[];images=[]
    with torch.no_grad():
        for lo in range(0,len(table),64):
            a=np.array(pad_image_batch_gpu(c[lo:lo+64],device=device,batch_size=64,cut_rows=50))
            b=np.array(pad_image_batch_gpu(i[lo:lo+64],device=device,batch_size=64,cut_rows=50))
            x=torch.stack([F.interpolate(torch.from_numpy(a).float().to(device).unsqueeze(1),size=(48,48),mode='bilinear',align_corners=False).squeeze(1),
                           F.interpolate(torch.from_numpy(b).float().to(device).unsqueeze(1),size=(48,48),mode='bilinear',align_corners=False).squeeze(1)],dim=1)
            x=torch.log1p(x);mu,_=net.encode(x);zs.append(mu.cpu().numpy());images.append(x.cpu().numpy())
    z=np.concatenate(zs);x=np.concatenate(images);np.save(OUT/'short_light_latents.npy',z);np.save(OUT/'short_light_images.npy',x)
    raw=np.load(BASE/'representations/vae_s0.npy');sc=StandardScaler().fit(raw[indices(d,'train')]);u=sc.transform(z)
    references={name:zz['vae_s0'][indices(d,'test',species)] for name,species in
                [('long_mip','muon'),('kaon_window','kaon'),('proton','proton')]}
    # Equal reference sizes avoid making the largest pool look closest merely
    # because it has more possible neighbours. Use repeated fixed subsamples.
    nref=min(map(len,references.values()));rng=np.random.default_rng(20260924);rows=[]
    for rep in range(10):
        for name,full in references.items():
            reference=full[rng.choice(len(full),nref,replace=False)]
            knn=NearestNeighbors(n_neighbors=6).fit(reference)
            query_dist=knn.kneighbors(u,return_distance=True)[0][:,4]
            within_dist=knn.kneighbors(reference,return_distance=True)[0][:,5]
            rows.append(dict(reference=name,replicate=rep,n_short=len(u),n_reference=nref,
                             median_distance=float(np.median(query_dist)),
                             q90_distance=float(np.quantile(query_dist,.9)),
                             reference_self_median=float(np.median(within_dist)),
                             relative_to_self=float(np.median(query_dist)/np.median(within_dist))))
    pd.DataFrame(rows).to_csv(OUT/'f2_short_light_support.csv',index=False)
    print('F2',rows,flush=True)


def compare():
    """Paired query-level comparison, with event-group bootstrap intervals."""
    table=pd.read_csv(OUT/'short_light_metadata.csv')
    z=np.load(OUT/'short_light_latents.npy')
    d,zz=data();raw=np.load(BASE/'representations/vae_s0.npy')
    u=StandardScaler().fit(raw[indices(d,'train')]).transform(z)
    pools={name:zz['vae_s0'][indices(d,'test',species)] for name,species in
           [('long_mip','muon'),('kaon_window','kaon'),('proton','proton')]}
    nref=min(map(len,pools.values()));rng=np.random.default_rng(20260924)
    distances={}
    for name,pool in pools.items():
        ref=pool[rng.choice(len(pool),nref,replace=False)]
        distances[name]=NearestNeighbors(n_neighbors=5).fit(ref).kneighbors(u,return_distance=True)[0][:,4]
    out=table[['run','subrun','event','height_col']].copy()
    for name,dist in distances.items():out['distance_'+name]=dist
    out['closer_kaon_than_mip']=out.distance_kaon_window<out.distance_long_mip
    out['closer_kaon_than_proton']=out.distance_kaon_window<out.distance_proton
    out['length_bin']=pd.cut(out.height_col,[10,50,100,150,179],labels=['11–50','51–100','101–150','151–178'])
    out.to_csv(OUT/'f2_short_light_query_distances.csv',index=False)
    summary=[]
    for name,part in [('all',out)]+[(str(b),out[out.length_bin.eq(b)]) for b in out.length_bin.cat.categories]:
        group=part.groupby([part.run,part.subrun,part.event],sort=False).closer_kaon_than_mip.mean().to_numpy()
        brng=np.random.default_rng(4409);draws=np.array([brng.choice(group,len(group),replace=True).mean() for _ in range(300)])
        summary.append(dict(length_bin=name,n_rows=len(part),n_events=len(group),
                            fraction_kaon_closer=float(group.mean()),
                            low=float(np.quantile(draws,.025)),high=float(np.quantile(draws,.975)),
                            median_kaon_distance=float(part.distance_kaon_window.median()),
                            median_mip_distance=float(part.distance_long_mip.median())))
    pd.DataFrame(summary).to_csv(OUT/'f2_short_light_pairwise.csv',index=False)
    print(pd.DataFrame(summary).to_string(index=False),flush=True)
    # Reverse direction: does a kaon-window query prefer the short-light
    # reference over the original long-MIP reference? Proton queries are a
    # negative control for interpreting this preference as light specificity.
    reverse=[];nr=min(len(pools['long_mip']),len(u),2470)
    for rep in range(10):
        rr=np.random.default_rng(91270+rep)
        short_ref=u[rr.choice(len(u),nr,replace=False)]
        long_ref=pools['long_mip'][rr.choice(len(pools['long_mip']),nr,replace=False)]
        short_knn=NearestNeighbors(n_neighbors=5).fit(short_ref)
        long_knn=NearestNeighbors(n_neighbors=5).fit(long_ref)
        for name in ['kaon_window','proton']:
            q=pools[name]
            ds=short_knn.kneighbors(q,return_distance=True)[0][:,4]
            dl=long_knn.kneighbors(q,return_distance=True)[0][:,4]
            reverse.append(dict(query=name,replicate=rep,n_query=len(q),n_reference=nr,
                                fraction_short_closer=float(np.mean(ds<dl)),
                                median_short_distance=float(np.median(ds)),
                                median_long_mip_distance=float(np.median(dl))))
    pd.DataFrame(reverse).to_csv(OUT/'f2_reverse_reference_preference.csv',index=False)
    print(pd.DataFrame(reverse).groupby('query').mean(numeric_only=True).to_string(),flush=True)


def method_controls():
    """Test whether the support failure persists in AE and random codes."""
    d,zz=data();tr=indices(d,'train')
    rng=np.random.default_rng(8254)
    take=np.sort(rng.choice(len(pd.read_csv(OUT/'short_light_metadata.csv')),5000,replace=False))
    images=np.load(OUT/'short_light_images.npy',mmap_mode='r')
    rows=[]
    for method,stem in [('vae','vae_s0'),('ae','ae_s0'),('random','random_s0')]:
        net,device=model(method);torch.set_num_threads(4);lat=[]
        with torch.no_grad():
            for lo in range(0,len(take),128):
                x=torch.from_numpy(np.asarray(images[take[lo:lo+128]]).copy()).to(device)
                mu,_=net.encode(x);lat.append(mu.cpu().numpy())
        raw=np.load(BASE/'representations'/f'{stem}.npy')
        short=StandardScaler().fit(raw[tr]).transform(np.concatenate(lat))
        pools={name:zz[stem][indices(d,'test',species)] for name,species in
               [('long_mip','muon'),('kaon_window','kaon'),('proton','proton')]}
        nr=min(map(len,pools.values()));rr=np.random.default_rng(7001)
        dist={}
        for name,full in pools.items():
            ref=full[rr.choice(len(full),nr,replace=False)]
            knn=NearestNeighbors(n_neighbors=6).fit(ref)
            q=knn.kneighbors(short,return_distance=True)[0][:,4]
            own=knn.kneighbors(ref,return_distance=True)[0][:,5]
            dist[name]=q
            rows.append(dict(method=method,reference=name,n_short=len(short),n_reference=nr,
                             median_distance=float(np.median(q)),
                             relative_to_self=float(np.median(q)/np.median(own))))
        print(method,'fraction short nearer K than long MIP',np.mean(dist['kaon_window']<dist['long_mip']),flush=True)
    pd.DataFrame(rows).to_csv(OUT/'f2_method_support_controls.csv',index=False)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['build','encode','compare','method_controls']);p.add_argument('--batch',type=int,default=2000);p.add_argument('--max-events',type=int)
    a=p.parse_args()
    if a.stage=='build':build(a.batch,a.max_events)
    elif a.stage=='encode':encode()
    elif a.stage=='compare':compare()
    else:method_controls()
