#!/usr/bin/env python3
"""Additional frozen-model diagnostics using the matched baseline manifest."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge,LogisticRegression
from sklearn.metrics import roc_auc_score,r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from _beam_data import apply_style,SINGLE_COL,DOUBLE_COL
from latent_limit_experiments import BASE,OUT,STAGE,data,indices,model,figsave
from evaluate_representation_baselines import mass_contrast
from src.cuts import image_cuts
import matplotlib.pyplot as plt


def b2_local_masks():
    d,z=data();net,device=model();imgs=np.load(BASE/'images_log1p.npy',mmap_mode='r')
    ii=indices(d,'test');rng=np.random.default_rng(718);ii=rng.choice(ii,size=min(900,len(ii)),replace=False)
    rawz=np.load(BASE/'representations/vae_s0.npy');sd=rawz[indices(d,'train')].std(0)
    x=np.asarray(imgs[ii]).copy();rows=[]
    for axis,start,stop in [('wire',0,8),('wire',20,28),('wire',40,48),('drift',0,8),('drift',20,28),('drift',40,48)]:
        a=x.copy()
        if axis=='wire':a[:,:,start:stop,:]=0
        else:a[:,:,:,start:stop]=0
        out=[]
        with torch.no_grad():
            for lo in range(0,len(ii),128):
                mu,_=net.encode(torch.from_numpy(a[lo:lo+128]).to(device));out.append(mu.cpu().numpy())
        shift=np.linalg.norm((np.concatenate(out)-rawz[ii])/sd,axis=1)
        rows.append(dict(axis=axis,start=start,stop=stop,n=len(ii),median_shift=float(np.median(shift)),q90=float(np.quantile(shift,.9)),
                         original_signal_fraction=float(np.sum(np.expm1(x-a))/np.sum(np.expm1(x)))))
    a=pd.DataFrame(rows);a.to_csv(OUT/'b2_local_mask_response.csv',index=False)
    print('B2',a.to_dict('records'),flush=True)


def b3_nuisance_subspaces():
    """Project out development-independent synthetic perturbation directions."""
    d,z=data();tr=indices(d,'train');raw=np.load(BASE/'representations/vae_s0.npy');scale=raw[tr].std(0)
    with np.load(OUT/'b1_response_vectors_dev.npz') as b:
        vectors={'gain':b['gain_plus10'].mean(0)/scale,
                 'drift_shift':b['drift_shift_one'].mean(0)/scale,
                 'both':np.stack([b['gain_plus10'].mean(0)/scale,b['drift_shift_one'].mean(0)/scale],axis=1)}
    rng=np.random.default_rng(2026);rows=[]
    for kind,v in vectors.items():
        matrix=v[:,None] if v.ndim==1 else v
        q=np.linalg.qr(matrix)[0];rnd=np.linalg.qr(rng.normal(size=(8,q.shape[1])))[0]
        for condition,axis in [('original',None),('remove',q),('random_same_rank',rnd)]:
            x=z['vae_s0'] if axis is None else z['vae_s0']-(z['vae_s0']@axis)@axis.T
            for target in ['mean_adc','solidity']:
                y=d[target].to_numpy()
                for sp in ['proton','kaon','muon']:
                    fit=indices(d,'dev',sp);reader=Ridge(alpha=10).fit(x[fit],y[fit])
                    for part in ['test','run_test']:
                        ii=indices(d,part,sp)
                        rows.append(dict(direction=kind,condition=condition,target=target,species=sp,
                                         partition=part,r2=r2_score(y[ii],reader.predict(x[ii]))))
    pd.DataFrame(rows).to_csv(OUT/'b3_nuisance_projection.csv',index=False)
    print('B3',len(rows),flush=True)


def f1_selective_stability():
    d,z=data();scores=[]
    for s in range(3):
        p=BASE/'evaluation'/f'vae_s{s}'/'cluster_scores.npz'
        with np.load(p) as f:scores.append(f['k37_s0_proton_score'])
    a=np.stack(scores);mean=a.mean(0);unc=a.std(0)
    test=indices(d,'test','kaon');test=test[(d.iloc[test].picky==1).to_numpy()]
    rows=[]
    for fraction in [.25,.5,.75,1.]:
        limit=np.quantile(unc[test],fraction) if fraction<1 else np.inf
        dd=d.copy();m=np.ones(len(d),dtype=bool);m[test]=unc[test]<=limit
        dd.loc[~m & dd.partition.eq('test'),'partition']='excluded_by_stability'
        r=mass_contrast(mean,dd,'test',.2,1)
        rows.append(dict(stable_fraction=fraction,n=int(sum(m[test])),mass_shift=r['shift'] if r else np.nan,
                         median_score_sd=float(np.median(unc[test[m[test]]]))))
    out=pd.DataFrame(rows);out.to_csv(OUT/'f1_score_stability_mass.csv',index=False)
    print('F1',rows,flush=True)


def f4_quality_readout():
    d,z=data();rows=[]
    for meth in ['vae_s0','vae_s1','vae_s2','ae_s0','endpoint_pca']:
        fit=indices(d,'dev','kaon');x=z[meth]
        y=d.picky.to_numpy();model=LogisticRegression(max_iter=1000).fit(x[fit],y[fit])
        for part in ['test','run_test']:
            ii=indices(d,part,'kaon');p=model.predict_proba(x[ii])[:,1]
            rows.append(dict(method=meth,partition=part,n=len(ii),picky_auc=roc_auc_score(y[ii],p)))
    pd.DataFrame(rows).to_csv(OUT/'f4_quality_readout.csv',index=False)
    print('F4',rows,flush=True)


def g2_plane_pairs():
    d,z=data();net,device=model();imgs=np.load(BASE/'images_log1p.npy',mmap_mode='r')
    rng=np.random.default_rng(113);ii=rng.choice(indices(d,'test'),size=650,replace=False)
    x=np.asarray(imgs[ii]).copy();ef=np.load(BASE/'endpoint.npy');pairs=[]
    for i,row in enumerate(ii):
        pool=ii[(d.iloc[ii].species==d.iloc[row].species).to_numpy() & (d.iloc[ii].event_key!=d.iloc[row].event_key).to_numpy()]
        if len(pool)==0:continue
        gap=((ef[pool,16:]-ef[row,16:])**2).mean(1)
        pairs.append((i,np.argmin(gap),int(pool[np.argmin(gap)]),int(rng.choice(pool))))
    matched=x.copy();random=x.copy()
    for i,_,j,k in pairs:
        matched[i,1]=imgs[j,1];random[i,1]=imgs[k,1]
    rows=[]
    for name,a in [('genuine',x),('matched_wrong_plane',matched),('random_wrong_plane',random)]:
        vals=[]
        with torch.no_grad():
            for lo in range(0,len(ii),128):
                xb=torch.from_numpy(a[lo:lo+128].copy()).to(device);mu,_=net.encode(xb);re=net.decode(mu)
                vals.append(((re-xb).square()*torch.where(xb>0,10.,1.)).sum((1,2,3)).cpu().numpy())
        vals=np.concatenate(vals)
        rows.append(dict(condition=name,n=len(vals),median_weighted_error=float(np.median(vals)),mean_weighted_error=float(vals.mean())))
        np.save(OUT/f'g2_{name}_errors.npy',vals)
    pd.DataFrame(rows).to_csv(OUT/'g2_plane_mismatch.csv',index=False)
    print('G2',rows,flush=True)


def e2_local_walks():
    d,z=data();net,device=model();raw=np.load(BASE/'representations/vae_s0.npy');tr=indices(d,'train')
    fit=indices(d,'dev','kaon');scaler=StandardScaler().fit(raw[tr]);x=scaler.transform(raw)
    ch=Ridge(alpha=10).fit(x[fit],d.iloc[fit].mean_adc);top=Ridge(alpha=10).fit(x[fit],d.iloc[fit].solidity)
    v=ch.coef_.copy();u=top.coef_.copy();v-=v@u/(u@u)*u;v/=np.linalg.norm(v)
    pool=indices(d,'test','kaon');rng=np.random.default_rng(932);anchors=rng.choice(pool,size=3,replace=False)
    nn=NearestNeighbors(n_neighbors=2).fit(x[tr]);base=nn.kneighbors(x[tr],return_distance=True)[0][:,1]
    support95=float(np.quantile(base,.95));rows=[];decoded=[]
    for anchor in anchors:
        for step in [-.6,-.3,0,.3,.6]:
            moved=x[anchor]+step*v;dist,j=nn.kneighbors(moved[None,:],return_distance=True)
            mu=scaler.inverse_transform(moved[None,:]).astype('float32')
            with torch.no_grad():image=net.decode(torch.from_numpy(mu).to(device)).cpu().numpy()[0,0]
            decoded.append(image)
            rows.append(dict(anchor_row=int(anchor),step=step,predicted_charge=float(ch.predict(moved[None,:])[0]),
                             predicted_solidity=float(top.predict(moved[None,:])[0]),nearest_train_distance=float(dist[0,0]),
                             nearest_train_row=int(tr[j[0,0]]),outside_local_support=bool(dist[0,0]>support95)))
    a=pd.DataFrame(rows);a.to_csv(OUT/'e2_local_walks.csv',index=False)
    apply_style(SINGLE_COL);fig,axes=plt.subplots(3,5,figsize=(DOUBLE_COL,3.5))
    for ax,img,r in zip(axes.flat,decoded,rows):
        ax.imshow(img,origin='lower',cmap='viridis',vmin=0,vmax=np.quantile(decoded,.995));ax.set_xticks([]);ax.set_yticks([])
        if r['outside_local_support']:ax.spines[:].set_color('#AA3377')
    for ax,s in zip(axes[0],[-.6,-.3,0,.3,.6]):ax.set_title(f'{s:+.1f}',fontsize=9)
    figsave(fig,'e2_local_walks')
    print('E2',{'rows':len(rows),'outside_support':int(a.outside_local_support.sum()),'support95':support95},flush=True)


def a2_unseen_upstream():
    """Read full source clusters and predict exclusively pre-endpoint charge."""
    d,z=data();base=Path('/Volumes/easystore/proton-kaon/clusters');values=np.full(len(d),np.nan)
    for source,types,kwargs in [('col.pkl',['proton','kaon'],dict(lower=10)),
                                ('muon_col.pkl',['muon'],dict(lower=175,upper=10_000_000,width=473))]:
        print('loading',source,flush=True)
        col=pd.read_pickle(base/source)
        partner='ind.pkl' if source=='col.pkl' else 'muon_ind.pkl'
        ind=pd.read_pickle(base/partner)
        col,_=image_cuts(col,ind,**kwargs)
        del ind
        for sp in types:
            sub=col[col.particle_type.eq(sp)].reset_index(drop=True)
            rows=np.flatnonzero(d.species.eq(sp).to_numpy())
            local=d.iloc[rows].source_tensor_row.to_numpy(dtype=int)
            assert local.max()<len(sub) and len(np.unique(local))==len(sub)
            key1=d.iloc[rows].event_key.to_numpy()
            key2=sub.iloc[local][['run','subrun','event']].astype(int).astype(str).agg(':'.join,axis=1).to_numpy()
            assert np.all(key1==key2),f'row mapping failed for {sp}'
            for position,j in enumerate(local):
                img=sub.iloc[j].image_intensity
                if img.shape[0]<100:continue
                before=np.asarray(img[-100:-50],dtype=float)
                positive=before[before>0]
                if len(positive):values[rows[position]]=float(np.median(positive))
            print(sp,'unseen targets',np.isfinite(values[rows]).sum(),flush=True)
        del col
    np.save(OUT/'a2_unseen_upstream_target.npy',values)
    rows=[]
    for sp in ['proton','kaon','muon']:
        fit=indices(d,'dev',sp);fit=fit[np.isfinite(values[fit])]
        if len(fit)<30:continue
        for meth in ['vae_s0','vae_s1','vae_s2','ae_s0','endpoint_pca']:
            x=z[meth];rd=Ridge(alpha=10).fit(x[fit],values[fit])
            for part in ['test','run_test']:
                ii=indices(d,part,sp);ii=ii[np.isfinite(values[ii])]
                if len(ii)<30:continue
                rows.append(dict(species=sp,method=meth,partition=part,n_train=len(fit),n_test=len(ii),
                                 r2=r2_score(values[ii],rd.predict(x[ii]))))
    pd.DataFrame(rows).to_csv(OUT/'a2_unseen_upstream_readout.csv',index=False)
    print('A2',rows,flush=True)


STAGES={'a2':a2_unseen_upstream,'b2':b2_local_masks,'b3':b3_nuisance_subspaces,'f1':f1_selective_stability,'f4':f4_quality_readout,
        'g2':g2_plane_pairs,'e2':e2_local_walks}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=STAGES);a=p.parse_args();STAGES[a.stage]()
