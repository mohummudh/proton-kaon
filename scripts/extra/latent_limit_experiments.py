#!/usr/bin/env python3
"""Frozen-embedding and frozen-model experiments for the eight-dimensional study.

Each stage writes an independent table; figures are staged for inspection before
copying to figs/. Run from repository root with the current baseline manifest.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / 'scripts/extra')]
import numpy as np
import pandas as pd
import torch
import yaml
import matplotlib.pyplot as plt
from scipy.linalg import orthogonal_procrustes
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr
from _beam_data import apply_style, SINGLE_COL, DOUBLE_COL, COLOURS
from src.models.build import build_vae

BASE = ROOT / 'output/representation_baselines'
OUT = ROOT / 'output/latent_limit'
STAGE = OUT / 'staging_figs'
OUT.mkdir(parents=True, exist_ok=True)
STAGE.mkdir(parents=True, exist_ok=True)
PARTS = ['test', 'run_test']
SPS = ['proton', 'kaon', 'muon']


def data():
    d = pd.read_csv(BASE / 'manifest.csv')
    tr = d.partition.eq('train').to_numpy()
    z = {}
    for stem in ['vae_s0', 'vae_s1', 'vae_s2', 'ae_s0', 'ae_s1', 'ae_s2', 'random_s0', 'endpoint_pca']:
        x = np.load(BASE / 'representations' / (stem + '.npy'))
        z[stem] = StandardScaler().fit(x[tr]).transform(x)
    return d, z


def indices(d, part, sp=None):
    m = d.partition.eq(part)
    if sp: m &= d.species.eq(sp)
    return np.flatnonzero(m.to_numpy())


def ci_r2(d, ii, pred, y, draws=200, seed=806):
    groups = d.iloc[ii].event_key.to_numpy()
    _, inv = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        w = np.bincount(rng.integers(inv.max()+1, size=inv.max()+1), minlength=inv.max()+1)[inv]
        vals.append(r2_score(y, pred, sample_weight=w))
    return np.quantile(vals, [.025, .975]).tolist()


def figsave(fig, name):
    fig.tight_layout()
    fig.savefig(STAGE / (name + '.pdf'), bbox_inches='tight', pad_inches=.03)
    fig.savefig(STAGE / (name + '.png'), dpi=300, bbox_inches='tight', pad_inches=.03)
    plt.close(fig)


def atlas():
    """A1: held-out physical-observable readout atlas."""
    d, zz = data()
    targets = ['mean_adc', 'median_adc', 'solidity', 'fill_fraction',
               'bragg_rise_slope', 'bragg_peak_ratio', 'profile_cv', 'n_local_maxima']
    methods = ['vae_s0', 'vae_s1', 'vae_s2', 'ae_s0', 'ae_s1', 'ae_s2', 'random_s0', 'endpoint_pca']
    rows = []
    for target in targets:
        y = pd.to_numeric(d[target], errors='coerce').to_numpy()
        for sp in SPS:
            fit = indices(d, 'dev', sp); fit = fit[np.isfinite(y[fit])]
            if len(fit) < 30 or np.std(y[fit]) < 1e-9: continue
            for meth in methods:
                x = zz[meth]
                model = Ridge(alpha=10).fit(x[fit], y[fit])
                for part in PARTS:
                    ii = indices(d, part, sp); ii = ii[np.isfinite(y[ii])]
                    pred = model.predict(x[ii]); lo, hi = ci_r2(d, ii, pred, y[ii])
                    rows.append(dict(target=target, species=sp, method=meth, partition=part,
                                     n=len(ii), r2=r2_score(y[ii], pred), low=lo, high=hi))
    a = pd.DataFrame(rows); a.to_csv(OUT / 'a1_physical_atlas.csv', index=False)
    f = a[a.partition.eq('test') & a.method.isin(['vae_s0', 'endpoint_pca', 'random_s0'])]
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL, 2.55), sharey=True)
    for ax, sp in zip(axes, SPS):
        sub = f[f.species.eq(sp)]
        for meth, color, label in [('vae_s0', '#0077BB', 'VAE'), ('endpoint_pca', '#EE7733', 'Endpoint PCA'), ('random_s0', '#999999', 'Random CNN')]:
            s = sub[sub.method.eq(meth)].set_index('target').reindex(targets)
            ax.plot(np.arange(len(targets)), s.r2, 'o-', color=color, lw=.9, ms=2.4, label=label)
        ax.axhline(0, color='.7', lw=.5); ax.set_title('MIPs' if sp=='muon' else sp.capitalize(), fontsize=9)
        ax.set_xticks(np.arange(len(targets)), [t.replace('_',' ') for t in targets], rotation=65, ha='right')
        ax.set_ylim(-.15, 1.02)
    axes[0].set_ylabel('Held-out linear $R^2$')
    axes[2].legend(frameon=False, loc='upper right')
    figsave(fig, 'a1_physical_atlas')
    print('A1',len(a),flush=True)


def transfer():
    """C1: cross-tag transfer within source target-range support."""
    d, zz = data(); targets = ['mean_adc','median_adc','solidity','bragg_rise_slope']
    rows = []
    for target in targets:
        y = pd.to_numeric(d[target], errors='coerce').to_numpy()
        for meth in ['vae_s0','vae_s1','vae_s2','ae_s0','endpoint_pca']:
            x = zz[meth]
            for source in SPS:
                fit = indices(d,'dev',source); fit=fit[np.isfinite(y[fit])]
                if len(fit)<30: continue
                model=Ridge(alpha=10).fit(x[fit],y[fit]); q=np.quantile(y[fit],[.05,.95])
                for target_sp in SPS:
                    for part in PARTS:
                        ii=indices(d,part,target_sp); ii=ii[np.isfinite(y[ii])]
                        overlap=ii[(y[ii]>=q[0]) & (y[ii]<=q[1])]
                        for support,jj in [('all',ii),('source_5_95',overlap)]:
                            if len(jj)<30: continue
                            pred=model.predict(x[jj]); lo,hi=ci_r2(d,jj,pred,y[jj])
                            rows.append(dict(target=target,method=meth,source=source,destination=target_sp,
                                             partition=part,support=support,n=len(jj),r2=r2_score(y[jj],pred),
                                             rho=spearmanr(y[jj],pred).statistic,low=lo,high=hi))
    a=pd.DataFrame(rows);a.to_csv(OUT/'c1_cross_tag_transfer.csv',index=False)
    f=a[(a.method=='vae_s0')&(a.partition=='test')&(a.support=='source_5_95')&(a.target.isin(['mean_adc','solidity']))]
    apply_style(SINGLE_COL); fig, axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.7))
    for ax,t in zip(axes,['mean_adc','solidity']):
        m=f[f.target.eq(t)].pivot(index='source',columns='destination',values='r2').reindex(index=SPS,columns=SPS)
        im=ax.imshow(m.to_numpy(),cmap='RdBu',vmin=-1,vmax=1)
        ax.set_xticks(range(3),['p','K','MIP']);ax.set_yticks(range(3),['p','K','MIP']);ax.set_xlabel('Applied to');ax.set_ylabel('Trained on')
        for i in range(3):
            for j in range(3):
                value=m.iloc[i,j]
                label=f'{value:.2f}' if abs(value)<10 else f'{value:.0f}'
                ax.text(j,i,label,ha='center',va='center',color='white' if abs(value)>.65 else 'black',fontsize=7)
        ax.set_title(t.replace('_',' '),fontsize=9)
    fig.colorbar(im,ax=axes,label='Held-out $R^2$',fraction=.03,pad=.03)
    figsave(fig,'c1_cross_tag_transfer')
    print('C1',len(a),flush=True)


def similarity():
    """C2/C3: training-seed and AE/VAE geometry; alignment fitted on train."""
    d,z=data();tr=indices(d,'train');test=indices(d,'test')
    rng=np.random.default_rng(12); test=rng.choice(test,size=min(1500,len(test)),replace=False)
    pairs=[('vae_s0','vae_s1'),('vae_s0','vae_s2'),('ae_s0','ae_s1'),('vae_s0','ae_s0'),('vae_s0','random_s0')]
    rows=[]
    for left,right in pairs:
        a,b=z[left],z[right];ma=a[tr].mean(0);mb=b[tr].mean(0)
        rot,_=orthogonal_procrustes(b[tr]-mb,a[tr]-ma)
        aa=a[test]-ma;bb=(b[test]-mb)@rot
        x=aa-aa.mean(0);y=bb-bb.mean(0)
        cka=np.linalg.norm(x.T@y,'fro')**2/(np.linalg.norm(x.T@x,'fro')*np.linalg.norm(y.T@y,'fro'))
        nbr1=NearestNeighbors(n_neighbors=11).fit(aa).kneighbors(aa,return_distance=False)[:,1:]
        nbr2=NearestNeighbors(n_neighbors=11).fit(bb).kneighbors(bb,return_distance=False)[:,1:]
        overlap=np.mean([len(set(q)&set(r))/10 for q,r in zip(nbr1,nbr2)])
        rows.append(dict(left=left,right=right,cka=cka,neighbor_overlap10=overlap,
                         aligned_mse=np.mean((aa-bb)**2),n=len(test)))
    pd.DataFrame(rows).to_csv(OUT/'c2_c3_seed_similarity.csv',index=False)
    print('C2/C3',rows,flush=True)


def bottleneck():
    """A4: unsupervised PCA compression of Z, fixed physical reader."""
    d,z=data();tr=indices(d,'train'); rows=[]
    for meth in ['vae_s0','vae_s1','vae_s2','ae_s0','endpoint_pca']:
        x=z[meth];pca=PCA(n_components=8,random_state=0).fit(x[tr]);u=pca.transform(x)
        for dim in range(1,9):
            for target in ['mean_adc','median_adc','solidity']:
                y=pd.to_numeric(d[target],errors='coerce').to_numpy()
                for sp in SPS:
                    fit=indices(d,'dev',sp);fit=fit[np.isfinite(y[fit])]
                    if len(fit)<30: continue
                    model=Ridge(alpha=10).fit(u[fit,:dim],y[fit])
                    for part in PARTS:
                        ii=indices(d,part,sp);ii=ii[np.isfinite(y[ii])]
                        pred=model.predict(u[ii,:dim]);rows.append(dict(method=meth,dimension=dim,target=target,
                                  species=sp,partition=part,n=len(ii),r2=r2_score(y[ii],pred)))
    a=pd.DataFrame(rows);a.to_csv(OUT/'a4_dimension_bottleneck.csv',index=False)
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.55),sharey=True)
    for ax,target in zip(axes,['mean_adc','solidity']):
        f=a[(a.partition=='test')&(a.target==target)&(a.method.isin(['vae_s0','endpoint_pca']))]
        for meth,color in [('vae_s0','#0077BB'),('endpoint_pca','#EE7733')]:
            for sp,style in [('proton','-'),('kaon','--'),('muon',':')]:
                s=f[(f.method==meth)&(f.species==sp)].sort_values('dimension')
                ax.plot(s.dimension,s.r2,style,color=color,lw=.95,label=f'{meth.replace("_s0", "")} {sp}')
        ax.set_xlabel('Retained PCA dimensions');ax.set_xticks(range(1,9));ax.set_title(target.replace('_',' '),fontsize=9)
    axes[0].set_ylabel('Held-out linear $R^2$');axes[1].legend(frameon=False,fontsize=6,loc='lower right',ncol=2)
    figsave(fig,'a4_dimension_bottleneck')
    print('A4',len(a),flush=True)


def reader_efficiency():
    """A3: nested label budgets for a frozen representation."""
    d,z=data();rng=np.random.default_rng(511);rows=[]
    for sp in SPS:
        dev=indices(d,'dev',sp)
        events=d.iloc[dev].event_key.unique();events=rng.permutation(events)
        for n in [25,50,100,200,500,len(events)]:
            chosen=set(events[:min(n,len(events))]);fit=dev[d.iloc[dev].event_key.isin(chosen).to_numpy()]
            for target in ['mean_adc','median_adc','solidity']:
                yy=pd.to_numeric(d[target],errors='coerce').to_numpy();ff=fit[np.isfinite(yy[fit])]
                if len(ff)<12:continue
                for meth in ['vae_s0','vae_s1','vae_s2','ae_s0','endpoint_pca','random_s0']:
                    model=Ridge(alpha=10).fit(z[meth][ff],yy[ff])
                    for part in PARTS:
                        ii=indices(d,part,sp);ii=ii[np.isfinite(yy[ii])]
                        rows.append(dict(species=sp,events_budget=n,images_used=len(ff),target=target,method=meth,
                                         partition=part,r2=r2_score(yy[ii],model.predict(z[meth][ii]))))
    a=pd.DataFrame(rows);a.to_csv(OUT/'a3_reader_label_efficiency.csv',index=False)
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,3,figsize=(DOUBLE_COL,2.55),sharey=True)
    for ax,sp in zip(axes,SPS):
        sub=a[(a.species==sp)&(a.partition=='test')&(a.target=='mean_adc')]
        for meth,col in [('vae_s0','#0077BB'),('endpoint_pca','#EE7733'),('random_s0','#999999')]:
            f=sub[sub.method==meth].sort_values('events_budget')
            ax.plot(f.events_budget,f.r2,'o-',lw=.85,ms=2.5,color=col,label=meth.replace('_s0',''))
        ax.set_xscale('log');ax.set_xlabel('Labeled development events');ax.set_title('MIPs' if sp=='muon' else sp.capitalize(),fontsize=9)
    axes[0].set_ylabel('Held-out charge $R^2$');axes[2].legend(frameon=False,loc='lower right',fontsize=7)
    figsave(fig,'a3_reader_label_efficiency')
    print('A3',len(a),flush=True)


def erasure():
    """B4: covariance-direction erasure of linear target information."""
    d,z=data();rng=np.random.default_rng(619);rows=[]
    for sp in SPS:
        fit=indices(d,'dev',sp)
        for erased in ['mean_adc','solidity']:
            yy=pd.to_numeric(d[erased],errors='coerce').to_numpy();ff=fit[np.isfinite(yy[fit])]
            x=z['vae_s0'];xc=x[ff]-x[ff].mean(0);yc=yy[ff]-yy[ff].mean()
            v=xc.T@yc/len(ff);v=v/np.linalg.norm(v)
            r=rng.normal(size=x.shape[1]);r-=r@v*v;r/=np.linalg.norm(r)
            for projection,w in [('original',None),('erase',v),('random_rank1',r)]:
                xx=x if w is None else x-(x@w)[:,None]*w[None,:]
                for target in ['mean_adc','solidity']:
                    t=pd.to_numeric(d[target],errors='coerce').to_numpy();f=fit[np.isfinite(t[fit])]
                    model=Ridge(alpha=10).fit(xx[f],t[f])
                    for part in PARTS:
                        ii=indices(d,part,sp);ii=ii[np.isfinite(t[ii])]
                        rows.append(dict(species=sp,erased=erased,projection=projection,target=target,
                                         partition=part,r2=r2_score(t[ii],model.predict(xx[ii]))))
    a=pd.DataFrame(rows);a.to_csv(OUT/'b4_linear_concept_erasure.csv',index=False)
    print('B4',len(a),flush=True)


def model(method='vae',seed=0):
    cfg=yaml.safe_load(next((ROOT/'configs').glob('run_0093*')).read_text())
    device='mps' if torch.backends.mps.is_available() else 'cpu'
    net=build_vae(cfg,device)
    c=torch.load(BASE/'models'/f'{method}_s{seed}.pt',map_location='cpu',weights_only=False)
    net.load_state_dict(c['state_dict']);net.eval()
    return net,device


def posterior():
    """E4: frozen VAE posterior variance and extra readout value."""
    d,z=data();net,device=model();imgs=np.load(BASE/'images_log1p.npy',mmap_mode='r')
    lv=[]
    with torch.no_grad():
        for lo in range(0,len(d),128):
            xb=torch.from_numpy(np.asarray(imgs[lo:lo+128]).copy()).to(device)
            _,v=net.encode(xb);lv.append(v.cpu().numpy())
    lv=np.concatenate(lv);np.save(OUT/'e4_logvar.npy',lv)
    tr=indices(d,'train'); v=StandardScaler().fit(lv[tr]).transform(lv)
    y=d['solidity'].to_numpy();rows=[]
    for feat,xx in [('mean',z['vae_s0']),('variance',v),('both',np.c_[z['vae_s0'],v])]:
        for sp in SPS:
            fit=indices(d,'dev',sp);fit=fit[np.isfinite(y[fit])]
            md=Ridge(alpha=10).fit(xx[fit],y[fit])
            for part in PARTS:
                ii=indices(d,part,sp);ii=ii[np.isfinite(y[ii])]
                rows.append(dict(features=feat,species=sp,partition=part,r2=r2_score(y[ii],md.predict(xx[ii]))))
    pd.DataFrame(rows).to_csv(OUT/'e4_posterior_readout.csv',index=False)
    print('E4',len(rows),flush=True)


def response():
    """B1: frozen encoder response to explicitly synthetic image perturbations."""
    d,z=data(); net,device=model(); x=np.load(BASE/'images_log1p.npy',mmap_mode='r')
    ii=indices(d,'test');rng=np.random.default_rng(99);ii=rng.choice(ii,size=min(1200,len(ii)),replace=False)
    xb=np.asarray(x[ii]).copy(); transforms={
        'gain_minus10':np.log1p(.9*np.expm1(xb)),
        'gain_plus10':np.log1p(1.1*np.expm1(xb)),
        'drift_shift_one':np.pad(xb[:,:,:,:-1],((0,0),(0,0),(0,0),(1,0))),
        'wire_shift_one':np.pad(xb[:,:,:-1,:],((0,0),(0,0),(1,0),(0,0))),
        'collection_zero':xb.copy(),
        'induction_zero':xb.copy(),
    }
    transforms['collection_zero'][:,0]=0;transforms['induction_zero'][:,1]=0
    rows=[];changes={}
    raw_z=np.load(BASE/'representations/vae_s0.npy')
    scale=raw_z[indices(d,'train')].std(0)
    ref=raw_z[ii]
    for name,arr in transforms.items():
        pred=[]
        with torch.no_grad():
            for lo in range(0,len(ii),128):
                mu,_=net.encode(torch.from_numpy(arr[lo:lo+128].astype(np.float32)).to(device));pred.append(mu.cpu().numpy())
        moved=np.concatenate(pred);delta=moved-ref;changes[name]=delta
        dist=np.linalg.norm(delta/scale,axis=1)
        rows.append(dict(perturbation=name,n=len(ii),median_distance=float(np.median(dist)),
                         q90_distance=float(np.quantile(dist,.9))))
    pd.DataFrame(rows).to_csv(OUT/'b1_detector_response.csv',index=False)
    np.savez_compressed(OUT/'b1_response_vectors.npz',row_indices=ii,**changes)
    dev=indices(d,'dev');dev=rng.choice(dev,size=min(1000,len(dev)),replace=False)
    xdev=np.asarray(x[dev]).copy();refdev=raw_z[dev]
    dev_changes={}
    for name,arr in [('gain_plus10',np.log1p(1.1*np.expm1(xdev))),
                     ('drift_shift_one',np.pad(xdev[:,:,:,:-1],((0,0),(0,0),(0,0),(1,0))))]:
        pred=[]
        with torch.no_grad():
            for lo in range(0,len(dev),128):
                mu,_=net.encode(torch.from_numpy(arr[lo:lo+128].astype(np.float32)).to(device));pred.append(mu.cpu().numpy())
        dev_changes[name]=np.concatenate(pred)-refdev
    np.savez_compressed(OUT/'b1_response_vectors_dev.npz',row_indices=dev,**dev_changes)
    print('B1',rows,flush=True)


def retrieval():
    """D1: real-event neighbor search across tags; no same-run matches."""
    d,z=data();q=indices(d,'test','kaon');ref=indices(d,'train','proton');rng=np.random.default_rng(319)
    q=rng.choice(q,size=min(400,len(q)),replace=False)
    rows=[]
    for meth in ['vae_s0','ae_s0','endpoint_pca']:
        x=z[meth]; nn=NearestNeighbors(n_neighbors=min(100,len(ref))).fit(x[ref]);ix=nn.kneighbors(x[q],return_distance=False)
        for j,row in enumerate(ix):
            chosen=next((ref[k] for k in row if d.iloc[ref[k]].run!=d.iloc[q[j]].run and d.iloc[ref[k]].event_key!=d.iloc[q[j]].event_key),None)
            if chosen is None: continue
            rows.append(dict(method=meth,query_row=int(q[j]),neighbor_row=int(chosen),
                             query_key=d.iloc[q[j]].event_key,neighbor_key=d.iloc[chosen].event_key,
                             mean_adc_difference=abs(d.iloc[q[j]].mean_adc-d.iloc[chosen].mean_adc),
                             solidity_difference=abs(d.iloc[q[j]].solidity-d.iloc[chosen].solidity)))
    a=pd.DataFrame(rows);a.to_csv(OUT/'d1_cross_tag_retrieval.csv',index=False)
    print('D1',a.groupby('method')[['mean_adc_difference','solidity_difference']].median().to_dict(),flush=True)


def matched_disagreements():
    """D2: prospectively find endpoint-feature matches separated in latent space."""
    d,z=data();pool=indices(d,'test');rng=np.random.default_rng(207)
    pool=rng.choice(pool,size=min(3500,len(pool)),replace=False)
    ef=z['endpoint_pca'][pool];va=z['vae_s0'][pool]
    nbr=NearestNeighbors(n_neighbors=35).fit(ef).kneighbors(ef,return_distance=False)[:,1:]
    pairs=[]
    for i,near in enumerate(nbr):
        candidates=[j for j in near if d.iloc[pool[i]].event_key!=d.iloc[pool[j]].event_key and d.iloc[pool[i]].run!=d.iloc[pool[j]].run]
        if not candidates:continue
        j=max(candidates,key=lambda k:np.linalg.norm(va[i]-va[k]))
        pairs.append(dict(left_row=int(pool[i]),right_row=int(pool[j]),endpoint_pca_distance=float(np.linalg.norm(ef[i]-ef[j])),
                          vae_distance=float(np.linalg.norm(va[i]-va[j])),left_species=d.iloc[pool[i]].species,
                          right_species=d.iloc[pool[j]].species,
                          mean_adc_difference=float(abs(d.iloc[pool[i]].mean_adc-d.iloc[pool[j]].mean_adc)),
                          solidity_difference=float(abs(d.iloc[pool[i]].solidity-d.iloc[pool[j]].solidity))))
    a=pd.DataFrame(pairs);a['distance_ratio']=a.vae_distance/(a.endpoint_pca_distance+1e-6)
    a.sort_values('distance_ratio',ascending=False).to_csv(OUT/'d2_feature_matched_latent_disagreements.csv',index=False)
    print('D2',len(a),flush=True)


def closure():
    """E1: physical-observable closure on deterministic reconstructions."""
    d,z=data();net,device=model();x=np.load(BASE/'images_log1p.npy',mmap_mode='r')
    ii=indices(d,'test');rng=np.random.default_rng(518);ii=rng.choice(ii,size=min(2000,len(ii)),replace=False)
    xb=np.asarray(x[ii]).copy();re=[]
    with torch.no_grad():
        for lo in range(0,len(ii),128):
            t=torch.from_numpy(xb[lo:lo+128]).to(device);mu,_=net.encode(t);re.append(net.decode(mu).cpu().numpy())
    re=np.concatenate(re)
    def metrics(a):
        raw=np.expm1(np.clip(a,0,20));p=raw>1.
        mean=raw.mean((1,2,3));occ=p.mean((1,2,3));profile=raw[:,0].max(axis=2)
        return {'mean_adc_image':mean,'occupancy_image':occ,'profile_peak':profile.max(1),
                'last_quarter_fraction':profile[:,-12:].sum(1)/np.maximum(profile.sum(1),1e-9)}
    m1,m2=metrics(xb),metrics(re);rows=[]
    for key in m1:
        y=m1[key];pr=m2[key]
        rows.append(dict(observable=key,n=len(ii),r2=r2_score(y,pr),bias=float(np.mean(pr-y)),
                         median_absolute_error=float(np.median(abs(pr-y)))))
    pd.DataFrame(rows).to_csv(OUT/'e1_observable_closure.csv',index=False)
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.55))
    for ax,key in zip(axes,['mean_adc_image','profile_peak']):
        y=m1[key];pr=m2[key];ax.scatter(y,pr,s=2,alpha=.13,color='#0077BB',rasterized=True)
        lo=min(np.quantile(y,.005),np.quantile(pr,.005));hi=max(np.quantile(y,.995),np.quantile(pr,.995))
        ax.plot([lo,hi],[lo,hi],color='.4',ls='--',lw=.8);ax.set_xlabel('Original');ax.set_ylabel('Decoded');ax.set_title(key.replace('_',' '),fontsize=9)
    figsave(fig,'e1_observable_closure')
    print('E1',rows,flush=True)


STAGES={'a1':atlas,'a3':reader_efficiency,'a4':bottleneck,'b1':response,'b4':erasure,
        'c1':transfer,'c2':similarity,'d1':retrieval,'d2':matched_disagreements,
        'e1':closure,'e4':posterior}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=STAGES)
    args=p.parse_args();STAGES[args.stage]()


if __name__=='__main__': main()
