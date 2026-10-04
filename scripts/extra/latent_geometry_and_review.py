#!/usr/bin/env python3
"""Real-event atlas, decoder local metric, anomaly review and blinded packet."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from latent_limit_experiments import BASE,OUT,STAGE,data,indices,model,figsave
from _beam_data import apply_style,SINGLE_COL,DOUBLE_COL
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def d3_atlas():
    d,z=data();ii=indices(d,'test');x=z['vae_s0'][ii]
    p=PCA(n_components=1).fit(z['vae_s0'][indices(d,'train')]);v=p.transform(x)[:,0]
    goals=np.quantile(v,[.1,.3,.5,.7,.9]);used=set();rows=[]
    for q,target in zip([.1,.3,.5,.7,.9],goals):
        for pos in np.argsort(abs(v-target)):
            key=d.iloc[ii[pos]].event_key
            if key not in used:
                used.add(key);rows.append(dict(quantile=q,row=int(ii[pos]),event_key=key,tag=d.iloc[ii[pos]].species,
                                                mean_adc=float(d.iloc[ii[pos]].mean_adc),solidity=float(d.iloc[ii[pos]].solidity),pc1=float(v[pos])));break
    pd.DataFrame(rows).to_csv(OUT/'d3_real_event_atlas.csv',index=False)
    imgs=np.load(BASE/'images_log1p.npy',mmap_mode='r');apply_style(SINGLE_COL)
    fig,axes=plt.subplots(2,5,figsize=(DOUBLE_COL,3.1));vmax=float(np.quantile(imgs[[r['row'] for r in rows]],.998))
    for c,r in enumerate(rows):
        for plane in range(2):
            ax=axes[plane,c];ax.imshow(imgs[r['row'],plane],origin='lower',vmin=0,vmax=vmax,cmap='viridis')
            ax.set_xticks([]);ax.set_yticks([])
            if plane==0:ax.set_title(f'{int(r["quantile"]*100)}th percentile',fontsize=8)
    axes[0,0].set_ylabel('Collection');axes[1,0].set_ylabel('Induction')
    figsave(fig,'d3_real_event_atlas')
    print('D3',rows,flush=True)


def e3_decoder_metric():
    d,z=data();net,device=model();raw=np.load(BASE/'representations/vae_s0.npy');tr=indices(d,'train')
    mu=raw[tr].mean(0);sd=raw[tr].std(0);ii=indices(d,'test');rng=np.random.default_rng(319);ii=rng.choice(ii,size=100,replace=False)
    x=np.load(BASE/'images_log1p.npy',mmap_mode='r')
    fit=indices(d,'dev','kaon');charge=Ridge(alpha=10).fit(z['vae_s0'][fit],d.iloc[fit].mean_adc).coef_
    topology=Ridge(alpha=10).fit(z['vae_s0'][fit],d.iloc[fit].solidity).coef_
    rows=[];eps=.03
    for start in range(0,len(ii),10):
        subset=ii[start:start+10];v=z['vae_s0'][subset];n=len(subset)
        latent=[]
        for k in range(8):
            u=np.zeros_like(v);u[:,k]=eps
            latent.extend([v+u,v-u])
        batch=np.concatenate(latent)
        with torch.no_grad():
            dec=net.decode(torch.from_numpy((mu+batch*sd).astype('float32')).to(device)).cpu().numpy()
        j=np.stack([(dec[(2*k)*n:(2*k+1)*n]-dec[(2*k+1)*n:(2*k+2)*n])/(2*eps) for k in range(8)],axis=1)
        orig=np.asarray(x[subset]);weights=np.where(orig>0,10.,1.)
        for a,row in enumerate(subset):
            jt=j[a].reshape(8,-1);jw=jt*np.sqrt(weights[a].reshape(-1))[None,:]
            g=jw@jw.T;eig,vectors=np.linalg.eigh(g);ix=np.argmax(eig);lead=vectors[:,ix]
            rows.append(dict(row=int(row),species=d.iloc[row].species,lambda_max=float(eig[-1]),
                             spectral_pr=float(eig.sum()**2/max((eig**2).sum(),1e-20)),
                             lead_charge_cos2=float((lead@charge)**2/max(charge@charge,1e-20)),
                             lead_topology_cos2=float((lead@topology)**2/max(topology@topology,1e-20)),
                             eig1=float(eig[-1]),eig2=float(eig[-2]),eig3=float(eig[-3]),eig4=float(eig[-4]),
                             eig5=float(eig[-5]),eig6=float(eig[-6]),eig7=float(eig[-7]),eig8=float(eig[-8])))
    a=pd.DataFrame(rows);a.to_csv(OUT/'e3_decoder_local_metric.csv',index=False)
    apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.5))
    eig=a[[f'eig{i}' for i in range(1,9)]].to_numpy();shares=eig/np.maximum(eig.sum(1,keepdims=True),1e-20)
    center=np.median(np.cumsum(shares,axis=1),axis=0)
    ax.plot(range(1,9),center,'o-',color='#0077BB',ms=3,lw=1)
    ax.set_xticks(range(1,9));ax.set_xlabel('Leading decoder metric directions');ax.set_ylabel('Cumulative local sensitivity')
    ax.set_ylim(0,1.05);figsave(fig,'e3_decoder_metric')
    print('E3',a[['spectral_pr','lead_charge_cos2','lead_topology_cos2']].median().to_dict(),flush=True)


def f3_candidates():
    d,z=data();rng=np.random.default_rng(130);selected=[]
    errors=np.load(BASE/'representations/vae_s0_reconstruction.npy')
    for sp in ['proton','kaon','muon']:
        ii=indices(d,'test',sp);top=ii[np.argsort(errors[ii])[-5:]]
        random=rng.choice(ii,size=5,replace=False)
        for name,rows in [('high_error',top),('random',random)]:
            for row in rows:selected.append(dict(row=int(row),group=name,species=sp,event_key=d.iloc[row].event_key,
                                                   weighted_error=float(errors[row]),mean_adc=float(d.iloc[row].mean_adc)))
    a=pd.DataFrame(selected);a.to_csv(OUT/'f3_anomaly_review_key.csv',index=False)
    images=np.load(BASE/'images_log1p.npy',mmap_mode='r');apply_style(SINGLE_COL)
    fig,axes=plt.subplots(6,5,figsize=(DOUBLE_COL,6.8));vmax=float(np.quantile(images[a.row],.998))
    for i,r in enumerate(selected):
        ax=axes.flat[i];ax.imshow(images[r['row'],0],origin='lower',vmin=0,vmax=vmax,cmap='viridis');ax.set_xticks([]);ax.set_yticks([])
        ax.text(.02,.96,f'{i+1:02}',transform=ax.transAxes,color='white',va='top',fontsize=8)
    figsave(fig,'f3_anomaly_gallery')
    print('F3','review candidates',len(a),flush=True)


def d4_blind_packet():
    d,z=data();rng=np.random.default_rng(270);pool=indices(d,'test');err=np.load(BASE/'representations/vae_s0_reconstruction.npy')
    top=pool[np.argsort(err[pool])[-40:]];random=rng.choice(pool,size=60,replace=False)
    disagreement=pd.read_csv(OUT/'d2_feature_matched_latent_disagreements.csv').head(40).left_row.to_numpy()
    rows=np.unique(np.r_[top,random,disagreement]);rows=rng.permutation(rows)
    key=pd.DataFrame(dict(study_id=[f'L{i+1:03}' for i in range(len(rows))],row=rows,
                          source_tag=d.iloc[rows].species.to_numpy(),event_key=d.iloc[rows].event_key.to_numpy(),
                          source_group=np.where(np.isin(rows,top),'high_error',np.where(np.isin(rows,disagreement),'latent_disagreement','random'))))
    key.to_csv(OUT/'d4_blind_answer_key.csv',index=False)
    pd.DataFrame(dict(study_id=key.study_id,straight='',endpoint_rise='',kink_branch='',multiple_tracks='',artifact='',ambiguous='',notes='')).to_csv(OUT/'d4_annotation_form.csv',index=False)
    imgs=np.load(BASE/'images_log1p.npy',mmap_mode='r');apply_style(SINGLE_COL)
    with PdfPages(OUT/'d4_blind_morphology_packet.pdf') as pdf:
        for lo in range(0,len(rows),12):
            fig,axes=plt.subplots(6,4,figsize=(8.5,10.8));axes=axes.reshape(6,4)
            for j,row in enumerate(rows[lo:lo+12]):
                rr=(j//2)*2;cc=(j%2)*2
                for plane in range(2):
                    ax=axes[rr:rr+2,cc+plane][0] if False else axes[j//2,cc+plane]
                    ax.imshow(imgs[row,plane],origin='lower',cmap='viridis',vmin=0,vmax=8)
                    ax.set_xticks([]);ax.set_yticks([])
                    if plane==0:ax.set_title(key.iloc[lo+j].study_id,fontsize=8)
            for ax in axes.flat:
                if not ax.images:ax.axis('off')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    print('D4','packet events',len(rows),flush=True)


STAGES={'d3':d3_atlas,'d4':d4_blind_packet,'e3':e3_decoder_metric,'f3':f3_candidates}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=STAGES);a=p.parse_args();STAGES[a.stage]()
