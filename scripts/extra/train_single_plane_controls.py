#!/usr/bin/env python3
"""Train matched single-plane VAE controls using the frozen baseline protocol."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score,adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from src.models.build import build_vae
from latent_limit_experiments import BASE,OUT,indices

DEST=OUT/'single_plane_models';DEST.mkdir(parents=True,exist_ok=True)
CONFIG=next((ROOT/'configs').glob('run_0093*'))


def train(plane,seed):
    path=DEST/f'{plane}_s{seed}.pt';zp=DEST/f'{plane}_s{seed}_latents.npy'
    if path.exists() and zp.exists():print('existing',plane,seed,flush=True);return
    d=pd.read_csv(BASE/'manifest.csv');tr=indices(d,'train');dv=indices(d,'dev')
    cfg=yaml.safe_load(CONFIG.read_text());device='mps' if torch.backends.mps.is_available() else 'cpu'
    torch.set_num_threads(4)
    # Match the baseline trainer's resident image tensor. Repeated random
    # memmap reads and host-to-MPS copies made every epoch needlessly slow.
    images=torch.from_numpy(np.load(BASE/'images_log1p.npy')).to(device)
    torch.manual_seed(seed);np.random.seed(seed);net=build_vae(cfg,device);opt=torch.optim.Adam(net.parameters(),lr=.001,weight_decay=.0001)
    rng=torch.Generator().manual_seed(10000+seed);best=float('inf');stale=0;history=[]
    channel=1 if plane=='collection' else 0
    bestpath=DEST/f'{plane}_s{seed}.best.pt'
    for epoch in range(200):
        net.train();order=tr[torch.randperm(len(tr),generator=rng).numpy()];total=0.
        for lo in range(0,len(order),128):
            xb=images[order[lo:lo+128]]
            xmasked=xb.clone();xmasked[:,channel]=0
            opt.zero_grad(set_to_none=True);mu,lv=net.encode(xmasked);re=net.decode(net.reparameterise(mu,lv))
            loss=((re-xb).square()*torch.where(xb>0,10.,1.)).sum((1,2,3)).mean()+.25*(mu.square()+lv.exp()-lv-1).sum(1).mean()
            loss.backward();opt.step();total+=float(loss.detach())*len(xb)
        net.eval();value=0.
        with torch.no_grad():
            for lo in range(0,len(dv),128):
                xb=images[dv[lo:lo+128]]
                xmasked=xb.clone();xmasked[:,channel]=0
                mu,lv=net.encode(xmasked);re=net.decode(mu)
                loss=((re-xb).square()*torch.where(xb>0,10.,1.)).sum((1,2,3)).mean()+.25*(mu.square()+lv.exp()-lv-1).sum(1).mean()
                value+=float(loss)*len(xb)
        value/=len(dv);history.append(dict(epoch=epoch+1,train_objective=total/len(tr),dev_objective=value))
        if value<best-1e-4:
            best=value;stale=0;torch.save({k:v.detach().cpu() for k,v in net.state_dict().items()},bestpath)
        else:stale+=1
        if epoch%5==0 or stale>=20:
            print(plane,seed,epoch+1,'dev',value,'best',best,'stale',stale,flush=True)
            pd.DataFrame(history).to_csv(DEST/f'{plane}_s{seed}_history.csv',index=False)
        if stale>=20:break
    net.load_state_dict(torch.load(bestpath,map_location='cpu',weights_only=True));net.eval()
    torch.save(dict(state_dict={k:v.detach().cpu() for k,v in net.state_dict().items()},plane=plane,seed=seed,
                    best=best,epochs=len(history),manifest_sha256=__import__('hashlib').sha256((BASE/'manifest.csv').read_bytes()).hexdigest()),path)
    bestpath.unlink(missing_ok=True)
    lat=[]
    with torch.no_grad():
        for lo in range(0,len(d),128):
            xb=images[lo:lo+128].clone();xb[:,channel]=0
            mu,_=net.encode(xb);lat.append(mu.cpu().numpy())
    np.save(zp,np.concatenate(lat));print('complete',plane,seed,flush=True)


def evaluate():
    d=pd.read_csv(BASE/'manifest.csv');tr=indices(d,'train');dev=indices(d,'dev');rows=[]
    y=pd.Categorical(d.species,categories=['proton','kaon','muon']).codes
    for plane in ['collection','induction']:
        for seed in [0,1,2]:
            path=DEST/f'{plane}_s{seed}_latents.npy'
            if not path.exists():continue
            a=np.load(path);x=StandardScaler().fit(a[tr]).transform(a)
            gm=GaussianMixture(3,covariance_type='full',reg_covar=1e-4,
                               n_init=5,max_iter=200,random_state=0).fit(x[tr])
            for part in ['test','run_test']:
                ii=indices(d,part);rows.append(dict(plane=plane,seed=seed,partition=part,metric='tag_ari',species='all',
                                                   value=adjusted_rand_score(y[ii],gm.predict(x[ii]))))
            for target in ['mean_adc','solidity']:
                vals=d[target].to_numpy()
                for sp in ['proton','kaon','muon']:
                    fit=indices(d,'dev',sp);reader=Ridge(alpha=10).fit(x[fit],vals[fit])
                    for part in ['test','run_test']:
                        ii=indices(d,part,sp)
                        rows.append(dict(plane=plane,seed=seed,partition=part,metric=target+'_r2',species=sp,
                                         value=r2_score(vals[ii],reader.predict(x[ii]))))
    pd.DataFrame(rows).to_csv(OUT/'g3_single_plane_controls.csv',index=False)
    print('G3',len(rows),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['train','evaluate']);p.add_argument('--plane',choices=['collection','induction']);p.add_argument('--seed',type=int)
    a=p.parse_args()
    if a.stage=='train':
        for plane in ([a.plane] if a.plane else ['collection','induction']):
            for seed in ([a.seed] if a.seed is not None else [0,1,2]):train(plane,seed)
    else:evaluate()
