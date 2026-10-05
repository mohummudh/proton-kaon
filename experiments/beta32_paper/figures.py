#!/usr/bin/env python3
"""Render isolated comparison figures with the repository's research style."""
import json
from pathlib import Path
import sys
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
sys.path[:0]=[str(HERE),str(ROOT/'scripts/extra'),str(ROOT)]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from _beam_data import apply_style,SINGLE_COL,DOUBLE_COL,COLOURS,DISPLAY,SPECIES
from evaluate_study import load,cluster_summary
from plot_tsne import embed

PRIMARY='bal9419_d8_s0_b32'
DEST=HERE/'figures'
CAPTIONS={
 'training_objective':'Historical and beta=32 primary training histories. Left: sampled validation reconstruction. Right: beta-weighted KL share. Checkpoint rules are identical; seed and hardware differ in this historical comparison.',
 'species_tsne':'Separate raw-latent t-SNE projections for historical beta=0.5 (top) and beta=32 seed zero (bottom), with proton, kaon-window and MIP tags highlighted in matching columns. Grey points are the other tags. Neither projection enters any clustering fit. Positions and orientations cannot be compared between models.',
 'proxy_tsne':'The same t-SNE coordinates within each model, coloured by mean ADC (left) and solidity (right). Historical beta=0.5 is above beta=32. Colour limits use the shared observed population; the projection is descriptive.',
 'proxy_auc':'Within-tag, validation-only median-split logistic probes. Open circles: historical beta=0.5; filled squares: beta=32 seed zero. Bars are percentile intervals from 2,000 resamples of frozen out-of-fold predictions, and omit VAE training uncertainty.',
 'cluster_count':'Original full-covariance GMM scan on raw posterior means. Left: majority tag agreement; right: fraction of all candidates in clusters at least 85% pure against the tags used to name them. Solid black: beta=32; dashed grey: historical beta=0.5. Fine clusters are descriptive rather than unique physical populations.',
 'composition_and_mass':'Primary beta=32 k=37 GMM. Left: cluster composition, widths proportional to population; kaon segments shaded by within-cluster median beamline mass where at least 50 kaon candidates are present. Right: mass in the kaon window split by majority-tag assignment. Mass does not enter training or GMM fitting. The tag itself is defined by this mass window.',
 'anchored_mass':'Kaon-window mass split by the tag-anchored comparison at beta=32. These densities use proton and MIP tags during fitting and are a supporting comparison, not fully unsupervised separation.',
 'gmm_stability':'Beta=32 k=36-41, five GMM initialization seeds. Left: mean proton-minus-kaon group mass shift with one seed standard deviation. Right: mean pairwise partition ARI. Error bars show initialization variability, not event-bootstrap confidence intervals.',
 'training_size':'Original training-size grid at beta=32. Points show means and standard deviations over three seeds; grey dashed curves show the historical beta=0.5 means. Panels: sampled validation reconstruction, GMM k=3 tag agreement, calorimetry AUC and topology AUC. Missing rungs remain absent.',
 'latent_capacity':'Original capacity grid at beta=32. Points show three-seed means and standard deviations; grey dashed curves show historical beta=0.5 means. Panels: active coordinates, sampled validation reconstruction, KL share, GMM k=3 agreement, calorimetry AUC and topology AUC. Missing capacities remain absent.'}

def finish(fig,stem):
    DEST.mkdir(exist_ok=True)
    fig.savefig(DEST/(stem+'.pdf'))
    fig.savefig(DEST/(stem+'.png'))
    plt.close(fig)
    print('FIGURE',stem,flush=True)

def clean(ax):
    ax.spines[['top','right']].set_visible(False)

def histories():
    old=HERE/'reference/historical_training.json';new=HERE/'runs'/PRIMARY/'history.json'
    if not old.exists() or not new.exists():return
    apply_style(SINGLE_COL)
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.5))
    for p,colour,style,label,beta in [(old,'.55','--',r'$\beta=0.5$ (historical)',.5),
        (new,'black','-',r'$\beta=32$',32)]:
        obj=json.loads(p.read_text());table=pd.DataFrame(obj['history'] if isinstance(obj,dict) else obj)
        axes[0].plot(table.epoch,table.val_recon,c=colour,ls=style,lw=1.,label=label)
        axes[1].plot(table.epoch,beta*table.val_kl/table.val_loss*100,c=colour,ls=style,lw=1.,label=label)
    axes[0].set_ylabel('Validation reconstruction (weighted)')
    axes[1].set_ylabel(r'$\beta\,KL$ / total objective [%]')
    for ax in axes:ax.set_xlabel('Epoch');clean(ax)
    axes[0].legend(frameon=False,loc='upper right');fig.tight_layout();finish(fig,'training_objective')

def embeddings():
    if not (HERE/'runs'/PRIMARY/'complete.json').exists():return
    apply_style(SINGLE_COL)
    datasets=[]
    for stem in ['historical',PRIMARY]:
        z,df,_,_=load(stem)
        e=embed(z,30,0,HERE/'cache'/stem)
        datasets.append((e,df))
    fig,axes=plt.subplots(2,3,figsize=(DOUBLE_COL,4.5))
    for row,(e,df) in enumerate(datasets):
        for col,sp in enumerate(SPECIES):
            ax=axes[row,col];m=(df.species==sp).to_numpy()
            ax.scatter(e[~m,0],e[~m,1],s=.55,c='.83',rasterized=True,linewidths=0)
            ax.scatter(e[m,0],e[m,1],s=.55,c=COLOURS[sp],rasterized=True,linewidths=0)
            ax.set_xticks([]);ax.set_yticks([])
            ax.set_xlabel('t-SNE 1')
            if col==0:ax.set_ylabel(('Historical '+r'$\beta=0.5$' if row==0 else r'$\beta=32$')+'\nt-SNE 2')
            ax.legend(handles=[Line2D([],[],marker='o',ls='none',ms=3,color=COLOURS[sp],label=DISPLAY[sp])],
                      frameon=False,loc='upper right')
    fig.tight_layout(pad=.7);finish(fig,'species_tsne')
    fig,axes=plt.subplots(2,2,figsize=(DOUBLE_COL,5.1))
    for col,(feat,label) in enumerate([('mean_adc','Calorimetry proxy'),('solidity','Topology proxy')]):
        v=datasets[0][1][feat].to_numpy();limits=np.nanpercentile(v,[2,98])
        for row,(e,df) in enumerate(datasets):
            ax=axes[row,col]
            idx=np.random.default_rng(0).permutation(len(e))
            sc=ax.scatter(e[idx,0],e[idx,1],c=df[feat].to_numpy()[idx],s=.65,cmap='viridis',
                          vmin=limits[0],vmax=limits[1],rasterized=True,linewidths=0)
            ax.set_xticks([]);ax.set_yticks([]);ax.set_xlabel('t-SNE 1')
            if col==0:ax.set_ylabel(('Historical '+r'$\beta=0.5$' if row==0 else r'$\beta=32$')+'\nt-SNE 2')
            fig.colorbar(sc,ax=ax,pad=.025).set_label(label)
    fig.tight_layout();finish(fig,'proxy_tsne')

def probes():
    paths=[HERE/'results'/s/'probes.csv' for s in ['historical',PRIMARY]]
    if not all(p.exists() for p in paths):return
    tables=[pd.read_csv(p) for p in paths];apply_style(SINGLE_COL)
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.6),sharey=True)
    for col,feat in enumerate(['mean_adc','solidity']):
        ax=axes[col]
        for i,sp in enumerate(SPECIES):
            for which,t in enumerate(tables):
                r=t[(t.feature==feat)&(t.species==sp)].iloc[0]
                y=i+(-.10 if which==0 else .10)
                ax.errorbar(r.auc,y,xerr=[[max(0,r.auc-r.lo)],[max(0,r.hi-r.auc)]],
                    fmt='o' if which==0 else 's',mfc='white' if which==0 else COLOURS[sp],
                    color=COLOURS[sp],ms=4,capsize=2,lw=.8)
        ax.set_yticks(range(3));ax.set_yticklabels([DISPLAY[s] for s in SPECIES])
        ax.set_xlabel(('Calorimetry' if col==0 else 'Topology')+' proxy AUC')
        ax.set_xlim(.70,.985);clean(ax)
    axes[1].legend(handles=[Line2D([],[],marker='o',mfc='white',c='.3',ls='none',label=r'Historical $\beta=0.5$'),
        Line2D([],[],marker='s',c='.3',ls='none',label=r'$\beta=32$')],frameon=False,loc='lower right',fontsize=7)
    axes[0].invert_yaxis()
    fig.tight_layout();finish(fig,'proxy_auc')

def clusters():
    apply_style(SINGLE_COL)
    fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.5));hasnew=False
    for stem,col,style,label in [('historical','.55','--',r'Historical $\beta=0.5$'),(PRIMARY,'black','-',r'$\beta=32$')]:
        rows=[]
        for k in range(3,51):
            p=HERE/f'results/{stem}/gmm_k{k}_seed0.json'
            if p.exists():rows.append(json.loads(p.read_text()))
        if not rows:continue
        if stem==PRIMARY:hasnew=True
        t=pd.DataFrame(rows)
        for ax,key in zip(axes,['purity','fraction_in_85']):
            ax.plot(t.k,t[key],c=col,ls=style,marker='o' if stem==PRIMARY else None,ms=2,lw=.9,label=label)
    if hasnew:
        axes[0].set_ylabel('Majority tag agreement');axes[1].set_ylabel('Fraction in >=85%-pure clusters')
        for ax in axes:ax.set_xlabel('GMM components k');clean(ax)
        axes[1].legend(frameon=False,loc='lower right');fig.tight_layout();finish(fig,'cluster_count')
    else:plt.close(fig)
    path=HERE/'cache'/PRIMARY/'gmm_labels_k37_seed0.npy'
    if not path.exists():return
    z,df,_,_=load(PRIMARY);lab=np.load(path);result,counts,mapped=cluster_summary(lab,df,37)
    mass=df.beamline_mass.to_numpy();truth=pd.Categorical(df.species,categories=SPECIES).codes
    sizes=counts.sum(1);frac=counts/sizes[:,None];order=np.argsort(-(frac[:,0]-frac[:,2]))
    fig,(ax,bx)=plt.subplots(1,2,figsize=(DOUBLE_COL,2.65))
    widths=sizes[order]/sizes.sum();edges=np.r_[0,np.cumsum(widths)]
    norm=Normalize(450,600);cm=plt.get_cmap('YlOrRd');bottom=np.zeros(37)
    for i,sp in enumerate(SPECIES):
        colours=[]
        for c in order:
            idx=(lab==c)&(truth==1)&np.isfinite(mass)
            colours.append(cm(norm(np.median(mass[idx]))) if sp=='kaon' and idx.sum()>=50
                           else '.78' if sp=='kaon' else COLOURS[sp])
        ax.bar(edges[:-1],frac[order,i]*100,width=widths,align='edge',bottom=bottom,
               color=colours,edgecolor='white',linewidth=.25)
        bottom+=frac[order,i]*100
    ax.set_ylim(0,100);ax.set_xlim(0,1);ax.set_xlabel('Cumulative fraction of candidates')
    ax.set_ylabel('Cluster composition [%]');ax.set_xticks([0,.25,.5,.75,1]);ax.set_xticklabels(['0','25%','50%','75%','100%'])
    ax.legend(handles=[Patch(color=COLOURS['proton'],label='Proton'),
        Patch(color='#EE7733',label='Kaon (mass shade)'),Patch(color=COLOURS['muon'],label='MIPs')],
        ncol=3,loc='lower left',bbox_to_anchor=(0,1.01),frameon=False,fontsize=8,
        handlelength=.8,columnspacing=.7,borderaxespad=0)
    fig.colorbar(plt.cm.ScalarMappable(norm=norm,cmap=cm),ax=ax,pad=.025).set_label('Kaon-window median mass [MeV]')
    edges=np.linspace(350,650,26)
    bx.hist(mass[(truth==1)&np.isfinite(mass)],bins=edges,density=True,histtype='step',lw=1.,
            color='.5',label='All kaon-window (8,227)')
    for i,sp in enumerate(SPECIES[:2]):
        vals=mass[(truth==1)&(mapped==i)&np.isfinite(mass)]
        if len(vals):bx.hist(vals,bins=edges,density=True,histtype='step',lw=1.,color=COLOURS[sp],
                            label=f'{DISPLAY[sp]} majority ({len(vals):,})')
    bx.set_xlabel('Beamline mass [MeV]');bx.set_ylabel('Density');clean(bx)
    bx.legend(frameon=False,loc='upper left',fontsize=8)
    fig.tight_layout();finish(fig,'composition_and_mass')

def supplementary():
    apply_style(SINGLE_COL)
    p=HERE/'results'/PRIMARY/'stability.csv'
    if p.exists():
        t=pd.read_csv(p);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.4))
        axes[0].errorbar(t.k,t.mean_mass_shift,yerr=t.sd_mass_shift,color='black',fmt='o-',ms=3,capsize=2,lw=.8)
        axes[0].set_ylabel('Median mass shift [MeV]')
        axes[1].plot(t.k,t.pairwise_ari_mean,'o-',color='black',ms=3,lw=.8)
        axes[1].set_ylabel('Mean pairwise partition ARI');axes[1].set_ylim(0,1)
        for ax in axes:ax.set_xlabel('GMM components k');clean(ax)
        fig.tight_layout();finish(fig,'gmm_stability')
    p=HERE/'cache'/PRIMARY/'anchored_labels.npy'
    if p.exists():
        _,df,_,_=load(PRIMARY);labs=np.load(p);mass=df.beamline_mass.to_numpy()
        fig,ax=plt.subplots(figsize=(SINGLE_COL,2.65))
        for i,sp in enumerate(SPECIES):
            vals=mass[(df.species=='kaon').to_numpy()&(labs==i)&np.isfinite(mass)]
            ax.hist(vals,bins=np.linspace(350,650,26),density=True,histtype='step',lw=1,color=COLOURS[sp],
                    label=f'{DISPLAY[sp]} ({len(vals):,})')
        ax.set_xlabel('Beamline mass [MeV]');ax.set_ylabel('Density');clean(ax)
        ax.legend(frameon=False,loc='upper left',fontsize=7);fig.tight_layout();finish(fig,'anchored_mass')

def curve(ax,table,col,x,colour,label=None,ls='-'):
    if col not in table or len(table)==0:return
    a=table.groupby(x)[col].agg(['mean','std']).sort_index()
    ax.errorbar(a.index,a['mean'],yerr=a['std'].fillna(0),color=colour,ls=ls,
                marker='o' if ls=='-' else None,ms=2.2,capsize=1.5,lw=.8,label=label)

def sweeps():
    rows=[]
    for p in (HERE/'results').glob('*/scan_metrics.json'):rows.append(json.loads(p.read_text()))
    if not rows:return
    allnew=pd.DataFrame(rows);apply_style(SINGLE_COL)
    for family,stem,x,naxes in [('training_size','training_size','n_train',4),('latent_capacity','latent_capacity','latent',6)]:
        new=allnew[allnew.family=='training_size'] if family=='training_size' else allnew[(allnew.tag=='pool8227_tr50')&(allnew.beta==32)]
        if not len(new):continue
        old=pd.read_csv(HERE/f'reference/{family}_beta05.csv')
        fig,axes=plt.subplots(2,2 if naxes==4 else 3,figsize=(DOUBLE_COL,4.8));axes=axes.ravel()
        recon=axes[0] if naxes==4 else axes[1]
        curve(recon,new,'val_recon',x,'black',r'$\beta=32$');curve(recon,old,'val_recon',x,'.55',r'Historical $\beta=0.5$','--')
        recon.set_ylabel('Validation reconstruction (weighted)');recon.legend(frameon=False,fontsize=6.5)
        agreement=axes[1] if naxes==4 else axes[3]
        for col,colour,label in [('ari','.25','ARI'),('purity','.65','Tag agreement')]:
            curve(agreement,new,col,x,colour,label);curve(agreement,old,col,x,colour,ls='--')
        agreement.set_ylabel('GMM k=3 agreement');agreement.legend(frameon=False,fontsize=6.5)
        for ax,feature,label in zip(axes[-2:],['mean_adc','solidity'],['Calorimetry proxy AUC','Topology proxy AUC']):
            for sp in SPECIES:
                curve(ax,new,f'auc_{feature}_{sp}',x,COLOURS[sp],DISPLAY[sp])
                curve(ax,old,f'auc_{feature}_{sp}',x,COLOURS[sp],ls='--')
            ax.set_ylabel(label);ax.legend(frameon=False,fontsize=6.5)
        if naxes==6:
            for col,colour,label in [('n_active_dims','black',r'Var($\mu$)>0.01'),
                    ('n_dims_95var','.5','95% of variance'),('participation_ratio','.3','Participation ratio')]:
                curve(axes[0],new,col,x,colour,label);curve(axes[0],old,col,x,colour,ls='--')
            limits=[4,128]
            axes[0].plot(limits,limits,':',color='.75',lw=.6,label='All available')
            axes[0].set_ylabel('Effective latent dimensions')
            axes[0].legend(frameon=False,fontsize=7,loc='upper left')
            curve(axes[2],new,'kl_fraction',x,'black');curve(axes[2],old,'kl_fraction',x,'.55',ls='--')
            axes[2].set_ylabel(r'$\beta\,KL$ / total objective')
            from matplotlib.ticker import PercentFormatter
            axes[2].yaxis.set_major_formatter(PercentFormatter(1.,decimals=0))
        for ax in axes:ax.set_xlabel('Total training images' if x=='n_train' else 'Latent dimension');clean(ax)
        fig.tight_layout();finish(fig,stem)

def main():
    with threadpool_limits(limits=2):
        histories();probes();clusters();supplementary();sweeps();embeddings()
    present=[stem for stem in CAPTIONS if (DEST/(stem+'.pdf')).exists()]
    text=['# Figure captions','']
    for stem in present:text += [f'## {stem}', '',CAPTIONS[stem],'',f'![{stem}](figures/{stem}.png)','']
    (HERE/'FIGURE_CAPTIONS.md').write_text('\n'.join(text))

if __name__=='__main__':main()
