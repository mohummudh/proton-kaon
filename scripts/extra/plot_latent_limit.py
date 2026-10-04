#!/usr/bin/env python3
"""Paper-sized draft figures from completed latent-limit tables; staging only."""
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts/extra')]
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from _beam_data import apply_style,SINGLE_COL,DOUBLE_COL
from latent_limit_experiments import OUT,figsave


def cross_transfer():
    a=pd.read_csv(OUT/'c1_cross_tag_transfer.csv')
    f=a[(a.method=='vae_s0')&(a.partition=='test')&(a.support=='source_5_95')]
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.65))
    for ax,target in zip(axes,['mean_adc','solidity']):
        m=f[f.target.eq(target)].pivot(index='source',columns='destination',values='r2').reindex(index=['proton','kaon','muon'],columns=['proton','kaon','muon'])
        im=ax.imshow(m,cmap='RdBu',vmin=-1,vmax=1)
        ax.set_xticks(range(3),['p','K','MIP']);ax.set_yticks(range(3),['p','K','MIP'])
        ax.set_xlabel('Evaluation sample');ax.set_ylabel('Reader source')
        ax.text(.02,1.03,target.replace('_',' '),transform=ax.transAxes,fontsize=9)
        for i in range(3):
            for j in range(3):
                v=m.iloc[i,j];label=f'{v:.2f}' if abs(v)<10 else f'{v:.0f}'
                ax.text(j,i,label,ha='center',va='center',color='white' if abs(v)>.65 else 'black',fontsize=7)
    fig.colorbar(im,ax=axes,fraction=.04,pad=.03,label='Held-out $R^2$ (colors clipped)')
    fig.savefig(OUT/'staging_figs/c1_cross_tag_transfer.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/'staging_figs/c1_cross_tag_transfer.png',dpi=300,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)

    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.65))
    for ax,target in zip(axes,['mean_adc','solidity']):
        m=f[f.target.eq(target)].pivot(index='source',columns='destination',values='rho').reindex(index=['proton','kaon','muon'],columns=['proton','kaon','muon'])
        im=ax.imshow(m,cmap='viridis',vmin=0,vmax=.85)
        ax.set_xticks(range(3),['p','K','MIP']);ax.set_yticks(range(3),['p','K','MIP'])
        ax.set_xlabel('Evaluation sample');ax.set_ylabel('Reader source')
        ax.text(.02,1.03,target.replace('_',' '),transform=ax.transAxes,fontsize=9)
        for i in range(3):
            for j in range(3):
                v=m.iloc[i,j];ax.text(j,i,f'{v:.2f}',ha='center',va='center',color='white' if v<.4 else 'black',fontsize=7)
    fig.colorbar(im,ax=axes,fraction=.04,pad=.03,label='Spearman $\\rho$')
    fig.savefig(OUT/'staging_figs/c1_rank_transfer.pdf',bbox_inches='tight',pad_inches=.03)
    fig.savefig(OUT/'staging_figs/c1_rank_transfer.png',dpi=300,bbox_inches='tight',pad_inches=.03)
    plt.close(fig)


def seed_stability():
    a=pd.read_csv(OUT/'c2_c3_seed_similarity.csv')
    labels=['VAE seed 0 : 1','VAE seed 0 : 2','AE seed 0 : 1','VAE : AE','VAE : random']
    apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.35))
    yp=np.arange(len(a));ax.barh(yp,a.neighbor_overlap10,color=['#0077BB']*3+['#EE7733','#999999'],height=.58)
    ax.set_yticks(yp,labels);ax.invert_yaxis();ax.set_xlabel('Shared 10-neighbor fraction');ax.set_xlim(0,.8)
    figsave(fig,'c2_seed_neighbor_stability')


def response():
    a=pd.read_csv(OUT/'b1_detector_response.csv')
    apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.4))
    names=[t.replace('_',' ') for t in a.perturbation]
    ax.barh(range(len(a)),a.median_distance,color=['#999999']*4+['#0077BB','#EE7733'],height=.58)
    ax.set_yticks(range(len(a)),names);ax.invert_yaxis();ax.set_xlabel('Median code displacement (train SD units)')
    figsave(fig,'b1_encoder_response')


def erasure():
    a=pd.read_csv(OUT/'b4_linear_concept_erasure.csv')
    f=a[(a.partition=='test')&(a.species=='kaon')]
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.4),sharey=True)
    for ax,erased in zip(axes,['mean_adc','solidity']):
        sub=f[f.erased==erased].pivot(index='target',columns='projection',values='r2').reindex(['mean_adc','solidity'])
        xp=np.arange(2);w=.23
        for i,(key,col,lab) in enumerate([('original','#0077BB','Original'),('erase','#AA3377','Removed'),('random_rank1','#999999','Random direction')]):
            ax.bar(xp+(i-1)*w,sub[key],width=w,color=col,label=lab)
        ax.set_xticks(xp,['Mean ADC','Solidity']);ax.text(.02,1.03,'Remove '+erased.replace('_',' '),transform=ax.transAxes,fontsize=9)
        ax.axhline(0,color='.6',lw=.6)
    axes[0].set_ylabel('Held-out linear $R^2$');axes[1].legend(frameon=False,loc='upper right',fontsize=7)
    figsave(fig,'b4_concept_erasure')


def posterior():
    a=pd.read_csv(OUT/'e4_posterior_readout.csv')
    f=a[a.partition=='test'].pivot(index='species',columns='features',values='r2').reindex(['proton','kaon','muon'])
    apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.4));xp=np.arange(3);w=.23
    for i,(name,color,label) in enumerate([('mean','#0077BB','Mean'),('variance','#EE7733','Log variance'),('both','#AA3377','Both')]):
        ax.bar(xp+(i-1)*w,f[name],width=w,color=color,label=label)
    ax.set_xticks(xp,['Proton','Kaon','MIPs']);ax.set_ylabel('Held-out solidity $R^2$')
    ax.set_ylim(0,.85);ax.legend(frameon=False,loc='upper right');figsave(fig,'e4_posterior_readout')


def plane_mismatch():
    a=pd.read_csv(OUT/'g2_plane_mismatch.csv')
    apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.35))
    keys=['genuine','matched_wrong_plane','random_wrong_plane']
    med=a.set_index('condition').loc[keys].median_weighted_error
    ax.bar(range(3),med,color=['#999999','#0077BB','#EE7733'],width=.65)
    ax.set_xticks(range(3),['Genuine','Matched\nwrong plane','Random\nwrong plane'])
    ax.set_ylabel('Median weighted error');figsave(fig,'g2_plane_mismatch')


def masks():
    a=pd.read_csv(OUT/'b2_local_mask_response.csv')
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.35),sharey=True)
    for ax,axis in zip(axes,['wire','drift']):
        f=a[a.axis==axis];ax.bar(range(3),f.median_shift,color='#0077BB',width=.65)
        ax.set_xticks(range(3),['0–7','20–27','40–47']);ax.set_xlabel(axis.capitalize()+' strip removed')
    axes[0].set_ylabel('Median code displacement (train SD units)')
    figsave(fig,'b2_local_mask_response')


def unseen_upstream():
    a=pd.read_csv(OUT/'a2_unseen_upstream_readout.csv')
    f=a[a.partition=='test'].copy();f['family']=f.method.map(lambda x:'VAE' if x.startswith('vae') else ('AE' if x.startswith('ae') else 'Endpoint PCA'))
    summary=f.groupby(['species','family']).r2.mean().unstack().reindex(['proton','kaon','muon'])
    apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.45));xp=np.arange(3);w=.23
    for i,(name,col) in enumerate([('VAE','#0077BB'),('AE','#999999'),('Endpoint PCA','#EE7733')]):
        ax.bar(xp+(i-1)*w,summary[name],width=w,color=col,label=name)
    ax.set_xticks(xp,['Proton','Kaon','MIPs']);ax.set_ylabel('Held-out upstream charge $R^2$')
    ax.set_ylim(-.05,.85);ax.legend(frameon=False,loc='upper right');figsave(fig,'a2_unseen_upstream')


def short_light_support():
    path=OUT/'f2_short_light_support.csv'
    if not path.exists():return
    a=pd.read_csv(path)
    short=pd.read_csv(OUT/'short_light_metadata.csv')
    original=pd.read_csv(ROOT/'output/representation_baselines/manifest.csv')
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,2,figsize=(DOUBLE_COL,2.65))
    for name,selection,col in [('Short light',short.height_col,'#AA3377'),
                               ('Long MIP',original.loc[original.species.eq('muon'),'height'],'#999999'),
                               ('Kaon window',original.loc[original.species.eq('kaon'),'height'],'#EE7733')]:
        axes[0].hist(selection,bins=np.arange(0,201,8),density=True,histtype='step',
                     linewidth=1.3,color=col,label=name)
    axes[0].set_xlabel('Collection-plane length (wires)');axes[0].set_ylabel('Density')
    axes[0].legend(frameon=False,loc='upper left',fontsize=7)
    names=['long_mip','kaon_window','proton'];labels=['Long MIP','Kaon window','Proton']
    colors=['#AA3377','#EE7733','#0077BB']
    med=a.groupby('reference').relative_to_self.median().reindex(names)
    axes[1].bar(range(3),med,color=colors,width=.64)
    axes[1].axhline(1,color='.4',lw=.7,linestyle='--')
    axes[1].set_xticks(range(3),labels);axes[1].set_ylabel('Short-light 5-NN distance /\nreference self-distance')
    for ax,letter in zip(axes,['a','b']):ax.text(.02,1.03,letter,transform=ax.transAxes,fontsize=9)
    figsave(fig,'f2_short_light_support')

    detail=OUT/'f2_short_light_pairwise.csv'
    if detail.exists():
        b=pd.read_csv(detail);b=b[b.length_bin.ne('all')]
        apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.35))
        xp=np.arange(len(b));val=b.fraction_kaon_closer.to_numpy()
        ax.errorbar(xp,val,yerr=[val-b.low.to_numpy(),b.high.to_numpy()-val],
                    color='#EE7733',marker='o',markersize=4,linewidth=1.1,capsize=2)
        ax.axhline(.5,color='.5',linestyle='--',linewidth=.7)
        ax.set_xticks(xp,b.length_bin);ax.set_ylim(.45,1.0)
        ax.set_xlabel('Short-light track length (wires)')
        ax.set_ylabel('Fraction nearer kaon window\nthan long-MIP reference')
        figsave(fig,'f2_short_light_by_length')
    reverse=OUT/'f2_reverse_reference_preference.csv'
    if reverse.exists():
        r=pd.read_csv(reverse)
        p=r.groupby('query').fraction_short_closer.agg(['mean','min','max']).reindex(['kaon_window','proton'])
        apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.25))
        xp=np.arange(2);v=p['mean'].to_numpy()
        ax.bar(xp,v,color=['#EE7733','#0077BB'],width=.58)
        ax.errorbar(xp,v,yerr=[v-p['min'].to_numpy(),p['max'].to_numpy()-v],
                    fmt='none',ecolor='black',elinewidth=.7,capsize=2)
        ax.set_xticks(xp,['Kaon window','Proton']);ax.set_ylim(0,1)
        ax.set_ylabel('Fraction nearer short-light\nthan long-MIP reference')
        figsave(fig,'f2_reverse_reference_control')
    methods=OUT/'f2_method_support_controls.csv'
    if methods.exists():
        m=pd.read_csv(methods)
        p=m.pivot(index='reference',columns='method',values='relative_to_self').reindex(
            ['long_mip','kaon_window','proton'])
        apply_style(SINGLE_COL);fig,ax=plt.subplots(figsize=(SINGLE_COL,2.4))
        xp=np.arange(3);w=.23
        for j,(name,col) in enumerate([('vae','#0077BB'),('ae','#EE7733'),('random','#999999')]):
            ax.bar(xp+(j-1)*w,p[name],width=w,color=col,label=name.upper() if name!='random' else 'Random CNN')
        ax.axhline(1,color='.5',linestyle='--',lw=.7)
        ax.set_xticks(xp,['Long MIP','Kaon window','Proton'])
        ax.set_ylabel('Short-light 5-NN distance /\nreference self-distance')
        ax.legend(frameon=False,loc='upper right',fontsize=7)
        figsave(fig,'f2_method_support_controls')


def single_plane_controls():
    path=OUT/'g3_single_plane_controls.csv'
    if not path.exists():return
    a=pd.read_csv(path);p=pd.read_csv(ROOT/'output/representation_baselines/probes_all.csv')
    c=pd.read_csv(ROOT/'output/representation_baselines/clustering_all.csv')
    a=a[(a.partition=='test') & ((a.species=='kaon') | a.metric.eq('tag_ari'))]
    rows=[]
    for metric in ['tag_ari','mean_adc_r2','solidity_r2']:
        for plane in ['collection','induction']:
            vals=a[(a.plane==plane)&(a.metric==metric)].value.to_numpy()
            rows.append(dict(plane=plane,metric=metric,mean=vals.mean(),minimum=vals.min(),maximum=vals.max()))
        if metric=='tag_ari':
            vals=c[(c.method=='vae')&(c.partition=='test')&(c.k==3)&(c.gmm_seed==0)].groupby('seed').ari.first().to_numpy()
        else:
            vals=p[(p.method=='vae')&(p.partition=='test')&(p.species=='kaon')&
                   (p.target==metric.removesuffix('_r2'))].groupby('seed').r2_linear.first().to_numpy()
        rows.append(dict(plane='both',metric=metric,mean=vals.mean(),minimum=vals.min(),maximum=vals.max()))
    s=pd.DataFrame(rows);s.to_csv(OUT/'g3_single_plane_summary.csv',index=False)
    apply_style(SINGLE_COL);fig,axes=plt.subplots(1,3,figsize=(DOUBLE_COL,2.45))
    for ax,metric,label in zip(axes,['tag_ari','mean_adc_r2','solidity_r2'],
                                ['Three-component tag ARI','Kaon mean ADC $R^2$','Kaon solidity $R^2$']):
        vals=s[s.metric.eq(metric)].set_index('plane').reindex(['collection','induction','both'])
        xp=np.arange(3);colors=['#0077BB','#EE7733','#999999']
        ax.bar(xp,vals['mean'],color=colors,width=.62)
        ax.errorbar(xp,vals['mean'],yerr=[vals['mean']-vals['minimum'],vals['maximum']-vals['mean']],
                    fmt='none',ecolor='black',elinewidth=.7,capsize=2)
        ax.set_xticks(xp,['Col.','Ind.','Both']);ax.set_ylabel(label)
        ax.axhline(0,color='.5',lw=.6)
    figsave(fig,'g3_single_plane_controls')


def main():
    for fn in [cross_transfer,seed_stability,response,erasure,posterior,plane_mismatch,masks,unseen_upstream]:fn()
    short_light_support();single_plane_controls()

if __name__=='__main__':main()
