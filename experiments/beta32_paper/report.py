#!/usr/bin/env python3
"""Build an honest, claim-by-claim report from completed measurements only."""
from pathlib import Path
import json
import sys
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from run_study import write_json
import pandas as pd
import numpy as np

PRIMARY='bal9419_d8_s0_b32'

def read(path):
    return json.loads(path.read_text()) if path.exists() else None

def fmt(value,digits=3):
    return 'pending' if value is None else f'{value:.{digits}f}'

def comparison(label,old,new,digits=3):
    delta=None if old is None or new is None else new-old
    return f'| {label} | {fmt(old,digits)} | {fmt(new,digits)} | {fmt(delta,digits)} |'

def make():
    plan=read(HERE/'plan.json')
    complete=[t for t in plan if (HERE/'runs'/t['id']/'complete.json').exists()]
    evaluated=[t for t in plan if (HERE/'results'/t['id']/'scan_metrics.json').exists()]
    old=read(HERE/'results/historical/gmm_k37_seed0.json')
    new=read(HERE/'results'/PRIMARY/'gmm_k37_seed0.json')
    hist=read(HERE/'runs'/PRIMARY/'history.json') or []
    # Track exactly the epoch accepted by the min_delta selection rule.
    best=float('inf'); selected=None
    for r in hist:
        if r['val_loss']<best-1e-4: best=r['val_loss']; selected=r
    final=(HERE/'runs'/PRIMARY/'complete.json').exists()
    required=['stability.csv','contamination.json','two_sample.json']
    supplementary=all((HERE/'results'/PRIMARY/name).exists() for name in required)
    counts=sum((HERE/f'results/{PRIMARY}/gmm_k{k}_seed0.json').exists() for k in range(3,51))
    full=len(complete)==len(plan) and len(evaluated)==len(plan) and supplementary and counts==48
    lines=['# Paper reproduction with beta = 32','',
        '**Status: '+('complete numerical reproduction.' if full else 'in progress; pending entries are not results.')+'**','',
        f'{len(complete)}/{len(plan)} training runs complete; {len(evaluated)}/{len(plan)} representation summaries complete.','',
        'The reference is the nine-page `paper_reference.pdf` copied from the supplied `neurips.pdf`. '
        'This study concerns that paper, including Appendices A-C. It does not substitute the later event/run-disjoint baseline protocol.','',
        *(['## First completed main-model findings','',
            f'- KL is {100*32*selected["val_kl"]/selected["val_loss"]:.2f}% of the selected validation objective; beta=32 does not yield 50/50 after retraining.',
            f'- k=37 overall tag agreement changes from {100*old["purity"]:.2f}% to {100*new["purity"]:.2f}%.',
            f'- Kaon-group tag purity changes from {100*old["group_purity_kaon"]:.2f}% to {100*new["group_purity_kaon"]:.2f}%; the primary run does not retain the paper\'s above-75%-for-all-tags claim.'
                if new['group_purity_kaon']<.75 else f'- Kaon-group tag purity is {100*new["group_purity_kaon"]:.2f}%.',
            f'- The unsupervised kaon-window mass shift remains positive: {new["mass_shift"]:+.2f} MeV versus {old["mass_shift"]:+.2f} MeV historically.',
            '- These are seed-zero main-model findings. The report below distinguishes the fresh paired control, additional seeds and still-pending appendix results.',''] if final and old and new else []),
        '## What was held fixed','',
        'Original 27,657 two-plane 48 x 48 images; log1p; four encoder blocks [32, 64, 128, 256]; '
        'latent dimension eight for the main model; weighted squared error summed over pixels with 10x '
        'weight above 0.01; summed Gaussian KL; Adam lr=0.001, weight_decay=0.0001; batch size 32; '
        '200-epoch cap; stochastic validation; patience 20; min_delta=0.0001; restore the best accepted checkpoint. '
        'The original 9,419/18,238 split is copied byte-for-byte: train 3,139/3,140/3,140 and '
        'validation 7,327/5,087/5,824 for proton/kaon/MIPs. Labels remain absent from the VAE loss and unsupervised GMM fitting.','',
        'The only intended objective change is beta=0.5 to beta=32. The original main checkpoint was '
        'unseeded, so identical original initialization cannot be reproduced. New main runs use explicit '
        'seeds 0/1/2, with seed zero primary. A fresh beta=0.5 seed-zero control uses the same initialization '
        'and shuffle seed as the primary beta=32 run. Historical comparisons and paired comparisons are distinguished. '
        'Hardware is the Mac MPS GPU; bitwise equality to historical GPU runs is not claimed.','',
        'The isolated trainer was checked against the original trainer on an uneven-batch, two-epoch '
        'fixture: losses and selected model weights matched exactly. Input, split and original source hashes '
        'are recorded in `provenance.json`, `plan.json`, and `reference/sources.json`. Checkpoints retain optimizer '
        'and CPU/device/shuffle RNG states for epoch-level restart.','',
        '## Loss balance and reconstruction','']
    if selected:
        share=32*selected['val_kl']/selected['val_loss']
        lines += [f'The {"selected checkpoint" if final else "best epoch so far (provisional)"} is epoch '
            f'{selected["epoch"]}. Its logged stochastic validation reconstruction is '
            f'{selected["val_recon"]:.2f}, unweighted KL {selected["val_kl"]:.2f}, beta x KL '
            f'{32*selected["val_kl"]:.2f}, and total loss {selected["val_loss"]:.2f}. '
            f'KL contributes **{100*share:.2f}%**, not an assumed 50%.','']
    else: lines += ['Training has not produced a validation checkpoint yet.','']
    metric=read(HERE/'results'/PRIMARY/'scan_metrics.json')
    historical=read(HERE/'reference/historical_training.json')
    if metric and historical:
        previous=min(historical['history'],key=lambda x:x['val_loss'])
        previous_capacity=read(HERE/'results/historical/latent_capacity.json') or {}
        lines += ['| Quantity | Historical beta=0.5 | beta=32 seed 0 | Change |',
            '|---|---:|---:|---:|',
            comparison('Validation reconstruction (stochastic)',previous['val_recon'],metric['val_recon'],2),
            comparison('Validation KL',previous['val_kl'],metric['val_kl'],2),
            comparison('KL share',.5*previous['val_kl']/previous['val_loss'],metric['kl_fraction'],4),
            f'| Active mean coordinates (Var > 0.01) | 8 | {metric["n_active_dims"]} | {metric["n_active_dims"]-8} |',
            comparison('Participation ratio',previous_capacity.get('participation_ratio'),metric['participation_ratio']),
            comparison('Total variance of posterior means',previous_capacity.get('latent_var_total'),metric['latent_var_total'],2),'',
            'Total objectives have different beta weights and should not be interpreted as a direct reconstruction ranking. '
            'Mean-decoded validation reconstruction and KL are also retained in `scan_metrics.json`; they are separate '
            'from the original sampled objective.','']
    lines += ['## Figures 1-3: species structure and physical probes','',
        'The detector example images and their interpretation are unchanged input data. New t-SNE projections '
        'use the original raw posterior means, perplexity 30, PCA initialization, automatic learning rate, '
        '1,000 iterations and seed zero. Projections are fitted separately; positions and orientations are not '
        'comparable between models, and visual spacing is not a clustering metric.','']
    oldprobes=HERE/'results/historical/probes.csv'; newprobes=HERE/'results'/PRIMARY/'probes.csv'
    if oldprobes.exists() and newprobes.exists():
        a=pd.read_csv(oldprobes); b=pd.read_csv(newprobes)
        lines += ['The original six AUC values reproduce the paper to its displayed precision. '
            'New probes use the same within-species validation median split, five stratified folds, '
            'standardization inside each fold, and logistic reader. Intervals resample frozen out-of-fold '
            'predictions 2,000 times; they omit VAE training uncertainty.','',
            '| Proxy / tag | Historical AUC | beta=32 AUC | Change | beta=32 95% interval |',
            '|---|---:|---:|---:|---|']
        for r in b.itertuples():
            before=a[(a.feature==r.feature)&(a.species==r.species)].iloc[0]
            lines.append(f'| {r.feature} / {r.species} | {before.auc:.3f} | {r.auc:.3f} | '
                         f'{r.auc-before.auc:+.3f} | {r.lo:.3f}-{r.hi:.3f} |')
        lines += ['']
        decreases=(b.merge(a,on=['feature','species'],suffixes=('_new','_old')).eval('auc_new < auc_old')).sum()
        lines += [f'{decreases} of six linear proxy AUCs decrease in the primary historical comparison. '
            'This measures access to the named observed proxies within each tag, not true-species accuracy.','']
    else:lines+=['New checkpoint probe results are pending.','']
    lines += ['## Figure 4 and Appendix B: unsupervised clustering and mass corroboration','',
        'Full-covariance GMMs use raw, unstandardized posterior means and the original pooled '
        'training/validation rows, 20 starts, default covariance regularization, and seed zero. '
        'Cluster majority tags are read back after fitting. Their purities are descriptive agreement '
        'with imperfect beamline tags; they are not independently held-out naming accuracy.','']
    if old and new:
        lines += ['| Quantity at k=37 | Historical beta=0.5 | beta=32 seed 0 | Change |',
            '|---|---:|---:|---:|']
        keys=[('Overall tag agreement','purity',3),('Clusters at least 80% pure','n_clusters_80',0),
            ('Sample fraction in those clusters','fraction_in_80',3),
            ('Proton-group tag purity','group_purity_proton',3),('Kaon-group tag purity','group_purity_kaon',3),
            ('MIP-group tag purity','group_purity_muon',3),('At least 90%-kaon clusters','clean_kaon_clusters',0),
            ('Kaon candidates in those clusters','clean_kaon_count',0),('Combined kaon-tag purity','clean_kaon_group_purity',3),
            ('Kaon-window candidates in proton-majority clusters','kaon_in_proton_group',0),
            ('Fraction of kaon window flagged','flagged_kaon_fraction',3),
            ('Median mass in proton-majority group [MeV]','mass_median_proton',2),
            ('Median mass in kaon-majority group [MeV]','mass_median_kaon',2),
            ('Median mass shift [MeV]','mass_shift',2)]
        lines.extend(comparison(label,old.get(key),new.get(key),digits) for label,key,digits in keys)
        lines += ['', 'The baseline k=37 composition, five clean kaon clusters, 2,981 clean-cluster kaon candidates, '
            'and +104.69 MeV mass shift reproduce the paper. The beta=32 mass split is '
            +('positive in the expected direction.' if new['mass_shift'] and new['mass_shift']>0 else 'not positive in the expected direction.')+'','',
            '| Tag | Historical mass-versus-proton-fraction rho | beta=32 rho | beta=32 p | Qualifying clusters |',
            '|---|---:|---:|---:|---:|']
        for a,b in zip(old['cluster_mass_correlations'],new['cluster_mass_correlations']):
            lines.append(f'| {b["species"]} | {a["rho"]:.3f} | {b["rho"]:.3f} | {b["p_value"]:.4g} | {b["n_clusters"]} |')
        lines+=['', 'Each correlation uses clusters containing at least 50 candidates of the tag being tested. '
            'Under this explicit rule the historical proton correlation is rho=0.457, p=0.0217, '
            'so the paper\'s statement of no proton correlation is not reproduced by this check. '
            'The historical kaon rho=0.644 does reproduce the published 0.64. These p-values are descriptive '
            'and uncorrected for multiple comparisons.','']
    else:lines+=['New k=37 composition and mass results are pending.','']
    lines+=['### Table 1 and full k=3-50 scan','',
        '| k | Historical purity | beta=32 purity | beta=32 best kaon purity | beta=32 fraction in >=85% clusters | beta=32 smallest cluster |',
        '|---:|---:|---:|---:|---:|---:|']
    for k in [3,8,12,15,37,50]:
        a=read(HERE/f'results/historical/gmm_k{k}_seed0.json');b=read(HERE/f'results/{PRIMARY}/gmm_k{k}_seed0.json')
        lines.append(f'| {k} | {fmt(a["purity"] if a else None)} | {fmt(b["purity"] if b else None)} | '
            f'{fmt(b["best_cluster_purity_kaon"] if b else None)} | {fmt(b["fraction_in_85"] if b else None)} | '
            f'{fmt(b["smallest_cluster"] if b else None,0)} |')
    counts=sum((HERE/f'results/{PRIMARY}/gmm_k{k}_seed0.json').exists() for k in range(3,51))
    lines += ['',f'{counts}/48 points in the full beta=32 cluster-count scan have completed.','',
        '### k=36-41 stability, five GMM seeds','']
    stability=HERE/'results'/PRIMARY/'stability.csv'
    if stability.exists():
        stab=pd.read_csv(stability)
        lines += ['| k | Mean mass shift [MeV] | Seed SD [MeV] | Mean pairwise ARI |',
            '|---:|---:|---:|---:|']
        lines.extend(f'| {r.k:.0f} | {r.mean_mass_shift:.2f} | {r.sd_mass_shift:.2f} | {r.pairwise_ari_mean:.3f} |'
                     for r in stab.itertuples())
        lines+=['']
    else:lines += ['Pending. No conclusion about fine-cluster stability at beta=32 is yet justified.','']
    lines += ['## Appendix A.1: training/validation distributions','']
    two=read(HERE/'results'/PRIMARY/'two_sample.json')
    if two:
        c=two['c2st']['mlp']; e=two['energy'];m=two['marginal']
        lines += [f'Mixture-matched pooled C2ST MLP AUC = {c["auc"]:.4f} +/- {c["auc_repeat_sd"]:.4f}; '
            f'energy permutation p = {e["p_value"]:.4f}; Holm-significant coordinates = '
            f'{m["n_holm_significant"]}/{two["n_dim"]}. The historical paper reports 0.5026 +/- 0.0029, '
            'p=0.64, and 0/8. The original protocol uses 1,999 marginal/energy permutations, five '
            'energy draws, five classifier draws, 199 classifier null permutations per draw and five folds.','']
    else: lines += ['Pending: mixture-matched C2ST, energy test and Holm-corrected coordinate tests. '
        'The original computational budgets are retained; a cheap preliminary test is not substituted.','']
    lines += ['## Appendix A.2, Figures 5-6: training size and latent capacity','',
        'The full original grids are configured: 30 nested training sizes x three seeds, '
        'and latent dimensions 4,8,...,128 x three seeds at the fixed 12,342-image split. '
        'The three latent-eight/tr50 runs are shared by the grids. Together with three main seeds '
        'and the paired weak-beta control this is 187 unique runs.','',
        'The reconstruction panels use original sampled validation loss at the selected epoch. '
        'The six proxy AUCs use each split\'s validation rows. GMM k=3, active coordinates, '
        '95%-variance coordinate count and participation ratio follow the original source. '
        'An extra common-validation weighted mean-decoding check is explicitly separate from the original metric.','']
    rows=[]
    for task in evaluated:
        r=read(HERE/'results'/task['id']/'scan_metrics.json');rows.append(r)
    if rows:
        table=pd.DataFrame(rows)
        for family in ['training_size','latent_capacity']:
            subset=table[table.family==family]
            subset.to_csv(HERE/f'{family}_beta32.csv',index=False)
        lines += [f'Completed training-size summaries: {(table.family=="training_size").sum()}/90. '
            f'Completed capacity-only summaries: {(table.family=="latent_capacity").sum()}/93 '
            '(plus the three shared tr50/latent-eight runs).','']
        size=table[table.family=='training_size']
        cap=table[(table.tag=='pool8227_tr50') & (table.beta==32)]
        if len(size)==90:
            from scipy.stats import linregress
            means=size.groupby('n_train').mean(numeric_only=True)
            cols=['val_recon','ari','purity']+[f'auc_{f}_{s}' for f in ['mean_adc','solidity'] for s in ['proton','kaon','muon']]
            lines+=['### Complete training-size scan','',
                'Values below are three-seed means. The full per-run and per-rung tables retain seed variation.','',
                '| Quantity | Smallest training set | 12,342 images | 22,212 images | High-data slope p |',
                '|---|---:|---:|---:|---:|']
            high=means[means.index>=12342]
            for col in cols:
                trend=linregress(high.index.to_numpy(),high[col].to_numpy())
                lines.append(f'| {col} | {means.iloc[0][col]:.3f} | {means.loc[12342,col]:.3f} | '
                    f'{means.loc[22212,col]:.3f} | {trend.pvalue:.4g} |')
            lines+=['','Trend p-values describe ordinary linear regression of rung means against training count '
                'over 12,342-22,212 images. Rungs are nested and their validation sets differ, so these '
                'descriptive tests do not provide independent-rung inference or correct multiple comparisons.','']
            size.groupby('n_train')[cols].agg(['mean','std']).to_csv(HERE/'training_size_summary.csv')
        if len(cap)==96:
            means=cap.groupby('latent').mean(numeric_only=True)
            r4,r128=means.loc[4,'val_recon'],means.loc[128,'val_recon']
            target=r4-.95*(r4-r128)
            crossing=means.index[means.val_recon<=target]
            saturation=int(crossing[0]) if r128<r4 and len(crossing) else None
            lines+=['### Complete latent-capacity scan','',
                '| Quantity | Latent 4 | Latent 8 | Latent 128 |',
                '|---|---:|---:|---:|']
            cols=['val_recon','kl_fraction','n_active_dims','n_dims_95var','participation_ratio','ari','purity']+[
                f'auc_{f}_{s}' for f in ['mean_adc','solidity'] for s in ['proton','kaon','muon']]
            for col in cols:
                lines.append(f'| {col} | {means.loc[4,col]:.3f} | {means.loc[8,col]:.3f} | {means.loc[128,col]:.3f} |')
            lines+=['',f'The first measured capacity reaching 95% of the latent-four-to-128 reconstruction '
                f'improvement is {saturation if saturation else "undefined (no net improvement)"} '
                '(paper: about 48). This uses the original sampled validation reconstruction and three-seed means; '
                'it does not impose a monotone fit.','',
                f'KL shares range from {100*cap.kl_fraction.min():.2f}% to {100*cap.kl_fraction.max():.2f}% '
                'across the new runs (paper beta=0.5: 0.5-6.5%).','']
            cap.to_csv(HERE/'latent_capacity_beta32.csv',index=False)
            cap.groupby('latent')[cols].agg(['mean','std']).to_csv(HERE/'latent_capacity_summary.csv')
    if not rows or len(size)<90 or len(cap)<96:
        lines += ['Until each grid is complete, neither the original few-thousand-image plateau nor '
            'stability over capacity nor the original 48-dimensional reconstruction saturation is '
            'established at beta=32. Stronger regularization could change all three.','']
    lines += [
        '## Appendix C, Figures 7-8: beamline context and anchored comparison','',
        'Figure 7 is independent of the VAE. Its measured beamline spectrum, 4,009-event Poisson fit, '
        '488.3 +/- 1.2 MeV kaon peak, 36.2 MeV width, chi-squared/dof=0.91 and fitted '
        '61.2%/25.1%/13.8% kaon/proton/light window fractions remain unchanged. '
        'They precede the analysis selection and remain contextual, rather than a measured composition '
        'of the selected 8,227 kaon candidates. No refit is needed to change beta.','']
    cont=read(HERE/'results'/PRIMARY/'contamination.json')
    if cont:
        a=cont['anchored'];lines += [f'At beta=32 the anchored comparison assigns '
            f'{a["counts"]["proton"]:,} kaon-window candidates to the proton density, '
            f'{a["counts"]["kaon"]:,} to the kaon density and {a["counts"]["muon"]:,} to the MIP density. '
            f'The proton-minus-kaon median mass shift is {fmt(a["mass_shift"],2)} MeV '
            '(paper: 1,692 proton assignments and +103.4 MeV).','',
            f'The tag-conditioned injection-recovery slope is {cont["recovery_slope"]:.3f}. '
            f'Calibrated proton estimates are {100*cont["subsamples"]["picky"]["corrected"]:.2f}% '
            f'for picky and {100*cont["subsamples"]["non-picky"]["corrected"]:.2f}% for non-picky '
            '(paper: 16.9% and 17.2%). These remain model-dependent density estimates, not truth labels.','']
        lines += [f'The picky/non-picky calibrated gap is '
            f'{100*abs(cont["subsamples"]["picky"]["corrected"]-cont["subsamples"]["non-picky"]["corrected"]):.2f} '
            'percentage points. The original near-equality across the quality flag is not preserved '
            'in this primary density-model comparison.','']
    else:lines += ['New anchored-mixture and tag-conditioned injection-recovery estimates are pending.','']
    paired=HERE/'results/bal9419_d8_s0_b0.5/probes.csv'
    lines+=['## Paired control and interpretation','']
    if paired.exists() and newprobes.exists():
        a=pd.read_csv(paired);b=pd.read_csv(newprobes)
        merged=b.merge(a,on=['feature','species'],suffixes=('_32','_05'))
        lines+=['| Proxy / tag | Fresh beta=0.5 seed 0 | beta=32 seed 0 | Paired change |',
                '|---|---:|---:|---:|']
        for r in merged.itertuples():
            lines.append(f'| {r.feature} / {r.species} | {r.auc_05:.3f} | {r.auc_32:.3f} | {r.auc_32-r.auc_05:+.3f} |')
        lines+=['']
        decreases=(merged.auc_32<merged.auc_05).sum()
        lines += [f'{decreases}/6 AUCs decrease in the same-seed comparison. The MIP calorimetry '
            'reader improves; the other five decrease. This distinguishes the beta intervention '
            'from the historical model\'s unknown initialization, while remaining one training-seed comparison.','']
        paired_cluster=read(HERE/'results/bal9419_d8_s0_b0.5/gmm_k37_seed0.json')
        paired_metric=read(HERE/'results/bal9419_d8_s0_b0.5/scan_metrics.json')
        if paired_cluster and paired_metric and new and metric:
            lines += ['| Quantity | Fresh beta=0.5 seed 0 | beta=32 seed 0 | Paired change |',
                '|---|---:|---:|---:|',
                comparison('Validation reconstruction',paired_metric['val_recon'],metric['val_recon'],2),
                comparison('Validation KL',paired_metric['val_kl'],metric['val_kl'],2),
                comparison('k=37 majority tag agreement',paired_cluster['purity'],new['purity']),
                comparison('k=37 kaon-group tag purity',paired_cluster['group_purity_kaon'],new['group_purity_kaon']),
                comparison('k=37 kaon candidates in >=90%-kaon clusters',paired_cluster['clean_kaon_count'],new['clean_kaon_count'],0),
                comparison('k=37 median mass shift [MeV]',paired_cluster['mass_shift'],new['mass_shift'],2),'',
                'The same-seed mass shift is essentially retained, despite lower clustering agreement '
                'and lower kaon-group tag purity. The small historical mass-shift difference should '
                'not therefore be attributed entirely to beta.','']
    else:lines+=['The fresh same-seed beta=0.5 comparison is pending. Historical changes combine '
        'the beta intervention with initialization and hardware differences, so they do not alone '
        'identify a pure beta effect.','']
    lines+=['The observational limitations of the paper persist: beamline tags are imperfect; '
        'mass defines the kaon window; pooled GMM descriptions reuse tags for naming and scoring; '
        't-SNE is visualization; crop and selection can shape the populations. Statements of novelty '
        'and detector physics are contextual and cannot change numerically with beta. Stronger '
        'regularization must be assessed by the measured KL share and the completed proxy/clustering '
        'results, not by treating beta=32 as a guaranteed 50/50 objective.','',
        '## Reproduce and resume','',
        'From the repository root:','',
        '```sh',
        '.venv/bin/python experiments/beta32_paper/test_study.py',
        '.venv/bin/python experiments/beta32_paper/orchestrate.py',
        '```','',
        'The driver resumes saved runs, evaluates completed checkpoints, updates this report, '
        'and creates local commits limited to this folder at the main-result and final milestones. '
        'Large input tensors, checkpoints, posterior arrays and intermediate caches stay local and '
        'are ignored by Git; configurations, hashes, numerical result tables, reference paper, '
        'report and final figures are committed. No existing paper figures or paper copies are replaced.','']
    (HERE/'REPORT.md').write_text('\n'.join(lines))
    write_json(HERE/'coverage.json',{'training_complete':len(complete),'training_expected':len(plan),
        'summaries_complete':len(evaluated),'primary_complete':final,'primary_supplementary_complete':supplementary,
        'full_numerical_reproduction_complete':full,'full_cluster_scan_complete':counts==48})

if __name__=='__main__':make()
