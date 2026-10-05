#!/usr/bin/env python3
"""Matched pixel-PCA figures and an inspectable t-SNE comparison."""
import argparse
import base64
import io
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT/'scripts/extra')]
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from experiments.pilarnet_lariat.latent_truth.prepare import BASE
from scripts.extra._beam_data import apply_style, SINGLE_COL, DOUBLE_COL

ORDER = ['proton', 'pion', 'muon', 'electron', 'photon']
SYMBOLS = {'proton':'o', 'pion':'v', 'muon':'P', 'electron':'^', 'photon':'s'}
COLORS = {'proton':'#1f77b4', **{s:'#b51f8c' for s in ORDER[1:]}}
LABELS = {'incoming_ke_mev':'Incoming KE', 'deposited_mev':'Deposited E',
    'geom_retained_energy_mev':'Retained E', 'mean_dedx_mev_cm':'Mean dE/dx',
    'endpoint_dedx_mev_cm':'End dE/dx', 'bragg_ratio':'Bragg ratio',
    'linearity_3d':'3D linearity', 'transverse_rms_cm':'3D width'}
METHODS = [('paper_vae', 'VAE (8D)'), ('input_pca8', 'Pixel PCA (8D)')]


def md_table(frame):
    rows = [['—' if pd.isna(v) else f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
             for v in row] for row in frame.to_numpy()]
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
        '| '+' | '.join(['---']*len(frame.columns))+' |',
        *['| '+' | '.join(row)+' |' for row in rows]])


def figures(source, destination, frame, readouts, formats):
    apply_style(SINGLE_COL)
    views = np.load(source/'tsne_views.npz')
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.3))
    for ax, (method, label), letter in zip(axes, METHODS, ['a','b']):
        xy = views[method]
        for species in ORDER:
            keep = frame.species.eq(species)
            ax.scatter(*xy[keep].T, color=COLORS[species], marker=SYMBOLS[species], s=7,
                alpha=.5, linewidths=0, label=species.capitalize(), rasterized=True)
        ax.set_xlabel('t-SNE 1 (arbitrary units)'); ax.set_ylabel('t-SNE 2 (arbitrary units)')
        ax.text(0, 1.02, f'({letter}) {label}', transform=ax.transAxes, va='bottom')
    axes[0].legend(loc='best', frameon=False, markerscale=1.5, handletextpad=.3)
    fig.tight_layout(w_pad=1)
    for suffix in formats:
        fig.savefig(destination/f'tsne_pixel_baseline.{suffix}')
    plt.close(fig)
    targets = list(LABELS)
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 5.4), layout='constrained')
    for ax, (method, label), letter in zip(axes, METHODS, ['a','b']):
        selected = readouts[(readouts.representation == method) & (readouts.probe == 'ridge')]
        values = selected.pivot(index='species', columns='target', values='r2').reindex(index=ORDER, columns=targets).to_numpy()
        image = ax.imshow(np.ma.masked_invalid(values), vmin=-.2, vmax=1, cmap='Blues', aspect='auto')
        ax.set_yticks(range(5), [s.capitalize() for s in ORDER])
        ax.set_xticks(range(len(targets)), [LABELS[t]+(' *' if t != 'linearity_3d' else '') for t in targets], rotation=30, ha='right')
        for i in range(5):
            for j in range(len(targets)):
                value = values[i,j]
                ax.text(j,i,'—' if not np.isfinite(value) else f'{value:.2f}',ha='center',va='center',fontsize=8,
                    color='white' if np.isfinite(value) and value > .65 else 'black')
        ax.text(0,1.035,f'({letter}) {label} · identical linear probes',transform=ax.transAxes,va='bottom')
    fig.colorbar(image, ax=axes, label='$R^2$ on held-out events', shrink=.8, pad=.015)
    for suffix in formats:
        fig.savefig(destination/f'truth_pixel_baseline.{suffix}')
    plt.close(fig)


def report(source, destination, frame, readouts, classes):
    protocol = json.loads((source/'protocol.json').read_text())
    pixels = json.loads((source/'representation_protocol.json').read_text())
    tsne = json.loads((source/'tsne_protocol.json').read_text())
    encoders = json.loads((source/'encoders.json').read_text())
    primary = classes[classes.representation.isin([m[0] for m in METHODS])]
    physical = readouts[readouts.representation.isin([m[0] for m in METHODS]) & readouts.species.ne('pooled') & readouts.probe.isin(['ridge','trees'])]
    lines = ['# VAE versus pixel PCA: simulation truth and t-SNE', '',
        f"Updated comparison on the same {len(frame):,} converted particles ({', '.join(f'{s}: {n}' for s,n in frame.species.value_counts().items())}). "
        'No engineered image descriptors or truth-derived predictors are used. '
        'Simulation truth supplies evaluation labels only. The main baseline is **eight principal components of the actual pixels**, '
        'matching the frozen VAE’s eight latent dimensions.', '',
        '## Pixel baseline', '',
        'Flatten both 48×48 planes into 4,608 values after the same single log1p(ADC) transform used by the VAE. '
        'PCA centres each pixel and fits **training events only**, without per-pixel variance scaling or labels. '
        f"Eight components retain {pixels['retained_training_variance']:.1%} of training pixel variance. "
        'Development/test images are projected with that fixed basis. '
        'This is an in-domain MC-fitted PCA baseline; the VAE remains frozen from real LArIAT training. '
        'Their representation-training data therefore differ, so this does not isolate architecture under identical training conditions. '
        'Prediction heads standardize each 8D representation on training events, select regularization on development events, '
        'and report test performance. VAE and pixel PCA use identical linear and nonlinear head settings.', '',
        md_table(pd.crosstab(frame.species,frame.partition).reindex(ORDER).reset_index()), '',
        '## t-SNE', '',
        f"Settings: perplexity {tsne['settings']['perplexity']}, seed {tsne['settings']['random_state']}, "
        f"PCA initialization, automatic learning rate, {tsne['settings']['max_iter']} iterations. "
        'Both 8D representations are standardized using their training events before separate t-SNE fits. '
        'All 2,500 particles participate in this descriptive visualization. '
        'PID labels are added only afterwards for colour/shape. '
        'The coordinates of the two plots are independent; global distances and cluster sizes cannot be compared across maps. '
        '**No t-SNE coordinates enter the prediction heads.**', '',
        md_table(pd.DataFrame([{'representation':label,**tsne[method]} for method,label in METHODS])), '',
        '## Held-out species and semantic classification', '',
        md_table(primary[['task','representation','probe','balanced_accuracy','low','high','n_test']]), '',
        'Five-species chance is 20%. Semantic classification predicts the dominant truth category of an isolated particle, '
        'not voxel segmentation. The generated species have different energy distributions; '
        'the five-class energy-balanced diagnostic lacks common support, so this remains a conditional simulation pilot.', '',
        '## Within-species physical readouts', '',
        'Energy, momentum, dE/dx, lengths and ratios use log1p targets; positions, angles, fractions and 3D linearity use native targets. '
        'R² is reported on the fitted target scale. Physical-unit mean absolute errors and R² are in the CSV. '
        'Confidence intervals resample source events (150 draws). '
        'The endpoint dE/dx and Bragg conclusions must be judged against pixel PCA, rather than the earlier descriptor baseline. '
        'Failure of both 8D representations does not demonstrate a VAE-specific loss of information.', '']
    for species in ORDER:
        lines += [f'### {species.capitalize()}', '', md_table(physical[physical.species.eq(species)][
            ['target','representation','probe','n_test','r2','r2_low','r2_high','physical_mae']]), '']
    lines += ['## Controls and limits', '',
        'Additional controls use raw pixels, the proton-only VAE, three VAE seeds, an AE, an untrained encoder and shuffled labels. '
        'These are pixel/learned representations. Original-coordinate truth and derived physical profiles remain targets, '
        'not additional model inputs. The interaction-pair head concatenates the two representations directly.', '',
        md_table(pd.read_csv(source/'interaction_readout.csv')), '',
        'This sample uses truth-isolated particles and one exploratory proton-calibrated readout response. '
        'Incoming kinematics are accepted only when fragment momenta agree within 5% and energy closes; '
        'the retained-energy label is a voxel-centre geometric proxy, not exact waveform ancestry. '
        'Original shared vertices/directions were removed during placement. '
        'Kaon truth, full-event segmentation/ancestry and calibrated real-data task transfer are not tested. '
        'The previous domain and response-sensitivity results remain diagnostics of the same fixed sample.', '',
        '## Provenance', '',
        f"Source: `{protocol['source_relative']}`, published SHA-256 `{protocol['published_sha256']}`.",
        f"Checkpoint SHA-256: `{encoders['paper_vae']['checkpoint_sha256']}`.",
        f"Response SHA-256: `{protocol['response_sha256']}`.",
        f"Ordered manifest SHA-256: `{pixels['manifest_sha256']}`.", '',
        f'Data and numerical results: `{source}`. Plots and copied CSVs: `{destination}`.', '',
        '`tsne_pixel_baseline.png`: identical t-SNE settings on VAE8 and pixel PCA8; all selected simulation particles. '
        '`truth_pixel_baseline.png`: identical linear probes on these two eight-dimensional representations; '
        'asterisks indicate log1p targets and dashes indicate unsupported targets. Numerical readouts use all eight dimensions.', '']
    path = Path(__file__).with_name('RESULTS.md')
    path.write_text('\n'.join(lines))
    return path


def alpha_png(values, vmax):
    rgba = np.zeros((48,48,4),np.uint8)
    rgba[...,3] = np.rint(np.clip(values/vmax,0,1)*255).astype(np.uint8)
    buf = io.BytesIO();Image.fromarray(rgba).save(buf,format='PNG',optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


def explorer(source, fragment, frame):
    rng = np.random.default_rng(9105)
    views = np.load(source/'tsne_views.npz')
    raw = np.log1p(np.load(source/'raw.npy'))
    decoded = np.load(source/'paper_vae_reconstruction.npy')
    selected = []
    for species in ORDER:
        f = frame[frame.species.eq(species) & frame.partition.eq('test')]
        selected.extend(rng.choice(f.index,min(20,len(f)),replace=False))
    vmax = float(np.quantile(raw[raw>0],.999))
    cp = pd.read_csv(source/'classification_predictions.csv').query("task=='species'")
    predictors = {m:cp[cp.representation.eq(m)].set_index('row') for m,_ in METHODS}
    pid_names = {0:'photon',1:'electron',2:'muon',3:'pion',4:'proton'}
    points = []
    for row in selected:
        f = frame.iloc[row]
        points.append({'row':int(row),'species':f.species,'event':int(f.event_index),'group':int(f.group_id),
            'xy':[np.round(views[m][row],4).tolist() for m,_ in METHODS],
            'ke':round(f.incoming_ke_mev,2) if f.kinematics_reliable else None,
            'edep':round(f.deposited_mev,2),'dedx':round(f.mean_dedx_mev_cm,3),
            'probe':[pid_names[int(predictors[m].loc[row,'predicted'])] for m,_ in METHODS],
            'images':[alpha_png(a,vmax) for a in [raw[row,0],raw[row,1],decoded[row,0],decoded[row,1]]]})
    c = pd.read_csv(source/'classification.csv')
    scores = [{'probe':probe,'values':[round(float(c[(c.task=='species')&(c.representation==m)&(c.probe==probe)].iloc[0].balanced_accuracy)*100,1)
              for m,_ in METHODS]} for probe in ('logistic','trees')]
    pixel_info = json.loads((source/'representation_protocol.json').read_text())
    payload = {'points':points,'cloud':[{'row':i,'species':s,'xy':[np.round(views[m][i],4).tolist() for m,_ in METHODS]}
        for i,s in enumerate(frame.species)],'scale':round(vmax,3),'scores':scores,
        'pixel_variance':round(pixel_info['retained_training_variance']*100,1)}
    template = Path(__file__).with_name('explorer.template.html').read_text()
    fragment.parent.mkdir(parents=True,exist_ok=True)
    fragment.write_text(template.replace('__PAYLOAD__',json.dumps(payload,separators=(',',':'))))
    if fragment.stat().st_size >= 1_000_000:raise RuntimeError('Explorer exceeds 1 MB')
    return fragment


def build(source,destination,fragment,formats=('png',)):
    destination.mkdir(parents=True,exist_ok=True)
    frame = pd.read_csv(source/'manifest.csv')
    readouts = pd.read_csv(source/'truth_readouts.csv')
    classes = pd.read_csv(source/'classification.csv')
    for path in source.glob('*.csv'):
        if path.name not in ('manifest.csv','rejected.csv','interaction_pairs.csv'):
            shutil.copy2(path,destination/path.name)
    figures(source,destination,frame,readouts,formats)
    print(report(source,destination,frame,readouts,classes))
    print(explorer(source,fragment,frame))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=BASE/'latent_truth/pilot_pixels_v2')
    p.add_argument('--destination',type=Path,default=ROOT/'output/pilarnet_lariat/latent_truth/pilot_pixels_v2')
    p.add_argument('--fragment',type=Path,required=True)
    p.add_argument('--pdf',action='store_true',help='Also export publication PDFs')
    args = p.parse_args()
    build(args.source,args.destination,args.fragment,('png','pdf') if args.pdf else ('png',))
