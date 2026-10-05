#!/usr/bin/env python3
"""Publish compact figures, a numerical audit and an inspectable latent sample."""
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
SYMBOLS = {'proton': 'o', 'pion': 'v', 'muon': 'P', 'electron': '^', 'photon': 's'}
COLORS = {'proton': '#1f77b4', **{s: '#b51f8c' for s in ORDER[1:]}}
LABELS = {'incoming_ke_mev': 'Incoming KE', 'deposited_mev': 'Deposited E',
    'geom_retained_energy_mev': 'Retained E', 'mean_dedx_mev_cm': 'Mean dE/dx',
    'endpoint_dedx_mev_cm': 'End dE/dx', 'bragg_ratio': 'Bragg ratio',
    'linearity_3d': '3D linearity', 'transverse_rms_cm': '3D width'}


def md_table(frame):
    rows = [['—' if pd.isna(v) else f'{v:.3f}' if isinstance(v, (float, np.floating)) else str(v)
             for v in row] for row in frame.to_numpy()]
    return '\n'.join(['| '+' | '.join(map(str, frame.columns))+' |',
                       '| '+' | '.join(['---']*len(frame.columns))+' |',
                       *['| '+' | '.join(row)+' |' for row in rows]])


def figures(source, destination, frame, readouts, classifications):
    apply_style(SINGLE_COL)
    ref = np.load(source/'real_reference.npz')
    rng = np.random.default_rng(9105)
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 3.25), gridspec_kw={'width_ratios': [1.05, 1]})
    ax = axes[0]
    real = ref['real_xy'][rng.choice(len(ref['real_xy']), 2000, replace=False)]
    ax.scatter(*real.T, color='.55', s=3, alpha=.18, linewidths=0, label='LArIAT reference', rasterized=True)
    xy = ref['simulation_xy']
    for species in ORDER:
        keep = frame.species.eq(species)
        ax.scatter(*xy[keep].T, color=COLORS[species], marker=SYMBOLS[species], s=7,
            alpha=.48, linewidths=0, label=species.capitalize(), rasterized=True)
    ax.set_xlabel(f"PC1 ({100*ref['pca_variance'][0]:.1f}% of real training variance)")
    ax.set_ylabel(f"PC2 ({100*ref['pca_variance'][1]:.1f}%)")
    ax.legend(loc='best', frameon=False, markerscale=1.5, handletextpad=.3)
    ax.text(0, 1.02, '(a)', transform=ax.transAxes, va='bottom')
    c = classifications.query("task == 'species' and representation == 'paper_vae' and probe == 'trees'").iloc[0]
    matrix = np.asarray(json.loads(c.confusion))
    labels = ['photon', 'electron', 'muon', 'pion', 'proton']
    reorder = [labels.index(s) for s in ORDER]
    matrix = matrix[np.ix_(reorder, reorder)]
    normalized = matrix/matrix.sum(axis=1, keepdims=True)
    ax = axes[1]
    ax.imshow(normalized, vmin=0, vmax=1, cmap='Blues')
    names = ['$p$', '$\\pi$', '$\\mu$', '$e$', '$\\gamma$']
    ax.set_xticks(range(5), names); ax.set_yticks(range(5), names)
    ax.set_xlabel('Predicted species'); ax.set_ylabel('True species')
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{normalized[i,j]:.0%}', ha='center', va='center',
                color='white' if normalized[i,j] > .55 else 'black', fontsize=8)
    ax.text(0, 1.02, '(b)', transform=ax.transAxes, va='bottom')
    fig.tight_layout(w_pad=1)
    for suffix in ('pdf', 'png'):
        fig.savefig(destination/f'latent_species.{suffix}')
    plt.close(fig)
    targets = list(LABELS)
    fig, axes = plt.subplots(2, 1, figsize=(DOUBLE_COL, 5.4), layout='constrained')
    for ax, method, letter in zip(axes, ['paper_vae', 'image_summaries'], ['a', 'b']):
        selected = readouts[(readouts.representation == method) & (readouts.probe == 'ridge')]
        table = selected.pivot(index='species', columns='target', values='r2').reindex(index=ORDER, columns=targets)
        values = table.to_numpy()
        image = ax.imshow(np.ma.masked_invalid(values), vmin=-.2, vmax=1, cmap='Blues', aspect='auto')
        ax.set_yticks(range(5), [s.capitalize() for s in ORDER])
        ax.set_xticks(range(len(targets)), [LABELS[t]+(' *' if t != 'linearity_3d' else '') for t in targets], rotation=30, ha='right')
        for i in range(5):
            for j in range(len(targets)):
                value = values[i,j]
                ax.text(j, i, '—' if not np.isfinite(value) else f'{value:.2f}',
                    ha='center', va='center', fontsize=8,
                    color='white' if np.isfinite(value) and value > .65 else 'black')
        description = 'Frozen VAE: 8 latent values' if method == 'paper_vae' else 'Image summaries: 32 values'
        ax.text(0, 1.035, f'({letter}) {description}', transform=ax.transAxes, va='bottom')
    fig.colorbar(image, ax=axes, label='$R^2$ on held-out events', shrink=.8, pad=.015)
    for suffix in ('pdf', 'png'):
        fig.savefig(destination/f'truth_preservation.{suffix}')
    plt.close(fig)


def reconstruction_audit(source, destination, frame):
    raw = np.load(source/'raw.npy')
    reconstructed = np.expm1(np.clip(np.load(source/'paper_vae_reconstruction.npy'), 0, 15))
    rows = []
    for species in ORDER:
        mask = frame.species.eq(species) & frame.partition.eq('test')
        for plane, name in enumerate(['collection', 'induction']):
            a, b = raw[mask, plane], reconstructed[mask, plane]
            total_ratio = b.sum((1, 2))/a.sum((1, 2)).clip(min=1e-9)
            peak_ratio = b.max((1, 2))/a.max((1, 2)).clip(min=1e-9)
            a_profile = a.sum(axis=2)/a.sum((1, 2)).clip(min=1e-9)[:, None]
            b_profile = b.sum(axis=2)/b.sum((1, 2)).clip(min=1e-9)[:, None]
            rows.append({'species': species, 'plane': name, 'n_test': int(mask.sum()),
                'median_decoded_input_adc_sum_ratio': np.median(total_ratio),
                'median_decoded_input_adc_peak_ratio': np.median(peak_ratio),
                'median_normalized_wire_profile_l1_error': np.median(np.abs(a_profile-b_profile).sum(axis=1))})
    result = pd.DataFrame(rows)
    result.to_csv(destination/'reconstruction_audit.csv', index=False)
    return result


def report(source, destination, frame, readouts, classes, reconstruction):
    protocol = json.loads((source/'protocol.json').read_text())
    encoders = json.loads((source/'encoders.json').read_text())
    def rv(species, target, probe='ridge', representation='paper_vae', field='r2'):
        matched = readouts[(readouts.species == species) & (readouts.target == target) &
            (readouts.probe == probe) & (readouts.representation == representation)]
        return float(matched.iloc[0][field]) if len(matched) else np.nan
    def cv(task, probe='logistic', representation='paper_vae', field='balanced_accuracy'):
        matched = classes[(classes.task == task) & (classes.probe == probe) & (classes.representation == representation)]
        return float(matched.iloc[0][field]) if len(matched) else np.nan
    support = frame.groupby('species').kinematics_reliable.mean()
    retained = frame.groupby('species').geometric_energy_fraction.mean()
    bb = frame.loc[frame.species.eq('proton'), ['proton_bb_density_ratio','proton_bb_log_distance']].median()
    domain = pd.read_csv(source/'domain_separation.csv').set_index('species')
    n_proton = rv('proton', 'incoming_ke_mev', 'trees', field='n_test')
    report_path = Path(__file__).with_name('RESULTS.md')
    lines = ['# PILArNet simulation truth in the frozen LArIAT latent space', '',
        f"This is a bounded diagnostic using **{len(frame):,} particles** "
        f"({', '.join(f'{s}: {n}' for s, n in protocol['counts'].items())}), selected from the first {protocol['events_scanned']:,} events of one checksum-verified shard. "
        'No kaons are available. The frozen paper VAE is run 0093, with eight latent dimensions; '
        'all predictions use its posterior mean. The response was calibrated on protons and remains exploratory for other species.', '',
        '## Main findings', '',
        '- Species and coarse track/shower information are recoverable. The nonlinear species probe reaches '
        f"{cv('species','trees'):.1%} balanced accuracy (event-bootstrap 95% interval {cv('species','trees',field='low'):.1%}–{cv('species','trees',field='high'):.1%}); the linear probe reaches {cv('species'):.1%}. "
        f"Five-species chance is 20%. Simple image summaries reach {cv('species',representation='image_summaries'):.1%} with the same linear classifier.",
        '- Proton incoming energy is recoverable, but incomplete: log-energy R² is '
        f"{rv('proton','incoming_ke_mev'):.3f} with a linear probe and {rv('proton','incoming_ke_mev','trees'):.3f} with trees. "
        f"The nonlinear mean absolute error is {rv('proton','incoming_ke_mev','trees',field='physical_mae'):.1f} MeV on {n_proton:.0f} held-out protons. "
        f"Image summaries reach {rv('proton','incoming_ke_mev',representation='image_summaries'):.3f} with a linear probe.",
        '- Fine proton endpoint calorimetry is poorly recoverable from the latent vectors. '
        f"Endpoint dE/dx R² is {rv('proton','endpoint_dedx_mev_cm'):.3f} (linear) and {rv('proton','endpoint_dedx_mev_cm','trees'):.3f} (trees), "
        f"versus {rv('proton','endpoint_dedx_mev_cm',representation='image_summaries'):.3f}/{rv('proton','endpoint_dedx_mev_cm','trees','image_summaries'):.3f} from image summaries. "
        f"The proton Bragg ratio gives {rv('proton','bragg_ratio'):.3f}/{rv('proton','bragg_ratio','trees'):.3f} in the latent, versus {rv('proton','bragg_ratio',representation='image_summaries'):.3f} from linear image summaries. "
        'This is evidence of a useful image-level signal that these latent readouts fail to retain, '
        'rather than proof that no conceivable readout can recover it.',
        f"- Full muon incoming energy is weakly recoverable: linear R² {rv('muon','incoming_ke_mev'):.3f}; nonlinear {rv('muon','incoming_ke_mev','trees'):.3f}. "
        f"Only {retained['muon']:.1%} of its full deposited energy is geometrically retained on average, compared with {retained['proton']:.1%} for protons. "
        'Endpoint-only images cannot supply an unrestricted calorimetric energy measurement.',
        '- Electron/photon deposited energy and broad transverse spread are recoverable. '
        'Pions are the hardest class: nonlinear recall is 37%, with 33% assigned to muons. '
        f"Electron incoming-energy truth is accepted for {support['electron']:.1%} of the sample, photon truth for {support['photon']:.1%}; "
        'their incoming-energy results are conditional on this metadata-quality subset.',
        '- Original vertices/directions are not recovered by linear probes. They are deliberately removed by independent particle placement. '
        'Same-interaction pairing reaches only AUROC 0.591, versus 1.000 using original vertices. '
        'Isolated endpoint vectors are insufficient for reconstructing the original event without retaining its geometry.',
        f"- The MC and real proton latents are strongly distinguishable (unmatched AUROC {domain.loc['proton','auc_real_vs_simulation']:.3f}). "
        'This is not yet a validated simulation-to-real transfer test; differences in incoming/TPC energy, selection and detector response remain.',
        f"- Bethe–Bloch truth diagnostic: median proton dE/dx/model ratio {bb.proton_bb_density_ratio:.3f}, median absolute log-profile deviation {bb.proton_bb_log_distance:.3f}. "
        f"The density-ratio readout gives latent tree R² {rv('proton','proton_bb_density_ratio','trees'):.3f}, versus {rv('proton','proton_bb_density_ratio','trees','image_summaries'):.3f} from image summaries. "
        'All three control VAE seeds and the AE also poorly recover proton endpoint dE/dx; this study does not isolate which architecture, bottleneck or loss choice causes it.', '',
        '## Protocol and leakage controls', '',
        f"Source: `{protocol['source_relative']}`; published SHA-256 `{protocol['published_sha256']}`.",
        f"Frozen checkpoint SHA-256: `{encoders['paper_vae']['checkpoint_sha256']}`.",
        f"Frozen response SHA-256: `{protocol['response_sha256']}`.", '',
        'One response, recombination rule and fit-only LArIAT direction bank are used for every species. '
        'No per-species image tuning, simulated model training or checkpoint modification is performed. '
        f"Candidates are randomized and capped at {protocol['requested_per_species']} accepted particles per species; {protocol['rejections']} attempted candidates fail the conversion/quality cuts. "
        'Selection requires 20–10,000 voxels, at least five occupied collection rows, and a dominant connected component carrying at least half the positive signal in each plane. '
        'These cuts bias the accepted shower/secondary sample. The same source event always occupies one 60/20/20 probe partition. '
        'Scaler/PCA fits use probe-training events only; regularization is chosen on development events. '
        'Test labels are used for reporting and explicitly labelled matched-population diagnostics only. '
        'CIs use 150 event-cluster bootstrap draws; this is an exploratory pilot with many correlated targets.', '',
        md_table(pd.crosstab(frame.species, frame.partition).reindex(ORDER).reset_index()), '',
        'Incoming KE = sqrt(p²+m²)−m with the recorded momentum convention GeV/c→MeV/c. '
        'First-fragment kinematics are used only if fragment momentum spread ≤5%, KE>0 and deposited energy≤1.15 KE. '
        'This is a consistency filter, not independent validation of generator metadata. '
        'dx numerically follows the validated cm convention in the conversion; the dataset-card mm description is not used. '
        'Retained energy is a mean two-plane voxel-centre geometric proxy; it is not exact waveform ancestry. '
        'Track endpoint targets require >90% unique deposition times; shower sum(dx) is collective path, not shower length. '
        '3D linearity and width come from unweighted voxel covariance. '
        'Bethe–Bloch scores use the existing proton-deuteron proton table without shifting or changing energy labels.', '',
        '## Held-out classification', '',
        md_table(classes[['task', 'representation', 'probe', 'balanced_accuracy', 'low', 'high', 'n_test']]), '',
        'Semantic classification is the dominant **truth** label of an isolated particle (shower/track/Michel/delta), '
        'not pixel segmentation. Species may be inferred partly from their generated energy distributions. '
        'The five-class energy-balanced diagnostic has insufficient common retained-energy support; '
        'the energy-only oracle is reported to expose that shortcut.', '',
        '## Within-species truth readouts', '',
        'R² uses log1p(target) for energy, momentum, dE/dx, sizes and ratios; '
        '3D linearity, fractions, angles and positions use their original scale. '
        'Physical-unit R²/MAE and support are in the CSV. R²≤0 means no advantage over the held-out target mean. '
        'Linear and nonlinear probes answer different accessibility questions. Input PCA has eight dimensions; '
        'image summaries have 32, and raw pixels have 4,608. A stronger summary baseline does not isolate latent dimension from architecture/objective effects.', '']
    main = readouts[(readouts.representation == 'paper_vae') & readouts.probe.isin(['ridge', 'trees']) & readouts.species.ne('pooled')]
    for species in ORDER:
        lines += [f'### {species.capitalize()}', '', md_table(main[main.species.eq(species)][
            ['target', 'probe', 'n_test', 'r2', 'r2_low', 'r2_high', 'physical_mae']]), '']
    lines += ['## Reconstruction and response sensitivity', '',
        'The decoder is evaluated at the posterior mean. Reconstruction ADC is expm1(max(decoded log image,0)). '
        'Ratios below are descriptive image quantities, not calibrated charge or energy closure. '
        'The normalized wire-profile error compares row marginals on the same resized image grid; it is not a physical dE/dx measurement.', '', md_table(reconstruction), '',
        md_table(pd.read_csv(source/'counterfactual_summary.csv')), '',
        'Counterfactuals use 25 held-out particles per species. ±20% waveform gain is applied before threshold and recropping; '
        'a +2° xz rotation preserves the source deposits but can change volume/crop support. '
        'Distances are standardized on simulation probe-training latents and divided by the median distance between random distinct particles of the same species. '
        'These are sensitivity diagnostics, not proof of invariance. Regenerating every baseline reproduced its saved image and latent vector.', '',
        '## Domain and original-event geometry', '', md_table(pd.read_csv(source/'domain_separation.csv')), '',
        md_table(pd.read_csv(source/'interaction_readout.csv')), '']
    matched_path = source/'energy_matched_proton_domain.csv'
    if matched_path.exists():
        lines += ['A second domain check equalizes real/MC counts within fixed 20 MeV incoming-KE bands in each event partition. '
            'Real labels use upstream beamline momentum; simulated labels refer to the generated particle. '
            'TPC energy, upstream material, reconstruction and selections remain unmatched.', '', md_table(pd.read_csv(matched_path)), '']
    lines += ['A real-trained closed-set proton/kaon/MIP head is also applied descriptively:', '',
        md_table(pd.read_csv(source/'real_reference_assignments.csv')), '',
        'These allocations are **not** validated species probabilities or beam-composition estimates. '
        'For example, assigning a simulated photon to the kaon region does not establish a real kaon contamination rate. '
        'The real MIP category is not pure simulated-muon truth.', '',
        '## Tasks this pilot cannot validate', '',
        '- Kaon recognition and proton/kaon decontamination: no kaon simulation truth in this release.',
        '- Full-scene instance/pixel segmentation and vertex finding: particles were extracted using truth; original shared geometry is removed.',
        '- Parent/daughter, decay ancestry and interaction-process classification: no adequate ancestry/process truth in these arrays.',
        '- Unrestricted total energy reconstruction or calibrated LArIAT task performance: endpoint cropping and unresolved domain differences.', '',
        '## Figure captions and reproducibility', '',
        '`latent_species.pdf`: (a) real-training PCA of standardized eight-dimensional latents, '
        'with a random real reference subset and all converted particles; this 2D view is illustrative. '
        '(b) held-out row-normalized five-class confusion matrix for the nonlinear latent probe. '
        'Numerical probes use all eight dimensions.', '',
        '`truth_preservation.pdf`: identical linear probes of the eight-dimensional latent and 32-dimensional image summaries. '
        'Entries are within-species held-out R²; asterisks denote log1p targets. '
        'Colour spans −0.2 to 1, while annotations retain exact values including any outside that range. '
        'Dashes indicate unsupported/unapplied targets. Confidence intervals and all controls are in the CSV.', '',
        f'Full caches and truth results: `{source}`. Figures and copied CSVs: `{destination}`.', '',
        'Run `prepare.py`, `encode.py`, `evaluate.py`, `counterfactual.py`, then `build_report.py`. '
        'Use another output path to change an existing pilot. All checkpoints remain frozen and large data stay on the external drive.', '']
    report_path.write_text('\n'.join(lines))
    return report_path


def alpha_png(values, vmax):
    rgba = np.zeros((48, 48, 4), np.uint8)
    rgba[..., 3] = np.rint(np.clip(values/vmax, 0, 1)*255).astype(np.uint8)
    buf = io.BytesIO(); Image.fromarray(rgba).save(buf, format='PNG', optimize=True)
    return 'data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()


def explorer(source, fragment, frame):
    rng = np.random.default_rng(9105)
    ref = np.load(source/'real_reference.npz')
    raw = np.log1p(np.load(source/'raw.npy'))
    decoded = np.load(source/'paper_vae_reconstruction.npy')
    selected = []
    for species in ORDER:
        f = frame[frame.species.eq(species) & frame.partition.eq('test')]
        selected.extend(rng.choice(f.index, 20, replace=False))
    vmax = float(np.quantile(raw[raw > 0], .999))
    cp = pd.read_csv(source/'classification_predictions.csv').query("task=='species'").set_index('row')
    points = []
    for row in selected:
        f = frame.iloc[row]
        points.append({'row': int(row), 'species': f.species, 'event': int(f.event_index), 'group': int(f.group_id),
            'xy': np.round(ref['simulation_xy'][row], 4).tolist(),
            'ke': round(f.incoming_ke_mev, 2) if f.kinematics_reliable else None,
            'edep': round(f.deposited_mev, 2), 'retained': round(f.geom_retained_energy_mev, 2),
            'dedx': round(f.mean_dedx_mev_cm, 3), 'width': round(f.transverse_rms_cm, 3),
            'probe': {0:'photon', 1:'electron', 2:'muon', 3:'pion', 4:'proton'}[int(cp.loc[row, 'predicted'])],
            'images': [alpha_png(a, vmax) for a in [raw[row,0], raw[row,1], decoded[row,0], decoded[row,1]]]})
    subset = rng.choice(len(ref['real_xy']), 1200, replace=False)
    payload = {'points': points, 'reference': np.round(ref['real_xy'][subset], 4).tolist(),
        'cloud': [{'xy': np.round(xy, 4).tolist(), 'species': s} for xy, s in zip(ref['simulation_xy'], frame.species)],
        'variance': np.round(ref['pca_variance']*100, 1).tolist(), 'scale': round(vmax, 3)}
    template = Path(__file__).with_name('explorer.template.html').read_text()
    fragment.parent.mkdir(parents=True, exist_ok=True)
    fragment.write_text(template.replace('__PAYLOAD__', json.dumps(payload, separators=(',', ':'))))
    if fragment.stat().st_size >= 1_000_000:
        raise RuntimeError('Inline latent explorer exceeds 1 MB')
    return fragment


def build(source, destination, fragment):
    destination.mkdir(parents=True, exist_ok=True)
    frame = pd.read_csv(source/'manifest.csv')
    readouts = pd.read_csv(source/'truth_readouts.csv')
    classes = pd.read_csv(source/'classification.csv')
    for path in source.glob('*.csv'):
        if path.name not in ('manifest.csv', 'rejected.csv', 'interaction_pairs.csv'):
            shutil.copy2(path, destination/path.name)
    figures(source, destination, frame, readouts, classes)
    reconstruction = reconstruction_audit(source, destination, frame)
    print(report(source, destination, frame, readouts, classes, reconstruction))
    print(explorer(source, fragment, frame))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=BASE/'latent_truth/pilot_v1')
    p.add_argument('--destination', type=Path, default=ROOT/'output/pilarnet_lariat/latent_truth/pilot_v1')
    p.add_argument('--fragment', type=Path, required=True)
    args = p.parse_args()
    build(args.source, args.destination, args.fragment)
