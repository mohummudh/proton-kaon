#!/usr/bin/env python3
"""Inspect approximate LArIAT projections at the paper's printed figure size."""

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.extra._beam_data import apply_style, DOUBLE_COL, SINGLE_COL, COLOURS


def plot_gallery(particles, input_dir, output_dir):
    """Eight particle pairs per page, preserving manifest order and ADC scale."""
    samples = []
    for record in particles:
        with np.load(input_dir / f'{record["id"]}.npz') as source:
            samples.append(source['log1p'].copy())
    vmax = max(float(sample.max()) for sample in samples)
    apply_style(SINGLE_COL)
    with PdfPages(output_dir / 'all_particles.pdf') as pdf:
        for start in range(0, len(particles), 8):
            fig = plt.figure(figsize=(DOUBLE_COL, 9.5))
            grid = fig.add_gridspec(4, 2, left=.065, right=.975, bottom=.12,
                                   top=.95, wspace=.22, hspace=.50)
            im = None
            for local, index in enumerate(range(start, min(start + 8, len(particles)))):
                record = particles[index]
                pair = grid[local // 2, local % 2].subgridspec(1, 2, wspace=.09)
                for plane in (0, 1):
                    ax = fig.add_subplot(pair[0, plane])
                    im = ax.imshow(samples[index][plane], cmap='Greys', vmin=0, vmax=vmax,
                                   origin='upper', aspect='equal', interpolation='nearest')
                    ax.set_xticks([0, 24, 47])
                    ax.set_yticks([0, 24, 47])
                    ax.tick_params(length=2, pad=2)
                    if plane == 1:
                        ax.tick_params(labelleft=False)
                    ax.text(.5, 1.045, ('Collection', 'Induction')[plane],
                            transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
                    if plane == 0:
                        color = COLOURS['proton' if record['species']=='Proton' else 'muon']
                        ax.text(0, 1.25, f'{index+1:02d}  {record["species"]}  '
                                f'({record["energy_mev"]:.1f} MeV deposited)',
                                transform=ax.transAxes, ha='left', va='bottom', color=color)
            fig.text(.52, .084, 'Time-axis pixel', ha='center')
            fig.text(.012, .54, 'Wire-axis pixel', rotation=90, va='center')
            cax = fig.add_axes([.28, .045, .46, .012])
            fig.colorbar(im, cax=cax, orientation='horizontal',
                         label='$\\log(1+\\mathrm{ADC})$')
            pdf.savefig(fig)
            fig.savefig(output_dir / f'all_particles_{start//8+1:02d}.png')
            plt.close(fig)
    print(f'Saved {len(particles)} pairs in all_particles.pdf and page PNGs')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('output/pilarnet_lariat'))
    parser.add_argument('--all', action='store_true', help='Show every pair in the manifest')
    args = parser.parse_args()
    manifest = json.loads((args.input/'manifest.json').read_text())
    particles = manifest['particles']
    args.output.mkdir(parents=True, exist_ok=True)
    if args.all:
        plot_gallery(particles, args.input, args.output)
        return
    selected = [next(r for r in particles if r['species'] == species) for species in ('Proton', 'Muon')]
    data = [np.load(args.input/f'{r["id"]}.npz') for r in selected]
    apply_style(SINGLE_COL)
    args.output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(DOUBLE_COL, 4.6))
    grid = fig.add_gridspec(2, 3, width_ratios=(1.15, 1, 1), wspace=.65, hspace=.75)
    vmax = max(float(d['log1p'].max()) for d in data)
    im = None
    for row, (record, sample) in enumerate(zip(selected, data)):
        ax = fig.add_subplot(grid[row, 0], projection='3d')
        xyz, energy = sample['xyz_cm'], sample['energy_mev']
        # Show only the deposits within the TPC, matching the accepted readout.
        r = manifest['response']
        valid = ((xyz[:, 0]>=0)&(xyz[:, 0]<=r['main_drift_cm'])
                 &(abs(xyz[:, 1])<=r['height_cm']/2)&(xyz[:, 2]>=0)&(xyz[:, 2]<=r['length_cm']))
        color = COLOURS['proton' if record['species']=='Proton' else 'muon']
        ax.scatter(xyz[valid, 2], xyz[valid, 0], xyz[valid, 1], s=2+8*np.sqrt(energy[valid]),
                   color=color, alpha=.65, linewidths=0)
        ax.set_xlabel('Beam $z$ (cm)', labelpad=-3)
        ax.set_ylabel('Drift $x$ (cm)', labelpad=-3)
        ax.set_zlabel('$y$ (cm)', labelpad=-3)
        ax.tick_params(pad=-1, labelsize=7)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.fill = False
            axis._axinfo['grid']['linewidth'] = .3
        ax.view_init(elev=22, azim=-65)
        ax.text2D(0, 1.05, record['species'], transform=ax.transAxes)
        for column, plane in ((1, 0), (2, 1)):
            ax = fig.add_subplot(grid[row, column])
            im = ax.imshow(sample['log1p'][plane], cmap='Greys', vmin=0, vmax=vmax,
                           origin='upper', aspect='equal', interpolation='nearest')
            ax.set_xlabel('Time-axis pixel')
            ax.set_ylabel('Wire-axis pixel')
            ax.set_xticks([0, 24, 47]);ax.set_yticks([0, 24, 47])
            if row == 0:
                ax.text(.5, 1.05, ['Collection', 'Induction'][plane], ha='center', transform=ax.transAxes)
    fig.subplots_adjust(left=.03, right=.90, top=.93, bottom=.09)
    cax = fig.add_axes([.93, .24, .016, .50])
    fig.colorbar(im, cax=cax, label='$\\log(1+\\mathrm{ADC})$')
    fig.savefig(args.output/'projection_examples.pdf')
    fig.savefig(args.output/'projection_examples.png')
    plt.close(fig)

    sample, record = data[0], selected[0]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL, 2.65))
    response = manifest['response']
    for plane, ax in enumerate(axes):
        crop = record['crop'][plane]['bbox_wire_tick']
        row = (crop[0]+crop[2])//2
        lo, hi = max(0,crop[1]-40), min(response['ticks'],crop[3]+80)
        ax.plot(np.arange(lo,hi)*response['sample_us'], sample['waveforms'][plane,row,lo:hi],
                color=COLOURS['proton'], lw=1)
        ax.axhline(0,color='grey',lw=.5)
        ax.set_xlabel('Readout time ($\\mu$s)');ax.set_ylabel('ADC')
        ax.text(.03,.95,['Collection','Induction'][plane],transform=ax.transAxes,va='top')
    fig.tight_layout()
    fig.savefig(args.output/'waveform_examples.pdf')
    fig.savefig(args.output/'waveform_examples.png')
    plt.close(fig)
    print('Saved projection_examples and waveform_examples as PNG/PDF')


if __name__ == '__main__':
    main()
