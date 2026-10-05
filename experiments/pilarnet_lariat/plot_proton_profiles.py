#!/usr/bin/env python3
"""Plot physical stopping profiles and ADC images for explicitly labelled energies."""

import argparse
from pathlib import Path
import sys
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.pilarnet_lariat.calibrate import BASE
from experiments.pilarnet_lariat.proton_profiles import RR_EDGES, range_energy
from scripts.extra._beam_data import apply_style, SINGLE_COL, DOUBLE_COL, COLOURS


def band(ax, coordinates, samples, color, label, simulated=False):
    samples = np.asarray(samples)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        low, median, high = np.nanquantile(samples, [.16, .5, .84], axis=0)
    valid = np.isfinite(median) & (np.isfinite(samples).sum(axis=0) >= min(3, len(samples)))
    ax.fill_between(coordinates[valid], low[valid], high[valid], color=color, alpha=.14, lw=0)
    ax.plot(coordinates[valid], median[valid], color=color,
            ls='--' if simulated else '-', lw=1, label=label)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=BASE/'profile_matching')
    parser.add_argument('--output', type=Path, default=Path('output/pilarnet_lariat/proton_profiles'))
    parser.add_argument('--fitted', action='store_true', help='Use the fit-only global amplitude correction')
    args = parser.parse_args()
    suffix = '_fitted' if args.fitted else ''
    args.output.mkdir(parents=True, exist_ok=True)
    real_profiles = np.load(args.input/'lariat_dedx_profiles.npy')
    fake_profiles = np.load(args.input/'pilarnet_dedx_profiles.npy')
    centers = (RR_EDGES[:-1]+RR_EDGES[1:])/2
    apply_style(SINGLE_COL)
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL, 4.8))
    labels = ('incoming', 'tpc_range')
    conditions = ('Incoming beam KE', 'TPC range-energy estimate')
    for column, (label, condition) in enumerate(zip(labels, conditions)):
        matches = pd.read_csv(args.input/f'{label}_matches.csv')
        ax = axes[0, column]
        band(ax, centers, real_profiles[matches.real_row.to_numpy()], '#777777', 'LArIAT')
        band(ax, centers, fake_profiles[matches.pilarnet_row.to_numpy()], COLOURS['proton'], 'PILArNet', True)
        expectation = np.diff(range_energy(RR_EDGES))/np.diff(RR_EDGES)
        ax.plot(centers, expectation, color='black', lw=.8, label='Proton model')
        ax.set_xlim(0, 24); ax.set_ylim(0, 65)
        ax.set_xlabel('Residual range (cm)'); ax.set_ylabel('$dE/dx$ (MeV/cm)')
        ax.text(.98, .97, f'{chr(97+column)}) {condition}\n{len(matches)} matched pairs',
                transform=ax.transAxes, ha='right', va='top', fontsize=8)
        ax.legend(loc='upper right', bbox_to_anchor=(1, .78), frameon=False)
        pairs = pd.read_csv(args.input/f'{label}{suffix}_image_pairs.csv')
        heldout = pairs.partition.to_numpy() == 'validation'
        if not heldout.any():
            raise ValueError(f'No held-out image pairs for {label}')
        with np.load(args.input/f'{label}{suffix}_image_pairs.npz') as data:
            # Show the downstream Bragg profile at the model's actual pixel scale.
            observed = data['real'][heldout, 0].max(axis=2)
            projected = data['pilarnet'][heldout, 0].max(axis=2)
        ax = axes[1, column]
        band(ax, np.arange(48), observed, '#777777', 'LArIAT')
        band(ax, np.arange(48), projected, COLOURS['proton'], 'PILArNet', True)
        ax.set_xlabel('Wire-axis pixel'); ax.set_ylabel('Collection row maximum (ADC)')
        ax.set_xlim(0, 47); ax.set_ylim(bottom=0)
        ax.text(.02, .96, f'{chr(99+column)}) ADC profiles: {heldout.sum()} held-out pairs',
                transform=ax.transAxes, va='top', fontsize=8)
    fig.subplots_adjust(left=.08, right=.985, bottom=.095, top=.98, wspace=.30, hspace=.38)
    fig.savefig(args.output/'proton_profile_comparison.pdf')
    fig.savefig(args.output/'proton_profile_comparison.png')
    plt.close(fig)

    selected, image_pairs = [], []
    for label in labels:
        table = pd.read_csv(args.input/f'{label}{suffix}_image_pairs.csv')
        validation = table[table.partition == 'validation']
        candidates = validation if len(validation) else table
        # Pick the middle physical-profile score, rather than a best-looking ADC match.
        index = candidates.sort_values('profile_log_distance').index[len(candidates)//2]
        selected.append((label, table.loc[index]))
        with np.load(args.input/f'{label}{suffix}_image_pairs.npz') as data:
            image_pairs.append((data['real'][index], data['pilarnet'][index]))
    vmax = max(np.log1p(pair).max() for row in image_pairs for pair in row)
    fig, axes = plt.subplots(2, 4, figsize=(DOUBLE_COL, 4.8))
    for row, ((label, record), (observed, projected)) in enumerate(zip(selected, image_pairs)):
        for column, image in enumerate((*observed, *projected)):
            ax = axes[row, column]
            im = ax.imshow(np.log1p(image), cmap='Greys', vmin=0, vmax=vmax,
                           origin='upper', interpolation='nearest')
            ax.set_xticks([0, 24, 47]); ax.set_yticks([0, 24, 47])
            ax.tick_params(length=2, pad=2)
            if column:
                ax.tick_params(labelleft=False)
            if row == 0:
                ax.text(.5, 1.05, ['LArIAT collection', 'LArIAT induction',
                        'PILArNet collection', 'PILArNet induction'][column],
                        transform=ax.transAxes, ha='center', va='bottom', fontsize=8)
        if label == 'incoming':
            text = (f'Incoming KE: LArIAT {record.real_incoming_ke_mev:.1f} MeV; '
                    f'PILArNet {record.pilarnet_incoming_ke_mev:.1f} MeV')
        else:
            text = (f'TPC range estimate: LArIAT {record.real_range_energy_mev:.1f} MeV; '
                    f'PILArNet {record.pilarnet_incoming_ke_mev:.1f} MeV\n'
                    f'LArIAT incoming beam KE remains {record.real_incoming_ke_mev:.1f} MeV')
        axes[row, 0].text(0, -.30 if row == 0 else -.26, text,
                          transform=axes[row, 0].transAxes, fontsize=8, va='top')
    fig.subplots_adjust(left=.055, right=.98, bottom=.27, top=.94, wspace=.13, hspace=.68)
    fig.text(.015, .55, 'Wire-axis pixel', rotation=90, va='center')
    fig.text(.51, .10, 'Time-axis pixel', ha='center')
    cax = fig.add_axes([.32, .045, .40, .017])
    fig.colorbar(im, cax=cax, orientation='horizontal', label='$\log(1+\mathrm{ADC})$')
    fig.savefig(args.output/'energy_matched_images.pdf')
    fig.savefig(args.output/'energy_matched_images.png')
    plt.close(fig)
    print(args.output.resolve())


if __name__ == '__main__':
    main()
