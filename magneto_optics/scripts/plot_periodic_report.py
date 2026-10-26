"""Plot actual saved periodic spectra, never synthetic stand-in output curves."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_spectrum(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows]) for key in rows[0]}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    square = read_spectrum(args.report / 'square_sheet' / 'spectrum.csv')
    diffractive = read_spectrum(args.report / 'diffractive_sheet' / 'spectrum.csv')
    config = json.loads((args.report / 'square_sheet' / 'config.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    ax = axes[0, 0]
    periods = np.array(config['geometry']['periods_nm'])
    radius = float(config['geometry']['radius_nm'])
    for ix in [-1, 0, 1]:
        for iy in [-1, 0, 1]:
            for position in config['geometry']['positions_nm']:
                xy = np.array(position[:2]) + periods * [ix, iy]
                ax.add_patch(plt.Circle(xy, radius, color='#287f9c' if ix == iy == 0 else '#b8d7df'))
    ax.add_patch(plt.Rectangle(-periods / 2, *periods, fill=False, ls='--', lw=1.7, color='#cf6138'))
    ax.set(xlim=(-60, 60), ylim=(-60, 60), xlabel='x (nm)', ylabel='y (nm)',
           title='Periodic sheet: radius 6 nm, period 36 nm', aspect='equal')
    ax.text(.02, .02, 'Dashed cell shifted for display; infinitely repeated', transform=ax.transAxes, fontsize=8)
    wavelength = square['wavelength_nm']
    for key, label in [('total_transmission', 'T, all orders'), ('total_reflection', 'R, all orders'), ('absorption', 'A, material work')]:
        axes[0, 1].plot(wavelength, square[f'{key}_Mplus_x'], label=label)
    axes[0, 1].set(title='Power budget: +M, x incidence', xlabel='Vacuum wavelength (nm)', ylabel='Incident power fraction')
    axes[0, 1].legend(fontsize=8)
    for index, quantity in enumerate(['faraday', 'kerr']):
        ax = axes[1, index]
        for sign, style in [('Mplus', '-'), ('Mminus', '--')]:
            ax.plot(wavelength, square[f'{quantity}_rotation_deg_{sign}'], style, label=f'{sign}: rotation')
        ax.plot(wavelength, square[f'{quantity}_ellipticity_deg_Mplus'], ':', label='+M: ellipticity')
        ax.set(title=f'{quantity.title()} zero-order ellipse, laboratory xy', xlabel='Vacuum wavelength (nm)', ylabel='Angle (degrees)')
        ax.legend(fontsize=8)
    ax = axes[0, 2]
    residual = [max(abs(square[f'energy_balance_residual_{m}_{p}'][i]) for m in ['Mplus', 'Mminus'] for p in ['x', 'y', 'plus', 'minus']) for i in range(len(wavelength))]
    ax.semilogy(wavelength, np.maximum(residual, np.finfo(float).eps), label='max |T + R + A - 1|')
    ax.set(title='Independent energy closure', xlabel='Vacuum wavelength (nm)', ylabel='Absolute residual (plot floor: machine epsilon)')
    ax.legend(fontsize=8)
    ax = axes[1, 2]
    with (args.report / 'diffractive_sheet' / 'diffraction_orders.csv').open(encoding='utf-8', newline='') as handle:
        order_rows = list(csv.DictReader(handle))
    higher = np.array([sum(float(row['transmission']) + float(row['reflection']) for row in order_rows
                          if float(row['wavelength_nm']) == lam and row['magnetization'] == 'Mplus'
                          and row['polarization'] == 'x' and (int(row['m']) != 0 or int(row['n']) != 0))
                       for lam in diffractive['wavelength_nm']])
    ax.semilogy(diffractive['wavelength_nm'], np.maximum(higher, 1e-18), 'o-', label='T + R outside zero order')
    for lam, val, count in zip(diffractive['wavelength_nm'], higher, diffractive['propagating_orders_per_side']):
        ax.annotate(f'{int(count)} orders', (lam, max(val, 1e-18)), xytext=(0, 7), textcoords='offset points', ha='center', fontsize=8)
    ax.set(title='650 x 570 nm diffractive cell', xlabel='Vacuum wavelength (nm)', ylabel='Nonzero-order power fraction (floor: 1e-18)')
    ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('Periodic magneto-optical metasurfaces | Synthetic materials, electric-dipole model', fontsize=14)
    fig.savefig(args.report / 'overview.png', dpi=160)
    fig.savefig(args.report / 'overview.svg')
    plt.close(fig)


if __name__ == '__main__':
    main()
