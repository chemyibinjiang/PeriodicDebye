"""Plot saved scientific results; no simulation or fabricated data."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def table(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows]) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    root = args.report
    benchmark = json.loads((root / 'performance/benchmark.json').read_text(encoding='utf-8'))
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig.suptitle('Finite magneto-optical particle arrays | synthetic validation', fontsize=16, fontweight='bold')
    geometry = np.load(root / 'finite_array/geometry.npz')
    ax = axes[0, 0]
    for position, radius in zip(geometry['positions_nm'], geometry['radii_nm']):
        ax.add_patch(Circle(position[:2], radius, facecolor='#0d7490', edgecolor='none'))
    ax.set(xlim=(-20, 416), ylim=(-20, 416), aspect='equal', xlabel='x (nm)', ylabel='y (nm)', title='A  12 x 12 separated spheres')
    ax.text(.04, .95, 'radius 6 nm; pitch 36 nm\nM and incident k along z', transform=ax.transAxes, va='top', fontsize=9,
            bbox={'facecolor': 'white', 'edgecolor': '#d4d4d8', 'alpha': .93})
    labels = [('single_sphere', 1, 'One sphere', '#64748b'), ('finite_array', 144, '144 spheres', '#0891b2'), ('large_array', 16384, '16,384 spheres', '#b45309')]
    for name, count, label, color in labels:
        data = table(root / name / 'spectrum.csv')
        axes[0, 1].plot(data['wavelength_nm'], data['mean_extinction_nm2'] / count, label=label, color=color, lw=1.8)
        axes[1, 0].plot(data['wavelength_nm'], 100 * data['magnetic_cd_normalized'], label=label, color=color, lw=1.8)
    axes[0, 1].set(xlabel='Vacuum wavelength (nm)', ylabel='Mean extinction / particle (nm²)', title='B  Self-consistent extinction spectrum')
    axes[0, 1].legend(frameon=False)
    axes[1, 0].axhline(0, color='#a1a1aa', lw=.7)
    axes[1, 0].set(xlabel='Vacuum wavelength (nm)', ylabel='Magnetization-odd CD / mean extinction (%)', title='C  Four channels at every wavelength')
    ax = axes[1, 1]
    small = benchmark['same_size_comparisons']
    for backend, label, color in [('dense_lu', 'Dense LU', '#64748b'), ('dense_gmres', 'Dense GMRES', '#a855f7'), ('fft_gmres', 'FFT + GMRES', '#0891b2')]:
        counts = [case['n_particles'] for case in small]
        times = [case['backends'][backend]['median_seconds']['total_seconds'] for case in small]
        if backend == 'fft_gmres':
            large = [case for case in benchmark['large_fft_cases'] if case['shape'][2] == 1]
            counts += [case['n_particles'] for case in large]
            times += [case['total_seconds'] for case in large]
        ax.loglog(counts, times, 'o-', label=label, color=color, markersize=5)
    ax.set(xlabel='Particles in finite planar array', ylabel='Four-channel calculation (s)', title='D  Measured CPU time at 500 nm')
    ax.legend(frameon=False)
    ax.text(.04, .94, '1 thread; numerical target 0.1%\n64–576: median of 3 runs\nLarge arrays: one measured run', transform=ax.transAxes, va='top', fontsize=9)
    for ax in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        ax.grid(alpha=.2)
    fig.savefig(root / 'overview.png', dpi=180)
    fig.savefig(root / 'overview.svg')
    plt.close(fig)
    print(root / 'overview.png')


if __name__ == '__main__':
    main()
