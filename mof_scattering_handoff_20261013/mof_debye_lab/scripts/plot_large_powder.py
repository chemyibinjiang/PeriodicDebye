#!/usr/bin/env python3
"""Render saved finite PXRD results without recalculating the model."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    record = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    data = np.load(args.run/'pattern.npz')
    q, intensity, bound = data['q'], data['intensity'], data['approximation_bound']
    region = q >= .1
    maximum = np.flatnonzero(region)[np.argmax(intensity[region])]
    scale = intensity[maximum]
    color = '#126f8c'
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(f"Finite MOF-5 powder | {record['edge_nm']:.3f} nm cube | "
                 f"{record['represented_atoms']:,} represented atoms", fontsize=15)
    axes[0,0].plot(q[region], intensity[region]/scale, lw=.8, color=color)
    axes[0,0].set(title='Resolved powder pattern', ylabel='Intensity / strongest non-forward peak')
    axes[0,1].plot(q[region], intensity[region]/scale, lw=.8, color=color)
    axes[0,1].set(yscale='log', title='Same pattern on a logarithmic scale', ylabel='Relative intensity')
    peak = abs(q-q[maximum]) <= .025
    axes[1,0].plot(q[peak], intensity[peak]/scale, '.-', ms=3, lw=1, color=color)
    axes[1,0].set(title=f"Strongest peak: q = {q[maximum]:.4f} $\mathrm{{\AA}}^{{-1}}$",
                  ylabel='Relative intensity')
    axes[1,1].semilogy(q[region], np.maximum(bound[region]/scale, 1e-18),
                        label='Analytical approximation bound*', color='#b06813', lw=1)
    refined_path = args.run/'refined.npz'
    if refined_path.is_file():
        refined = np.load(refined_path)['intensity']
        discrepancy = abs(intensity-refined)/scale
        axes[1,1].semilogy(q[region], np.maximum(discrepancy[region], 1e-18),
                            label='Bin/order refinement discrepancy', lw=.8, color=color)
    exact_path = args.run/'selected_exact.npz'
    if exact_path.is_file():
        exact = np.load(exact_path)
        discrepancy = abs(exact['actual']-exact['reference'])/scale
        axes[1,1].scatter(exact['q'], np.maximum(discrepancy, 1e-18),
                         label='Selected exact Debye checks', color='#b93145', s=24, zorder=4)
    axes[1,1].axhline(.001, linestyle='--', color='#555555', label='0.1% peak-scaled target')
    axes[1,1].set(title='Numerical accuracy (forward beam excluded)', ylabel='Absolute error / non-forward peak')
    axes[1,1].legend(fontsize=8, loc='best')
    for axis in axes.flat:
        axis.set_xlabel(r'$q$ ($\mathrm{\AA}^{-1}$)')
        axis.grid(alpha=.15)
        axis.spines[['top','right']].set_visible(False)
    fig.supxlabel('*Bound excludes floating-point roundoff. Error display floor: 1e-18. '
                  'Complete cells; neutral IT92 factors; no instrumental broadening.', fontsize=9)
    for extension in ('png','pdf'):
        fig.savefig(args.run/f'finite_pxrd.{extension}', dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
