#!/usr/bin/env python3
"""Plot saved synthetic pore-distribution patterns and numerical uncertainty."""
import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--coarse', type=Path)
    args = parser.parse_args()
    meta = json.loads((args.run/'summary.json').read_text(encoding='utf-8'))
    analytic = np.load(args.run/'analytic'/'curves.npz')
    wall = np.load(args.run/'active_wall'/'curves.npz')
    q = wall['q']
    selected = q>=.1
    peak = np.max(abs(wall['total'][selected]))
    significant = selected & (abs(wall['total'])>=.01*peak)
    plt.rcParams.update({'font.size':10, 'axes.spines.top':False, 'axes.spines.right':False})
    fig = plt.figure(figsize=(11, 8), layout='constrained')
    axes = fig.subplot_mosaic([['pattern', 'pattern'], ['parts', 'uncertainty']])
    ax = axes['pattern']
    ax.semilogy(q[selected], analytic['total'][selected], color='#2563eb', lw=1.1,
                label='Centre ball 2 A: analytic confined distribution')
    ax.semilogy(q[selected], wall['total'][selected], color='#dc6b20', lw=1.1,
                label='Proposal ball 4 A: active wall conditioning')
    ax.semilogy(q[selected], wall['host'][selected], color='#4b5563', lw=.75, alpha=.8,
                label='Empty host')
    ax.set(xlabel='q (inverse angstrom)', ylabel='Intensity (electron squared)', xlim=(.1, 6))
    ax.legend(loc='upper right', fontsize=9)
    ax.set_title(f"Synthetic {meta['edge_nm']:g} nm porous crystal: independent Au size and position distributions")
    ax = axes['parts']
    for name, label, color in [('guest_internal', 'Within particles', '#15803d'),
                                ('host_guest', 'Host-particle', '#9333ea'),
                                ('guest_guest', 'Between particles', '#c2410c')]:
        ax.plot(q[selected], wall[name][selected]/peak, label=label, color=color, lw=1)
    ax.axhline(0, color='black', lw=.5)
    ax.set(xlim=(.1, 1.5), xlabel='q (inverse angstrom)', ylabel='Contribution / non-forward total peak',
           title='Signed scattering contributions (active wall)')
    ax.legend(fontsize=8)
    ax = axes['uncertainty']
    values = np.full(len(q), np.nan)
    values[significant] = 300*wall['total_mc_sem'][significant]/abs(wall['total'][significant])
    ax.semilogy(q, values, color='#dc6b20', label='3 x SEM, refined sampling', lw=1)
    comparison = {}
    if args.coarse:
        coarse = np.load(args.coarse/'active_wall'/'curves.npz')
        if not np.array_equal(coarse['q'], q):
            raise ValueError('Coarse and refined q grids must match.')
        values[significant] = 300*coarse['total_mc_sem'][significant]/abs(coarse['total'][significant])
        ax.semilogy(q, values, color='#9ca3af', label='3 x SEM, initial sampling', lw=.85)
        difference = abs(coarse['total']-wall['total'])
        comparison = dict(max_difference_over_nonforward_peak=float(np.max(difference[selected])/peak),
            max_difference_fraction_at_significant_points=float(np.max(difference[significant]/abs(wall['total'][significant]))),
            caveat='Sampling refinement, not an exact reference; consult the saved seeds and sampler metadata for stream independence.')
    ax.axhline(.1, color='black', linestyle='--', lw=.8, label='0.1% target')
    ax.set(xlim=(.1, 6), xlabel='q (inverse angstrom)', ylabel='Estimated integration uncertainty (%)',
           title='Numerical sampling uncertainty; significant points')
    ax.legend(fontsize=8, loc='lower left')
    fig.suptitle('60% occupancy; number weights Au13 / Au19 / Au43 = 0.5 / 0.3 / 0.2\n'
                 'Haar rotations; spherical pore radius 8 A, atom-centre clearance 0.5 A', fontsize=11)
    fig.savefig(args.run/'pore_distribution.png', dpi=180)
    fig.savefig(args.run/'pore_distribution.pdf')
    if comparison:
        (args.run/'sampling_refinement.json').write_text(json.dumps(comparison, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(comparison, indent=2))


if __name__ == '__main__':
    main()
