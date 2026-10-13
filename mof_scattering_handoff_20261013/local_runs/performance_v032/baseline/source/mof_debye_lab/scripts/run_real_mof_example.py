#!/usr/bin/env python3
"""Run the bundled IRMOF-1 CIF in every scattering case without network access.

Writes raw tables, metadata, an expanded cell XYZ and scientific comparison plots.
This is an ordered empty-framework reference model, not an experimental guest fit.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))

import numpy as np

from mofdebye.crystal import read_cif
from mofdebye.runner import run_configuration, write_result


CONFIGURATIONS = ('periodic_single_crystal', 'finite_single_crystal',
                  'pxrd_finite', 'pxrd_bulk', 'pxrd_atom_cut')


def plot_results(results, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    with plt.rc_context({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.titleweight': 'bold'}):
        fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), constrained_layout=True)
        fig.suptitle('IRMOF-1 (MOF-5): symmetry-expanded empty framework', fontsize=16)

        finite = results['finite_single_crystal']['tables']['intensity']
        h = finite['qx'] * 25.832 / (2 * np.pi)
        axes[0, 0].plot(h, finite['total'], color='#176B87', linewidth=1.8)
        axes[0, 0].set(title='Finite single crystal: 2 x 1 x 1 cells',
                       xlabel='h along the (h, 0, 0) line', ylabel=r'Intensity / electron$^2$')
        axes[0, 0].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        reflections = results['periodic_single_crystal']['tables']['reflections']
        nonzero = (reflections['l'] == 0) & (reflections['f_squared'] > 1e-10 * reflections['f_squared'].max())
        values = reflections['f_squared'][nonzero]
        scatter = axes[0, 1].scatter(reflections['h'][nonzero], reflections['k'][nonzero],
                                     c=values, cmap='viridis', s=25,
                                     norm=LogNorm(vmin=values.min(), vmax=values.max()))
        fig.colorbar(scatter, ax=axes[0, 1], label=r'$|F_{hkl}|^2$ / electron$^2$')
        axes[0, 1].set(title='Periodic single crystal: l = 0 slice', xlabel='h', ylabel='k', aspect='equal')
        axes[0, 1].text(0.02, 0.02, 'Zero/very weak symbols omitted; all reflections saved',
                         transform=axes[0, 1].transAxes, fontsize=7)

        for key, label, color in [('pxrd_finite', 'Complete cell: 424 atoms', '#176B87'),
                                   ('pxrd_atom_cut', '12 A atom cut: 232 atoms', '#B86A24')]:
            finite_powder = results[key]['tables']['intensity']
            axes[1, 0].plot(finite_powder['q'], finite_powder['total'], label=label, color=color)
        axes[1, 0].set(title='Finite PXRD: geometric objects', xlabel=r'q / angstrom$^{-1}$',
                       ylabel=r'Intensity / electron$^2$', yscale='log')
        axes[1, 0].legend(frameon=False, fontsize=9)

        shells = results['pxrd_bulk']['tables']['powder_shells']
        axes[1, 1].vlines(shells['q'], 0, shells['radial_weight'], color='#4B5F9E', linewidth=1.1)
        axes[1, 1].set(title='Bulk PXRD: radial delta weights', xlabel=r'q / angstrom$^{-1}$',
                       ylabel=r'Radial weight / electron$^2$ angstrom$^{-1}$')
        axes[1, 1].ticklabel_format(axis='y', style='sci', scilimits=(0, 0))
        axes[1, 1].text(0.97, 0.97, 'Forward delta omitted\nStatic diffuse = 0\nNo instrumental broadening',
                         ha='right', va='top', transform=axes[1, 1].transAxes, fontsize=9)
        for ax in axes.flat:
            ax.grid(alpha=0.16)
        fig.savefig(output / 'real_mof_scattering.png', dpi=180)
        fig.savefig(output / 'real_mof_scattering.pdf')
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path,
                        default=PROJECT.parent / 'local_runs' / 'r07_real_mof')
    parser.add_argument('--no-plot', action='store_true', help='Write all numerical outputs without matplotlib plots.')
    args = parser.parse_args()
    data = PROJECT / 'examples' / 'real_mof'
    output = args.out.resolve()
    provenance = json.loads((data / 'provenance.json').read_text(encoding='utf-8'))
    source = data / provenance['cif']['file']
    if hashlib.sha256(source.read_bytes()).hexdigest() != provenance['cif']['sha256']:
        raise ValueError('Bundled CIF differs from its recorded upstream SHA-256; inspect provenance before running.')
    crystal = read_cif(source)
    expectations = provenance['structural_expectations']
    composition = dict(Counter(crystal.basis.elements))
    if len(crystal.basis) != expectations['expanded_unit_cell_atoms'] or composition != expectations['expanded_composition']:
        raise ValueError('IRMOF-1 expansion differs from its independently checked 424-atom composition.')

    output.mkdir(parents=True, exist_ok=True)
    crystal.basis.write_xyz(output / 'expanded_unit_cell.xyz')
    cell_metadata = {'cell_vectors_A': crystal.cell_vectors_A.tolist(),
                     'expanded_composition': composition, 'crystallography': crystal.metadata,
                     'provenance': provenance}
    (output / 'cell_metadata.json').write_text(json.dumps(cell_metadata, indent=2, allow_nan=False), encoding='utf-8')
    results = {}
    summary = {'material': provenance['material'], 'record_type': provenance['record_type'],
               'source_sha256': provenance['cif']['sha256'],
               'source_url': provenance['cif']['source_url'],
               'expanded_unit_cell_atoms': len(crystal.basis), 'composition': composition,
               'pore_sites': 'Not supplied or inferred; all runs are empty-framework host-only models.',
               'runs': {}}
    for name in CONFIGURATIONS:
        path = data / f'{name}.json'
        configuration = json.loads(path.read_text(encoding='utf-8'))
        result = run_configuration(configuration, data)
        write_result(result, output / name)
        results[name] = result
        summary['runs'][name] = {
            'config_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'scattering_case': configuration['scattering_case'],
            'boundary': configuration['boundary'],
            'host_atoms': result['metadata']['input_processing']['host_atoms'],
            'seconds': result['metadata']['seconds'],
            'tables': {key: len(next(iter(table.values()))) for key, table in result['tables'].items()},
            'actual_method': result['metadata']['actual_method'],
        }
        print(json.dumps({'completed': name, **summary['runs'][name]}), flush=True)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    if not args.no_plot:
        plot_results(results, output)
    print(f'IRMOF-1 examples and raw results: {output}', flush=True)


if __name__ == '__main__':
    main()
