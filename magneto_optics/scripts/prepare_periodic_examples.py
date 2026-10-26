"""Reproduce synthetic periodic metasurface configurations; no measured data."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np


def configurations():
    wavelengths = np.linspace(400, 700, 61)
    energy = 1239.841984 / wavelengths
    plus = 5 - 49 / (energy * (energy + 0.15j - 0.025))
    minus = 5 - 49 / (energy * (energy + 0.15j + 0.025))
    pairs = lambda values: [[float(v.real), float(v.imag)] for v in values]
    base = {
        'schema_version': 'magnetooptics-periodic-1.0',
        'description': 'SYNTHETIC infinite rectangular array of small gyrotropic spheres. One electric dipole per sphere; homogeneous host; normal incidence.',
        'geometry': {'periods_nm': [36., 36.], 'positions_nm': [[0., 0., 0.]],
                     'radius_nm': 6., 'material': 'synthetic_mo'},
        'materials': {'synthetic_mo': {
            'model': 'gyrotropic_table', 'wavelength_nm': wavelengths.tolist(),
            'epsilon_diagonal': pairs((plus + minus) / 2), 'g': pairs((minus - plus) / 2),
            'magnetization': [0., 0., 1.],
            'provenance': 'Synthetic passive circular Drude channels: eps_inf=5, plasma_energy=7 eV, damping=.15 eV, splitting=.025 eV; NOT experimental material data. z diagonal equals mean transverse diagonal.'}},
        'wavelength_nm': wavelengths.tolist(), 'host_index': 1.,
        'incidence': {'direction': [0., 0., 1.]},
        'solver': {'backend': 'cached', 'rtol': 1e-10, 'ewald_tolerance': 1e-10,
                   'max_memory_mib': 1024},
        'threads': 1,
    }
    dimer = copy.deepcopy(base)
    dimer['geometry'] = {'periods_nm': [72., 60.],
                        'positions_nm': [[18., 15., 0.], [54., 45., 0.]],
                        'radius_nm': [6., 8.], 'material': 'synthetic_mo'}
    diffractive = copy.deepcopy(base)
    diffractive['geometry'] = {'periods_nm': [650., 570.],
                              'positions_nm': [[80., 70., 0.], [320., 280., 0.]],
                              'radius_nm': [10., 12.], 'material': 'synthetic_mo'}
    diffractive['wavelength_nm'] = [450., 500., 550., 610., 680.]
    diffractive['description'] += ' Multiple propagating diffraction orders; selected wavelengths avoid Rayleigh thresholds.'
    lossless = copy.deepcopy(base)
    lossless['materials']['synthetic_mo'] = {
        'model': 'gyrotropic_constant', 'epsilon_diagonal': [2.5, 0.],
        'g': [0.15, 0.], 'magnetization': [0., 0., 1.],
        'provenance': 'Synthetic lossless Hermitian gyrotropic tensor for conservation checks.'}
    lossless['wavelength_nm'] = [400., 500., 600., 700.]
    return {'square_sheet': base, 'two_site_sheet': dimer,
            'diffractive_sheet': diffractive, 'lossless_sheet': lossless}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    configs = configurations()
    args.out.mkdir(parents=True, exist_ok=False)
    for name, config in configs.items():
        (args.out / f'{name}.json').write_bytes(
            (json.dumps(config, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps({'output': str(args.out), 'examples': list(configs)}))


if __name__ == '__main__':
    main()
