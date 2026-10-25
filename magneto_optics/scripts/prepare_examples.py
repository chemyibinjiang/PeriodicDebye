"""Generate explicitly synthetic dispersive magneto-optical particle fixtures."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('use a new example directory')
    wavelengths = np.linspace(400, 700, 61)
    energy = 1239.841984 / wavelengths
    # A synthetic passive circular-channel Drude table, in eV units.
    # These values are NOT a fit or tabulation for Au, Fe, Co, garnet, etc.
    epsilon_plus = 5 - 7**2 / (energy * (energy + 0.15j - 0.025))
    epsilon_minus = 5 - 7**2 / (energy * (energy + 0.15j + 0.025))
    diagonal = (epsilon_plus + epsilon_minus) / 2
    coupling = (epsilon_minus - epsilon_plus) / 2
    pairs = lambda array: [[float(x.real), float(x.imag)] for x in array]
    config = {
        'schema_version': 'magnetooptics-1.0',
        'description': 'SYNTHETIC gyrotropic sphere array, one electric dipole per sphere. Not measured material data.',
        'geometry': {'kind': 'grid', 'shape': [12, 12, 1], 'spacing_nm': 36, 'radius_nm': 6, 'material': 'synthetic_mo'},
        'materials': {'synthetic_mo': {'model': 'gyrotropic_table', 'wavelength_nm': wavelengths.tolist(),
            'epsilon_diagonal': pairs(diagonal), 'g': pairs(coupling), 'magnetization': [0, 0, 1],
            'provenance': 'Synthetic circular Drude channels: eps_inf=5, plasma_energy=7eV, damping=.15eV, splitting=.025eV; no real-material claim. z diagonal set equal to mean transverse diagonal.'}},
        'host_index': 1., 'wavelength_nm': wavelengths.tolist(),
        'incidence': {'direction': [0, 0, 1], 'reference_axis': [1, 0, 0]},
        'solver': {'backend': 'fft', 'rtol': 1e-10, 'restart': 40, 'maxiter': 200, 'max_memory_mib': 1024},
        'threads': 1,
    }
    variants = {'finite_array': config}
    single = copy.deepcopy(config)
    single['geometry']['shape'] = [1, 1, 1]
    single['solver']['backend'] = 'dense'
    variants['single_sphere'] = single
    volume = copy.deepcopy(config)
    volume['geometry']['shape'] = [8, 8, 8]
    volume['wavelength_nm'] = [430., 460., 490., 550., 650.]
    variants['volume_array'] = volume
    large = copy.deepcopy(config)
    large['geometry']['shape'] = [128, 128, 1]
    variants['large_array'] = large
    args.out.mkdir(parents=True)
    for name, value in variants.items():
        (args.out / f'{name}.json').write_bytes((json.dumps(value, indent=2) + '\n').encode('utf-8'))
    print(json.dumps({'status': 'created', 'files': list(variants), 'output': str(args.out)}))


if __name__ == '__main__':
    main()
