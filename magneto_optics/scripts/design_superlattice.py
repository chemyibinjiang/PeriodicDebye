"""Reproduce the synthetic 600 nm, 45-degree Bragg-cavity benchmark input."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
from scipy.optimize import root, brentq
from magnetooptics.multilayer import layer_stack_scattering, stack_mirror_return


def design():
    wavelength, high, low, pairs = 600., 2.3, 1.45, 5

    def layers(g, thickness):
        h = {'name': 'H', 'epsilon_diagonal': high**2, 'g': 0., 'thickness_nm': wavelength/(4*high)}
        l = {'name': 'L', 'epsilon_diagonal': low**2, 'g': 0., 'thickness_nm': wavelength/(4*low)}
        mo = {'name': 'MO', 'epsilon_diagonal': high**2, 'g': float(g), 'thickness_nm': float(thickness)}
        return [dict(v) for v in ([h, l]*pairs + [mo] + [l, h]*pairs)]

    def residual(parameters):
        t = layer_stack_scattering(layers(*parameters), wavelength)['transmission_eigenvalues']
        ratio = t[1]/t[0]
        # Equal amplitudes give a line; phase difference pi/2 gives 45 degrees.
        return [np.angle(ratio)-np.pi/2, np.log(abs(ratio))]

    solution = root(residual, [.029, wavelength/(2*high)], tol=1e-10)
    if np.max(np.abs(residual(solution.x))) > 1e-10 or not (.001 < solution.x[0] < .06):
        raise RuntimeError(f'synthetic design failed: {solution.message}')
    stack = layer_stack_scattering(layers(*solution.x), wavelength)

    def returned_axis(gap):
        field = stack_mirror_return(stack, gap)['returned_jones'][:, 0]
        # Near the chosen lossless 90-degree solution Ey is nonzero and Ex/Ey is real.
        return float((field[0]/field[1]).real)

    gap = brentq(returned_axis, 40., 60., xtol=1e-11)
    final_layers = layers(*solution.x)
    for layer in final_layers:
        for key in ('epsilon_diagonal', 'g'):
            layer[key] = [float(layer[key]), 0.]
    return {
        'schema_version': 'synthetic-superlattice-fixture-1.0',
        'provenance': 'Entirely synthetic nondispersive materials, not measured garnet, silica or any named material. g and MO thickness fitted to equal-amplitude circular transmission with pi/2 phase difference at 600 nm. PEC gap then fitted to zero returned Ex.',
        'wavelength_nm': wavelength, 'host_index': 1., 'gap_nm': float(gap),
        'mirror_amplitude': [-1., 0.], 'magnetization_z': 1.,
        'architecture': '(H L)^5 MO (L H)^5; H,L quarter-wave dielectric layers; central axial gyrotropic defect',
        'layers': final_layers,
        'loss_stress': 'Demo adds +0.001i to the MO diagonal permittivity without retuning; synthetic sensitivity test, not an estimate of measured loss.',
        'scope': 'Normal incidence, mu_r=1, identical lossless hosts, axial magnetization, laterally uniform planar layers; no 2D nanodisk array.',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    config = design()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x', encoding='utf-8') as f:
        json.dump(config, f, indent=2, allow_nan=False)
        f.write('\n')
    print(json.dumps({'output': str(args.out), 'MO': config['layers'][10], 'gap_nm': config['gap_nm']}, indent=2))


if __name__ == '__main__':
    main()
