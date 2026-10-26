"""Exact synthetic Bragg-superlattice rotator, including the full mirror cavity."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from magnetooptics.multilayer import layer_stack_scattering, stack_mirror_return
from magnetooptics.periodic_observables import polarization_ellipse


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return np.stack((value.real, value.imag), axis=-1).tolist() if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def make_layers(config, loss=0.):
    layers = []
    for spec in config['layers']:
        diagonal = complex(*spec['epsilon_diagonal'])
        if spec['name'] == 'MO':
            diagonal += 1j * loss
        layers.append({'epsilon_diagonal': diagonal, 'g': complex(*spec['g']),
                       'thickness_nm': spec['thickness_nm']})
    return layers


def field_record(field):
    return {'field_xy': field, 'power': float(np.vdot(field, field).real),
            'x_power': float(abs(field[0])**2), 'y_power': float(abs(field[1])**2),
            **polarization_ellipse(field)}


def calculate(config, wavelength, loss=0., magnetization=1.):
    stack = layer_stack_scattering(make_layers(config, loss), wavelength,
                                   config['host_index'], magnetization * config['magnetization_z'])
    returned = stack_mirror_return(stack, config['gap_nm'], complex(*config['mirror_amplitude']))
    x = np.array([1., 0.], dtype=complex)
    single = field_record(stack['transmission'] @ x)
    reflected = field_record(stack['reflection'] @ x)
    back = field_record(returned['returned_jones'] @ x)
    first = field_record(returned['first_return_jones'] @ x)
    return {'stack': stack, 'mirror_system': returned, 'transmitted': single,
            'reflected': reflected, 'returned': back, 'first_return_path': first,
            'loaded_gap_forward': field_record(returned['gap_forward_jones'] @ x)}


def row(wavelength, result):
    record = {'wavelength_nm': float(wavelength)}
    for label in ('transmitted', 'returned'):
        for key in ('rotation_deg', 'ellipticity_deg', 'power', 'x_power', 'y_power'):
            record[label + '_' + key] = result[label][key]
    record['reflected_power'] = result['reflected']['power']
    record['stack_absorption'] = 1 - result['transmitted']['power'] - result['reflected']['power']
    record['mirror_system_absorption'] = 1 - result['returned']['power']
    return record


def plot(config, cases, centers, output):
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[.55, 1.2, 1.])
    ax = fig.add_subplot(grid[0, :])
    z = 0.
    for layer in config['layers']:
        width = layer['thickness_nm']
        color = {'H': '#38779b', 'L': '#cbe2eb', 'MO': '#e7a05c'}[layer['name']]
        ax.add_patch(plt.Rectangle((z, -.3), width, .6, facecolor=color, edgecolor='white', lw=.6))
        z += width
    ax.plot([z + config['gap_nm']] * 2, [-.5, .5], color='#444444', lw=5)
    ax.annotate('incident +z', xy=(0, .65), xytext=(-250, .65), arrowprops={'arrowstyle': '->'}, va='center')
    ax.text(z / 2, .68, '(H L)^5  |  MO  |  (L H)^5', ha='center', va='center')
    ax.text(z + config['gap_nm'] + 30, 0, 'PEC\nmirror', va='center')
    ax.set(xlim=(-350, z + 270), ylim=(-.55, .95), xlabel='Depth z (nm; dimensions to scale)', yticks=[])
    ax.spines[['left', 'right', 'top']].set_visible(False)
    ax = fig.add_subplot(grid[1, :2])
    for name, rows in cases.items():
        waves = [v['wavelength_nm'] for v in rows]
        for port, color in [('transmitted', '#18709e'), ('returned', '#bd552e')]:
            values = np.array([v[port + '_rotation_deg'] for v in rows], dtype=float)
            values = np.degrees(np.unwrap(2 * np.radians(values)) / 2)
            if port == 'returned':
                values += 180 * np.round((90 - values[len(values)//2]) / 180)
            ax.plot(waves, values, color=color, ls='-' if name == 'ideal' else '--',
                    label=f'{name}: ' + ('stack alone' if port == 'transmitted' else 'full mirror return'))
    ax.axvline(config['wavelength_nm'], color='#999999', ls=':', lw=1)
    ax.set(xlabel='Vacuum wavelength (nm)', ylabel='Ellipse axis (deg, unwrapped)', title='Resonant rotation, not a broadband 45-degree plate')
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[1, 2:])
    for name, rows in cases.items():
        for key, color, label in [('transmitted_power', '#18709e', 'stack transmission'),
                                  ('returned_y_power', '#bd552e', 'returned y-analyzer power')]:
            ax.plot([v['wavelength_nm'] for v in rows], [v[key] for v in rows], color=color,
                    ls='-' if name == 'ideal' else '--', label=f'{name}: {label}')
    ax.set(xlabel='Vacuum wavelength (nm)', ylabel='Fraction of incident power', ylim=(-.02, 1.04),
           title='Lossy case: add Im(epsilon_MO) = 0.001, no retuning')
    ax.legend(fontsize=8)
    labels = [('ideal', 'transmitted', 'Stack alone'), ('ideal', 'returned', 'Stack + mirror'),
              ('lossy', 'transmitted', 'Lossy stack alone'), ('lossy', 'returned', 'Lossy stack + mirror')]
    phase = np.linspace(0, 2 * np.pi, 301)
    for i, (case, port, label) in enumerate(labels):
        ax = fig.add_subplot(grid[2, i])
        rec = centers[case][port]
        field = rec['field_xy']
        ellipse = np.real(field[:, None] * np.exp(-1j * phase))
        ax.plot(ellipse[0], ellipse[1], lw=2.5, color='#18709e' if port == 'transmitted' else '#bd552e')
        ax.axhline(0, color='#dddddd', lw=.8); ax.axvline(0, color='#dddddd', lw=.8)
        ax.set(aspect='equal', xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), xlabel='lab Ex', ylabel='lab Ey')
        theta = rec['rotation_deg']
        theta = abs(theta) if abs(abs(theta)-90) < 1e-5 else theta
        ax.set_title(f'{label}\naxis {theta:.3f} deg; ellipticity {rec["ellipticity_deg"]:.3f} deg\npower {rec["power"]:.4f}', fontsize=9)
    fig.suptitle('Magnetophotonic superlattice + mirror | Synthetic constitutive data\n21 layers, normal incidence, fixed lab axes and magnetization', fontsize=14)
    fig.savefig(output / 'overview.png', dpi=160)
    fig.savefig(output / 'overview.svg')
    plt.close(fig)


def plot_sensitivity(config, cases, gaps, output):
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for name, color in [('ideal', '#18709e'), ('lossy', '#bd552e')]:
        near = [v for v in cases[name] if abs(v['wavelength_nm']-config['wavelength_nm']) <= .3]
        x = np.array([v['wavelength_nm']-config['wavelength_nm'] for v in near])
        theta = np.degrees(np.unwrap(2*np.radians([v['returned_rotation_deg'] for v in near]))/2)
        theta += 180*np.round((90-theta[len(theta)//2])/180)
        axes[0, 0].plot(x, theta, color=color, label=f'{name}: returned axis')
        axes[0, 1].plot(x, [v['returned_y_power'] for v in near], color=color, label=name)
        x = np.array([v['gap_offset_nm'] for v in gaps[name]])
        theta = np.degrees(np.unwrap(2*np.radians([v['rotation_deg'] for v in gaps[name]]))/2)
        theta += 180*np.round((90-theta[len(theta)//2])/180)
        axes[1, 0].plot(x, theta, color=color, label=name)
        axes[1, 1].plot(x, [v['y_power'] for v in gaps[name]], color=color, label=name)
    for i in range(2):
        for j in range(2):
            axes[i, j].axvline(0, color='#aaaaaa', ls=':', lw=1)
            axes[i, j].set_xlabel('Wavelength offset (nm)' if i == 0 else 'Mirror gap offset (nm)')
            axes[i, j].set_ylabel('Returned axis (deg, unwrapped)' if j == 0 else 'Returned y power / incident power')
            axes[i, j].legend(fontsize=8)
    fig.suptitle('Sensitivity of the synthetic superlattice + mirror\nWavelength scan holds gap fixed; gap scan holds wavelength at 600 nm')
    fig.savefig(output/'sensitivity.png', dpi=160)
    fig.savefig(output/'sensitivity.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT/'examples/superlattice/bragg_cavity.json')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('use a new output directory')
    raw = args.config.read_bytes()
    config = json.loads(raw)
    start = perf_counter()
    centers, cases, gaps = {}, {}, {}
    wavelengths = config['wavelength_nm'] + np.unique(np.round(np.concatenate((
        np.linspace(-10, 10, 401), np.linspace(-.5, .5, 201))), 9))
    for name, loss in [('ideal', 0.), ('lossy', .001)]:
        centers[name] = calculate(config, config['wavelength_nm'], loss)
        cases[name] = [row(w, calculate(config, w, loss)) for w in wavelengths]
        gaps[name] = []
        for offset in np.linspace(-5, 5, 201):
            field = stack_mirror_return(centers[name]['stack'], config['gap_nm']+offset,
                                        complex(*config['mirror_amplitude']))['returned_jones'][:, 0]
            gaps[name].append({'gap_offset_nm': float(offset), **field_record(field)})
    mo_layer = [v for v, s in zip(make_layers(config), config['layers']) if s['name'] == 'MO']
    bare = layer_stack_scattering(mo_layer, config['wavelength_nm'], config['host_index'])
    bare_record = field_record(bare['transmission'][:, 0])
    reverse = calculate(config, config['wavelength_nm'], magnetization=-1.)
    zero = calculate(config, config['wavelength_nm'], magnetization=0.)
    ideal = centers['ideal']
    checks = {
        'transmitted_axis_error_deg': abs(ideal['transmitted']['rotation_deg'] - 45),
        'transmitted_ellipticity_deg': abs(ideal['transmitted']['ellipticity_deg']),
        'returned_x_power': ideal['returned']['x_power'],
        'max_lossless_stack_energy_error': max(abs(v['stack_absorption']) for v in cases['ideal']),
        'max_lossless_mirror_energy_error': max(abs(1 - v['returned_power']) for v in cases['ideal']),
        'magnetization_reversal_axis_error_deg': abs(ideal['transmitted']['rotation_deg'] + reverse['transmitted']['rotation_deg']),
        'unmagnetized_transmitted_y_power': zero['transmitted']['y_power'],
        'unmagnetized_returned_y_power': zero['returned']['y_power'],
        'lossy_minimum_absorption': min(min(v['stack_absorption'], v['mirror_system_absorption']) for v in cases['lossy']),
    }
    if (max(checks[k] for k in checks if k.endswith('_deg')) > 1e-5
            or checks['returned_x_power'] > 1e-12
            or max(checks['max_lossless_stack_energy_error'], checks['max_lossless_mirror_energy_error']) > 1e-10
            or checks['lossy_minimum_absorption'] < -1e-10
            or checks['unmagnetized_transmitted_y_power'] > 1e-20
            or checks['unmagnetized_returned_y_power'] > 1e-20):
        raise RuntimeError(f'superlattice fixture acceptance failed: {checks}')
    result = {'input': config, 'input_sha256': hashlib.sha256(raw).hexdigest(),
              'design_results': centers, 'same_MO_layer_without_Bragg_layers': bare_record,
              'magnetization_reversed': reverse, 'unmagnetized': zero, 'checks': checks,
              'wavelength_points_per_case': len(wavelengths),
              'wavelength_plus_0p1_nm': {name: next(v for v in rows if abs(v['wavelength_nm']-config['wavelength_nm']-.1) < 1e-9) for name, rows in cases.items()},
              'calculation_seconds': perf_counter() - start,
              'interpretation': 'Stack-alone transmission, first return path, and full coherent mirror return are distinct fields. The ordinary mirror adds no ellipse rotation. All data are synthetic; no experimental garnet performance is asserted.',
              'source_sha256': {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in [Path(__file__).resolve(), ROOT/'scripts/design_superlattice.py', *sorted((ROOT/'src/magnetooptics').glob('*.py'))]}}
    args.out.mkdir(parents=True)
    (args.out/'input.json').write_bytes(raw)
    (args.out/'result.json').write_text(json.dumps(serial(result), indent=2, allow_nan=False)+'\n', encoding='utf-8')
    for name, rows in cases.items():
        with (args.out/f'{name}_spectrum.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
        gap_rows = [{k: v for k, v in row.items() if k != 'field_xy'} for row in gaps[name]]
        with (args.out/f'{name}_gap.csv').open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(gap_rows[0])); writer.writeheader(); writer.writerows(gap_rows)
    plot(config, cases, centers, args.out)
    plot_sensitivity(config, cases, gaps, args.out)
    print(json.dumps(serial({'checks': checks, 'center': {n: {p: v[p] for p in ('transmitted', 'returned', 'first_return_path')} for n, v in centers.items()},
                             'bare_MO': bare_record, 'calculation_seconds': result['calculation_seconds']}), indent=2))


if __name__ == '__main__':
    main()
