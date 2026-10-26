"""Build and plot an exact synthetic XX+XX Faraday-mirror demonstration."""
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

from magnetooptics.faraday_mirror import design_transparent_rotator, slab_scattering, mirror_return
from magnetooptics.periodic_observables import polarization_ellipse


def make_slab(design, wavelength=None, magnetization=1.):
    return slab_scattering(design['epsilon_diagonal'], design['g'], design['thickness_nm'],
                           design['wavelength_nm'] if wavelength is None else wavelength,
                           design['host_index'], magnetization)


def json_value(value):
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return np.stack((value.real, value.imag), axis=-1).tolist() if np.iscomplexobj(value) else value.tolist()
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def stage(label, field):
    return {'stage': label, 'electric_field_xy': field, 'power': float(np.vdot(field, field).real),
            **polarization_ellipse(field)}


def draw(design, gap, stages, rows, output):
    fig = plt.figure(figsize=(13, 9), constrained_layout=True)
    grid = fig.add_gridspec(3, 4, height_ratios=[.8, 1.25, 1.4])
    ax = fig.add_subplot(grid[0, :])
    ax.set(xlim=(0, 10), ylim=(-1.1, 1.15))
    ax.axis('off')
    ax.add_patch(plt.Rectangle((3, -.9), 2, 1.8, color='#8dc6ce', alpha=.55))
    ax.plot([8, 8], [-1, 1], color='#555555', lw=7)
    ax.annotate('', xy=(7.8, .45), xytext=(.6, .45), arrowprops={'arrowstyle':'->','lw':2,'color':'#18709e'})
    ax.annotate('', xy=(.6, -.45), xytext=(7.8, -.45), arrowprops={'arrowstyle':'->','lw':2,'color':'#cb5e39'})
    ax.text(4, 0, f"Gyrotropic slab\nd = {design['thickness_nm']:.4g} nm\nM fixed along +z", ha='center', va='center')
    ax.text(6.5, 0, f'Host gap\n{gap:.4g} nm', ha='center', va='center')
    ax.text(8.3, 0, 'Ordinary\nPEC mirror', va='center')
    ax.text(1.7, .7, 'Forward +z', ha='center', color='#18709e')
    ax.text(1.7, -.85, 'Return -z', ha='center', color='#cb5e39')
    phase = np.linspace(0, 2*np.pi, 300)
    labels = ['Input', 'After slab (forward)', 'Just after mirror', 'Returned through slab']
    for index, (record, label) in enumerate(zip(stages, labels)):
        ax = fig.add_subplot(grid[1, index])
        field = np.array(record['electric_field_xy'])
        ellipse = np.real(field[:, None] * np.exp(-1j * phase)[None])
        scale = np.sqrt(record['power'])
        ax.plot(ellipse[0] / scale, ellipse[1] / scale, lw=3, color='#18709e' if index < 2 else '#cb5e39')
        ax.axhline(0, color='#dddddd', lw=.8)
        ax.axvline(0, color='#dddddd', lw=.8)
        ax.set(aspect='equal', xlim=(-1.1,1.1), ylim=(-1.1,1.1), xlabel='lab x', ylabel='lab y')
        angle = record['rotation_deg']
        # +/-90 is the same ellipse axis; keep the raw signed value in JSON.
        display = abs(angle) if abs(abs(angle)-90) < 1e-8 else angle
        ax.set_title(f'{label}\naxis {display:.6g} deg | power {record["power"]:.6g}', fontsize=10)
    waves = np.array([row['wavelength_nm'] for row in rows])
    ax = fig.add_subplot(grid[2, :2])
    def continuous_axes(values):
        return np.degrees(np.unwrap(2*np.radians(values)) / 2)
    ax.plot(waves, continuous_axes([row['single_rotation_deg'] for row in rows]), label='Slab alone: transmitted axis')
    returned = continuous_axes([row['returned_rotation_deg'] for row in rows])
    centre = np.argmin(abs(waves-design['wavelength_nm']))
    returned += 180 * np.round((2*design['design_angle_deg']-returned[centre])/180)
    ax.plot(waves, returned, label='Slab + mirror: returned axis')
    ax.axvline(design['wavelength_nm'], color='#999999', ls=':', label='Design wavelength')
    ax.set(xlabel='Vacuum wavelength (nm)', ylabel='Axis angle (degrees, continuously unwrapped)', title='Fixed synthetic material and thickness')
    ax.legend(fontsize=8)
    ax = fig.add_subplot(grid[2, 2:])
    ax.plot(waves, [row['single_transmission'] for row in rows], label='Slab-alone transmission')
    ax.plot(waves, [row['returned_power'] for row in rows], '--', label='Full mirror return')
    ax.plot(waves, [row['returned_x_power'] for row in rows], label='Return through x analyzer')
    ax.plot(waves, [row['returned_y_power'] for row in rows], label='Return through y analyzer')
    ax.set(xlabel='Vacuum wavelength (nm)', ylabel='Fraction of incident power', ylim=(-.03, 1.05), title='Includes interfaces and all coherent cavity reflections')
    ax.legend(fontsize=8)
    fig.suptitle(f"Faraday double pass: {design['design_angle_deg']:g} deg + {design['design_angle_deg']:g} deg\nSynthetic exact slab benchmark; fixed laboratory axes", fontsize=15)
    fig.savefig(output / 'overview.png', dpi=160)
    fig.savefig(output / 'overview.svg')
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--angle-deg', type=float, default=45.)
    parser.add_argument('--wavelength-nm', type=float, default=600.)
    parser.add_argument('--host-index', type=float, default=1.)
    parser.add_argument('--gap-nm', type=float, default=300.)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('use a new output directory')
    started = perf_counter()
    design = design_transparent_rotator(args.angle_deg, args.wavelength_nm, args.host_index)
    slab = make_slab(design)
    result = mirror_return(slab, args.gap_nm)
    incident = np.array([1., 0.], dtype=complex)
    forward = slab['transmission'] @ incident
    after_mirror = -np.exp(1j * slab['k_host_nm_inv'] * args.gap_nm) * result['gap_forward_jones'] @ incident
    returned = result['returned_jones'] @ incident
    stages = [stage('input', incident), stage('forward_after_material', forward),
              stage('immediately_after_isotropic_mirror', after_mirror), stage('returned_after_material', returned)]
    target = np.radians(2 * args.angle_deg)
    expected_axis = np.array([np.cos(target), np.sin(target)])
    rejected_axis = np.array([-expected_axis[1], expected_axis[0]])
    orthogonal_error = float(abs(np.vdot(rejected_axis, returned))**2)
    rows = []
    for wavelength in np.linspace(.8 * args.wavelength_nm, 1.2 * args.wavelength_nm, 121):
        spectral_slab = make_slab(design, wavelength)
        spectral_return = mirror_return(spectral_slab, args.gap_nm)
        single = spectral_slab['transmission'] @ incident
        back = spectral_return['returned_jones'] @ incident
        rows.append({'wavelength_nm': float(wavelength),
                     'single_rotation_deg': polarization_ellipse(single)['rotation_deg'],
                     'single_ellipticity_deg': polarization_ellipse(single)['ellipticity_deg'],
                     'returned_rotation_deg': polarization_ellipse(back)['rotation_deg'],
                     'returned_ellipticity_deg': polarization_ellipse(back)['ellipticity_deg'],
                     'single_transmission': float(np.vdot(single,single).real),
                     'returned_power': float(np.vdot(back,back).real),
                     'returned_x_power': float(abs(back[0])**2), 'returned_y_power': float(abs(back[1])**2)})
    data = {'design': design, 'gap_nm': args.gap_nm, 'mirror_amplitude': -1., 'stages': stages,
            'slab': slab, 'mirror_system': result,
            'checks': {'slab_reflection_max_amplitude': float(abs(slab['reflection_eigenvalues']).max()),
                       'power_outside_expected_returned_axis': orthogonal_error,
                       'max_spectral_lossless_return_power_error': max(abs(row['returned_power']-1) for row in rows)},
            'scope': 'Exact 1D uniform gyrotropic slab, normal incidence, mu=1, isotropic PEC mirror; synthetic material. This does not implement a mirror Green function for the periodic nanoparticle sheet.',
            'interpretation': 'Mirror itself adds no lab-frame ellipse rotation. Extra angle accumulates on the return through material. Axes modulo180; +/-90 are identical. Offdesign CW response includes prompt and multiple cavity reflections.',
            'source_sha256': {str(path.relative_to(ROOT)).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest()
                              for path in [Path(__file__).resolve(), *sorted((ROOT/'src/magnetooptics').glob('*.py'))]},
            'calculation_seconds': perf_counter()-started}
    if orthogonal_error > 1e-20 or data['checks']['max_spectral_lossless_return_power_error'] > 1e-10:
        raise RuntimeError('Faraday mirror demonstration failed its field or energy acceptance')
    serialized = json.dumps(json_value(data), indent=2, allow_nan=False) + '\n'
    args.out.mkdir(parents=True)
    (args.out/'result.json').write_text(serialized, encoding='utf-8')
    with (args.out/'spectrum.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    draw(design, args.gap_nm, stages, rows, args.out)
    print(json.dumps(json_value({'design':design, 'stages':stages, 'checks':data['checks'], 'output':str(args.out)}), indent=2))


if __name__ == '__main__':
    main()
