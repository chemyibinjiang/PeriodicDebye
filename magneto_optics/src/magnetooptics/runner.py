"""Strict JSON-to-spectrum interface for finite magneto-optical particle arrays."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy
from threadpoolctl import threadpool_limits

from . import __version__
from .geometry import from_config, _real_array
from .interaction import DenseGreenOperator, FFTGreenOperator
from .materials import gyrotropic_permittivity, sphere_polarizability
from .observables import circular_polarizations, plane_wave, cross_sections, far_field
from .solver import CoupledDipoleSystem


def _complex(value, name):
    if np.isscalar(value) and not isinstance(value, (str, bool)):
        result = complex(value)
    else:
        raw = _real_array(value, name)
        if raw.shape != (2,) or raw.dtype.kind not in 'fiu':
            raise ValueError(f'{name} must be a number or [real, imaginary]')
        result = complex(float(raw[0]), float(raw[1]))
    if not np.isfinite(result):
        raise ValueError(f'{name} must be finite')
    return result


def material_tensor(spec, wavelength_nm, magnetization_sign=1, *, _validated=False):
    allowed = {'model', 'epsilon_diagonal', 'g', 'magnetization', 'wavelength_nm', 'provenance', 'label'}
    if unknown := set(spec) - allowed:
        raise ValueError(f'unknown material fields: {sorted(unknown)}')
    model = spec.get('model', 'gyrotropic_constant')
    if model == 'gyrotropic_constant':
        if 'wavelength_nm' in spec:
            raise ValueError('constant material cannot contain a wavelength table')
        diagonal = _complex(spec['epsilon_diagonal'], 'epsilon_diagonal')
        g = _complex(spec.get('g', 0), 'g')
    elif model == 'gyrotropic_table':
        wavelength = _real_array(spec['wavelength_nm'], 'material wavelength_nm')
        if wavelength.ndim != 1 or len(wavelength) < 2 or not np.isfinite(wavelength).all() or np.any(wavelength <= 0) or np.any(np.diff(wavelength) <= 0):
            raise ValueError('material wavelength_nm must be positive and strictly increasing')
        diagonal_table = np.array([_complex(v, 'epsilon_diagonal') for v in spec['epsilon_diagonal']])
        g_table = np.array([_complex(v, 'g') for v in spec['g']])
        if diagonal_table.shape != wavelength.shape or g_table.shape != wavelength.shape:
            raise ValueError('material tables must have the same length')
        if not _validated:
            for diagonal_knot, g_knot in zip(diagonal_table, g_table):
                gyrotropic_permittivity(diagonal_knot, g_knot, spec.get('magnetization', [0, 0, 1]))
        if wavelength_nm < wavelength[0] or wavelength_nm > wavelength[-1]:
            raise ValueError('material extrapolation is forbidden; extend the supplied table')
        diagonal = np.interp(wavelength_nm, wavelength, diagonal_table)
        g = np.interp(wavelength_nm, wavelength, g_table)
    else:
        raise ValueError('material model must be gyrotropic_constant or gyrotropic_table')
    return gyrotropic_permittivity(diagonal, g, magnetization_sign * np.asarray(spec.get('magnetization', [0, 0, 1]), dtype=float))


def validate_config(config):
    allowed = {'schema_version', 'description', 'geometry', 'materials', 'wavelength_nm', 'host_index', 'incidence', 'solver', 'threads'}
    if unknown := set(config) - allowed:
        raise ValueError(f'unknown configuration fields: {sorted(unknown)}')
    if config.get('schema_version') != 'magnetooptics-1.0':
        raise ValueError('schema_version must be magnetooptics-1.0')
    geometry = from_config(config['geometry'])
    wavelengths = _real_array(config['wavelength_nm'], 'wavelength_nm')
    if wavelengths.ndim != 1 or not len(wavelengths) or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError('wavelength_nm must be a positive strictly increasing list')
    nh = config.get('host_index', 1.)
    if isinstance(nh, bool) or not np.isscalar(nh) or not np.isrealobj(nh) or not np.isfinite(nh) or nh <= 0:
        raise ValueError('host_index must be finite, real and positive')
    materials = config['materials']
    if not isinstance(materials, dict) or not materials:
        raise ValueError('materials must be a nonempty named mapping')
    gspec = config['geometry']
    if 'material' in gspec and 'site_materials' in gspec:
        raise ValueError('specify material or site_materials, not both')
    names = gspec.get('site_materials', [gspec.get('material', next(iter(materials)))] * len(geometry.positions_nm))
    if not isinstance(names, list) or len(names) != len(geometry.positions_nm) or any(not isinstance(name, str) or name not in materials for name in names):
        raise ValueError('site_materials must name one existing material per particle')
    for spec in materials.values():
        # A fixed-magnetization table interpolates passive Hermitian loss matrices
        # convexly. Validate every knot once and check both requested endpoints.
        for wavelength in [wavelengths[0], wavelengths[-1]]:
            material_tensor(spec, float(wavelength))  # Includes passivity before expensive allocation.
    solver = dict(config.get('solver', {}))
    if unknown := set(solver) - {'backend', 'rtol', 'restart', 'maxiter', 'max_memory_mib'}:
        raise ValueError(f'unknown solver fields: {sorted(unknown)}')
    backend = solver.get('backend', 'fft')
    if backend not in ('fft', 'dense', 'dense_gmres'):
        raise ValueError('backend must be fft, dense or dense_gmres')
    if backend == 'fft' and geometry.indices is None:
        raise ValueError('FFT requires exact orthogonal-grid geometry; no automatic snapping')
    for name, default in [('restart', 40), ('maxiter', 200), ('max_memory_mib', 1024)]:
        value = solver.setdefault(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f'{name} must be a positive integer')
    rtol = solver.setdefault('rtol', 1e-9)
    if isinstance(rtol, bool) or not np.isscalar(rtol) or not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError('rtol must be between zero and one')
    solver['backend'] = backend
    threads = config.get('threads', 1)
    if isinstance(threads, bool) or not isinstance(threads, int) or threads <= 0:
        raise ValueError('threads must be a positive integer')
    incidence = config.get('incidence', {})
    if unknown := set(incidence) - {'direction', 'reference_axis'}:
        raise ValueError(f'unknown incidence fields: {sorted(unknown)}')
    direction = np.asarray(incidence.get('direction', [0, 0, 1]), dtype=float)
    plus, minus = circular_polarizations(direction, incidence.get('reference_axis'))
    return geometry, wavelengths, float(nh), names, solver, threads, direction, plus, minus


def _ellipse(field, first, second):
    ex, ey = np.vdot(first, field), np.vdot(second, field)
    s0 = float(abs(ex)**2 + abs(ey)**2)
    if s0 == 0:
        return {'rotation_deg': None, 'ellipticity_deg': None, 'cross_fraction': None}
    s1 = float(abs(ex)**2 - abs(ey)**2)
    s2, s3 = 2 * np.real(np.conj(ex) * ey), 2 * np.imag(np.conj(ex) * ey)
    circular = np.hypot(s1, s2) <= 64 * np.finfo(float).eps * s0
    return {'rotation_deg': None if circular else float(np.degrees(0.5 * np.arctan2(s2, s1))),
            'ellipticity_deg': float(np.degrees(0.5 * np.arcsin(np.clip(s3 / s0, -1, 1)))),
            'cross_fraction': float(abs(ey)**2 / s0)}


def simulate(config):
    geometry, wavelengths, nh, names, settings, threads, direction, plus, minus = validate_config(config)
    started = perf_counter()
    rows, records = [], []
    direction = direction / np.linalg.norm(direction)
    basis_x, basis_y = (plus + minus) / np.sqrt(2), (plus - minus) / (1j * np.sqrt(2))
    audit = geometry.audit()
    audit['maximum_host_size_parameter_ka'] = float(2 * np.pi * nh * geometry.radii_nm.max() / wavelengths.min())
    maximum_internal_size = 0.
    with threadpool_limits(limits=threads):
        for wavelength in wavelengths:
            k = 2 * np.pi * nh / wavelength
            memory = settings['max_memory_mib'] * 1024**2
            if settings['backend'] == 'fft':
                operator = FFTGreenOperator(geometry.indices, geometry.spacing_nm, k, max_memory_bytes=memory)
            else:
                operator = DenseGreenOperator(geometry.positions_nm, k, max_memory_bytes=memory)
            row = {'wavelength_nm': float(wavelength)}
            cases, errors = {}, []
            for sign, label in [(1, 'Mplus'), (-1, 'Mminus')]:
                tensors = {name: material_tensor(spec, float(wavelength), sign, _validated=True) for name, spec in config['materials'].items()}
                epsilon = np.array([tensors[name] for name in names])
                internal_size = 2 * np.pi / wavelength * geometry.radii_nm * np.sqrt(np.linalg.norm(epsilon, ord=2, axis=(-2, -1)))
                maximum_internal_size = max(maximum_internal_size, float(internal_size.max()))
                alpha = sphere_polarizability(epsilon, geometry.radii_nm, wavelength, host_index=nh)
                system = CoupledDipoleSystem(operator, alpha, method='dense' if settings['backend'] == 'dense' else 'gmres')
                dipoles = []
                for helicity, polarization in [('plus', plus), ('minus', minus)]:
                    incident = plane_wave(geometry.positions_nm, wavelength, direction, polarization, host_index=nh)
                    result = system.solve(incident, rtol=settings['rtol'], restart=settings['restart'], maxiter=settings['maxiter'])
                    sections = cross_sections(result.dipoles, incident, result.local_field, k)
                    for name, value in sections.items():
                        row[f'{name}_{label}_{helicity}'] = float(value)
                    cases[f'{label}_{helicity}'] = dict(result.metadata)
                    errors.append(result.relative_residual)
                    dipoles.append(result.dipoles)
                # Linearity supplies a linear-incident scattered field without another solve.
                linear = (dipoles[0] + dipoles[1]) / np.sqrt(2)
                field = far_field(linear, geometry.positions_nm, k, direction[None])[0]
                for name, value in _ellipse(field, basis_x, basis_y).items():
                    row[f'forward_scattered_{name}_{label}'] = value
            delta_positive = row['extinction_nm2_Mplus_plus'] - row['extinction_nm2_Mplus_minus']
            delta_negative = row['extinction_nm2_Mminus_plus'] - row['extinction_nm2_Mminus_minus']
            average = sum(row[f'extinction_nm2_{m}_{h}'] for m in ('Mplus', 'Mminus') for h in ('plus', 'minus')) / 4
            row['magnetic_cd_extinction_nm2'] = (delta_positive - delta_negative) / 2
            row['even_cd_extinction_nm2'] = (delta_positive + delta_negative) / 2
            row['mean_extinction_nm2'] = average
            row['magnetic_cd_normalized'] = None if average == 0 else row['magnetic_cd_extinction_nm2'] / average
            row['max_relative_residual'] = max(errors)
            rows.append(row)
            records.append({'wavelength_nm': float(wavelength), 'operator': dict(operator.metadata), 'solves': cases})
    audit['maximum_internal_size_parameter'] = maximum_internal_size
    advisories = []
    if audit['maximum_host_size_parameter_ka'] > 0.3 or maximum_internal_size > 0.5:
        advisories.append('Particle size parameters exceed the conservative small-particle screening values ka=0.3 or internal=0.5; use multipole or converged volumetric EM validation.')
    if audit['minimum_center_distance_nm'] is not None and audit['maximum_radius_nm'] / audit['minimum_center_distance_nm'] > 0.2:
        advisories.append('Particle separation is close: maximum radius/minimum centre distance exceeds 0.2; omitted multipoles may matter.')
    metadata = {
        'package_version': __version__, 'model': 'finite electric coupled-dipole homogeneous spheres',
        'time_convention': 'exp(-i omega t)', 'length_unit': 'nm', 'cross_section_unit': 'nm^2',
        'polarizability_unit': 'nm^3', 'host_index': nh, 'geometry': audit,
        'material_provenance': {name: spec.get('provenance', 'user supplied; not independently validated') for name, spec in config['materials'].items()},
        'helicity_convention': 'e_plus=(e1+i e2)/sqrt(2), e_minus=(e1-i e2)/sqrt(2); e1 cross e2 = propagation direction',
        'magnetic_cd_definition': '[(Cplus(+M)-Cminus(+M))-(Cplus(-M)-Cminus(-M))]/2',
        'normalized_cd_denominator': 'mean of the four extinction cross sections; undefined for zero denominator',
        'polarization_observable': 'forward SCATTERED field of linear incidence; not total transmitted Faraday rotation or planar Kerr reflection',
        'scope': 'homogeneous lossless host; finite nonoverlapping spheres; no substrate, infinite periodic boundary, magnetic-dipole particle modes or magnetization dynamics',
        'applicability_advisories': advisories,
        'accuracy': 'linear solver residual is reported; it is not a bound on model/multipole error or relative error of a vanishing CD signal',
        'threads_blas': threads, 'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()},
        'elapsed_seconds': perf_counter() - started, 'records': records,
    }
    return rows, metadata, geometry


def write_run(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new output directory')
    serialized = (json.dumps(config, indent=2, allow_nan=False) + '\n').encode('utf-8')
    rows, metadata, geometry = simulate(config)
    output.mkdir(parents=True)
    with (output / 'spectrum.csv').open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(output / 'geometry.npz', positions_nm=geometry.positions_nm, radii_nm=geometry.radii_nm)
    (output / 'config.json').write_bytes(serialized)
    metadata['configuration_sha256'] = hashlib.sha256(serialized).hexdigest()
    source_dir = Path(__file__).resolve().parent
    metadata['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob('*.py'))}
    (output / 'metadata.json').write_text(json.dumps(metadata, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return metadata


def _unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'), object_pairs_hook=_unique_keys, parse_constant=_reject_constant)
    metadata = write_run(config, args.out)
    print(json.dumps({'status': 'completed', 'particles': metadata['geometry']['particle_count'], 'seconds': metadata['elapsed_seconds'], 'output': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
