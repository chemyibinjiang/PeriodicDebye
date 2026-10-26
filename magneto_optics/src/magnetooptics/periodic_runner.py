"""Strict JSON workflow for normally illuminated periodic dipole metasurfaces."""
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
from .geometry import _real_array
from .materials import sphere_polarizability
from .periodic_geometry import RectangularCell
from .periodic_interaction import PeriodicGreenOperator
from .periodic_observables import diffraction_orders, evaluate_ports, jones_matrices, polarization_ellipse
from .runner import material_tensor, _unique_keys, _reject_constant
from .solver import CoupledDipoleSystem, ConvergenceError


def _mapping(value, name, allowed, required=()):
    if not isinstance(value, dict):
        raise ValueError(f'{name} must be a mapping')
    if unknown := set(value) - set(allowed):
        raise ValueError(f'unknown {name} fields: {sorted(unknown)}')
    if missing := set(required) - set(value):
        raise ValueError(f'missing {name} fields: {sorted(missing)}')
    return value


def _positive(value, name, *, integer=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a positive number')
    if not np.isfinite(value) or value <= 0 or (integer and not isinstance(value, (int, np.integer))):
        raise ValueError(f'{name} must be a positive {"integer" if integer else "number"}')
    return int(value) if integer else float(value)


def validate_config(config):
    _mapping(config, 'configuration', {'schema_version', 'description', 'geometry', 'materials',
             'wavelength_nm', 'host_index', 'incidence', 'solver', 'threads'},
             {'schema_version', 'geometry', 'materials', 'wavelength_nm'})
    if config['schema_version'] != 'magnetooptics-periodic-1.0':
        raise ValueError('schema_version must be magnetooptics-periodic-1.0')
    gspec = _mapping(config['geometry'], 'geometry',
                     {'periods_nm', 'positions_nm', 'radius_nm', 'material', 'site_materials'},
                     {'periods_nm', 'positions_nm', 'radius_nm'})
    cell = RectangularCell(gspec['periods_nm'], gspec['positions_nm'], gspec['radius_nm'])
    wavelengths = _real_array(config['wavelength_nm'], 'wavelength_nm')
    if wavelengths.ndim != 1 or not len(wavelengths) or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError('wavelength_nm must be a positive strictly increasing list')
    nh = _positive(config.get('host_index', 1.), 'host_index')
    materials = config['materials']
    if not isinstance(materials, dict) or not materials or any(not isinstance(name, str) for name in materials):
        raise ValueError('materials must be a nonempty named mapping')
    if 'material' in gspec and 'site_materials' in gspec:
        raise ValueError('specify material or site_materials, not both')
    names = gspec.get('site_materials', [gspec.get('material', next(iter(materials)))] * len(cell.positions_nm))
    if not isinstance(names, list) or len(names) != len(cell.positions_nm) or any(not isinstance(name, str) or name not in materials for name in names):
        raise ValueError('site_materials must name one existing material per particle')
    for spec in materials.values():
        if not isinstance(spec, dict):
            raise ValueError('each material must be a mapping')
        # Includes every table knot, not only the requested endpoints.
        for wavelength in [wavelengths[0], wavelengths[-1]]:
            material_tensor(spec, float(wavelength))
    incidence = _mapping(config.get('incidence', {}), 'incidence', {'direction'})
    direction = _real_array(incidence.get('direction', [0., 0., 1.]), 'direction')
    if direction.shape != (3,) or not np.array_equal(direction, [0., 0., 1.]):
        raise ValueError('this periodic release supports normal incidence [0,0,1] only')
    settings = dict(_mapping(config.get('solver', {}), 'solver',
                    {'backend', 'rtol', 'ewald_tolerance', 'max_memory_mib'}))
    settings.setdefault('backend', 'cached')
    if settings['backend'] not in ('cached', 'vectorized', 'reference'):
        raise ValueError('backend must be cached, vectorized or reference')
    for name, default in [('rtol', 1e-10), ('ewald_tolerance', 1e-10)]:
        settings[name] = _positive(settings.get(name, default), name)
        if not 1e-14 <= settings[name] <= 1e-3:
            raise ValueError(f'{name} must be between 1e-14 and 1e-3')
    settings['max_memory_mib'] = _positive(settings.get('max_memory_mib', 1024), 'max_memory_mib', integer=True)
    threads = _positive(config.get('threads', 1), 'threads', integer=True)
    # Reject grazing inputs before allocating or solving any wavelength.
    for wavelength in wavelengths:
        diffraction_orders(cell.periods_nm, float(wavelength), nh)
    return cell, wavelengths, nh, names, settings, threads


def _json_value(value):
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        if np.iscomplexobj(value):
            return np.stack((value.real, value.imag), axis=-1).tolist()
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _odd_axis_angle(positive, negative):
    if positive is None or negative is None:
        return None
    # Ellipse axes are modulo 180 degrees. The odd angle is modulo 90 degrees.
    difference = (positive - negative + 90.) % 180. - 90.
    if abs(abs(difference) - 90.) < 1e-12:
        return None  # Equally short branches: there is no unique signed half difference.
    return float(difference / 2.)


def simulate(config):
    cell, wavelengths, nh, names, settings, threads = validate_config(config)
    started = perf_counter()
    rows, records, order_rows = [], [], []
    max_internal_size = 0.
    max_balance = 0.
    with threadpool_limits(limits=threads):
        for wavelength in wavelengths:
            wavelength = float(wavelength)
            k = 2 * np.pi * nh / wavelength
            operator = PeriodicGreenOperator(cell.positions_nm, cell.periods_nm, k,
                tolerance=settings['ewald_tolerance'], backend=settings['backend'],
                max_memory_bytes=settings['max_memory_mib'] * 1024**2)
            orders = diffraction_orders(cell.periods_nm, wavelength, nh)
            zero = int(orders['zero_index'])
            row = {'wavelength_nm': wavelength, 'propagating_orders_per_side': len(orders['indices'])}
            record = {'wavelength_nm': wavelength, 'operator': dict(operator.metadata), 'solves': {}, 'jones': {}}
            residuals = []
            for sign, label in [(1, 'Mplus'), (-1, 'Mminus')]:
                tensors = {name: material_tensor(spec, wavelength, sign, _validated=True)
                           for name, spec in config['materials'].items()}
                epsilon = np.array([tensors[name] for name in names])
                size = 2 * np.pi / wavelength * cell.radii_nm * np.sqrt(np.linalg.norm(epsilon, ord=2, axis=(-2, -1)))
                max_internal_size = max(max_internal_size, float(size.max()))
                alpha = sphere_polarizability(epsilon, cell.radii_nm, wavelength, nh)
                system = CoupledDipoleSystem(operator, alpha, method='dense')
                solutions, linear_ports = [], []
                for axis, axis_label in [(0, 'x'), (1, 'y')]:
                    incident = np.broadcast_to(np.eye(3)[axis], (len(cell.positions_nm), 3)).copy()
                    result = system.solve(incident, rtol=settings['rtol'])
                    solutions.append(result)
                    record['solves'][f'{label}_{axis_label}'] = dict(result.metadata)
                    residuals.append(result.relative_residual)
                    linear_ports.append(evaluate_ports(cell, result.dipoles, incident, result.local_field,
                                                      wavelength, nh, orders=orders))
                jones = jones_matrices(*linear_ports)
                record['jones'][label] = jones
                for observable, matrix_name in [('faraday', 'transmission'), ('kerr', 'reflection')]:
                    ellipse = polarization_ellipse(jones[matrix_name][:, 0])
                    for quantity, value in ellipse.items():
                        row[f'{observable}_{quantity}_{label}'] = value
                ports = {'x': linear_ports[0], 'y': linear_ports[1]}
                for name, helicity in [('plus', 1), ('minus', -1)]:
                    incident = np.array([1., helicity * 1j, 0.]) / np.sqrt(2.)
                    dipoles = (solutions[0].dipoles + helicity * 1j * solutions[1].dipoles) / np.sqrt(2.)
                    local = (solutions[0].local_field + helicity * 1j * solutions[1].local_field) / np.sqrt(2.)
                    ports[name] = evaluate_ports(cell, dipoles, incident, local, wavelength, nh, orders=orders)
                for polarization, output in ports.items():
                    for quantity in ('total_transmission', 'total_reflection', 'absorption', 'energy_balance_residual'):
                        row[f'{quantity}_{label}_{polarization}'] = float(output[quantity])
                    row[f'zero_transmission_{label}_{polarization}'] = float(output['transmitted_power'][zero])
                    row[f'zero_reflection_{label}_{polarization}'] = float(output['reflected_power'][zero])
                    max_balance = max(max_balance, abs(float(output['energy_balance_residual'])))
                    for n, (m1, m2) in enumerate(output['order_indices']):
                        entry = {'wavelength_nm': wavelength, 'magnetization': label, 'polarization': polarization,
                                 'm': int(m1), 'n': int(m2), 'transmission': float(output['transmitted_power'][n]),
                                 'reflection': float(output['reflected_power'][n])}
                        for side in ('transmitted', 'reflected'):
                            for axis, field in zip('xyz', output[f'{side}_fields'][n]):
                                entry[f'{side}_E{axis}_real'] = float(field.real)
                                entry[f'{side}_E{axis}_imag'] = float(field.imag)
                        order_rows.append(entry)
            for name in ('faraday', 'kerr'):
                row[f'{name}_magnetic_odd_rotation_deg'] = _odd_axis_angle(
                    row[f'{name}_rotation_deg_Mplus'], row[f'{name}_rotation_deg_Mminus'])
            for side in ('transmission', 'reflection'):
                positive = row[f'total_{side}_Mplus_plus'] - row[f'total_{side}_Mplus_minus']
                negative = row[f'total_{side}_Mminus_plus'] - row[f'total_{side}_Mminus_minus']
                row[f'{side}_magnetic_odd_cd'] = float((positive - negative) / 2)
            row['max_relative_residual'] = max(residuals)
            rows.append(row)
            records.append(record)
    if not np.isfinite(max_balance) or max_balance > max(1e-7, 100 * settings['ewald_tolerance']):
        raise ConvergenceError(f'Periodic energy closure failed: maximum |T+R+A-1|={max_balance:.6g}')
    audit = cell.audit()
    audit['maximum_host_size_parameter_ka'] = float(2 * np.pi * nh * cell.radii_nm.max() / wavelengths.min())
    audit['maximum_internal_size_parameter'] = max_internal_size
    advisories = []
    if audit['maximum_host_size_parameter_ka'] > 0.3 or max_internal_size > 0.5:
        advisories.append('Particle size screening exceeded: omitted multipoles require independent converged Maxwell validation.')
    if audit.get('minimum_center_distance_nm') and cell.radii_nm.max() / audit['minimum_center_distance_nm'] > 0.2:
        advisories.append('Radius/minimum periodic centre distance exceeds 0.2; omitted near-field multipoles may matter.')
    metadata = {
        'package_version': __version__, 'model': 'infinite 2D rectangular coplanar electric-dipole metasurface',
        'time_convention': 'exp(-i omega t)', 'length_unit': 'nm', 'polarizability_unit': 'nm^3',
        'power_unit': 'fraction of incident normal energy flux through one unit cell', 'host_index': nh,
        'geometry': audit, 'scope': 'normal incidence +z; identical real lossless host on both sides; one electric dipole per separated sphere; no substrate, oblique incidence, multilayers or higher multipoles',
        'jones_convention': 'columns incident unit x,y; rows outgoing fixed laboratory x,y on BOTH sides; complex numbers saved [real,imag]; amplitudes referred to z=0',
        'polarization_convention': 'ellipse S3=2 Im(conj(Ex) Ey); angles in fixed laboratory xy, including reflected light; angles modulo180, magnetic odd axis angle modulo90; null if undefined',
        'faraday_kerr_definition': 'raw zero-order TOTAL transmitted / reflected ellipse for x incidence, including possible nonmagnetic geometry rotation; use magnetization-odd outputs for reversal-odd response; reflected lab basis is not right handed with -z',
        'absorption_definition': 'independent local material work minus dipole self-radiation, divided by incident cell flux; NOT inferred from 1-T-R',
        'max_energy_balance_residual': max_balance,
        'accuracy': 'Ewald tolerance and true linear residual are numerical controls, not physical dipole-model accuracy guarantees; weak/zero signals require absolute error metrics',
        'material_provenance': {name: spec.get('provenance', 'user supplied, not independently validated') for name, spec in config['materials'].items()},
        'applicability_advisories': advisories, 'threads_blas': threads,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform()},
        'elapsed_seconds': perf_counter() - started, 'records': records,
    }
    return rows, _json_value(metadata), cell, order_rows


def write_run(config, output):
    output = Path(output)
    if output.exists():
        raise FileExistsError('use a new output directory')
    config_bytes = (json.dumps(config, indent=2, allow_nan=False) + '\n').encode('utf-8')
    rows, metadata, cell, order_rows = simulate(config)
    metadata['configuration_sha256'] = hashlib.sha256(config_bytes).hexdigest()
    source_dir = Path(__file__).resolve().parent
    metadata['source_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source_dir.glob('*.py'))}
    # Serialize first: invalid or nonfinite results must not leave a claimed run.
    metadata_bytes = (json.dumps(metadata, indent=2, allow_nan=False) + '\n').encode('utf-8')
    json.dumps([rows, order_rows], allow_nan=False)
    output.mkdir(parents=True)
    for filename, values in [('spectrum.csv', rows), ('diffraction_orders.csv', order_rows)]:
        with (output / filename).open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=list(values[0]))
            writer.writeheader()
            writer.writerows(values)
    np.savez_compressed(output / 'geometry.npz', positions_nm=cell.positions_nm,
                        radii_nm=cell.radii_nm, periods_nm=cell.periods_nm)
    (output / 'config.json').write_bytes(config_bytes)
    (output / 'metadata.json').write_bytes(metadata_bytes)
    return metadata


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    config = json.loads(args.config.read_text(encoding='utf-8'), object_pairs_hook=_unique_keys,
                        parse_constant=_reject_constant)
    metadata = write_run(config, args.out)
    print(json.dumps({'status': 'completed', 'geometry': metadata['geometry'],
                      'seconds': metadata['elapsed_seconds'],
                      'max_energy_balance_residual': metadata['max_energy_balance_residual'],
                      'output': str(args.out)}, indent=2))


if __name__ == '__main__':
    main()
