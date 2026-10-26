"""Periodic workflow regression checks: physics outputs and auditable files."""
import copy
import csv
import hashlib
import json

import numpy as np
import pytest
from numpy.testing import assert_allclose

from magnetooptics import periodic_runner as runner


def configuration():
    return {
        'schema_version': 'magnetooptics-periodic-1.0',
        'geometry': {'periods_nm': [84., 90.], 'positions_nm': [[13., 15., 0.], [50., 55., 0.]],
                     'radius_nm': [5., 7.], 'site_materials': ['mo', 'dielectric']},
        'materials': {'mo': {'epsilon_diagonal': [-2.5, .5], 'g': [.1, .01], 'magnetization': [0, 0, 1]},
                      'dielectric': {'epsilon_diagonal': [2.2, .1], 'g': 0}},
        'wavelength_nm': [450., 550.], 'host_index': 1.2,
        'solver': {'backend': 'cached', 'ewald_tolerance': 1e-11, 'rtol': 1e-11},
        'threads': 1,
    }


def test_reference_and_accelerated_spectra_and_ports_match():
    config = configuration()
    untouched = copy.deepcopy(config)
    fast, fast_metadata, _, fast_orders = runner.simulate(config)
    assert config == untouched
    config['solver']['backend'] = 'reference'
    reference, metadata, _, reference_orders = runner.simulate(config)
    for actual, expected in zip(fast, reference):
        assert actual.keys() == expected.keys()
        for key in actual:
            if expected[key] is None:
                assert actual[key] is None
            else:
                assert_allclose(actual[key], expected[key], atol=2e-9, rtol=2e-9, err_msg=key)
    for actual, expected in zip(fast_orders, reference_orders):
        for key in actual:
            if isinstance(expected[key], str):
                assert actual[key] == expected[key]
            else:
                assert_allclose(actual[key], expected[key], atol=1e-12, rtol=1e-10)
    assert fast_metadata['max_energy_balance_residual'] < 1e-10
    for record in metadata['records']:
        assert len(record['solves']) == 4
        for label in ('Mplus', 'Mminus'):
            assert not record['solves'][f'{label}_x']['factorization_reused']
            assert record['solves'][f'{label}_y']['factorization_reused']
            assert np.asarray(record['jones'][label]['transmission']).shape == (2, 2, 2)


def test_zero_contrast_is_transparent_and_has_no_kerr_angle():
    config = configuration()
    config['materials'] = {'invisible': {'epsilon_diagonal': config['host_index']**2, 'g': 0}}
    config['geometry'].pop('site_materials')
    config['geometry']['material'] = 'invisible'
    rows, _, _, orders = runner.simulate(config)
    for row in rows:
        for sign in ('Mplus', 'Mminus'):
            assert row[f'kerr_rotation_deg_{sign}'] is None
            assert row[f'kerr_ellipticity_deg_{sign}'] is None
            assert row[f'faraday_rotation_deg_{sign}'] == 0
            for pol in ('x', 'y', 'plus', 'minus'):
                assert_allclose(row[f'total_transmission_{sign}_{pol}'], 1., atol=1e-15)
                assert row[f'total_reflection_{sign}_{pol}'] == 0
                assert row[f'absorption_{sign}_{pol}'] == 0
    assert all(entry['reflection'] == 0 for entry in orders)


def test_saved_results_have_actual_byte_hashes_and_all_order_power(tmp_path):
    config = configuration()
    config['geometry']['periods_nm'] = [650., 570.]
    config['host_index'] = 1.
    output = tmp_path / 'result'
    metadata = runner.write_run(config, output)
    assert metadata['configuration_sha256'] == hashlib.sha256((output / 'config.json').read_bytes()).hexdigest()
    assert json.loads((output / 'metadata.json').read_text(encoding='utf-8')) == metadata
    with (output / 'spectrum.csv').open(encoding='utf-8', newline='') as handle:
        spectra = list(csv.DictReader(handle))
    with (output / 'diffraction_orders.csv').open(encoding='utf-8', newline='') as handle:
        orders = list(csv.DictReader(handle))
    for row in spectra:
        selected = [entry for entry in orders if entry['wavelength_nm'] == row['wavelength_nm']
                    and entry['magnetization'] == 'Mplus' and entry['polarization'] == 'x']
        assert len(selected) > 1
        assert_allclose(sum(float(entry['transmission']) for entry in selected),
                        float(row['total_transmission_Mplus_x']), atol=1e-15)
    with pytest.raises(FileExistsError):
        runner.write_run(config, output)


@pytest.mark.parametrize('location,key,value', [
    ('', 'unexpected', 1), ('solver', 'backend', 'fft'), ('solver', 'rtol', True),
    ('solver', 'ewald_tolerance', 0), ('solver', 'max_memory_mib', 1.5),
    ('', 'host_index', '1'), ('', 'threads', False), ('', 'wavelength_nm', [550, 450]),
    ('geometry', 'positions_nm', [[0, 0, 1]]),
    ('geometry', 'site_materials', ['missing', 'mo']),
])
def test_invalid_input_is_rejected_before_output(tmp_path, location, key, value):
    config = configuration()
    target = config[location] if location else config
    target[key] = value
    output = tmp_path / 'rejected'
    with pytest.raises(ValueError):
        runner.write_run(config, output)
    assert not output.exists()


def test_oblique_and_grazing_incidence_rejected():
    config = configuration()
    config['incidence'] = {'direction': [.1, 0, 1]}
    with pytest.raises(ValueError, match='normal incidence'):
        runner.validate_config(config)
    config['incidence']['direction'] = [0, 0, 1]
    config['host_index'] = 1.
    config['geometry']['periods_nm'] = [450., 600.]
    with pytest.raises(ValueError, match='Rayleigh'):
        runner.validate_config(config)


def test_unsampled_active_material_knot_is_rejected():
    config = configuration()
    config['materials']['mo'] = {'model': 'gyrotropic_table',
        'wavelength_nm': [400, 500, 600], 'epsilon_diagonal': [[2, .2], [2, -.5], [2, .2]],
        'g': [0, 0, 0]}
    with pytest.raises(ValueError, match='active'):
        runner.validate_config(config)


def test_cli_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / 'bad.json'
    path.write_text('{"schema_version":1,"schema_version":2}', encoding='utf-8')
    with pytest.raises(ValueError, match='duplicate JSON key'):
        runner.main(['--config', str(path), '--out', str(tmp_path / 'output')])


def test_axis_odd_angle_respects_180_degree_equivalence():
    assert runner._odd_axis_angle(89, -89) == -1
    assert runner._odd_axis_angle(None, 0) is None
    assert runner._odd_axis_angle(45, -45) is None
