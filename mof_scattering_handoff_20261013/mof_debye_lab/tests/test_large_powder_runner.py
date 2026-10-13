"""Runner contracts for the opt-in approximate finite host PXRD backend."""
import json

import numpy as np
import pytest

from mofdebye import runner
from mofdebye.core import AtomCloud, concatenate, direct_iq


@pytest.fixture
def config(tmp_path):
    (tmp_path/'host.xyz').write_text('3\nheterogeneous basis\nC 0.1 0.3 0.2\nO 1.2 0.8 1.4\nZn 2.1 1.1 0.9\n', encoding='utf-8')
    return dict(schema_version='1.1', scattering_case='pxrd', boundary='finite',
                guest_estimator='explicit', backend='radial_moments', device='cpu',
                host=dict(xyz='host.xyz', cell_vectors_A=(np.eye(3)*4).tolist(),
                          cell_mask=np.ones((2, 2, 2), dtype=int).tolist()),
                sampling=dict(q=dict(min=0., max=6., points=121)),
                radial=dict(bin_width_A=.04, order=8, threads=1, merge_decimals=12))


def test_runner_matches_explicit_pairs_and_saves_error_provenance(config, tmp_path):
    pytest.importorskip('numba', reason='The radial moment backend uses the optional fast extra.')
    result = runner.run_configuration(config, tmp_path)
    table = result['tables']['intensity']
    basis = AtomCloud.read_xyz(tmp_path/'host.xyz')
    cloud = concatenate([basis.moved(np.asarray(index)*4) for index in np.ndindex((2, 2, 2))])
    expected = direct_iq(cloud, table['q'])
    np.testing.assert_allclose(table['total'], expected, rtol=2e-9, atol=1e-7)
    metadata = result['metadata']
    assert metadata['backend_requested'] == 'radial_moments'
    assert metadata['device_actual'] == 'cpu'
    assert metadata['input_processing']['host_atoms'] == 24
    assert metadata['input_processing']['host_atoms_materialized'] is False
    assert metadata['actual_method']
    error_table = result['tables']['numerical_error']
    np.testing.assert_array_equal(error_table['q'], table['q'])
    bounds = [value for key, value in error_table.items() if key != 'q']
    assert len(bounds) == 1
    bound = np.asarray(bounds[0])
    assert bound.shape == table['q'].shape
    assert np.isfinite(bound).all() and np.all(bound >= 0)
    output = tmp_path/'output'
    runner.write_result(result, output)
    saved = np.loadtxt(output/'numerical_error.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(saved[:, 1], bound)
    saved_metadata = json.loads((output/'metadata.json').read_text(encoding='utf-8'))
    assert saved_metadata['actual_method'] == metadata['actual_method']


def test_input_preparation_keeps_100nm_cells_implicit(config, tmp_path, monkeypatch):
    config['host']['cell_mask'] = np.ones((40, 40, 40), dtype=int).tolist()
    def forbid_materialization(*args, **kwargs):
        raise AssertionError('The finite crystal must remain an implicit repeated basis.')
    monkeypatch.setattr(runner, 'concatenate', forbid_materialization)
    runner.validate_config(config)
    host, basis, cell, mask, population, hashes, processing = runner._inputs(config, tmp_path)
    assert host is basis and len(host) == 3
    assert processing['host_atoms'] == 3*40**3
    assert processing['retained_cells'] == 40**3
    assert processing['host_atoms_materialized'] is False
    assert population is None


@pytest.mark.parametrize('case,boundary', [
    ('pxrd', 'bulk'), ('single_crystal_finite', 'finite'), ('single_crystal_periodic', 'bulk')])
def test_rejects_nonfinite_powder_modes(config, case, boundary):
    config.update(scattering_case=case, boundary=boundary)
    with pytest.raises(ValueError):
        runner.validate_config(config)


@pytest.mark.parametrize('field,value', [
    ('particles', [dict(xyz='guest.xyz', weight=1.)]),
    ('pores', dict(centres_A=[[1., 1., 1.]])),
    ('population', dict(occupancy=.4)),
])
@pytest.mark.parametrize('estimator', ['explicit', 'analytic', 'moments'])
def test_rejects_guest_inputs_for_all_estimators(config, field, value, estimator):
    config[field] = value
    config['guest_estimator'] = estimator
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_rejects_atom_cut_and_cuda(config):
    config['host']['atom_cut'] = dict(kind='sphere', centre_A=[0., 0., 0.], radius_A=5.)
    with pytest.raises(ValueError):
        runner.validate_config(config)
    del config['host']['atom_cut']
    config['device'] = 'cuda'
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_rejects_host_without_repeat_geometry(config):
    del config['host']['cell_vectors_A']
    del config['host']['cell_mask']
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_radial_options_cannot_silently_affect_other_backends(config):
    config['backend'] = 'reference'
    with pytest.raises(ValueError):
        runner.validate_config(config)


def test_unknown_radial_setting_is_rejected(config):
    config['radial']['typo'] = True
    with pytest.raises(ValueError):
        runner.validate_config(config)


@pytest.mark.parametrize('q', [[0.], [.1, .2, .3], [0., .1, .25], [0., 0., 0.], [0., .1, .05]])
def test_rejects_unsupported_q_grids(config, tmp_path, q):
    config['sampling']['q'] = q
    with pytest.raises(ValueError):
        runner.run_configuration(config, tmp_path)


@pytest.mark.parametrize('setting,value', [
    ('bin_width_A', 0.), ('bin_width_A', float('nan')), ('order', -1),
    ('threads', 0), ('threads', True), ('merge_decimals', 2.5)])
def test_invalid_radial_controls_fail_before_calculation(config, tmp_path, setting, value):
    config['radial'][setting] = value
    with pytest.raises((ValueError, TypeError)):
        runner.run_configuration(config, tmp_path)
