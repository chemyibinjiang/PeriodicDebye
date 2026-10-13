"""Runner-level correctness and bounded representation of independent pores."""
import copy
import json

import numpy as np
import pytest

from mofdebye import runner
from mofdebye.core import AtomCloud, direct_iq


@pytest.fixture
def pore_config(tmp_path):
    (tmp_path/'host.xyz').write_text('1\nhost\nC 0 0 0\n', encoding='utf-8')
    (tmp_path/'guest.xyz').write_text('1\nguest\nAu 0 0 0\n', encoding='utf-8')
    return dict(schema_version='1.1', scattering_case='pxrd', boundary='finite',
                guest_estimator='analytic', backend='radial_moments', device='cpu',
                host=dict(xyz='host.xyz', cell_vectors_A=(np.eye(3)*8).tolist(), cell_mask=[[[1]]]),
                particles=[dict(xyz='guest.xyz', weight=1)], pores=dict(centres_A=[[2., 0., 0.]]),
                population=dict(occupancy=.3, displacement_sigma_A=0., pore_radius_A=1.),
                sampling=dict(q=dict(min=0., max=6., points=121)),
                radial=dict(bin_width_A=.04, order=8, threads=1),
                statistics=dict(n_samples=32, n_batches=4, seed=678))


def test_one_cell_bernoulli_ensemble_and_exported_uncertainty(pore_config, tmp_path):
    pytest.importorskip('numba')
    result = runner.run_configuration(pore_config, tmp_path)
    table = result['tables']['intensity']
    q = table['q']
    empty = direct_iq(AtomCloud(('C',), [[0, 0, 0]]), q)
    occupied = direct_iq(AtomCloud(('C', 'Au'), [[0, 0, 0], [2, 0, 0]]), q)
    reference = .7*empty+.3*occupied
    np.testing.assert_allclose(table['total'], reference, rtol=3e-10, atol=1e-8)
    assert set(table) == {'q', 'host', 'guest_internal', 'host_guest', 'guest_guest', 'total'}
    np.testing.assert_allclose(table['total'], sum(table[name] for name in
                                                ('host', 'guest_internal', 'host_guest', 'guest_guest')))
    np.testing.assert_allclose(table['guest_guest'], 0, atol=1e-11)
    assert np.min(table['host_guest']) < 0
    uncertainty = result['tables']['numerical_error']
    assert set(uncertainty) == {'q', 'approximation_bound_excluding_roundoff', 'total_mc_sem'}
    np.testing.assert_array_equal(uncertainty['q'], q)
    np.testing.assert_array_equal(uncertainty['total_mc_sem'], np.zeros(len(q)))
    assert np.all(uncertainty['approximation_bound_excluding_roundoff'] >= 0)
    assert result['metadata']['moment_metadata']['method'] == 'analytic'
    assert result['metadata']['output_statistic'] == 'ensemble_mean'
    destination = tmp_path/'export'
    runner.write_result(result, destination)
    saved = np.loadtxt(destination/'intensity.csv', delimiter=',', skiprows=1)
    assert saved.shape == (len(q), 6)
    saved_error = np.loadtxt(destination/'numerical_error.csv', delimiter=',', skiprows=1)
    np.testing.assert_allclose(saved_error[:, -1], 0)
    metadata = json.loads((destination/'metadata.json').read_text(encoding='utf-8'))
    assert 'Monte Carlo' in metadata['uncertainty']
    assert 'roundoff' in metadata['uncertainty']
    assert metadata['actual_method'][0]['method'] == 'finite_full_cell_independent_pore_radial_moments'


def test_sampled_wall_outputs_integration_sem_and_exact_forward(pore_config, tmp_path):
    pytest.importorskip('numba')
    config = copy.deepcopy(pore_config)
    config['guest_estimator'] = 'moments'
    config['population'].update(displacement_sigma_A=.4, pore_radius_A=.5)
    config['statistics']['moment_method'] = 'sampled'
    result = runner.run_configuration(config, tmp_path)
    error = result['tables']['numerical_error']
    assert np.all(error['total_mc_sem'] >= 0)
    assert np.max(error['total_mc_sem'][1:]) > 0
    assert abs(error['total_mc_sem'][0]) < 1e-10
    metadata = result['metadata']['moment_metadata']
    assert metadata['method'] == 'sampled'
    assert metadata['batches'] == 4
    assert metadata['accepted_placements'] == 2*32
    assert metadata['accepted_samples_per_template_pore_replica_batch'] == 8
    assert metadata['spherical_wall'] == 'conditioned_by_rejection'
    assert 'physical pore disorder' in result['metadata']['uncertainty']


def test_40_cube_input_keeps_host_and_pore_basis_implicit(pore_config, tmp_path, monkeypatch):
    config = copy.deepcopy(pore_config)
    config['host']['cell_mask'] = np.ones((40, 40, 40), dtype=int).tolist()
    def disallow(*args, **kwargs):
        raise AssertionError('No expanded full-crystal atom cloud may be created.')
    monkeypatch.setattr(runner, 'concatenate', disallow)
    runner.validate_config(config)
    host, basis, cell, mask, population, hashes, processing = runner._inputs(config, tmp_path)
    assert host is basis and len(host) == 1
    assert len(population.centres) == 1
    assert population.occupancy.shape == (1,)
    assert processing['host_atoms'] == 40**3
    assert processing['retained_pore_sites'] == 40**3
    assert processing['host_atoms_materialized'] is False
    assert processing['pore_sites_materialized'] is False
    assert processing['expected_guest_atoms'] == .3*40**3
    assert len(processing['pore_centres_A']) == 1


@pytest.mark.parametrize('schema', ['1.0', '1.1'])
def test_duplicate_periodic_pores_rejected_for_both_schemas(pore_config, tmp_path, schema):
    config = copy.deepcopy(pore_config)
    config['schema_version'] = schema
    config['pores']['centres_A'] = [[2, 0, 0], [10, 0, 0]]
    with pytest.raises(ValueError, match='Duplicate periodic'):
        runner.run_configuration(config, tmp_path)


@pytest.mark.parametrize('value', [1, 0, -1, True, 2.5])
def test_invalid_batch_counts_rejected_at_configuration_stage(pore_config, value):
    config = copy.deepcopy(pore_config)
    config['statistics']['n_batches'] = value
    with pytest.raises(ValueError, match='n_batches'):
        runner.validate_config(config)


@pytest.mark.parametrize('change', ['explicit', 'clearance', 'external', 'atom_cut'])
def test_unsupported_physics_rejected_before_geometry(pore_config, change):
    config = copy.deepcopy(pore_config)
    if change == 'explicit':
        config['guest_estimator'] = 'explicit'
    elif change == 'clearance':
        config['population']['min_host_distance_A'] = .2
    elif change == 'external':
        config['host']['guest_retention'] = 'whole_particle'
    else:
        config['host']['atom_cut'] = dict(kind='sphere', centre_A=[0, 0, 0], radius_A=2)
    with pytest.raises(ValueError):
        runner.validate_config(config)


@pytest.mark.parametrize('change', ['orientation', 'anisotropy', 'active_wall', 'bad_sample_division'])
def test_invalid_local_statistics_fail_without_fast_geometry(pore_config, tmp_path, change, monkeypatch):
    from mofdebye import pore_powder
    def disallow(*args, **kwargs):
        raise AssertionError('Unsupported local physics should fail before constructing geometry.')
    monkeypatch.setattr(pore_powder, 'PorePowderCache', disallow)
    config = copy.deepcopy(pore_config)
    if change == 'orientation':
        config['population']['orientation'] = 'fixed'
    elif change == 'anisotropy':
        config['population']['displacement_covariance_A2'] = np.diag([.1, .2, .3]).tolist()
    elif change == 'active_wall':
        config['population']['displacement_sigma_A'] = .1
    else:
        config['guest_estimator'] = 'moments'
        config['statistics'].update(moment_method='sampled', n_samples=31)
    with pytest.raises(ValueError):
        runner.run_configuration(config, tmp_path)


def test_periodic_wall_overlap_rejected_before_cache(pore_config, tmp_path):
    config = copy.deepcopy(pore_config)
    config['population']['pore_radius_A'] = 4.1
    with pytest.raises(ValueError, match='overlap|Overlap'):
        runner.run_configuration(config, tmp_path)
