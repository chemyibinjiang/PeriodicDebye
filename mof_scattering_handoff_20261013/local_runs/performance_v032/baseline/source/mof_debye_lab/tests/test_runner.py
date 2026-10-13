from itertools import product
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors
from mofdebye.directional import amplitude, intensity
from mofdebye.moments import GuestMoments
from mofdebye.periodic import reciprocal_lattice
from mofdebye.runner import run_configuration, validate_config, write_result


@pytest.fixture
def fixture_files(tmp_path):
    host = AtomCloud(('C', 'O'), [[0.1, 0.2, 0.3], [0.7, -0.2, 0.1]])
    particle = AtomCloud(('Au',), [[0, 0, 0]])
    host.write_xyz(tmp_path / 'host.xyz')
    particle.write_xyz(tmp_path / 'guest.xyz')
    return tmp_path, host, particle


def configuration(case='pxrd', boundary='finite', estimator='analytic', guests=True):
    result = dict(schema_version='1.0', scattering_case=case, boundary=boundary,
                  guest_estimator=estimator, backend='reference',
                  host={'xyz': 'host.xyz'}, statistics={'n_samples': 32, 'seed': 218},
                  angular={'n_mu': 12, 'n_phi': 24})
    if case == 'pxrd':
        result['sampling'] = {'q': [0, 0.4, 1.0]}
    elif case == 'single_crystal_finite':
        result['sampling'] = {'q_vectors': [[0, 0, 0], [0.3, -0.5, 0.1], [1.1, 0.2, -0.4]]}
    else:
        result['sampling'] = {'reflection_q_max': 2.3,
                              'q_vectors': [[0, 0, 0], [0.2, 0.3, -0.4]]}
    if boundary == 'bulk':
        result['host']['cell_vectors_A'] = [[4, 0, 0], [0.4, 5, 0], [0.2, 0.3, 6]]
    if guests:
        result['particles'] = [{'xyz': 'guest.xyz', 'weight': 1}]
        result['pores'] = {'centres_A': [[1.2, 0.3, 0.7]]}
        result['population'] = {'occupancy': 0.35}
    return result


@pytest.mark.parametrize('case', ['pxrd', 'single_crystal_finite'])
def test_finite_runner_preserves_host_guest_interference_and_occupancy(fixture_files, case):
    root, host, particle = fixture_files
    config = configuration(case)
    result = run_configuration(config, root)
    p = config['population']['occupancy']
    full = concatenate([host, particle.moved(config['pores']['centres_A'][0])])
    if case == 'pxrd':
        q = config['sampling']['q']
        expected = (1 - p) * direct_iq(host, q) + p * direct_iq(full, q)
    else:
        vectors = config['sampling']['q_vectors']
        expected = (1 - p) * intensity(host, vectors) + p * intensity(full, vectors)
        assert 'mean_amplitude_sem' in result['tables']['intensity']
    np.testing.assert_allclose(result['tables']['intensity']['total'], expected, rtol=1e-12)
    metadata = result['metadata']
    assert metadata['scattering_case'] == case
    assert metadata['boundary'] == 'finite'
    assert metadata['output_statistic'] == 'ensemble_mean'
    assert metadata['device_actual'] == 'cpu'
    assert len(metadata['input_sha256']) == 2
    assert all(len(digest) == 64 for digest in metadata['input_sha256'].values())
    json.dumps(metadata, allow_nan=False)


@pytest.mark.parametrize('case', ['pxrd', 'single_crystal_finite'])
def test_explicit_analytic_and_sampled_moments_agree_for_deterministic_population(fixture_files, case):
    root, _, _ = fixture_files
    outputs = []
    for estimator in ('explicit', 'analytic', 'moments'):
        config = configuration(case, estimator=estimator)
        config['population']['occupancy'] = 1.0
        result = run_configuration(config, root)
        outputs.append(result['tables']['intensity']['total'])
        assert result['metadata']['output_statistic'] == ('realization' if estimator == 'explicit' else 'ensemble_mean')
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=1e-12)
    np.testing.assert_allclose(outputs[0], outputs[2], rtol=1e-11)


def test_periodic_runner_reflection_measure_and_diffuse_are_separate(fixture_files):
    root, host, particle = fixture_files
    config = configuration('single_crystal_periodic', 'bulk')
    result = run_configuration(config, root)
    table = result['tables']['reflections']
    vectors = np.column_stack([table[k] for k in ('qx', 'qy', 'qz')])
    p = config['population']['occupancy']
    g = amplitude(particle.moved(config['pores']['centres_A'][0]), vectors)
    expected_coherent = np.abs(amplitude(host, vectors) + p * g)**2
    expected_diffuse = p * (1 - p) * np.abs(g)**2
    np.testing.assert_allclose(table['f_squared'], expected_coherent, rtol=1e-12)
    np.testing.assert_allclose(table['diffuse_at_reflections'], expected_diffuse, rtol=1e-12)
    volume = abs(np.linalg.det(config['host']['cell_vectors_A']))
    np.testing.assert_allclose(table['bragg_weight'], (2 * np.pi)**3 / volume * expected_coherent)
    assert result['metadata']['forward_weight'] > 0
    assert 'peak height' in result['metadata']['bragg_convention']
    assert result['tables']['diffuse']['diffuse'].shape == (2,)


def test_bulk_powder_runner_shell_measure_and_forward_separation(fixture_files):
    root, _, _ = fixture_files
    config = configuration('pxrd', 'bulk')
    config['sampling']['q'] = [0, 1, 2.3]
    result = run_configuration(config, root)
    reflections = result['tables']['reflections']
    shells = result['tables']['powder_shells']
    assert np.all(shells['q'] > 0)
    assert shells['multiplicity'].sum() == len(reflections['q'])
    np.testing.assert_allclose(shells['radial_weight'].sum(),
                               np.sum(reflections['bragg_weight'] / (4 * np.pi * reflections['q']**2)))
    p = config['population']['occupancy']
    q = np.asarray(config['sampling']['q'])
    np.testing.assert_allclose(result['tables']['diffuse']['diffuse'],
                               p * (1 - p) * form_factors(('Au',), q)['Au']**2, rtol=1e-12)
    assert 'Forward weight separate' in result['metadata']['powder_convention']


def test_hkl_and_fractional_pores_use_row_vectors_in_triclinic_cell(fixture_files):
    root, host, particle = fixture_files
    cell = np.array([[4, 0.2, 0], [1.1, 3.8, 0.3], [0.2, 0.7, 5.1]])
    hkl = np.array([[0, 0, 0], [1, 0, 0], [-1, 1, 2]])
    config = configuration('single_crystal_finite', estimator='explicit')
    config['host']['cell_vectors_A'] = cell.tolist()
    config['sampling'] = {'hkl': hkl.tolist()}
    config['pores'] = {'centres_fractional': [[0.2, 0.3, 0.4]]}
    config['population']['occupancy'] = 1
    result = run_configuration(config, root)
    vectors = hkl @ reciprocal_lattice(cell)
    table = result['tables']['intensity']
    np.testing.assert_allclose(np.column_stack([table[k] for k in ('qx', 'qy', 'qz')]), vectors)
    full = concatenate([host, particle.moved(np.array([0.2, 0.3, 0.4]) @ cell)])
    np.testing.assert_allclose(table['total'], intensity(full, vectors), rtol=1e-12)


def test_full_cell_mask_repeats_pores_offsets_and_independent_occupancies(fixture_files):
    root, host, particle = fixture_files
    config = configuration('single_crystal_finite')
    cell = np.diag([4, 5, 6])
    config['host'].update(cell_vectors_A=cell.tolist(), cell_mask=[[[1]], [[0]], [[1]]])
    centres = np.array([[0.2, 0.3, 0.4], [1.2, -0.1, 0.2]])
    offsets = np.array([[0.1, 0, 0.2], [-0.1, 0.3, 0]])
    config['pores']['centres_A'] = centres.tolist()
    config['population'].update(occupancy=[0.2, 0.7], mean_displacement_A=offsets.tolist())
    result = run_configuration(config, root)
    vectors = config['sampling']['q_vectors']
    shifts = np.array([[0, 0, 0], [8, 0, 0]])
    finite_host = concatenate([host.moved(shift) for shift in shifts])
    guest_centres = np.concatenate([centres + offsets + shift for shift in shifts])
    probabilities = np.tile([0.2, 0.7], 2)
    expected = np.zeros(3)
    for flags in product((0, 1), repeat=4):
        probability = np.prod([p if flag else 1 - p for p, flag in zip(probabilities, flags)])
        clouds = [finite_host] + [particle.moved(centre) for centre, flag in zip(guest_centres, flags) if flag]
        expected += probability * intensity(concatenate(clouds), vectors)
    np.testing.assert_allclose(result['tables']['intensity']['total'], expected, rtol=1e-12)
    assert result['metadata']['surface_policy'] == 'complete cells'


@pytest.mark.parametrize('case', ['pxrd', 'single_crystal_finite'])
def test_host_only_lattice_backend_agrees_with_expanded_reference(fixture_files, case):
    root, _, _ = fixture_files
    config = configuration(case, estimator='explicit', guests=False)
    config['host'].update(cell_vectors_A=[[4, 0, 0], [0.6, 5, 0], [0.2, 0.3, 6]],
                          cell_mask=[[[1], [0]], [[1], [1]]])
    reference = run_configuration(config, root)
    config['backend'] = 'lattice_grouped'
    grouped = run_configuration(config, root)
    np.testing.assert_allclose(grouped['tables']['intensity']['total'],
                               reference['tables']['intensity']['total'], rtol=1e-11, atol=1e-8)


@pytest.mark.parametrize('case', ['pxrd', 'single_crystal_finite'])
def test_host_only_lattice_backend_never_expands_atoms(fixture_files, monkeypatch, case):
    import mofdebye.runner as runner
    root, _, _ = fixture_files
    config = configuration(case, estimator='explicit', guests=False)
    config['backend'] = 'lattice_grouped'
    config['host'].update(cell_vectors_A=np.diag([4, 5, 6]).tolist(),
                          cell_mask=np.ones((4, 3, 2), dtype=int).tolist())
    def no_expansion(*args, **kwargs):
        raise AssertionError('Host-only lattice evaluation must not construct expanded atom clouds.')
    monkeypatch.setattr(runner, 'concatenate', no_expansion)
    result = run_configuration(config, root)
    assert np.isfinite(result['tables']['intensity']['total']).all()


def test_sampled_periodic_weights_use_unbiased_complex_variance_correction(fixture_files):
    root, host, particle = fixture_files
    config = configuration('single_crystal_periodic', 'bulk', estimator='moments')
    result = run_configuration(config, root)
    table = result['tables']['reflections']
    vectors = np.column_stack([table[k] for k in ('qx', 'qy', 'qz')])
    metadata = result['metadata']['moment_metadata']
    count = config['statistics']['n_samples']
    fraction = metadata['occupancy_counts'][0] / count
    guest_amplitude = amplitude(particle.moved(config['pores']['centres_A'][0]), vectors)
    variance = fraction * (1 - fraction) * count / (count - 1) * np.abs(guest_amplitude)**2
    coherent = np.abs(amplitude(host, vectors) + fraction * guest_amplitude)**2 - variance / count
    np.testing.assert_allclose(table['f_squared'], coherent, rtol=1e-12)
    np.testing.assert_allclose(table['diffuse_at_reflections'], variance, rtol=1e-12)
    np.testing.assert_allclose(table['mean_variance'], variance / count, rtol=1e-12)


def test_periodic_runner_rejects_pores_overlapping_across_cell_faces(fixture_files):
    root, _, _ = fixture_files
    config = configuration('single_crystal_periodic', 'bulk', estimator='moments')
    config['population']['pore_radius_A'] = 2.1
    with pytest.raises(ValueError, match='Overlapping periodic hard-wall pores'):
        run_configuration(config, root)


@pytest.mark.parametrize('change, message', [
    ({'schema_version': '0.1'}, 'schema_version'),
    ({'scattering_case': 'single_crystal_finite', 'boundary': 'bulk'}, 'incompatible'),
    ({'backend': 'installed_package'}, 'native factors'),
    ({'backend': 'adapted_kernel', 'scattering_case': 'single_crystal_finite'}, 'Scalar Debye'),
    ({'backend': 'adapted_kernel', 'guest_estimator': 'moments'}, 'reference backend'),
    ({'device': 'cuda'}, 'CPU'),
    ({'correlated_cells': True}, 'Unsupported configuration'),
    ({'statistics': {'n_samples': 1}}, 'n_samples'),
    ({'statistics': {'seed': -1}}, 'seed'),
    ({'statistics': {'moment_method': 'uncorrected'}}, 'moment_method'),
    ({'sampling': {'q_vectors': [[0, 0, 0]]}}, 'scalar'),
])
def test_config_rejects_incompatible_or_unsupported_requests(change, message):
    config = configuration()
    config.update(change)
    with pytest.raises(ValueError, match=message):
        validate_config(config)


def test_bulk_explicit_random_cell_and_ambiguous_directional_inputs_rejected():
    bulk = configuration('single_crystal_periodic', 'bulk', estimator='explicit')
    with pytest.raises(ValueError, match='repeated explicit random cell'):
        validate_config(bulk)
    config = configuration('single_crystal_finite')
    config['sampling']['hkl'] = [[0, 0, 0]]
    with pytest.raises(ValueError, match='exactly q_vectors or hkl'):
        validate_config(config)
    config['sampling'] = {'q': [0, 1]}
    with pytest.raises(ValueError, match='never scalar q'):
        validate_config(config)


@pytest.mark.parametrize('field,value', [('correlated_occupancy', True), ('host_relaxation', True),
                                         ('inter_pore_covariance', [[1]])])
def test_unknown_population_models_are_rejected(fixture_files, field, value):
    root, _, _ = fixture_files
    config = configuration()
    config['population'][field] = value
    with pytest.raises(ValueError, match='Unsupported population'):
        run_configuration(config, root)


def test_periodic_host_clearance_and_missing_fractional_cell_are_rejected(fixture_files):
    root, _, _ = fixture_files
    config = configuration('single_crystal_periodic', 'bulk', estimator='moments')
    config['population']['min_host_distance_A'] = 0.5
    with pytest.raises(ValueError, match='Periodic host clearance'):
        run_configuration(config, root)
    finite = configuration()
    finite['pores'] = {'centres_fractional': [[0, 0, 0]]}
    with pytest.raises(ValueError, match='Fractional pore centres require'):
        run_configuration(finite, root)


def test_negative_unbiased_estimates_are_preserved_in_output(fixture_files, monkeypatch):
    import mofdebye.runner as runner
    root, host, _ = fixture_files
    config = configuration('single_crystal_finite', estimator='moments')
    vectors = np.asarray(config['sampling']['q_vectors'])
    mean = -amplitude(host, vectors)
    monkeypatch.setattr(runner, 'guest_moments', lambda *args, **kwargs:
                        GuestMoments(mean, np.zeros(3), np.ones(3), {'method': 'sampled'}))
    result = run_configuration(config, root)
    np.testing.assert_allclose(result['tables']['intensity']['total'], -np.ones(3))
    assert 'negative estimates are retained' in result['metadata']['sampled_moment_policy']


def test_result_files_and_cli_roundtrip_with_empty_reflections(fixture_files):
    root, _, _ = fixture_files
    config = configuration('single_crystal_periodic', 'bulk', guests=False)
    config['sampling']['reflection_q_max'] = 0.01
    result = run_configuration(config, root)
    output = root / 'written'
    write_result(result, output)
    assert (output / 'reflections.csv').read_text(encoding='utf-8').count('\n') == 1
    saved = json.loads((output / 'metadata.json').read_text(encoding='utf-8'))
    assert saved['schema_version'] == '1.0'
    path = root / 'request.json'
    path.write_text(json.dumps(config), encoding='utf-8')
    script = Path(__file__).resolve().parents[1] / 'scripts' / 'run_scattering.py'
    run = subprocess.run([sys.executable, str(script), '--config', str(path), '--out', str(root / 'cli')],
                         cwd=root, check=True, capture_output=True, text=True, timeout=60)
    summary = json.loads(run.stdout)
    assert summary['tables'] == ['reflections', 'diffuse']
    assert (root / 'cli' / 'metadata.json').is_file()
