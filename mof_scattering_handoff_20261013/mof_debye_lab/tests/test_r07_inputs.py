"""CIF/pore/surface integration checks against independent explicit structures."""
from copy import deepcopy

import numpy as np
import pytest

from mofdebye.core import AtomCloud, concatenate
from mofdebye.directional import amplitude, intensity
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.geometry import contains_points
from mofdebye.moments import guest_moments
from mofdebye.periodic import reciprocal_lattice, validate_periodic_population
from mofdebye.runner import _inputs, run_configuration, validate_config


@pytest.fixture
def inputs(tmp_path):
    host = AtomCloud(('C', 'O'), [[.1, .2, .3], [1.8, .2, .3]])
    guest = AtomCloud(('Au',), [[0, 0, 0]])
    dimer = AtomCloud(('Au', 'Au'), [[-.4, 0, 0], [.4, 0, 0]])
    host.write_xyz(tmp_path / 'host.xyz')
    guest.write_xyz(tmp_path / 'guest.xyz')
    dimer.write_xyz(tmp_path / 'dimer.xyz')
    return tmp_path, host, guest


def config(guests=True):
    value = {
        'schema_version': '1.1', 'scattering_case': 'single_crystal_finite',
        'boundary': 'finite', 'guest_estimator': 'explicit', 'backend': 'reference',
        'host': {'xyz': 'host.xyz'},
        'sampling': {'q_vectors': [[0, 0, 0], [.2, -.3, .4], [1.1, .5, -.2]]},
        'statistics': {'n_samples': 8, 'seed': 17},
    }
    if guests:
        value.update(particles=[{'xyz': 'guest.xyz', 'weight': 1}],
                     pores={'centres_A': [[.5, .5, .5]]},
                     population={'occupancy': 1., 'orientation': 'fixed'})
    return value


def cut_box(lower=(0., 0., 0.), upper=(1., 1., 1.)):
    return {'kind': 'box', 'min_A': list(lower), 'max_A': list(upper),
            'surface_policy': 'unpassivated_atom_cut'}


def prepared(value, root):
    validate_config(value)
    return _inputs(value, root)


def test_fractional_pores_and_hkl_match_cartesian_structure_with_schema_11(inputs):
    root, host, guest = inputs
    cell = np.array([[4., .4, .1], [.8, 5., .2], [.3, .7, 6.]])
    hkl = np.array([[0, 0, 0], [1, 0, 0], [1, -1, 2]])
    fractional = np.array([[.2, .3, .4]])
    value = config()
    value['host']['cell_vectors_A'] = cell.tolist()
    value['pores'] = {'centres_fractional': fractional.tolist()}
    value['sampling'] = {'hkl': hkl.tolist()}
    result = run_configuration(value, root)
    cartesian = deepcopy(value)
    cartesian['pores'] = {'centres_A': (fractional @ cell).tolist()}
    other = run_configuration(cartesian, root)
    vectors = hkl @ reciprocal_lattice(cell)
    expected = intensity(concatenate([host, guest.moved((fractional @ cell)[0])]), vectors)
    np.testing.assert_allclose(result['tables']['intensity']['total'], expected, rtol=2e-13)
    np.testing.assert_allclose(other['tables']['intensity']['total'], expected, rtol=2e-13)


def test_cell_replication_and_surface_filter_preserve_per_site_parameters(inputs):
    root, _, _ = inputs
    value = config()
    value['host'].update(cell_vectors_A=(np.eye(3) * 4).tolist(), cell_mask=[[[1]], [[1]]],
                         atom_cut=cut_box((1, 0, 0), (5, 1, 1)), guest_retention='whole_particle')
    value['pores']['centres_A'] = [[.5, .5, .5], [2.5, .5, .5]]
    value['population'].update(occupancy=[.2, .8], pore_radius_A=[.2, .3],
                               mean_displacement_A=[[.1, 0, 0], [-.1, 0, 0]])
    host, _, _, _, population, _, metadata = prepared(value, root)
    np.testing.assert_allclose(host.xyz[:, 0], [1.8, 4.1])
    np.testing.assert_allclose(population.centres, [[2.5, .5, .5], [4.5, .5, .5]])
    np.testing.assert_allclose(population.occupancy, [.8, .2])
    np.testing.assert_allclose(population.pore_radius, [.3, .2])
    np.testing.assert_allclose(population.mean_displacement, [[-.1, 0, 0], [.1, 0, 0]])
    assert metadata['pore_sites_before_surface_filter'] == 4
    assert metadata['retained_pore_sites'] == 2
    assert metadata['atom_cut']['candidate_atom_count'] == 4
    assert metadata['host_atoms'] == 2


def test_all_nominal_pores_removed_becomes_the_cut_host_only(inputs):
    root, host, _ = inputs
    value = config()
    value['host'].update(atom_cut=cut_box((0, 0, 0), (.3, .4, .5)), guest_retention='centre')
    result = run_configuration(value, root)
    expected = intensity(AtomCloud(('C',), host.xyz[:1]), value['sampling']['q_vectors'])
    np.testing.assert_allclose(result['tables']['intensity']['total'], expected)
    assert result['metadata']['input_processing']['retained_pore_sites'] == 0
    assert prepared(value, root)[4] is None


def test_centre_retention_keeps_rigid_overhanging_guest_whole_particle_rejects(inputs):
    root, _, _ = inputs
    value = config()
    value['host'].update(atom_cut=cut_box(), guest_retention='centre')
    value['particles'][0]['xyz'] = 'dimer.xyz'
    value['pores']['centres_A'] = [[.85, .5, .5]]
    value['population']['max_attempts'] = 3
    population = prepared(value, root)[4]
    draw = population.sample(np.random.default_rng(1))[0]
    np.testing.assert_allclose(draw.cloud.xyz, [[.45, .5, .5], [1.25, .5, .5]])
    assert not contains_points(draw.cloud.xyz, value['host']['atom_cut']).all()
    value['host']['guest_retention'] = 'whole_particle'
    with pytest.raises(RuntimeError, match='No valid placement'):
        run_configuration(value, root)


def test_external_centre_rule_conditions_displacement_after_nominal_site_selection(inputs):
    root, _, _ = inputs
    value = config()
    value['host'].update(atom_cut=cut_box(), guest_retention='centre')
    value['population'].update(mean_displacement_A=[.6, 0, 0], max_attempts=2)
    with pytest.raises(RuntimeError, match='No valid placement'):
        run_configuration(value, root)


def test_complete_cell_boundary_intersects_atom_cut_for_external_retention(inputs):
    root, _, _ = inputs
    value = config()
    value['host'].update(cell_vectors_A=(np.eye(3) * 4).tolist(), cell_mask=[[[1]]],
                         atom_cut=cut_box((0, 0, 0), (10, 10, 10)), guest_retention='centre')
    value['population'].update(mean_displacement_A=[4, 0, 0], max_attempts=2)
    with pytest.raises(RuntimeError, match='No valid placement'):
        run_configuration(value, root)


def test_default_site_retention_preserves_existing_overhang_behavior(inputs):
    root, _, guest = inputs
    value = config()
    value['host'].update(cell_vectors_A=(np.eye(3) * 4).tolist(), cell_mask=[[[1]]])
    value['population']['mean_displacement_A'] = [10, 0, 0]
    host, _, _, _, population, _, metadata = prepared(value, root)
    assert population.external_boundary is None
    assert population.guest_retention == metadata['guest_retention'] == 'site'
    expected = intensity(concatenate([host, guest.moved([10.5, .5, .5])]), value['sampling']['q_vectors'])
    np.testing.assert_allclose(run_configuration(value, root)['tables']['intensity']['total'], expected)


@pytest.mark.parametrize('estimator, statistics', [('analytic', {}), ('moments', {'moment_method': 'analytic'})])
def test_analytic_paths_reject_external_conditioning(inputs, estimator, statistics):
    root, _, _ = inputs
    value = config()
    value['guest_estimator'] = estimator
    value['statistics'].update(statistics)
    value['population']['orientation'] = 'uniform'
    value['host'].update(atom_cut=cut_box(), guest_retention='whole_particle')
    with pytest.raises(ValueError, match='External-boundary conditioning'):
        run_configuration(value, root)


def test_external_rejection_keeps_number_weighted_sizes_and_rigid_templates():
    monomer = AtomCloud(('Au',), [[0, 0, 0]])
    dimer = AtomCloud(('Au', 'Au'), [[-.8, 0, 0], [.8, 0, 0]])
    library = ParticleLibrary((monomer, dimer), [.35, .65])
    boundary = cut_box((-1, -1, -1), (1, 1, 1))
    population = PorePopulation([[0, 0, 0]], library, occupancy=1., orientation='fixed',
        position_mode='uniform_ball', uniform_ball_radius_A=.8,
        external_boundary=boundary, guest_retention='whole_particle')
    rng = np.random.default_rng(168)
    counts = np.zeros(2)
    rejected = 0
    for _ in range(600):
        draw = population.sample(rng)[0]
        counts[draw.template_index] += 1
        rejected += draw.attempts - 1
        assert contains_points(draw.cloud.xyz, boundary).all()
        np.testing.assert_allclose(draw.cloud.xyz - draw.displacement_A, library.templates[draw.template_index].xyz, atol=1e-15)
    np.testing.assert_allclose(counts / counts.sum(), library.weights, atol=.06)
    assert rejected > 150


def test_failed_placement_does_not_redraw_selected_particle_size():
    class ControlledRandom:
        choices = 0

        def random(self):
            return 0.

        def choice(self, *args, **kwargs):
            self.choices += 1
            return 1 if self.choices == 1 else 0

        def normal(self, size):
            return np.zeros(size)

    small = AtomCloud(('Au',), [[0, 0, 0]])
    large = AtomCloud(('Au', 'Au'), [[-2, 0, 0], [2, 0, 0]])
    population = PorePopulation([[0, 0, 0]], ParticleLibrary((small, large), [.5, .5]),
        orientation='fixed', external_boundary=cut_box((-1, -1, -1), (1, 1, 1)),
        guest_retention='whole_particle', max_attempts=3)
    rng = ControlledRandom()
    with pytest.raises(RuntimeError, match='template 1'):
        population.sample(rng)
    assert rng.choices == 1


@pytest.mark.parametrize('periodic', [False, True])
def test_sampled_moments_share_clearance_tree_and_boundary_without_mutating_population(monkeypatch, periodic):
    guest = AtomCloud(('Au',), [[0, 0, 0]])
    host = AtomCloud(('C',), [[0, 0, 0]])
    cell = np.eye(3) * 4
    centres = np.array([[.5, .5, .5], [2.5, .5, .5]])
    extra = {'periodic_cell_A': cell} if periodic else {
        'external_boundary': cut_box((0, 0, 0), (4, 4, 4)), 'guest_retention': 'whole_particle'}
    population = PorePopulation(centres, ParticleLibrary((guest,), [1]),
        orientation='fixed', host=host, min_host_distance_A=.1, **extra)
    original_sample = PorePopulation.sample
    seen = []

    def checked_sample(local, rng):
        assert local._host_tree is population._host_tree
        assert local.external_boundary is population.external_boundary
        assert len(local.centres) == 1
        seen.append(local.centres[0].copy())
        return original_sample(local, rng)

    monkeypatch.setattr(PorePopulation, 'sample', checked_sample)
    vectors = np.array([[0, 0, 0], [.2, .3, .4]])
    result = guest_moments(population, vectors, method='sampled', n_samples=4)
    expected = amplitude(concatenate([guest.moved(centre) for centre in centres]), vectors)
    np.testing.assert_allclose(result.mean, expected)
    np.testing.assert_allclose(result.variance, 0., atol=1e-20)
    np.testing.assert_array_equal(population.centres, centres)
    assert len(seen) == 8


def test_runner_periodic_clearance_checks_host_across_cell_face(inputs):
    root, _, _ = inputs
    AtomCloud(('C',), [[.1, .5, .5]]).write_xyz(root / 'host.xyz')
    finite = config()
    finite['host']['cell_vectors_A'] = (np.eye(3) * 4).tolist()
    finite['pores']['centres_A'] = [[3.9, .5, .5]]
    finite['population'].update(min_host_distance_A=.3, max_attempts=2)
    assert np.isfinite(run_configuration(finite, root)['tables']['intensity']['total']).all()
    bulk = deepcopy(finite)
    bulk.update(scattering_case='single_crystal_periodic', boundary='bulk', guest_estimator='moments')
    bulk['sampling'] = {'reflection_q_max': 1.7, 'q_vectors': [[0, 0, 0]]}
    population = prepared(bulk, root)[4]
    assert population._host_tree.is_periodic
    np.testing.assert_allclose(population._host_tree.query(population.centres)[0], [.2])
    with pytest.raises(RuntimeError, match='No valid placement'):
        run_configuration(bulk, root)
    bulk['population']['min_host_distance_A'] = .1
    result = run_configuration(bulk, root)
    assert result['metadata']['input_processing']['host_clearance']['method'] == 'exact_periodic_nearest_host'
    np.testing.assert_allclose(result['tables']['diffuse']['diffuse'], 0., atol=1e-20)


def test_periodic_clearance_cell_mismatch_and_finite_tree_rejected():
    guest = AtomCloud(('Au',), [[0, 0, 0]])
    host = AtomCloud(('C',), [[0, 0, 0]])
    library = ParticleLibrary((guest,), [1])
    periodic = PorePopulation([[1, 1, 1]], library, host=host, min_host_distance_A=.1,
                              periodic_cell_A=np.eye(3) * 4)
    with pytest.raises(ValueError, match='differs from the scattering cell'):
        validate_periodic_population(periodic, np.eye(3) * 5)
    finite = PorePopulation([[1, 1, 1]], library, host=host, min_host_distance_A=.1)
    with pytest.raises(ValueError, match='periodic neighbor handling'):
        validate_periodic_population(finite, np.eye(3) * 4)


@pytest.mark.parametrize('host_change, message', [
    ({'atom_cut': cut_box()}, 'explicit centre or whole_particle'),
    ({'guest_retention': 'centre'}, 'external boundary'),
    ({'guest_retention': 'clip'}, 'guest_retention'),
    ({'cif': 'structure.cif'}, 'exactly one'),
    ({'cif_options': {}}, 'requires host.cif'),
])
def test_new_input_contract_rejects_ambiguous_requests(host_change, message):
    value = config()
    value['host'].update(host_change)
    with pytest.raises(ValueError, match=message):
        validate_config(value)


def test_atom_cut_rejects_complete_cell_factorization_and_legacy_schema():
    value = config(False)
    value['host'].update(cell_vectors_A=(np.eye(3) * 4).tolist(), cell_mask=[[[1]]], atom_cut=cut_box())
    value['backend'] = 'lattice_grouped'
    with pytest.raises(ValueError, match='break complete-cell factorization'):
        validate_config(value)
    value['backend'] = 'reference'
    value['schema_version'] = '1.0'
    with pytest.raises(ValueError, match='schema_version "1.1"'):
        validate_config(value)


@pytest.fixture
def cif_inputs(inputs):
    root, _, guest = inputs
    (root / 'nonorthogonal.cif').write_text('''data_nonorthogonal
_cell_length_a 5
_cell_length_b 6
_cell_length_c 7
_cell_angle_alpha 90
_cell_angle_beta 105
_cell_angle_gamma 90
_space_group_name_H-M_alt 'P -1'
_space_group_IT_number 2
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,-y,-z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C .1 .2 .3 1
Zn1 Zn 0 0 0 1
''', encoding='utf-8')
    beta = np.deg2rad(105.)
    cell = np.array([[5., 0., 0.], [0., 6., 0.], [7 * np.cos(beta), 0., 7 * np.sin(beta)]])
    basis = AtomCloud(('C', 'C', 'Zn'), np.array([[.1, .2, .3], [.9, .8, .7], [0, 0, 0]]) @ cell)
    basis.write_xyz(root / 'expanded.xyz')
    return root, basis, guest, cell


@pytest.mark.parametrize('case, boundary', [
    ('single_crystal_finite', 'finite'), ('single_crystal_periodic', 'bulk'),
    ('pxrd', 'finite'), ('pxrd', 'bulk'),
])
def test_cif_expansion_connects_all_scattering_modes_to_explicit_nonorthogonal_reference(cif_inputs, case, boundary):
    root, _, _, cell = cif_inputs
    value = config(False)
    value.update(scattering_case=case, boundary=boundary, guest_estimator='analytic')
    value['host'] = {'cif': 'nonorthogonal.cif'}
    if case == 'pxrd':
        value['sampling'] = {'q': [0., .7, 1.2]}
        value['angular'] = {'n_mu': 4, 'n_phi': 8}
    elif boundary == 'bulk':
        value['sampling'] = {'reflection_q_max': 1.4, 'hkl': [[0, 0, 0], [1, -1, 1]]}
    else:
        value['sampling'] = {'hkl': [[0, 0, 0], [1, 0, 0], [1, -1, 2]]}
    reference = deepcopy(value)
    reference['host'] = {'xyz': 'expanded.xyz', 'cell_vectors_A': cell.tolist()}
    result = run_configuration(value, root)
    expected = run_configuration(reference, root)
    assert set(result['tables']) == set(expected['tables'])
    for table, columns in result['tables'].items():
        for name, values in columns.items():
            np.testing.assert_allclose(values, expected['tables'][table][name], rtol=3e-12, atol=2e-9)
    processing = result['metadata']['input_processing']
    assert processing['basis_atoms'] == 3
    assert processing['retained_cells'] == (1 if boundary == 'finite' else None)
    assert processing['crystallography']['symmetry_operation_count'] == 2
    assert processing['crystallography']['duplicates_removed'] == 1
    assert len(result['metadata']['input_sha256']) == 1


def test_cif_complete_cell_factorization_and_atom_cut_use_expanded_basis(cif_inputs):
    root, basis, _, cell = cif_inputs
    value = config(False)
    value['host'] = {'cif': 'nonorthogonal.cif', 'cell_mask': [[[1]], [[0]], [[1]]]}
    reference = run_configuration(value, root)
    value['backend'] = 'lattice_grouped'
    grouped = run_configuration(value, root)
    np.testing.assert_allclose(grouped['tables']['intensity']['total'], reference['tables']['intensity']['total'], rtol=3e-12)
    assert grouped['metadata']['input_processing']['host_atoms_materialized'] is False
    assert grouped['metadata']['input_processing']['host_atoms'] == 6
    value['backend'] = 'reference'
    value['host']['atom_cut'] = cut_box((-1, -1, -1), (7, 10, 10))
    result = run_configuration(value, root)
    expanded = concatenate([basis, basis.moved(2 * cell[0])])
    keep = np.all((expanded.xyz >= [-1, -1, -1]) & (expanded.xyz <= [7, 10, 10]), axis=1)
    explicit_cut = AtomCloud(tuple(e for e, retained in zip(expanded.elements, keep) if retained), expanded.xyz[keep])
    np.testing.assert_allclose(result['tables']['intensity']['total'], intensity(explicit_cut, value['sampling']['q_vectors']), rtol=3e-12)
    assert result['metadata']['input_processing']['host_atoms'] == len(explicit_cut)


def test_cif_pore_symmetry_expansion_and_coordinate_frames_are_equivalent(cif_inputs):
    root, basis, guest, cell = cif_inputs
    value = config()
    value['host'] = {'cif': 'nonorthogonal.cif'}
    value['pores'] = {'centres_fractional': [[.2, .3, .4]], 'expand_symmetry': True}
    value['sampling'] = {'hkl': [[0, 0, 0], [1, -1, 2], [0, 1, 1]]}
    result = run_configuration(value, root)
    expanded_centres = np.array([[.2, .3, .4], [.8, .7, .6]]) @ cell
    explicit = concatenate([basis] + [guest.moved(centre) for centre in expanded_centres])
    np.testing.assert_allclose(result['tables']['intensity']['total'],
                               intensity(explicit, np.asarray(value['sampling']['hkl']) @ reciprocal_lattice(cell)), rtol=3e-12)
    assert result['metadata']['input_processing']['input_pore_sites'] == 1
    assert result['metadata']['input_processing']['pore_sites_per_cell'] == 2
    cartesian = deepcopy(value)
    cartesian['pores'] = {'centres_A': expanded_centres[:1].tolist(), 'expand_symmetry': True}
    np.testing.assert_allclose(run_configuration(cartesian, root)['tables']['intensity']['total'],
                               result['tables']['intensity']['total'], rtol=3e-12)


@pytest.mark.parametrize('parameters, message', [
    ({'occupancy': [.3]}, 'shared scalar occupancy/radius'),
    ({'pore_radius_A': [.3]}, 'shared scalar occupancy/radius'),
    ({'mean_displacement_A': [[0, 0, 0]]}, 'shared mean displacement'),
])
def test_cif_symmetry_expanded_pores_require_shared_parameters(cif_inputs, parameters, message):
    root, _, _, _ = cif_inputs
    value = config()
    value['host'] = {'cif': 'nonorthogonal.cif'}
    value['pores'] = {'centres_fractional': [[.2, .3, .4]], 'expand_symmetry': True}
    value['population'].update(parameters)
    with pytest.raises(ValueError, match=message):
        run_configuration(value, root)


def test_pore_symmetry_requires_cif_and_explicit_boolean(inputs):
    root, _, _ = inputs
    value = config()
    value['pores']['expand_symmetry'] = True
    with pytest.raises(ValueError, match='requires host.cif'):
        run_configuration(value, root)
    value['pores']['expand_symmetry'] = 'yes'
    with pytest.raises(ValueError, match='must be a boolean'):
        run_configuration(value, root)


def duplicate_site_config(mode):
    value = config()
    if mode != 'cartesian':
        value['host']['cell_vectors_A'] = (np.eye(3) * 4).tolist()
    if mode == 'cell_mask':
        value['host']['cell_mask'] = [[[1]]]
    if mode == 'bulk':
        value.update(scattering_case='single_crystal_periodic', boundary='bulk', guest_estimator='moments')
        value['sampling'] = {'reflection_q_max': 1.7, 'q_vectors': [[0, 0, 0]]}
    value['pores'] = {'centres_A': [[.5, .5, .5], [.5 if mode == 'cartesian' else 4.5, .5, .5]]}
    return value


@pytest.mark.parametrize('mode', ['cartesian', 'cell_mask', 'bulk'])
def test_schema11_rejects_duplicate_pores_in_the_applicable_coordinate_domain(inputs, mode):
    root, _, _ = inputs
    value = duplicate_site_config(mode)
    with pytest.raises(ValueError, match='[Dd]uplicate|equivalent'):
        prepared(value, root)


@pytest.mark.parametrize('mode', ['cartesian', 'cell_mask', 'bulk'])
def test_schema10_duplicate_pore_behavior_remains_compatible(inputs, mode):
    root, _, _ = inputs
    value = duplicate_site_config(mode)
    value['schema_version'] = '1.0'
    population = prepared(value, root)[4]
    assert len(population.centres) == 2
    assert len(population.sample(np.random.default_rng(1))) == 2


@pytest.mark.parametrize('mode', ['cartesian', 'cell_mask', 'bulk'])
def test_schema11_allows_distinct_pore_centres(inputs, mode):
    root, _, _ = inputs
    value = duplicate_site_config(mode)
    value['pores']['centres_A'][1] = [1.5, .5, .5]
    population = prepared(value, root)[4]
    np.testing.assert_allclose(population.centres, [[.5, .5, .5], [1.5, .5, .5]])


def test_finite_explicit_xyz_pores_can_span_multiple_cells_without_cell_mask(inputs):
    root, _, _ = inputs
    value = duplicate_site_config('explicit_multiple_cells')
    population = prepared(value, root)[4]
    np.testing.assert_allclose(population.centres, [[.5, .5, .5], [4.5, .5, .5]])
    assert len(population.sample(np.random.default_rng(1))) == 2
