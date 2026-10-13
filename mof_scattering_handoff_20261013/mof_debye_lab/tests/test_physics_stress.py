"""Independent R07 physics checks, including full probability enumeration.

The small-system oracles below deliberately do not call production scattering,
pair-distance, moment, symmetry, or angular-grid routines. IT92 coefficients are
the shared physical input, not an independently calibrated scattering model.
"""
from itertools import product
import json
from pathlib import Path

import gemmi
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from mofdebye.core import AtomCloud, concatenate
from mofdebye.crystal import read_cif
from mofdebye.directional import powder_average
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.moments import exact_guest_moments, guest_moments
from mofdebye.periodic import finite_cell_ensemble
from mofdebye.runner import run_configuration


def record(request, fixture, equation, **measurements):
    """Retain numerical evidence in JUnit without writing into the source tree."""
    request.node.user_properties.append(('physics_record', json.dumps(
        dict(fixture=fixture, equation=equation, **measurements), allow_nan=False)))


def scaled_error(actual, expected):
    return float(np.max(abs(np.asarray(actual) - expected)) / max(1., np.max(abs(expected))))


def atom_factors(elements, q):
    """One factor per atom, using the published IT92 coefficient definition."""
    s2 = (np.asarray(q) / (4 * np.pi)) ** 2
    rows = []
    for element in elements:
        c = gemmi.Element(element).it92
        rows.append(sum(a * np.exp(-b * s2) for a, b in zip(c.a, c.b)) + c.c)
    return np.asarray(rows).reshape(len(elements), len(s2))


def atomic_sum(cloud, vectors):
    vectors = np.asarray(vectors)
    factors = atom_factors(cloud.elements, np.linalg.norm(vectors, axis=1))
    return np.sum(factors * np.exp(1j * cloud.xyz @ vectors.T), axis=0)


def pair_sum(cloud, q):
    """Full ordered i,j sum, independent of production pair decomposition."""
    factors = atom_factors(cloud.elements, q)
    distances = np.linalg.norm(cloud.xyz[:, None, :] - cloud.xyz[None, :, :], axis=2)
    return np.einsum('iq,jq,ijq->q', factors, factors,
                     np.sinc(distances[:, :, None] * np.asarray(q) / np.pi))


def distribution(templates, weights, centres, occupancy):
    """Enumerate independent vacancy/template choices without production draws."""
    states, probabilities = [], []
    for choices in product(range(len(templates) + 1), repeat=len(centres)):
        clouds, probability = [], 1.
        for choice, centre, occupied in zip(choices, centres, occupancy):
            probability *= 1 - occupied if choice == 0 else occupied * weights[choice - 1]
            if choice:
                clouds.append(templates[choice - 1].moved(centre))
        states.append(concatenate(clouds))
        probabilities.append(probability)
    return states, np.asarray(probabilities)


@pytest.fixture
def triclinic(tmp_path):
    (tmp_path / 'host.cif').write_text('''data_triclinic
_cell_length_a 5.3
_cell_length_b 6.1
_cell_length_c 7.4
_cell_angle_alpha 78
_cell_angle_beta 97
_cell_angle_gamma 105
_space_group_name_H-M_alt 'P -1'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C .13 .27 .39 1
Zn1 Zn .04 .11 .19 1
''', encoding='utf-8')
    alpha, beta, gamma = np.deg2rad([78, 97, 105])
    cx = 7.4 * np.cos(beta)
    cy = 7.4 * (np.cos(alpha) - np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cell = np.array([[5.3, 0, 0], [6.1 * np.cos(gamma), 6.1 * np.sin(gamma), 0],
                     [cx, cy, np.sqrt(7.4**2 - cx**2 - cy**2)]])
    expected = AtomCloud(('C', 'C', 'Zn', 'Zn'),
                         np.array([[.13, .27, .39], [.87, .73, .61],
                                   [.04, .11, .19], [.96, .89, .81]]) @ cell)
    crystal = read_cif(tmp_path / 'host.cif')
    np.testing.assert_allclose(crystal.cell_vectors_A, cell, atol=2e-14)
    expected_rows = sorted((e, *np.round(x, 10)) for e, x in zip(expected.elements, expected.xyz))
    actual_rows = sorted((e, *np.round(x, 10)) for e, x in zip(crystal.basis.elements, crystal.basis.xyz))
    assert actual_rows == expected_rows
    return tmp_path, crystal, expected


@pytest.mark.parametrize('seed', [31, 1701, 98271])
def test_imported_triclinic_mixed_sizes_full_cell_enumeration_and_frame_invariance(triclinic, seed, request):
    _, crystal, host = triclinic
    templates = (AtomCloud(('Au',), [[0, 0, 0]]),
                 AtomCloud(('Au', 'Au', 'O'), [[-.4, 0, .1], [.4, .2, 0], [0, -.3, .2]]))
    centres = np.array([[.21, .31, .41], [.72, .63, .54]]) @ crystal.cell_vectors_A
    states, probabilities = distribution(templates, [.4, .6], centres, [.27, .82])
    rng = np.random.default_rng(seed)
    vectors = np.vstack((np.zeros((1, 3)), rng.normal(size=(19, 3))))
    # An irregular disconnected shape prevents rectangular-shape shortcuts.
    mask = np.array([[[0], [1]], [[1], [0]]])
    shifts = np.array([[0, 1, 0], [1, 0, 0]]) @ crystal.cell_vectors_A
    expected = np.zeros(len(vectors))
    for i, j in product(range(len(states)), repeat=2):
        complete = concatenate([host.moved(shifts[0]), host.moved(shifts[1]),
                                states[i].moved(shifts[0]), states[j].moved(shifts[1])])
        expected += probabilities[i] * probabilities[j] * abs(atomic_sum(complete, vectors))**2
    moments = exact_guest_moments(states, probabilities, vectors)
    result = finite_cell_ensemble(crystal.basis, crystal.cell_vectors_A, mask, vectors, moments)
    np.testing.assert_allclose(result['total'], expected, rtol=3e-13, atol=3e-9)
    rotation = Rotation.from_rotvec([.31, -.73, .27]).as_matrix()
    translation = np.array([13.4, -7.8, 2.1])
    rotated_vectors = vectors @ rotation.T
    transformed = [state.moved(translation, rotation) for state in states]
    transformed_moments = exact_guest_moments(transformed, probabilities, rotated_vectors)
    transformed_result = finite_cell_ensemble(crystal.basis.moved(translation, rotation),
        crystal.cell_vectors_A @ rotation.T, mask, rotated_vectors, transformed_moments)
    np.testing.assert_allclose(transformed_result['total'], expected, rtol=3e-13, atol=3e-9)
    record(request, 'Two inversion-expanded triclinic cells at indices (0,1,0) and (1,0,0); two independent pores per cell; vacancy/monomer/trimer states; 81 complete configurations.',
        'E[I(q)] = sum_s P(s) |sum_i f_i(q) exp(i q.r_i,s)|^2 = |S(q)|^2 |H(q)+mu(q)|^2 + M Var(A_guest(q))',
        seed=seed, exact_state_count=81, q_count=len(vectors),
        enumeration_scaled_error=scaled_error(result['total'], expected),
        rotated_translated_scaled_error=scaled_error(transformed_result['total'], expected),
        forward_intensity_electron2=float(expected[0]))


def test_powder_angular_convergence_matches_full_state_ordered_pair_oracle(triclinic, request):
    _, crystal, host = triclinic
    templates = (AtomCloud(('Au',), [[0, 0, 0]]),
                 AtomCloud(('Au', 'Au'), [[-.5, 0, 0], [.5, 0, 0]]))
    centres = np.array([[.2, .3, .4], [.7, .6, .5]]) @ crystal.cell_vectors_A
    states, probabilities = distribution(templates, [.3, .7], centres, [.4, .8])
    mask = np.ones((2, 1, 1), dtype=int)
    q = np.array([0, .19, .71, 1.23, 2.7])
    expected = np.zeros(len(q))
    for i, j in product(range(len(states)), repeat=2):
        cloud = concatenate([host, states[i], host.moved(crystal.cell_vectors_A[0]),
                             states[j].moved(crystal.cell_vectors_A[0])])
        expected += probabilities[i] * probabilities[j] * pair_sum(cloud, q)
    def evaluate(vectors):
        moments = exact_guest_moments(states, probabilities, vectors)
        return finite_cell_ensemble(crystal.basis, crystal.cell_vectors_A, mask, vectors, moments)['total']
    coarse = powder_average(evaluate, q, n_mu=8, n_phi=16)
    fine = powder_average(evaluate, q, n_mu=48, n_phi=96)
    refined = powder_average(evaluate, q, n_mu=64, n_phi=128)
    # This object has q*extent ~ 35: a small grid really is insufficient.
    assert np.max(abs(coarse - expected)) > .1
    np.testing.assert_allclose(fine, expected, rtol=2e-11, atol=2e-7)
    np.testing.assert_allclose(refined, expected, rtol=2e-11, atol=2e-7)
    np.testing.assert_allclose(refined, fine, rtol=2e-11, atol=2e-7)
    record(request, 'Two triclinic cells with independent Bernoulli monomer/dimer guests; 81 states; q=0..2.7 angstrom^-1.',
        '<I(q)>_sphere = sum_s P(s) sum_i,j f_i(q) f_j(q) sinc(q |r_i,s-r_j,s|)',
        grids=[[8, 16], [48, 96], [64, 128]],
        coarse_scaled_error=scaled_error(coarse, expected),
        fine_scaled_error=scaled_error(fine, expected),
        refined_scaled_error=scaled_error(refined, expected),
        coarse_max_absolute_error_electron2=float(np.max(abs(coarse - expected))))


@pytest.mark.parametrize('case,boundary', [('single_crystal_finite', 'finite'),
    ('pxrd', 'finite'), ('single_crystal_periodic', 'bulk'), ('pxrd', 'bulk')])
def test_runner_imported_host_with_bernoulli_template_mixture_all_modes(triclinic, case, boundary, request):
    root, crystal, host = triclinic
    # Monatomic species make every orientation identical and permit exact finite
    # probability enumeration at nonzero q, including interference terms.
    templates = (AtomCloud(('Au',), [[0, 0, 0]]), AtomCloud(('Ag',), [[0, 0, 0]]))
    for template, name in zip(templates, ('gold.xyz', 'silver.xyz')):
        template.write_xyz(root / name)
    fractional = np.array([[.21, .31, .41], [.79, .69, .59]])
    centres = fractional @ crystal.cell_vectors_A
    states, probabilities = distribution(templates, [.37, .63], centres, [.27, .82])
    q = np.array([0, .37, 1.19, 2.03])
    vectors = np.array([[0, 0, 0], [.17, -.29, .37], [1.1, .2, -.7]])
    config = dict(schema_version='1.1', scattering_case=case, boundary=boundary,
        guest_estimator='analytic', backend='reference', host={'cif': 'host.cif'},
        particles=[{'xyz': 'gold.xyz', 'weight': .37}, {'xyz': 'silver.xyz', 'weight': .63}],
        pores={'centres_fractional': fractional.tolist()}, population={'occupancy': [.27, .82]},
        angular={'n_mu': 4, 'n_phi': 8})
    if case == 'pxrd':
        config['sampling'] = {'q': q.tolist()}
    elif boundary == 'bulk':
        config['sampling'] = {'reflection_q_max': 2.03, 'q_vectors': vectors.tolist()}
    else:
        config['sampling'] = {'q_vectors': vectors.tolist()}
    result = run_configuration(config, root)
    if boundary == 'finite':
        evaluator = pair_sum if case == 'pxrd' else lambda c, v: abs(atomic_sum(c, v))**2
        expected = sum(p * evaluator(concatenate([host, state]), q if case == 'pxrd' else vectors)
                       for state, p in zip(states, probabilities))
        np.testing.assert_allclose(result['tables']['intensity']['total'], expected,
                                   rtol=2e-13, atol=2e-9)
        record(request, 'Imported triclinic C/Zn host; two Bernoulli sites with Au/Ag monatomic number mixture; nine exact states.',
            'E[I] = sum_s P(s) I(host+guest_s); powder I = sum_i,j f_i f_j sinc(q r_ij)',
            scattering_case=case, boundary=boundary,
            intensity_scaled_error=scaled_error(result['tables']['intensity']['total'], expected))
    else:
        table = result['tables']['reflections']
        reciprocal = 2 * np.pi * np.linalg.inv(crystal.cell_vectors_A).T
        oracle_hkl = [h for h in product(range(-4, 5), repeat=3)
                      if h != (0, 0, 0) and np.linalg.norm(np.asarray(h) @ reciprocal) <= 2.03 + 1e-12]
        actual_hkl = np.column_stack([table[k] for k in ('h', 'k', 'l')])
        assert set(map(tuple, actual_hkl)) == set(oracle_hkl)
        reciprocal_vectors = actual_hkl @ reciprocal
        values = np.array([atomic_sum(state, reciprocal_vectors) for state in states])
        mean = probabilities @ values
        variance = probabilities @ abs(values - mean)**2
        coherent = abs(atomic_sum(host, reciprocal_vectors) + mean)**2
        prefactor = (2 * np.pi)**3 / abs(np.linalg.det(crystal.cell_vectors_A))
        np.testing.assert_allclose(table['f_squared'], coherent, rtol=2e-13, atol=2e-9)
        np.testing.assert_allclose(table['diffuse_at_reflections'], variance, rtol=2e-13)
        np.testing.assert_allclose(table['bragg_weight'], prefactor * coherent, rtol=2e-13, atol=2e-9)
        # A monatomic mixed catalogue has isotropic variance even for two pores.
        sample_q = q if case == 'pxrd' else np.linalg.norm(vectors, axis=1)
        f = atom_factors(('Au', 'Ag'), sample_q)
        e1, e2 = np.array([.37, .63]) @ f, np.array([.37, .63]) @ f**2
        expected_diffuse = sum(p * e2 - p**2 * e1**2 for p in (.27, .82))
        np.testing.assert_allclose(result['tables']['diffuse']['diffuse'], expected_diffuse, rtol=2e-13)
        if case == 'pxrd':
            shells = result['tables']['powder_shells']
            assert shells['multiplicity'].sum() == len(table['q'])
            for shell_q, multiplicity, weight in zip(shells['q'], shells['multiplicity'], shells['radial_weight']):
                members = np.isclose(table['q'], shell_q, rtol=1e-10, atol=1e-12)
                assert members.sum() == multiplicity
                assert weight == pytest.approx(np.sum(prefactor * coherent[members] / (4 * np.pi * shell_q**2)), rel=2e-13)
        record(request, 'Imported triclinic C/Zn host; two Bernoulli Au/Ag guest sites; independent cells and exact nine-state cell distribution.',
            'W_G=(2*pi)^3/V |H(G)+E[A_guest(G)]|^2; diffuse=Var(A_guest); W_shell=sum_G W_G/(4*pi*|G|^2)',
            scattering_case=case, boundary=boundary, reflection_count=len(actual_hkl),
            coherent_scaled_error=scaled_error(table['f_squared'], coherent),
            diffuse_scaled_error=scaled_error(table['diffuse_at_reflections'], variance),
            measure_scaled_error=scaled_error(table['bragg_weight'], prefactor * coherent))


@pytest.mark.parametrize('retention', ['centre', 'whole_particle'])
@pytest.mark.parametrize('seed', [8842, 3901])
def test_conditioned_mixed_size_guest_moments_against_independent_truncated_gaussian_quadrature(retention, seed, request):
    templates = (AtomCloud(('Au',), [[0, 0, 0]]),
                 AtomCloud(('Au', 'Au'), [[-.65, 0, 0], [.65, 0, 0]]))
    weights = np.array([.29, .71])
    occupancy = .73
    mean = np.array([.25, -.1, .05])
    sigma = np.array([.55, .30, .20])
    boundary = {'kind': 'box', 'min_A': [-1, -1, -1], 'max_A': [1, 1, 1]}
    pop = PorePopulation([[0, 0, 0]], ParticleLibrary(templates, weights), occupancy=occupancy,
        displacement_covariance_A2=np.diag(sigma**2), mean_displacement_A=mean,
        orientation='fixed', external_boundary=boundary, guest_retention=retention)
    vectors = np.array([[0, 0, 0], [.2, -.4, .7], [1.3, .6, -.2], [2.9, -1.1, .9]])
    conditional_mean = np.zeros(len(vectors), dtype=complex)
    conditional_second = np.zeros(len(vectors))
    nodes, quadrature_weights = np.polynomial.legendre.leggauss(128)
    for weight, template in zip(weights, templates):
        lower = -np.ones(3)
        upper = np.ones(3)
        if retention == 'whole_particle':
            lower -= template.xyz.min(axis=0)
            upper -= template.xyz.max(axis=0)
        characteristic = np.ones(len(vectors), dtype=complex)
        for axis in range(3):
            x = .5 * (upper[axis] - lower[axis]) * nodes + .5 * (upper[axis] + lower[axis])
            density = np.exp(-.5 * ((x - mean[axis]) / sigma[axis])**2)
            integration_weights = quadrature_weights * density
            characteristic *= (np.exp(1j * vectors[:, axis, None] * x) @ integration_weights) / integration_weights.sum()
        internal = atomic_sum(template, vectors)
        conditional_mean += weight * internal * characteristic
        conditional_second += weight * abs(internal)**2
    expected_mean = occupancy * conditional_mean
    expected_variance = occupancy * conditional_second - abs(expected_mean)**2
    samples = 4096
    sampled = guest_moments(pop, vectors, method='sampled', n_samples=samples, seed=seed)
    # Fixed seeds plus conservative seven-sigma simultaneous limits avoid fragile
    # Monte Carlo tolerances. The oracle integrates the actual conditional law,
    # not a replay of production sampling or unconditioned Gaussian moments.
    error_in_se = abs(sampled.mean - expected_mean) / np.sqrt(expected_variance / samples)
    assert error_in_se.max() < 7
    np.testing.assert_allclose(sampled.variance, expected_variance, rtol=.12)
    counts = np.asarray(sampled.metadata['template_counts'])
    assert abs(counts.sum() / samples - occupancy) < 7 * np.sqrt(occupancy * (1 - occupancy) / samples)
    assert abs(counts[0] / counts.sum() - weights[0]) < 7 * np.sqrt(weights[0] * weights[1] / counts.sum())
    assert sampled.metadata['mean_placement_attempts'] > 1.05
    record(request, 'Fixed-orientation Au monomer/dimer at one Bernoulli site in [-1,1]^3; diagonal Gaussian displacement; fixed-size rejection; independently integrated truncated-normal characteristic functions.',
        'phi_t(q)=product_a integral_Lta^Uta exp(i q_a x) N(x;m_a,sigma_a) dx / integral_Lta^Uta N dx; mu=p sum_t w_t A_t phi_t; variance=p sum_t w_t |A_t|^2-|mu|^2',
        retention=retention, seed=seed, samples=samples, quadrature_order=128,
        maximum_mean_error_in_exact_sem=float(error_in_se.max()),
        mean_error_sem_limit=7., maximum_variance_relative_error=float(np.max(abs(sampled.variance / expected_variance - 1))),
        variance_relative_tolerance=.12, occupied_draws=int(counts.sum()),
        mean_placement_attempts=sampled.metadata['mean_placement_attempts'])


def test_real_mof_synthetic_algebraic_guest_sites_keep_bragg_absences_and_add_diffuse(tmp_path, request):
    """Synthetic F-centering algebra only, not identified physical MOF pores.

    These arbitrary guest sites have no validated host clearance or loading
    interpretation. The real host tests imported symmetry in the same algebra.
    """
    source = Path(__file__).resolve().parents[1] / 'examples' / 'real_mof' / 'IRMOF-1.cif'
    crystal = read_cif(source)
    AtomCloud(('Au',), [[0, 0, 0]]).write_xyz(tmp_path / 'gold.xyz')
    sites = np.array([[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0]])
    occupancy = .61
    # Synthetic algebraic guest configuration; not a physical MOF loading model.
    config = dict(schema_version='1.1', scattering_case='single_crystal_periodic', boundary='bulk',
        guest_estimator='analytic', backend='reference', host={'cif': str(source)},
        particles=[{'xyz': 'gold.xyz'}], pores={'centres_fractional': sites.tolist()},
        population={'occupancy': occupancy}, sampling={'reflection_q_max': 1.0})
    result = run_configuration(config, tmp_path)
    table = result['tables']['reflections']
    hkl = np.column_stack([table[k] for k in ('h', 'k', 'l')])
    parity = hkl % 2
    forbidden = ~np.all(parity == parity[:, :1], axis=1)
    assert forbidden.sum() > 50
    assert table['f_squared'][forbidden].max() < 1e-18
    f = atom_factors(('Au',), table['q'])[0]
    np.testing.assert_allclose(table['diffuse_at_reflections'], 4 * occupancy * (1 - occupancy) * f**2, rtol=3e-14)
    assert np.all(table['diffuse_at_reflections'][forbidden] > 1000)
    vectors = np.column_stack([table[k] for k in ('qx', 'qy', 'qz')])
    host_amplitude = atomic_sum(crystal.basis, vectors)
    guest_mean = occupancy * f * np.exp(2j * np.pi * sites @ hkl.T).sum(axis=0)
    np.testing.assert_allclose(table['f_squared'], abs(host_amplitude + guest_mean)**2, rtol=3e-12, atol=3e-8)
    record(request, 'Imported IRMOF-1 host plus four SYNTHETIC algebraic F-centering guest sites, p=0.61. Sites are not identified physical pores; host clearance/loading was not validated.',
        'F_F(hkl)=1+(-1)^(k+l)+(-1)^(h+l)+(-1)^(h+k); mixed parity implies Bragg=0 while independent guest diffuse=4*p*(1-p)*f_Au^2',
        reflection_count=len(hkl), forbidden_reflection_count=int(forbidden.sum()),
        largest_forbidden_bragg_electron2=float(table['f_squared'][forbidden].max()),
        smallest_forbidden_diffuse_electron2=float(table['diffuse_at_reflections'][forbidden].min()),
        oracle_scaled_error=scaled_error(table['f_squared'], abs(host_amplitude + guest_mean)**2))
