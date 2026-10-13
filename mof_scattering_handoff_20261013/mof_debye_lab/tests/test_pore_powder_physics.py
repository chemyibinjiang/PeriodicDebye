"""Independent geometry/probability references for pore-distribution PXRD."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip('numba', reason='The fast finite-powder backend requires the optional fast extra.')
from mofdebye.core import AtomCloud, form_factors
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.pore_powder import PorePowderCache
from mofdebye.radial_guests import radial_guest_statistics

_spec = importlib.util.spec_from_file_location('pore_powder_oracles',
    Path(__file__).parents[1]/'scripts'/'validate_pore_powder.py')
_oracle = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_oracle)


@pytest.mark.parametrize('cells', [1, 2, 3])
def test_body_centered_integer_shells_match_all_host_pore_pairs(cells):
    spacing = 3.7
    positions = np.array(list(np.ndindex((cells,)*3)), dtype=float)*spacing
    pores = positions+spacing/2
    distances = np.linalg.norm(positions[:, None, :]-pores[None, :, :], axis=2)
    q = np.linspace(0, 6, 61)
    expected = np.array([np.sinc(value*distances/np.pi).sum() for value in q])
    radii, counts = _oracle.body_centered_cross_shells(cells, spacing)
    actual = _oracle.shell_sum(q, radii, counts)
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-12)


@pytest.mark.parametrize('probability', [0., .3, 1.])
@pytest.mark.parametrize('dimer_weight', [0., .65, 1.])
def test_finite_distribution_matches_closed_pair_components(probability, dimer_weight):
    q = np.linspace(0, 6, 601)
    population = _oracle.synthetic_population(occupancy=probability, dimer_weight=dimer_weight)
    stats = radial_guest_statistics(population, q, method='analytic')
    cache = PorePowderCache(AtomCloud(('C',), [[0., 0., 0.]]), np.eye(3)*25.832,
                           np.ones((2, 2, 2), dtype=int), population, q, threads=1)
    actual = cache.iq(stats)
    expected = _oracle.body_centered_pore_reference(q, 2, occupancy=probability, dimer_weight=dimer_weight)
    scale = max(expected['total'][q >= .1])
    for name in expected:
        assert np.max(abs(actual[name][q >= .1]-expected[name][q >= .1]))/scale < 3e-10, name
        np.testing.assert_allclose(actual[name][0], expected[name][0], rtol=3e-13, atol=1e-7)
    assert not np.any(actual['total_mc_sem'])
    # Same-particle pairs carry p, while distinct-pore interference carries p².
    m, internal = _oracle.monomer_dimer_statistics(q, 2., dimer_weight)
    f_host = form_factors(('C',), q)['C']
    expected_zero = (8*f_host[0]+8*probability*m[0])**2 + 8*(probability*internal[0]-(probability*m[0])**2)
    np.testing.assert_allclose(actual['total'][0], expected_zero, rtol=3e-13)
    if probability:
        negative = expected['host_guest'] < -1e-4
        assert np.any(negative)
        assert np.all(actual['host_guest'][negative] < 0), 'Interference must not be clipped.'


@pytest.mark.parametrize('shifted', [False, True])
def test_different_pore_occupancies_use_independent_pair_probabilities(shifted):
    q = np.linspace(0, 6, 121)
    spacing = 10.
    basis = AtomCloud(('C', 'O'), [[0., 0., 0.], [1.3, .7, .4]])
    library = ParticleLibrary((AtomCloud(('Au',), [[0., 0., 0.]]),
                               AtomCloud(('Au', 'Au'), [[-.7, 0., 0.], [.7, 0., 0.]])), [.4, .6])
    centres = np.array([[2., 2., 2.], [7., 2., 2.], [2., 7., 2.]])
    probabilities = np.array([0., .3, 1.])
    shifts = np.array([[.15, -.2, .1], [-.1, .2, .1], [.2, .1, -.1]]) if shifted else np.zeros((3, 3))
    population = PorePopulation(centres, library, occupancy=probabilities,
        position_mode='uniform_ball', uniform_ball_radius_A=.4, pore_radius_A=2.,
        mean_displacement_A=shifts, orientation='uniform')
    mask = np.ones((2, 1, 1), dtype=int)
    stats = radial_guest_statistics(population, q, method='analytic')
    actual = PorePowderCache(basis, np.eye(3)*spacing, mask, population, q, threads=1).iq(stats)
    translations = np.array([[0., 0., 0.], [spacing, 0., 0.]])
    host_xyz = (translations[:, None, :]+basis.xyz).reshape(-1, 3)
    pore_xyz = (translations[:, None, :]+centres+shifts).reshape(-1, 3)
    ff = form_factors(basis.elements+('Au',), q)
    host_f = np.array([ff[e] for e in basis.elements*2])
    mean, internal = _oracle.monomer_dimer_statistics(q, 1.4, .6)
    pore_f = np.tile(probabilities, 2)[:, None]*mean*_oracle.ball_characteristic(q, .4)
    expected = {name: np.zeros(len(q)) for name in ('host', 'host_guest', 'guest_guest')}
    for a in range(len(host_xyz)):
        for b in range(len(host_xyz)):
            expected['host'] += host_f[a]*host_f[b]*np.sinc(q*np.linalg.norm(host_xyz[a]-host_xyz[b])/np.pi)
        for b in range(len(pore_xyz)):
            expected['host_guest'] += 2*host_f[a]*pore_f[b]*np.sinc(q*np.linalg.norm(host_xyz[a]-pore_xyz[b])/np.pi)
    for a in range(len(pore_xyz)):
        for b in range(a):
            expected['guest_guest'] += 2*pore_f[a]*pore_f[b]*np.sinc(q*np.linalg.norm(pore_xyz[a]-pore_xyz[b])/np.pi)
    expected['guest_internal'] = 2*probabilities.sum()*internal
    expected['total'] = sum(expected.values())
    for name in expected:
        np.testing.assert_allclose(actual[name], expected[name], rtol=3e-9, atol=1e-7)


def test_active_monatomic_wall_matches_exact_conditioned_ball_and_intensity_sem():
    q = np.linspace(0, 6, 121)
    samples = 2048
    population = _oracle.synthetic_population(spacing_A=12., occupancy=.37, dimer_weight=0.,
                                               radius_A=4., wall_radius_A=2.)
    stats = radial_guest_statistics(population, q, method='sampled',
                                   n_samples=samples, n_batches=16, seed=20261013)
    factor = form_factors(('Au',), q)['Au']
    expected_mean = factor*_oracle.ball_characteristic(q, 2.)
    # Exact variance of one radial sample, integrated independently over the
    # accepted uniform ball. There are two independent replicas of samples.
    nodes, weights = np.polynomial.legendre.leggauss(96)
    radii = nodes+1
    probability_weights = weights*3*radii*radii/8
    kernel = np.sinc(q[:, None]*radii[None, :]/np.pi)
    radial_mean = kernel@probability_weights
    variance = np.maximum((kernel*kernel)@probability_weights-radial_mean*radial_mean, 0.)
    exact_sem = abs(factor)*np.sqrt(variance/(2*samples))
    sample_mean = stats.means[:, :, 0].mean(axis=(0, 1))
    standardized = abs(sample_mean[1:]-expected_mean[1:])/exact_sem[1:]
    assert standardized.max() < 5., standardized.max()
    np.testing.assert_allclose(stats.internal[0], factor*factor, rtol=2e-14)
    result = PorePowderCache(AtomCloud(('C',), [[0., 0., 0.]]), np.eye(3)*12,
                            np.ones((2, 2, 2), dtype=int), population, q, threads=1).iq(stats)
    reference = _oracle.body_centered_pore_reference(q, 2, spacing_A=12., occupancy=.37,
                                                    dimer_weight=0., radius_A=2.)
    allowance = 5*result['total_mc_sem']+result['approximation_bound_excluding_roundoff']+1e-8
    assert np.all(abs(result['total']-reference['total']) <= allowance)
    assert np.any(result['total_mc_sem'][1:] > 0)
    np.testing.assert_allclose(result['total'][0], reference['total'][0], rtol=2e-13)


def test_active_dimer_wall_requires_coupled_lens_average():
    q = np.linspace(0, 6, 121)
    dimer = AtomCloud(('Au', 'Au'), [[-.5, 0., 0.], [.5, 0., 0.]])
    population = PorePopulation([[0., 0., 0.]], ParticleLibrary((dimer,), [1.]),
        occupancy=.4, position_mode='uniform_ball', uniform_ball_radius_A=2.,
        pore_radius_A=2., orientation='uniform')
    stats = radial_guest_statistics(population, q, method='sampled',
                                   n_samples=2048, n_batches=16, seed=20261014)
    expected = _oracle.confined_dimer_lens_mean(q)
    batch_replica = stats.means[:, :, 0].reshape(-1, len(q))
    sem = batch_replica.std(axis=0, ddof=1)/np.sqrt(len(batch_replica))
    discrepancy = abs(batch_replica.mean(axis=0)-expected)
    assert np.all(discrepancy[1:] <= 5*sem[1:]+1e-9)
    factor = form_factors(('Au',), q)['Au']
    expected_internal = 2*factor*factor*(1+np.sinc(q/np.pi))
    np.testing.assert_allclose(stats.internal[0], expected_internal, rtol=2e-13)
    separable_wrong = 2*factor*np.sinc(q*.5/np.pi)*_oracle.ball_characteristic(q, 2.)
    assert np.max(abs(expected-separable_wrong)) > 1., 'This fixture must distinguish coupled rejection.'


def test_103nm_guest_distribution_matches_integer_shell_oracle():
    q = np.linspace(0, 6, 6001)
    population = _oracle.synthetic_population()
    stats = radial_guest_statistics(population, q, method='analytic')
    actual = PorePowderCache(AtomCloud(('C',), [[0., 0., 0.]]), np.eye(3)*25.832,
                            np.ones((40, 40, 40), dtype=int), population, q, threads=1).iq(stats)
    expected = _oracle.body_centered_pore_reference(q, 40)
    metrics = _oracle.compare_powder_curves(q, actual['total'], expected['total'])
    assert metrics['intensity_checks_passed'], metrics
    assert metrics['max_peak_q_shift_A_inv'] <= .00100000001
    assert metrics['forward_relative_error'] < 2e-12
    for name in ('host', 'guest_internal', 'host_guest', 'guest_guest'):
        assert np.max(abs(actual[name][q >= .1]-expected[name][q >= .1]))/metrics['nonforward_peak_scale'] < 1e-8, name
