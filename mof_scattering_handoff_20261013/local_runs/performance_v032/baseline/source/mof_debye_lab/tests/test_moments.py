from itertools import product

import numpy as np
import pytest

from mofdebye.core import AtomCloud, concatenate, form_factors
from mofdebye.directional import amplitude, intensity, powder_average
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.ensemble import analytic_ensemble
from mofdebye.moments import (GuestMoments, exact_guest_moments,
                             finite_ensemble_intensity, guest_moments)


def empty():
    return AtomCloud((), np.empty((0, 3)))


def test_single_particle_occupancy_is_p_not_p_squared():
    guest = AtomCloud(('Au',), [[0, 0, 0]])
    vectors = np.array([[0, 0, 0], [0.3, -0.5, 0.7], [1.4, 0.2, 0]])
    probability = 0.3
    pop = PorePopulation([[2, -1, 0.5]], ParticleLibrary((guest,), [1]),
                         occupancy=probability)
    moments = guest_moments(pop, vectors)
    f = form_factors(('Au',), np.linalg.norm(vectors, axis=1))['Au']
    np.testing.assert_allclose(moments.mean,
                               probability * amplitude(guest.moved(pop.centres[0]), vectors))
    np.testing.assert_allclose(moments.variance, probability * (1 - probability) * f**2)
    result = finite_ensemble_intensity(empty(), vectors, moments)
    np.testing.assert_allclose(result['total'], probability * f**2)


def test_exact_states_match_full_host_guest_configuration_enumeration():
    host = AtomCloud(('C', 'O'), [[0, 1, 0], [2.1, -0.3, 0.5]])
    states = [empty(), AtomCloud(('Au',), [[3, 0, 1]]),
              AtomCloud(('Au', 'C'), [[3, 1, 0], [-1, 0, 0.4]])]
    probabilities = np.array([0.2, 0.35, 0.45])
    vectors = np.array([[0, 0, 0], [0.3, 0.4, -0.1], [1.2, -1, 0.7]])
    moments = exact_guest_moments(states, probabilities, vectors)
    result = finite_ensemble_intensity(host, vectors, moments)
    expected = sum(p * intensity(concatenate([host, state]), vectors)
                   for p, state in zip(probabilities, states))
    np.testing.assert_allclose(result['total'], expected, rtol=1e-13, atol=1e-10)
    np.testing.assert_allclose(result['total'], result['host'] + result['host_guest']
                               + result['guest_coherent'] + result['diffuse'], rtol=1e-13)
    assert not moments.mean_variance.any()


def test_correlated_occupancies_differ_and_require_joint_states():
    particle = AtomCloud(('Au',), [[0, 0, 0]])
    vectors = np.zeros((1, 3))
    pop = PorePopulation([[0, 0, 0], [3, 0, 0]], ParticleLibrary((particle,), [1]),
                         occupancy=0.5)
    independent = guest_moments(pop, vectors)
    correlated = exact_guest_moments([empty(), concatenate([particle, particle.moved([3, 0, 0])])],
                                     [0.5, 0.5], vectors)
    np.testing.assert_allclose(correlated.mean, independent.mean)
    np.testing.assert_allclose(correlated.variance, 2 * independent.variance)
    with pytest.raises(ValueError, match='Correlated pores'):
        guest_moments(pop, vectors, independent_pores=False)


def test_analytic_directional_powder_average_recovers_baseline():
    host = AtomCloud(('C', 'O', 'Zn'), [[0, 0, 0], [0.4, 1.1, -0.2], [-1, 0.3, 0.4]])
    small = AtomCloud(('Au', 'C'), [[0.2, 0, 0], [0, 0.6, -0.1]])
    large = AtomCloud(('Au', 'Au', 'O'), [[0.4, 0.1, 0], [-0.5, 0.2, 0.2], [0.1, -0.3, 0.7]])
    pop = PorePopulation([[2.5, 0.3, 0.2], [-1.5, 2, 0.5]],
                         ParticleLibrary((small, large), [0.7, 0.3]),
                         occupancy=[0.3, 0.8], displacement_sigma_A=0.4,
                         mean_displacement_A=[[0.1, 0.2, 0], [-0.2, 0, 0.3]])
    q = np.array([0, 0.4, 1.3, 2.1])
    evaluate = lambda vectors: finite_ensemble_intensity(host, vectors, guest_moments(pop, vectors))['total']
    expected = analytic_ensemble(host, pop, q)['total']
    first = powder_average(evaluate, q, n_mu=20, n_phi=40)
    second = powder_average(evaluate, q, n_mu=32, n_phi=64)
    np.testing.assert_allclose(first, second, rtol=1e-11, atol=1e-8)
    np.testing.assert_allclose(second, expected, rtol=1e-11, atol=1e-8)


def test_sampled_unbounded_anisotropic_fixed_orientation_with_mean_sem():
    particle = AtomCloud(('Au',), [[0.4, -0.3, 0.2]])
    covariance = np.array([[0.3, 0.1, 0], [0.1, 0.6, 0], [0, 0, 0.2]])
    pop = PorePopulation([[2, 0.3, -1]], ParticleLibrary((particle,), [1]),
                         occupancy=0.6, displacement_covariance_A2=covariance,
                         mean_displacement_A=[0.2, -0.1, 0.4], orientation='fixed')
    vectors = np.array([[0, 0, 0], [0.3, 0.8, -0.2], [1.1, -0.4, 0.5]])
    result = guest_moments(pop, vectors, method='sampled', n_samples=4096, seed=781)
    f = form_factors(('Au',), np.linalg.norm(vectors, axis=1))['Au']
    centre = pop.centres[0] + pop.mean_displacement[0] + particle.xyz[0]
    exact_mean = 0.6 * f * np.exp(1j * (vectors @ centre)) * np.exp(
        -0.5 * np.einsum('qi,ij,qj->q', vectors, covariance, vectors))
    exact_variance = 0.6 * f**2 - np.abs(exact_mean)**2
    assert np.max(np.abs(result.mean - exact_mean) / np.sqrt(result.mean_variance)) < 4
    np.testing.assert_allclose(result.variance, exact_variance, rtol=0.05)
    np.testing.assert_allclose(result.mean_variance, result.variance / 4096)
    with pytest.raises(ValueError):
        guest_moments(pop, vectors, method='analytic')


def test_sampled_conditioned_distribution_matches_actual_configuration_average():
    host = AtomCloud(('C',), [[-3, 0.1, 0]])
    small = AtomCloud(('Au',), [[0, 0, 0]])
    large = AtomCloud(('Au', 'Au'), [[-0.6, 0, 0], [0.6, 0, 0]])
    pop = PorePopulation([[0, 0, 0]], ParticleLibrary((small, large), [0.4, 0.6]),
                         occupancy=0.8, displacement_sigma_A=0.5, pore_radius_A=1.1,
                         mean_displacement_A=[0.2, 0, 0], orientation='wobble',
                         wobble_std_deg=25, host=host, min_host_distance_A=0.5)
    vectors = np.array([[0, 0, 0], [0.7, -0.4, 0.1], [1.2, 0.3, -0.5]])
    count = 512
    moments = guest_moments(pop, vectors, method='sampled', n_samples=count, seed=309)
    result = finite_ensemble_intensity(host, vectors, moments)
    rng = np.random.default_rng(309)
    explicit = np.zeros(len(vectors))
    for _ in range(count):
        draws = pop.sample(rng)
        for draw in draws:
            assert np.linalg.norm(draw.cloud.xyz, axis=1).max() <= 1.1 + 1e-12
        explicit += intensity(concatenate([host] + [d.cloud for d in draws]), vectors) / count
    # For a single pore, the unbiased moment identity equals the arithmetic
    # sample intensity exactly, even when rejection couples placement variables.
    np.testing.assert_allclose(result['total'], explicit, rtol=2e-13, atol=1e-8)
    frequencies = np.asarray(moments.metadata['template_counts'], dtype=float)
    np.testing.assert_allclose(frequencies / frequencies.sum(), [0.4, 0.6], atol=0.08)
    assert moments.metadata['mean_placement_attempts'] > 1
    with pytest.raises(ValueError, match='Hard-wall|Haar-uniform'):
        guest_moments(pop, vectors)


def test_sampled_impossible_template_is_not_removed_or_reweighted():
    impossible = AtomCloud(('Au',), [[2, 0, 0]])
    pop = PorePopulation([[0, 0, 0]], ParticleLibrary((impossible,), [1]),
                         orientation='fixed', pore_radius_A=1, max_attempts=2)
    with pytest.raises(RuntimeError, match='No valid placement'):
        guest_moments(pop, [[0, 0, 0]], method='sampled', n_samples=2)


def test_sample_mean_squared_bias_correction_by_exact_sequence_enumeration():
    host = empty()
    vectors = np.zeros((1, 3))
    raw = corrected = total = 0.0
    samples = 4
    for sequence in product((0.0, 1.0), repeat=samples):
        values = np.asarray(sequence)
        variance = values.var(ddof=1)
        moments = GuestMoments([values.mean()], [variance], [variance / samples], {'method': 'sampled'})
        result = finite_ensemble_intensity(host, vectors, moments)
        raw += result['coherent_raw'][0] / 2**samples
        corrected += result['coherent'][0] / 2**samples
        total += result['total'][0] / 2**samples
    assert raw == pytest.approx(0.25 + 0.25 / samples)
    assert corrected == pytest.approx(0.25)
    assert total == pytest.approx(0.5)


def test_sampled_deterministic_and_empty_limits_and_seed_reproducibility():
    host = AtomCloud(('C',), [[1, 2, 0]])
    particle = AtomCloud(('Au', 'O'), [[-0.3, 0, 0.1], [0.2, 0.7, 0]])
    vectors = np.array([[0, 0, 0], [0.3, 0.1, -0.5]])
    pop = PorePopulation([[2, 0, 1]], ParticleLibrary((particle,), [1]), orientation='fixed')
    result = guest_moments(pop, vectors, method='sampled', n_samples=8, seed=1)
    assert not result.variance.any()
    np.testing.assert_allclose(finite_ensemble_intensity(host, vectors, result)['total'],
                               intensity(concatenate([host, particle.moved(pop.centres[0])]), vectors))
    no_guests = guest_moments(None, vectors)
    np.testing.assert_allclose(finite_ensemble_intensity(host, vectors, no_guests)['total'],
                               intensity(host, vectors))
    pop.occupancy[:] = 0.4
    a = guest_moments(pop, vectors, method='sampled', n_samples=50, seed=28)
    b = guest_moments(pop, vectors, method='sampled', n_samples=50, seed=28)
    np.testing.assert_array_equal(a.mean, b.mean)
    np.testing.assert_array_equal(a.variance, b.variance)


@pytest.mark.parametrize('mean, variance, mean_variance', [
    ([], [], []), ([np.nan], [0], [0]), ([0], [-1], [0]),
    ([0], [0], [-1]), ([0], [0, 0], [0]), ([0], [0], [np.inf]),
])
def test_moment_arrays_reject_invalid_values(mean, variance, mean_variance):
    with pytest.raises(ValueError):
        GuestMoments(mean, variance, mean_variance, {})


def test_moment_input_and_probability_validation():
    vectors = np.zeros((1, 3))
    particle = AtomCloud(('Au',), [[0, 0, 0]])
    pop = PorePopulation([[0, 0, 0]], ParticleLibrary((particle,), [1]))
    for count in (0, 1, 2.5, True):
        with pytest.raises(ValueError):
            guest_moments(pop, vectors, method='sampled', n_samples=count)
    for probabilities in ([0.2], [np.nan], [-1], [1, 0]):
        with pytest.raises(ValueError):
            exact_guest_moments([particle], probabilities, vectors)
    with pytest.raises(ValueError, match='method'):
        guest_moments(pop, vectors, method='unbounded-approximation')
    with pytest.raises(ValueError, match='q-vector count'):
        finite_ensemble_intensity(particle, vectors, GuestMoments([0, 0], [0, 0], [0, 0], {}))
