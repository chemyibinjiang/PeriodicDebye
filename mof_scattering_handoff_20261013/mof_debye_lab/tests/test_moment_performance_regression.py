"""Accuracy guards for reusable rigid-template amplitude preparation."""
from copy import copy

import numpy as np
import pytest

from mofdebye.core import AtomCloud
from mofdebye.directional import amplitude
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.moments import _SampledAmplitudePlan, guest_moments


def _population():
    rng = np.random.default_rng(29)
    small = AtomCloud(('Au', 'O', 'C'), rng.uniform(-0.3, 0.3, (3, 3)))
    large = AtomCloud(tuple(('C', 'Au', 'O')[i % 3] for i in range(263)),
                      rng.uniform(-0.3, 0.3, (263, 3)))
    return PorePopulation([[0, 0, 0], [5, 0, 0]],
                          ParticleLibrary((small, large), [0.4, 0.6]),
                          occupancy=[0.35, 0.9], displacement_sigma_A=0.4,
                          pore_radius_A=1.0, orientation='wobble',
                          wobble_std_deg=31)


def _explicit_pore_statistics(population, vectors, count, seed):
    """Keep configurations and use two-pass statistics as a small oracle."""
    rng = np.random.default_rng(seed)
    expected_mean = np.zeros(len(vectors), dtype=complex)
    expected_variance = np.zeros(len(vectors))
    occupied_counts = []
    template_counts = np.zeros(len(population.library.templates), dtype=int)
    attempts = occupied = 0
    for pore in range(len(population.centres)):
        local = copy(population)
        local.centres = population.centres[pore:pore + 1]
        local.occupancy = population.occupancy[pore:pore + 1]
        local.mean_displacement = population.mean_displacement[pore:pore + 1]
        if population.pore_radius is not None:
            local.pore_radius = population.pore_radius[pore:pore + 1]
        values = []
        current_occupied = 0
        for _ in range(count):
            draws = local.sample(rng)
            values.append(amplitude(draws[0].cloud, vectors) if draws
                          else np.zeros(len(vectors), dtype=complex))
            if draws:
                current_occupied += 1
                template_counts[draws[0].template_index] += 1
                attempts += draws[0].attempts
        occupied_counts.append(current_occupied)
        occupied += current_occupied
        values = np.asarray(values)
        mean = values.mean(axis=0)
        expected_mean += mean
        expected_variance += np.sum(np.abs(values - mean)**2, axis=0) / (count - 1)
    return (expected_mean, expected_variance, occupied_counts, template_counts,
            attempts / occupied if occupied else 0.0)


def test_sampled_cached_templates_match_explicit_conditioned_states():
    population = _population()
    before = population.centres.copy()
    vectors = np.random.default_rng(30).normal(size=(265, 3))
    vectors[0] = 0
    result = guest_moments(population, vectors, method='sampled', n_samples=31, seed=41)
    mean, variance, occupied, counts, attempts = _explicit_pore_statistics(
        population, vectors, 31, 41)
    np.testing.assert_allclose(result.mean, mean, rtol=2e-14, atol=2e-11)
    np.testing.assert_allclose(result.variance, variance, rtol=2e-14, atol=2e-8)
    np.testing.assert_allclose(result.mean_variance, variance / 31, rtol=2e-14, atol=1e-9)
    assert result.metadata['occupancy_counts'] == occupied
    assert result.metadata['template_counts'] == counts.tolist()
    assert result.metadata['mean_placement_attempts'] == attempts
    assert attempts > 1  # Conditioned placement was actually exercised.
    assert result.metadata['cached_form_factor_values'] == 3 * len(vectors)
    np.testing.assert_array_equal(population.centres, before)


def test_cached_phase_tiles_preserve_reference_amplitude_exactly():
    population = _population()
    population.occupancy[:] = 1
    vectors = np.random.default_rng(31).normal(size=(521, 3))
    plan = _SampledAmplitudePlan(population.library.templates, vectors)
    rng = np.random.default_rng(32)
    seen = set()
    for _ in range(12):
        for draw in population.sample(rng):
            seen.add(draw.template_index)
            np.testing.assert_array_equal(plan.evaluate(draw), amplitude(draw.cloud, vectors))
    assert seen == {0, 1}


@pytest.mark.parametrize('q_block', [1, 13, 257])
def test_sampled_q_blocking_preserves_draws_and_statistics(q_block):
    population = _population()
    vectors = np.random.default_rng(33).normal(size=(261, 3))
    whole = guest_moments(population, vectors, method='sampled', n_samples=7, seed=34)
    chunks = [guest_moments(population, vectors[start:start + q_block],
                            method='sampled', n_samples=7, seed=34)
              for start in range(0, len(vectors), q_block)]
    for name in ('mean', 'variance', 'mean_variance'):
        np.testing.assert_allclose(np.concatenate([getattr(part, name) for part in chunks]),
                                   getattr(whole, name), rtol=2e-13, atol=1e-8)
    assert all(part.metadata['template_counts'] == whole.metadata['template_counts']
               for part in chunks)


def test_unused_templates_and_vacant_populations_do_not_prepare_factors(monkeypatch):
    import mofdebye.moments as module
    old_factors = module.form_factors
    calls = []

    def recorded(elements, q):
        calls.append(set(elements))
        return old_factors(elements, q)

    monkeypatch.setattr(module, 'form_factors', recorded)
    library = ParticleLibrary((AtomCloud(('Au',), [[0, 0, 0]]),
                               AtomCloud(('O',), [[0, 0, 0]])), [1, 0])
    population = PorePopulation([[0, 0, 0]], library, occupancy=0)
    vectors = np.array([[0, 0, 0], [1, 0, 0]])
    empty = guest_moments(population, vectors, method='sampled', n_samples=5, seed=35)
    assert not calls
    assert empty.metadata['cached_form_factor_values'] == 0
    assert not empty.mean.any() and not empty.variance.any()
    population.occupancy[:] = 1
    occupied = guest_moments(population, vectors, method='sampled', n_samples=5, seed=35)
    assert calls == [{'Au'}]
    assert occupied.metadata['cached_form_factor_values'] == len(vectors)
    # A new call observes population/library changes instead of a stale cache.
    population.library = ParticleLibrary(library.templates, [0, 1])
    changed = guest_moments(population, vectors, method='sampled', n_samples=5, seed=35)
    assert calls == [{'Au'}, {'O'}]
    np.testing.assert_array_equal(changed.mean, amplitude(library.templates[1], vectors))
