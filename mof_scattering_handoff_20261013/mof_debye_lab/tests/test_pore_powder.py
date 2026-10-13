"""Geometry, replica statistics and error propagation for finite pore powder."""
from dataclasses import replace
from itertools import combinations, product

import numpy as np
import pytest

pytest.importorskip('numba')

from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.pore_powder import PorePowderCache
from mofdebye.radial_guests import radial_guest_statistics


def _model(occupancy=(.35, .8), mask=None, **radial_options):
    host = AtomCloud(('C', 'O'), [[0., 0., 0.], [1.1, .4, .2]])
    library = ParticleLibrary((AtomCloud(('Au',), [[0., 0., 0.]]),), [1.])
    population = PorePopulation([[2., 2., 2.], [7., 6., 5.]], library,
                                 occupancy=occupancy, position_mode='uniform_ball',
                                 uniform_ball_radius_A=.7, pore_radius_A=1.1)
    cell = np.eye(3) * 10
    mask = np.ones((2, 1, 1), bool) if mask is None else mask
    q = np.linspace(0, 6, 121)
    stats = radial_guest_statistics(population, q)
    cache = PorePowderCache(host, cell, mask, population, q, **radial_options)
    return host, population, cell, mask, q, stats, cache


def _explicit_replica_reference(host, population, cell, mask, q, statistics):
    """Enumerate physical atoms/sites; never use cell-offset or self subtraction."""
    translations = np.argwhere(mask) @ cell
    full_host = concatenate([host.moved(vector) for vector in translations])
    host_ff = form_factors(full_host.elements, q)
    pores = [(centre + vector, index) for vector in translations
             for index, centre in enumerate(statistics.centres)]
    mean = statistics.means
    batches = len(mean)
    host_part = direct_iq(full_host, q)
    internal = len(translations) * np.sum(population.occupancy[:, None] * statistics.internal, axis=0)
    values = {name: np.zeros((batches, len(q)))
              for name in ('host', 'guest_internal', 'host_guest', 'guest_guest')}
    values['host'][:] = host_part
    values['guest_internal'][:] = internal
    for batch in range(batches):
        for centre, index in pores:
            b = .5 * (mean[batch, 0, index] + mean[batch, 1, index])
            for element, atom in zip(full_host.elements, full_host.xyz):
                values['host_guest'][batch] += (
                    2 * population.occupancy[index] * b * host_ff[element]
                    * np.sinc(q * np.linalg.norm(atom - centre) / np.pi))
        for (ca, ia), (cb, ib) in combinations(pores, 2):
            amplitude = (mean[batch, 0, ia] * mean[batch, 1, ib]
                         + mean[batch, 1, ia] * mean[batch, 0, ib])
            values['guest_guest'][batch] += (
                population.occupancy[ia] * population.occupancy[ib] * amplitude
                * np.sinc(q * np.linalg.norm(ca - cb) / np.pi))
    values['total'] = sum(values.values())
    return values


def test_every_component_matches_explicit_physical_site_pairs():
    host, population, cell, mask, q, stats, cache = _model()
    result = cache.iq(stats)
    expected = _explicit_replica_reference(host, population, cell, mask, q, stats)
    for name, values in expected.items():
        np.testing.assert_allclose(result[name], values.mean(axis=0), rtol=3e-12, atol=2e-8)
        np.testing.assert_array_equal(result['mc_sem'][name], 0.)
    np.testing.assert_allclose(result['total'], sum(result[name] for name in expected if name != 'total'))
    assert 'represented_atoms' not in result['metadata']
    assert result['metadata']['physical_host_atoms'] == len(host) * mask.sum()
    assert result['metadata']['geometry_scattering_sites_per_cell'] == len(host) + 2
    assert result['metadata']['expected_guest_atoms'] == pytest.approx(mask.sum() * sum(population.occupancy))
    assert any(np.any(result[name] < 0) for name in ('host_guest', 'guest_guest'))


def test_two_independent_replicas_remove_reused_mean_squared_bias():
    host, population, cell, mask, q, stats, cache = _model()
    exact = cache.iq(stats)
    means = np.repeat(stats.means, 4, axis=0)
    delta = .2 * stats.means[0, 0] * np.sin(q)[None, :]
    # Four equally weighted Cartesian combinations enumerate independent signs.
    for batch, (a_sign, b_sign) in enumerate(product((-1, 1), repeat=2)):
        means[batch, 0] += a_sign * delta
        means[batch, 1] += b_sign * delta
    sampled = replace(stats, means=means, mean_bounds=np.zeros_like(means),
                       metadata={**stats.metadata, 'method': 'sampled', 'purpose': 'algebraic independent-replica check'})
    actual = cache.iq(sampled)
    reference = _explicit_replica_reference(host, population, cell, mask, q, sampled)
    for name, patterns in reference.items():
        np.testing.assert_allclose(actual[name], patterns.mean(axis=0), rtol=3e-12, atol=2e-8)
        np.testing.assert_allclose(actual[name], exact[name], rtol=3e-12, atol=2e-8)
        expected_sem = patterns.std(axis=0, ddof=1) / 2
        np.testing.assert_allclose(actual['mc_sem'][name], expected_sem, rtol=3e-11, atol=2e-9)
    assert np.max(actual['total_mc_sem']) > 1
    # Deliberately correlated replicas demonstrate the positive square bias.
    wrong = means.copy()
    wrong[:, 1] = wrong[:, 0]
    biased = cache.iq(replace(sampled, means=wrong))
    assert np.max(np.abs(biased['guest_guest'] - exact['guest_guest'])) > 1


def test_local_mean_and_internal_bounds_propagate_through_all_four_parts():
    host, population, cell, mask, q, stats, cache = _model(bin_width=1., order=4)
    error = np.broadcast_to(.02 * np.cos(q), stats.means.shape).copy()
    internal_error = np.broadcast_to(.3 * np.cos(q), stats.internal.shape).copy()
    approximate = replace(stats, means=stats.means + error, mean_bounds=np.abs(error),
                          internal=stats.internal + internal_error,
                          internal_bounds=np.abs(internal_error))
    actual = cache.iq(approximate)
    reference = _explicit_replica_reference(host, population, cell, mask, q, stats)
    for name in ('host', 'guest_internal', 'host_guest', 'guest_guest'):
        observed = np.abs(actual[name] - reference[name][0])
        bound = actual['component_approximation_bounds'][name]
        assert np.all(observed <= bound + 3e-8)
    assert np.all(np.abs(actual['total'] - reference['total'][0])
                  <= actual['approximation_bound_excluding_roundoff'] + 3e-8)


@pytest.mark.parametrize('occupancy', [(0., 0.), (1., 1.), (.17, .83)])
def test_forward_limit_uses_local_occupancy_variance_not_repeated_random_cell(occupancy):
    host, population, cell, mask, q, stats, cache = _model(occupancy=occupancy,
                                                        mask=np.ones((3, 1, 1), bool))
    result = cache.iq(stats)
    n = int(mask.sum())
    ff = form_factors(host.elements, [0.])
    h = n * sum(ff[element][0] for element in host.elements)
    b = stats.means[0, 0, :, 0]
    p = population.occupancy
    expected = (h + n * np.sum(p*b))**2 + n * np.sum(p*stats.internal[:, 0] - (p*b)**2)
    assert result['total'][0] == pytest.approx(expected, rel=3e-15)
    assert result['guest_internal'][0] == pytest.approx(n * np.sum(p * stats.internal[:, 0]))
    assert result['approximation_bound_excluding_roundoff'][0] == 0
    if sum(occupancy) == 0:
        for name in ('guest_internal', 'host_guest', 'guest_guest'):
            np.testing.assert_array_equal(result[name], 0.)


def test_same_physical_pore_diagonal_is_removed_but_other_cells_are_retained():
    host = AtomCloud((), np.empty((0, 3)))
    library = ParticleLibrary((AtomCloud(('Au',), [[0, 0, 0]]),), [1])
    population = PorePopulation([[2., 2., 2.]], library, occupancy=.4)
    q = np.linspace(0, 6, 121)
    stats = radial_guest_statistics(population, q)
    one = PorePowderCache(host, np.eye(3)*10, np.ones((1, 1, 1)), population, q).iq(stats)
    np.testing.assert_allclose(one['guest_guest'], 0., rtol=0, atol=1e-12)
    two = PorePowderCache(host, np.eye(3)*10, np.ones((2, 1, 1)), population, q).iq(stats)
    ff = form_factors(('Au',), q)['Au']
    expected = 2 * .4**2 * ff**2 * np.sinc(q*10/np.pi)
    np.testing.assert_allclose(two['guest_guest'], expected, rtol=3e-12, atol=3e-9)
    np.testing.assert_array_equal(two['host'], 0.)


def test_periodically_equivalent_pore_sites_are_rejected_without_wall_radii():
    library = ParticleLibrary((AtomCloud(('Au',), [[0, 0, 0]]),), [1])
    population = PorePopulation([[1, 2, 3], [11, 2, 3]], library)
    with pytest.raises(ValueError, match='periodically equivalent'):
        PorePowderCache(AtomCloud(('C',), [[0, 0, 0]]), np.eye(3)*10,
                       np.ones((1, 1, 1)), population, np.linspace(0, 6, 121))


@pytest.mark.parametrize('case', ['other_image', 'self_image', 'skew_self_image'])
def test_periodic_hard_wall_pore_overlap_is_rejected(case):
    library = ParticleLibrary((AtomCloud(('Au',), [[0, 0, 0]]),), [1])
    cell = np.eye(3) * 10
    centres, radius = [[.1, 2, 3], [9.9, 2, 3]], .2
    if case == 'self_image':
        cell = np.eye(3) * 3
        centres, radius = [[0, 0, 0]], 2.
    elif case == 'skew_self_image':
        cell = np.array([[5., 0, 0], [4.9, .1, 0], [0, 0, 6.]])
        centres, radius = [[0, 0, 0]], .1
    population = PorePopulation(centres, library, pore_radius_A=radius)
    with pytest.raises(ValueError, match='overlap under a lattice translation'):
        PorePowderCache(AtomCloud(('C',), [[0, 0, 0]]), cell,
                       np.ones((1, 1, 1)), population, np.linspace(0, 6, 121))


@pytest.mark.parametrize('invalid', ['q', 'centres', 'occupancy', 'shape', 'bounds_shape',
                                    'nan', 'negative_bound', 'internal_shape', 'internal_bound',
                                    'method', 'one_sample_batch', 'different_analytic_replicas', 'complex',
                                    'negative_internal'])
def test_mismatched_or_invalid_statistics_are_rejected(invalid):
    _, _, _, _, _, stats, cache = _model()
    changes = {}
    if invalid == 'q':
        changes['q'] = stats.q + .01
    elif invalid == 'centres':
        changes['centres'] = stats.centres + 1e-9
    elif invalid == 'occupancy':
        changes['occupancy'] = stats.occupancy * .9
    elif invalid == 'shape':
        changes['means'] = stats.means[0]
    elif invalid == 'bounds_shape':
        changes['mean_bounds'] = stats.mean_bounds[0]
    elif invalid == 'nan':
        changes['means'] = stats.means * np.nan
    elif invalid == 'negative_bound':
        changes['mean_bounds'] = stats.mean_bounds - 1
    elif invalid == 'internal_shape':
        changes['internal'] = stats.internal[0]
    elif invalid == 'internal_bound':
        changes['internal_bounds'] = stats.internal_bounds - 1
    elif invalid == 'negative_internal':
        changes['internal'] = -stats.internal
    elif invalid == 'method':
        changes['metadata'] = {'method': 'unknown'}
    elif invalid == 'one_sample_batch':
        changes['metadata'] = {'method': 'sampled'}
    elif invalid == 'complex':
        changes['means'] = stats.means.astype(complex) + 1j
    else:
        means = stats.means.copy()
        means[:, 1] += 1
        changes['means'] = means
    with pytest.raises(ValueError):
        cache.iq(replace(stats, **changes))


def test_population_geometry_and_occupancy_are_copied():
    _, population, _, _, _, stats, cache = _model()
    before = cache.iq(stats)
    population.centres[:] = 99
    population.mean_displacement[:] = 99
    population.occupancy[:] = 0
    after = cache.iq(stats)
    for name in ('host', 'guest_internal', 'host_guest', 'guest_guest', 'total', 'total_mc_sem'):
        np.testing.assert_array_equal(before[name], after[name])
    assert cache.metadata['evaluation_count'] == 2


@pytest.mark.parametrize('occupancy', [-.1, 1.1, np.nan])
def test_invalid_mutated_population_occupancy_is_rejected(occupancy):
    host, population, cell, mask, q, _, _ = _model()
    population.occupancy[:] = occupancy
    with pytest.raises(ValueError, match='occupancy'):
        PorePowderCache(host, cell, mask, population, q)


def test_size_catalogue_reuse_updates_atom_count_metadata_without_rebuilding_geometry():
    host, population, cell, mask, q, stats, cache = _model()
    before = cache.iq(stats)
    geometry = cache._cache
    monomer = AtomCloud(('Au',), [[0., 0., 0.]])
    dimer = AtomCloud(('Au', 'Au'), [[-.2, 0., 0.], [.2, 0., 0.]])
    revised_population = PorePopulation(
        population.centres, ParticleLibrary((monomer, dimer), [.25, .75]),
        occupancy=population.occupancy, position_mode='uniform_ball',
        uniform_ball_radius_A=.7, pore_radius_A=1.1)
    revised_stats = radial_guest_statistics(revised_population, q)
    assert revised_stats.metadata['conditional_mean_atom_count'] == 1.75
    actual = cache.iq(revised_stats)
    rebuilt = PorePowderCache(host, cell, mask, revised_population, q).iq(revised_stats)
    assert cache._cache is geometry
    for name in ('host', 'guest_internal', 'host_guest', 'guest_guest', 'total'):
        np.testing.assert_array_equal(actual[name], rebuilt[name])
    expected_particles = int(mask.sum()) * population.occupancy.sum()
    assert before['metadata']['expected_guest_atoms'] == pytest.approx(expected_particles)
    assert actual['metadata']['expected_guest_atoms'] == pytest.approx(expected_particles * 1.75)
    assert actual['metadata']['guest_atom_count_source'] == 'current_guest_statistics'
    unknown_metadata = {key: value for key, value in revised_stats.metadata.items()
                        if key != 'conditional_mean_atom_count'}
    unknown = cache.iq(replace(revised_stats, metadata=unknown_metadata))
    assert unknown['metadata']['expected_guest_atoms'] is None
    assert unknown['metadata']['guest_atom_count_source'] == 'unavailable_in_supplied_statistics'
    np.testing.assert_array_equal(unknown['total'], actual['total'])


@pytest.mark.parametrize('count', [-1, np.nan, True, '2', [2]])
def test_invalid_conditional_mean_atom_count_metadata_is_rejected(count):
    _, _, _, _, _, stats, cache = _model()
    metadata = {**stats.metadata, 'conditional_mean_atom_count': count}
    with pytest.raises(ValueError, match='Conditional mean atom count'):
        cache.iq(replace(stats, metadata=metadata))
