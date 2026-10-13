import numpy as np
import pytest
import torch

import mofdebye.core as core
import mofdebye.ensemble as ensemble
import mofdebye.lattice as lattice
from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors, upstream_kernel_iq
from mofdebye.distributions import ParticleLibrary, PorePopulation


def dense_reference(cloud, q):
    factors = form_factors(cloud.elements, q)
    per_atom = np.stack([factors[el] for el in cloud.elements])
    distances = np.linalg.norm(cloud.xyz[:, None, :] - cloud.xyz[None, :, :], axis=-1)
    return np.sum(per_atom[:, None, :] * per_atom[None, :, :]
                  * np.sinc(distances[:, :, None] * q[None, None, :] / np.pi), axis=(0, 1))


def test_reference_distance_and_sinc_allocations_are_bounded(monkeypatch):
    rng = np.random.default_rng(22)
    cloud = AtomCloud(tuple(['C'] * 19 + ['O'] * 11 + ['Au'] * 7), rng.normal(size=(37, 3)))
    q = np.linspace(0., 10., 101)
    expected = dense_reference(cloud, q)
    calls = []
    original_pdist, original_cdist, original_sinc = core.pdist, core.cdist, core.np.sinc
    def pdist(x):
        calls.append(('pdist', len(x)))
        assert len(x) <= 4
        return original_pdist(x)
    def cdist(x, y):
        calls.append(('cdist', len(x), len(y)))
        assert len(x) <= 4 and len(y) <= 4
        return original_cdist(x, y)
    def sinc(x):
        assert x.size <= 64
        return original_sinc(x)
    monkeypatch.setattr(core, 'pdist', pdist)
    monkeypatch.setattr(core, 'cdist', cdist)
    monkeypatch.setattr(core.np, 'sinc', sinc)
    got = core.direct_iq(cloud, q, atom_chunk=4, max_matrix_entries=64)
    np.testing.assert_allclose(got, expected, rtol=2e-14, atol=2e-9)
    a = AtomCloud(cloud.elements[:20], cloud.xyz[:20])
    b = AtomCloud(cloud.elements[20:], cloud.xyz[20:])
    cross = core.cross_iq(a, b, q, atom_chunk=4, max_matrix_entries=64)
    # The algebraic all-pairs identity also catches missing inter-block pairs.
    np.testing.assert_allclose(cross, got - core.direct_iq(a, q, atom_chunk=4, max_matrix_entries=64)
                               - core.direct_iq(b, q, atom_chunk=4, max_matrix_entries=64),
                               rtol=5e-13, atol=2e-9)
    assert any(call[0] == 'pdist' for call in calls)
    assert any(call[0] == 'cdist' for call in calls)


def test_streaming_ensemble_matches_retained_draws_and_batch_statistics(monkeypatch):
    host = AtomCloud(('C', 'Zn'), [[0., 0., 0.], [2., .4, -.2]])
    library = ParticleLibrary((AtomCloud(('Au',), [[0., 0., 0.]]),
                               AtomCloud(('Au', 'O'), [[-.2, 0., 0.], [.2, 0., 0.]])), [.6, .4])
    population = PorePopulation([[4., 0., 0.], [-4., 0., 0.]], library, occupancy=.7,
                                displacement_sigma_A=.1, pore_radius_A=1.)
    q = np.linspace(0., 8., 17)
    n_samples = 48
    allocations = []
    original_zeros = ensemble.np.zeros
    def zeros(shape, *args, **kwargs):
        if isinstance(shape, tuple):
            allocations.append(shape)
        return original_zeros(shape, *args, **kwargs)
    monkeypatch.setattr(ensemble.np, 'zeros', zeros)
    streamed = ensemble.sampled_ensemble(host, population, q, n_samples=n_samples, seed=77)
    assert (n_samples, len(ensemble.COMPONENTS), len(q)) not in allocations
    retained = ensemble.sampled_ensemble(host, population, q, n_samples=n_samples, seed=77, retain_samples=True)
    assert (n_samples, len(ensemble.COMPONENTS), len(q)) in allocations
    assert 'samples' not in streamed
    for k, name in enumerate(ensemble.COMPONENTS):
        np.testing.assert_array_equal(streamed['mean'][name], retained['mean'][name])
        np.testing.assert_array_equal(streamed['sem'][name], retained['sem'][name])
        np.testing.assert_allclose(streamed['mean'][name], retained['samples'][:, k].mean(axis=0),
                                   rtol=1e-13, atol=1e-9)
        np.testing.assert_allclose(streamed['sem'][name], retained['samples'][:, k].std(axis=0, ddof=1)
                                   / np.sqrt(n_samples), rtol=1e-13, atol=1e-10)
    for name in ('occupancy_counts', 'template_counts'):
        np.testing.assert_array_equal(streamed[name], retained[name])
    for name in ('mean_placement_attempts', 'max_full_sum_relative_error', 'seed', 'n_samples'):
        assert streamed[name] == retained[name]


@pytest.mark.parametrize('method,actual', [('direct', 'direct'), ('grid', 'direct_fallback')])
def test_adapted_direct_and_fallback_workspace_cap(monkeypatch, method, actual):
    cloud = AtomCloud(tuple(['C'] * 18 + ['Au'] * 5),
                      np.random.default_rng(213).normal(size=(23, 3)) * 3)
    q = np.linspace(0., 12., 601)
    expected = direct_iq(cloud, q)
    sizes = []
    original_sinc = torch.sinc
    def sinc(value):
        sizes.append(value.numel())
        assert value.numel() <= 1024
        return original_sinc(value)
    monkeypatch.setattr(torch, 'sinc', sinc)
    got, info = upstream_kernel_iq(cloud, q, method=method, batch_size=3_000_000,
                                   max_pair_q_entries=1024, return_info=True)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=1e-8)
    assert sizes and info['actual_method'] == actual
    assert info['direct_q_chunk'] < len(q)
    assert info['direct_pair_q_entries_limit'] == 1024
    assert info['grid_memory_is_capped'] is False


def test_too_small_vendor_species_row_cap_is_explicit():
    cloud = AtomCloud(('C',) * 4, np.zeros((4, 3)))
    with pytest.raises(ValueError, match='one species row'):
        upstream_kernel_iq(cloud, [0., 1.], method='direct', max_pair_q_entries=3)


def cache_fixture():
    basis = AtomCloud(('C', 'C', 'Au', 'O'), [[0., 0., 0.], [.7, .2, -.1], [1., -.3, .5], [.2, .4, .8]])
    cell = np.array([[3., .2, 0.], [.4, 4., .3], [.1, .8, 5.]])
    mask = np.ones((3, 2, 2), dtype=bool)
    mask[1, 0, 1] = False
    return basis, cell, mask


def test_lattice_cache_reuses_geometry_and_matches_explicit_debye(monkeypatch):
    basis, cell, mask = cache_fixture()
    explicit = concatenate([basis.moved(index @ cell) for index in np.argwhere(mask)])
    q = np.linspace(0., 9., 71)
    expected = direct_iq(explicit, q)
    cache = lattice.LatticePairCache(basis, cell, mask)
    assert cache.metadata['distance_storage'] == 'cached'
    assert 0 < cache.metadata['distance_cache_bytes'] <= cache.cache_max_bytes
    # Mutating the caller's geometry cannot silently invalidate a prepared cache.
    cell[:] = 0.
    mask[:] = False
    def forbidden(*args, **kwargs):
        raise AssertionError('Geometry was rebuilt during cached evaluation.')
    monkeypatch.setattr(lattice, 'shape_autocorrelation', forbidden)
    monkeypatch.setattr(lattice.np.linalg, 'norm', forbidden)
    np.testing.assert_allclose(cache.iq(q), expected, rtol=2e-12, atol=1e-8)
    np.testing.assert_allclose(cache.iq(q[::2]), expected[::2], rtol=2e-12, atol=1e-8)
    assert cache.metadata['evaluation_count'] == 2
    assert cache.metadata['preprocessing_seconds'] >= 0
    assert cache.metadata['last_evaluation_seconds'] >= 0


def test_lattice_cache_budget_selects_streaming_without_changing_physics(monkeypatch):
    basis, cell, mask = cache_fixture()
    q = np.array([0., .31, 2., 6., 12.])
    cached = lattice.LatticePairCache(basis, cell, mask).iq(q)
    streamed = lattice.LatticePairCache(basis, cell, mask, cache_max_bytes=1, translation_chunk=7)
    calls = []
    original_norm = lattice.np.linalg.norm
    def norm(array, *args, **kwargs):
        calls.append(array.shape)
        assert len(array) <= 7
        return original_norm(array, *args, **kwargs)
    monkeypatch.setattr(lattice.np.linalg, 'norm', norm)
    np.testing.assert_allclose(streamed.iq(q, max_matrix_entries=13), cached, rtol=1e-12, atol=1e-8)
    assert calls
    assert streamed.metadata['distance_storage'] == 'streamed'
    assert streamed.metadata['distance_cache_bytes'] == 0


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_memory_limits_are_validated(value):
    cloud = AtomCloud(('C',), [[0., 0., 0.]])
    with pytest.raises(ValueError):
        core.direct_iq(cloud, [0.], atom_chunk=value)
    with pytest.raises(ValueError):
        core.direct_iq(cloud, [0.], max_matrix_entries=value)
    with pytest.raises(ValueError):
        core.upstream_kernel_iq(cloud, [0.], max_pair_q_entries=value)


@pytest.mark.parametrize('value', [-1, 1.5, True])
def test_lattice_cache_budget_is_validated(value):
    basis, cell, mask = cache_fixture()
    with pytest.raises(ValueError, match='cache_max_bytes'):
        lattice.LatticePairCache(basis, cell, mask, cache_max_bytes=value)
