"""Independent finite-atom checks for the compiled radial-moment wrapper."""
from pathlib import Path

import numpy as np
import pytest

numba = pytest.importorskip('numba')

from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors
from mofdebye.crystal import read_cif
from mofdebye.large_powder import LargePowderCache, _merge_offsets


def _basis():
    return AtomCloud(('C', 'O', 'C', 'Zn', 'O'),
                     [[0., 0., 0.], [.3, 1.1, .2], [.7, -.2, 1.3],
                      [1.2, .6, -.1], [-.8, .9, 1.3]])


def _explicit(basis, cell, mask):
    return concatenate([basis.moved(index @ cell) for index in np.argwhere(mask)])


@pytest.mark.parametrize('threads', [1, 2])
@pytest.mark.parametrize('shape', ['cube', 'hole', 'skew', 'negative_axis'])
def test_radial_backend_matches_independent_explicit_debye(shape, threads):
    basis = _basis()
    cell = np.eye(3) * 4
    mask = np.ones((2, 2, 2), bool)
    if shape == 'hole':
        mask[0, 1, 0] = False
    elif shape == 'skew':
        cell = np.array([[4., .2, -.1], [.3, 3.8, .2], [-.1, .4, 4.2]])
    elif shape == 'negative_axis':
        cell[1, 1] *= -1
    q = np.linspace(0, 12, 241)
    original_threads = numba.get_num_threads()
    cache = LargePowderCache(basis, cell, mask, q, bin_width=.08, threads=threads)
    actual, bound = cache.iq()
    expected = direct_iq(_explicit(basis, cell, mask), q)
    scale = max(expected)
    assert np.max(np.abs(actual - expected)) < 2e-12 * scale
    assert np.all(np.abs(actual - expected) <= bound + 2e-12 * scale)
    selected = np.array([0., 1e-8, .3123, 2.51, 12.])
    np.testing.assert_allclose(cache.exact_selected(selected),
                               direct_iq(_explicit(basis, cell, mask), selected),
                               rtol=3e-13, atol=2e-9)
    assert numba.get_num_threads() == original_threads
    metadata = cache.metadata
    assert metadata['represented_atoms'] == len(basis) * mask.sum()
    assert metadata['max_absolute_approximation_bound_electron2'] >= 0
    assert not metadata['fastmath']
    for pair in metadata['species_pairs']:
        a, b = pair['elements']
        expected_count = basis.elements.count(a) * basis.elements.count(b) * int(mask.sum())**2
        assert pair['pair_count'] == expected_count


def test_coarse_radial_approximation_has_a_nontrivial_valid_bound():
    basis = _basis()
    mask = np.ones((2, 2, 2), bool)
    cell = np.eye(3) * 4
    q = np.linspace(0, 8, 161)
    cache = LargePowderCache(basis, cell, mask, q, bin_width=1.0, order=4,
                             merge_decimals=None)
    actual, bound = cache.iq()
    expected = direct_iq(_explicit(basis, cell, mask), q)
    error = np.abs(actual - expected)
    assert error.max() > 1e-3
    assert np.all(error <= bound + 1e-9 * expected[0])
    assert bound[0] == 0
    assert cache.metadata['max_absolute_grouping_bound_electron2'] == 0


def test_offset_merging_retains_actual_representatives_and_bounds_displacement():
    offsets = np.array([[.12340, 0, 0], [.12341, 0, 0], [1.3, 2.1, 3.2]])
    weights = np.array([3, 7, 11], dtype=np.int64)
    result, counts, weighted_error, max_error = _merge_offsets(offsets, weights, 3)
    np.testing.assert_array_equal(result, offsets[[0, 2]])
    np.testing.assert_array_equal(counts, [10, 11])
    assert weighted_error == pytest.approx(7e-5)
    assert max_error == pytest.approx(1e-5)
    exact, exact_weights, weighted_error, max_error = _merge_offsets(offsets, weights, None)
    np.testing.assert_array_equal(exact, offsets)
    np.testing.assert_array_equal(exact_weights, weights)
    assert weighted_error == max_error == 0


def test_nontrivial_offset_merging_error_is_included_in_returned_bound():
    basis = AtomCloud(('C', 'C', 'O'), [[1., 0, 0], [1.003, 0, 0], [0., 0, 0]])
    cell = np.eye(3) * 4
    mask = np.ones((2, 2, 2), bool)
    q = np.linspace(0, 12, 241)
    cache = LargePowderCache(basis, cell, mask, q, merge_decimals=2)
    actual, bound = cache.iq()
    expected = direct_iq(_explicit(basis, cell, mask), q)
    assert np.max(np.abs(actual - expected)) > 1e-3
    assert np.all(np.abs(actual - expected) <= bound + 2e-10)
    assert cache.metadata['max_absolute_grouping_bound_electron2'] > 0
    assert cache.metadata['grouped_basis_displacements'] < cache.metadata['original_basis_displacements']
    unmerged = cache.exact_selected(q[::20])
    merged = cache.exact_selected(q[::20], merged=True)
    np.testing.assert_allclose(unmerged, expected[::20], rtol=1e-12, atol=2e-9)
    assert np.max(np.abs(unmerged - merged)) > 1e-3


def test_coincident_mixed_atoms_and_forward_intensity_are_exact():
    basis = AtomCloud(('C', 'C', 'O', 'Zn'), np.zeros((4, 3)))
    q = np.linspace(0, 6, 121)
    cache = LargePowderCache(basis, np.eye(3) * 4, np.ones((1, 1, 1), bool), q)
    actual, bound = cache.iq()
    ff = form_factors(basis.elements, q)
    expected = sum(ff[element] for element in basis.elements)**2
    np.testing.assert_allclose(actual, expected, rtol=2e-15)
    np.testing.assert_array_equal(bound, 0.)
    assert all(pair['pair_count'] == pair['zero_distance_count']
               for pair in cache.metadata['species_pairs'])


def test_empty_basis_returns_zero():
    basis = AtomCloud((), np.empty((0, 3)))
    q = np.linspace(0, 6, 121)
    cache = LargePowderCache(basis, np.eye(3), np.ones((1, 1, 1), bool), q)
    actual, bound = cache.iq()
    np.testing.assert_array_equal(actual, 0.)
    np.testing.assert_array_equal(bound, 0.)
    np.testing.assert_array_equal(cache.exact_selected([0., 1.3]), 0.)


def test_input_geometry_is_copied_and_repeated_evaluation_is_deterministic():
    cell = np.eye(3) * 4
    mask = np.ones((2, 2, 2), bool)
    q = np.linspace(0, 6, 121)
    cache = LargePowderCache(_basis(), cell, mask, q, threads=2)
    before, before_bound = cache.iq()
    cell[:] = 99
    mask[:] = False
    q[:] = 99
    after, after_bound = cache.iq()
    np.testing.assert_array_equal(after, before)
    np.testing.assert_array_equal(after_bound, before_bound)
    assert cache.metadata['evaluation_count'] == 2


def test_private_histogram_budget_reduces_partitions(monkeypatch):
    import mofdebye.large_powder as module
    q = np.linspace(0, 6, 121)
    probe = LargePowderCache(_basis(), np.eye(3) * 4, np.ones((2, 2, 2), bool), q)
    one_partition = probe.metadata['histogram_private_workspace_bytes']
    monkeypatch.setattr(module, '_MAX_PRIVATE_HISTOGRAM_BYTES', one_partition)
    cache = LargePowderCache(_basis(), np.eye(3) * 4, np.ones((2, 2, 2), bool), q, threads=2)
    assert cache.metadata['histogram_partitions'] == 1
    assert cache.metadata['histogram_private_workspace_bytes'] == one_partition
    np.testing.assert_allclose(cache.iq()[0], probe.iq()[0], rtol=0, atol=0)


@pytest.mark.parametrize('options', [
    {'threads': 0}, {'threads': True}, {'threads': 1.5},
    {'merge_decimals': True}, {'merge_decimals': -1}, {'merge_decimals': 16},
    {'merge_decimals': 12.0}, {'order': 3}, {'order': True},
    {'bin_width': 0}, {'bin_width': np.nan},
])
def test_invalid_options_are_rejected(options):
    with pytest.raises(ValueError):
        LargePowderCache(_basis(), np.eye(3) * 4, np.ones((1, 1, 1), bool),
                         np.linspace(0, 6, 121), **options)


@pytest.mark.parametrize('q', [[0], [.1, .2], [0, .1, .3], [0, 1, 2], [0, np.nan]])
def test_invalid_or_underresolved_q_grid_is_rejected(q):
    with pytest.raises(ValueError):
        LargePowderCache(_basis(), np.eye(3) * 4, np.ones((2, 2, 2), bool), q)


def test_real_mof_one_cell_matches_independent_atom_pairs_and_compresses_offsets():
    cif = Path(__file__).parents[1] / 'examples' / 'real_mof' / 'IRMOF-1.cif'
    structure = read_cif(cif)
    q = np.linspace(0, 6, 121)
    cache = LargePowderCache(structure.basis, structure.cell_vectors_A,
                             np.ones((1, 1, 1), bool), q)
    actual, bound = cache.iq()
    expected = direct_iq(structure.basis, q)
    np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-7)
    assert cache.metadata['original_basis_displacements'] == 11339
    assert cache.metadata['grouped_basis_displacements'] == 1526
    assert cache.metadata['max_merged_displacement_A'] < 1e-12
    assert np.all(bound >= 0)
