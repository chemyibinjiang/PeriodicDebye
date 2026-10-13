"""Independent Debye checks for exact displacement compression and batching."""
from pathlib import Path

import numpy as np
import pytest

from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors
from mofdebye.crystal import read_cif
from mofdebye.lattice import LatticePairCache


def _explicit(basis, cell, mask):
    return concatenate([basis.moved(index @ cell) for index in np.argwhere(mask)])


@pytest.mark.parametrize('cache_bytes', [0, 64 * 1024**2])
@pytest.mark.parametrize('shape', ['cube', 'cuboid', 'hole', 'skew', 'negative_axes'])
def test_compressed_powder_matches_independent_atom_pairs(shape, cache_bytes):
    basis = AtomCloud(('C', 'C', 'C', 'O', 'Zn', 'O'),
                      [[0., 0., 0.], [1., 0., 0.], [0., -1., 0.],
                       [.5, .7, .2], [.2, .3, .9], [-.5, -.7, -.2]])
    cell = np.eye(3) * 3.5
    mask = np.ones((2, 2, 2), dtype=bool)
    if shape == 'cuboid':
        mask = np.ones((2, 3, 1), dtype=bool)
    elif shape == 'hole':
        mask[0, 1, 0] = False
    elif shape == 'skew':
        cell = np.array([[3.5, .2, -.1], [0., 3.5, .3], [.1, .4, 4.]])
    elif shape == 'negative_axes':
        cell[1, 1] *= -1
    q = np.r_[0., np.geomspace(1e-8, 45., 83)]
    expected = direct_iq(_explicit(basis, cell, mask), q, atom_chunk=13)
    cache = LatticePairCache(basis, cell, mask, cache_max_bytes=cache_bytes, translation_chunk=23)
    actual = cache.iq(q, max_matrix_entries=197)
    np.testing.assert_allclose(actual, expected, rtol=3e-12, atol=2e-8)
    assert cache.metadata['displacement_symmetry'] == (
        'cubic_signed_permutations' if shape in ('cube', 'negative_axes') else 'inversion')
    assert cache.metadata['grouped_basis_displacements'] < cache.metadata['raw_basis_pairs']


@pytest.mark.parametrize('perturbation', ['length', 'off_diagonal', 'rotation'])
def test_cubic_specialization_requires_exact_metric(perturbation):
    basis = AtomCloud(('C', 'O'), [[0., 0., 0.], [1., 2., 3.]])
    cell = np.eye(3) * 4
    if perturbation == 'length':
        cell[0, 0] = np.nextafter(4., 5.)
    elif perturbation == 'off_diagonal':
        cell[0, 1] = 1e-20
    else:
        angle = .37
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                             [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
        cell = cell @ rotation
    cache = LatticePairCache(basis, cell, np.ones((2, 2, 2), dtype=bool))
    assert cache.metadata['displacement_symmetry'] == 'inversion'


def test_nearby_offsets_are_not_rounded_or_binned():
    separation = np.nextafter(1., 2.)
    basis = AtomCloud(('C', 'C', 'O'), [[1., 0., 0.], [separation, 0., 0.], [0., 0., 0.]])
    cache = LatticePairCache(basis, np.eye(3) * 4, np.ones((1, 1, 1), dtype=bool))
    np.testing.assert_array_equal(cache._offsets['C', 'O'][:, 2], [1., separation])
    np.testing.assert_array_equal(cache._offset_counts['C', 'O'], [1, 1])


def test_degenerate_basis_self_terms_and_exact_q_zero():
    basis = AtomCloud(('C', 'C', 'O', 'Zn'), np.zeros((4, 3)))
    mask = np.ones((4, 4, 4), dtype=bool)
    cache = LatticePairCache(basis, np.eye(3) * 6, mask)
    ff = form_factors(basis.elements, [0.])
    expected = (mask.sum() * sum(ff[element][0] for element in basis.elements))**2
    np.testing.assert_allclose(cache.iq([0.]), expected, rtol=1e-15)
    assert cache.grouped_basis_displacements == 6
    for array in (*cache._offsets.values(), *cache._offset_counts.values()):
        assert not array.flags.writeable


def test_real_mof_symmetry_compression_matches_explicit_one_cell():
    cif = Path(__file__).parents[1] / 'examples' / 'real_mof' / 'IRMOF-1.cif'
    structure = read_cif(cif)
    cache = LatticePairCache(structure.basis, structure.cell_vectors_A,
                            np.ones((1, 1, 1), dtype=bool))
    q = np.r_[0., np.geomspace(1e-6, 20., 47)]
    expected = direct_iq(structure.basis, q)
    np.testing.assert_allclose(cache.iq(q), expected, rtol=8e-13, atol=1e-7)
    assert cache.raw_basis_pairs == 118848
    assert cache.grouped_basis_displacements < cache.raw_basis_pairs / 8
    assert cache.metadata['radial_cache_entries'] < cache.grouped_basis_displacements / 2
    actual_bytes = sum(radii.nbytes + counts.nbytes for radii, counts in cache._distances.values())
    assert cache.metadata['distance_cache_bytes'] == actual_bytes
    assert actual_bytes <= cache.cache_max_bytes


def test_radial_aggregation_retains_adjacent_float_radii_and_integer_weights():
    from mofdebye.lattice import _aggregate_radii
    near = np.nextafter(1., 2.)
    radii, counts = _aggregate_radii(np.array([near, 1., near, 2., 1.]),
                                    np.array([2**53, 7, 3, 11, 13], dtype=np.int64))
    np.testing.assert_array_equal(radii, [1., near, 2.])
    np.testing.assert_array_equal(counts, [20, 2**53 + 3, 11])


@pytest.mark.parametrize('budget', [0, 16, 96, 1024, 65536])
def test_radial_cache_payload_budget_and_fallback_preserve_full_pair_count(budget):
    basis = AtomCloud(('C', 'C', 'O', 'O', 'Zn'),
                      [[0., 0., 0.], [1., 0., 0.], [.1, .2, .3],
                       [.1, -.2, -.3], [.5, .5, .5]])
    cell = np.eye(3) * 3
    mask = np.ones((3, 3, 3), dtype=bool)
    cache = LatticePairCache(basis, cell, mask, cache_max_bytes=budget, translation_chunk=19)
    assert cache.metadata['cache_candidate_payload_peak_bytes'] <= budget
    assert cache.metadata['distance_cache_bytes'] <= budget
    if cache._distances is None:
        assert cache.metadata['radial_cache_entries'] == 0
        assert cache.metadata['distance_cache_bytes'] == 0
    else:
        actual_bytes = 0
        for (a, b), (radii, counts) in cache._distances.items():
            assert not radii.flags.writeable and not counts.flags.writeable
            assert np.all(np.diff(radii) > 0)
            assert counts.dtype == np.int64
            assert counts.sum() == len(cache._positions[a]) * len(cache._positions[b]) * int(mask.sum())**2
            actual_bytes += radii.nbytes + counts.nbytes
        assert cache.metadata['distance_cache_bytes'] == actual_bytes
    q = np.array([0., .07, .21, 1.5, 8., 21.])
    expected = direct_iq(_explicit(basis, cell, mask), q)
    np.testing.assert_allclose(cache.iq(q, max_matrix_entries=31), expected,
                               rtol=2e-12, atol=5e-8)


def test_streamed_radial_sorting_is_bounded_by_distance_block(monkeypatch):
    import mofdebye.lattice as lattice
    basis = AtomCloud(('C', 'O', 'Zn'), [[0., 0., 0.], [.5, 0., .5], [0., .25, .25]])
    cache = LatticePairCache(basis, np.eye(3) * 4, np.ones((3, 3, 3), bool),
                            cache_max_bytes=0, translation_chunk=17)
    original = lattice._aggregate_radii
    calls = []
    def aggregate(distances, weights):
        calls.append(len(distances))
        assert len(distances) <= 17
        assert len(weights) == len(distances)
        return original(distances, weights)
    monkeypatch.setattr(lattice, '_aggregate_radii', aggregate)
    cache.iq([0., .1, 1., 10.], max_matrix_entries=23)
    assert len(calls) > 1
