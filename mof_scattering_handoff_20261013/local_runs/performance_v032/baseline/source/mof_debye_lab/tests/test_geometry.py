from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation

from mofdebye.core import AtomCloud
from mofdebye.geometry import (PeriodicHostTree, apply_atom_cut,
                               contains_cell_mask, contains_points)


def test_atom_cut_has_explicit_termination_and_preserves_species_order():
    cloud = AtomCloud(('C', 'O', 'Zn', 'H'), [[0, 0, 0], [1, 0, 0], [1.1, 0, 0], [-1, 0, 0]])
    sphere = {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 1.,
              'surface_policy': 'unpassivated_atom_cut'}
    retained, metadata = apply_atom_cut(cloud, sphere)
    assert retained.elements == ('C', 'O', 'H')
    np.testing.assert_array_equal(retained.xyz, cloud.xyz[[0, 1, 3]])
    assert metadata['candidate_atom_count'] == 4
    assert metadata['retained_atom_count'] == 3
    assert metadata['removed_atom_count'] == 1
    assert metadata['surface_policy'] == 'unpassivated_atom_cut'
    assert len(cloud) == 4
    box = {'kind': 'box', 'min_A': [-.5, -.5, -.5], 'max_A': [.5, .5, .5],
           'surface_policy': 'unpassivated_atom_cut'}
    assert apply_atom_cut(cloud, box)[0].elements == ('C',)


@pytest.mark.parametrize('specification, message', [
    ({'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 1}, 'surface_policy'),
    ({'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 1,
      'surface_policy': 'reconstructed'}, 'surface_policy'),
    ({'kind': 'box', 'min_A': [3, 3, 3], 'max_A': [4, 4, 4],
      'surface_policy': 'unpassivated_atom_cut'}, 'removed all'),
    ({'kind': 'cell_mask', 'surface_policy': 'unpassivated_atom_cut'}, 'sphere or box'),
])
def test_cut_rejects_implicit_chemistry_or_empty_result(specification, message):
    with pytest.raises(ValueError, match=message):
        apply_atom_cut(AtomCloud(('C',), [[0, 0, 0]]), specification)


@pytest.mark.parametrize('specification', [
    {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 0},
    {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': True},
    {'kind': 'sphere', 'centre_A': [0, 0], 'radius_A': 1},
    {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': float('nan')},
    {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 1, 'typo': 1},
    {'kind': 'box', 'min_A': [0, 0, 0], 'max_A': [1, 0, 1]},
    {'kind': 'unknown'},
    {},
])
def test_boundary_validation(specification):
    with pytest.raises(ValueError):
        contains_points([[0, 0, 0]], specification)


def test_cell_mask_includes_closed_faces_and_handles_holes_and_oblique_cells():
    cell = np.array([[4., 1., .2], [.5, 3., .7], [.2, .1, 5.]])
    mask = np.array([1, 0, 1]).reshape(3, 1, 1)
    fractions = np.array([[.5, .5, .5], [1., .5, .5], [1.5, .5, .5],
                          [2., .5, .5], [3., .5, .5], [3.1, .5, .5],
                          [0., 0., 0.], [1., 1., 1.], [-.1, .5, .5]])
    expected = [True, True, False, True, True, False, True, True, False]
    np.testing.assert_array_equal(contains_cell_mask(fractions @ cell, cell, mask), expected)
    np.testing.assert_array_equal(contains_points(fractions @ cell, {
        'kind': 'cell_mask', 'cell_vectors_A': cell, 'mask': mask}), expected)
    assert contains_cell_mask([1., .5, .5], np.eye(3), np.ones((1, 1, 1)), tolerance_A=0.)
    with pytest.raises(ValueError, match='binary'):
        contains_cell_mask([[0, 0, 0]], cell, [[[2]]])
    with pytest.raises(ValueError, match='nonnegative'):
        contains_cell_mask([[0, 0, 0]], cell, mask, tolerance_A=-1)


def test_intersection_uses_cut_and_actual_candidate_mask():
    boundary = {'kind': 'intersection', 'boundaries': [
        {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 2.},
        {'kind': 'cell_mask', 'cell_vectors_A': np.eye(3), 'mask': [[[1]]]},
    ]}
    points = [[.5, .5, .5], [-.5, .5, .5], [1.5, .5, .5], [0., 0., 0.]]
    np.testing.assert_array_equal(contains_points(points, boundary), [True, False, False, True])
    assert contains_points([.5, .5, .5], boundary)
    with pytest.raises(ValueError, match='nonempty'):
        contains_points(points, {'kind': 'intersection', 'boundaries': []})


def _brute_distances(basis, cell, points, image_extent):
    inverse = np.linalg.inv(cell)
    canonical = (basis.xyz @ inverse % 1.) @ cell
    shifts = np.asarray(list(product(range(-image_extent, image_extent + 1), repeat=3))) @ cell
    images = (canonical[None, :, :] + shifts[:, None, :]).reshape(-1, 3)
    wrapped = (np.asarray(points) @ inverse % 1.) @ cell
    return cKDTree(images).query(wrapped, k=1)[0]


@pytest.mark.parametrize('cell', [
    np.diag([4., 5., 6.]),
    np.array([[4., 0., 0.], [3.7, 2.1, 0.], [1.2, .8, 3.3]]),
    np.array([[1., 0., 0.], [5.1, .2, 0.], [0., 0., 4.]]),
])
def test_periodic_clearance_matches_explicit_images_including_skew(cell):
    rng = np.random.default_rng(42)
    basis = AtomCloud(('C', 'O', 'Zn'), rng.random((3, 3)) @ cell)
    fractions = rng.random((17, 3))
    # Query coordinates may lie many cells away, as allowed by Gaussian jitter.
    points = (fractions + rng.integers(-10000, 10000, size=(17, 3))) @ cell
    tree = PeriodicHostTree(basis, cell, query_chunk=3)
    distances, indices = tree.query(points)
    np.testing.assert_allclose(distances, _brute_distances(basis, cell, points, 12), atol=2e-11)
    assert indices.shape == distances.shape == (17,)
    assert np.all((indices >= 0) & (indices < len(basis)))
    assert tree.is_periodic
    assert tree.metadata['basis_atom_count'] == 3


def test_periodic_nearest_image_can_be_outside_adjacent_cells():
    cell = np.array([[1., 0., 0.], [5.1, .2, 0.], [0., 0., 4.]])
    basis = AtomCloud(('C',), [[0, 0, 0]])
    point = np.array([.49, .49, 0.]) @ cell
    got, index = PeriodicHostTree(basis, cell).query(point)
    expected = _brute_distances(basis, cell, point[None], 10)[0]
    nearby = _brute_distances(basis, cell, point[None], 1)[0]
    np.testing.assert_allclose(got, expected, atol=1e-14)
    assert got < nearby - .5
    assert index == 0


def test_periodic_tree_rotates_with_cell_and_accepts_basis_lattice_shifts():
    cell = np.array([[4., 0., 0.], [1.7, 3.1, 0.], [1.2, .8, 3.3]])
    fractional = np.array([[.03, .4, .5], [.6, .2, .1]])
    points = np.array([[.97, .4, .5], [.01, .4, .5], [2.3, -5.6, 1.2]]) @ cell
    basis = AtomCloud(('O', 'C'), fractional @ cell)
    expected = PeriodicHostTree(basis, cell).query(points)[0]
    np.testing.assert_allclose(expected[:2], [.24, .08], atol=1e-14)
    rotation = Rotation.from_rotvec([.4, .5, .6]).as_matrix()
    shifted = AtomCloud(basis.elements, (fractional + [[11, -2, 5], [-20, 5, 7]]) @ cell)
    rotated = PeriodicHostTree(shifted.moved(rotation=rotation), cell @ rotation.T)
    np.testing.assert_allclose(rotated.query(points @ rotation.T)[0], expected, atol=4e-14)
    assert rotated.query(np.empty((0, 3)))[0].shape == (0,)
    assert rotated.query(points.reshape(1, 3, 3))[0].shape == (1, 3)


def test_periodic_search_is_chunked_and_guarded(monkeypatch):
    import mofdebye.geometry as geometry
    original = geometry.cKDTree
    observed = []

    class TrackingTree:
        def __init__(self, values):
            self.tree = original(values)

        def query(self, values, k=1):
            observed.append(len(values))
            return self.tree.query(values, k=k)

    monkeypatch.setattr(geometry, 'cKDTree', TrackingTree)
    basis = AtomCloud(('C',), [[0, 0, 0]])
    tree = PeriodicHostTree(basis, np.eye(3) * 3, query_chunk=2)
    tree.query([[1.4, 1.4, 1.4]])
    assert max(observed) <= 2
    with pytest.raises(ValueError, match='No image truncation'):
        PeriodicHostTree(basis, np.eye(3) * 3, max_images=1).query([[1.4, 1.4, 1.4]])
    with pytest.raises(ValueError, match='k=1'):
        tree.query([[0, 0, 0]], k=2)
    with pytest.raises(ValueError, match='nonempty'):
        PeriodicHostTree(AtomCloud((), np.empty((0, 3))), np.eye(3))
    with pytest.raises(ValueError, match='finite'):
        tree.query([[np.inf, 0, 0]])
