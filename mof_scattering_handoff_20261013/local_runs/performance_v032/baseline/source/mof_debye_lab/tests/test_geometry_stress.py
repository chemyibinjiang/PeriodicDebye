"""Independent geometric and statistical stress checks for R07.

The nearest-image oracle explicitly enumerates a large lattice neighborhood;
the membership oracle tests every selected parallelepiped separately. Neither
uses the production reciprocal bound or its candidate-cell selection logic.
"""
from itertools import product

import numpy as np
import pytest
from scipy.spatial.distance import cdist, pdist
from scipy.spatial.transform import Rotation
from scipy.stats import kstest, truncnorm

from mofdebye.core import AtomCloud
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.geometry import PeriodicHostTree, apply_atom_cut, contains_cell_mask


def _exhaustive_images(fractional_basis, cell, fractional_queries, extent=14):
    integer_shifts = np.asarray(list(product(range(-extent, extent + 1), repeat=3)))
    images = ((fractional_basis[None] + integer_shifts[:, None]) @ cell).reshape(-1, 3)
    distances = cdist(fractional_queries @ cell, images)
    nearest = np.argmin(distances, axis=1)
    # This fixed-halo oracle is trustworthy only when the minimizer is interior.
    chosen_shifts = integer_shifts[nearest // len(fractional_basis)]
    assert np.max(np.abs(chosen_shifts)) < extent - 2
    return distances[np.arange(len(fractional_queries)), nearest], nearest % len(fractional_basis)


@pytest.mark.parametrize('seed', range(12))
def test_nearest_periodic_atom_against_exhaustive_images(seed):
    rng = np.random.default_rng(87100 + seed)
    # Use moderate triclinic cells and skew cells whose best images may lie
    # several translations beyond a 3x3x3 halo.
    cell = (np.array([[2.1, 0, 0], [8.7, 1.4, 0], [.8, -.3, 2.8]])
            if seed % 3 == 0 else np.array([[4.2, 0, 0], [1.7, 3.1, 0], [.8, -.9, 5.3]]))
    cell = cell @ Rotation.random(random_state=rng).as_matrix().T
    if seed % 2:
        cell[1] *= -1  # Valid left-handed direct-lattice basis.
    fractional_basis = rng.uniform(.01, .99, (5, 3))
    fractional_queries = rng.uniform(.01, .99, (15, 3))
    expected, expected_indices = _exhaustive_images(fractional_basis, cell, fractional_queries)
    # Independently move every host atom and query by a different lattice vector.
    atom_shifts = rng.integers(-100_000, 100_000, (5, 3))
    query_shifts = rng.integers(-100_000, 100_000, (15, 3))
    basis = AtomCloud(('C', 'H', 'O', 'Zn', 'Au'), (fractional_basis + atom_shifts) @ cell)
    points = (fractional_queries + query_shifts) @ cell
    distances, indices = PeriodicHostTree(basis, cell, query_chunk=7).query(points.reshape(3, 5, 3))
    np.testing.assert_allclose(distances.ravel(), expected, rtol=0., atol=3e-9)
    np.testing.assert_array_equal(indices.ravel(), expected_indices)


def _all_cells_oracle(points, cell, mask, tolerance_A):
    fractional = np.linalg.solve(cell.T, points.T).T
    # Each fractional coordinate is an affine signed plane distance times the
    # reciprocal-vector norm. Include both closed faces of each selected cell.
    tolerance = tolerance_A * np.linalg.norm(np.linalg.solve(cell, np.eye(3)), axis=0)
    selected = np.argwhere(mask)
    return np.any(np.all((fractional[:, None] >= selected - tolerance) &
                        (fractional[:, None] <= selected + 1 + tolerance), axis=2), axis=1)


@pytest.mark.parametrize('seed', range(8))
def test_cell_mask_matches_union_of_every_selected_parallelepiped(seed):
    rng = np.random.default_rng(6600 + seed)
    cell = np.array([[2.1, 0, 0], [3.7, 1.4, 0], [.8, -.3, 2.8]])
    cell = cell @ Rotation.random(random_state=rng).as_matrix().T
    if seed % 2:
        cell[2] *= -1
    mask = rng.random((4, 3, 2)) < .45
    mask[0, 0, 0] = True
    fractions = rng.uniform([-1, -1, -1], [5, 4, 3], (700, 3))
    # Probe corners, shared faces, and both sides of those faces, including holes.
    corners = np.asarray(list(product(range(5), range(4), range(3))), dtype=float)
    face_points = rng.uniform([0, 0, 0], [4, 3, 2], (90, 3))
    axis = np.arange(90) % 3
    face_points[np.arange(90), axis] = np.round(face_points[np.arange(90), axis])
    fractions = np.vstack([fractions, corners, face_points,
                           face_points + 2e-8, face_points - 2e-8])
    points = fractions @ cell
    expected = _all_cells_oracle(points, cell, mask, 1e-10)
    np.testing.assert_array_equal(contains_cell_mask(points, cell, mask), expected)
    # Also exercise the documented zero-tolerance behavior on exact Cartesian
    # integer coordinates, without an oblique inverse rounding ambiguity.
    unit_points = np.vstack([corners, [[1, .3, .7], [2, .3, .7]]])
    np.testing.assert_array_equal(contains_cell_mask(unit_points, np.eye(3), mask, 0.),
                                 _all_cells_oracle(unit_points, np.eye(3), mask, 0.))


@pytest.mark.parametrize('seed', range(4))
def test_spherical_atom_cut_commutes_with_rotation_translation_and_order(seed):
    rng = np.random.default_rng(9130 + seed)
    coordinates = rng.uniform(-4., 4., (150, 3))
    elements = tuple(('C', 'O', 'Zn')[i % 3] for i in range(len(coordinates)))
    cloud = AtomCloud(elements, coordinates)
    centre = np.array([.3, -.2, .7])
    specification = {'kind': 'sphere', 'centre_A': centre, 'radius_A': 2.5,
                     'surface_policy': 'unpassivated_atom_cut'}
    retained, metadata = apply_atom_cut(cloud, specification)
    original_xyz = cloud.xyz.copy()
    rotation = Rotation.random(random_state=rng).as_matrix()
    translation = rng.uniform(-1000., 1000., 3)
    moved = cloud.moved(translation, rotation)
    transformed = dict(specification, centre_A=centre @ rotation.T + translation)
    moved_retained, moved_metadata = apply_atom_cut(moved, transformed)
    assert moved_retained.elements == retained.elements
    assert moved_metadata == metadata
    np.testing.assert_allclose(moved_retained.xyz, retained.xyz @ rotation.T + translation, atol=1e-12)
    np.testing.assert_array_equal(cloud.xyz, original_xyz)
    permutation = rng.permutation(len(cloud))
    reordered = AtomCloud(tuple(elements[i] for i in permutation), coordinates[permutation])
    cut_reordered = apply_atom_cut(reordered, specification)[0]
    assert set(map(tuple, cut_reordered.xyz)) == set(map(tuple, retained.xyz))


@pytest.mark.parametrize('seed', [2814, 6901])
def test_boundary_conditioning_preserves_occupancy_size_and_exact_conditional_law(seed):
    # Sampling only x gives two independently calculable truncated Gaussian
    # distributions: the dimer's accepted interval is narrower than the atom's.
    monomer = AtomCloud(('Au',), [[0, 0, 0]])
    dimer = AtomCloud(('Au', 'Au'), [[-.75, 0, 0], [.75, 0, 0]])
    library = ParticleLibrary((monomer, dimer), [.35, .65])
    box = {'kind': 'box', 'min_A': [-1, -1, -1], 'max_A': [1, 1, 1]}
    sigma, mean, occupancy, samples = .65, .3, .63, 6000
    population = PorePopulation([[0, 0, 0]], library, occupancy=occupancy,
        displacement_covariance_A2=np.diag([sigma**2, 0., 0.]),
        mean_displacement_A=[mean, 0, 0], orientation='fixed',
        external_boundary=box, guest_retention='whole_particle')
    rng = np.random.default_rng(seed)
    positions = [[], []]
    attempts = [[], []]
    for _ in range(samples):
        for draw in population.sample(rng):
            positions[draw.template_index].append(draw.displacement_A[0])
            attempts[draw.template_index].append(draw.attempts)
            assert np.all(np.abs(draw.cloud.xyz) <= 1)
            np.testing.assert_allclose(pdist(draw.cloud.xyz), pdist(library.templates[draw.template_index].xyz), atol=1e-14)
    counts = np.asarray([len(group) for group in positions])
    observed_occupancy = counts.sum() / samples
    assert abs(observed_occupancy - occupancy) < 6 * np.sqrt(occupancy * (1 - occupancy) / samples)
    observed_sizes = counts / counts.sum()
    size_se = np.sqrt(library.weights * (1 - library.weights) / counts.sum())
    assert np.all(np.abs(observed_sizes - library.weights) < 6 * size_se)
    assert np.mean(attempts[1]) > 2 * np.mean(attempts[0])
    for template, half_width in enumerate([1., .25]):
        a, b = (-half_width - mean) / sigma, (half_width - mean) / sigma
        distribution = truncnorm(a, b, loc=mean, scale=sigma)
        sample = np.asarray(positions[template])
        expected_mean, expected_var = distribution.stats(moments='mv')
        assert abs(sample.mean() - expected_mean) < 6 * np.sqrt(expected_var / len(sample))
        # A fixed numerical KS statistic bound avoids fragile p-value cutoffs.
        assert kstest(sample, distribution.cdf).statistic < .065


def test_sampling_is_seed_reproducible_with_periodic_clearance_and_external_boundary():
    host = AtomCloud(('C', 'O'), [[.1, .2, .3], [2.2, 1.2, .4]])
    guest = AtomCloud(('Au', 'Au'), [[-.2, 0, 0], [.2, 0, 0]])
    library = ParticleLibrary((guest,), [1.])
    populations = [
        PorePopulation([[.4, .4, .4]], library, displacement_sigma_A=.5,
            host=host, min_host_distance_A=.4, periodic_cell_A=np.diag([3., 4., 5.])),
        PorePopulation([[.4, .4, .4]], library, displacement_sigma_A=.5,
            external_boundary={'kind': 'box', 'min_A': [0, 0, 0], 'max_A': [1, 1, 1]},
            guest_retention='whole_particle'),
    ]
    for population in populations:
        first, second = np.random.default_rng(9124), np.random.default_rng(9124)
        for _ in range(30):
            left, right = population.sample(first)[0], population.sample(second)[0]
            assert left.attempts == right.attempts
            assert left.template_index == right.template_index
            np.testing.assert_array_equal(left.cloud.xyz, right.cloud.xyz)
            np.testing.assert_array_equal(left.rotation_matrix, right.rotation_matrix)


