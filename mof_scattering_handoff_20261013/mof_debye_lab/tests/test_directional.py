import numpy as np
import pytest
from scipy.optimize import brentq
from scipy.spatial.transform import Rotation

from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors
from mofdebye.directional import (
    amplitude, amplitude_components, angular_grid, cell_shape_amplitude,
    intensity, lattice_shape_amplitude, powder_average, validate_q_vectors,
)


def test_one_two_atoms_and_forward_amplitude():
    vectors = np.array([[0., 0., 0.], [.2, -.7, 1.1], [2., 3., -4.]])
    q = np.linalg.norm(vectors, axis=1)
    a = AtomCloud(('Au',), [[.7, -1.2, 3.]])
    factors = form_factors(('Au', 'O'), q)
    expected_a = factors['Au'] * np.exp(1j * (vectors @ a.xyz[0]))
    np.testing.assert_allclose(amplitude(a, vectors, atom_chunk=1, q_chunk=1), expected_a, rtol=1e-14)
    np.testing.assert_allclose(intensity(a, vectors), factors['Au'] ** 2, rtol=1e-14)
    b = AtomCloud(('O',), [[-1.5, .6, -.2]])
    combined = concatenate([a, b])
    expected = expected_a + factors['O'] * np.exp(1j * (vectors @ b.xyz[0]))
    np.testing.assert_allclose(amplitude(combined, vectors), expected, rtol=1e-14)
    np.testing.assert_allclose(intensity(combined, vectors), np.abs(expected) ** 2, rtol=1e-14)
    assert intensity(combined, vectors)[0] == pytest.approx((factors['Au'][0] + factors['O'][0]) ** 2)


def test_empty_cloud_and_coincident_atoms():
    vectors = np.array([[0., 0., 0.], [1., 2., 3.]])
    empty = AtomCloud((), np.empty((0, 3)))
    np.testing.assert_array_equal(amplitude(empty, vectors), np.zeros(2, dtype=complex))
    cloud = AtomCloud(('C', 'C', 'O'), np.zeros((3, 3)))
    factors = form_factors(cloud.elements, np.linalg.norm(vectors, axis=1))
    np.testing.assert_allclose(intensity(cloud, vectors), (2 * factors['C'] + factors['O']) ** 2)


def test_translation_phase_rotation_covariance_and_tiling():
    rng = np.random.default_rng(917)
    cloud = AtomCloud(('C', 'Au', 'O', 'Zn', 'C'), rng.normal(size=(5, 3)) * 2)
    vectors = rng.normal(size=(29, 3)) * 4
    shift = np.array([2.1, -.7, 4.2])
    rotation = Rotation.from_euler('xyz', [31, -17, 28], degrees=True).as_matrix()
    original = amplitude(cloud, vectors)
    np.testing.assert_allclose(amplitude(cloud, vectors, atom_chunk=2, q_chunk=3), original,
                               rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(amplitude(cloud.moved(shift), vectors),
                               original * np.exp(1j * (vectors @ shift)), rtol=1e-13, atol=1e-12)
    np.testing.assert_allclose(intensity(cloud.moved(shift), vectors), intensity(cloud, vectors), rtol=1e-13)
    np.testing.assert_allclose(amplitude(cloud.moved(rotation=rotation), vectors),
                               amplitude(cloud, vectors @ rotation), rtol=1e-13, atol=1e-12)


def test_directional_components_retain_interference_and_empty_limits():
    host = AtomCloud(('C', 'Zn'), [[0., .3, .2], [2., -.5, .7]])
    guests = [AtomCloud(('Au', 'Au'), [[1., .1, .2], [1.5, .8, -.3]]),
              AtomCloud(('Au',), [[-1., .7, 1.2]])]
    vectors = np.random.default_rng(2).normal(size=(35, 3))
    parts = amplitude_components(host, guests, vectors, atom_chunk=1, q_chunk=7)
    np.testing.assert_allclose(parts['total'], intensity(concatenate([host] + guests), vectors), rtol=2e-14)
    np.testing.assert_allclose(sum(parts[k] for k in ('host', 'guest_internal', 'host_guest', 'guest_guest')),
                               parts['total'], rtol=2e-14, atol=1e-10)
    assert np.any(parts['host_guest'] < 0)
    assert np.any(parts['guest_guest'] < 0)
    host_only = amplitude_components(host, [], vectors)
    np.testing.assert_allclose(host_only['total'], host_only['host'])
    np.testing.assert_array_equal(host_only['guest_internal'], np.zeros(len(vectors)))


def test_angular_grid_normalization_and_second_moments():
    directions, weights = angular_grid(12, 24)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=1), 1., atol=3e-16)
    assert weights.sum() == pytest.approx(1., abs=3e-16)
    np.testing.assert_allclose(weights @ directions, 0., atol=3e-16)
    np.testing.assert_allclose(directions.T @ (weights[:, None] * directions), np.eye(3) / 3,
                               atol=3e-15)


def test_converged_angular_average_matches_independent_debye():
    cloud = AtomCloud(('C', 'O', 'Au', 'Zn', 'C'),
                      [[-2., .4, 1.], [1.2, 3.1, -.7], [.6, -.8, 2.4],
                       [3.3, 1.8, -2.], [-1.7, -2.2, .9]])
    q = np.array([0., .2, 1.7, 4.1, 8.])
    expected = direct_iq(cloud, q)
    evaluator = lambda vectors: intensity(cloud, vectors, atom_chunk=2, q_chunk=256)
    coarse = powder_average(evaluator, q, n_mu=8, n_phi=16, q_chunk=2)
    medium = powder_average(evaluator, q, n_mu=40, n_phi=80, q_chunk=2)
    fine = powder_average(evaluator, q, n_mu=80, n_phi=160, q_chunk=2)
    assert np.max(np.abs(coarse - expected) / expected) > 1e-3
    np.testing.assert_allclose(medium, expected, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(fine, medium, rtol=1e-9, atol=1e-8)
    np.testing.assert_allclose(fine, expected, rtol=2e-12, atol=1e-8)


def test_powder_evaluation_is_chunked_and_keeps_signed_estimators():
    sizes = []
    def evaluator(vectors):
        sizes.append(len(vectors))
        return -np.ones(len(vectors))
    result = powder_average(evaluator, np.arange(11.), n_mu=3, n_phi=6, q_chunk=2)
    assert sizes == [36] * 5 + [18]
    np.testing.assert_allclose(result, -1.)


def test_irregular_triclinic_full_cell_factorization():
    basis = AtomCloud(('C', 'O', 'Zn'), [[.1, .4, .3], [1.2, .5, .9], [.8, 1.1, 1.3]])
    cell = np.array([[6., 0., 0.], [1.1, 5.2, 0.], [.3, .8, 7.1]])
    rng = np.random.default_rng(100)
    mask = rng.random((4, 3, 2)) > .25
    explicit = concatenate([basis.moved(index @ cell) for index in np.argwhere(mask)])
    vectors = np.vstack([np.zeros((1, 3)), rng.normal(size=(37, 3)) * 3])
    factored = cell_shape_amplitude(basis, cell, mask, vectors, atom_chunk=1, cell_chunk=3, q_chunk=5)
    np.testing.assert_allclose(factored, amplitude(explicit, vectors, atom_chunk=4, q_chunk=7),
                               rtol=1e-12, atol=1e-10)
    assert lattice_shape_amplitude(cell, mask, [[0., 0., 0.]])[0] == mask.sum()


def test_chain_geometric_sum_peak_width_and_integrated_scaling():
    spacing = 3.2
    cell = np.diag([spacing, 4., 5.])
    reciprocal = 2 * np.pi / spacing
    half_widths = []
    for n in (16, 32):
        mask = np.ones((n, 1, 1), dtype=bool)
        delta = np.linspace(-np.pi / spacing, np.pi / spacing, 4097)
        vectors = np.column_stack([reciprocal + delta, np.zeros((len(delta), 2))])
        shape_intensity = np.abs(lattice_shape_amplitude(cell, mask, vectors, cell_chunk=7, q_chunk=31)) ** 2
        expected = n ** 2 * (np.sinc(n * delta * spacing / (2 * np.pi))
                            / np.sinc(delta * spacing / (2 * np.pi))) ** 2
        np.testing.assert_allclose(shape_intensity, expected, rtol=1e-10, atol=2e-10)
        assert shape_intensity[len(delta) // 2] == pytest.approx(n ** 2, rel=1e-14)
        # Orthogonality over one reciprocal period: area / number of cells = 2pi/a.
        area = np.sum((shape_intensity[1:] + shape_intensity[:-1]) * np.diff(delta) / 2)
        assert area / n == pytest.approx(reciprocal, rel=2e-13)
        first_zero = reciprocal / n
        zero = lattice_shape_amplitude(cell, mask, [[reciprocal + first_zero, 0., 0.]])[0]
        assert abs(zero) < 1e-11
        half_widths.append(brentq(lambda x: (np.sinc(n * x * spacing / (2 * np.pi))
                                             / np.sinc(x * spacing / (2 * np.pi))) ** 2 - .5,
                                  0., first_zero))
    assert half_widths[0] / half_widths[1] == pytest.approx(2., rel=.003)


@pytest.mark.parametrize('vectors', [[], [1., 2., 3.], [[1., 2.]], [[0., np.nan, 0.]],
                                    np.empty((0, 3)), np.zeros((2, 3, 1))])
def test_bad_q_vectors_rejected(vectors):
    with pytest.raises(ValueError, match='q_vectors'):
        validate_q_vectors(vectors)


@pytest.mark.parametrize('value', [0, -1, 1.5, True])
def test_bad_chunk_and_grid_counts_rejected(value):
    cloud = AtomCloud(('C',), [[0., 0., 0.]])
    with pytest.raises(ValueError):
        amplitude(cloud, [[0., 0., 0.]], atom_chunk=value)
    with pytest.raises(ValueError):
        amplitude(cloud, [[0., 0., 0.]], q_chunk=value)
    with pytest.raises(ValueError):
        angular_grid(value, 4)
    with pytest.raises(ValueError):
        angular_grid(4, value)


@pytest.mark.parametrize('mask', [np.zeros((2, 2, 2)), np.ones((2, 2)), np.full((1, 1, 1), .5)])
def test_bad_mask_rejected(mask):
    with pytest.raises(ValueError, match='mask'):
        lattice_shape_amplitude(np.eye(3), mask, [[0., 0., 0.]])


@pytest.mark.parametrize('cell', [np.zeros((3, 3)), np.ones((2, 2)), np.eye(3) * np.nan])
def test_bad_lattice_rejected(cell):
    with pytest.raises(ValueError, match='cell_vectors'):
        lattice_shape_amplitude(cell, np.ones((1, 1, 1)), [[0., 0., 0.]])


@pytest.mark.parametrize('evaluator', [lambda v: np.zeros((len(v), 1)),
                                     lambda v: np.full(len(v), np.nan),
                                     lambda v: np.ones(len(v), dtype=complex)])
def test_bad_directional_evaluator_rejected(evaluator):
    with pytest.raises(ValueError, match='evaluator'):
        powder_average(evaluator, [1.], n_mu=2, n_phi=4)
