"""Independent correctness oracles for exact directional/periodic speedups."""
import itertools

import numpy as np
import pytest

from mofdebye.core import AtomCloud, form_factors
from mofdebye.directional import lattice_shape_amplitude
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.periodic import enumerate_reflections, powder_diffuse, powder_shells


CELLS = [np.diag([25.8, 24.1, 26.7]),
         np.array([[25.8, 0., 0.], [8.1, 24.1, 0.], [-4.3, 7.2, 26.7]]),
         np.array([[0., 24.1, 0.], [25.8, 0., 0.], [-4.3, 7.2, 26.7]])]


@pytest.mark.parametrize('cell', CELLS)
@pytest.mark.parametrize('shape', [(1, 1, 1), (7, 5, 3), (2, 9, 1)])
def test_rectangular_factorization_matches_direct_cell_sum(cell, shape):
    rng = np.random.default_rng(951)
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    bragg = np.array([3, -2, 1]) @ reciprocal
    vectors = np.vstack((np.zeros(3), rng.normal(size=(31, 3)) * 5,
                         bragg, bragg + 1e-11, bragg - 1e-11))
    indices = np.array(list(itertools.product(*(range(n) for n in shape))))
    positions = indices @ cell
    expected = np.exp(1j * (vectors @ positions.T)).sum(axis=1)
    got = lattice_shape_amplitude(cell, np.ones(shape, dtype=bool), vectors,
                                  q_chunk=7, cell_chunk=3)
    np.testing.assert_allclose(got, expected, rtol=3e-11, atol=2e-10)
    assert got[0] == np.prod(shape)


def test_large_chain_forward_bragg_and_first_zero():
    n = 100_000
    phases = np.array([0., 1e-12, 2 * np.pi / n, (2 * np.pi + .01) / n,
                       2 * np.pi, 2 * np.pi - 1e-11, 2 * np.pi + 1e-11])
    vectors = np.column_stack((phases, np.zeros((len(phases), 2))))
    indices = np.arange(n)
    expected = np.array([np.exp(1j * phase * indices).sum() for phase in phases])
    got = lattice_shape_amplitude(np.eye(3), np.ones((n, 1, 1), bool), vectors)
    # Absolute, count-normalized tolerance remains meaningful at exact zeros.
    assert np.max(abs(got - expected)) / n < 2e-10
    assert got[0] == n
    assert abs(got[2]) / n < 1e-14


def test_irregular_mask_still_matches_explicit_sum():
    cell = CELLS[1]
    mask = np.random.default_rng(63).random((7, 5, 3)) > .3
    vectors = np.random.default_rng(64).normal(size=(23, 3))
    positions = np.argwhere(mask) @ cell
    expected = np.exp(1j * (vectors @ positions.T)).sum(axis=1)
    got = lattice_shape_amplitude(cell, mask, vectors, q_chunk=5, cell_chunk=7)
    np.testing.assert_allclose(got, expected, rtol=2e-12, atol=3e-12)


@pytest.mark.parametrize('cell', [np.diag([4., 5., 6.]),
                                np.array([[4., 0., 0.], [3.7, 2.1, 0.], [1.2, .8, -3.3]])])
@pytest.mark.parametrize('include_origin,q_min', [(False, 0.), (True, 0.), (False, 1.7)])
def test_reflection_enumeration_matches_scalar_bruteforce(cell, include_origin, q_min):
    q_max = 4.3
    reciprocal = 2 * np.pi * np.linalg.inv(cell).T
    bounds = np.ceil(q_max * np.linalg.norm(cell, axis=1) / (2 * np.pi)).astype(int)
    retained = []
    for hkl in itertools.product(*(range(-int(b) - 1, int(b) + 2) for b in bounds)):
        if hkl == (0, 0, 0) and not include_origin:
            continue
        radius = np.linalg.norm(np.array(hkl) @ reciprocal)
        if q_min - 1e-12 <= radius <= q_max + 1e-12:
            retained.append(hkl)
    hkl = np.asarray(retained).reshape(-1, 3)
    vectors = hkl @ reciprocal
    radii = np.linalg.norm(vectors, axis=1)
    order = np.argsort(radii, kind='stable')
    got_hkl, got_vectors, got_q = enumerate_reflections(cell, q_max, q_min, include_origin)
    np.testing.assert_array_equal(got_hkl, hkl[order])
    np.testing.assert_allclose(got_vectors, vectors[order], rtol=0, atol=1e-14)
    np.testing.assert_allclose(got_q, radii[order], rtol=0, atol=1e-14)


def test_reflection_candidate_tile_boundary_and_q_cutoffs():
    # 47^3 candidates crosses the internal candidate tile boundary.
    cell = np.eye(3) * 35
    hkl, _, q = enumerate_reflections(cell, 4., q_min=1.1)
    bound = 23
    expected = {h for h in itertools.product(range(-bound, bound + 1), repeat=3)
                if 1.1 - 1e-12 <= 2 * np.pi / 35 * np.sqrt(sum(x*x for x in h)) <= 4. + 1e-12}
    assert set(map(tuple, hkl)) == expected
    assert np.all(np.diff(q) >= 0)
    empty, vectors, radii = enumerate_reflections(np.eye(3), 0.)
    assert empty.shape == vectors.shape == (0, 3)
    assert radii.shape == (0,)


@pytest.mark.parametrize('rtol,atol', [(0., .1), (1e-10, 1e-12), (0., 0.)])
def test_shell_groups_preserve_first_anchor_and_boundary_semantics(rtol, atol):
    tolerance = atol + rtol
    boundary = 1. + tolerance
    q = np.array([1., 1., np.nextafter(boundary, 0.), boundary,
                  np.nextafter(boundary, np.inf), 1. + 1.8 * tolerance, 2., 3.])
    q = q[[3, 7, 0, 5, 2, 6, 1, 4]]
    f_squared = np.arange(len(q), dtype=float) - 3.5
    bragg_weight = .017 * f_squared
    reflection = dict(q=q, f_squared=f_squared, bragg_weight=bragg_weight)
    groups = []
    for index in np.argsort(q):
        if not groups or not np.isclose(q[index], q[groups[-1][0]], rtol=rtol, atol=atol):
            groups.append([index])
        else:
            groups[-1].append(index)
    got = powder_shells(reflection, rtol=rtol, atol=atol)
    np.testing.assert_array_equal(got['q'], [q[g[0]] for g in groups])
    np.testing.assert_array_equal(got['multiplicity'], [len(g) for g in groups])
    np.testing.assert_allclose(got['sum_f_squared'], [f_squared[g].sum() for g in groups], atol=1e-15)
    np.testing.assert_allclose(got['radial_weight'],
                               [(bragg_weight[g] / (4 * np.pi * q[g]**2)).sum() for g in groups],
                               rtol=2e-15, atol=1e-17)


def test_radial_analytic_diffuse_matches_closed_form_mixed_size_dimer():
    spacing, sigma = 1.2, .37
    atom = AtomCloud(('Au',), [[0., 0., 0.]])
    dimer = AtomCloud(('Au', 'Au'), [[-.5 * spacing, 0., 0.], [.5 * spacing, 0., 0.]])
    library = ParticleLibrary((atom, dimer), [.25, .75])
    probabilities = np.array([.2, .8])
    population = PorePopulation([[1., 2., 3.], [-3., .2, 1.]], library,
                                occupancy=probabilities, displacement_sigma_A=sigma)
    q = np.linspace(0., 12., 97)
    factor = form_factors(('Au',), q)['Au']
    radial = factor * (.25 + .75 * 2 * np.sinc(q * spacing / (2 * np.pi)))
    internal = factor**2 * (.25 + .75 * 2 * (1 + np.sinc(q * spacing / np.pi)))
    expected = sum(p * internal - p*p * radial**2 * np.exp(-(sigma*q)**2) for p in probabilities)
    for grid in [(1, 1), (4, 8), (24, 48)]:
        got = powder_diffuse(population, q, n_mu=grid[0], n_phi=grid[1], cell=np.eye(3) * 30)
        np.testing.assert_allclose(got, expected, rtol=3e-14, atol=3e-11)


@pytest.mark.parametrize('kwargs,match', [({'orientation': 'fixed'}, 'Haar-uniform'),
                                       ({'displacement_covariance_A2': np.diag([1., 2., 3.])}, 'isotropic'),
                                       ({'pore_radius_A': .3}, 'Hard-wall')])
def test_radial_diffuse_retains_analytic_model_restrictions(kwargs, match):
    atom = AtomCloud(('Au',), [[0., 0., 0.]])
    population = PorePopulation([[0., 0., 0.]], ParticleLibrary((atom,), [1.]), **kwargs)
    with pytest.raises(ValueError, match=match):
        powder_diffuse(population, [0., 1.], cell=np.eye(3) * 30)


@pytest.mark.parametrize('grid', [(0, 8), (4, True), (4, 1.5)])
def test_radial_diffuse_retains_grid_validation(grid):
    atom = AtomCloud(('Au',), [[0., 0., 0.]])
    population = PorePopulation([[0., 0., 0.]], ParticleLibrary((atom,), [1.]))
    with pytest.raises(ValueError, match='positive integer'):
        powder_diffuse(population, [0., 1.], n_mu=grid[0], n_phi=grid[1])
