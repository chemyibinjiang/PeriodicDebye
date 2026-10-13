"""Finite-to-bulk integration through the production directional/cell APIs."""
import numpy as np

from mofdebye.core import AtomCloud, form_factors
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.moments import guest_moments
from mofdebye.periodic import bragg_reflections, finite_cell_ensemble, reciprocal_lattice


def reciprocal_window(cell, hkl, points):
    """Midpoint quadrature of one reciprocal primitive parallelepiped about G."""
    angles = -np.pi + (np.arange(points) + .5) * 2 * np.pi / points
    reduced = np.stack(np.meshgrid(angles, angles, angles, indexing='ij'), axis=-1).reshape(-1, 3)
    reciprocal = reciprocal_lattice(cell)
    vectors = (reduced / (2 * np.pi) + np.asarray(hkl)) @ reciprocal
    volume = abs(np.linalg.det(reciprocal))
    return vectors, reduced, volume


def target_reflection(table, hkl):
    rows = np.flatnonzero(np.all(table['hkl'] == hkl, axis=1))
    assert len(rows) == 1
    return rows[0]


def test_production_finite_crystals_converge_to_bulk_bragg_measure():
    cell = np.array([[4., 0., 0.], [.8, 5., 0.], [.3, .7, 6.]])
    host = AtomCloud(('C',), [[.2, -.1, .4]])
    target = np.array([1, 0, 0])
    vectors, reduced, reciprocal_volume = reciprocal_window(cell, target, points=20)
    factors_squared = form_factors(('C',), np.linalg.norm(vectors, axis=1))['C'] ** 2
    target_q = np.linalg.norm(target @ reciprocal_lattice(cell))
    bragg = bragg_reflections(host, cell, target_q + .01)
    row = target_reflection(bragg, target)
    factor_at_G_squared = form_factors(('C',), [bragg['q'][row]])['C'][0] ** 2
    bulk_mass = bragg['bragg_weight'][row] / factor_at_G_squared
    np.testing.assert_allclose(bulk_mass, reciprocal_volume, rtol=2e-14)
    empty_moments = guest_moments(None, vectors)
    errors = []
    # Divide out the known q-dependent factor, yielding a constant-amplitude
    # cell. The quadrature integrates the production |S|^2/M exactly because its
    # largest Fourier order is below the grid resolution. The nonconstant smooth
    # observable below then probes weak convergence to the Bragg delta at G.
    observable = np.prod(np.cos(reduced), axis=1)
    for n in (2, 4, 8):
        mask = np.ones((n, n, n), dtype=bool)
        finite = finite_cell_ensemble(host, cell, mask, vectors, empty_moments)
        per_cell_density = finite['total'] / (int(mask.sum()) * factors_squared)
        integrated_mass = reciprocal_volume * np.mean(per_cell_density)
        np.testing.assert_allclose(integrated_mass, bulk_mass, rtol=2e-13, atol=1e-13)
        smooth_integral = reciprocal_volume * np.mean(per_cell_density * observable)
        # The observable equals one at G, so its bulk integral is bulk_mass.
        errors.append(abs(smooth_integral / bulk_mass - 1))
        np.testing.assert_allclose(smooth_integral / bulk_mass, (1 - 1 / n) ** 3,
                                   rtol=2e-13, atol=1e-13)
        np.testing.assert_array_equal(finite['diffuse'], np.zeros(len(vectors)))
    assert np.all(np.diff(errors) < 0)
    assert errors[-1] < .34


def test_production_occupancy_disorder_integrates_bragg_plus_diffuse_per_cell():
    cell = np.array([[4., 0., 0.], [.8, 5., 0.], [.3, .7, 6.]])
    host = AtomCloud((), np.empty((0, 3)))
    guest = AtomCloud(('Au',), [[0., 0., 0.]])
    probability = .37
    population = PorePopulation([[.4, .2, .1]], ParticleLibrary((guest,), [1.]), occupancy=probability)
    target = np.array([1, 0, 0])
    vectors, reduced, reciprocal_volume = reciprocal_window(cell, target, points=12)
    factors_squared = form_factors(('Au',), np.linalg.norm(vectors, axis=1))['Au'] ** 2
    target_q = np.linalg.norm(target @ reciprocal_lattice(cell))
    bragg = bragg_reflections(host, cell, target_q + .01, population=population)
    row = target_reflection(bragg, target)
    factor_at_G_squared = form_factors(('Au',), [bragg['q'][row]])['Au'][0] ** 2
    bulk_bragg_mass = bragg['bragg_weight'][row] / factor_at_G_squared
    bulk_diffuse_density = bragg['diffuse_at_reflections'][row] / factor_at_G_squared
    np.testing.assert_allclose(bulk_bragg_mass, reciprocal_volume * probability ** 2, rtol=2e-14)
    np.testing.assert_allclose(bulk_diffuse_density, probability * (1 - probability), rtol=2e-14)
    moments = guest_moments(population, vectors)
    mask = np.ones((4, 4, 4), dtype=bool)
    finite = finite_cell_ensemble(host, cell, mask, vectors, moments)
    normalization = int(mask.sum()) * factors_squared
    integrated_coherent = reciprocal_volume * np.mean(finite['coherent'] / normalization)
    integrated_diffuse = reciprocal_volume * np.mean(finite['diffuse'] / normalization)
    integrated_total = reciprocal_volume * np.mean(finite['total'] / normalization)
    np.testing.assert_allclose(integrated_coherent, bulk_bragg_mass, rtol=2e-13)
    np.testing.assert_allclose(integrated_diffuse, reciprocal_volume * bulk_diffuse_density, rtol=2e-13)
    np.testing.assert_allclose(integrated_total, bulk_bragg_mass + reciprocal_volume * bulk_diffuse_density,
                               rtol=2e-13)
    # A smooth observable with zero uniform-cell integral leaves no diffuse
    # contribution. This detects an erroneous |S|^2 scaling of disorder variance.
    observable = np.prod(np.cos(reduced), axis=1)
    diffuse_observable = reciprocal_volume * np.mean(finite['diffuse'] / normalization * observable)
    np.testing.assert_allclose(diffuse_observable, 0., atol=1e-14)
