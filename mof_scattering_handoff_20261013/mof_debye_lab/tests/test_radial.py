"""Independent sinc references for controlled radial moment interpolation."""
from math import factorial
import numpy as np
import pytest

from mofdebye.radial import RadialMomentHistogram


def exact_sum(radii, weights, q):
    return np.sum(weights[:, None] * np.sinc(radii[:, None] * q[None, :] / np.pi), axis=0)


@pytest.mark.parametrize('order', [4, 5, 8, 10, 12])
def test_random_radial_sum_and_forward_limit(order):
    rng = np.random.default_rng(87342)
    q = np.linspace(0, 6, 601)
    radii = np.r_[0, 0, rng.uniform(0.1, 100, 400)]
    weights = rng.integers(1, 10000, len(radii))
    histogram = RadialMomentHistogram(q, 100, order=order)
    histogram.add(radii[:111], weights[:111]).add(radii[111:], weights[111:])
    result, bound = histogram.evaluate()
    expected = exact_sum(radii, weights, q)
    assert result[0] == int(weights.sum())
    assert bound[0] == 0
    assert np.all(np.abs(result - expected) <= bound + 3e-13 * weights.sum())
    assert histogram.dr <= 0.04
    assert np.isfinite(result).all()


def test_nontrivial_truncation_bound_and_order_convergence():
    q = np.linspace(0, 6, 301)
    radii = np.array([0.033, 0.14, 2.339, 3.856, 9.005, 25.55, 70.19])
    weights = np.array([5., 7., 13., 31., 6., 89., 9.])
    expected = exact_sum(radii, weights, q)
    errors = []
    for order in (4, 8, 12):
        histogram = RadialMomentHistogram(q, 71, bin_width=0.4, order=order).add(radii, weights)
        values, bound = histogram.evaluate()
        error = np.abs(values - expected)
        assert np.all(error <= bound + 2e-11)
        assert bound[-1] > 0
        errors.append(np.max(error))
    assert errors[0] > 1e-6
    assert errors[1] < errors[0] / 100
    assert errors[2] < errors[1] / 100


def test_very_short_positive_radius_does_not_pollute_transform():
    q = np.linspace(0, 6, 601)
    radii = np.array([0, 1e-100, 1e-14, 0.001, 2.37, 12.65])
    weights = np.array([7., 31., 8., 6., 2., 1.])
    histogram = RadialMomentHistogram(q, 13).add(radii, weights)
    values, bound = histogram.evaluate()
    expected = exact_sum(radii, weights, q)
    np.testing.assert_allclose(values, expected, rtol=3e-14, atol=3e-13)
    assert np.isfinite(bound).all()


def test_self_terms_empty_histogram_and_zero_weights():
    histogram = RadialMomentHistogram(np.linspace(0, 6, 101), 0)
    assert np.array_equal(histogram.evaluate()[0], np.zeros(101))
    histogram.add(np.array([0., 0.]), np.array([17., 0.]))
    assert np.array_equal(histogram.evaluate()[0], np.full(101, 17.))
    assert np.array_equal(histogram.evaluate()[1], np.zeros(101))


def test_prebuilt_moment_contract_and_copy():
    q = np.linspace(0, 6, 301)
    first = RadialMomentHistogram(q, 40).add(np.array([0., 0.01, 2.5, 39.3]), np.array([9., 2., 13., 31.]))
    second = RadialMomentHistogram(q, 40)
    second.load_moments(first.moments, first.total_weight, first.zero_weight,
                        first.inverse_radius_weight, first.remainder_mass)
    for expected, actual in zip(first.evaluate(), second.evaluate()):
        np.testing.assert_array_equal(actual, expected)
    saved = second.evaluate()[0]
    first.moments[:] = 0
    np.testing.assert_array_equal(second.evaluate()[0], saved)


def test_conservative_half_bin_remainder():
    q = np.linspace(0, 6, 101)
    h = RadialMomentHistogram(q, 30).add(np.array([2., 13.3, 26.]))
    conservative = RadialMomentHistogram(q, 30).load_moments(
        h.moments, h.total_weight, h.zero_weight, h.inverse_radius_weight)
    np.testing.assert_allclose(conservative.evaluate()[1][1:],
                               h.inverse_radius_weight * (h.dr / 2)**9 * q[1:]**8 / factorial(9))
    assert np.all(conservative.evaluate()[1] >= h.evaluate()[1])


@pytest.mark.parametrize('q', [[0], [1, 2], [0, 0], [0, 2, 1], [0, 1, 2.001], [0, 1e-12]])
def test_invalid_or_impractical_q_grid(q):
    with pytest.raises(ValueError):
        RadialMomentHistogram(q, 0)


@pytest.mark.parametrize('options', [
    {'max_radius': -1}, {'max_radius': np.inf}, {'max_radius': True},
    {'bin_width': 0}, {'bin_width': np.nan}, {'bin_width': True},
    {'order': 3}, {'order': 13}, {'order': 8.0}, {'order': True},
])
def test_invalid_histogram_options(options):
    kwargs = {'max_radius': 1}
    kwargs.update(options)
    with pytest.raises(ValueError):
        RadialMomentHistogram(np.linspace(0, 6, 101), **kwargs)


def test_explicit_coarse_q_guard_instead_of_radius_folding():
    with pytest.raises(ValueError, match='coarse'):
        RadialMomentHistogram([0, 6], 100)


@pytest.mark.parametrize('radii,weights', [
    ([-1], [1]), ([np.nan], [1]), ([[1]], [[1]]), ([1], [-1]),
    ([1], [np.inf]), ([1, 2], [1]), ([11], [1]),
])
def test_invalid_distance_blocks_do_not_mutate_histogram(radii, weights):
    histogram = RadialMomentHistogram(np.linspace(0, 6, 101), 10)
    with pytest.raises(ValueError):
        histogram.add(radii, weights)
    assert histogram.total_weight == 0
    assert not np.any(histogram.moments)


@pytest.mark.parametrize('field,value', [
    ('total_weight', -1), ('zero_weight', 20), ('inverse_radius_weight', -1),
    ('inverse_radius_weight', 15), ('remainder_mass', -1),
])
def test_invalid_external_moments(field, value):
    histogram = RadialMomentHistogram(np.linspace(0, 6, 101), 10)
    histogram.add(np.array([0., 1.]), np.array([1., 1.]))
    options = dict(total_weight=2., zero_weight=1., inverse_radius_weight=1., remainder_mass=0.)
    options[field] = value
    with pytest.raises(ValueError):
        histogram.load_moments(histogram.moments, **options)
