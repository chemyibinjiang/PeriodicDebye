"""Independent physical references for the approximate finite Debye backend."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

from mofdebye.core import AtomCloud, form_factors

pytest.importorskip('numba', reason='The radial moment backend uses the optional fast extra.')
from mofdebye.large_powder import LargePowderCache


_path = Path(__file__).parents[1] / 'scripts' / 'validate_large_powder.py'
_spec = importlib.util.spec_from_file_location('large_powder_physics_reference', _path)
_validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validation)


def _explicit_ordered_pairs(basis, cell, mask, q):
    """No production Debye/cache/symmetry code; every atom pair is explicit."""
    shifts = np.argwhere(mask) @ cell
    positions = (shifts[:, None, :] + basis.xyz[None, :, :]).reshape(-1, 3)
    elements = basis.elements * len(shifts)
    ff = form_factors(elements, q)
    factors = np.array([ff[element] for element in elements])
    radii = np.linalg.norm(positions[:, None, :] - positions[None, :, :], axis=2)
    return np.array([np.sum(factors[:, i, None] * factors[None, :, i] *
                           np.sinc(value*radii/np.pi)) for i, value in enumerate(q)])


@pytest.mark.parametrize('geometry', ['cube', 'cuboid', 'hole', 'skew', 'left_handed'])
def test_species_resolved_powder_matches_explicit_ordered_pairs(geometry):
    basis = AtomCloud(('Zn', 'C', 'O', 'H'),
                      [[.23, .12, .41], [1.1, .8, .2], [1.6, 1.7, .9], [.3, 1.4, 2.1]])
    cell = np.eye(3)*3.7
    mask = np.ones((2, 2, 2), dtype=int)
    if geometry == 'cuboid':
        mask = np.ones((2, 3, 1), dtype=int)
    elif geometry == 'hole':
        mask[0, 1, 0] = 0
    elif geometry == 'skew':
        cell = np.array([[3.7, .6, -.2], [0., 4.1, .3], [.4, .2, 3.5]])
    elif geometry == 'left_handed':
        cell[1, 1] *= -1
    q = np.linspace(0, 6, 121)
    expected = _explicit_ordered_pairs(basis, cell, mask, q)
    cache = LargePowderCache(basis, cell, mask, q, bin_width=.04, order=8, threads=1, merge_decimals=12)
    actual, bound = cache.iq()
    metrics = _validation.compare_powder_curves(q, actual, expected)
    assert metrics['intensity_checks_passed'], metrics
    np.testing.assert_allclose(actual, expected, rtol=3e-9, atol=1e-7)
    # The analytic bound excludes floating-point roundoff; account for that
    # explicitly rather than describing the bound as covering every error.
    roundoff_allowance = 2e-12 * expected[0]
    assert np.all(abs(actual-expected) <= bound + roundoff_allowance)


def test_single_atom_has_no_shape_or_origin_dependence():
    basis = AtomCloud(('Zn',), [[1.234, -2.7, .5]])
    q = np.linspace(0, 10, 201)
    expected = form_factors(basis.elements, q)['Zn']**2
    actual, bound = LargePowderCache(basis, np.eye(3)*4, np.ones((1, 1, 1)), q,
                                   bin_width=.04, order=8, threads=1).iq()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-10)
    assert np.isfinite(bound).all() and np.all(bound >= 0)


def test_identical_position_species_keep_self_and_cross_terms():
    basis = AtomCloud(('C', 'C', 'O', 'Zn'), np.zeros((4, 3)))
    q = np.linspace(0, 6, 121)
    factors = form_factors(basis.elements, q)
    expected = sum(factors[element] for element in basis.elements)**2
    actual, bound = LargePowderCache(basis, np.eye(3)*4, np.ones((1, 1, 1)), q,
                                   bin_width=.04, order=8, threads=1).iq()
    np.testing.assert_allclose(actual, expected, rtol=2e-12, atol=1e-9)


def test_global_rotation_translation_does_not_change_powder():
    basis = AtomCloud(('C', 'O', 'Zn'), [[.1, .3, .4], [1.4, .2, 1.], [2., 1.8, .5]])
    cell = np.array([[3.7, .6, -.2], [0., 4.1, .3], [.4, .2, 3.5]])
    angle = .713
    rotation = np.array([[np.cos(angle), -np.sin(angle), 0.],
                         [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]])
    mask = np.ones((2, 3, 2), dtype=int)
    mask[1, 1, 0] = 0
    q = np.linspace(0, 8, 161)
    moved = basis.moved([17.1, -9.3, 13.2], rotation=rotation)
    a, _ = LargePowderCache(basis, cell, mask, q, bin_width=.04, order=8, threads=1).iq()
    b, _ = LargePowderCache(moved, cell@rotation.T, mask, q,
                           bin_width=.04, order=8, threads=1).iq()
    np.testing.assert_allclose(a, b, rtol=4e-9, atol=1e-7)


def test_103nm_simple_cube_matches_exact_integer_shell_oracle():
    """Full 6001-point geometric extent, independently grouped by integer shells."""
    basis = AtomCloud(('C',), [[.37, 1.23, .98]])
    spacing = 25.832
    q = np.linspace(0, 6, 6001)
    factors = form_factors(basis.elements, q)['C']
    expected = _validation.integer_cube_iq(q, 40, spacing, factors)
    actual, bound = LargePowderCache(basis, np.eye(3)*spacing,
                                   np.ones((40, 40, 40), dtype=int), q,
                                   bin_width=.04, order=8, threads=1).iq()
    metrics = _validation.compare_powder_curves(q, actual, expected)
    assert metrics['intensity_checks_passed'], metrics
    assert metrics['max_peak_q_shift_A_inv'] <= .00100000001
    assert metrics['forward_relative_error'] < 2e-12
    assert np.isfinite(bound).all() and np.all(bound >= 0)
