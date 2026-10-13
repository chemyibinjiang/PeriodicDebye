"""Validate the independent oracle and error metrics before using them as evidence."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


_path = Path(__file__).parents[1] / 'scripts' / 'validate_large_powder.py'
_spec = importlib.util.spec_from_file_location('large_powder_validation', _path)
_validation = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_validation)
integer_cube_shells = _validation.integer_cube_shells
integer_cube_iq = _validation.integer_cube_iq
compare_powder_curves = _validation.compare_powder_curves


def test_two_by_two_cube_has_closed_form_shell_multiplicities():
    radii, counts = integer_cube_shells(2, 2.3)
    np.testing.assert_allclose(radii, 2.3 * np.sqrt([0, 1, 2, 3]))
    np.testing.assert_array_equal(counts, [8, 24, 24, 8])


@pytest.mark.parametrize('cells', [1, 2, 3, 4])
def test_integer_shell_oracle_matches_explicit_ordered_atom_pairs(cells):
    spacing = 2.3
    xyz = np.array(list(np.ndindex((cells, cells, cells))), dtype=float) * spacing
    distances = np.linalg.norm(xyz[:, None, :] - xyz[None, :, :], axis=2)
    q = np.r_[0., 1e-8, np.linspace(.1, 15, 53)]
    factor = 3.1 * np.exp(-q*q/40)
    expected = np.array([np.sinc(value * distances / np.pi).sum() for value in q]) * factor**2
    actual = integer_cube_iq(q, cells, spacing, factor, q_chunk=11)
    np.testing.assert_allclose(actual, expected, rtol=1e-13, atol=2e-10)


def test_large_cube_pair_counts_and_self_terms_use_int64():
    radii, counts = integer_cube_shells(40, 25.832)
    assert radii[0] == 0
    assert counts[0] == 40**3
    assert int(counts.sum()) == 40**6
    assert counts.dtype == np.int64
    assert integer_cube_iq([0.], 40, 25.832)[0] == 40**6


def test_forward_beam_cannot_hide_nonforward_errors():
    q = np.array([0., .1, 1., 2.])
    reference = np.array([1e20, 2., 1., 1.5])
    actual = np.array([1e20, 2.02, 1.01, 1.515])
    result = compare_powder_curves(q, actual, reference)
    assert result['forward_relative_error'] == 0
    assert result['max_abs_error_over_nonforward_peak'] == pytest.approx(.01)
    assert result['max_relative_error_above_floor'] == pytest.approx(.01)
    assert not result['intensity_checks_passed']


def test_extinction_regions_are_measured_without_clipping_or_division_by_zero():
    q = np.array([.1, .2, .3, .4, .5])
    reference = np.array([0., 1., 0., .5, 0.])
    actual = reference.copy()
    actual[2] = -.0002
    result = compare_powder_curves(q, actual, reference)
    assert result['minimum_actual'] == -.0002
    assert result['max_abs_error_over_nonforward_peak'] == .0002
    assert result['max_relative_error_above_floor'] == 0
    assert result['pointwise_points_above_floor'] == 2


def test_peak_and_integral_metrics_identify_shifted_curve():
    q = np.linspace(.1, 2, 1901)
    reference = np.exp(-((q-.7)/.04)**2) + .4*np.exp(-((q-1.4)/.07)**2)
    actual = np.exp(-((q-.702)/.04)**2) + .4*np.exp(-((q-1.4)/.07)**2)
    result = compare_powder_curves(q, actual, reference, reference_kind='refinement')
    assert result['reference_kind'] == 'refinement'
    assert result['peak_count'] == 2
    assert result['peaks'][0]['q_shift_A_inv'] == pytest.approx(.002)
    assert result['relative_integral_error'] < 1e-13
    assert result['relative_integrated_absolute_error'] > .01
    assert not result['intensity_checks_passed']


@pytest.mark.parametrize('replacement', [np.nan, np.inf, -np.inf])
def test_nonfinite_curve_cannot_pass(replacement):
    with pytest.raises(ValueError):
        compare_powder_curves([0., .1, .2], [1., replacement, 3.], [1., 2., 3.])


def test_identity_curves_have_zero_errors():
    q = np.linspace(0, 6, 121)
    reference = 3 + np.cos(q*5)
    result = compare_powder_curves(q, reference, reference)
    assert result['intensity_checks_passed']
    assert result['max_relative_error_above_floor'] == 0
    assert result['relative_integrated_absolute_error'] == 0
    assert all(p['q_shift_A_inv'] == 0 and p['relative_area_error'] == 0 for p in result['peaks'])
