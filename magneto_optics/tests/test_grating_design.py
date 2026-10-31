"""Physical meaning of grating screening and incoherent spectrum metrics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from magnetooptics.grating_design import (nbk7_index, prototype_layers,
    relative_rotation, transmission_metrics, stokes_band_average)


def _scan_module():
    path = Path(__file__).resolve().parents[1] / "scripts" / "scan_grating.py"
    specification = importlib.util.spec_from_file_location("tested_scan_grating", path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


@pytest.mark.parametrize("axis", [0, 1])
@pytest.mark.parametrize("angle", [-89., -12., 0., 30., 89.])
def test_relative_rotation_removes_incident_axis_and_global_phase(axis, angle):
    radians = np.deg2rad(angle)
    rotation = np.array([[np.cos(radians), -np.sin(radians)],
                         [np.sin(radians), np.cos(radians)]])
    actual, ellipticity = relative_rotation(3.4*np.exp(.93j)*rotation[:, axis], axis)
    assert_allclose(actual, angle, atol=3e-13)
    assert_allclose(ellipticity, 0., atol=2e-13)


def test_zero_order_metrics_include_port_flux_and_distinguish_diffraction():
    angle = np.deg2rad(20.)
    transmission = .5*np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    result = dict(transmission=transmission, n_incident=1., n_exit=1.5,
                  transmitted_power=np.array([.6, .6]), reflected_power=np.array([.3, .3]),
                  absorbed_power=np.array([.1, .1]))
    for axis in (0, 1):
        actual = transmission_metrics(result, axis)
        assert_allclose(actual["rotation_deg"], 20., atol=2e-13)
        assert_allclose(actual["zero_order_transmission"], .375, atol=2e-15)
        assert_allclose(actual["total_transmission"], .6, atol=2e-15)
        assert_allclose(actual["cross_power"], .375*np.sin(angle)**2, atol=2e-15)
        assert_allclose(actual["fom_deg"], 20*np.sqrt(.375), atol=2e-13)
        assert actual["qualified"]


@pytest.mark.parametrize("field", [[0., 0.], [.01, .01], [1., 1j]])
def test_dark_or_circular_field_cannot_win_rotation_screen(field):
    result = dict(transmission=np.column_stack((field, field)), n_incident=1., n_exit=1.,
                  transmitted_power=np.ones(2), reflected_power=np.zeros(2), absorbed_power=np.zeros(2))
    assert not transmission_metrics(result, 0)["qualified"]


def test_seed_ranking_excludes_unqualified_large_angle():
    common = dict(period_nm=930., fill_fraction=.6, height_nm=593., cap_nm=50., incident_axis=0)
    rows = [dict(**common, qualified=True, fom_deg=.4, marker="usable"),
            dict(**common, qualified=False, fom_deg=60., marker="dark or circular")]
    assert _scan_module().choose_seeds(rows, 3)[0]["marker"] == "usable"
    assert len(_scan_module().choose_seeds(rows, 3)) == 1


def test_incoherent_average_combines_stokes_not_complex_fields_or_angles():
    # Weighted intensities are 1 and .75; the perpendicular components are
    # incoherent, so S2=S3=0, unlike a sum of these complex field amplitudes.
    result = stokes_band_average([[2., 0.], [0., 1j]], [1., 3.], 0)
    assert_allclose(result["stokes"], [1.75, .25, 0., 0.], atol=2e-15)
    assert_allclose(result["degree_of_polarization"], 1/7, atol=2e-15)
    assert result["rotation_deg"] == 0.
    assert result["ellipticity_deg"] == 0.


def test_equal_opposite_circular_spectra_are_unpolarized():
    fields = np.array([[1., 1j], [1., -1j]])/np.sqrt(2)
    result = stokes_band_average(fields, [1., 1.], 0)
    assert_allclose(result["power"], 1., atol=2e-15)
    assert result["degree_of_polarization"] == 0.
    assert result["rotation_deg"] is None
    assert result["ellipticity_deg"] is None


@pytest.mark.parametrize("axis", [0, 1])
def test_identical_polarization_spectrum_retains_ellipse_and_loses_phase(axis):
    field = np.array([.9, .2+.1j]) if axis == 0 else np.array([-.2-.1j, .9])
    weights = np.array([1., 2., 3.])
    amplitudes = np.array([.3, .7, .4])*np.exp(1j*np.array([.2, 2.3, -1.8]))
    average = stokes_band_average(amplitudes[:, None]*field, weights, axis)
    theta, chi = relative_rotation(field, axis)
    assert_allclose(average["rotation_deg"], theta, atol=2e-13)
    assert_allclose(average["ellipticity_deg"], chi, atol=2e-13)
    assert_allclose(average["degree_of_polarization"], 1., atol=2e-15)
    assert_allclose(average["power"], weights @ abs(amplitudes)**2 / weights.sum()
                    * np.vdot(field, field).real, atol=2e-15)


@pytest.mark.parametrize("weights", [[0., 0.], [-1., 2.], [np.nan, 1.], [1.]])
def test_invalid_spectral_weight_rejected(weights):
    with pytest.raises(ValueError):
        stokes_band_average([[1., 0.], [0., 1.]], weights, 0)


def test_geometry_preserves_fixed_tensor_and_accounts_for_cap_and_overetch_volume():
    geometry = dict(period_nm=966., fill_fraction=608/966, height_nm=593., cap_nm=50., overetch_nm=30.)
    epsilon, coupling, substrate = 2.856064+.02028j, -.00375-.0018j, 1.5
    layers = prototype_layers(geometry, epsilon, coupling, substrate)
    assert [layer["thickness_nm"] for layer in layers] == [50., 593., 30.]
    assert layers[0]["fill_fraction"] == 1.
    assert layers[0]["epsilon_a"] == epsilon
    assert layers[0]["g_a"] == coupling
    assert layers[1]["epsilon_a"] == 1.95**2
    assert layers[2]["epsilon_a"] == substrate**2
    for layer in layers:
        assert layer["epsilon_b"] == epsilon
        assert layer["g_b"] == coupling
    thickness = sum(layer["thickness_nm"] for layer in layers)
    mo_volume_per_area = sum(layer["thickness_nm"] * (
        layer["fill_fraction"]*(layer["g_a"] == coupling)
        + (1-layer["fill_fraction"])*(layer["g_b"] == coupling)) for layer in layers)
    assert_allclose(thickness, 673., atol=2e-13)
    assert_allclose(mo_volume_per_area, 50.+(1-608/966)*(593.+30.), atol=2e-13)


@pytest.mark.parametrize("wavelength", [1549., 1571., np.nan, np.inf, 1550+1j])
def test_substrate_proxy_does_not_extend_material_study_band(wavelength):
    with pytest.raises(ValueError):
        nbk7_index(wavelength)
