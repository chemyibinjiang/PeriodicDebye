import numpy as np
import pytest

from magnetooptics.periodic_geometry import RectangularCell
from magnetooptics.periodic_observables import diffraction_orders, evaluate_ports, jones_matrices, polarization_ellipse


def test_subwavelength_cell_has_only_forward_backward_zero_order():
    orders = diffraction_orders([200, 300], 600)
    np.testing.assert_array_equal(orders["indices"], [[0, 0]])
    np.testing.assert_array_equal(orders["directions_transmitted"], [[0, 0, 1]])
    np.testing.assert_array_equal(orders["directions_reflected"], [[0, 0, -1]])
    assert orders["zero_index"] == 0
    assert orders["flux_weights"][0] == 1


def test_all_nine_rectangular_orders_and_their_flux_factors():
    orders = diffraction_orders([900, 700], 500)
    expected = {(m, n) for m in (-1, 0, 1) for n in (-1, 0, 1)}
    assert set(map(tuple, orders["indices"])) == expected
    for (m, n), weight, direction in zip(orders["indices"], orders["flux_weights"], orders["directions_transmitted"]):
        cosine = np.sqrt(1 - (500 * m / 900)**2 - (500 * n / 700)**2)
        assert weight == pytest.approx(cosine, rel=2e-15)
        assert direction[2] == pytest.approx(cosine, rel=2e-15)
        assert np.linalg.norm(direction) == pytest.approx(1, abs=3e-16)


@pytest.mark.parametrize("offset", [0, -1e-10, 1e-10])
def test_grazing_order_rejected_from_both_sides(offset):
    with pytest.raises(ValueError, match="Rayleigh/grazing"):
        diffraction_orders([600 * (1 + offset), 200], 600)


def test_order_memory_guard_and_host_wavelength_are_applied():
    with pytest.raises(ValueError, match="max_orders"):
        diffraction_orders([1e9, 1e9], 500, max_orders=100)
    with pytest.raises(ValueError, match="Rayleigh/grazing"):
        diffraction_orders([400, 200], 600, host_index=1.5)


def test_transmitted_zero_order_contains_incident_and_reflected_does_not():
    cell = RectangularCell([200, 300], [[10, 20, 0]], 5)
    dipoles = np.zeros((1, 3), complex)
    incident = np.array([[1., 0, 0]])
    ports = evaluate_ports(cell, dipoles, incident, incident, 600)
    assert ports["total_transmission"] == 1
    assert ports["total_reflection"] == 0
    assert ports["absorption"] == 0
    assert ports["energy_balance_residual"] == 0
    np.testing.assert_array_equal(ports["zero_transmitted"], [1, 0, 0])
    np.testing.assert_array_equal(ports["zero_reflected"], [0, 0, 0])


def test_prescribed_uniform_dipole_sheet_has_known_i_k_over_2_area_amplitude():
    cell = RectangularCell([200, 300], [[10, 20, 0]], 5)
    dipoles = np.array([[120 + 70j, -15j, 123]], dtype=complex)
    incident = np.array([[1., 0, 0]])
    wavelength = 600
    k = 2 * np.pi / wavelength
    ports = evaluate_ports(cell, dipoles, incident, incident, wavelength)
    expected_reflected = 1j * k / (2 * cell.area_nm2) * np.array([dipoles[0, 0], dipoles[0, 1], 0])
    np.testing.assert_allclose(ports["zero_reflected"], expected_reflected, rtol=2e-15, atol=0)
    np.testing.assert_allclose(ports["zero_transmitted"], expected_reflected + incident[0], rtol=2e-15, atol=0)
    expected_absorption = k / cell.area_nm2 * (dipoles[0, 0].imag - k**3 * np.sum(np.abs(dipoles)**2) / (6 * np.pi))
    assert ports["absorption"] == pytest.approx(expected_absorption, rel=2e-15)


def test_diffraction_phase_transversality_and_power_for_prescribed_x_dipole():
    cell = RectangularCell([900, 700], [[130, 230, 0]], 5)
    dipoles = np.array([[1000 + 200j, 0, 0]])
    incident = np.array([[0., 1, 0]])
    k = 2 * np.pi / 500
    orders = diffraction_orders(cell.periods_nm, 500)
    ports = evaluate_ports(cell, dipoles, incident, incident, 500, orders=orders)
    for i, ((m, n), direction, kz) in enumerate(zip(orders["indices"], orders["directions_reflected"], orders["kz"])):
        dx, dy, dz = direction
        phase = np.exp(-2j * np.pi * (m * 130 / 900 + n * 230 / 700))
        expected = 1j * k**2 / (2 * cell.area_nm2 * kz) * dipoles[0, 0] * phase * np.array([1 - dx**2, -dx * dy, -dx * dz])
        np.testing.assert_allclose(ports["reflected_fields"][i], expected, rtol=3e-15, atol=1e-20)
        assert abs(direction @ ports["reflected_fields"][i]) < 1e-19
        expected_power = kz / k * abs(1j * k**2 * dipoles[0, 0] / (2 * cell.area_nm2 * kz))**2 * (1 - dx**2)
        assert ports["reflected_power"][i] == pytest.approx(expected_power, rel=3e-15)


def test_incident_amplitude_scaling_leaves_power_ratios_and_absorption_unchanged():
    cell = RectangularCell([250, 270], [[20, 30, 0]], 5)
    incident = np.array([[1., 0, 0]], complex)
    dipoles = np.array([[800 + 250j, 20 - 10j, 15]], complex)
    local = np.array([[1.2 + 0.1j, 0.1j, 0.2]], complex)
    scale = 2 - 3j
    reference = evaluate_ports(cell, dipoles, incident, local, 600)
    scaled = evaluate_ports(cell, scale * dipoles, scale * incident, scale * local, 600)
    for key in ("total_transmission", "total_reflection", "absorption", "energy_balance_residual"):
        assert scaled[key] == pytest.approx(reference[key], rel=3e-12, abs=2e-16)
    np.testing.assert_allclose(scaled["zero_transmitted"], scale * reference["zero_transmitted"], rtol=2e-15)


def test_jones_identity_for_empty_contrast_and_unit_source_validation():
    cell = RectangularCell([200, 300], [[10, 20, 0]], 5)
    zero = np.zeros((1, 3), complex)
    x = evaluate_ports(cell, zero, [1, 0, 0], [[1, 0, 0]], 600)
    y = evaluate_ports(cell, zero, [0, 1, 0], [[0, 1, 0]], 600)
    jones = jones_matrices(x, y)
    np.testing.assert_array_equal(jones["transmission"], np.eye(2))
    np.testing.assert_array_equal(jones["reflection"], np.zeros((2, 2)))
    with pytest.raises(ValueError, match="unit x"):
        jones_matrices(y, x)


def test_cached_order_mismatch_and_nontransverse_incidence_rejected():
    cell = RectangularCell([200, 300], [[10, 20, 0]], 5)
    zero = np.zeros((1, 3), complex)
    wrong = diffraction_orders([300, 200], 600)
    with pytest.raises(ValueError, match="cached"):
        evaluate_ports(cell, zero, [1, 0, 0], zero, 600, orders=wrong)
    with pytest.raises(ValueError, match="normal-incidence"):
        evaluate_ports(cell, zero, [0, 0, 1], zero, 600)


@pytest.mark.parametrize("sign", [-1, 1])
def test_circular_port_ellipticity_has_no_ellipse_axis(sign):
    result = polarization_ellipse([1, sign * 1j])
    assert result["rotation_deg"] is None
    assert result["ellipticity_deg"] == pytest.approx(sign * 45)
    assert result["cross_fraction"] == 0.5


def test_ellipse_angle_branch_is_modulo_180_and_zero_field_undefined():
    for angle in (-89, -32, 21, 89):
        rad = np.radians(angle)
        result = polarization_ellipse([np.cos(rad), np.sin(rad)])
        assert result["rotation_deg"] == pytest.approx(angle)
        assert result["ellipticity_deg"] == 0
    assert polarization_ellipse([0, 0]) == {"rotation_deg": None, "ellipticity_deg": None, "cross_fraction": None}


def test_nonmagnetic_anisotropic_cell_can_rotate_light_but_has_no_magnetic_odd_signal():
    from magnetooptics.periodic_runner import simulate

    # A diagonal two-sphere motif has ordinary linear anisotropy. Its raw
    # transmitted/reflected ellipse can rotate even though g=0 everywhere.
    # Solving the full infinite array must retain that even background while
    # giving exactly zero magnetization-odd signal; no mocked fields are used.
    config = {
        "schema_version": "magnetooptics-periodic-1.0",
        "geometry": {"periods_nm": [80, 100], "positions_nm": [[10, 10, 0], [40, 35, 0]], "radius_nm": 6, "material": "nonmagnetic"},
        "materials": {"nonmagnetic": {"epsilon_diagonal": [-2.5, 0.3], "g": 0}},
        "wavelength_nm": [550],
        "solver": {"backend": "cached", "ewald_tolerance": 1e-11, "rtol": 1e-11},
    }
    rows, metadata, _, _ = simulate(config)
    row = rows[0]
    assert row["zero_reflection_Mplus_x"] > 1e-6
    assert abs(row["kerr_rotation_deg_Mplus"]) > 1e-3
    assert abs(row["faraday_rotation_deg_Mplus"]) > 1e-6
    for name in ("kerr", "faraday"):
        assert row[f"{name}_rotation_deg_Mplus"] == row[f"{name}_rotation_deg_Mminus"]
        assert row[f"{name}_magnetic_odd_rotation_deg"] == 0
    assert row["transmission_magnetic_odd_cd"] == 0
    assert row["reflection_magnetic_odd_cd"] == 0
    assert metadata["max_energy_balance_residual"] < 1e-8
