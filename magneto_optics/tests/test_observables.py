import numpy as np
import pytest

from magnetooptics.materials import sphere_polarizability
from magnetooptics.observables import (
    circular_polarizations,
    cross_sections,
    far_field,
    integrated_scattering,
    plane_wave,
    stokes_parameters,
)


@pytest.mark.parametrize("direction", [[0, 0, 1], [1, 0, 0], [1, -2, 3]])
def test_circular_channels_are_transverse_orthonormal_and_have_declared_sign(direction):
    direction = np.asarray(direction) / np.linalg.norm(direction)
    plus, minus = circular_polarizations(direction)
    np.testing.assert_allclose(np.vdot(plus, plus), 1, atol=3e-16)
    np.testing.assert_allclose(np.vdot(plus, minus), 0, atol=3e-16)
    np.testing.assert_allclose(direction @ plus, 0, atol=3e-16)
    np.testing.assert_allclose(np.cross(direction, plus), -1j * plus, atol=3e-16)


def test_default_circular_basis_and_parallel_reference_rejection():
    plus, minus = circular_polarizations([0, 0, 1])
    np.testing.assert_array_equal(plus, np.array([1, 1j, 0]) / np.sqrt(2))
    np.testing.assert_array_equal(minus, plus.conj())
    with pytest.raises(ValueError, match="parallel"):
        circular_polarizations([0, 0, 1], [0, 0, -1])


def test_plane_wave_host_wavelength_phase_and_normalization():
    field = plane_wave([[0, 0, 0], [0, 0, 100]], 600, [0, 0, 2], [3, 0, 0], host_index=1.5)
    np.testing.assert_allclose(field, [[1, 0, 0], [1j, 0, 0]], atol=3e-16)


@pytest.mark.parametrize("polarization", [[0, 0, 1], [1, 0, 0.1], [0, 0, 0], [np.nan, 0, 0]])
def test_invalid_plane_wave_polarization(polarization):
    with pytest.raises(ValueError):
        plane_wave([[0, 0, 0]], 600, [0, 0, 1], polarization)


def test_single_arbitrary_complex_dipole_integrated_radiation():
    dipole = np.array([[1 + 2j, -3j, 4 - 1j]])
    k = 0.021
    expected = k**4 * np.sum(np.abs(dipole)**2) / (6 * np.pi)
    actual = integrated_scattering(dipole, [[1, 3, -2]], k, n_theta=4, n_phi=8)
    assert actual == pytest.approx(expected, rel=2e-15)


def test_far_field_is_transverse_and_translation_changes_only_phase():
    rng = np.random.default_rng(23)
    positions = rng.normal(size=(7, 3)) * 25
    dipoles = rng.normal(size=(7, 3)) + 1j * rng.normal(size=(7, 3))
    directions = rng.normal(size=(13, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    shift = np.array([17, -11, 31])
    k = 0.02
    initial = far_field(dipoles, positions, k, directions)
    translated = far_field(dipoles, positions + shift, k, directions)
    np.testing.assert_allclose(np.sum(initial * directions, axis=1), 0, atol=1e-19)
    np.testing.assert_allclose(translated, initial * np.exp(-1j * k * (directions @ shift))[:, None], rtol=1e-14, atol=1e-19)


def test_phase_memory_chunks_preserve_results():
    rng = np.random.default_rng(14)
    positions = rng.normal(size=(37, 3)) * 25
    dipoles = rng.normal(size=(37, 3)) + 1j * rng.normal(size=(37, 3))
    directions = rng.normal(size=(45, 3))
    reference = far_field(dipoles, positions, 0.01, directions)
    tiny_chunks = far_field(dipoles, positions, 0.01, directions, chunk_size=3, max_pair_elements=11)
    np.testing.assert_allclose(tiny_chunks, reference, rtol=1e-13, atol=1e-19)


@pytest.mark.parametrize("epsilon", [2.25, 2.25 + 0.2j])
def test_single_sphere_cross_sections_match_analytic_formula(epsilon):
    k = 2 * np.pi / 600
    alpha = sphere_polarizability(epsilon, 12, 600)[0, 0]
    incident = np.array([[1, 0, 0]], dtype=complex)
    values = cross_sections(alpha * incident, incident, incident, k)
    expected_scattering = k**4 * abs(alpha)**2 / (6 * np.pi)
    assert values["extinction_nm2"] == pytest.approx(k * alpha.imag, rel=2e-15)
    assert values["scattering_nm2"] == pytest.approx(expected_scattering, rel=2e-13)
    assert values["absorption_nm2"] == pytest.approx(k * alpha.imag - expected_scattering, abs=1e-14)


def test_negative_absorption_is_exposed_not_clipped():
    incident = np.array([[1, 0, 0]], dtype=complex)
    values = cross_sections(incident, incident, incident, 0.1)
    assert values["absorption_nm2"] < 0


def test_stokes_circular_sign_and_purity():
    states = np.array([[1, 0], [1, 1j], [1, -1j], [1 + 2j, -2 + 1j]], dtype=complex)
    stokes = stokes_parameters(states)
    assert stokes["S3"][1] > 0
    assert stokes["S3"][2] < 0
    np.testing.assert_allclose(stokes["S0"]**2, stokes["S1"]**2 + stokes["S2"]**2 + stokes["S3"]**2)


def test_empty_particle_system_radiates_nothing():
    empty = np.empty((0, 3), dtype=complex)
    np.testing.assert_array_equal(far_field(empty, empty.real, 0.01, [[0, 0, 1]]), np.zeros((1, 3)))
    assert cross_sections(empty, empty, empty, 0.01) == {"extinction_nm2": 0, "absorption_nm2": 0, "scattering_nm2": 0}


@pytest.mark.parametrize("orders", [(1, 4), (4, 1), (3.5, 4), (4, 3.5)])
def test_invalid_angular_quadrature_orders(orders):
    with pytest.raises(ValueError):
        integrated_scattering([[1, 0, 0]], [[0, 0, 0]], 0.01, n_theta=orders[0], n_phi=orders[1])


def test_circular_scattered_field_has_no_defined_ellipse_orientation():
    from magnetooptics.runner import simulate

    # epsilon_d=2,g=1 has one host-matched circular eigenchannel epsilon=1
    # and one scattering channel epsilon=3. Linear illumination consequently
    # scatters exactly circularly: its major-axis angle is physically undefined.
    config = {
        "schema_version": "magnetooptics-1.0",
        "geometry": {"kind": "explicit", "positions_nm": [[0, 0, 0]], "radius_nm": 4, "material": "circular_filter"},
        "materials": {"circular_filter": {"epsilon_diagonal": 2, "g": 1}},
        "wavelength_nm": [600], "solver": {"backend": "dense", "rtol": 1e-12},
    }
    rows, _, _ = simulate(config)
    row = rows[0]
    for magnetization, ellipticity in (("Mplus", -45), ("Mminus", 45)):
        assert row[f"forward_scattered_rotation_deg_{magnetization}"] is None
        assert row[f"forward_scattered_ellipticity_deg_{magnetization}"] == pytest.approx(ellipticity, abs=2e-6)
        assert row[f"forward_scattered_cross_fraction_{magnetization}"] == pytest.approx(0.5, abs=3e-15)


def test_onsager_source_exchange_reverses_all_noncollinear_magnetizations():
    from magnetooptics.interaction import DenseGreenOperator
    from magnetooptics.materials import gyrotropic_permittivity
    from magnetooptics.solver import CoupledDipoleSystem

    # The response to arbitrary local Cartesian excitations obeys the Onsager
    # source/detector exchange identity T(-M)=T(+M).T, with NO conjugation.
    # Heterogeneous, lossy, noncollinear gyrotropic sites prevent accidental
    # scalar, geometric, Hermitian or ordinary reciprocal symmetries.
    positions = np.array([[0, 0, 0], [45, 17, -3], [-13, 42, 21]], dtype=float)
    radii = np.array([5, 6, 7], dtype=float)
    epsilon_diagonal = [2.8 + 0.2j, 3.1 + 0.3j, 2.2 + 0.1j]
    gyrotropy = [0.3 + 0.02j, 0.2 + 0.01j, 0.4 + 0.03j]
    magnetizations = np.array([[0.2, 0.3, 0.8], [-0.5, 0.1, 0.4], [0.4, -0.6, 0.1]])
    wavelength, host_index = 530.0, 1.25
    operator = DenseGreenOperator(positions, 2 * np.pi * host_index / wavelength)
    cartesian_sources = np.eye(9, dtype=complex)
    responses = []
    for sign in (1, -1):
        tensors = np.array([
            gyrotropic_permittivity(epsilon, g, sign * magnetization)
            for epsilon, g, magnetization in zip(epsilon_diagonal, gyrotropy, magnetizations)
        ])
        alpha = sphere_polarizability(tensors, radii, wavelength, host_index)
        system = CoupledDipoleSystem(operator, alpha, method="dense")
        responses.append(np.column_stack([
            system.solve(cartesian_sources[:, column].reshape(3, 3), rtol=1e-12).dipoles.ravel()
            for column in range(9)
        ]))
    positive, negative = responses
    np.testing.assert_allclose(negative, positive.T, rtol=2e-12, atol=1e-11)
    assert np.linalg.norm(positive - positive.T) > 1e-3 * np.linalg.norm(positive)
    assert np.linalg.norm(negative - positive.conj().T) > 1e-2 * np.linalg.norm(positive)
