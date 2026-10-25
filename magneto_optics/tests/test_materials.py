import numpy as np
import pytest

from magnetooptics.materials import (
    gyrotropic_permittivity,
    sphere_polarizability,
    validate_permittivity,
)


def test_gyrotropic_eigenchannels_and_lossless_hermitian_tensor():
    epsilon = gyrotropic_permittivity(3.2, 0.4)
    plus = np.array([1, 1j, 0]) / np.sqrt(2)
    minus = plus.conj()
    np.testing.assert_allclose(epsilon @ plus, 2.8 * plus)
    np.testing.assert_allclose(epsilon @ minus, 3.6 * minus)
    np.testing.assert_array_equal(epsilon, epsilon.conj().T)


def test_partial_and_zero_magnetization_are_not_normalized():
    zero = gyrotropic_permittivity(2 + 0.1j, 0.8, [0, 0, 0])
    half = gyrotropic_permittivity(2 + 0.1j, 0.8, [0, 0, 0.5])
    np.testing.assert_array_equal(zero, (2 + 0.1j) * np.eye(3))
    assert half[0, 1] == 0.4j
    np.testing.assert_array_equal(half.T, gyrotropic_permittivity(2 + 0.1j, 0.8, [0, 0, -0.5]))


def test_passivity_uses_hermitian_loss_not_entrywise_imaginary_part():
    passive = gyrotropic_permittivity(2 + 0.3j, 0.8 + 0.2j)
    assert passive[1, 0].imag < 0
    validate_permittivity(passive)
    with pytest.raises(ValueError, match="active"):
        gyrotropic_permittivity(2 + 0.1j, 0.8 + 0.2j)


@pytest.mark.parametrize("epsilon", [1 - 1e-3j, np.diag([2 + 0.1j, 2 - 0.001j, 2]), np.nan, np.inf])
def test_invalid_or_active_permittivity(epsilon):
    with pytest.raises(ValueError):
        validate_permittivity(epsilon)


@pytest.mark.parametrize("magnetization", [[0, 0, 1.01], [0, 0], [0, 0, np.nan], [0, 0, 1j]])
def test_invalid_magnetization(magnetization):
    with pytest.raises(ValueError):
        gyrotropic_permittivity(2, 0.1, magnetization)


def test_zero_contrast_and_electrostatic_resonance_are_finite():
    n = 1.33
    np.testing.assert_array_equal(sphere_polarizability(n**2, 10, 600, n), np.zeros((3, 3)))
    k = 2 * np.pi * n / 600
    resonant = sphere_polarizability(-2 * n**2, 10, 600, n)
    np.testing.assert_allclose(resonant, (6j * np.pi / k**3) * np.eye(3), rtol=2e-15)


def test_tensor_and_radius_broadcast_preserve_site_values():
    epsilon = np.stack([np.eye(3) * (2 + 0.1j), np.eye(3) * (3 + 0.2j)])
    radii = np.array([5, 8])
    batch = sphere_polarizability(epsilon, radii, 620)
    assert batch.shape == (2, 3, 3)
    for i in range(2):
        np.testing.assert_array_equal(batch[i], sphere_polarizability(epsilon[i], radii[i], 620))
    assert sphere_polarizability(2.2, radii, 620).shape == (2, 3, 3)


def test_rotating_general_tensor_rotates_polarizability():
    angle = 0.37
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    epsilon = np.diag([2 + 0.2j, 3 + 0.3j, 4 + 0.4j])
    rotated = rotation @ epsilon @ rotation.T
    expected = rotation @ sphere_polarizability(epsilon, 9, 550) @ rotation.T
    np.testing.assert_allclose(sphere_polarizability(rotated, 9, 550), expected, rtol=2e-15, atol=1e-12)


def test_radiative_polarizability_satisfies_passivity_quadratic_form():
    epsilon = gyrotropic_permittivity(3 + 0.5j, 0.1 + 0.2j, [0.3, -0.4, 0.5])
    alpha = sphere_polarizability(epsilon, 8, 600, 1.4)
    k = 2 * np.pi * 1.4 / 600
    absorbed = (alpha - alpha.conj().T) / (2j) - (k**3 / (6 * np.pi)) * alpha.conj().T @ alpha
    assert np.linalg.eigvalsh(absorbed).min() > 0


@pytest.mark.parametrize("radius", [0, -1, np.nan, np.inf, 1j])
def test_invalid_radii(radius):
    with pytest.raises(ValueError):
        sphere_polarizability(2, radius, 500)


@pytest.mark.parametrize("host", [0, -1, np.nan, np.inf, 1 + 0j])
def test_invalid_hosts(host):
    with pytest.raises(ValueError):
        sphere_polarizability(2, 10, 500, host)


@pytest.mark.parametrize("wavelength", [0, -1, np.nan, np.inf, 500 + 0j])
def test_invalid_wavelengths(wavelength):
    with pytest.raises(ValueError):
        sphere_polarizability(2, 10, wavelength)
