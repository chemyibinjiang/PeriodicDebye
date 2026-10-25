"""Independent physical acceptance oracles for the finite electric-dipole model.

These tests do not establish continuum Maxwell accuracy for arbitrary particles.
They distinguish algebraic/FFT accuracy from the measurable Rayleigh sphere error.

Conventions: exp(-i omega t), outgoing exp(+ikr), alpha in nm^3, real host index.
References for the underlying equations (not external software dependencies):
* Draine & Flatau, JOSA A 11, 1491 (1994), doi:10.1364/JOSAA.11.001491.
* Bohren & Huffman, Absorption and Scattering of Light by Small Particles,
  equations 4.62 and 4.88; coefficient/cross-section formulas also documented at
  https://miepython.readthedocs.io/en/3.0.0/07_algorithm.html .
The Mie oracle below uses SciPy Bessel functions and outgoing h^(1), with positive
imaginary permittivity for absorption; it does not call the production solver or
copy a production Green tensor into its independent reference calculations.
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.special import spherical_jn, spherical_yn

from magnetooptics.interaction import DenseGreenOperator, FFTGreenOperator
from magnetooptics.materials import gyrotropic_permittivity, sphere_polarizability
from magnetooptics.observables import (
    circular_polarizations,
    cross_sections,
    integrated_scattering,
    plane_wave,
)
from magnetooptics.solver import CoupledDipoleSystem


def _radiation_corrected_scalar(epsilon, radius, wavelength, host_index=1.0):
    """Closed sphere eigenchannel, written independently of tensor inversion."""
    k = 2 * np.pi * host_index / wavelength
    host_epsilon = host_index**2
    electrostatic = 4 * np.pi * radius**3 * (
        (epsilon - host_epsilon) / (epsilon + 2 * host_epsilon)
    )
    return electrostatic / (1 - 1j * k**3 * electrostatic / (6 * np.pi))


def _direct_dipole_field(positions, dipoles, k):
    """Retarded field using cross products, independent of matrix assembly."""
    result = np.zeros_like(dipoles, dtype=complex)
    for i, target in enumerate(positions):
        for j, source in enumerate(positions):
            if i == j:
                continue
            displacement = target - source
            distance = np.linalg.norm(displacement)
            normal = displacement / distance
            transverse = np.cross(np.cross(normal, dipoles[j]), normal)
            near_field = 3 * normal * np.dot(normal, dipoles[j]) - dipoles[j]
            result[i] += np.exp(1j * k * distance) / (4 * np.pi) * (
                k**2 * transverse / distance
                + (1 / distance**3 - 1j * k / distance**2) * near_field
            )
    return result


def _sphere_integrated_radiation(positions, dipoles, k):
    """Integrate the transverse radiation explicitly with an independent grid."""
    mu, mu_weights = np.polynomial.legendre.leggauss(21)
    phi = 2 * np.pi * np.arange(47) / 47
    result = 0.0
    for cosine, weight in zip(mu, mu_weights):
        sine = np.sqrt(1 - cosine**2)
        directions = np.column_stack(
            (sine * np.cos(phi), sine * np.sin(phi), np.full(phi.size, cosine))
        )
        for direction in directions:
            phase = np.exp(-1j * k * (positions @ direction))
            moment = np.sum(phase[:, None] * dipoles, axis=0)
            transverse = moment - direction * np.dot(direction, moment)
            result += weight * 2 * np.pi / phi.size * np.vdot(
                transverse, transverse
            ).real
    return result * k**4 / (16 * np.pi**2)


def _exact_mie_cross_sections(epsilon, radius, wavelength, host_index=1.0):
    """Full electric + magnetic multipole series for the small scalar spheres.

    Six orders are ample for the tested ka <= 0.08; convergence is also checked
    against eight orders below.  This helper is deliberately local to the tests.
    """
    k = 2 * np.pi * host_index / wavelength
    x = k * radius
    m = np.sqrt(complex(epsilon)) / host_index
    values = []
    for nmax in (6, 8):
        orders = np.arange(1, nmax + 1)
        mx = m * x
        psi_x = x * spherical_jn(orders, x)
        dpsi_x = spherical_jn(orders, x) + x * spherical_jn(orders, x, True)
        psi_m = mx * spherical_jn(orders, mx)
        dpsi_m = spherical_jn(orders, mx) + mx * spherical_jn(orders, mx, True)
        xi_x = psi_x + 1j * x * spherical_yn(orders, x)
        dxi_x = dpsi_x + 1j * (
            spherical_yn(orders, x) + x * spherical_yn(orders, x, True)
        )
        electric = (m * psi_m * dpsi_x - psi_x * dpsi_m) / (
            m * psi_m * dxi_x - xi_x * dpsi_m
        )
        magnetic = (psi_m * dpsi_x - m * psi_x * dpsi_m) / (
            psi_m * dxi_x - m * xi_x * dpsi_m
        )
        weights = 2 * orders + 1
        extinction = 2 * np.pi / k**2 * np.sum(
            weights * np.real(electric + magnetic)
        )
        scattering = 2 * np.pi / k**2 * np.sum(
            weights * (np.abs(electric)**2 + np.abs(magnetic)**2)
        )
        values.append(np.array([extinction, scattering]))
    assert_allclose(values[0], values[1], rtol=1e-13, atol=0)
    return values[1]


@pytest.mark.parametrize("backend", ["dense", "fft"])
@pytest.mark.parametrize("component", [0, 1, 2])
@pytest.mark.parametrize("parity", [1, -1])
def test_two_dipole_closed_longitudinal_and_transverse_modes(backend, component, parity):
    """Bright/dark dimer modes have p=alpha E/(1-parity alpha G_mode)."""
    separation, k = 28.0, 0.05
    positions = np.array([[0, 0, 0], [separation, 0, 0]])
    alpha = 20_000 + 400j
    kr = k * separation
    if component == 0:
        eigenvalue = 2 * (1 - 1j * kr)
    else:
        eigenvalue = kr**2 - 1 + 1j * kr
    eigenvalue *= np.exp(1j * kr) / (4 * np.pi * separation**3)
    incident = np.zeros((2, 3), dtype=complex)
    incident[:, component] = [1, parity]
    expected = alpha * incident / (1 - parity * alpha * eigenvalue)
    if backend == "dense":
        operator = DenseGreenOperator(positions, k)
        method = "dense"
    else:
        operator = FFTGreenOperator([[0, 0, 0], [1, 0, 0]], separation, k)
        method = "gmres"
    solution = CoupledDipoleSystem(operator, alpha * np.eye(3), method).solve(
        incident, rtol=1e-12
    )
    assert_allclose(solution.dipoles, expected, rtol=2e-12, atol=2e-9)
    assert_allclose(solution.local_field, expected / alpha, rtol=2e-12, atol=1e-13)
    assert solution.relative_residual < 1e-12


@pytest.mark.parametrize("host_index", [1.0, 1.33])
@pytest.mark.parametrize("helicity", [1, -1])
def test_single_sphere_circular_eigenpolarizability(host_index, helicity):
    """For m=+z, (x +/- i y) sees epsilon_diagonal -/+ g."""
    epsilon, g, radius, wavelength = 3 + 0.2j, 0.25 + 0.02j, 7.0, 650.0
    k = 2 * np.pi * host_index / wavelength
    polarization = np.array([1, helicity * 1j, 0]) / np.sqrt(2)
    expected = _radiation_corrected_scalar(
        epsilon - helicity * g, radius, wavelength, host_index
    )
    tensor = gyrotropic_permittivity(epsilon, g, (0, 0, 1))
    alpha = sphere_polarizability(tensor, radius, wavelength, host_index=host_index)
    assert_allclose(alpha @ polarization, expected * polarization, rtol=2e-13, atol=1e-12)
    incident = polarization[None, :]
    solution = CoupledDipoleSystem(DenseGreenOperator([[0, 0, 0]], k), alpha, "dense").solve(incident)
    measured = cross_sections(solution.dipoles, incident, solution.local_field, k)
    assert_allclose(measured["extinction_nm2"], k * expected.imag, rtol=3e-13)
    assert_allclose(measured["scattering_nm2"], k**4 * abs(expected)**2 / (6 * np.pi), rtol=1e-11)


@pytest.mark.parametrize("g", [0.0, 0.30 + 0.03j])
def test_square_array_circular_degeneracy_and_magnetization_reversal(g):
    """A mirror-symmetric square cannot retain CD when its magnetization flips."""
    positions = np.array([[-30, -30, 0], [-30, 30, 0], [30, -30, 0], [30, 30, 0]])
    wavelength, k = 620.0, 2 * np.pi / 620.0
    operator = DenseGreenOperator(positions, k)
    eplus, eminus = circular_polarizations((0, 0, 1))
    assert_allclose(eplus, np.array([1, 1j, 0]) / np.sqrt(2), atol=1e-15)
    records = []
    for sign in (1, -1):
        epsilon = gyrotropic_permittivity(2.8 + 0.15j, g, (0, 0, sign))
        alpha = sphere_polarizability(epsilon, 8.0, wavelength)
        system = CoupledDipoleSystem(operator, alpha, "dense")
        row = []
        for polarization in (eplus, eminus):
            incident = plane_wave(positions, wavelength, (0, 0, 1), polarization)
            solution = system.solve(incident)
            row.append(cross_sections(solution.dipoles, incident, solution.local_field, k)["extinction_nm2"])
        records.append(row)
    values = np.asarray(records)
    assert_allclose(values[0], values[1, ::-1], rtol=2e-12, atol=1e-14)
    if g == 0:
        assert_allclose(values[:, 0], values[:, 1], rtol=2e-12, atol=1e-14)
    else:
        assert abs(values[0, 0] - values[0, 1]) > 1e-3 * values.max()
        assert_allclose(np.diff(values, axis=1)[0], -np.diff(values, axis=1)[1], rtol=2e-12)


@pytest.mark.parametrize("absorbing", [False, True])
def test_optical_theorem_against_independent_sphere_integration(absorbing):
    """Angular radiation, work on dipoles, and material heating must agree."""
    positions = np.array([[0, 0, 0], [41, 7, 12], [-29, 39, 3], [9, -46, 17], [14, 13, 53], [-20, -16, -37]])
    host_index, wavelength = 1.2, 500.0
    k = 2 * np.pi * host_index / wavelength
    epsilon = 3.0 + (0.2j if absorbing else 0)
    tensor = gyrotropic_permittivity(epsilon, 0.3, (0, 0, 1))
    alpha = sphere_polarizability(tensor, 8.0, wavelength, host_index=host_index)
    direction = np.array([1, 2, 3]) / np.sqrt(14)
    polarization = circular_polarizations(direction)[0]
    incident = plane_wave(positions, wavelength, direction, polarization, host_index=host_index)
    solution = CoupledDipoleSystem(DenseGreenOperator(positions, k), alpha, "dense").solve(incident, rtol=1e-12)
    measured = cross_sections(solution.dipoles, incident, solution.local_field, k)
    independent_radiation = _sphere_integrated_radiation(positions, solution.dipoles, k)
    assert_allclose(measured["scattering_nm2"], independent_radiation, rtol=5e-11, atol=1e-14)
    assert_allclose(integrated_scattering(solution.dipoles, positions, k), independent_radiation, rtol=2e-12)
    assert measured["extinction_nm2"] > 0
    if absorbing:
        assert measured["absorption_nm2"] > 0
        assert measured["extinction_nm2"] > independent_radiation
    else:
        assert abs(measured["absorption_nm2"]) < 3e-11 * measured["extinction_nm2"]
        assert_allclose(measured["extinction_nm2"], independent_radiation, rtol=5e-11)


def test_sparse_fft_field_matches_independent_retarded_dipole_field():
    """Signed 3-D indices and unequal spacings exercise every tensor component."""
    indices = np.array([[-2, 1, 0], [1, -1, 2], [0, 0, -1], [2, 2, 1], [-1, 0, 2], [1, 1, -1]])
    spacing = np.array([31.0, 43.0, 27.0])
    positions = indices * spacing
    k = 2 * np.pi / 470.0
    rng = np.random.default_rng(20261025)
    moments = rng.normal(size=(len(indices), 3)) + 1j * rng.normal(size=(len(indices), 3))
    expected = _direct_dipole_field(positions, moments, k)
    fft = FFTGreenOperator(indices, spacing, k, shape=(7, 5, 6))
    dense = DenseGreenOperator(positions, k)
    assert_allclose(fft.apply(moments), expected, rtol=4e-13, atol=1e-20)
    assert_allclose(dense.apply(moments), expected, rtol=4e-13, atol=1e-20)
    direction = np.array([1, 2, 3]) / np.sqrt(14)
    epsilon = gyrotropic_permittivity(3 + 0.1j, 0.2 + 0.01j, direction)
    alpha = sphere_polarizability(epsilon, 9.0, 470.0)
    incident = plane_wave(positions, 470.0, direction, circular_polarizations(direction)[0])
    reference = CoupledDipoleSystem(dense, alpha, "dense").solve(incident, rtol=1e-12)
    accelerated = CoupledDipoleSystem(fft, alpha, "gmres").solve(incident, rtol=1e-12)
    assert_allclose(accelerated.dipoles, reference.dipoles, rtol=2e-11, atol=2e-9)
    independent_local = incident + _direct_dipole_field(positions, accelerated.dipoles, k)
    assert_allclose(accelerated.local_field, independent_local, rtol=2e-13, atol=1e-13)
    assert accelerated.relative_residual < 1e-12


def test_small_sphere_exact_mie_quantifies_dipole_physical_error(record_property):
    """Tiny-sphere agreement is checked; O((ka)^2) model error is not hidden."""
    wavelength, epsilon = 600.0, 2.25 + 0.1j
    k = 2 * np.pi / wavelength
    errors = []
    for size_parameter in (0.02, 0.04, 0.08):
        radius = size_parameter / k
        alpha = sphere_polarizability(epsilon * np.eye(3), radius, wavelength)
        incident = np.array([[1.0, 0, 0]], dtype=complex)
        solution = CoupledDipoleSystem(DenseGreenOperator([[0, 0, 0]], k), alpha, "dense").solve(incident)
        sections = cross_sections(solution.dipoles, incident, solution.local_field, k)
        dipole = np.array([sections["extinction_nm2"], sections["scattering_nm2"]])
        mie = _exact_mie_cross_sections(epsilon, radius, wavelength)
        assert np.all(mie > 0)
        assert mie[0] > mie[1]
        relative_error = np.abs(dipole - mie) / mie
        errors.append(relative_error)
        record_property(f"ka_{size_parameter:.2f}_extinction_relative_error", float(relative_error[0]))
        record_property(f"ka_{size_parameter:.2f}_scattering_relative_error", float(relative_error[1]))
    errors = np.asarray(errors)
    assert np.all(errors[0] < 0.001)
    # Finite-size corrections are detectable and scale quadratically here.
    assert np.all(errors[0] > 1e-6)
    assert np.all((errors[1] / errors[0] > 3.8) & (errors[1] / errors[0] < 4.2))
    assert errors[2, 0] > 0.001
