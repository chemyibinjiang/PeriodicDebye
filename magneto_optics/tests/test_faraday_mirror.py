"""Independent Maxwell boundary checks for a Faraday slab and ordinary mirror.

The oracle solves simultaneous E/H continuity equations for forward/backward
field amplitudes, rather than using Fresnel or scattering-matrix recursion.
The Jones basis is fixed laboratory x,y on both propagation directions.

Physical background: operator algebra for birefringence and mirror reflection,
https://www.sciencedirect.com/science/article/abs/pii/S0030401802011239 ;
normal-incidence gyrotropic slab/PEC boundary solutions are also described in
https://www.jpier.org/ac_api/download.php?id=23041602 .
"""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from magnetooptics.faraday_mirror import (
    design_transparent_rotator,
    mirror_return,
    slab_scattering,
)


def _boundary_channel(epsilon, thickness_nm, wavelength_nm, host_index=1.0,
                      mirror=None, gap_nm=0.0):
    """Solve four (free slab) or five (mirror-backed slab) unknown amplitudes."""
    index = np.sqrt(complex(epsilon))
    k0 = 2 * np.pi / wavelength_nm
    forward = np.exp(1j * k0 * index * thickness_nm)
    backward = 1 / forward
    if mirror is None:
        # Unknowns r, A, B, t; t is referenced to the slab's exit plane.
        matrix = np.array([
            [1, -1, -1, 0],
            [-host_index, -index, index, 0],
            [0, forward, backward, -1],
            [0, index * forward, -index * backward, -host_index],
        ], dtype=complex)
        reflected, a, b, transmitted = np.linalg.solve(matrix, [-1, -host_index, 0, 0])
        result = {"reflection": reflected, "transmission": transmitted}
    else:
        # Unknowns r, A, B, C, D. Gap fields C,D are referenced at slab exit.
        # D=r_m exp(2 i k_host gap) C is the ordinary mirror boundary.
        mirror_at_exit = mirror * np.exp(2j * k0 * host_index * gap_nm)
        matrix = np.array([
            [1, -1, -1, 0, 0],
            [-host_index, -index, index, 0, 0],
            [0, forward, backward, -1, -1],
            [0, index * forward, -index * backward, -host_index, host_index],
            [0, 0, 0, -mirror_at_exit, 1],
        ], dtype=complex)
        reflected, a, b, c, d = np.linalg.solve(matrix, [-1, -host_index, 0, 0, 0])
        result = {"reflection": reflected, "gap_forward": c, "gap_backward": d,
                  "mirror_absorption": (1 - abs(mirror)**2) * abs(c)**2}
    # Independent volume Joule-heating integral, without using 1-R-T or any
    # production absorption routine. Fixed moderate optical depths suffice here.
    nodes, weights = np.polynomial.legendre.leggauss(64)
    z = .5 * thickness_nm * (nodes + 1)
    field = a * np.exp(1j * k0 * index * z) + b * np.exp(-1j * k0 * index * z)
    integral = .5 * thickness_nm * np.dot(weights, np.abs(field)**2)
    result["material_absorption"] = float(k0 * complex(epsilon).imag / host_index * integral)
    return result


def _channel_to_jones(plus, minus):
    # Construct from projectors, independently of any production conversion.
    eplus = np.array([1, 1j]) / np.sqrt(2)
    eminus = np.array([1, -1j]) / np.sqrt(2)
    return plus * np.outer(eplus, eplus.conj()) + minus * np.outer(eminus, eminus.conj())


def _oracle(epsilon_diagonal, g, thickness_nm, wavelength_nm, host_index=1.0,
            magnetization=1.0, mirror=None, gap_nm=0.0):
    channels = [_boundary_channel(epsilon_diagonal - helicity * magnetization * g,
                thickness_nm, wavelength_nm, host_index, mirror, gap_nm) for helicity in (1, -1)]
    result = {"reflection": _channel_to_jones(*(entry["reflection"] for entry in channels)),
              "material_absorption": sum(entry["material_absorption"] for entry in channels) / 2,
              "material_loss_operator": _channel_to_jones(*(entry["material_absorption"] for entry in channels))}
    if mirror is None:
        result["transmission"] = _channel_to_jones(*(entry["transmission"] for entry in channels))
    else:
        result["mirror_absorption"] = sum(entry["mirror_absorption"] for entry in channels) / 2
        result["mirror_loss_operator"] = _channel_to_jones(*(entry["mirror_absorption"] for entry in channels))
    return result


def _power_x(matrix):
    return float(np.vdot(matrix[:, 0], matrix[:, 0]).real)


def _rotation(angle_deg):
    angle = np.radians(angle_deg)
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


@pytest.mark.parametrize("host_index", [1.0, 1.33])
@pytest.mark.parametrize("loss", [0.0, 0.08])
@pytest.mark.parametrize("thickness", [0.0, 230.0, 851.0])
def test_free_slab_matches_independent_maxwell_boundary_solution(host_index, loss, thickness):
    epsilon, g, wavelength = 2.7 + 1j * loss, .12 + .125j * loss, 633.0
    actual = slab_scattering(epsilon, g, thickness, wavelength, host_index=host_index)
    expected = _oracle(epsilon, g, thickness, wavelength, host_index)
    assert_allclose(actual["transmission"], expected["transmission"], rtol=2e-12, atol=2e-13)
    assert_allclose(actual["reflection"], expected["reflection"], rtol=2e-12, atol=2e-13)
    absorption = 1 - _power_x(actual["transmission"]) - _power_x(actual["reflection"])
    assert_allclose(absorption, expected["material_absorption"], rtol=2e-11, atol=5e-13)
    loss_operator = np.eye(2) - actual["transmission"].conj().T @ actual["transmission"] - actual["reflection"].conj().T @ actual["reflection"]
    assert_allclose(loss_operator, expected["material_loss_operator"], rtol=2e-11, atol=5e-13)
    if loss and thickness:
        assert absorption > 0
    else:
        assert abs(absorption) < 5e-13


@pytest.mark.parametrize("host_index", [1.0, 1.33])
@pytest.mark.parametrize("loss", [0.0, 0.08])
@pytest.mark.parametrize("gap", [0.0, 137.0])
@pytest.mark.parametrize("mirror", [-1.0, .82 * np.exp(.37j)])
def test_mirror_cavity_matches_five_unknown_boundary_solution_and_heating(host_index, loss, gap, mirror):
    epsilon, g, thickness, wavelength = 2.7 + 1j * loss, .12 + .125j * loss, 230.0, 633.0
    slab = slab_scattering(epsilon, g, thickness, wavelength, host_index=host_index)
    actual = mirror_return(slab, gap_nm=gap, mirror_amplitude=mirror)
    expected = _oracle(epsilon, g, thickness, wavelength, host_index, mirror=mirror, gap_nm=gap)
    assert_allclose(actual["returned_jones"], expected["reflection"], rtol=2e-12, atol=2e-13)
    mirror_loss = actual["mirror_loss_operator"][0, 0]
    assert abs(mirror_loss.imag) < 1e-13
    assert_allclose(mirror_loss.real, expected["mirror_absorption"], rtol=2e-11, atol=5e-13)
    inferred_material_loss = 1 - _power_x(actual["returned_jones"]) - mirror_loss.real
    assert_allclose(inferred_material_loss, expected["material_absorption"], rtol=2e-11, atol=5e-13)
    for key in ("material_loss_operator", "mirror_loss_operator"):
        assert_allclose(actual[key], expected[key], rtol=2e-11, atol=5e-13)
        assert_allclose(actual[key], actual[key].conj().T, atol=2e-13)
        assert np.linalg.eigvalsh(actual[key]).min() >= -5e-13
    assert_allclose(_power_x(actual["returned_jones"]) + expected["material_absorption"] + expected["mirror_absorption"], 1, atol=5e-13)
    if loss:
        assert inferred_material_loss > 0
    else:
        assert abs(inferred_material_loss) < 5e-13


@pytest.mark.parametrize("epsilon,g", [(-1.5 + .2j, .3 + .02j), (.1 + .2j, .35)])
def test_evanescent_or_metallic_channels_use_outgoing_passive_index(epsilon, g):
    slab = slab_scattering(epsilon, g, 120, 633, host_index=1.33)
    expected = _oracle(epsilon, g, 120, 633, 1.33)
    assert np.all(np.asarray(slab["indices"]).imag >= 0)
    assert_allclose(slab["transmission"], expected["transmission"], rtol=2e-12, atol=2e-13)
    assert_allclose(slab["reflection"], expected["reflection"], rtol=2e-12, atol=2e-13)
    backed = mirror_return(slab, gap_nm=93)
    mirror_expected = _oracle(epsilon, g, 120, 633, 1.33, mirror=-1, gap_nm=93)
    assert_allclose(backed["returned_jones"], mirror_expected["reflection"], rtol=2e-12, atol=2e-13)
    assert_allclose(1 - _power_x(backed["returned_jones"]), mirror_expected["material_absorption"], rtol=2e-11, atol=5e-13)


@pytest.mark.parametrize("angle", [0, 30, 45, -45, -60])
@pytest.mark.parametrize("host_index", [1.0, 1.33])
def test_transparent_design_rotates_once_then_twice_in_exact_maxwell_model(angle, host_index):
    design = design_transparent_rotator(angle_deg=angle, wavelength_nm=600, host_index=host_index)
    slab = slab_scattering(design["epsilon_diagonal"], design["g"], design["thickness_nm"], 600, host_index=host_index)
    expected = _oracle(design["epsilon_diagonal"], design["g"], design["thickness_nm"], 600, host_index)
    assert_allclose(slab["transmission"], expected["transmission"], rtol=2e-12, atol=2e-13)
    assert_allclose(slab["reflection"], np.zeros((2, 2)), atol=2e-13)
    rotation = _rotation(angle)
    common_phase = np.trace(rotation.T @ slab["transmission"]) / 2
    assert_allclose(abs(common_phase), 1, atol=2e-13)
    assert_allclose(slab["transmission"], common_phase * rotation, atol=2e-13)
    returned = mirror_return(slab)
    mirror_expected = _oracle(design["epsilon_diagonal"], design["g"], design["thickness_nm"], 600, host_index, mirror=-1)
    assert_allclose(returned["returned_jones"], mirror_expected["reflection"], rtol=2e-12, atol=2e-13)
    assert_allclose(returned["returned_jones"], -common_phase**2 * _rotation(2 * angle), atol=3e-13)
    assert_allclose(abs(slab["transmission"][1, 0])**2, np.sin(np.radians(angle))**2, atol=2e-13)
    assert_allclose(abs(returned["returned_jones"][1, 0])**2, np.sin(np.radians(2 * angle))**2, atol=3e-13)
    if abs(angle) == 45:
        assert abs(returned["returned_jones"][0, 0])**2 < 1e-25
        assert_allclose(abs(returned["returned_jones"][1, 0])**2, 1, atol=3e-13)


def test_exact_45_degree_example_has_two_transparent_circular_channels():
    design = design_transparent_rotator()
    assert_allclose([design["epsilon_diagonal"], design["g"], design["thickness_nm"]], [2.5, 1.5, 150])
    slab = slab_scattering(2.5, 1.5, 150, 600)
    assert_allclose(slab["indices"], [1, 2], atol=1e-14)
    assert_allclose(slab["transmission_eigenvalues"], [1j, -1], atol=2e-14)
    assert_allclose(mirror_return(slab)["returned_eigenvalues"], [1, -1], atol=3e-14)


def test_faraday_adds_while_a_reciprocal_rotator_cancels_in_fixed_lab_basis():
    slab = slab_scattering(2.5, 1.5, 150, 600)
    forward = slab["propagation"]
    bare_mirror = -np.eye(2)
    faraday_round_trip = forward @ bare_mirror @ forward
    # A reciprocal element has backward Jones matrix equal to its transpose
    # in this fixed laboratory basis; an ordinary mirror is a scalar here.
    reciprocal_round_trip = forward.T @ bare_mirror @ forward
    assert_allclose(mirror_return(slab)["returned_jones"], faraday_round_trip, atol=3e-13)
    assert_allclose(abs(faraday_round_trip[1, 0])**2, 1, atol=3e-13)
    assert abs(reciprocal_round_trip[1, 0])**2 < 1e-25
    assert_allclose(abs(reciprocal_round_trip[0, 0])**2, 1, atol=3e-13)
    assert_allclose(reciprocal_round_trip, np.eye(2) * reciprocal_round_trip[0, 0], atol=2e-13)


@pytest.mark.parametrize("gap", [0, 137])
def test_magnetization_reversal_transposes_full_slab_and_mirror_return(gap):
    plus = slab_scattering(2.7 + .08j, .12 + .01j, 230, 633, magnetization=1)
    minus = slab_scattering(2.7 + .08j, .12 + .01j, 230, 633, magnetization=-1)
    for key in ("transmission", "reflection", "propagation"):
        assert_allclose(minus[key], plus[key].T, atol=3e-13)
    returned_plus = mirror_return(plus, gap_nm=gap, mirror_amplitude=.82 * np.exp(.37j))
    returned_minus = mirror_return(minus, gap_nm=gap, mirror_amplitude=.82 * np.exp(.37j))
    assert_allclose(returned_minus["returned_jones"], returned_plus["returned_jones"].T, atol=3e-13)
    assert abs(returned_plus["returned_jones"][1, 0]) > 1e-4


@pytest.mark.parametrize("thickness", [0, 230])
def test_ordinary_mirror_and_zero_g_material_do_not_create_rotation(thickness):
    slab = slab_scattering(2.7 + .08j, 0, thickness, 633)
    returned = mirror_return(slab, gap_nm=137)
    assert_allclose(returned["returned_jones"], returned["returned_jones"][0, 0] * np.eye(2), atol=2e-13)
    if thickness == 0:
        expected = -np.exp(2j * 2 * np.pi / 633 * 137) * np.eye(2)
        assert_allclose(returned["returned_jones"], expected, atol=2e-13)


def test_internal_reflections_are_measurable_and_not_just_naive_double_pass():
    slab = slab_scattering(4 + .05j, .3 + .01j, 173, 633)
    returned = mirror_return(slab, gap_nm=79)
    expected = _oracle(4 + .05j, .3 + .01j, 173, 633, mirror=-1, gap_nm=79)
    assert_allclose(returned["returned_jones"], expected["reflection"], rtol=2e-12, atol=2e-13)
    truncated = returned["front_reflection_jones"] + returned["first_return_jones"]
    assert np.linalg.norm(returned["returned_jones"] - truncated) > 1e-2
    assert_allclose(returned["returned_jones"], returned["front_reflection_jones"] + returned["cavity_return_jones"], atol=2e-13)


@pytest.mark.parametrize("arguments", [
    {"epsilon_diagonal": 2 - .1j}, {"g": .2j},
    {"thickness_nm": -1}, {"wavelength_nm": 0},
    {"host_index": 1 + .1j}, {"host_index": 0},
    {"magnetization": 1.1}, {"magnetization": True},
    {"epsilon_diagonal": True}, {"g": float("nan")},
    {"epsilon_diagonal": 1, "g": 1},
])
def test_invalid_or_active_slab_is_rejected(arguments):
    parameters = {"epsilon_diagonal": 2.7 + .08j, "g": .12 + .01j, "thickness_nm": 230, "wavelength_nm": 633}
    parameters.update(arguments)
    with pytest.raises(ValueError):
        slab_scattering(**parameters)


@pytest.mark.parametrize("arguments", [
    {"gap_nm": -1}, {"gap_nm": True},
    {"mirror_amplitude": 1.01}, {"mirror_amplitude": float("nan")},
])
def test_invalid_or_active_mirror_is_rejected(arguments):
    with pytest.raises(ValueError):
        mirror_return(slab_scattering(2.5, 1.5, 150, 600), **arguments)


@pytest.mark.parametrize("angle", [90, -90, 91, float("nan"), True])
def test_invalid_transparent_design_angle_is_rejected(angle):
    with pytest.raises(ValueError):
        design_transparent_rotator(angle_deg=angle)
