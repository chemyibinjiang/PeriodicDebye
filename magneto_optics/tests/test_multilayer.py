"""Independent E/H boundary and volume-heating oracles for layered Faraday media.

Every modest stack is solved as simultaneous boundary equations for all internal
fields, independently of the production scattering recursion. Exact zero-index
layers use E=A+Bz and H=B/(i*k0), rather than two degenerate plane waves.
All Jones vectors use fixed laboratory x,y for forward and reverse incidence.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

from magnetooptics.faraday_mirror import slab_scattering, mirror_return
from magnetooptics.multilayer import layer_stack_scattering, stack_mirror_return

_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(64)


def _jones(plus, minus):
    eplus, eminus = np.array([1, 1j]) / np.sqrt(2), np.array([1, -1j]) / np.sqrt(2)
    return plus * np.outer(eplus, eplus.conj()) + minus * np.outer(eminus, eminus.conj())


def _boundary_channel(epsilons, thicknesses, wavelength, host, mirror=None, gap=0):
    """Independent full boundary linear solve, not a product of transfer blocks."""
    k0 = 2 * np.pi / wavelength
    count = len(epsilons)
    load = None if mirror is None else mirror * np.exp(2j * k0 * host * gap)
    if count == 0:
        return {"reflection": 0 if mirror is None else load, "transmission": 1,
                "material_loss": 0, "mirror_loss": 0 if mirror is None else 1 - abs(mirror)**2}
    indices = np.sqrt(np.asarray(epsilons, dtype=complex))
    # Each layer stores E and admittance-scaled H coefficients at both ends.
    basis = []
    for epsilon, index, thickness in zip(epsilons, indices, thicknesses):
        if epsilon == 0:
            basis.append((np.array([1, 0]), np.array([0, 1 / (1j * k0)]),
                          np.array([1, thickness]), np.array([0, 1 / (1j * k0)])))
        else:
            phase = np.exp(1j * k0 * index * thickness)
            basis.append((np.array([1, 1]), np.array([index, -index]),
                          np.array([phase, 1 / phase]), np.array([index * phase, -index / phase])))
    size = 2 * count + (2 if mirror is None else 3)
    matrix, rhs = np.zeros((size, size), dtype=complex), np.zeros(size, dtype=complex)
    matrix[0, 0], matrix[0, 1:3], rhs[0] = 1, -basis[0][0], -1
    matrix[1, 0], matrix[1, 1:3], rhs[1] = -host, -basis[0][1], -host
    row = 2
    for i in range(count - 1):
        left, right = slice(1 + 2 * i, 3 + 2 * i), slice(3 + 2 * i, 5 + 2 * i)
        matrix[row, left], matrix[row, right] = basis[i][2], -basis[i + 1][0]
        matrix[row + 1, left], matrix[row + 1, right] = basis[i][3], -basis[i + 1][1]
        row += 2
    last = slice(2 * count - 1, 2 * count + 1)
    matrix[row, last], matrix[row + 1, last] = basis[-1][2], basis[-1][3]
    right = 2 * count + 1
    matrix[row, right], matrix[row + 1, right] = -1, -host
    if mirror is not None:
        matrix[row, right + 1], matrix[row + 1, right + 1] = -1, host
        matrix[row + 2, right], matrix[row + 2, right + 1] = -load, 1
    solution = np.linalg.solve(matrix, rhs)
    absorption = 0.0
    for i, (epsilon, index, thickness) in enumerate(zip(epsilons, indices, thicknesses)):
        a, b = solution[1 + 2 * i:3 + 2 * i]
        z = .5 * thickness * (_NODES + 1)
        field = a + b * z if epsilon == 0 else a * np.exp(1j * k0 * index * z) + b * np.exp(-1j * k0 * index * z)
        absorption += k0 / host * complex(epsilon).imag * .5 * thickness * np.dot(_WEIGHTS, abs(field)**2)
    return {"reflection": solution[0], "transmission": solution[right], "material_loss": float(absorption),
            "mirror_loss": 0.0 if mirror is None else float((1 - abs(mirror)**2) * abs(solution[right])**2)}


def _oracle(layers, wavelength=633, host=1.0, magnetization=1.0, mirror=None, gap=0):
    channels = []
    for helicity in (1, -1):
        epsilons = [layer["epsilon_diagonal"] - helicity * magnetization * layer.get("g", 0) for layer in layers]
        thicknesses = [layer["thickness_nm"] for layer in layers]
        channels.append(_boundary_channel(epsilons, thicknesses, wavelength, host, mirror, gap))
    return {name: _jones(*(channel[name] for channel in channels)) for name in ("reflection", "transmission", "material_loss", "mirror_loss")}


STACKS = [
    [{"epsilon_diagonal": 2.3 + .05j, "g": .09 + .01j, "thickness_nm": 43}],
    [{"epsilon_diagonal": 1.8, "g": .05, "thickness_nm": 63},
     {"epsilon_diagonal": 4.2, "g": .1, "thickness_nm": 127}],
    [{"epsilon_diagonal": 2.4 + .07j, "g": .08 + .01j, "thickness_nm": 55},
     {"epsilon_diagonal": 5 + .2j, "g": -.14 + .01j, "thickness_nm": 97},
     {"epsilon_diagonal": 1.7 + .03j, "thickness_nm": 83}],
    [{"epsilon_diagonal": -1.5 + .2j, "g": .13 + .02j, "thickness_nm": 35},
     {"epsilon_diagonal": 2.7 + .04j, "g": .05, "thickness_nm": 80}],
    [{"epsilon_diagonal": 1, "g": 1, "thickness_nm": 75},
     {"epsilon_diagonal": 2.4, "g": .03, "thickness_nm": 51}],
]


@pytest.mark.parametrize("layers", STACKS)
@pytest.mark.parametrize("host", [1.0, 1.33])
@pytest.mark.parametrize("magnetization", [1.0, -1.0])
def test_both_incidence_sides_match_boundary_equations_and_volume_heating(layers, host, magnetization):
    original = copy.deepcopy(layers)
    actual = layer_stack_scattering(layers, 633, host_index=host, magnetization=magnetization)
    assert layers == original
    for prefix, ordered in [("", layers), ("reverse_", layers[::-1])]:
        expected = _oracle(ordered, host=host, magnetization=magnetization)
        t, r = actual[prefix + "transmission"], actual[prefix + "reflection"]
        assert_allclose(t, expected["transmission"], rtol=3e-11, atol=3e-13)
        assert_allclose(r, expected["reflection"], rtol=3e-11, atol=3e-13)
        missing_flux = np.eye(2) - t.conj().T @ t - r.conj().T @ r
        assert_allclose(missing_flux, expected["material_loss"], rtol=3e-10, atol=8e-13)
        name = "right_material_loss_operator" if prefix else "left_material_loss_operator"
        assert_allclose(actual[name], expected["material_loss"], rtol=3e-10, atol=8e-13)
        if not prefix:
            assert_allclose(actual["material_loss_operator"], expected["material_loss"], rtol=3e-10, atol=8e-13)
        assert np.linalg.eigvalsh(missing_flux).min() >= -1e-12


@pytest.mark.parametrize("layers", [STACKS[1], STACKS[2], STACKS[4]])
@pytest.mark.parametrize("gap", [0, 137])
@pytest.mark.parametrize("mirror", [-1, .82 * np.exp(.37j)])
@pytest.mark.parametrize("host", [1.0, 1.33])
def test_asymmetric_mirror_stack_matches_direct_boundary_and_independent_loss(layers, gap, mirror, host):
    stack = layer_stack_scattering(layers, 633, host_index=host)
    actual = stack_mirror_return(stack, gap_nm=gap, mirror_amplitude=mirror)
    expected = _oracle(layers, host=host, mirror=mirror, gap=gap)
    assert_allclose(actual["returned_jones"], expected["reflection"], rtol=3e-11, atol=3e-13)
    for name in ("material", "mirror"):
        operator = actual[name + "_loss_operator"]
        assert_allclose(operator, expected[name + "_loss"], rtol=3e-10, atol=1e-12)
        assert_allclose(operator, operator.conj().T, atol=2e-13)
        assert np.linalg.eigvalsh(operator).min() >= -1e-12
    returned = actual["returned_jones"]
    assert_allclose(returned.conj().T @ returned + expected["material_loss"] + expected["mirror_loss"], np.eye(2), atol=1e-12)


def test_asymmetric_back_reflection_is_required_in_mirror_cavity():
    layers = STACKS[2]
    stack = layer_stack_scattering(layers, 633)
    actual = stack_mirror_return(stack, gap_nm=79)
    expected = _oracle(layers, mirror=-1, gap=79)
    assert np.linalg.norm(stack["reflection"] - stack["reverse_reflection"]) > .05
    assert_allclose(actual["returned_jones"], expected["reflection"], rtol=3e-11, atol=3e-13)
    # This deliberately incorrect symmetric-slab substitution must disagree.
    load = -np.exp(2j * 2 * np.pi / 633 * 79)
    wrong = stack["reflection"] + stack["transmission"] @ np.linalg.solve(np.eye(2) - load * stack["reflection"], load * stack["transmission"])
    assert np.linalg.norm(actual["returned_jones"] - wrong) > .01


@pytest.mark.parametrize("host", [1.0, 1.33])
def test_single_layer_reduces_to_existing_exact_slab(host):
    layer = {"epsilon_diagonal": 2.7 + .08j, "g": .12 + .01j, "thickness_nm": 230}
    stack = layer_stack_scattering([layer], 633, host_index=host)
    slab = slab_scattering(**layer, wavelength_nm=633, host_index=host)
    for key in ("transmission", "reflection"):
        assert_allclose(stack[key], slab[key], rtol=2e-12, atol=2e-13)
        assert_allclose(stack["reverse_" + key], slab[key], rtol=2e-12, atol=2e-13)
    result, reference = stack_mirror_return(stack, gap_nm=137), mirror_return(slab, gap_nm=137)
    for key in ("returned_jones", "gap_forward_jones", "material_loss_operator", "mirror_loss_operator"):
        assert_allclose(result[key], reference[key], rtol=2e-12, atol=5e-13)


@pytest.mark.parametrize("layers", [[], [{"epsilon_diagonal": 2.7 + .08j, "g": .12 + .01j, "thickness_nm": 0}],
    [{"epsilon_diagonal": 0, "thickness_nm": 0}, {"epsilon_diagonal": 4, "g": 1, "thickness_nm": 0}]])
def test_empty_and_zero_thickness_stacks_are_identity(layers):
    stack = layer_stack_scattering(layers, 633, host_index=1.33)
    for prefix in ("", "reverse_"):
        assert_allclose(stack[prefix + "transmission"], np.eye(2), atol=2e-13)
        assert_allclose(stack[prefix + "reflection"], np.zeros((2, 2)), atol=2e-13)
    returned = stack_mirror_return(stack, gap_nm=137)
    assert_allclose(returned["returned_jones"], -np.exp(2j * 2 * np.pi * 1.33 / 633 * 137) * np.eye(2), atol=3e-13)
    assert_allclose(returned["material_loss_operator"], np.zeros((2, 2)), atol=5e-13)


def test_splitting_homogeneous_layers_and_inserting_zero_layers_changes_nothing():
    layers = STACKS[2]
    subdivided = []
    for layer in layers:
        for _ in range(7):
            subdivided.append({**layer, "thickness_nm": layer["thickness_nm"] / 7})
        subdivided.append({"epsilon_diagonal": 9 + .2j, "g": .5, "thickness_nm": 0})
    original, split = layer_stack_scattering(layers, 633), layer_stack_scattering(subdivided, 633)
    for prefix in ("", "reverse_"):
        for name in ("transmission", "reflection"):
            assert_allclose(split[prefix + name], original[prefix + name], rtol=3e-12, atol=5e-13)
    assert_allclose(stack_mirror_return(split, gap_nm=79)["returned_jones"], stack_mirror_return(original, gap_nm=79)["returned_jones"], rtol=3e-12, atol=5e-13)


def test_long_absorbing_stack_reaches_semi_infinite_boundary_without_overflow():
    epsilon, host = 4 + 3j, 1.33
    layers = [{"epsilon_diagonal": epsilon, "thickness_nm": 500}] * 120
    stack = layer_stack_scattering(layers, 633, host_index=host)
    # Semi-infinite medium has only a forward wave. Solve its two E/H equations.
    index = np.sqrt(epsilon)
    reflected, _ = np.linalg.solve(np.array([[1, -1], [-host, -index]], dtype=complex), [-1, -host])
    assert_allclose(stack["reflection"], reflected * np.eye(2), atol=2e-13)
    assert np.linalg.norm(stack["transmission"]) < 1e-100
    for value in (stack["reflection"], stack["transmission"], stack["reverse_reflection"], stack["reverse_transmission"]):
        assert np.isfinite(value).all()
    returned = stack_mirror_return(stack, gap_nm=137)
    assert_allclose(returned["returned_jones"], reflected * np.eye(2), atol=3e-13)
    assert_allclose(returned["material_loss_operator"], (1 - abs(reflected)**2) * np.eye(2), atol=5e-13)


def test_exact_zero_index_limit_matches_linear_internal_field():
    layers = [{"epsilon_diagonal": 0, "thickness_nm": 100}]
    actual = layer_stack_scattering(layers, 600)
    expected = _oracle(layers, wavelength=600)
    assert_allclose(actual["transmission"], expected["transmission"], atol=3e-13)
    assert_allclose(actual["reflection"], expected["reflection"], atol=3e-13)
    returned = stack_mirror_return(actual)
    assert_allclose(returned["returned_jones"], _oracle(layers, wavelength=600, mirror=-1)["reflection"], atol=3e-13)
    assert_allclose(returned["returned_jones"].conj().T @ returned["returned_jones"], np.eye(2), atol=5e-13)


def test_magnetization_reversal_transposes_all_jones_blocks_and_mirror_return():
    plus = layer_stack_scattering(STACKS[2], 633, magnetization=1)
    minus = layer_stack_scattering(STACKS[2], 633, magnetization=-1)
    for prefix in ("", "reverse_"):
        for name in ("transmission", "reflection"):
            assert_allclose(minus[prefix + name], plus[prefix + name].T, atol=3e-13)
    a, b = stack_mirror_return(plus, gap_nm=97), stack_mirror_return(minus, gap_nm=97)
    assert_allclose(b["returned_jones"], a["returned_jones"].T, atol=3e-13)
    assert abs(a["returned_jones"][1, 0]) > 1e-4


def test_zero_gyrotropy_has_no_polarization_conversion_even_in_asymmetric_stack():
    layers = [{**layer, "g": 0} for layer in STACKS[2]]
    stack = layer_stack_scattering(layers, 633)
    for prefix in ("", "reverse_"):
        for name in ("transmission", "reflection"):
            value = stack[prefix + name]
            assert_allclose(value, value[0, 0] * np.eye(2), atol=1e-14)
    returned = stack_mirror_return(stack, gap_nm=97)["returned_jones"]
    assert_allclose(returned, returned[0, 0] * np.eye(2), atol=1e-14)


@pytest.mark.parametrize("layer", [
    {"epsilon_diagonal": 2 - .1j, "thickness_nm": 20},
    {"epsilon_diagonal": 2 + .01j, "g": .1j, "thickness_nm": 20},
    {"epsilon_diagonal": 2, "thickness_nm": -1},
    {"epsilon_diagonal": 2, "thickness_nm": True},
    {"epsilon_diagonal": 2, "thickness_nm": float("inf")},
    {"epsilon_diagonal": 2, "thickness_nm": 20, "thikness_nm": 30},
])
def test_active_or_invalid_layer_rejected(layer):
    with pytest.raises(ValueError):
        layer_stack_scattering([layer], 633)


@pytest.mark.parametrize("kwargs", [{"wavelength_nm": 0}, {"host_index": 0}, {"host_index": 1 + .1j}, {"magnetization": 1.1}])
def test_invalid_global_controls_rejected(kwargs):
    parameters = {"layers": STACKS[0], "wavelength_nm": 633}
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        layer_stack_scattering(**parameters)


@pytest.mark.parametrize("kwargs", [{"gap_nm": -1}, {"mirror_amplitude": 1.01}, {"mirror_amplitude": complex(float("nan"), 0)}])
def test_invalid_mirror_termination_rejected(kwargs):
    with pytest.raises(ValueError):
        stack_mirror_return(layer_stack_scattering(STACKS[0], 633), **kwargs)


def test_lossless_quarter_wave_bragg_stack_matches_closed_impedance_solution():
    # Each quarter-wave layer transforms load admittance Y into n^2/Y.
    # An H,L pair multiplies it by (nH/nL)^2, giving a closed independent oracle.
    wavelength, high, low, pairs, host = 600.0, 3.0, 1.5, 8, 1.33
    layers = [{"epsilon_diagonal": index**2, "thickness_nm": wavelength / (4 * index)}
              for _ in range(pairs) for index in (high, low)]
    actual = layer_stack_scattering(layers, wavelength, host_index=host)
    impedance_ratio = (high / low)**(2 * pairs)
    expected_r = (1 - impedance_ratio) / (1 + impedance_ratio)
    expected_t = (-1)**pairs * 2 * np.sqrt(impedance_ratio) / (1 + impedance_ratio)
    assert_allclose(actual["reflection"], expected_r * np.eye(2), atol=5e-13)
    assert_allclose(actual["reverse_reflection"], -expected_r * np.eye(2), atol=5e-13)
    assert_allclose(actual["transmission"], expected_t * np.eye(2), rtol=3e-12, atol=2e-15)
    for prefix in ("", "reverse_"):
        t, r = actual[prefix + "transmission"], actual[prefix + "reflection"]
        assert_allclose(t.conj().T @ t + r.conj().T @ r, np.eye(2), atol=5e-13)


@pytest.mark.parametrize("wavelength_offset", [-.1, 0.0, .1])
@pytest.mark.parametrize("extra_mo_loss", [0.0, .001])
def test_shipped_21_layer_bragg_cavity_against_independent_boundary_oracle(wavelength_offset, extra_mo_loss):
    path = Path(__file__).resolve().parents[1] / "examples" / "superlattice" / "bragg_cavity.json"
    fixture = json.loads(path.read_text(encoding="utf-8"))
    layers, mo_count = [], 0
    for entry in fixture["layers"]:
        # Parse the stored complex pairs explicitly, without a production loader.
        epsilon = complex(entry["epsilon_diagonal"][0], entry["epsilon_diagonal"][1])
        coupling = complex(entry["g"][0], entry["g"][1])
        if entry["name"] == "MO":
            epsilon += 1j * extra_mo_loss
            mo_count += 1
        # Names are provenance, not constitutive parameters in the oracle.
        layers.append({"epsilon_diagonal": epsilon, "g": coupling, "thickness_nm": entry["thickness_nm"]})
    assert len(layers) == 21 and mo_count == 1
    wavelength = fixture["wavelength_nm"] + wavelength_offset
    host, magnetization, gap = fixture["host_index"], fixture["magnetization_z"], fixture["gap_nm"]
    mirror = complex(*fixture["mirror_amplitude"])
    actual = layer_stack_scattering(layers, wavelength, host_index=host, magnetization=magnetization)
    for prefix, ordered, loss_key in [("", layers, "left_material_loss_operator"), ("reverse_", layers[::-1], "right_material_loss_operator")]:
        expected = _oracle(ordered, wavelength=wavelength, host=host, magnetization=magnetization)
        for name in ("transmission", "reflection"):
            assert_allclose(actual[prefix + name], expected[name], rtol=3e-10, atol=3e-12)
        assert_allclose(actual[loss_key], expected["material_loss"], rtol=3e-9, atol=3e-12)
    returned = stack_mirror_return(actual, gap_nm=gap, mirror_amplitude=mirror)
    expected_return = _oracle(layers, wavelength=wavelength, host=host, magnetization=magnetization, mirror=mirror, gap=gap)
    assert_allclose(returned["returned_jones"], expected_return["reflection"], rtol=3e-10, atol=3e-12)
    for name in ("material", "mirror"):
        assert_allclose(returned[name + "_loss_operator"], expected_return[name + "_loss"], rtol=3e-9, atol=3e-12)
    returned_field = returned["returned_jones"][:, 0]
    returned_power = float(np.vdot(returned_field, returned_field).real)
    assert_allclose(returned_power + expected_return["material_loss"][0, 0].real + expected_return["mirror_loss"][0, 0].real, 1, atol=3e-12)
    if extra_mo_loss:
        assert 0 < returned_power < 1
        assert expected_return["material_loss"][0, 0].real > 0
        if wavelength_offset == 0:
            normalized_s3 = 2 * np.imag(np.conj(returned_field[0]) * returned_field[1]) / returned_power
            assert abs(normalized_s3) > 1e-6  # Detectably elliptical, not a pure 90-degree line.
    else:
        assert_allclose(returned_power, 1, atol=3e-12)
        if wavelength_offset == 0:
            transmitted = actual["transmission"][:, 0]
            assert_allclose(transmitted[1] / transmitted[0], 1, atol=2e-9)
            # Forward power is set by the independently solved resonant stack;
            # no rounded 0.500037 golden value is used as a reference.
            independent = _oracle(layers, wavelength=wavelength, host=host, magnetization=magnetization)
            assert_allclose(np.vdot(transmitted, transmitted).real,
                            np.vdot(independent["transmission"][:, 0], independent["transmission"][:, 0]).real,
                            rtol=3e-10, atol=3e-12)
            assert abs(returned_field[0]) < 1e-9
            assert_allclose(abs(returned_field[1])**2, 1, atol=3e-12)
