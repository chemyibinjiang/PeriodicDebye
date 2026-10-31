"""Independent Maxwell boundary, heating, symmetry and convergence tests.

The patterned-layer oracle integrates Fourier coefficients by quadrature and
solves a global boundary equation using a matrix exponential. It does not call
the production modal construction or scattering recursion. This oracle is used
only for thin layers, where its growing transfer exponentials remain harmless.
"""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.linalg import expm
from threadpoolctl import threadpool_limits

from magnetooptics.grating import grating_scattering
from magnetooptics.multilayer import layer_stack_scattering


@pytest.fixture(autouse=True)
def _single_blas_thread():
    with threadpool_limits(limits=1):
        yield


def _layer(epsilon=3.8025, g=0., fill=.63, thickness=593.,
           epsilon_b=2.856064 + .02028j, g_b=-.00375 - .0018j):
    return dict(epsilon_a=epsilon, g_a=g, epsilon_b=epsilon_b, g_b=g_b,
                fill_fraction=fill, thickness_nm=thickness)


def _uniform_boundary(epsilon, g, thickness, wavelength, left, right):
    """Solve four E/H continuity equations for each circular eigenchannel."""
    eigenvectors = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    values = {"transmission": [], "reflection": [], "absorption": []}
    nodes, weights = np.polynomial.legendre.leggauss(48)
    k0 = 2 * np.pi / wavelength
    for circular_epsilon in (epsilon - g, epsilon + g):
        index = np.sqrt(complex(circular_epsilon))
        phase = np.exp(1j * k0 * index * thickness)
        matrix = np.array([
            [1, -1, -1, 0], [-left, -index, index, 0],
            [0, phase, 1 / phase, -1],
            [0, index * phase, -index / phase, -right]], complex)
        r, a, b, t = np.linalg.solve(matrix, [-1, -left, 0, 0])
        z = thickness * (nodes + 1) / 2
        field = a * np.exp(1j * k0 * index * z) + b * np.exp(-1j * k0 * index * z)
        heating = (k0 / left * complex(circular_epsilon).imag * thickness / 2
                   * np.dot(weights, abs(field)**2))
        values["reflection"].append(r)
        values["transmission"].append(t)
        values["absorption"].append(heating)
    return {name: eigenvectors @ np.diag(value) @ eigenvectors.conj().T
            for name, value in values.items()}


def _quadrature_convolution(value_a, value_b, fill, orders):
    """Integral of a centered strip, with no sinc/Toeplitz production helper."""
    nodes, weights = np.polynomial.legendre.leggauss(40)
    result = np.zeros((len(orders), len(orders)), complex)
    for lower, upper, value in [(-.5, -fill/2, value_b),
                                (-fill/2, fill/2, value_a),
                                (fill/2, .5, value_b)]:
        positions = (upper - lower) / 2 * nodes + (upper + lower) / 2
        weighted = (upper - lower) / 2 * weights
        for row, m in enumerate(orders):
            for column, n in enumerate(orders):
                result[row, column] += value * np.dot(
                    weighted, np.exp(-2j * np.pi * (m - n) * positions))
    return result


def _transfer_boundary_oracle(layer, wavelength, period, order, left, right, *,
                              heating=False, state_front=None, operators=False):
    """Independent first-order Maxwell propagator and simultaneous boundary solve."""
    modes = np.arange(-order, order + 1)
    count = len(modes)
    identity, zero = np.eye(count), np.zeros((count, count))
    kx = np.diag(modes * wavelength / period)
    ea, eb, ga, gb = (layer[name] for name in ("epsilon_a", "epsilon_b", "g_a", "g_b"))
    conv = lambda a, b: _quadrature_convolution(a, b, layer["fill_fraction"], modes)
    # Dx and Ey are continuous across vertical interfaces. The inverse rule
    # below follows from Ex=(Dx-epsilon_xy Ey)/epsilon_xx, before truncation.
    a = np.linalg.inv(conv(1 / ea, 1 / eb))
    b = a @ conv(1j * ga / ea, 1j * gb / eb)
    c = conv(-1j * ga / ea, -1j * gb / eb) @ a
    d = conv(ea - ga**2 / ea, eb - gb**2 / eb) + conv(-1j * ga / ea, -1j * gb / eb) @ b
    longitudinal = identity - kx @ np.linalg.solve(conv(ea, eb), kx)
    maxwell = np.block([[zero, zero, zero, longitudinal],
                        [zero, zero, -identity, zero],
                        [-c, kx @ kx - d, zero, zero],
                        [a, b, zero, zero]])
    propagation = expm(2j * np.pi / wavelength * layer["thickness_nm"] * maxwell)

    def port(index, direction):
        q = np.sqrt((index**2 - np.diag(kx)**2).astype(complex))
        return np.block([[identity, zero], [zero, identity],
                         [zero, -direction * np.diag(q)],
                         [direction * np.diag(index**2 / q), zero]])

    if operators:
        return propagation, port
    matrix = np.column_stack((propagation @ port(left, -1), -port(right, 1)))
    rhs = np.column_stack((-propagation @ port(left, 1), port(right, -1)))
    result = np.linalg.solve(matrix, rhs)
    size = 2 * count
    if heating:
        # Reconstruct discontinuous Ex from continuous Dx and Ey pointwise,
        # then integrate the ORIGINAL constituent's Hermitian loss tensor.
        # Neither flux differences nor the production eigenmodes enter this.
        if state_front is None:
            state_front = (port(left, 1) + port(left, -1) @ result[:size, :size])[:, [order, count+order]]
        nodes, weights = np.polynomial.legendre.leggauss(32)
        integral = np.zeros((2, 2), complex)
        for z, weight_z in zip(layer["thickness_nm"]*(nodes+1)/2,
                               layer["thickness_nm"]*weights/2):
            state = expm(2j*np.pi/wavelength*z*maxwell) @ state_front
            dx = a @ state[:count] + b @ state[count:2*count]
            ez = -np.linalg.solve(conv(ea, eb), kx @ state[3*count:])
            fill = layer["fill_fraction"]
            for lo, hi, epsilon, coupling in [(-.5, -fill/2, eb, gb),
                                               (-fill/2, fill/2, ea, ga),
                                               (fill/2, .5, eb, gb)]:
                x = (hi-lo)*nodes/2 + (hi+lo)/2
                phase = np.exp(2j*np.pi*x[:, None]*modes)
                ey_field = phase @ state[count:2*count]
                ex_field = (phase @ dx - 1j*coupling*ey_field)/epsilon
                electric = np.stack((ex_field, ey_field, phase @ ez), axis=1)
                loss = np.array([[complex(epsilon).imag, 1j*complex(coupling).imag, 0],
                                 [-1j*complex(coupling).imag, complex(epsilon).imag, 0],
                                 [0, 0, complex(epsilon).imag]])
                integral += weight_z*np.einsum("xia,ij,xjb,x->ab", electric.conj(), loss,
                                               electric, (hi-lo)*weights/2)
        return 2*np.pi/wavelength/left*integral
    return {"r_left": result[:size, :size], "t_reverse": result[:size, size:],
            "t_forward": result[size:, :size], "r_right": result[size:, size:]}


def _stack_boundary_heating_oracle(layers, wavelength, period, order, left, right):
    """All interface states solved simultaneously; no production S cascade."""
    count = 2*order+1
    half, state_size = 2*count, 4*count
    size = (len(layers)+1)*state_size
    operators = [_transfer_boundary_oracle(layer, wavelength, period, order, left, right,
                                           operators=True) for layer in layers]
    port = operators[0][1]
    matrix, rhs = np.zeros((size, size), complex), np.zeros((size, 2), complex)
    central = [order, count+order]
    matrix[:state_size, :half] = -port(left, -1)
    matrix[:state_size, half:half+state_size] = np.eye(state_size)
    rhs[:state_size] = port(left, 1)[:, central]
    for number, (propagation, _) in enumerate(operators):
        row = (number+1)*state_size
        column = half+number*state_size
        matrix[row:row+state_size, column:column+state_size] = propagation
        if number < len(layers)-1:
            matrix[row:row+state_size, column+state_size:column+2*state_size] = -np.eye(state_size)
        else:
            matrix[row:row+state_size, -half:] = -port(right, 1)
    solution = np.linalg.solve(matrix, rhs)
    heating = np.zeros((2, 2), complex)
    for number, layer in enumerate(layers):
        column = half+number*state_size
        heating += _transfer_boundary_oracle(layer, wavelength, period, order, left, right,
            heating=True, state_front=solution[column:column+state_size])
    return dict(reflection=solution[:half][central], transmission=solution[-half:][central],
                material_loss_operator=heating)


@pytest.mark.parametrize("left,right", [(1., 1.5), (1.33, 1.), (1., 1.)])
@pytest.mark.parametrize("order", [0, 3])
def test_empty_stack_matches_both_directions_of_fresnel_interface(left, right, order):
    result = grating_scattering(1550, 966, [], n_incident=left, n_exit=right, order=order)
    identity = np.eye(2)
    assert_allclose(result["reflection"], (left-right)/(left+right)*identity, atol=2e-13)
    assert_allclose(result["transmission"], 2*left/(left+right)*identity, atol=2e-13)
    assert_allclose(result["reverse_reflection"], (right-left)/(left+right)*identity, atol=2e-13)
    assert_allclose(result["reverse_transmission"], 2*right/(left+right)*identity, atol=2e-13)
    assert_allclose(result["transmitted_power"], 4*left*right/(left+right)**2, atol=2e-13)
    assert_allclose(result["reflected_power"] + result["transmitted_power"], 1., atol=3e-13)


@pytest.mark.parametrize("fill", [0., 1.])
@pytest.mark.parametrize("magnetization", [-1., 0., 1.])
def test_uniform_gyrotropic_layer_matches_boundary_and_volume_heating(fill, magnetization):
    epsilon, g, thickness = 2.856064+.02028j, -.00375-.0018j, 593.
    layer = _layer(epsilon=epsilon if fill else 4.1, g=g if fill else .12,
                   fill=fill, thickness=thickness,
                   epsilon_b=4.1 if fill else epsilon, g_b=.12 if fill else g)
    original = copy.deepcopy(layer)
    result = grating_scattering(1550, 966, [layer], n_incident=1., n_exit=1.5,
                                order=3, magnetization=magnetization)
    assert layer == original
    for prefix, left, right in [("", 1., 1.5), ("reverse_", 1.5, 1.)]:
        expected = _uniform_boundary(epsilon, g*magnetization, thickness, 1550, left, right)
        for name in ("transmission", "reflection"):
            assert_allclose(result[prefix+name], expected[name], rtol=3e-11, atol=2e-12)
        t, r = result[prefix+"transmission"], result[prefix+"reflection"]
        missing = np.eye(2) - r.conj().T @ r - right/left * t.conj().T @ t
        assert_allclose(missing, expected["absorption"], rtol=3e-10, atol=3e-12)
        if not prefix:
            assert_allclose(result["absorbed_power"], np.diag(expected["absorption"]).real,
                            rtol=3e-10, atol=3e-12)


def test_uniform_multilayer_matches_existing_exact_solver():
    uniform = [dict(epsilon_diagonal=3.8025, g=0., thickness_nm=193.),
               dict(epsilon_diagonal=2.856064+.02028j, g=-.00375-.0018j, thickness_nm=593.),
               dict(epsilon_diagonal=2.2, g=0., thickness_nm=73.)]
    layers = [_layer(epsilon=s["epsilon_diagonal"], g=s["g"], fill=1.,
                     thickness=s["thickness_nm"]) for s in uniform]
    result = grating_scattering(1550, 966, layers, n_incident=1.33, n_exit=1.33, order=3)
    expected = layer_stack_scattering(uniform, 1550, host_index=1.33)
    for name in ("transmission", "reflection", "reverse_transmission", "reverse_reflection"):
        assert_allclose(result[name], expected[name], rtol=3e-11, atol=2e-12)


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("lossy", [False, True])
def test_patterned_layer_matches_independent_matrix_exponential_boundary_solve(order, lossy):
    layer = _layer(epsilon=3.8+.03j if lossy else 3.8, g=.04+.007j if lossy else .04,
                   fill=.47, thickness=83., epsilon_b=2.7+.04j if lossy else 2.7,
                   g_b=-.03-.008j if lossy else -.03)
    result = grating_scattering(1550, 966, [layer], n_incident=1., n_exit=1.5, order=order)
    expected = _transfer_boundary_oracle(layer, 1550, 966, order, 1., 1.5)
    for name in expected:
        assert_allclose(result["scattering"][name], expected[name], rtol=2e-10, atol=3e-12)


@pytest.mark.parametrize("order", [1, 2])
def test_patterned_loss_matches_independent_pointwise_volume_heating(order):
    layer = _layer(epsilon=3.8+.03j, g=.04+.007j, fill=.47, thickness=83.,
                   epsilon_b=2.7+.04j, g_b=-.03-.008j)
    result = grating_scattering(1550, 966, [layer], n_incident=1., n_exit=1.5, order=order)
    heating = _transfer_boundary_oracle(layer, 1550, 966, order, 1., 1.5, heating=True)
    assert_allclose(result["material_loss_operator"], heating, rtol=2e-10, atol=3e-12)
    assert np.linalg.eigvalsh(heating).min() > 0


@pytest.mark.parametrize("magnetization", [-1., 0., 1.])
def test_lossless_diffracting_grating_conserves_all_order_flux(magnetization):
    layer = _layer(epsilon=3.8, g=.03, fill=.57, thickness=371., epsilon_b=2.7, g_b=-.02)
    result = grating_scattering(1550, 2400, [layer], n_incident=1., n_exit=1.5,
                                order=4, magnetization=magnetization)
    assert_allclose(result["transmitted_power"]+result["reflected_power"], 1., atol=2e-11)
    assert_allclose(result["transmission_by_order"].sum(axis=0), result["transmitted_power"], atol=2e-13)
    assert_allclose(result["reflection_by_order"].sum(axis=0), result["reflected_power"], atol=2e-13)
    zero = 4
    diffracted = result["transmission_by_order"].sum(axis=0)-result["transmission_by_order"][zero]
    assert np.min(diffracted) > .005
    assert_allclose(result["transmission_by_order"][[0, 1, 7, 8]], 0., atol=1e-15)
    assert_allclose(result["reflection_by_order"][[0, 1, 2, 6, 7, 8]], 0., atol=1e-15)


def test_magnetization_reversal_and_zero_magnetization_conversion():
    layer = _layer()
    results = {m: grating_scattering(1560, 966, [layer], order=5, magnetization=m)
               for m in (-1, 0, 1)}
    parity = np.diag([1., -1.])
    for name in ("transmission", "reflection", "reverse_transmission", "reverse_reflection"):
        assert_allclose(results[-1][name], parity @ results[1][name] @ parity, atol=2e-11)
        assert abs(results[0][name][0, 1]) < 2e-12
        assert abs(results[0][name][1, 0]) < 2e-12
    assert abs(results[1]["transmission"][1, 0]) > 1e-4
    assert_allclose(results[-1]["transmitted_power"], results[1]["transmitted_power"], atol=2e-11)


def test_subdividing_pattern_and_inserting_zero_layer_preserves_response():
    layer = _layer(thickness=593.)
    original = grating_scattering(1550, 966, [layer], order=5)
    split = [_layer(thickness=0., fill=.37)] + [{**layer, "thickness_nm": 593/3}]*3
    actual = grating_scattering(1550, 966, split, order=5)
    for name in ("transmission", "reflection", "reverse_transmission", "reverse_reflection"):
        assert_allclose(actual[name], original[name], atol=2e-11, rtol=2e-11)
    assert_allclose(actual["absorbed_power"], original["absorbed_power"], atol=2e-11)


def test_opaque_uniform_layer_has_finite_semi_infinite_reflection():
    epsilon, g = 2.856064+.02028j, -.00375-.0018j
    layer = _layer(epsilon=epsilon, g=g, fill=1., thickness=1e8)
    result = grating_scattering(1550, 966, [layer], n_incident=1., n_exit=1.5, order=3)
    indices = np.sqrt(np.array([epsilon-g, epsilon+g]))
    eigenvectors = np.array([[1, 1], [1j, -1j]]) / np.sqrt(2)
    expected = eigenvectors @ np.diag((1-indices)/(1+indices)) @ eigenvectors.conj().T
    assert_allclose(result["reflection"], expected, atol=3e-12)
    assert np.linalg.norm(result["transmission"]) < 1e-100
    for name in ("transmission", "reflection", "reverse_transmission", "reverse_reflection"):
        assert np.isfinite(result[name]).all()


@pytest.mark.parametrize("kwargs", [dict(order=-1), dict(order=1.5), dict(order=True),
                                    dict(wavelength_nm=0), dict(period_nm=0),
                                    dict(n_incident=1+.1j), dict(n_exit=-1),
                                    dict(magnetization=1.01), dict(magnetization=np.nan)])
def test_invalid_geometry_or_illumination_rejected(kwargs):
    parameters = dict(wavelength_nm=1550, period_nm=966, layers=[_layer()], order=2)
    parameters.update(kwargs)
    with pytest.raises(ValueError):
        grating_scattering(**parameters)


@pytest.mark.parametrize("override", [dict(thickness_nm=-1), dict(fill_fraction=-.1),
                                       dict(fill_fraction=1.1), dict(epsilon_a=3-.1j),
                                       dict(epsilon_b=2+.001j, g_b=.01+.1j),
                                       dict(epsilon_a=np.nan), dict(g_a=np.inf)])
def test_invalid_layers_and_active_tensors_rejected(override):
    layer = {**_layer(), **override}
    with pytest.raises(ValueError):
        grating_scattering(1550, 966, [layer], order=2)


def test_exact_rayleigh_anomaly_is_explicitly_rejected():
    with pytest.raises(ValueError, match="(?i)(rayleigh|grazing|cutoff)"):
        grating_scattering(1550, 1550, [_layer()], n_incident=1., n_exit=1.5, order=3)


@pytest.mark.parametrize("order", [0, 1])
def test_truncation_cannot_silently_omit_open_diffraction_orders(order):
    # The substrate supports m=+/-2 although the air port only supports +/-1.
    with pytest.raises(ValueError, match="(?i)(propagating|open|retain|order)"):
        grating_scattering(1550, 2400, [_layer()], n_incident=1., n_exit=1.5, order=order)


@pytest.mark.parametrize("wavelength", [1550., 1560., 1570.])
def test_reference_geometry_fourier_convergence(wavelength):
    """A local numerical convergence check, not a fit to the published device."""
    layer = _layer(fill=608/966, thickness=593.)
    results = {order: grating_scattering(wavelength, 966, [layer], n_exit=1.5, order=order)
               for order in (11, 21, 31)}
    names = ("transmission", "reflection", "reverse_transmission", "reverse_reflection")
    changes = [max(np.max(abs(results[a][name]-results[b][name])) for name in names)
               for a, b in ((11, 21), (21, 31))]
    assert changes[1] < 1e-5
    assert changes[1] < .5*changes[0]
    assert_allclose(results[21]["transmitted_power"], results[31]["transmitted_power"],
                    rtol=0, atol=2e-6)


@pytest.mark.parametrize("order", [2, 3])
def test_selected_real_material_stack_matches_independent_boundary_and_heating(order):
    """Selected geometry at modest truncation; separate tests assess convergence.

    The global expm oracle becomes ill-conditioned at large harmonic cutoffs;
    using it at orders 2 and 3 validates the resonant three-layer Maxwell/loss
    implementation, without claiming these low orders predict final accuracy.
    """
    from magnetooptics.grating_design import nbk7_index, prototype_layers
    geometry = dict(period_nm=925.0420768002252, fill_fraction=.7475748750287126,
                    height_nm=573.7393944066401, cap_nm=82.05068975768243, overetch_nm=30.)
    wavelength, left, right = 1560., 1., nbk7_index(1560.)
    layers = prototype_layers(geometry, 2.856064+.02028j, -.00375-.0018j, right)
    result = grating_scattering(wavelength, geometry["period_nm"], layers,
                                n_incident=left, n_exit=right, order=order)
    expected = _stack_boundary_heating_oracle(layers, wavelength, geometry["period_nm"],
                                             order, left, right)
    for name in ("transmission", "reflection", "material_loss_operator"):
        assert_allclose(result[name], expected[name], rtol=3e-10, atol=3e-12)
    assert expected["material_loss_operator"][1, 1].real > .1
