"""Mirror feedback checked against independent boundary fields and Joule heat."""
import copy

import numpy as np
import pytest
from numpy.testing import assert_allclose

from magnetooptics.grating import grating_scattering
from magnetooptics.grating_mirror import grating_mirror_return
from magnetooptics.multilayer import layer_stack_scattering, stack_mirror_return


def _uniform(epsilon, thickness, g=0):
    return {"epsilon_a": epsilon, "epsilon_b": epsilon, "g_a": g, "g_b": g,
            "fill_fraction": 1., "thickness_nm": thickness}


def _boundary_oracle(layers, wavelength, left_index, right_index, gap, mirror, magnetization=1):
    """Solve every E/H interface simultaneously and integrate volume heating."""
    result = []
    nodes, weights = np.polynomial.legendre.leggauss(64)
    k0 = 2*np.pi/wavelength
    for helicity in (1, -1):
        eps = np.array([v["epsilon_a"]-helicity*magnetization*v["g_a"] for v in layers])
        indices = np.sqrt(eps.astype(complex))
        thickness = np.array([v["thickness_nm"] for v in layers])
        phases = np.exp(1j*k0*indices*thickness)
        count = len(layers)
        # Unknowns: external reflection, F/B for each layer, gap F/B.
        matrix = np.zeros((2*count+3, 2*count+3), complex)
        rhs = np.zeros(2*count+3, complex)
        matrix[0, :3], rhs[0] = [1, -1, -1], -1
        matrix[1, :3], rhs[1] = [-left_index, -indices[0], indices[0]], -left_index
        for i in range(count):
            row, own = 2+2*i, 1+2*i
            nxt = own+2
            next_index = indices[i+1] if i+1 < count else right_index
            matrix[row, own:own+2] = [phases[i], 1/phases[i]]
            matrix[row+1, own:own+2] = [indices[i]*phases[i], -indices[i]/phases[i]]
            matrix[row, nxt:nxt+2] = [-1, -1]
            matrix[row+1, nxt:nxt+2] = [-next_index, next_index]
        load = mirror*np.exp(2j*k0*right_index*gap)
        matrix[-1, -2:] = [-load, 1]
        solution = np.linalg.solve(matrix, rhs)
        heat = 0.
        for i in range(count):
            z = .5*thickness[i]*(nodes+1)
            a, b = solution[1+2*i:3+2*i]
            field = a*np.exp(1j*k0*indices[i]*z)+b*np.exp(-1j*k0*indices[i]*z)
            heat += k0/left_index*eps[i].imag*.5*thickness[i]*np.dot(weights, abs(field)**2)
        mirror_heat = right_index/left_index*(1-abs(mirror)**2)*abs(solution[-2])**2
        result.append((solution[0], heat, mirror_heat))
    plus, minus = np.array([1, 1j])/np.sqrt(2), np.array([1, -1j])/np.sqrt(2)
    return tuple(result[0][i]*np.outer(plus, plus.conj())
                 + result[1][i]*np.outer(minus, minus.conj()) for i in range(3))


@pytest.mark.parametrize("order", [0, 3])
@pytest.mark.parametrize("gap", [0., 137.])
def test_empty_structure_pec_has_known_sign_phase_and_unit_power(order, gap):
    wavelength, host = 633., 1.33
    bare = grating_scattering(wavelength, 400., [], n_incident=host, n_exit=host, order=order)
    actual = grating_mirror_return(bare, gap)
    expected = -np.exp(2j*2*np.pi/wavelength*host*gap)*np.eye(2)
    assert_allclose(actual["returned_jones"], expected, atol=2e-13)
    assert_allclose(actual["returned_power"], [1, 1], atol=2e-13)
    assert_allclose(actual["material_loss_operator"], 0, atol=3e-13)
    assert_allclose(actual["mirror_loss_operator"], 0, atol=3e-13)


@pytest.mark.parametrize("magnetization", [-1., 1.])
@pytest.mark.parametrize("right_index", [1., 1.47])
@pytest.mark.parametrize("mirror,order", [(-1., 3), (-.8, 3), (.8*np.exp(.37j), 0)])
def test_uniform_stack_matches_independent_boundary_fields_and_volume_heat(magnetization, right_index, mirror, order):
    layers = [_uniform(2.4+.07j, 55., .08+.01j),
              _uniform(5+.2j, 97., -.14+.01j), _uniform(1.7+.03j, 83.)]
    bare = grating_scattering(633., 400., layers, n_exit=right_index,
                              order=order, magnetization=magnetization)
    actual = grating_mirror_return(bare, gap_nm=79., mirror_amplitude=mirror)
    expected = _boundary_oracle(layers, 633., 1., right_index, 79., mirror, magnetization)
    assert_allclose(actual["returned_jones"], expected[0], rtol=2e-10, atol=2e-12)
    assert_allclose(actual["material_loss_operator"], expected[1], rtol=3e-10, atol=2e-12)
    assert_allclose(actual["mirror_loss_operator"], expected[2], rtol=3e-10, atol=2e-12)
    assert_allclose(actual["returned_power"]+actual["material_absorbed_power"]
                    + actual["mirror_absorbed_power"], 1., atol=2e-13)
    if right_index != 1:
        assert np.linalg.norm(bare["transmission"]-bare["reverse_transmission"]) > .1


def test_uniform_rcwa_termination_reduces_to_existing_multilayer_reference():
    layers = [{"epsilon_diagonal": 2.6+.04j, "g": .05+.002j, "thickness_nm": 150.},
              {"epsilon_diagonal": 1.5**2, "g": 0., "thickness_nm": 500000.}]
    exact = stack_mirror_return(layer_stack_scattering(layers, 1560.), gap_nm=230.)
    grating_layers = [_uniform(v["epsilon_diagonal"], v["thickness_nm"], v["g"]) for v in layers]
    actual = grating_mirror_return(grating_scattering(1560., 966., grating_layers, n_exit=1., order=3), 230.)
    assert_allclose(actual["returned_jones"], exact["returned_jones"], atol=2e-10)
    assert_allclose(actual["first_return_jones"], exact["first_return_jones"], atol=2e-10)
    assert_allclose(actual["material_loss_operator"], exact["material_loss_operator"], atol=2e-10)


def _pattern(g=.03):
    return {"epsilon_a": 4.1, "g_a": 0., "epsilon_b": 2.5+.08j, "g_b": g,
            "fill_fraction": .55, "thickness_nm": 230.}


def test_gap_equals_explicit_exit_medium_layer_including_evanescent_feedback():
    layer = _pattern()
    bare = grating_scattering(700., 430., [layer], n_exit=1., order=5)
    actual = grating_mirror_return(bare, 21.)
    explicit = grating_mirror_return(grating_scattering(700., 430.,
        [layer, _uniform(1., 21.)], n_exit=1., order=5), 0.)
    assert_allclose(actual["returned_jones"], explicit["returned_jones"], atol=2e-11)
    assert_allclose(actual["material_absorbed_power"], explicit["material_absorbed_power"], atol=2e-11)
    # Deliberately discarding nonzero-order feedback must change this near-field
    # geometry, even though all its nonzero exterior orders are evanescent.
    central = [5, 16]
    zero_only = {**bare, "harmonic_orders": np.array([0]), "q_exit": np.array([1.]),
                 "flux_weights_incident": np.ones(2), "flux_weights_exit": np.ones(2),
                 "scattering": {key: value[np.ix_(central, central)]
                                for key, value in bare["scattering"].items()}}
    wrong = grating_mirror_return(zero_only, 21.)
    assert np.linalg.norm(actual["returned_jones"]-wrong["returned_jones"]) > 1e-4


def test_magnetic_reversal_transposes_patterned_return_and_zero_g_has_no_conversion():
    params = dict(wavelength_nm=700., period_nm=430., n_exit=1., order=5)
    plus = grating_mirror_return(grating_scattering(**params, layers=[_pattern()], magnetization=1), 61.)
    minus = grating_mirror_return(grating_scattering(**params, layers=[_pattern()], magnetization=-1), 61.)
    zero = grating_mirror_return(grating_scattering(**params, layers=[_pattern(0)]), 61.)
    assert_allclose(plus["returned_jones"], minus["returned_jones"].T, atol=2e-11)
    assert_allclose(zero["returned_jones"][[0, 1], [1, 0]], 0, atol=2e-12)
    assert abs(plus["returned_jones"][1, 0]) > 1e-4


def test_all_returned_diffraction_orders_are_counted_for_lossless_pattern():
    layer = _pattern(g=.03)
    layer["epsilon_b"] = 2.5
    result = grating_mirror_return(grating_scattering(700., 1000., [layer], n_exit=1., order=5), 60.)
    assert_allclose(result["returned_power"], [1, 1], atol=2e-10)
    assert_allclose(result["material_absorbed_power"], 0, atol=2e-10)
    assert result["reflection_by_order"][[4, 6]].sum() > .01
    assert np.sum(abs(result["returned_jones"])**2, axis=0).max() < .99


@pytest.mark.parametrize("kwargs", [{"gap_nm": -1}, {"gap_nm": np.inf}, {"gap_nm": True},
                                    {"mirror_amplitude": 1.1}, {"mirror_amplitude": np.nan},
                                    {"mirror_amplitude": -.9+.01j}])
def test_invalid_mirror_or_gap_is_rejected(kwargs):
    bare = grating_scattering(700., 430., [], n_exit=1., order=3)
    with pytest.raises(ValueError):
        grating_mirror_return(bare, **kwargs)


def test_invalid_result_data_is_rejected_without_mutating_original():
    bare = grating_scattering(700., 430., [], n_exit=1., order=3)
    original = copy.deepcopy(bare)
    grating_mirror_return(bare, 21.)
    assert_allclose(bare["scattering"]["r_left"], original["scattering"]["r_left"])
    for key in ["scattering", "q_exit", "flux_weights_incident"]:
        broken = copy.deepcopy(bare)
        del broken[key]
        with pytest.raises(ValueError):
            grating_mirror_return(broken)
    broken = copy.deepcopy(bare)
    broken["scattering"]["t_reverse"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        grating_mirror_return(broken)
