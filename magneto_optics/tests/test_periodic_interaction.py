"""Ewald convergence, analytic derivatives, phase conventions, and guard tests."""
import numpy as np
import pytest
from scipy.special import erfc

from magnetooptics.periodic_interaction import PeriodicGreenOperator, EwaldConvergenceError, _short_tensor
from magnetooptics.solver import CoupledDipoleSystem


POSITIONS = [[5., 8., 0.], [25., 31., 0.], [66., 75., 0.]]


def test_three_backends_identical_with_multi_site_tensor():
    operators = [PeriodicGreenOperator(POSITIONS, [100., 120.], .017, backend=backend)
                 for backend in ("reference", "vectorized", "cached")]
    for operator in operators[1:]:
        np.testing.assert_allclose(operator.matrix, operators[0].matrix, rtol=2e-13, atol=3e-20)
    assert operators[-1].metadata["pair_evaluations"] < len(POSITIONS)**2
    assert operators[1].metadata["pair_evaluations"] == len(POSITIONS)**2
    alpha = np.array([[800+40j, 60j, 0], [-60j, 800+40j, 0], [0, 0, 600+20j]])
    electric = np.tile([1, .3j, 0], (len(POSITIONS), 1))
    systems = [CoupledDipoleSystem(operator, alpha, "dense") for operator in operators]
    results = [system.solve(electric) for system in systems]
    for result in results[1:]:
        np.testing.assert_allclose(result.dipoles, results[0].dipoles, rtol=3e-14, atol=1e-12)
    systems[-1].solve(2*electric)
    assert systems[-1].factorization_count == 1


@pytest.mark.parametrize("q", [[0., 0.], [.003, -.002]])
def test_splitting_eta_invariance(q):
    eta = np.sqrt(np.pi/(100*120))
    operators = [PeriodicGreenOperator(POSITIONS, [100., 120.], .017, ewald_eta_nm_inv=eta*factor,
                                       bloch_nm_inv=q, tolerance=1e-12) for factor in (.7, 1., 1.4)]
    for operator in operators:
        np.testing.assert_allclose(operator.matrix, operators[1].matrix, rtol=5e-12, atol=5e-20)
        assert operator.metadata["convergence"]["scaled_change"] <= 1e-12


def test_cutoff_refinement_and_insufficient_cutoff_detection():
    operator = PeriodicGreenOperator(POSITIONS, [100., 120.], .017)
    finer = PeriodicGreenOperator(POSITIONS, [100., 120.], .017, cutoff=operator.metadata["ewald_cutoff"]+2)
    np.testing.assert_allclose(operator.matrix, finer.matrix, rtol=2e-14, atol=1e-20)
    with pytest.raises(EwaldConvergenceError, match="cutoff refinement"):
        PeriodicGreenOperator(POSITIONS, [100., 120.], .017, cutoff=0)


def test_bloch_reversal_is_transpose_not_conjugate():
    q = np.array([.004, .002])
    plus = PeriodicGreenOperator(POSITIONS, [100., 120.], .017, bloch_nm_inv=q)
    minus = PeriodicGreenOperator(POSITIONS, [100., 120.], .017, bloch_nm_inv=-q)
    np.testing.assert_allclose(plus.matrix, minus.matrix.T, rtol=2e-13, atol=2e-20)
    assert np.max(np.abs(plus.matrix-minus.matrix.conj().T)) > 1e-7


def test_analytic_short_hessian_against_independent_finite_difference():
    # Differentiation is used only by this oracle, never in production Ewald evaluation.
    position = np.array([27., 19., 0.])
    k, eta, step = .014, .018, .01
    def scalar(r):
        radius = np.linalg.norm(r)
        return np.real(np.exp(1j*k*radius)*erfc(eta*radius+1j*k/(2*eta)))/(4*np.pi*radius)
    value = scalar(position)
    hessian = np.empty((3, 3))
    for a in range(3):
        ea = np.eye(3)[a]*step
        for b in range(3):
            eb = np.eye(3)[b]*step
            if a == b:
                hessian[a, b] = (-scalar(position+2*ea)+16*scalar(position+ea)-30*value+16*scalar(position-ea)-scalar(position-2*ea))/(12*step**2)
            else:
                hessian[a, b] = (scalar(position+ea+eb)-scalar(position+ea-eb)-scalar(position-ea+eb)+scalar(position-ea-eb))/(4*step**2)
    np.testing.assert_allclose(_short_tensor(position, k, eta), hessian+k*k*value*np.eye(3), rtol=2e-6, atol=2e-12)


def test_large_cell_eta_avoids_exponential_overflow():
    operator = PeriodicGreenOperator([[0., 0., 0.]], [1700., 1900.], .013)
    assert .013/(2*operator.ewald_eta_nm_inv) <= 2
    assert np.isfinite(operator.matrix).all()


@pytest.mark.parametrize("periods", [[500., 700.], [500*(1+1e-10), 700.]])
def test_rayleigh_threshold_rejected_on_both_sides(periods):
    with pytest.raises(ValueError, match="Rayleigh"):
        PeriodicGreenOperator([[0., 0., 0.]], periods, 2*np.pi/500)


@pytest.mark.parametrize("kwargs", [
    {"positions_nm": []}, {"positions_nm": [[0, 0, 0], [0, 0, 0]]},
    {"positions_nm": [[0, 0, 0], [20, 20, 1]]}, {"positions_nm": [[100, 0, 0]]},
    {"positions_nm": [[-1, 0, 0]]}, {"positions_nm": [[np.nan, 0, 0]]},
    {"positions_nm": [[True, 0, 0]]}, {"positions_nm": [["0", "0", "0"]]},
    {"periods_nm": [0, 100]}, {"periods_nm": [True, 100]}, {"periods_nm": ["100", "100"]},
    {"periods_nm": [1e-200, 1e-200]}, {"bloch_nm_inv": [np.nan, 0]},
    {"bloch_nm_inv": [True, 0]}, {"k_nm_inv": "0.01"}, {"k_nm_inv": True},
    {"tolerance": "1e-10"}, {"tolerance": 1e-16}, {"cutoff": 1.5}, {"cutoff": 300},
    {"ewald_eta_nm_inv": 1e-8}, {"backend": "finite_array"},
])
def test_invalid_kernel_inputs(kwargs):
    values = {"positions_nm": [[0., 0., 0.]], "periods_nm": [100., 120.], "k_nm_inv": .013}
    values.update(kwargs)
    with pytest.raises(ValueError):
        PeriodicGreenOperator(**values)


def test_memory_limit_rejects_before_pair_matrix_allocation():
    with pytest.raises(MemoryError):
        PeriodicGreenOperator(POSITIONS, [100., 120.], .017, max_memory_bytes=100)


def test_geometry_arrays_are_owned_and_immutable():
    positions = np.asarray(POSITIONS)
    periods = np.array([100., 120.])
    operator = PeriodicGreenOperator(positions, periods, .017)
    positions[:] = 0
    periods[:] = 0
    np.testing.assert_array_equal(operator.positions_nm, POSITIONS)
    np.testing.assert_array_equal(operator.periods_nm, [100., 120.])
    assert not operator.matrix.flags.writeable
