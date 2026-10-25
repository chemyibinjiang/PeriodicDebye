"""Dense factorization, iteration failure, tensor coupling, and memory checks."""
import numpy as np
import pytest

from magnetooptics.interaction import DenseGreenOperator, FFTGreenOperator, green_tensor
from magnetooptics.solver import CoupledDipoleSystem, ConvergenceError


@pytest.mark.parametrize("method", ["dense", "gmres"])
def test_one_dipole_exact_tensor_response(method):
    operator = DenseGreenOperator([[2, 4, 8]], .012)
    alpha = np.array([[17+2j, 3j, 0], [-3j, 17+2j, 0], [0, 0, 12+1j]])
    electric = np.array([[1, .3j, .2]], complex)
    result = CoupledDipoleSystem(operator, alpha, method=method).solve(electric)
    np.testing.assert_allclose(result.dipoles[0], alpha @ electric[0], rtol=1e-14)
    np.testing.assert_array_equal(result.local_field, electric)
    assert result.relative_residual < 1e-14


@pytest.mark.parametrize("polarization", [0, 1, 2])
@pytest.mark.parametrize("antisymmetric", [False, True])
def test_two_dipole_closed_eigenmodes(polarization, antisymmetric):
    distance, k, a = 21., .011, 500+30j
    electric = np.zeros((2, 3), complex)
    electric[:, polarization] = [1, -1 if antisymmetric else 1]
    operator = DenseGreenOperator([[0, 0, 0], [distance, 0, 0]], k)
    coupling = green_tensor([distance, 0, 0], k)[polarization, polarization]
    expected = a*electric/(1 + (1 if antisymmetric else -1)*a*coupling)
    result = CoupledDipoleSystem(operator, a*np.eye(3), method="dense").solve(electric)
    np.testing.assert_allclose(result.dipoles, expected, rtol=2e-15, atol=1e-14)


def test_spatially_varying_nonreciprocal_tensors_fft_gmres_match_dense():
    rng = np.random.default_rng(348)
    indices = np.indices((4, 3, 2)).reshape(3, -1).T[::2]
    n = len(indices)
    alpha = np.tile((800+50j)*np.eye(3), (n, 1, 1))
    alpha[:, 0, 1] = 70j
    alpha[:, 1, 0] = -70j
    alpha *= rng.uniform(.8, 1.2, (n, 1, 1))
    electric = rng.normal(size=(n, 3)) + 1j*rng.normal(size=(n, 3))
    dense = DenseGreenOperator(indices*18., .014)
    fft = FFTGreenOperator(indices, 18., .014)
    expected = CoupledDipoleSystem(dense, alpha, "dense").solve(electric)
    actual = CoupledDipoleSystem(fft, alpha).solve(electric, rtol=1e-12)
    np.testing.assert_allclose(actual.dipoles, expected.dipoles, rtol=2e-11, atol=3e-9)
    np.testing.assert_allclose(actual.local_field, expected.local_field, rtol=2e-12, atol=1e-12)
    assert actual.relative_residual < 1e-12
    assert actual.iterations > 0


def test_dense_factorization_is_reused_and_alpha_copied():
    operator = DenseGreenOperator([[0, 0, 0], [20, 0, 0]], .01)
    alpha = np.eye(3)*100
    system = CoupledDipoleSystem(operator, alpha, "dense")
    alpha[:] = 0
    first = system.solve(np.ones((2, 3)))
    second = system.solve(np.ones((2, 3))*2)
    np.testing.assert_allclose(second.dipoles, first.dipoles*2)
    assert system.factorization_count == 1
    assert first.metadata["factorization_reused"] is False
    assert second.metadata["factorization_reused"] is True


@pytest.mark.parametrize("method", ["dense", "gmres"])
def test_zero_alpha_and_zero_incident_are_exact(method):
    operator = DenseGreenOperator([[0, 0, 0], [20, 0, 0]], .01)
    for alpha, field in [(np.zeros((3, 3)), np.ones((2, 3))), (np.eye(3), np.zeros((2, 3)))]:
        result = CoupledDipoleSystem(operator, alpha, method).solve(field)
        np.testing.assert_array_equal(result.dipoles, np.zeros((2, 3)))
        assert result.relative_residual == 0


def test_nonconvergence_raises_instead_of_publishing_result():
    indices = np.indices((4, 3, 1)).reshape(3, -1).T
    operator = FFTGreenOperator(indices, 12., .02)
    electric = np.random.default_rng(4).normal(size=(len(indices), 3))
    system = CoupledDipoleSystem(operator, (2000+20j)*np.eye(3))
    with pytest.raises(ConvergenceError, match="did not converge"):
        system.solve(electric, rtol=1e-14, restart=1, maxiter=1)


def test_invalid_alpha_method_tolerances_and_vectors():
    operator = DenseGreenOperator([[0, 0, 0]], .01)
    for alpha in [np.zeros(3), np.full((3, 3), np.nan)]:
        with pytest.raises(ValueError):
            CoupledDipoleSystem(operator, alpha)
    with pytest.raises(ValueError):
        CoupledDipoleSystem(operator, np.eye(3), "invalid")
    with pytest.raises(ValueError):
        CoupledDipoleSystem(FFTGreenOperator([[0, 0, 0]], 10, .01), np.eye(3), "dense")
    system = CoupledDipoleSystem(operator, np.eye(3))
    for options in [{"rtol": -1}, {"atol": np.nan}, {"rtol": 0, "atol": 0},
                    {"restart": True}, {"maxiter": 0}, {"initial": [[1, 2]]}]:
        with pytest.raises(ValueError):
            system.solve([[1, 0, 0]], **options)
    with pytest.raises(ValueError):
        system.solve([[1, np.nan, 0]])


def test_memory_guards_include_lu_and_krylov():
    positions = np.arange(60).reshape(20, 3)*10.
    dense = DenseGreenOperator(positions, .01)
    dense.max_memory_bytes = 4*144*20*20 - 1
    with pytest.raises(MemoryError, match="factorization"):
        CoupledDipoleSystem(dense, np.eye(3), "dense").solve(np.ones((20, 3)))
    fft = FFTGreenOperator([[0, 0, 0], [1, 0, 0]], 10., .01)
    fft.max_memory_bytes = fft.metadata["estimated_peak_array_bytes"]
    with pytest.raises(MemoryError, match="Krylov"):
        CoupledDipoleSystem(fft, np.eye(3)).solve(np.ones((2, 3)))


def test_exact_singular_system_raises():
    # Deliberate mathematical singularity (not a passive physical material).
    operator = DenseGreenOperator([[0, 0, 0], [20, 0, 0]], .01)
    g = operator.matrix[0, 3]
    alpha = np.diag([1/g, 0, 0])
    with pytest.raises(ConvergenceError):
        CoupledDipoleSystem(operator, alpha, "dense").solve([[1, 0, 0], [1, 0, 0]])


def test_overflowing_incident_norm_fails_before_iteration():
    operator = DenseGreenOperator([[0, 0, 0]], .01)
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ConvergenceError, match="norm overflow"):
            CoupledDipoleSystem(operator, 1e200*np.eye(3)).solve([[1e200, 0, 0]])
