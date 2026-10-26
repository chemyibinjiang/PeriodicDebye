"""Self-consistent electric dipoles with a dense oracle and FFT-ready GMRES."""
from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
import warnings
import numpy as np
from scipy.linalg import lu_factor, lu_solve, get_lapack_funcs, LinAlgWarning
from scipy.sparse.linalg import LinearOperator, gmres

from .interaction import DenseGreenOperator, _vectors, _check_memory


class ConvergenceError(RuntimeError):
    """The requested residual was not achieved; no scientific result is returned."""


@dataclass(frozen=True)
class SolveResult:
    dipoles: np.ndarray
    local_field: np.ndarray
    relative_residual: float
    iterations: int
    metadata: dict


def _tolerance(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isrealobj(value):
        raise ValueError(f"{name} must be finite and nonnegative.")
    value = float(value)
    if not np.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and nonnegative.")
    return value


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


class CoupledDipoleSystem:
    """Solve (I - alpha G) p = alpha E without inverting singular/zero alpha.

    alpha is copied and fixed for this system. Reuse the Green operator for a
    different material or magnetization; repeated dense solves reuse one LU.
    Residual means ||p-alpha(E+Gp)||/||alpha E||, not a bound on solution error.
    """

    def __init__(self, operator, alpha, method="gmres"):
        if method not in ("dense", "gmres"):
            raise ValueError("method must be 'dense' or 'gmres'.")
        if method == "dense" and not isinstance(operator, DenseGreenOperator):
            raise ValueError("The dense solver requires a DenseGreenOperator.")
        if not hasattr(operator, "apply") or not hasattr(operator, "n_dipoles"):
            raise ValueError("operator must provide apply and n_dipoles.")
        self.operator = operator
        self.n_dipoles = operator.n_dipoles
        supplied = np.asarray(alpha, dtype=np.complex128)
        if supplied.shape == (3, 3):
            supplied = np.broadcast_to(supplied, (self.n_dipoles, 3, 3))
        if supplied.shape != (self.n_dipoles, 3, 3) or not np.isfinite(supplied).all():
            raise ValueError("alpha must be a finite (3,3) or (N,3,3) array in nm^3.")
        self.alpha = np.array(supplied, copy=True)
        self.alpha.setflags(write=False)
        self.method = method
        self._lu = None
        self.reciprocal_condition_estimate = None
        self.factorization_count = 0

    def _multiply_alpha(self, value):
        return np.einsum("nij,nj->ni", self.alpha, value)

    def _matvec(self, vector):
        values = vector.reshape(self.n_dipoles, 3)
        return (values - self._multiply_alpha(self.operator.apply(values))).reshape(-1)

    def _factorize(self):
        if self._lu is not None:
            return
        n = self.n_dipoles
        required = 4 * 144 * n * n + 512 * n
        _check_memory(required, self.operator.max_memory_bytes, "Dense coupled-dipole factorization")
        matrix = np.empty((3*n, 3*n), dtype=np.complex128, order="F")
        for i in range(n):
            matrix[3*i:3*i+3] = -self.alpha[i] @ self.operator.matrix[3*i:3*i+3]
            matrix[3*i:3*i+3, 3*i:3*i+3] += np.eye(3)
        if not np.isfinite(matrix).all():
            raise ConvergenceError("Coupled-dipole matrix overflow; check material and geometry scales.")
        matrix_norm = float(np.linalg.norm(matrix, ord=1))
        if not np.isfinite(matrix_norm):
            raise ConvergenceError("Coupled-dipole matrix norm overflow; check material and geometry scales.")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error", LinAlgWarning)
                factorization = lu_factor(matrix, overwrite_a=True, check_finite=False)
        except (LinAlgWarning, ValueError) as exc:
            raise ConvergenceError("Dense coupled-dipole matrix is singular or could not be factorized.") from exc
        condition, info = get_lapack_funcs("gecon", (factorization[0],))(factorization[0], matrix_norm)
        if info != 0 or not np.isfinite(condition) or condition <= 3*n*np.finfo(float).eps:
            raise ConvergenceError("Dense coupled-dipole matrix is singular to working precision.")
        self.reciprocal_condition_estimate = float(condition)
        self._lu = factorization
        self.factorization_count += 1

    def solve(self, incident, rtol=1e-9, atol=0., restart=40, maxiter=200, initial=None):
        started = perf_counter()
        electric = _vectors(incident, self.n_dipoles, "incident")
        rtol, atol = _tolerance(rtol, "rtol"), _tolerance(atol, "atol")
        if rtol == 0 and atol == 0:
            raise ValueError("At least one of rtol and atol must be positive.")
        restart, maxiter = _integer(restart, "restart"), _integer(maxiter, "maxiter")
        guess = None if initial is None else _vectors(initial, self.n_dipoles, "initial").reshape(-1)
        rhs = self._multiply_alpha(electric).reshape(-1)
        rhs_norm = float(np.linalg.norm(rhs))
        if not np.isfinite(rhs_norm):
            raise ConvergenceError("Incident polarization norm overflow; check incident and alpha scales.")
        iterations = 0
        previous_factorizations = self.factorization_count
        if rhs_norm == 0:
            solution = np.zeros_like(rhs)
            info = 0
        elif self.method == "dense":
            self._factorize()
            solution = lu_solve(self._lu, rhs, check_finite=False)
            info = 0
        else:
            # Krylov basis can dominate FFT memory; account for it before GMRES allocates.
            dimension = 3 * self.n_dipoles
            effective_restart = min(restart, dimension)
            required = (self.operator.metadata.get("estimated_peak_array_bytes", 0)
                        + 16 * dimension * (effective_restart + 8)
                        + 16 * (effective_restart + 1)**2)
            _check_memory(required, self.operator.max_memory_bytes, "GMRES Krylov basis and operator workspace")
            def count_iteration(_):
                nonlocal iterations
                iterations += 1
            system = LinearOperator((dimension, dimension), matvec=self._matvec, dtype=np.complex128)
            solution, info = gmres(system, rhs, x0=guess, rtol=rtol, atol=atol,
                                   restart=restart, maxiter=maxiter, callback=count_iteration,
                                   callback_type="pr_norm")
        if not np.isfinite(solution).all():
            raise ConvergenceError("Coupled-dipole solution contains nonfinite values.")
        dipoles = solution.reshape(-1, 3)
        local = electric + self.operator.apply(dipoles)
        residual = dipoles - self._multiply_alpha(local)
        absolute = float(np.linalg.norm(residual))
        relative = absolute / rhs_norm if rhs_norm else absolute
        threshold = max(atol, rtol * rhs_norm)
        # A tiny roundoff allowance prevents re-evaluation order from falsely rejecting a converged solve.
        roundoff = 8 * np.finfo(float).eps * rhs_norm
        if info != 0 or not np.isfinite(absolute) or absolute > threshold + roundoff:
            raise ConvergenceError(f"Coupled-dipole solve did not converge: info={info}, iterations={iterations}, "
                                   f"relative_residual={relative:.6g}, absolute_residual={absolute:.6g}, "
                                   f"requested_absolute_tolerance={threshold:.6g}.")
        return SolveResult(dipoles=dipoles, local_field=local, relative_residual=relative,
                           iterations=iterations, metadata={
                               "method": self.method, "rtol": rtol, "atol": atol,
                               "absolute_residual": absolute, "relative_residual": relative,
                               "residual_definition": "norm(p-alpha(E_inc+Gp))/norm(alpha E_inc)",
                               "preconditioner": "equivalent left-polarizability form I-alpha G; self-image blocks retained when periodic",
                               "iterations": iterations, "maxiter_restart_cycles": maxiter,
                               "restart": restart, "factorization_reused": previous_factorizations > 0,
                               "factorization_count": self.factorization_count,
                               "reciprocal_condition_estimate": self.reciprocal_condition_estimate,
                               "solve_seconds": perf_counter() - started,
                               "operator": dict(self.operator.metadata),
                           })
