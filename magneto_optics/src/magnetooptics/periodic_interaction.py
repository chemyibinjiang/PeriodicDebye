"""Outgoing two-periodic electric Green dyadics, evaluated by Ewald splitting.

Coplanar sites in a rectangular cell, homogeneous lossless host, exp(-i omega t).
The sum is sum_R exp(i q.R) G(r_i-r_j-R), excluding only i=j,R=0.
It is not a finite-array approximation.  Both Ewald sums converge exponentially.

The scalar Ewald representation follows Lunnemann, Sersic and Koenderink,
Phys. Rev. B 88, 245109 (2013), Appendix A, with our 1/(4*pi) normalization.
Here its radial derivatives, coplanar spectral Hessian, and coincident-site
subtraction are evaluated analytically (no numerical differentiation).
https://doi.org/10.1103/PhysRevB.88.245109
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy.special import erfc, erfi

from .interaction import DenseGreenOperator, _positive_real, _memory_limit, _check_memory
from .geometry import _real_array


class EwaldConvergenceError(RuntimeError):
    """Measured cutoff refinement failed the requested Ewald tolerance."""


def _short_tensor(vectors, k, eta):
    """(k^2 I + Hessian) [Re(exp(ikr) erfc(eta*r+ik/2eta))/(4*pi*r)]."""
    radius = np.linalg.norm(vectors, axis=-1)
    if np.any(radius == 0):
        raise ValueError("Internal short-range evaluation includes an unremoved self site.")
    unit = vectors / radius[..., None]
    h = np.exp(1j*k*radius) * erfc(eta*radius + 1j*k/(2*eta))
    exponential = np.exp(-(eta*radius)**2 + (k/(2*eta))**2)
    h0 = 2*h.real
    h1 = -2*k*h.imag - 4*eta/np.sqrt(np.pi)*exponential
    h2 = -k*k*h0 + 8*eta**3*radius/np.sqrt(np.pi)*exponential
    isotropic = (k*k*h0/radius + h1/radius**2 - h0/radius**3)/(8*np.pi)
    radial = (h2/radius - 3*h1/radius**2 + 3*h0/radius**3)/(8*np.pi)
    return isotropic[..., None, None]*np.eye(3) + radial[..., None, None]*unit[..., :, None]*unit[..., None, :]


def _self_correction(k, eta):
    ratio = k/(2*eta)
    real = (k**3*erfi(ratio) + 2*eta*(eta**2-k*k)*np.exp(ratio**2)/np.sqrt(np.pi))/(6*np.pi)
    return real - 1j*k**3/(6*np.pi)


class PeriodicGreenOperator(DenseGreenOperator):
    """Small-unit-cell dense operator with a genuine infinite 2D lattice sum.

    ``cached`` groups exactly equal pair displacements (and inversion at q=0)
    and vectorizes lattice sums. ``vectorized`` omits the displacement cache.
    ``reference`` evaluates every ordered pair
    with scalar lattice loops. Both use the same analytically derived Ewald
    split; independent radiation/static identities validate that shared formula.

    ``tolerance`` controls measured change on increasing both real/reciprocal
    square cutoffs by two, scaled by max(max|G|, Lmin^-3 + k^2/Lmin). This is
    a convergence diagnostic, not a rigorous interval bound. A further eta
    invariance check is provided by the validation suite. The optional integer
    ``cutoff`` selects the starting cutoff; inadequate values raise explicitly.
    Bloch wavevectors are a kernel option; user-facing ports currently use q=0.
    """

    def __init__(self, positions_nm, periods_nm, k_nm_inv, *, tolerance=1e-10,
                 backend="cached", max_memory_bytes=512*1024**2,
                 ewald_eta_nm_inv=None, cutoff=None, bloch_nm_inv=(0., 0.)):
        started = perf_counter()
        if backend not in ("cached", "vectorized", "reference"):
            raise ValueError("backend must be 'cached', 'vectorized', or 'reference'.")
        positions = _real_array(positions_nm, "positions_nm")
        periods = _real_array(periods_nm, "periods_nm")
        q = _real_array(bloch_nm_inv, "bloch_nm_inv")
        _real_array(k_nm_inv, "k_nm_inv")
        _real_array(tolerance, "tolerance")
        if ewald_eta_nm_inv is not None:
            _real_array(ewald_eta_nm_inv, "ewald_eta_nm_inv")
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions) or not np.isfinite(positions).all():
            raise ValueError("positions_nm must be a nonempty finite (N,3) array.")
        if periods.shape != (2,) or not np.isfinite(periods).all() or np.any(periods <= 0):
            raise ValueError("periods_nm must contain two positive finite lengths.")
        if q.shape != (2,) or not np.isfinite(q).all():
            raise ValueError("bloch_nm_inv must contain two finite wavevector components.")
        if np.any(positions[:, :2] < 0) or np.any(positions[:, :2] >= periods):
            raise ValueError("In-plane positions must lie in the half-open rectangular unit cell [0,Lx)x[0,Ly).")
        if np.any(positions[:, 2] != positions[0, 2]):
            raise ValueError("This periodic kernel requires exactly coplanar particle centres.")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("Duplicate particle centres are not allowed.")
        if isinstance(tolerance, (bool, np.bool_)) or not np.isscalar(tolerance) or not np.isrealobj(tolerance):
            raise ValueError("tolerance must be a finite real number between 1e-14 and 0.01.")
        tolerance = float(tolerance)
        if not np.isfinite(tolerance) or not 1e-14 <= tolerance < .01:
            raise ValueError("tolerance must be a finite real number between 1e-14 and 0.01.")
        self.k_nm_inv = _positive_real(k_nm_inv, "k_nm_inv")
        self.max_memory_bytes = _memory_limit(max_memory_bytes)
        self.n_dipoles = len(positions)
        self.positions_nm = np.array(positions, copy=True)
        self.positions_nm.setflags(write=False)
        self.periods_nm = np.array(periods, copy=True)
        self.periods_nm.setflags(write=False)
        self.bloch_nm_inv = np.array(q, copy=True)
        self.bloch_nm_inv.setflags(write=False)
        self.area_nm2 = float(np.prod(periods))
        if not np.isfinite(self.area_nm2) or self.area_nm2 <= 0:
            raise ValueError("Unit-cell area underflow or overflow; check length scales.")
        k = self.k_nm_inv
        eta = max(np.sqrt(np.pi/self.area_nm2), k/4) if ewald_eta_nm_inv is None else _positive_real(ewald_eta_nm_inv, "ewald_eta_nm_inv")
        if k/(2*eta) > 4:
            raise ValueError("Ewald eta is too small: exponential cancellation would compromise float64 accuracy; increase eta.")
        self.ewald_eta_nm_inv = eta
        if cutoff is not None and (isinstance(cutoff, (bool, np.bool_)) or not isinstance(cutoff, (int, np.integer)) or cutoff < 0):
            raise ValueError("cutoff must be a nonnegative integer.")
        # Use both real and reciprocal Gaussian decay scales, allowing pair offsets up to one period.
        decay = np.sqrt(-np.log(tolerance)+8+(k/(2*eta))**2)
        proposed = max(4, int(np.ceil(max(decay/(eta*periods.min())+1,
                                        (2*eta*decay+np.linalg.norm(q)+k)*periods.max()/(2*np.pi)))))
        base_cutoff = proposed if cutoff is None else int(cutoff)
        if base_cutoff+2 > 256:
            raise ValueError("Ewald cutoff would exceed 256; reduce cell aspect/size or change eta. No truncation was performed.")
        n = self.n_dipoles
        lattice_count = (2*(base_cutoff+2)+1)**2
        # Includes simultaneous lower/refined blocks, advanced-index assembly,
        # final matrix, displacement/inverse caches, and bounded lattice temporaries.
        estimated = 800*n*n + 2048*lattice_count + 512*n
        _check_memory(estimated, self.max_memory_bytes, "Periodic Ewald matrix, pair cache, and lattice workspace")
        offsets = (positions[:, None, :2]-positions[None, :, :2]).reshape(-1, 2)
        if backend == "cached":
            if np.all(q == 0):
                leading = np.argmax(offsets != 0, axis=1)
                negative = offsets[np.arange(len(offsets)), leading] < 0
                offsets[negative] *= -1
            unique, inverse = np.unique(offsets, axis=0, return_inverse=True)
        else:
            unique, inverse = offsets, np.arange(n*n)
        self._backend = backend
        prior = self._evaluate_offsets(unique, base_cutoff)
        accepted_cutoff = base_cutoff+2
        refined = self._evaluate_offsets(unique, accepted_cutoff)
        change = float(np.max(np.abs(refined-prior)))
        scale = max(float(np.max(np.abs(refined))), periods.min()**-3+k*k/periods.min())
        scaled_change = change/scale
        if not np.isfinite(refined).all() or not np.isfinite(scaled_change) or scaled_change > tolerance:
            raise EwaldConvergenceError(f"Ewald cutoff refinement {base_cutoff}->{accepted_cutoff} gives "
                                        f"scaled change {scaled_change:.3g}, exceeding tolerance {tolerance:.3g}; increase cutoff or adjust eta.")
        blocks = refined[inverse].reshape(n, n, 3, 3)
        self.matrix = np.ascontiguousarray(blocks.transpose(0, 2, 1, 3).reshape(3*n, 3*n))
        self.matrix.setflags(write=False)
        self.metadata = {
            "backend": "periodic_ewald_"+backend, "n_dipoles": n,
            "boundary": "infinite_2d_periodic_open_z", "periods_nm": periods.tolist(),
            "bloch_nm_inv": q.tolist(), "area_nm2": self.area_nm2,
            "self_term": "exclude only same-particle zero image; all lattice images retained; radiation remains in alpha",
            "ewald_eta_nm_inv": eta, "ewald_cutoff": accepted_cutoff,
            "real_terms_per_pair": (2*accepted_cutoff+1)**2,
            "reciprocal_terms_per_pair": (2*accepted_cutoff+1)**2,
            "pair_evaluations": len(unique), "ordered_pair_count": n*n,
            "displacement_cache": "exact_float_equality_and_inversion_at_zero_bloch" if backend == "cached" else "none",
            "lattice_evaluation": "scalar_reference" if backend == "reference" else "vectorized",
            "convergence": {"verified": True, "tolerance": tolerance, "lower_cutoff": base_cutoff,
                            "upper_cutoff": accepted_cutoff, "max_absolute_change_nm_inv3": change,
                            "scale_nm_inv3": scale, "scaled_change": scaled_change,
                            "scope": "measured real+reciprocal cutoff refinement; not a rigorous error bound"},
            "rayleigh_grazing_tolerance": 1e-8,
            "matrix_bytes": self.matrix.nbytes, "estimated_peak_array_bytes": estimated,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_scope": "conservative ndarray/workspace estimate; excludes interpreter/native-library overhead",
            "setup_seconds": perf_counter()-started,
            "reference": "https://doi.org/10.1103/PhysRevB.88.245109",
        }

    def _lattice_data(self, cutoff):
        orders = np.indices((2*cutoff+1, 2*cutoff+1)).reshape(2, -1).T-cutoff
        real = orders*self.periods_nm
        reciprocal = orders*(2*np.pi/self.periods_nm)+self.bloch_nm_inv
        k2 = self.k_nm_inv**2
        margin = k2-np.sum(reciprocal**2, axis=1)
        grazing = np.abs(margin/k2) <= 1e-8
        if np.any(grazing):
            raise ValueError(f"Rayleigh/grazing diffraction threshold at orders {orders[grazing].tolist()}; change wavelength or periods.")
        kz = np.sqrt(margin.astype(complex))
        eta = self.ewald_eta_nm_inv
        scalar = 1j*erfc(-1j*kz/(2*eta))/(2*self.area_nm2*kz)
        spectral = np.zeros((len(orders), 3, 3), complex)
        for a in range(2):
            for b in range(2):
                spectral[:, a, b] = ((k2 if a == b else 0)-reciprocal[:, a]*reciprocal[:, b])*scalar
        spectral[:, 2, 2] = np.sum(reciprocal**2, axis=1)*scalar - eta*np.exp(margin/(4*eta**2))/(self.area_nm2*np.sqrt(np.pi))
        phases = np.exp(1j*(real@self.bloch_nm_inv))
        return real, reciprocal, spectral, phases

    def _evaluate_offsets(self, offsets, cutoff):
        real, reciprocal, spectral, phases = self._lattice_data(cutoff)
        result = np.empty((len(offsets), 3, 3), complex)
        for i, offset in enumerate(offsets):
            vectors = np.zeros((len(real), 3))
            vectors[:, :2] = offset-real
            retained = np.any(vectors != 0, axis=1)
            if self._backend != "reference":
                short = np.einsum("n,nij->ij", phases[retained], _short_tensor(vectors[retained], self.k_nm_inv, self.ewald_eta_nm_inv))
                long = np.einsum("n,nij->ij", np.exp(1j*(reciprocal@offset)), spectral)
            else:
                short = np.zeros((3, 3), complex)
                for vector, phase in zip(vectors[retained], phases[retained]):
                    short += phase*_short_tensor(vector, self.k_nm_inv, self.ewald_eta_nm_inv)
                long = np.zeros((3, 3), complex)
                for wavevector, dyadic in zip(reciprocal, spectral):
                    long += np.exp(1j*np.dot(wavevector, offset))*dyadic
            result[i] = short+long
            if not np.any(offset):
                result[i] += np.eye(3)*_self_correction(self.k_nm_inv, self.ewald_eta_nm_inv)
        return result
