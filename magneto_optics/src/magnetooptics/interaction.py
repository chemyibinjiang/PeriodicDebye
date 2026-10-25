"""Retarded electric dipole interactions in a homogeneous, lossless host.

Convention: exp(-i omega t), distances in nm, p_tilde = p/(epsilon_0 epsilon_b).
The outgoing Green tensor maps p_tilde to electric field and has units nm^-3.
Self interaction is zero: radiation reaction must already be in alpha.
"""
from __future__ import annotations

from time import perf_counter
import numpy as np
from scipy import fft


def _positive_real(value, name):
    if isinstance(value, (bool, np.bool_)) or not np.isscalar(value) or not np.isrealobj(value):
        raise ValueError(f"{name} must be a finite positive real number.")
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive real number.")
    return value


def _memory_limit(value):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError("max_memory_bytes must be a positive integer.")
    return int(value)


def _check_memory(required, limit, operation):
    if required > limit:
        raise MemoryError(f"{operation} requires an estimated {required:,} array/workspace bytes; "
                          f"max_memory_bytes is {limit:,}. Reduce the geometry or raise the explicit limit.")


def _vectors(values, n, name="dipoles"):
    values = np.asarray(values, dtype=np.complex128)
    if values.shape != (n, 3) or not np.isfinite(values).all():
        raise ValueError(f"{name} must be finite with shape ({n}, 3).")
    return values


def green_tensor(displacements_nm, k_nm_inv):
    """Return (..., 3, 3) dyadics; exactly zero displacement has zero self term."""
    k = _positive_real(k_nm_inv, "k_nm_inv")
    if np.iscomplexobj(displacements_nm):
        raise ValueError("displacements_nm must be real coordinates.")
    r = np.asarray(displacements_nm, dtype=float)
    if r.ndim < 1 or r.shape[-1] != 3 or not np.isfinite(r).all():
        raise ValueError("displacements_nm must be finite with last dimension 3.")
    radius = np.linalg.norm(r, axis=-1)
    nonzero = radius > 0
    safe = np.where(nonzero, radius, 1.)
    direction = r / safe[..., None]
    nn = direction[..., :, None] * direction[..., None, :]
    kr = k * safe
    result = np.exp(1j * kr)[..., None, None] / (4 * np.pi * safe**3)[..., None, None] * (
        kr[..., None, None]**2 * (np.eye(3) - nn)
        + np.asarray(1 - 1j * kr)[..., None, None] * (3 * nn - np.eye(3)))
    result = np.where(nonzero[..., None, None], result, 0.)
    if not np.isfinite(result).all():
        raise ValueError("Green tensor overflow; check physical length and wavenumber scales.")
    return result


class DenseGreenOperator:
    """Independent all-pairs reference; supports arbitrary distinct positions."""

    def __init__(self, positions_nm, k_nm_inv, max_memory_bytes=512 * 1024**2):
        started = perf_counter()
        if np.iscomplexobj(positions_nm):
            raise ValueError("positions_nm must be real coordinates.")
        positions = np.asarray(positions_nm, dtype=float)
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions) or not np.isfinite(positions).all():
            raise ValueError("positions_nm must be a nonempty finite (N, 3) array.")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("Distinct dipoles cannot occupy duplicate positions.")
        self.k_nm_inv = _positive_real(k_nm_inv, "k_nm_inv")
        self.max_memory_bytes = _memory_limit(max_memory_bytes)
        self.n_dipoles = len(positions)
        n = self.n_dipoles
        matrix_bytes = 144 * n * n
        # Build bounded row blocks; tensor construction holds several real/complex temporaries.
        row_chunk = min(n, 32)
        estimated = matrix_bytes + row_chunk * n * 1024 + n * 96
        _check_memory(estimated, self.max_memory_bytes, "Dense Green construction")
        self.positions_nm = np.array(positions, copy=True)
        self.positions_nm.setflags(write=False)
        self.matrix = np.empty((3 * n, 3 * n), dtype=np.complex128)
        for start in range(0, n, row_chunk):
            stop = min(start + row_chunk, n)
            dyadic = green_tensor(self.positions_nm[start:stop, None] - self.positions_nm[None], self.k_nm_inv)
            self.matrix[3 * start:3 * stop] = dyadic.transpose(0, 2, 1, 3).reshape(3 * (stop - start), 3 * n)
        self.matrix.setflags(write=False)
        self.metadata = {
            "backend": "dense_all_pairs", "n_dipoles": n, "matrix_bytes": matrix_bytes,
            "estimated_peak_array_bytes": estimated, "max_memory_bytes": self.max_memory_bytes,
            "memory_scope": "owned arrays and conservative workspace estimate; excludes Python/BLAS process overhead",
            "boundary": "finite_open", "self_term": "zero; radiation reaction belongs in alpha",
            "setup_seconds": perf_counter() - started,
        }

    def apply(self, dipoles):
        values = _vectors(dipoles, self.n_dipoles)
        return (self.matrix @ values.reshape(-1)).reshape(-1, 3)


class FFTGreenOperator:
    """Exact linear convolution on occupied orthogonal-grid sites.

    Empty sites remain zero. Negative indices are translated by their minimum;
    no geometric rounding, periodic boundary, cutoff, or interpolation is used.
    The six symmetric Green tensor components are transformed once and reused.
    """

    def __init__(self, indices, spacing_nm, k_nm_inv, shape=None,
                 max_memory_bytes=512 * 1024**2):
        started = perf_counter()
        raw = np.asarray(indices)
        if raw.ndim != 2 or raw.shape[1:] != (3,) or not len(raw) or raw.dtype.kind not in "iuf":
            raise ValueError("indices must be a nonempty finite integer (N, 3) array.")
        if not np.isfinite(raw).all() or np.any(raw != np.floor(raw)) or np.max(np.abs(raw)) > 2**50:
            raise ValueError("indices must contain exact, safely representable integers.")
        integer = raw.astype(np.int64)
        if len(np.unique(integer, axis=0)) != len(integer):
            raise ValueError("Duplicate occupied grid indices are not allowed.")
        origin = integer.min(axis=0)
        integer = integer - origin
        span = integer.max(axis=0) + 1
        if shape is None:
            grid_shape = tuple(map(int, span))
        else:
            raw_shape = np.asarray(shape)
            if raw_shape.shape != (3,) or raw_shape.dtype.kind not in "iu" or np.any(raw_shape <= 0):
                raise ValueError("shape must contain three positive integers.")
            if np.any(raw_shape < span):
                raise ValueError("shape must contain all indices after translation by their minimum.")
            grid_shape = tuple(map(int, raw_shape))
        if np.iscomplexobj(spacing_nm) or np.asarray(spacing_nm).dtype.kind == "b":
            raise ValueError("spacing_nm must be positive real lengths.")
        spacing = np.asarray(spacing_nm, dtype=float)
        if spacing.ndim == 0:
            spacing = np.repeat(spacing, 3)
        if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
            raise ValueError("spacing_nm must be a positive scalar or length-three vector.")
        self.k_nm_inv = _positive_real(k_nm_inv, "k_nm_inv")
        self.max_memory_bytes = _memory_limit(max_memory_bytes)
        self.n_dipoles = len(integer)
        self.shape = grid_shape
        self.padded_shape = tuple(fft.next_fast_len(2 * n - 1) for n in grid_shape)
        padded_count = int(np.prod(self.padded_shape, dtype=object))
        # Six retained kernels + sources/outputs/FFT workspaces; kernel assembly is scalar-component streamed.
        estimated = 16 * 24 * padded_count + 128 * self.n_dipoles
        _check_memory(estimated, self.max_memory_bytes, "FFT Green construction/application")
        self.indices = integer
        self.indices.setflags(write=False)
        self.origin_index = origin
        self.origin_index.setflags(write=False)
        self.spacing_nm = np.array(spacing, copy=True)
        self.spacing_nm.setflags(write=False)
        self.positions_nm = self.indices * self.spacing_nm
        self.positions_nm.setflags(write=False)
        self._site_tuple = tuple(self.indices.T)
        self._components = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
        self._kernels = []
        axes = []
        for n, length, step in zip(grid_shape, self.padded_shape, self.spacing_nm):
            axis = np.arange(length, dtype=float)
            # Valid positive and negative displacements are embedded at displacement modulo L.
            axis[axis >= n] -= length
            axes.append(axis * step)
        x, y, z = np.meshgrid(*axes, indexing="ij", sparse=True)
        radius = np.sqrt(x*x + y*y + z*z)
        safe = np.where(radius == 0, 1., radius)
        directions = (x / safe, y / safe, z / safe)
        kr = self.k_nm_inv * safe
        prefactor = np.exp(1j * kr) / (4 * np.pi * safe**3)
        isotropic = kr**2 - (1 - 1j * kr)
        longitudinal = 3 * (1 - 1j * kr) - kr**2
        for a, b in self._components:
            kernel = prefactor * (longitudinal * directions[a] * directions[b] + (isotropic if a == b else 0))
            kernel[0, 0, 0] = 0.
            if not np.isfinite(kernel).all():
                raise ValueError("Green tensor overflow; check physical length and wavenumber scales.")
            transformed = fft.fftn(kernel, overwrite_x=True)
            transformed.setflags(write=False)
            self._kernels.append(transformed)
        self.metadata = {
            "backend": "fft_linear_convolution", "n_dipoles": self.n_dipoles,
            "shape": list(self.shape), "padded_shape": list(self.padded_shape),
            "kernel_bytes": 6 * 16 * padded_count, "estimated_peak_array_bytes": estimated,
            "max_memory_bytes": self.max_memory_bytes,
            "memory_scope": "owned arrays and conservative workspace estimate; excludes Python/FFT process overhead",
            "boundary": "finite_open", "geometry_approximation": "none; exact occupied grid sites",
            "self_term": "zero; radiation reaction belongs in alpha", "setup_seconds": perf_counter() - started,
        }

    def apply(self, dipoles):
        values = _vectors(dipoles, self.n_dipoles)
        source = np.zeros((3,) + self.padded_shape, dtype=np.complex128)
        source[(slice(None),) + self._site_tuple] = values.T
        source = fft.fftn(source, axes=(1, 2, 3), overwrite_x=True)
        output = np.zeros_like(source)
        for kernel, (a, b) in zip(self._kernels, self._components):
            output[a] += kernel * source[b]
            if a != b:
                output[b] += kernel * source[a]
        output = fft.ifftn(output, axes=(1, 2, 3), overwrite_x=True)
        return output[(slice(None),) + self._site_tuple].T.copy()
