"""Plane waves and observables for normalized electric particle dipoles.

Time dependence is exp(-i omega t).  Fields are divided by incident E0 and
dipoles are u=p/(epsilon_0 epsilon_host E0), in nm**3.  Cross sections are nm**2.
All formulas assume a homogeneous real, lossless host.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .materials import _positive_real


def _real_vectors(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    array = np.asarray(raw, dtype=float)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (N, 3)")
    return array


def _complex_vectors(value, name):
    array = np.asarray(value, dtype=np.complex128)
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite with shape (N, 3)")
    return array


def _unit_vector(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real")
    vector = np.asarray(raw, dtype=float)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-vector")
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError(f"{name} must be nonzero")
    return vector / norm


def circular_polarizations(direction, reference_axis=None):
    """Return explicit e_plus=(e1+i e2)/sqrt(2), e_minus=(e1-i e2)/sqrt(2).

    e1 is the transverse projection of reference_axis, e2=direction cross e1.
    The default uses x, or y when x is nearly parallel to direction.  For
    propagation along +z the default is (x +/- i y)/sqrt(2).  Labels '+' and
    '-' avoid observer-dependent naming of left/right circular polarization.
    """
    direction = _unit_vector(direction, "direction")
    if reference_axis is None:
        reference_axis = [1, 0, 0] if abs(direction[0]) < 0.9 else [0, 1, 0]
    reference = _unit_vector(reference_axis, "reference_axis")
    e1 = reference - np.dot(reference, direction) * direction
    norm = np.linalg.norm(e1)
    if norm < 1e-12:
        raise ValueError("reference_axis must not be parallel to direction")
    e1 /= norm
    e2 = np.cross(direction, e1)
    return (e1 + 1j * e2) / np.sqrt(2), (e1 - 1j * e2) / np.sqrt(2)


def plane_wave(positions_nm, wavelength_nm, direction, polarization, host_index=1):
    """Return unit-amplitude transverse E(r)=e exp(i k direction dot r).

    Direction and polarization are normalized.  A longitudinal polarization
    is rejected rather than silently projected to a different illumination.
    """
    positions = _real_vectors(positions_nm, "positions_nm")
    wavelength = _positive_real(wavelength_nm, "wavelength_nm")
    index = _positive_real(host_index, "host_index")
    direction = _unit_vector(direction, "direction")
    polarization = np.asarray(polarization, dtype=np.complex128)
    if polarization.shape != (3,) or not np.all(np.isfinite(polarization)):
        raise ValueError("polarization must be a finite complex three-vector")
    norm = np.linalg.norm(polarization)
    if norm == 0:
        raise ValueError("polarization must be nonzero")
    polarization = polarization / norm
    if abs(np.dot(direction, polarization)) > 1e-12:
        raise ValueError("polarization must be transverse to direction")
    k = 2 * np.pi * index / wavelength
    return np.exp(1j * k * (positions @ direction))[:, None] * polarization


def cross_sections(dipoles, incident, local_field, k_nm_inv):
    """Return extinction, material absorption, and scattering in nm**2.

    ``local_field`` is the exciting field E_inc + sum(G_ij u_j), excluding
    each dipole's self field.  Incident and local fields use E0=1.  Absorption
    subtracts the already-radiation-corrected dipole's self-radiation.  Small
    negative values are retained so accuracy or passivity failures are visible.
    Scattering here is extinction minus absorption; independent far-field
    quadrature is provided by ``integrated_scattering`` for energy validation.
    """
    dipoles = _complex_vectors(dipoles, "dipoles")
    incident = _complex_vectors(incident, "incident")
    local_field = _complex_vectors(local_field, "local_field")
    if incident.shape != dipoles.shape or local_field.shape != dipoles.shape:
        raise ValueError("dipoles, incident and local_field must have identical shapes")
    k = _positive_real(k_nm_inv, "k_nm_inv")
    extinction = k * np.imag(np.vdot(incident, dipoles))
    absorption = k * (np.imag(np.vdot(local_field, dipoles)) - k**3 * np.vdot(dipoles, dipoles).real / (6 * np.pi))
    return {
        "extinction_nm2": float(extinction),
        "absorption_nm2": float(absorption),
        "scattering_nm2": float(extinction - absorption),
    }


def far_field(dipoles, positions_nm, k_nm_inv, directions, *, chunk_size=256, max_pair_elements=1048576):
    """Return scattered vector amplitude f(n), in nm, with dC/dOmega=|f|**2.

    E_sc/E0 = exp(i k r) f(n)/r in the far zone.  Directions are normalized.
    Particle and direction chunks bound the intermediate phase matrix, so
    memory does not grow as the product of every angle and every particle.
    This is scattered light only, not a finite sample transmission coefficient.
    """
    dipoles = _complex_vectors(dipoles, "dipoles")
    positions = _real_vectors(positions_nm, "positions_nm")
    directions = _real_vectors(directions, "directions")
    if positions.shape != dipoles.shape:
        raise ValueError("positions_nm and dipoles must have identical shapes")
    k = _positive_real(k_nm_inv, "k_nm_inv")
    if isinstance(chunk_size, bool) or int(chunk_size) != chunk_size or chunk_size < 1:
        raise ValueError("chunk_size must be a positive integer")
    if isinstance(max_pair_elements, bool) or int(max_pair_elements) != max_pair_elements or max_pair_elements < 1:
        raise ValueError("max_pair_elements must be a positive integer")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms == 0):
        raise ValueError("directions must be nonzero")
    directions = directions / norms[:, None]
    output = np.zeros((len(directions), 3), dtype=np.complex128)
    direction_chunk = min(int(chunk_size), int(max_pair_elements))
    for first in range(0, len(directions), direction_chunk):
        n = directions[first : first + direction_chunk]
        summed = np.zeros((len(n), 3), dtype=np.complex128)
        particle_chunk = max(1, int(max_pair_elements) // len(n))
        for start in range(0, len(dipoles), particle_chunk):
            phases = np.exp(-1j * k * (n @ positions[start : start + particle_chunk].T))
            summed += phases @ dipoles[start : start + particle_chunk]
        transverse = summed - n * np.sum(n * summed, axis=1)[:, None]
        output[first : first + len(n)] = (k**2 / (4 * np.pi)) * transverse
    return output


def integrated_scattering(dipoles, positions_nm, k_nm_inv, *, n_theta=32, n_phi=64, chunk_size=256):
    """Independently integrate |f|**2 on Gauss-Legendre cos(theta) x phi.

    Increase both orders to verify angular convergence, especially when the
    array spans many wavelengths.  This is a numerical angular integral and
    is not guaranteed accurate merely because the linear solve converged.
    """
    for value, name in ((n_theta, "n_theta"), (n_phi, "n_phi")):
        if isinstance(value, bool) or int(value) != value or value < 2:
            raise ValueError(f"{name} must be an integer of at least two")
    n_theta, n_phi = int(n_theta), int(n_phi)
    mu, weights = leggauss(int(n_theta))
    phi = np.arange(int(n_phi)) * (2 * np.pi / n_phi)
    sine = np.sqrt(1 - mu**2)
    directions = np.stack(
        [sine[:, None] * np.cos(phi), sine[:, None] * np.sin(phi), np.broadcast_to(mu[:, None], (n_theta, n_phi))], axis=-1
    ).reshape(-1, 3)
    field = far_field(dipoles, positions_nm, k_nm_inv, directions, chunk_size=chunk_size)
    intensity = np.sum(np.abs(field)**2, axis=1).reshape(n_theta, n_phi)
    return float((2 * np.pi / n_phi) * np.sum(weights[:, None] * intensity))


def stokes_parameters(jones):
    """Return S0,S1,S2,S3 for a declared orthonormal transverse Jones basis.

    S3=2 Im(E1* E2), hence the explicit '+' circular channel has positive S3.
    The caller supplies two components, not an unlabelled Cartesian vector.
    """
    field = np.asarray(jones, dtype=np.complex128)
    if field.ndim < 1 or field.shape[-1] != 2 or not np.all(np.isfinite(field)):
        raise ValueError("jones must be finite with shape (..., 2)")
    a, b = field[..., 0], field[..., 1]
    cross = a.conj() * b
    return {
        "S0": np.abs(a)**2 + np.abs(b)**2,
        "S1": np.abs(a)**2 - np.abs(b)**2,
        "S2": 2 * cross.real,
        "S3": 2 * cross.imag,
    }
