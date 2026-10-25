"""Passive dielectric tensors and electric sphere polarizabilities.

The time convention is exp(-i omega t).  Lengths are nm and polarizabilities
are volumes in nm**3: p = epsilon_0 * epsilon_host * alpha * E_exciting.
These are particle dipoles, not a converged volume discretization of a sphere.
"""

from __future__ import annotations

import numpy as np


def _positive_real(value, name):
    array = np.asarray(value)
    if array.ndim != 0 or np.iscomplexobj(array):
        raise ValueError(f"{name} must be a positive real scalar")
    result = float(array)
    if not np.isfinite(result) or result <= 0:
        raise ValueError(f"{name} must be a positive real scalar")
    return result


def validate_permittivity(epsilon, *, rtol=1e-12):
    """Return a finite passive tensor, accepting a scalar or (..., 3, 3).

    Passivity means that the Hermitian loss matrix (epsilon-epsilon.H)/(2j)
    is positive semidefinite.  Elementwise imaginary parts are insufficient
    for magneto-optical tensors.  A relative tolerance allows roundoff only.
    """
    if not np.isfinite(rtol) or rtol < 0:
        raise ValueError("rtol must be finite and nonnegative")
    tensor = np.asarray(epsilon, dtype=np.complex128)
    if not np.all(np.isfinite(tensor)):
        raise ValueError("epsilon must contain finite values")
    if tensor.ndim == 0:
        tensor = tensor * np.eye(3, dtype=np.complex128)
    if tensor.ndim < 2 or tensor.shape[-2:] != (3, 3):
        raise ValueError("epsilon must be scalar or have shape (..., 3, 3)")
    loss = (tensor - tensor.conj().swapaxes(-1, -2)) / (2j)
    minimum = np.linalg.eigvalsh(loss)[..., 0]
    scale = np.maximum(1.0, np.linalg.norm(tensor, axis=(-2, -1)))
    if np.any(minimum < -rtol * scale):
        raise ValueError("epsilon is active: Hermitian loss matrix must be positive semidefinite")
    return tensor.copy()


def gyrotropic_permittivity(epsilon_diagonal, g, magnetization=(0, 0, 1)):
    """Construct epsilon_d I - i g C(m), where C(m) v = m cross v.

    ``magnetization`` is a real dimensionless vector with magnitude at most
    one; its magnitude is retained, and zero gives no magneto-optical term.
    For m=+z, epsilon_xy=+i g.  The explicitly defined vectors
    e_+=(x+i y)/sqrt(2) and e_-=(x-i y)/sqrt(2) see epsilon_d-g and
    epsilon_d+g, respectively.  No material-specific data are implied.
    """
    diagonal = np.asarray(epsilon_diagonal, dtype=np.complex128)
    coupling = np.asarray(g, dtype=np.complex128)
    if diagonal.ndim != 0 or coupling.ndim != 0:
        raise ValueError("epsilon_diagonal and g must be scalar")
    raw_m = np.asarray(magnetization)
    if np.iscomplexobj(raw_m):
        raise ValueError("magnetization must be real")
    m = np.asarray(raw_m, dtype=float)
    if m.shape != (3,) or not np.all(np.isfinite(m)):
        raise ValueError("magnetization must be a finite three-vector")
    if np.linalg.norm(m) > 1 + 1e-12:
        raise ValueError("magnetization magnitude must not exceed one")
    mx, my, mz = m
    cross = np.array([[0, -mz, my], [mz, 0, -mx], [-my, mx, 0]])
    return validate_permittivity(diagonal * np.eye(3) - 1j * coupling * cross)


def sphere_polarizability(epsilon, radius_nm, wavelength_nm, host_index=1):
    """Radiation-corrected quasistatic sphere alpha, in nm**3.

    ``epsilon`` is an absolute relative dielectric tensor (not divided by
    the host permittivity).  Scalar epsilon and batches (...,3,3) are allowed;
    positive radii broadcast against the tensor's leading dimensions.
    Wavelength is the vacuum wavelength.  The host is real and lossless.

    With V=4*pi*a**3/3, chi=epsilon-epsilon_h I and c=k**3/(6*pi),
    alpha_0=3V chi (epsilon+2epsilon_h I)^-1 and
    alpha=(I-i c alpha_0)^-1 alpha_0.  The equivalent combined solve below
    remains finite at zero contrast and at the electrostatic resonance.
    """
    tensor = validate_permittivity(epsilon)
    wavelength = _positive_real(wavelength_nm, "wavelength_nm")
    index = _positive_real(host_index, "host_index")
    raw_radius = np.asarray(radius_nm)
    if np.iscomplexobj(raw_radius):
        raise ValueError("radius_nm must be positive and real")
    radius = np.asarray(raw_radius, dtype=float)
    if not np.all(np.isfinite(radius)) or np.any(radius <= 0):
        raise ValueError("radius_nm must be positive and finite")
    try:
        shape = np.broadcast_shapes(tensor.shape[:-2], radius.shape)
    except ValueError as error:
        raise ValueError("radius_nm cannot broadcast with epsilon sites") from error
    tensor = np.broadcast_to(tensor, shape + (3, 3))
    radius = np.broadcast_to(radius, shape)
    epsilon_h = index**2
    k = 2 * np.pi * index / wavelength
    volume_three = (4 * np.pi * radius**3)[..., None, None]
    contrast = tensor - epsilon_h * np.eye(3)
    numerator = volume_three * contrast
    denominator = tensor + 2 * epsilon_h * np.eye(3) - 1j * (k**3 / (6 * np.pi)) * numerator
    alpha = np.linalg.solve(denominator, numerator)
    if not np.all(np.isfinite(alpha)):
        raise ValueError("polarizability overflow: check length and material scales")
    return alpha
