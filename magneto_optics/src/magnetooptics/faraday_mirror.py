"""Exact normal-incidence homogeneous gyrotropic slab and isotropic mirror.

This is a one-dimensional Maxwell boundary solution, separate from the periodic
particle-sheet solver. Both sides have the same positive real host index; mu=1.
Time exp(-i omega t), lengths nm, fixed laboratory x,y on BOTH propagation legs.
Magnetization stays fixed in the laboratory during the return pass.
"""
from __future__ import annotations

import numpy as np

from .materials import gyrotropic_permittivity


def _real_scalar(value, name, *, positive=False, nonnegative=False):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f'{name} must be a real number')
    value = float(value)
    if not np.isfinite(value) or (positive and value <= 0) or (nonnegative and value < 0):
        raise ValueError(f'invalid {name}')
    return value


def _complex_scalar(value, name):
    if isinstance(value, (bool, np.bool_, str)) or np.asarray(value).ndim != 0:
        raise ValueError(f'{name} must be a finite complex scalar')
    value = complex(value)
    if not np.isfinite(value):
        raise ValueError(f'{name} must be finite')
    return value


def circular_jones(eigenvalues):
    """Transform fixed-lab e+ and e- eigenvalues to Cartesian Jones coordinates."""
    values = np.asarray(eigenvalues, dtype=complex)
    if values.shape != (2,) or not np.isfinite(values).all():
        raise ValueError('eigenvalues must contain two finite complex numbers')
    # Explicit formula preserves exactly diagonal/zero matrices when channels coincide.
    mean, difference = (values[0] + values[1]) / 2, (values[0] - values[1]) / 2
    return np.array([[mean, -1j * difference], [1j * difference, mean]])


def design_transparent_rotator(angle_deg=45., wavelength_nm=600., host_index=1.):
    """Construct a passive SYNTHETIC slab with exact target rotation at one lambda.

    One circular channel is index matched; the other is at a half-wave
    Fabry-Perot transmission resonance. Neither ideal AR coatings nor imposed
    Jones rotations are used. Real materials need independent constitutive data.
    Larger target angles approaching 90 degrees require unbounded synthetic index.
    """
    angle = _real_scalar(angle_deg, 'angle_deg')
    wavelength = _real_scalar(wavelength_nm, 'wavelength_nm', positive=True)
    host = _real_scalar(host_index, 'host_index', positive=True)
    if abs(angle) >= 90:
        raise ValueError('this construction requires abs(angle_deg) < 90')
    large_index = host / (1 - abs(angle) / 90.)
    diagonal = (host**2 + large_index**2) / 2
    coupling = np.sign(angle) * (large_index**2 - host**2) / 2
    thickness = wavelength / (2 * large_index)
    if not np.isfinite([diagonal, coupling, thickness]).all() or thickness <= 0:
        raise ValueError('synthetic design overflows; reduce the requested angle or index')
    return {'epsilon_diagonal': float(diagonal), 'g': float(coupling),
            'thickness_nm': float(thickness), 'wavelength_nm': wavelength,
            'host_index': host, 'design_angle_deg': angle,
            'provenance': 'Synthetic lossless gyrotropic slab; one matched circular channel and one half-wave Fabry-Perot resonance. Not measured material data.'}


def slab_scattering(epsilon_diagonal, g, thickness_nm, wavelength_nm, host_index=1., magnetization=1.):
    """Return exact circular and Cartesian slab amplitudes including interfaces.

    The slab occupies 0<z<d. Incident/reflected amplitudes are referenced to z=0;
    transmitted amplitudes to z=d. Same-host field amplitudes are power normalized
    by their modulus squared. The backward S blocks equal the forward ones in
    fixed lab x,y for this uniform gyrotropic material, with M unchanged.
    """
    diagonal = _complex_scalar(epsilon_diagonal, 'epsilon_diagonal')
    coupling = _complex_scalar(g, 'g')
    thickness = _real_scalar(thickness_nm, 'thickness_nm', nonnegative=True)
    wavelength = _real_scalar(wavelength_nm, 'wavelength_nm', positive=True)
    host = _real_scalar(host_index, 'host_index', positive=True)
    magnetization = _real_scalar(magnetization, 'magnetization')
    gyrotropic_permittivity(diagonal, coupling, [0, 0, magnetization])
    epsilon_channels = np.array([diagonal - coupling * magnetization, diagonal + coupling * magnetization])
    indices = np.sqrt(epsilon_channels.astype(complex))
    # Positive loss chooses Re(n)>=0, Im(n)>=0 for the passive outgoing branch.
    if np.any(indices.imag < -1e-12) or np.any(abs(indices) < 1e-12):
        raise ValueError('zero-index or nonpassive circular channel is outside this slab implementation')
    k0 = 2 * np.pi / wavelength
    phase = np.exp(1j * k0 * indices * thickness)
    interface = (host - indices) / (host + indices)
    denominator = 1 - interface**2 * phase**2
    if np.any(abs(denominator) <= 64 * np.finfo(float).eps):
        raise ValueError('slab resonance is singular to working precision')
    reflection = interface * (1 - phase**2) / denominator
    transmission = (1 - interface**2) * phase / denominator
    values = np.concatenate([reflection, transmission, phase])
    if not np.isfinite(values).all():
        raise ValueError('slab amplitude overflow; check input scales')
    return {'indices': indices, 'epsilon_channels': epsilon_channels,
            'transmission_eigenvalues': transmission, 'reflection_eigenvalues': reflection,
            'propagation_eigenvalues': phase, 'transmission': circular_jones(transmission),
            'reflection': circular_jones(reflection), 'propagation': circular_jones(phase),
            'k_host_nm_inv': float(k0 * host), 'thickness_nm': thickness,
            'wavelength_nm': wavelength, 'host_index': host, 'magnetization': magnetization,
            'phase_rotation_deg': float(np.degrees(k0 * thickness * (indices[1].real - indices[0].real) / 2)),
            'basis': 'fixed laboratory x,y for both directions, e+=(x+i y)/sqrt(2)',
            'model': 'exact homogeneous gyrotropic slab, normal incidence, nonmagnetic mu=1, identical lossless hosts'}


def mirror_return(slab, gap_nm=0., mirror_amplitude=-1.):
    """Coherent return of slab plus scalar mirror, including ALL cavity returns.

    Mirror at z=d+gap has local tangential-electric reflection amplitude rm.
    The return load at z=d is D=rm exp(2 i kh gap). An ideal PEC has rm=-1.
    Jret=R + T D (I-R D)^-1 T. First return T D T excludes prompt reflection
    and later cavity returns; it equals the total only when those are absent.
    The scalar mirror does not itself rotate the lab-frame polarization ellipse.
    """
    gap = _real_scalar(gap_nm, 'gap_nm', nonnegative=True)
    mirror = _complex_scalar(mirror_amplitude, 'mirror_amplitude')
    if abs(mirror) > 1:
        raise ValueError('mirror_amplitude must be passive: abs(rm) <= 1')
    k = _real_scalar(slab['k_host_nm_inv'], 'k_host_nm_inv', positive=True)
    r = np.asarray(slab['reflection_eigenvalues'], dtype=complex)
    t = np.asarray(slab['transmission_eigenvalues'], dtype=complex)
    if r.shape != (2,) or t.shape != (2,) or not np.isfinite(r).all() or not np.isfinite(t).all():
        raise ValueError('slab must contain two finite reflection and transmission eigenvalues')
    load = mirror * np.exp(2j * k * gap)
    denominator = 1 - r * load
    if np.any(abs(denominator) <= 64 * np.finfo(float).eps):
        raise ValueError('mirror-loaded slab cavity is singular to working precision')
    forward_gap = t / denominator
    cavity = t * load * forward_gap
    returned = r + cavity
    if not np.isfinite(returned).all():
        raise ValueError('mirror-loaded return contains nonfinite amplitudes')
    forward = circular_jones(forward_gap)
    mirror_loss = (1 - abs(mirror)**2) * forward.conj().T @ forward
    total = circular_jones(returned)
    # The host gap is lossless. Remaining missing flux is material heating.
    material_loss = np.eye(2) - total.conj().T @ total - mirror_loss
    return {'returned_jones': total, 'returned_eigenvalues': returned,
            'gap_forward_jones': forward, 'gap_return_jones': load * forward,
            'front_reflection_jones': circular_jones(r),
            'first_return_jones': circular_jones(t**2 * load),
            'cavity_return_jones': circular_jones(cavity),
            'mirror_loss_operator': mirror_loss, 'material_loss_operator': material_loss,
            'mirror_load': load, 'gap_nm': gap, 'mirror_amplitude': mirror,
            'energy_note': 'Material-loss operator is obtained by flux difference here; tests independently integrate volume Joule heating. Mirror loss excludes a transmission port, i.e. rm models an opaque absorbing termination.'}
