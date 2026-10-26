"""Exact normal-incidence scattering of planar axial magneto-optical layers.

This is a one-dimensional Maxwell solution, not an effective-medium treatment
of the periodic particle array. All layers have mu=1 and the same fixed axial
magnetization direction in laboratory coordinates. Their scalar circular
channels are cascaded by scattering matrices, avoiding growing propagation
exponentials. Identical lossless hosts terminate the two sides.

Conventions: exp(-i omega t), lengths nm, epsilon_xy=+i*g*M, and the fixed lab
basis e_+=(x+i*y)/sqrt(2), e_-=(x-i*y)/sqrt(2) on BOTH propagation directions.
Reversing light propagation does not reverse M or exchange circular channels.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import fsum
import numpy as np

from .faraday_mirror import _real_scalar, _complex_scalar, circular_jones
from .materials import gyrotropic_permittivity


def _channel_slab(epsilon_channels, thickness, wavelength, host):
    """Same-host scalar slab S blocks, including the exact epsilon=0 limit.

    Let n^2=epsilon, p=exp(i*k0*n*d), and f=(1-p^2)/n. Then
    t=4*h*p/[4*h+(h-n)^2*f], r=(h^2-n^2)*f/[4*h+(h-n)^2*f].
    expm1 preserves thin/low-index layers, and f -> -2i*k0*d when n -> 0.
    The outgoing passive branch has Im(n)>=0, so no growing exponent appears.
    """
    channels = np.asarray(epsilon_channels, dtype=np.complex128)
    if np.any(channels.imag < 0):
        raise ValueError("circular permittivities must be passive: imaginary parts must be nonnegative")
    indices = np.sqrt(channels)
    if thickness == 0:
        return np.ones(2, complex), np.zeros(2, complex), indices
    k0d = (2*np.pi/wavelength)*thickness
    if not np.isfinite(k0d):
        raise ValueError("layer optical thickness overflows; check physical length scales")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore", under="ignore"):
        argument = 1j*k0d*indices
        if not np.isfinite(argument).all():
            raise ValueError("layer propagation phase overflows; check physical scales")
        propagation = np.exp(argument)
        quotient = np.empty(2, complex)
        nonzero = indices != 0
        quotient[nonzero] = -np.expm1(2*argument[nonzero])/indices[nonzero]
        quotient[~nonzero] = -2j*k0d
        denominator = 4*host + (host-indices)**2*quotient
        scale = 4*host + np.abs((host-indices)**2*quotient)
        if not np.isfinite(denominator).all() or not np.isfinite(scale).all():
            raise ValueError("layer boundary amplitudes overflow; check material scales")
        if np.any(np.abs(denominator) <= 64*np.finfo(float).eps*scale):
            raise ValueError("layer boundary solution is singular to working precision")
        transmission = 4*host*propagation/denominator
        reflection = (host**2-channels)*quotient/denominator
    if not np.isfinite(transmission).all() or not np.isfinite(reflection).all():
        raise ValueError("layer scattering amplitudes are nonfinite")
    return transmission, reflection, indices


def _check_passive_blocks(left_r, forward_t, reverse_t, right_r, *, context):
    """Same-host two-port field amplitudes have ordinary Euclidean flux norm."""
    blocks = np.stack((np.stack((left_r, reverse_t), axis=-1),
                       np.stack((forward_t, right_r), axis=-1)), axis=-2)
    if not np.isfinite(blocks).all():
        raise ValueError(f"{context} contains nonfinite scattering amplitudes")
    largest = np.linalg.svd(blocks, compute_uv=False)[:, 0]
    if np.any(largest > 1 + 2e-10):
        raise ValueError(f"{context} is active or numerically unstable: two-port scattering is not passive")
    return float(np.max(np.maximum(largest-1, 0)))


def _cascade(left_r, forward_t, reverse_t, right_r, layer_t, layer_r):
    """Redheffer composition with the new symmetric scalar slab on the right."""
    denominator = 1-right_r*layer_r
    if np.any(np.abs(denominator) <= 64*np.finfo(float).eps):
        raise ValueError("multilayer internal feedback is singular to working precision")
    # Use the previous blocks throughout: right and left reflections differ.
    return (left_r + reverse_t*layer_r*forward_t/denominator,
            layer_t*forward_t/denominator,
            reverse_t*layer_t/denominator,
            layer_r + layer_t*right_r*layer_t/denominator)


def layer_stack_scattering(layers, wavelength_nm, host_index=1., magnetization=1.):
    """Return exact forward/backward S blocks of a planar gyrotropic stack.

    ``layers`` is an ordered left-to-right sequence of mappings with
    ``epsilon_diagonal``, ``thickness_nm`` and optional ``g`` (default zero).
    Optional ``name`` and ``metadata`` fields are retained as provenance.
    Thicknesses are nonnegative; zero-thickness layers and the empty stack
    have the exact identity response. Passive scalar circular permittivities,
    including exactly zero index, are supported. Magnetization is a signed
    dimensionless scalar in [-1,1] common to all layers, fixed in lab +z.

    Each layer is first represented between the common host. Cascading these
    S matrices with zero host gap cancels the virtual host interfaces exactly,
    leaving the actual contiguous material boundaries. The two external
    reference planes are the stack's front and back faces. Consequently an
    empty stack has t=1,r=0, whereas a finite host-matched layer retains phase.

    Transmission eigenvalues coincide in both directions for these decoupled
    scalar Maxwell channels. The Jones matrix need not be symmetric, and an
    asymmetric stack generally has distinct left and right reflections.
    """
    wavelength = _real_scalar(wavelength_nm, "wavelength_nm", positive=True)
    host = _real_scalar(host_index, "host_index", positive=True)
    magnetization = _real_scalar(magnetization, "magnetization")
    if abs(magnetization) > 1:
        raise ValueError("magnetization magnitude must not exceed one")
    if isinstance(layers, (str, bytes, Mapping)) or not isinstance(layers, Sequence):
        raise ValueError("layers must be a sequence of layer mappings")
    k_host = 2*np.pi/wavelength*host
    if not np.isfinite(k_host) or k_host <= 0:
        raise ValueError("host wavenumber overflows or underflows")
    normalized = []
    left_r, right_r = np.zeros(2, complex), np.zeros(2, complex)
    forward_t, reverse_t = np.ones(2, complex), np.ones(2, complex)
    for number, specification in enumerate(layers):
        if not isinstance(specification, Mapping):
            raise ValueError(f"layer {number} must be a mapping")
        unknown = set(specification)-{"epsilon_diagonal", "g", "thickness_nm", "name", "metadata"}
        if unknown:
            raise ValueError(f"layer {number} has unsupported fields: {sorted(unknown)}")
        if "epsilon_diagonal" not in specification or "thickness_nm" not in specification:
            raise ValueError(f"layer {number} requires epsilon_diagonal and thickness_nm")
        diagonal = _complex_scalar(specification["epsilon_diagonal"], "epsilon_diagonal")
        coupling = _complex_scalar(specification.get("g", 0.), "g")
        thickness = _real_scalar(specification["thickness_nm"], "thickness_nm", nonnegative=True)
        gyrotropic_permittivity(diagonal, coupling, [0., 0., magnetization])
        channels = np.array([diagonal-coupling*magnetization,
                             diagonal+coupling*magnetization], complex)
        layer_t, layer_r, indices = _channel_slab(channels, thickness, wavelength, host)
        left_r, forward_t, reverse_t, right_r = _cascade(left_r, forward_t, reverse_t, right_r, layer_t, layer_r)
        if not np.isfinite(np.concatenate((left_r, forward_t, reverse_t, right_r))).all():
            raise ValueError("multilayer cascade contains nonfinite amplitudes")
        item = {"epsilon_diagonal": diagonal, "g": coupling, "thickness_nm": thickness,
                "epsilon_channels": channels.copy(), "indices": indices.copy()}
        if "name" in specification:
            if not isinstance(specification["name"], str):
                raise ValueError("layer name must be a string")
            item["name"] = specification["name"]
        if "metadata" in specification:
            if not isinstance(specification["metadata"], Mapping):
                raise ValueError("layer metadata must be a mapping")
            item["metadata"] = dict(specification["metadata"])
        normalized.append(item)
    passive_error = _check_passive_blocks(left_r, forward_t, reverse_t, right_r, context="multilayer")
    try:
        thickness = fsum(layer["thickness_nm"] for layer in normalized)
    except OverflowError as exc:
        raise ValueError("total stack thickness overflows") from exc
    if not np.isfinite(thickness):
        raise ValueError("total stack thickness overflows")
    transmission = circular_jones(forward_t)
    reverse_transmission = circular_jones(reverse_t)
    reflection = circular_jones(left_r)
    reverse_reflection = circular_jones(right_r)
    left_loss = np.eye(2)-transmission.conj().T@transmission-reflection.conj().T@reflection
    right_loss = np.eye(2)-reverse_transmission.conj().T@reverse_transmission-reverse_reflection.conj().T@reverse_reflection
    return {
        "transmission_eigenvalues": forward_t, "reverse_transmission_eigenvalues": reverse_t,
        "reflection_eigenvalues": left_r, "reverse_reflection_eigenvalues": right_r,
        "transmission": transmission, "reverse_transmission": reverse_transmission,
        "reflection": reflection, "reverse_reflection": reverse_reflection,
        "material_loss_operator": left_loss, "left_material_loss_operator": left_loss,
        "right_material_loss_operator": right_loss,
        "k_host_nm_inv": float(k_host), "thickness_nm": thickness,
        "wavelength_nm": wavelength, "host_index": host, "magnetization": magnetization,
        "layer_count": len(normalized), "layers": normalized,
        "passivity_roundoff_excess": passive_error,
        "basis": "fixed laboratory x,y in both directions; e+=(x+i y)/sqrt(2), e-=(x-i y)/sqrt(2)",
        "reference_planes": "front and back faces of the contiguous stack",
        "method": "stable scalar circular-channel scattering-matrix cascade; no growing transfer exponentials",
        "model": "exact planar axial gyrotropic multilayer, normal incidence, mu=1, identical lossless hosts",
        "zero_index_policy": "analytic (1-exp(2 i k0 n d))/n limit; exactly zero-index circular channels are supported",
        "energy_note": "Bare-stack material loss operators use incoming-minus-outgoing flux; independent volume-heating validation is separate.",
    }


def stack_mirror_return(stack, gap_nm=0., mirror_amplitude=-1.):
    """Return an asymmetric stack terminated by an isotropic opaque mirror.

    For D=rm exp(2 i kh gap), the exact coherent return is
    J=Rl+Trl D (I-Rr D)^-1 Tlr. The denominator uses reflection seen from
    INSIDE the gap (Rr); replacing it with Rl is incorrect for asymmetric stacks.
    All returns are included. The separately reported first return Trl D Tlr
    omits prompt reflection and subsequent cavity trips.
    """
    if not isinstance(stack, Mapping):
        raise ValueError("stack must be a layer_stack_scattering result mapping")
    gap = _real_scalar(gap_nm, "gap_nm", nonnegative=True)
    mirror = _complex_scalar(mirror_amplitude, "mirror_amplitude")
    if abs(mirror) > 1:
        raise ValueError("mirror_amplitude must be passive: abs(rm) <= 1")
    try:
        k = _real_scalar(stack["k_host_nm_inv"], "k_host_nm_inv", positive=True)
        names = ("reflection_eigenvalues", "transmission_eigenvalues",
                 "reverse_transmission_eigenvalues", "reverse_reflection_eigenvalues")
        values = [np.asarray(stack[name], dtype=complex) for name in names]
    except KeyError as exc:
        raise ValueError(f"stack is missing {exc.args[0]}") from exc
    if any(value.shape != (2,) or not np.isfinite(value).all() for value in values):
        raise ValueError("stack must contain two finite eigenvalues for each of its four S blocks")
    left_r, forward_t, reverse_t, right_r = values
    _check_passive_blocks(left_r, forward_t, reverse_t, right_r, context="stack")
    phase = 2*k*gap
    if not np.isfinite(phase):
        raise ValueError("mirror gap phase overflows; check length scales")
    load = mirror*np.exp(1j*phase)
    denominator = 1-right_r*load
    if np.any(np.abs(denominator) <= 64*np.finfo(float).eps):
        raise ValueError("mirror-loaded stack cavity is singular to working precision")
    forward_gap = forward_t/denominator
    cavity = reverse_t*load*forward_gap
    returned = left_r+cavity
    first = reverse_t*load*forward_t
    if not np.isfinite(np.concatenate((forward_gap, cavity, returned, first))).all():
        raise ValueError("mirror-loaded stack contains nonfinite amplitudes")
    forward = circular_jones(forward_gap)
    total = circular_jones(returned)
    mirror_loss = (1-abs(mirror)**2)*forward.conj().T@forward
    material_loss = np.eye(2)-total.conj().T@total-mirror_loss
    return {
        "returned_jones": total, "returned_eigenvalues": returned,
        "gap_forward_jones": forward, "gap_return_jones": load*forward,
        "front_reflection_jones": circular_jones(left_r),
        "back_reflection_jones": circular_jones(right_r),
        "first_return_jones": circular_jones(first),
        "cavity_return_jones": circular_jones(cavity),
        "mirror_loss_operator": mirror_loss, "material_loss_operator": material_loss,
        "mirror_load": load, "gap_nm": gap, "mirror_amplitude": mirror,
        "basis": "fixed laboratory x,y on both propagation legs; magnetization unchanged",
        "energy_note": "Material-loss operator is a flux difference; validation independently integrates volume Joule heating. The scalar mirror is an opaque absorbing termination, not an extra transmission port.",
    }
