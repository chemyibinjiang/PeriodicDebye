"""Fourier-modal Maxwell solver for one-dimensional gyrotropic lamellar layers.

The grating is periodic in x, invariant in y and stacked along z. Incidence is
normal, mu=1, time dependence exp(-i omega t), and each constituent has
epsilon=[[e, i*g*M, 0],[-i*g*M,e,0],[0,0,e]]. M stays fixed in laboratory z.
Lengths are nm. Both exterior half spaces are isotropic and lossless.

At a vertical material boundary Dx and Ey are continuous. Writing Ex in terms
of those continuous fields yields the inverse-rule Fourier factorization used
below. This avoids applying the ordinary product rule to discontinuous Ex.
The approach follows Li's Fourier factorization rules (JOSA A 13, 1870, 1996;
doi:10.1364/JOSAA.13.001870). Retained harmonics remain an approximation that
must be converged. Interfaces and decaying propagation are cascaded with
scattering matrices; growing transfer exponentials are never constructed.

This module describes rectangular grooves with a specified, uniform filling
profile. It does not infer a real dip coating's cap thickness, roughness or
infiltration from a material name, nor describe a two-dimensional pillar array.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import numpy as np
from scipy.linalg import eig, solve

from .faraday_mirror import _real_scalar, _complex_scalar
from .materials import gyrotropic_permittivity


def _lamellar_convolution(a, b, fraction, orders):
    """Exact coefficients of material a in a centered strip, b elsewhere."""
    difference = orders[:, None] - orders[None, :]
    if fraction == 0:
        return b * np.eye(len(orders), dtype=complex)
    if fraction == 1:
        return a * np.eye(len(orders), dtype=complex)
    return ((a-b)*fraction*np.sinc(difference*fraction)
            + b*np.eye(len(orders), dtype=complex))


def _layer_operators(epsilon_a, g_a, epsilon_b, g_b, fraction, orders, kx):
    """Return P,Q for e'=ik0 P f, f'=ik0 Q e; e=(Ex,Ey), f=(Hy,-Hx)."""
    eye = np.eye(len(orders), dtype=complex)
    zero = np.zeros_like(eye)
    conv = lambda a, b: _lamellar_convolution(a, b, fraction, orders)
    # Ex=C(1/e) Dx-C(i*g/e) Ey. Dy=C(-i*g/e) Dx+C(e-g*g/e) Ey.
    inverse = conv(1/epsilon_a, 1/epsilon_b)
    ratio = conv(1j*g_a/epsilon_a, 1j*g_b/epsilon_b)
    axx = solve(inverse, eye, assume_a="gen", check_finite=False)
    axy = axx @ ratio
    ayx = -ratio @ axx
    ayy = conv(epsilon_a-g_a*g_a/epsilon_a,
               epsilon_b-g_b*g_b/epsilon_b) - ratio @ axy
    # Ez is tangential to the x interfaces: Dz=C(e) Ez, not C(1/e)^-1 Ez.
    ez = conv(epsilon_a, epsilon_b)
    diagonal_k = np.diag(kx)
    px = eye - diagonal_k @ solve(ez, diagonal_k, check_finite=False)
    p = np.block([[px, zero], [zero, eye]])
    q = np.block([[axx, axy], [ayx, ayy-np.diag(kx*kx)]])
    return p, q


def _layer_modes(layer, orders, kx):
    p, q = _layer_operators(layer["epsilon_a"], layer["g_a"],
                            layer["epsilon_b"], layer["g_b"],
                            layer["fill_fraction"], orders, kx)
    values, electric = eig(p @ q, check_finite=False)
    propagation = np.sqrt(values.astype(complex))
    scale = max(1., float(np.max(np.abs(propagation))))
    if np.any(np.abs(propagation) < 1e-12*scale):
        raise ValueError("a retained internal mode is at cutoff; change wavelength or geometry")
    # Im(q)>0 gives decaying forward waves. For a roundoff-real mode the flux
    # selects the forward branch. Real lossless roots can acquire tiny Im parts
    # during the non-Hermitian eigensolve; removing only those avoids exponential
    # growth in otherwise lossless, very thick slabs.
    nearly_real = np.abs(propagation.imag) < 64*np.finfo(float).eps*scale
    propagation[nearly_real] = propagation[nearly_real].real
    propagation[propagation.imag < 0] *= -1
    magnetic = (q @ electric)/propagation[None, :]
    flux = np.real(np.sum(electric.conj()*magnetic, axis=0))
    flip = nearly_real & (flux < 0)
    propagation[flip] *= -1
    magnetic[:, flip] *= -1
    if not np.isfinite(electric).all() or not np.isfinite(magnetic).all():
        raise ValueError("nonfinite grating modes; check physical scales")
    return electric, magnetic, propagation


def _port_modes(index, kx):
    propagation = np.sqrt(index*index-kx*kx+0j)
    if np.any(np.abs(propagation) < 1e-9*max(1., index)):
        raise ValueError("an exterior diffraction order is at a Rayleigh cutoff")
    # E amplitudes use Ex first, then Ey. For oblique diffraction Ex's flux
    # weight is n^2/q, not q: its longitudinal electric component matters.
    admittance = np.concatenate((index*index/propagation, propagation))
    return np.eye(len(admittance), dtype=complex), np.diag(admittance), propagation


def _interface(left_e, left_f, right_e, right_f):
    """Tangential E and H continuity, outgoing=[left-,right+]."""
    matrix = np.block([[left_e, -right_e], [left_f, right_f]])
    rhs = np.block([[-left_e, right_e], [left_f, right_f]])
    solution = solve(matrix, rhs, check_finite=False)
    size = left_e.shape[1]
    return (solution[:size, :size], solution[size:, :size],
            solution[:size, size:], solution[size:, size:])


def _cascade(first, second):
    """Redheffer star product, S block order (Rl,Tlr,Trl,Rr)."""
    al, at, au, ar = first
    bl, bt, bu, br = second
    size = len(al)
    inside = solve(np.eye(size)-ar @ bl, np.concatenate((at, ar @ bu), axis=1),
                   check_finite=False)
    forward, feedback = inside[:, :size], inside[:, size:]
    return (al + au @ bl @ forward, bt @ forward,
            au @ (bu + bl @ feedback), br + bt @ feedback)


def _normalize_layers(layers, magnetization):
    if isinstance(layers, (str, bytes, Mapping)) or not isinstance(layers, Sequence):
        raise ValueError("layers must be a sequence of layer mappings")
    normalized = []
    allowed = {"thickness_nm", "fill_fraction", "epsilon_a", "epsilon_b", "g_a", "g_b", "name"}
    required = {"thickness_nm", "fill_fraction", "epsilon_a", "epsilon_b"}
    for number, layer in enumerate(layers):
        if not isinstance(layer, Mapping) or not required.issubset(layer):
            raise ValueError(f"layer {number} requires {sorted(required)}")
        if set(layer)-allowed:
            raise ValueError(f"layer {number} has unsupported fields: {sorted(set(layer)-allowed)}")
        item = {"thickness_nm": _real_scalar(layer["thickness_nm"], "thickness_nm", nonnegative=True),
                "fill_fraction": _real_scalar(layer["fill_fraction"], "fill_fraction")}
        if not 0 <= item["fill_fraction"] <= 1:
            raise ValueError("fill_fraction must lie in [0,1]")
        for suffix in ("a", "b"):
            diagonal = _complex_scalar(layer["epsilon_"+suffix], "epsilon_"+suffix)
            coupling = _complex_scalar(layer.get("g_"+suffix, 0), "g_"+suffix)
            gyrotropic_permittivity(diagonal, coupling, [0., 0., magnetization])
            if diagonal == 0:
                raise ValueError("zero diagonal permittivity is unsupported by the inverse-rule factorization")
            item["epsilon_"+suffix] = diagonal
            item["g_"+suffix] = coupling*magnetization
        if "name" in layer:
            if not isinstance(layer["name"], str):
                raise ValueError("layer name must be a string")
            item["name"] = layer["name"]
        normalized.append(item)
    return normalized


def grating_scattering(wavelength_nm, period_nm, layers, n_incident=1., n_exit=1.5,
                       order=7, magnetization=1.):
    """Solve a stack of centered, rectangular 1D gratings at normal incidence.

    ``order`` retains m=-order,...,+order, with both Ex and Ey. A layer mapping
    requires ``thickness_nm, fill_fraction, epsilon_a, epsilon_b`` and optionally
    ``g_a, g_b, name``. Material a occupies the centered fraction of each period;
    material b fills the remainder. Fractions 0 or 1 describe uniform films.
    Every layer shares one period and strip center. No effective-medium model
    is used. Increasing order is necessary to establish numerical accuracy.

    ``transmission`` and ``reflection`` are zeroth-order 2x2 Jones matrices.
    Full ``scattering`` blocks use harmonic tangential E amplitudes (all Ex
    orders followed by all Ey orders); they are not power-normalized. The
    supplied port flux weights account for both longitudinal E and different
    indices. Power arrays contain x- and y-incident results. ``absorbed_power``
    is incoming minus all propagating outgoing flux, not an independent volume
    heating integral. The substrate is a half space unless explicitly supplied
    as a finite uniform layer followed by another exterior medium.

    The reference planes are the stack's front/back faces. M remains fixed in
    laboratory coordinates for reverse incidence; the two backward Jones axes
    are the same fixed x,y axes as the forward ones.
    """
    wavelength = _real_scalar(wavelength_nm, "wavelength_nm", positive=True)
    period = _real_scalar(period_nm, "period_nm", positive=True)
    incident = _real_scalar(n_incident, "n_incident", positive=True)
    exit_index = _real_scalar(n_exit, "n_exit", positive=True)
    magnetic_sign = _real_scalar(magnetization, "magnetization")
    if abs(magnetic_sign) > 1:
        raise ValueError("magnetization magnitude must not exceed one")
    if isinstance(order, (bool, np.bool_)) or not isinstance(order, (int, np.integer)) or not 0 <= order <= 128:
        raise ValueError("order must be an integer in [0,128]")
    if (order+1)*(wavelength/period) <= max(incident, exit_index)*(1+1e-12):
        raise ValueError("order truncation omits propagating diffraction channels or an exterior cutoff")
    normalized = _normalize_layers(layers, magnetic_sign)
    orders = np.arange(-order, order+1)
    kx = orders*(wavelength/period)
    left_e, left_f, left_q = _port_modes(incident, kx)
    right_e, right_f, right_q = _port_modes(exit_index, kx)
    size = len(left_e)
    zero = np.zeros((size, size), complex)
    identity = np.eye(size, dtype=complex)
    scattering = (zero.copy(), identity.copy(), identity.copy(), zero.copy())
    current_e, current_f = left_e, left_f
    for layer in normalized:
        if layer["thickness_nm"] == 0:
            continue
        electric, magnetic, propagation = _layer_modes(layer, orders, kx)
        scattering = _cascade(scattering, _interface(current_e, current_f, electric, magnetic))
        phase = 2*np.pi/wavelength*layer["thickness_nm"]*propagation
        if not np.isfinite(phase).all():
            raise ValueError("layer propagation phase overflows; check physical scales")
        decay = np.diag(np.exp(1j*phase))
        scattering = _cascade(scattering, (zero, decay, decay, zero))
        current_e, current_f = electric, magnetic
    scattering = _cascade(scattering, _interface(current_e, current_f, right_e, right_f))
    if not all(np.isfinite(block).all() for block in scattering):
        raise ValueError("nonfinite grating scattering solution")
    left_r, forward_t, reverse_t, right_r = scattering
    left_weights = np.diag(left_f).real.copy()
    right_weights = np.diag(right_f).real.copy()
    # Test the complete propagating two-port S matrix, including nonzero orders.
    # Evanescent amplitudes do not have a positive individual flux norm.
    propagating_left = np.flatnonzero(left_weights > 0)
    propagating_right = np.flatnonzero(right_weights > 0)
    combined = np.block([[left_r, reverse_t], [forward_t, right_r]])
    selected = np.concatenate((propagating_left, size+propagating_right))
    weights = np.concatenate((left_weights[propagating_left], right_weights[propagating_right]))
    power_s = combined[np.ix_(selected, selected)]*np.sqrt(weights[:, None]/weights[None, :])
    passive_excess = max(0., float(np.linalg.svd(power_s, compute_uv=False)[0])-1.)
    if passive_excess > 2e-7:
        raise ValueError(f"grating scattering violates passivity (singular-value excess {passive_excess:.3g})")
    count = len(orders)
    central = [order, count+order]
    reflected_fields = left_r[:, central]
    transmitted_fields = forward_t[:, central]
    reflected = (left_weights[:count, None]*abs(reflected_fields[:count])**2
                 + left_weights[count:, None]*abs(reflected_fields[count:])**2)/incident
    transmitted = (right_weights[:count, None]*abs(transmitted_fields[:count])**2
                   + right_weights[count:, None]*abs(transmitted_fields[count:])**2)/incident
    r_operator = reflected_fields.conj().T @ (left_weights[:, None]*reflected_fields)/incident
    t_operator = transmitted_fields.conj().T @ (right_weights[:, None]*transmitted_fields)/incident
    return {
        "transmission": forward_t[np.ix_(central, central)].copy(),
        "reflection": left_r[np.ix_(central, central)].copy(),
        "reverse_transmission": reverse_t[np.ix_(central, central)].copy(),
        "reverse_reflection": right_r[np.ix_(central, central)].copy(),
        "scattering": dict(zip(("r_left", "t_forward", "t_reverse", "r_right"), scattering)),
        "transmitted_power": transmitted.sum(axis=0),
        "reflected_power": reflected.sum(axis=0),
        "absorbed_power": 1-transmitted.sum(axis=0)-reflected.sum(axis=0),
        "transmission_by_order": transmitted,
        "reflection_by_order": reflected,
        "transmission_power_operator": t_operator,
        "reflection_power_operator": r_operator,
        "material_loss_operator": np.eye(2)-r_operator-t_operator,
        "flux_weights_incident": left_weights,
        "flux_weights_exit": right_weights,
        "harmonic_orders": orders,
        "q_incident": left_q,
        "q_exit": right_q,
        "order": int(order), "retained_harmonic_count": count,
        "wavelength_nm": wavelength, "period_nm": period,
        "n_incident": incident, "n_exit": exit_index,
        "magnetization": magnetic_sign,
        "layers": normalized,
        "passivity_roundoff_excess": passive_excess,
        "basis": "fixed laboratory x,y; full S uses Ex[-order:order], then Ey[-order:order] tangential fields",
        "method": "1D tensor Fourier-modal Maxwell solver; Li inverse factorization; stable scattering cascade",
        "absorption_note": "incoming minus all propagating outgoing flux; not an independent volume-heating integral",
    }
