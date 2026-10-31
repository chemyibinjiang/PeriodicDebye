"""Full-order planar mirror termination of the tensor grating scattering matrix.

An ideal PEC has tangential-electric reflection -1 for every harmonic and both
polarizations. The feedback includes retained evanescent orders. A complex,
angle-independent scalar reflector is not generally passive for evanescent TE
and TM waves simultaneously; this helper rejects it when those waves exist.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np

from .faraday_mirror import _real_scalar, _complex_scalar


def grating_mirror_return(result, gap_nm=0., mirror_amplitude=-1.):
    """Terminate the exit half space with an opaque planar scalar mirror.

    ``result`` is a ``grating_scattering`` result. The lossless gap has its
    ``n_exit`` index; to place a mirror behind a finite glass substrate and an
    air gap, include the glass as a finite layer and choose ``n_exit=1``.

    All blocks refer to the fixed laboratory x,y basis. With P containing
    exp(i*k0*q_exit*gap), D=rm*P**2, the returned matrix is
    Rl + Trl D (I-Rr D)^-1 Tlr. The independently exposed first return is
    Trl D Tlr; prompt reflection and later cavity trips are excluded from it.
    Backward blocks are those actually computed by the grating solver.

    PEC rm=-1 is physical for all retained harmonics. Other real scalars in
    [-1,1] are ideal mathematical loads, not measured mirror optical data.
    A complex scalar with abs(rm)<=1 is accepted only when every retained exit
    order propagates. Real mirrors generally require polarization- and
    angle-dependent coefficients. No mirror coating or material is inferred.
    """
    if not isinstance(result, Mapping):
        raise ValueError("result must be a grating_scattering result mapping")
    gap = _real_scalar(gap_nm, "gap_nm", nonnegative=True)
    mirror = _complex_scalar(mirror_amplitude, "mirror_amplitude")
    if abs(mirror) > 1:
        raise ValueError("mirror_amplitude must satisfy abs(rm) <= 1")
    try:
        wavelength = _real_scalar(result["wavelength_nm"], "wavelength_nm", positive=True)
        incident = _real_scalar(result["n_incident"], "n_incident", positive=True)
        exit_index = _real_scalar(result["n_exit"], "n_exit", positive=True)
        orders = np.asarray(result["harmonic_orders"])
        q_exit = np.asarray(result["q_exit"], dtype=complex)
        left_weights = np.asarray(result["flux_weights_incident"], dtype=float)
        right_weights = np.asarray(result["flux_weights_exit"], dtype=float)
        scattering = result["scattering"]
        blocks = [np.asarray(scattering[name], dtype=complex)
                  for name in ("r_left", "t_forward", "t_reverse", "r_right")]
    except (KeyError, TypeError) as exc:
        raise ValueError("result is missing valid grating scattering or port data") from exc
    if orders.ndim != 1 or len(orders) % 2 != 1 or len(orders) < 1:
        raise ValueError("harmonic_orders must be an odd, centered consecutive sequence")
    count = len(orders)
    central_order = count//2
    if not np.array_equal(orders, np.arange(-central_order, central_order+1)):
        raise ValueError("harmonic_orders must be centered and consecutive")
    size = 2*count
    if any(block.shape != (size, size) or not np.isfinite(block).all() for block in blocks):
        raise ValueError("all four scattering blocks must be finite 2N by 2N matrices")
    if (q_exit.shape != (count,) or not np.isfinite(q_exit).all()
            or np.any(q_exit.real < 0) or np.any(q_exit.imag < 0)
            or np.any(abs(q_exit) == 0)
            or not np.isclose(q_exit[central_order], exit_index, rtol=1e-12, atol=1e-14)):
        raise ValueError("q_exit must contain outgoing, nonzero lossless-port wavenumbers")
    for weights in (left_weights, right_weights):
        if weights.shape != (size,) or not np.isfinite(weights).all() or np.any(weights < 0):
            raise ValueError("port flux weights must be finite nonnegative arrays of length 2N")
    if np.any(q_exit.imag > 0) and mirror.imag != 0:
        raise ValueError("a complex scalar mirror is not a passive model for both evanescent polarizations; use PEC or a real scalar")
    phase = (2*np.pi/wavelength)*gap*q_exit
    if not np.isfinite(phase).all():
        raise ValueError("mirror gap phase overflows; check physical scales")
    propagation = np.tile(np.exp(1j*phase), 2)
    load = mirror*propagation*propagation
    left_r, forward_t, reverse_t, right_r = blocks
    feedback = np.eye(size)-right_r*load[None, :]
    try:
        forward_gap = np.linalg.solve(feedback, forward_t)
    except np.linalg.LinAlgError as exc:
        raise ValueError("mirror-loaded grating feedback is singular") from exc
    if not np.isfinite(forward_gap).all():
        raise ValueError("mirror-loaded grating contains nonfinite amplitudes")
    residual = np.linalg.norm(feedback @ forward_gap-forward_t)/max(np.linalg.norm(forward_t), 1e-300)
    if not np.isfinite(residual) or residual > 1e-9:
        raise ValueError("mirror-loaded grating feedback is numerically unresolved")
    cavity = reverse_t @ (load[:, None]*forward_gap)
    returned = left_r+cavity
    first = reverse_t @ (load[:, None]*forward_t)
    central = [central_order, count+central_order]
    fields = returned[:, central]
    by_order = (left_weights[:count, None]*abs(fields[:count])**2
                + left_weights[count:, None]*abs(fields[count:])**2)/incident
    returned_power_operator = fields.conj().T @ (left_weights[:, None]*fields)/incident
    # Evaluate absorption at the mirror plane. Propagation preserves the flux
    # of propagating harmonics; evanescent amplitudes decay and carry no separate
    # flux. For accepted real scalar loads their interference flux also vanishes.
    at_mirror = propagation[:, None]*forward_gap[:, central]
    mirror_loss = ((1-abs(mirror)**2)/incident
                   * at_mirror.conj().T @ (right_weights[:, None]*at_mirror))
    material_loss = np.eye(2)-returned_power_operator-mirror_loss
    if not np.isfinite(returned).all() or not np.isfinite(material_loss).all():
        raise ValueError("mirror-loaded grating contains nonfinite fields or powers")
    if np.linalg.eigvalsh(material_loss).min() < -2e-7:
        raise ValueError("mirror-loaded grating violates passivity or is numerically unresolved")
    return {
        "returned_jones": returned[np.ix_(central, central)].copy(),
        "first_return_jones": first[np.ix_(central, central)].copy(),
        "cavity_return_jones": cavity[np.ix_(central, central)].copy(),
        "front_reflection_jones": left_r[np.ix_(central, central)].copy(),
        "returned_scattering": returned,
        "first_return_scattering": first,
        "gap_forward_fields": forward_gap[:, central],
        "gap_return_fields": load[:, None]*forward_gap[:, central],
        "returned_power": by_order.sum(axis=0),
        "reflection_by_order": by_order,
        "returned_power_operator": returned_power_operator,
        "mirror_loss_operator": mirror_loss,
        "material_loss_operator": material_loss,
        "mirror_absorbed_power": np.diag(mirror_loss).real.copy(),
        "material_absorbed_power": np.diag(material_loss).real.copy(),
        "mirror_load": load, "gap_nm": gap, "gap_index": exit_index,
        "mirror_amplitude": mirror, "feedback_relative_residual": float(residual),
        "harmonic_orders": orders.copy(),
        "basis": "fixed laboratory x,y on both legs; magnetization unchanged",
        "model": "full retained-harmonic feedback to a planar scalar mirror in the exit medium",
        "mirror_model": "ideal PEC" if mirror == -1 else "ideal opaque scalar load; not measured mirror data",
        "absorption_note": "material absorption is incoming minus all returned orders and mirror loss; not an independent volume integral",
    }
