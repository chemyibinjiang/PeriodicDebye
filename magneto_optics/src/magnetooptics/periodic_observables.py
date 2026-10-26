"""Diffraction ports and polarization of a normally illuminated periodic sheet.

Time dependence is exp(-i omega t).  All complex port fields are referenced to
z=0.  The transmitted zero order contains the incident wave.  Jones matrices
use the fixed laboratory (x,y) basis for both propagation directions; the
reflected basis therefore is not right-handed with its -z wavevector.
"""

from __future__ import annotations

import numpy as np

from .materials import _positive_real
from .observables import cross_sections
from .periodic_geometry import RectangularCell, validate_periods


def diffraction_orders(periods_nm, wavelength_nm, host_index=1, *, grazing_tolerance=1e-8, max_orders=100000):
    """Enumerate every propagating diffraction order at normal incidence.

    An order at or close to a Rayleigh threshold is rejected on either side:
    abs(1-|G|**2/k**2) <= grazing_tolerance.  Such singular channels require
    a dedicated limiting treatment, not silently setting their power to zero.
    Returned arrays have a common order index and include the (0,0) channel.
    """
    periods = validate_periods(periods_nm)
    wavelength = _positive_real(wavelength_nm, "wavelength_nm")
    index = _positive_real(host_index, "host_index")
    if isinstance(grazing_tolerance, (bool, np.bool_)) or not np.isscalar(grazing_tolerance) or not np.isrealobj(grazing_tolerance) or not np.isfinite(grazing_tolerance) or not 0 < grazing_tolerance < 1:
        raise ValueError("grazing_tolerance must be a real number between zero and one")
    if isinstance(max_orders, (bool, np.bool_)) or not isinstance(max_orders, (int, np.integer)) or max_orders < 1:
        raise ValueError("max_orders must be a positive integer")
    k = 2 * np.pi * index / wavelength
    if not np.isfinite(k) or k <= 0:
        raise ValueError("wavenumber overflows or underflows; check physical length scales")
    limits = np.ceil(index * periods / wavelength * np.sqrt(1 + grazing_tolerance)).astype(object)
    if any(not np.isfinite(value) or value > max_orders for value in limits):
        raise ValueError("diffraction order range exceeds max_orders")
    nx, ny = (int(value) for value in limits)
    if (2 * nx + 1) * (2 * ny + 1) > 9 * max_orders:
        raise ValueError("diffraction candidate grid exceeds max_orders memory guard")
    indices = np.stack(np.meshgrid(np.arange(-nx, nx + 1), np.arange(-ny, ny + 1), indexing="ij"), axis=-1).reshape(-1, 2)
    transverse = 2 * np.pi * indices / periods
    margin = 1 - np.sum((transverse / k)**2, axis=1)
    near_grazing = np.abs(margin) <= grazing_tolerance
    if np.any(near_grazing):
        offending = indices[near_grazing].tolist()
        raise ValueError(f"Rayleigh/grazing diffraction threshold at orders {offending}; change wavelength or periods")
    propagating = margin > 0
    indices, transverse, margin = indices[propagating], transverse[propagating], margin[propagating]
    if len(indices) > max_orders:
        raise ValueError("number of propagating orders exceeds max_orders")
    kz = k * np.sqrt(margin)
    plus = np.column_stack((transverse, kz)) / k
    minus = np.column_stack((transverse, -kz)) / k
    zero = int(np.flatnonzero(np.all(indices == 0, axis=1))[0])
    return {
        "indices": indices,
        "kx_ky": transverse,
        "kz": kz,
        "directions_transmitted": plus,
        "directions_reflected": minus,
        "flux_weights": kz / k,
        "zero_index": zero,
        "k_nm_inv": float(k),
        "area_nm2": float(np.prod(periods)),
        "periods_nm": periods,
        "grazing_tolerance": float(grazing_tolerance),
    }


def _vectors(value, n, name):
    result = np.asarray(value, dtype=np.complex128)
    if result.shape != (n, 3) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite with shape ({n},3)")
    return result


def evaluate_ports(cell, dipoles, incident, local_field, wavelength_nm, host_index=1, *, orders=None):
    """Return all complex outgoing fields, order powers, total T/R, and heating A.

    The incident field is spatially constant and transverse at z=0, with
    shape (3,) or (N,3).  Fields retain its amplitude; powers and absorption
    are divided by its norm squared.  Heating is obtained independently from
    work on the dipoles minus their self radiation, not defined as 1-T-R.
    """
    if not isinstance(cell, RectangularCell):
        raise ValueError("cell must be a RectangularCell")
    n = len(cell.positions_nm)
    dipoles = _vectors(dipoles, n, "dipoles")
    local_field = _vectors(local_field, n, "local_field")
    incident = np.asarray(incident, dtype=np.complex128)
    if incident.shape == (3,):
        incident = np.broadcast_to(incident, (n, 3))
    incident = _vectors(incident, n, "incident")
    illumination = incident[0]
    input_power = float(np.vdot(illumination, illumination).real)
    if not np.isfinite(input_power) or input_power <= 0:
        raise ValueError("incident field must have finite nonzero amplitude")
    amplitude = np.sqrt(input_power)
    if abs(illumination[2]) > 1e-12 * amplitude or not np.allclose(incident, illumination, rtol=1e-12, atol=1e-14 * amplitude):
        raise ValueError("only a spatially constant transverse normal-incidence field is supported")
    wavelength = _positive_real(wavelength_nm, "wavelength_nm")
    index = _positive_real(host_index, "host_index")
    k = 2 * np.pi * index / wavelength
    if orders is None:
        orders = diffraction_orders(cell.periods_nm, wavelength, index)
    elif not np.array_equal(orders.get("periods_nm"), cell.periods_nm) or not np.isclose(orders.get("k_nm_inv", np.nan), k, rtol=8 * np.finfo(float).eps, atol=0):
        raise ValueError("cached diffraction orders do not match cell periods and wavenumber")
    # Bound phase workspace even for many orders and many occupied cell sites.
    summed = np.zeros((len(orders["indices"]), 3), dtype=np.complex128)
    for first in range(0, len(summed), 256):
        stop = min(first + 256, len(summed))
        particle_chunk = max(1, 1048576 // (stop - first))
        for start in range(0, n, particle_chunk):
            phase = np.exp(-1j * (orders["kx_ky"][first:stop] @ cell.positions_nm[start:start + particle_chunk, :2].T))
            summed[first:stop] += phase @ dipoles[start:start + particle_chunk]
    coefficient = 1j * k**2 / (2 * cell.area_nm2 * orders["kz"])
    fields = []
    for key in ("directions_transmitted", "directions_reflected"):
        direction = orders[key]
        transverse = summed - direction * np.sum(direction * summed, axis=1)[:, None]
        fields.append(coefficient[:, None] * transverse)
    transmitted, reflected = fields
    zero = orders["zero_index"]
    transmitted[zero] += illumination
    transmission = orders["flux_weights"] * np.sum(np.abs(transmitted)**2, axis=1) / input_power
    reflection = orders["flux_weights"] * np.sum(np.abs(reflected)**2, axis=1) / input_power
    material_absorption = cross_sections(dipoles, incident, local_field, k)["absorption_nm2"] / input_power
    absorption = material_absorption / cell.area_nm2
    total_t, total_r = float(transmission.sum()), float(reflection.sum())
    return {
        "order_indices": orders["indices"].copy(),
        "transmitted_fields": transmitted,
        "reflected_fields": reflected,
        "transmitted_power": transmission,
        "reflected_power": reflection,
        "total_transmission": total_t,
        "total_reflection": total_r,
        "absorption": float(absorption),
        "energy_balance_residual": float(total_t + total_r + absorption - 1),
        "zero_transmitted": transmitted[zero].copy(),
        "zero_reflected": reflected[zero].copy(),
        "material_absorption_nm2": float(material_absorption),
        "incident_vector": illumination.copy(),
        "incident_norm_squared": input_power,
        "zero_order_index": zero,
        "flux_weights": orders["flux_weights"].copy(),
    }


def jones_matrices(x_ports, y_ports):
    """Assemble zero-order Jones T and R from unit x and unit y solutions.

    Columns are incident x/y; rows are outgoing lab x/y in both hemispheres.
    Reflected polarization angles are consequently measured in the lab frame,
    not in a propagation-following frame with an implicitly flipped y axis.
    """
    for ports, expected in ((x_ports, [1, 0, 0]), (y_ports, [0, 1, 0])):
        if not np.allclose(ports["incident_vector"], expected, rtol=0, atol=1e-12):
            raise ValueError("Jones assembly requires unit x then unit y incident solutions")
    return {
        "transmission": np.column_stack((x_ports["zero_transmitted"][:2], y_ports["zero_transmitted"][:2])),
        "reflection": np.column_stack((x_ports["zero_reflected"][:2], y_ports["zero_reflected"][:2])),
        "basis": "fixed laboratory x,y for both transmitted +z and reflected -z; columns incident x,y",
    }


def polarization_ellipse(jones_vector):
    """Return lab-frame ellipse orientation, ellipticity and cross fraction.

    Angles are degrees. S3=2 Im(Ex* Ey), so (x+i y)/sqrt(2) has +45 degree
    ellipticity. Orientation is defined modulo 180 degrees and is undefined
    for circular polarization. All observables are undefined for zero field.
    """
    field = np.asarray(jones_vector, dtype=np.complex128)
    if field.shape != (2,) or not np.isfinite(field).all():
        raise ValueError("jones_vector must be a finite complex two-vector")
    ex, ey = field
    s0 = float(abs(ex)**2 + abs(ey)**2)
    if s0 == 0:
        return {"rotation_deg": None, "ellipticity_deg": None, "cross_fraction": None}
    if not np.isfinite(s0):
        raise ValueError("Jones intensity overflows; rescale the input field")
    s1 = float(abs(ex)**2 - abs(ey)**2)
    s2, s3 = 2 * np.real(np.conj(ex) * ey), 2 * np.imag(np.conj(ex) * ey)
    rotation = None if np.hypot(s1, s2) <= 64 * np.finfo(float).eps * s0 else float(np.degrees(0.5 * np.arctan2(s2, s1)))
    return {
        "rotation_deg": rotation,
        "ellipticity_deg": float(np.degrees(0.5 * np.arcsin(np.clip(s3 / s0, -1, 1)))),
        "cross_fraction": float(abs(ey)**2 / s0),
    }
