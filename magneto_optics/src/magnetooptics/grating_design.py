"""Fixed-material design family for the first Si3N4/composite grating scan.

Geometry and substrate/coating assumptions are explicit. This is a 1D grating,
invariant along y, not a 2D post array or a fabrication-qualified device.
"""
from __future__ import annotations

import numpy as np

from .periodic_observables import polarization_ellipse


def nbk7_index(wavelength_nm):
    """SCHOTT N-BK7 Sellmeier proxy, restricted to this study's material band.

    Source: https://media.schott.com/api/public/content/41e799d0bf874807a0bb8e702fbb75b5?v=54856406
    The paper specifies BK7, not a vendor or a measured substrate dispersion.
    Glass absorption is neglected; this is not a calibrated thick-glass model.
    """
    if np.ndim(wavelength_nm) or np.iscomplexobj(wavelength_nm):
        raise ValueError("wavelength_nm must be real and within 1550-1570 nm")
    wavelength = float(wavelength_nm)
    if not 1550 <= wavelength <= 1570:
        raise ValueError("wavelength_nm must be within the material band 1550-1570 nm")
    b = np.array([1.039612120, .231792344, 1.010469450])
    c = np.array([.006000699, .0200179144, 103.56065300])
    square = (wavelength / 1000)**2
    return float(np.sqrt(1 + np.sum(b * square / (square - c))))


def prototype_layers(geometry, epsilon_composite, g_composite, substrate_index):
    """Air -> optional MO cap -> filled SiN grating -> filled overetch -> glass.

    All gaps are completely infiltrated, walls are rectangular, and strips are
    aligned in both patterned layers. These are hypotheses, not AFM measurements.
    """
    period, fill, height, cap = (float(geometry[key]) for key in
                                ("period_nm", "fill_fraction", "height_nm", "cap_nm"))
    overetch = float(geometry.get("overetch_nm", 30.))
    if not np.isfinite([period, fill, height, cap, overetch]).all():
        raise ValueError("geometry must be finite")
    if period <= 0 or height <= 0 or not 0 < fill < 1 or cap < 0 or overetch < 0:
        raise ValueError("invalid grating geometry")
    layers = []
    if cap:
        layers.append(dict(thickness_nm=cap, fill_fraction=1.,
                           epsilon_a=epsilon_composite, g_a=g_composite,
                           epsilon_b=epsilon_composite, g_b=g_composite))
    layers.append(dict(thickness_nm=height, fill_fraction=fill,
                       epsilon_a=1.95**2, g_a=0.,
                       epsilon_b=epsilon_composite, g_b=g_composite))
    if overetch:
        layers.append(dict(thickness_nm=overetch, fill_fraction=fill,
                           epsilon_a=substrate_index**2, g_a=0.,
                           epsilon_b=epsilon_composite, g_b=g_composite))
    return layers


def relative_rotation(jones, incident_axis):
    """Ellipse-axis change modulo 180 degrees relative to x (0) or y (1)."""
    ellipse = polarization_ellipse(jones)
    theta = ellipse['rotation_deg']
    if theta is not None:
        theta = float((theta - 90 * incident_axis + 90) % 180 - 90)
    return theta, ellipse['ellipticity_deg']


def transmission_metrics(result, incident_axis, *, min_power=.25, max_ellipticity_deg=5.):
    """Keep total diffraction flux distinct from usable zero-order power.

    Screening cutoffs are study choices, not user-specified device tolerances.
    Undefined near-circular axes cannot win the ranking.
    """
    field = result['transmission'][:, incident_axis]
    theta, ellipticity = relative_rotation(field, incident_axis)
    ratio = result['n_exit'] / result['n_incident']
    power = float(ratio * np.vdot(field, field).real)
    cross_power = float(ratio * abs(field[1-incident_axis])**2)
    qualified = bool(theta is not None and ellipticity is not None and
                     power >= min_power and abs(ellipticity) <= max_ellipticity_deg)
    return dict(polarization='TM_x' if incident_axis == 0 else 'TE_y',
                rotation_deg=theta, ellipticity_deg=ellipticity,
                zero_order_transmission=power, cross_power=cross_power,
                total_transmission=float(result['transmitted_power'][incident_axis]),
                total_reflection=float(result['reflected_power'][incident_axis]),
                absorption_flux_difference=float(result['absorbed_power'][incident_axis]),
                fom_deg=0. if theta is None else float(abs(theta)*np.sqrt(power)),
                qualified=qualified)


def stokes_band_average(jones_fields, weights, incident_axis):
    """Incoherent spectral averaging before extracting the polarization ellipse.

    Fields must already be multiplied by sqrt(n_out/n_in) for power. Averaging
    angles would be wrong, especially around a sharp resonance. Mixed spectra
    may be partially polarized: return the degree of polarization explicitly.
    """
    fields = np.asarray(jones_fields, complex)
    weights = np.asarray(weights, float)
    if fields.ndim != 2 or fields.shape[1] != 2 or weights.shape != (len(fields),):
        raise ValueError("fields must be (N,2) and weights (N,)")
    if (not np.isfinite(fields).all() or not np.isfinite(weights).all() or
            np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError("finite fields and nonnegative nonzero weights required")
    weights = weights / weights.sum()
    a, b = fields.T
    cross = np.conj(a)*b
    s = np.array([weights@(abs(a)**2+abs(b)**2), weights@(abs(a)**2-abs(b)**2),
                  weights@(2*cross.real), weights@(2*cross.imag)])
    polarized = float(np.linalg.norm(s[1:]))
    axis = None if np.hypot(s[1], s[2]) <= 1e-14*s[0] else float(
        (np.degrees(.5*np.arctan2(s[2], s[1]))-90*incident_axis+90) % 180-90)
    ellipticity = None if polarized == 0 else float(np.degrees(.5*np.arcsin(np.clip(s[3]/polarized, -1, 1))))
    return dict(power=float(s[0]), rotation_deg=axis, ellipticity_deg=ellipticity,
                degree_of_polarization=None if s[0] == 0 else polarized/float(s[0]),
                stokes=s.tolist())
