"""Write quantitative independent physical acceptance without requiring pytest.

Run from any directory: python validate_magneto.py --out NEW_DIRECTORY.
All materials are synthetic. Exact dipole-model agreement and continuum sphere
model error are reported separately; no universal 0.1% Maxwell accuracy is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy
from scipy.special import spherical_jn, spherical_yn
from threadpoolctl import threadpool_info, threadpool_limits

from magnetooptics import __version__
from magnetooptics.interaction import DenseGreenOperator, FFTGreenOperator
from magnetooptics.materials import gyrotropic_permittivity, sphere_polarizability
from magnetooptics.observables import circular_polarizations, cross_sections, plane_wave
from magnetooptics.solver import CoupledDipoleSystem


def relative_norm(actual, expected):
    denominator = float(np.linalg.norm(expected))
    if denominator == 0:
        raise ValueError("relative error requires a nonzero reference")
    return float(np.linalg.norm(np.asarray(actual) - expected) / denominator)


def scalar_sphere_alpha(epsilon, radius, wavelength, host_index=1.0):
    """Independent scalar electrostatic solution plus radiation reaction."""
    k = 2 * np.pi * host_index / wavelength
    alpha0 = 4 * np.pi * radius**3 * (epsilon - host_index**2) / (epsilon + 2 * host_index**2)
    return alpha0 / (1 - 1j * k**3 * alpha0 / (6 * np.pi))


def angular_radiation(positions, dipoles, k, n_mu=21, n_phi=47):
    """Independent transverse-field integral, without production far_field."""
    mu, weights = np.polynomial.legendre.leggauss(n_mu)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    total = 0.0
    for cosine, weight in zip(mu, weights):
        sine = np.sqrt(1 - cosine**2)
        directions = np.column_stack((sine * np.cos(phi), sine * np.sin(phi), np.full(n_phi, cosine)))
        for direction in directions:
            moment = np.sum(np.exp(-1j * k * (positions @ direction))[:, None] * dipoles, axis=0)
            transverse = moment - direction * np.dot(direction, moment)
            total += weight * (2 * np.pi / n_phi) * np.vdot(transverse, transverse).real
    return float(total * k**4 / (16 * np.pi**2))


def exact_mie(epsilon, radius, wavelength, nmax):
    """Independent outgoing h^(1) electric + magnetic multipole series.

    Bohren & Huffman equations 4.62 and 4.88. Only the well-conditioned small
    test range ka=0.02...0.08 is used, with an explicit truncation comparison.
    """
    k = 2 * np.pi / wavelength
    x = k * radius
    m = np.sqrt(complex(epsilon))
    orders = np.arange(1, nmax + 1)
    mx = m * x
    psi_x = x * spherical_jn(orders, x)
    dpsi_x = spherical_jn(orders, x) + x * spherical_jn(orders, x, True)
    psi_m = mx * spherical_jn(orders, mx)
    dpsi_m = spherical_jn(orders, mx) + mx * spherical_jn(orders, mx, True)
    xi_x = psi_x + 1j * x * spherical_yn(orders, x)
    dxi_x = dpsi_x + 1j * (spherical_yn(orders, x) + x * spherical_yn(orders, x, True))
    electric = (m * psi_m * dpsi_x - psi_x * dpsi_m) / (m * psi_m * dxi_x - xi_x * dpsi_m)
    magnetic = (psi_m * dpsi_x - m * psi_x * dpsi_m) / (psi_m * dxi_x - m * xi_x * dpsi_m)
    weights = 2 * orders + 1
    return np.array([
        2 * np.pi / k**2 * np.sum(weights * np.real(electric + magnetic)),
        2 * np.pi / k**2 * np.sum(weights * (np.abs(electric)**2 + np.abs(magnetic)**2)),
    ])


def sphere_and_circular_eigenchannels():
    records = []
    epsilon, g, radius, wavelength = 3 + 0.2j, 0.25 + 0.02j, 7.0, 650.0
    for host_index in (1.0, 1.33):
        k = 2 * np.pi * host_index / wavelength
        alpha = sphere_polarizability(gyrotropic_permittivity(epsilon, g), radius, wavelength, host_index)
        system = CoupledDipoleSystem(DenseGreenOperator([[0, 0, 0]], k), alpha, "dense")
        for helicity in (1, -1):
            polarization = np.array([1, helicity * 1j, 0]) / np.sqrt(2)
            expected = scalar_sphere_alpha(epsilon - helicity * g, radius, wavelength, host_index)
            result = system.solve(polarization[None], rtol=1e-12)
            sections = cross_sections(result.dipoles, polarization[None], result.local_field, k)
            reference_sections = np.array([k * expected.imag, k**4 * abs(expected)**2 / (6 * np.pi)])
            measured_sections = np.array([sections["extinction_nm2"], sections["scattering_nm2"]])
            errors = np.abs(measured_sections - reference_sections) / reference_sections
            records.append({
                "host_index": host_index, "helicity_sign": helicity,
                "eigenpolarizability_relative_error": relative_norm(alpha @ polarization, expected * polarization),
                "dipole_relative_error": relative_norm(result.dipoles[0], expected * polarization),
                "extinction_relative_error": float(errors[0]), "scattering_relative_error": float(errors[1]),
                "reference_extinction_nm2": float(reference_sections[0]),
                "reference_scattering_nm2": float(reference_sections[1]),
            })
    return {"inputs": {"epsilon_diagonal": [3, .2], "g": [.25, .02], "radius_nm": radius, "wavelength_nm": wavelength, "magnetization": [0, 0, 1]}, "cases": records}


def dimer_modes():
    # This prescribed scalar point polarizability is passive at the chosen k:
    # Im(alpha) > k^3 |alpha|^2/(6 pi). No continuum sphere claim is attached.
    separation, k, alpha = 28.0, 0.05, 20_000 + 4_000j
    positions = np.array([[0, 0, 0], [separation, 0, 0]])
    operators = {
        "dense": DenseGreenOperator(positions, k),
        "fft": FFTGreenOperator([[0, 0, 0], [1, 0, 0]], separation, k),
    }
    records = []
    for backend, operator in operators.items():
        system = CoupledDipoleSystem(operator, alpha * np.eye(3), "dense" if backend == "dense" else "gmres")
        for component in (0, 1, 2):
            kr = k * separation
            coupling = (2 * (1 - 1j * kr) if component == 0 else kr**2 - 1 + 1j * kr)
            coupling *= np.exp(1j * kr) / (4 * np.pi * separation**3)
            for parity in (1, -1):
                incident = np.zeros((2, 3), dtype=complex)
                incident[:, component] = [1, parity]
                expected = alpha * incident / (1 - parity * alpha * coupling)
                solved = system.solve(incident, rtol=1e-12)
                records.append({
                    "backend": backend, "component": "xyz"[component], "parity": parity,
                    "dipole_relative_error": relative_norm(solved.dipoles, expected),
                    "relative_residual": solved.relative_residual,
                })
    return {"inputs": {"separation_nm": separation, "k_nm_inv": k, "scalar_alpha_nm3": [alpha.real, alpha.imag]}, "cases": records}


def magnetic_symmetry():
    positions = np.array([[-30, -30, 0], [-30, 30, 0], [30, -30, 0], [30, 30, 0]])
    wavelength, k = 620.0, 2 * np.pi / 620.0
    operator = DenseGreenOperator(positions, k)
    channels = circular_polarizations([0, 0, 1])
    records = []
    for g in (0j, .30 + .03j):
        sections = []
        for sign in (1, -1):
            epsilon = gyrotropic_permittivity(2.8 + .15j, g, [0, 0, sign])
            alpha = sphere_polarizability(epsilon, 8.0, wavelength)
            system = CoupledDipoleSystem(operator, alpha, "dense")
            row = []
            for polarization in channels:
                incident = plane_wave(positions, wavelength, [0, 0, 1], polarization)
                solved = system.solve(incident, rtol=1e-12)
                row.append(cross_sections(solved.dipoles, incident, solved.local_field, k)["extinction_nm2"])
            sections.append(row)
        values = np.asarray(sections)
        scale = float(values.max())
        records.append({
            "g": [g.real, g.imag], "extinction_nm2_rows_Mplus_Mminus_columns_plus_minus": values.tolist(),
            "reversal_channel_swap_relative_error": float(np.max(np.abs(values[0] - values[1, ::-1])) / scale),
            "circular_difference_divided_by_max_extinction": float((values[0, 0] - values[0, 1]) / scale),
        })
    return {"inputs": {"positions_nm": positions.tolist(), "radius_nm": 8.0, "wavelength_nm": wavelength, "epsilon_diagonal": [2.8, .15]}, "cases": records}


def energy_closure():
    positions = np.array([[0, 0, 0], [41, 7, 12], [-29, 39, 3], [9, -46, 17], [14, 13, 53], [-20, -16, -37]])
    host_index, wavelength = 1.2, 500.0
    k = 2 * np.pi * host_index / wavelength
    direction = np.array([1, 2, 3]) / np.sqrt(14)
    polarization = circular_polarizations(direction)[0]
    incident = plane_wave(positions, wavelength, direction, polarization, host_index)
    operator = DenseGreenOperator(positions, k)
    records = []
    for loss in (0.0, 0.2):
        alpha = sphere_polarizability(gyrotropic_permittivity(3 + 1j * loss, .3), 8.0, wavelength, host_index)
        solved = CoupledDipoleSystem(operator, alpha, "dense").solve(incident, rtol=1e-12)
        sections = cross_sections(solved.dipoles, incident, solved.local_field, k)
        radiation = angular_radiation(positions, solved.dipoles, k)
        refined = angular_radiation(positions, solved.dipoles, k, 31, 61)
        records.append({
            "epsilon_diagonal": [3, loss], **sections,
            "independent_radiation_nm2": radiation,
            "angular_refinement_relative_change": abs(refined - radiation) / refined,
            "scattering_vs_angular_relative_error": abs(sections["scattering_nm2"] - refined) / refined,
            "energy_closure_divided_by_extinction": abs(sections["extinction_nm2"] - sections["absorption_nm2"] - refined) / sections["extinction_nm2"],
            "absorption_divided_by_extinction": sections["absorption_nm2"] / sections["extinction_nm2"],
        })
    return {"inputs": {"positions_nm": positions.tolist(), "radius_nm": 8.0, "wavelength_nm": wavelength, "host_index": host_index, "g": [.3, 0], "magnetization": [0, 0, 1], "direction": direction.tolist(), "quadrature_orders": [[21, 47], [31, 61]]}, "cases": records}


def mie_model_error():
    epsilon, wavelength = 2.25 + .1j, 600.0
    k = 2 * np.pi / wavelength
    incident = np.array([[1.0, 0, 0]], dtype=complex)
    records = []
    for ka in (.02, .08):
        radius = ka / k
        alpha = sphere_polarizability(epsilon, radius, wavelength)
        solved = CoupledDipoleSystem(DenseGreenOperator([[0, 0, 0]], k), alpha, "dense").solve(incident)
        sections = cross_sections(solved.dipoles, incident, solved.local_field, k)
        measured = np.array([sections["extinction_nm2"], sections["scattering_nm2"]])
        mie6, mie8 = exact_mie(epsilon, radius, wavelength, 6), exact_mie(epsilon, radius, wavelength, 8)
        errors = np.abs(measured - mie8) / mie8
        records.append({
            "ka": ka, "radius_nm": radius,
            "dipole_extinction_scattering_nm2": measured.tolist(), "mie_extinction_scattering_nm2": mie8.tolist(),
            "mie_order_6_to_8_relative_change": relative_norm(mie6, mie8),
            "extinction_relative_error": float(errors[0]), "scattering_relative_error": float(errors[1]),
            "both_within_0_1_percent": bool(np.all(errors < .001)),
            "included_in_0_1_percent_acceptance": ka == .02,
        })
    return {"inputs": {"epsilon": [epsilon.real, epsilon.imag], "wavelength_nm": wavelength, "host_index": 1.0},
            "interpretation": "ka=0.02 is the specified small-sphere physical acceptance case. ka=0.08 is a deliberate model-limit illustration; its error is reported, not hidden by linear-solver accuracy.", "cases": records}


def acceptance_report():
    started = perf_counter()
    with threadpool_limits(limits=1):
        sphere = sphere_and_circular_eigenchannels()
        dimer = dimer_modes()
        symmetry = magnetic_symmetry()
        energy = energy_closure()
        mie = mie_model_error()
        libraries = [{key: info.get(key) for key in ("internal_api", "prefix", "version", "num_threads")} for info in threadpool_info()]
    criteria = []
    def upper(name, value, tolerance):
        criteria.append({"name": name, "measured": float(value), "maximum": tolerance, "passed": bool(np.isfinite(value) and value <= tolerance)})
    upper("sphere_eigenpolarizability_relative_error", max(case["eigenpolarizability_relative_error"] for case in sphere["cases"]), 2e-12)
    upper("sphere_cross_section_relative_error", max(max(case["extinction_relative_error"], case["scattering_relative_error"]) for case in sphere["cases"]), 2e-10)
    upper("closed_dimer_dipole_relative_error", max(case["dipole_relative_error"] for case in dimer["cases"]), 2e-12)
    upper("dimer_relative_residual", max(case["relative_residual"] for case in dimer["cases"]), 1e-12)
    upper("magnetization_reversal_channel_error", max(case["reversal_channel_swap_relative_error"] for case in symmetry["cases"]), 2e-12)
    upper("nonmagnetic_circular_degeneracy", abs(symmetry["cases"][0]["circular_difference_divided_by_max_extinction"]), 2e-12)
    signal = abs(symmetry["cases"][1]["circular_difference_divided_by_max_extinction"])
    criteria.append({"name": "magnetic_signal_is_nontrivial", "measured": signal, "minimum": .001, "passed": bool(signal > .001)})
    upper("independent_angular_scattering_relative_error", max(case["scattering_vs_angular_relative_error"] for case in energy["cases"]), 5e-11)
    upper("angular_refinement_relative_change", max(case["angular_refinement_relative_change"] for case in energy["cases"]), 2e-12)
    upper("lossless_absorption_relative_to_extinction", abs(energy["cases"][0]["absorption_divided_by_extinction"]), 3e-11)
    absorption = energy["cases"][1]["absorption_nm2"]
    criteria.append({"name": "lossy_material_absorbs", "measured_nm2": absorption, "minimum": 0, "passed": bool(absorption > 0)})
    upper("Mie_series_order_convergence", max(case["mie_order_6_to_8_relative_change"] for case in mie["cases"]), 1e-13)
    upper("specified_ka_0_02_physical_error", max(mie["cases"][0]["extinction_relative_error"], mie["cases"][0]["scattering_relative_error"]), .001)
    source_paths = [Path(__file__).resolve()] + sorted((ROOT / "src" / "magnetooptics").glob("*.py"))
    return {
        "schema_version": "magnetooptics-acceptance-1.0", "package_version": __version__,
        "passed": all(check["passed"] for check in criteria), "criteria": criteria,
        "scope": "Finite electric point dipoles in a homogeneous real host; synthetic material values. Numerical dipole-model acceptance and limited continuum sphere acceptance are distinct.",
        "time_convention": "exp(-i omega t)", "units": {"length": "nm", "cross_section": "nm^2", "polarizability": "nm^3"},
        "single_sphere": sphere, "closed_dimer": dimer, "magnetic_symmetry": symmetry,
        "independent_energy_closure": energy, "Mie_model_error": mie,
        "references": ["https://doi.org/10.1364/JOSAA.11.001491", "https://miepython.readthedocs.io/en/3.0.0/07_algorithm.html"],
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "blas_threads": 1, "libraries": libraries},
        "elapsed_seconds": perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="fresh output directory")
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("use a fresh output directory; validation evidence is never overwritten")
    report = acceptance_report()
    serialized = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    args.out.mkdir(parents=True)
    path = args.out / "acceptance.json"
    path.write_bytes(serialized)
    print(json.dumps({"passed": report["passed"], "criteria": len(report["criteria"]), "output": str(path), "elapsed_seconds": report["elapsed_seconds"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
