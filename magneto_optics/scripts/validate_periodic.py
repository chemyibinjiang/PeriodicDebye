"""Independent numerical and conservation acceptance for a periodic electric sheet.

Synthetic homogeneous spheres, normal incidence, rectangular coplanar cell.
Numerical accuracy here is accuracy of the electric-dipole model, not a bound on
neglected particle multipoles or the accuracy of a real material data set.
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
from scipy.special import zeta
from threadpoolctl import threadpool_limits

from magnetooptics.materials import gyrotropic_permittivity, sphere_polarizability
from magnetooptics.periodic_geometry import RectangularCell
from magnetooptics.periodic_interaction import PeriodicGreenOperator
from magnetooptics.periodic_observables import evaluate_ports, jones_matrices, polarization_ellipse
from magnetooptics.solver import CoupledDipoleSystem


def relative_error(actual, expected):
    return float(np.linalg.norm(np.asarray(actual) - expected) / np.linalg.norm(expected))


def complex_json(value):
    array = np.asarray(value)
    return {"real": array.real.tolist(), "imaginary": array.imag.tolist()}


def scalar_alpha(epsilon, radius, k, host_epsilon=1.0):
    electrostatic = 4 * np.pi * radius**3 * (epsilon - host_epsilon) / (epsilon + 2 * host_epsilon)
    return electrostatic / (1 - 1j * k**3 * electrostatic / (6 * np.pi))


def gaussian_abel_square(dampings=(.06, .05, .04, .03), ka=1.3):
    """Independent retarded real-space sum, with Gaussian Abel extrapolation.

    Square symmetry gives Gxx=Gyy. Terms are multiplied by exp[-(eta*r/a)^2],
    cut where eta*|coordinate|>=6, then extrapolated cubically in eta^2.
    This is a converged independent numerical reference, not a rigorous bound.
    """
    values = []
    for damping in dampings:
        extent = int(np.ceil(6 / damping))
        x, y = np.meshgrid(np.arange(-extent, extent + 1), np.arange(-extent, extent + 1), indexing="ij")
        distance = np.sqrt(x*x + y*y)
        distance = distance[distance > 0]
        kr = ka * distance
        base = np.exp(1j * kr - (damping * distance)**2) / (4 * np.pi * distance**3)
        xx = np.sum(base * .5 * (kr**2 + 1 - 1j * kr))
        zz = np.sum(base * (kr**2 - 1 + 1j * kr))
        values.append([xx, zz])
    limit = np.polynomial.polynomial.polyfit(np.asarray(dampings)**2, np.asarray(values), 3)[0]
    return np.diag([limit[0], limit[0], limit[1]])


def lattice_references():
    a = 100.0
    beta = 4**(-1.5) * (zeta(1.5, .25) - zeta(1.5, .75))
    square_sum = 4 * zeta(1.5, 1) * beta
    static = square_sum / (8 * np.pi) * np.diag([1, 1, -2])
    slow = PeriodicGreenOperator([[0, 0, 0]], [a, a], 1e-4 / a, tolerance=1e-11)
    dynamic = PeriodicGreenOperator([[0, 0, 0]], [a, a], 1.3 / a, tolerance=1e-11)
    abel = gaussian_abel_square()
    refined = gaussian_abel_square((.04, .035, .03, .025))
    return {
        "period_nm": a, "static_ka": 1e-4, "dynamic_ka": 1.3,
        "square_inverse_cube_sum": float(square_sum),
        "static_reference_dimensionless": static.tolist(),
        "static_real_relative_error": relative_error(slow.matrix.real * a**3, static),
        "dynamic_ewald_dimensionless": complex_json(dynamic.matrix * a**3),
        "dynamic_abel_dimensionless": complex_json(refined),
        "abel_refinement_relative_change": relative_error(abel, refined),
        "dynamic_abel_relative_error": relative_error(dynamic.matrix * a**3, refined),
        "reference_note": "Static sum is 4*zeta(3/2)*DirichletBeta(3/2). Dynamic reference sums outgoing real-space fields with Gaussian damping and independently extrapolates damping to zero.",
    }


def independent_radiative_self(periods, k, bloch=(0, 0)):
    """All open plane-wave channels minus the isolated-dipole self radiation.

    This is derived from outgoing plane-wave flux, independent of Ewald.
    The +/-z average removes mixed xz/yz entries for a coplanar self block.
    """
    periods, bloch = np.asarray(periods), np.asarray(bloch)
    area = float(np.prod(periods))
    bounds = np.ceil((k + np.abs(bloch)) * periods / (2 * np.pi)).astype(int) + 1
    result = -k**3 / (6 * np.pi) * np.eye(3)
    count = 0
    for m in range(-bounds[0], bounds[0] + 1):
        for n in range(-bounds[1], bounds[1] + 1):
            transverse = bloch + 2 * np.pi * np.array([m, n]) / periods
            kz_squared = k*k - np.dot(transverse, transverse)
            if kz_squared <= 0:
                continue
            kz = np.sqrt(kz_squared)
            kxy = np.array([*transverse, 0.0])
            result += (k*k * np.eye(3) - np.outer(kxy, kxy) - np.diag([0, 0, kz_squared])) / (2 * area * kz)
            count += 1
    return result, count


def radiative_identity():
    records = []
    for periods, wavelength, q_fraction in [([180, 220], 650, [0, 0]), ([900, 760], 500, [0, 0]), ([900, 760], 500, [.13, .09])]:
        k = 2 * np.pi / wavelength
        q = k * np.asarray(q_fraction)
        operator = PeriodicGreenOperator([[0, 0, 0]], periods, k, tolerance=1e-11, bloch_nm_inv=q)
        expected, count = independent_radiative_self(periods, k, q)
        records.append({"periods_nm": periods, "wavelength_nm": wavelength, "bloch_nm_inv": q.tolist(), "open_orders": count,
                        "relative_error": relative_error(operator.matrix.imag, expected)})
    return records


def solve_ports(cell, wavelength, epsilon, radius, polarization, operator=None):
    k = 2 * np.pi / wavelength
    if operator is None:
        operator = PeriodicGreenOperator(cell.positions_nm, cell.periods_nm, k, tolerance=1e-11)
    alpha = sphere_polarizability(epsilon, radius, wavelength)
    incident = np.broadcast_to(polarization, (len(cell.positions_nm), 3)).astype(complex)
    solved = CoupledDipoleSystem(operator, alpha, "dense").solve(incident, rtol=1e-12)
    ports = evaluate_ports(cell, solved.dipoles, incident, solved.local_field, wavelength)
    return ports, solved


def circular_sheet():
    cell = RectangularCell([200, 200], [[0, 0, 0]], 8)
    wavelength, k = 620.0, 2 * np.pi / 620.0
    operator = PeriodicGreenOperator(cell.positions_nm, cell.periods_nm, k, tolerance=1e-11)
    epsilon_d, g = -1.8 + .18j, .08 + .01j
    records, matrices = [], []
    for magnetization_sign in (1, -1):
        epsilon = gyrotropic_permittivity(epsilon_d, g, [0, 0, magnetization_sign])
        channels = []
        for helicity in (1, -1):
            polarization = np.array([1, 1j * helicity, 0]) / np.sqrt(2)
            alpha = scalar_alpha(epsilon_d - helicity * magnetization_sign * g, 8, k)
            effective = alpha / (1 - alpha * operator.matrix[0, 0])
            r = 1j * k * effective / (2 * cell.area_nm2)
            t = 1 + r
            ports, solved = solve_ports(cell, wavelength, epsilon, 8, polarization, operator)
            records.append({"magnetization_sign": magnetization_sign, "helicity": helicity,
                            "dipole_relative_error": relative_error(solved.dipoles[0], effective * polarization),
                            "transmitted_field_relative_error": relative_error(ports["zero_transmitted"], t * polarization),
                            "reflected_field_relative_error": relative_error(ports["zero_reflected"], r * polarization),
                            "transmission": ports["total_transmission"], "reflection": ports["total_reflection"], "absorption": ports["absorption"]})
            channels.append((t, r))
        x_ports, _ = solve_ports(cell, wavelength, epsilon, 8, [1, 0, 0], operator)
        matrices.append(jones_matrices(x_ports, solve_ports(cell, wavelength, epsilon, 8, [0, 1, 0], operator)[0]))
        for column, name in [(0, "transmitted"), (1, "reflected")]:
            plus, minus = channels[0][column], channels[1][column]
            expected_rotation = np.degrees(.5 * np.angle(minus / plus))
            expected_ellipticity = np.degrees(.5 * np.arcsin((abs(plus)**2 - abs(minus)**2) / (abs(plus)**2 + abs(minus)**2)))
            ellipse = polarization_ellipse(x_ports[f"zero_{name}"][:2])
            rotation_error = abs((ellipse["rotation_deg"] - expected_rotation + 90) % 180 - 90)
            records.append({"magnetization_sign": magnetization_sign, "port": name,
                            "rotation_deg": ellipse["rotation_deg"], "ellipticity_deg": ellipse["ellipticity_deg"],
                            "rotation_error_deg": rotation_error,
                            "ellipticity_error_deg": abs(ellipse["ellipticity_deg"] - expected_ellipticity)})
    reverse_error = max(relative_error(matrices[0][key], matrices[1][key].T) for key in ("transmission", "reflection"))
    no_mo = gyrotropic_permittivity(epsilon_d, 0)
    a = solve_ports(cell, wavelength, no_mo, 8, np.array([1, 1j, 0]) / np.sqrt(2), operator)[0]
    b = solve_ports(cell, wavelength, no_mo, 8, np.array([1, -1j, 0]) / np.sqrt(2), operator)[0]
    return {"inputs": {"periods_nm": [200, 200], "radius_nm": 8, "wavelength_nm": wavelength, "epsilon_diagonal": [-1.8, .18], "g": [.08, .01]},
            "cases": records, "Onsager_Jones_transpose_relative_error": reverse_error,
            "g_zero_circular_power_difference": max(abs(a[key] - b[key]) for key in ("total_transmission", "total_reflection", "absorption")),
            "note": "Closed scalar sheet equations use the independently validated Ewald lattice S as input: p=alpha/(1-alpha*S); r=i*k*p/(2A), t=1+r. Angle references come independently from the two circular eigenphases and magnitudes."}


def all_order_energy():
    periods, wavelength, radius = [900, 760], 500.0, 14.0
    positions = [[80, 93, 0], [400, 350, 0]]
    cell = RectangularCell(periods, positions, radius)
    k = 2 * np.pi / wavelength
    operator = PeriodicGreenOperator(positions, periods, k, tolerance=1e-11)
    records = []
    for loss in (0, .2):
        epsilon = gyrotropic_permittivity(-1.8 + 1j * loss, .05)
        ports, solved = solve_ports(cell, wavelength, epsilon, radius, [1, 0, 0], operator)
        # Work dissipated in the quasistatic material, without port radiation or
        # production cross_sections/local_field, provides an absorption oracle.
        inverse_alpha0 = (epsilon + 2 * np.eye(3)) @ np.linalg.inv(4 * np.pi * radius**3 * (epsilon - np.eye(3)))
        independent_absorption = -k / cell.area_nm2 * np.imag(np.vdot(solved.dipoles, solved.dipoles @ inverse_alpha0.T))
        zero = ports["zero_order_index"]
        nonzero_power = ports["total_transmission"] + ports["total_reflection"] - ports["transmitted_power"][zero] - ports["reflected_power"][zero]
        records.append({"material_loss": loss, "open_orders_per_hemisphere": len(ports["order_indices"]),
                        "transmission": ports["total_transmission"], "reflection": ports["total_reflection"], "absorption": ports["absorption"],
                        "independent_material_absorption": float(independent_absorption),
                        "absorption_absolute_error": abs(ports["absorption"] - independent_absorption),
                        "energy_balance_absolute_error": abs(ports["energy_balance_residual"]),
                        "nonzero_order_power": float(nonzero_power), "order_indices": ports["order_indices"].tolist()})
    return {"inputs": {"periods_nm": periods, "wavelength_nm": wavelength, "radius_nm": radius, "positions_nm": positions, "epsilon_diagonal_real": -1.8, "g": .05}, "cases": records}


def supercell_invariance():
    period, wavelength, radius = np.array([200.0, 240.0]), 620.0, 8.0
    offset = np.array([37.0, 53.0, 0])
    primitive = RectangularCell(period, offset[None], radius)
    indices = np.indices((2, 3)).reshape(2, -1).T
    positions = np.column_stack((indices * period, np.zeros(len(indices)))) + offset
    supercell = RectangularCell(period * [2, 3], positions, radius)
    epsilon = gyrotropic_permittivity(-1.8 + .18j, .08 + .01j)
    reference, p = solve_ports(primitive, wavelength, epsilon, radius, [1, 0, 0])
    expanded, ps = solve_ports(supercell, wavelength, epsilon, radius, [1, 0, 0])
    zero = expanded["zero_order_index"]
    mask = np.arange(len(expanded["order_indices"])) != zero
    return {"primitive_periods_nm": period.tolist(), "supercell_periods_nm": supercell.periods_nm.tolist(), "basis_sites": len(positions),
            "wavelength_nm": wavelength, "open_orders_primitive": len(reference["order_indices"]), "open_orders_supercell": len(expanded["order_indices"]),
            "dipole_relative_error": relative_error(ps.dipoles, np.broadcast_to(p.dipoles, ps.dipoles.shape)),
            "zero_transmitted_relative_error": relative_error(expanded["zero_transmitted"], reference["zero_transmitted"]),
            "zero_reflected_relative_error": relative_error(expanded["zero_reflected"], reference["zero_reflected"]),
            "total_power_absolute_error": max(abs(expanded[key] - reference[key]) for key in ("total_transmission", "total_reflection", "absorption")),
            "spurious_folded_order_power": float(np.sum(expanded["transmitted_power"][mask] + expanded["reflected_power"][mask]))}


def acceptance_report():
    started = perf_counter()
    with threadpool_limits(limits=1):
        lattice = lattice_references()
        radiative = radiative_identity()
        sheet = circular_sheet()
        energy = all_order_energy()
        supercell = supercell_invariance()
    criteria = []
    def upper(name, value, tolerance):
        criteria.append({"name": name, "measured": float(value), "maximum": tolerance, "passed": bool(np.isfinite(value) and value <= tolerance)})
    upper("square_static_zeta_reference", lattice["static_real_relative_error"], 2e-7)
    upper("independent_Abel_refinement", lattice["abel_refinement_relative_change"], 1e-7)
    upper("dynamic_Ewald_vs_independent_Abel", lattice["dynamic_abel_relative_error"], 1e-7)
    upper("radiative_self_all_open_channels", max(case["relative_error"] for case in radiative), 2e-10)
    field_errors = [case[key] for case in sheet["cases"] for key in ("dipole_relative_error", "transmitted_field_relative_error", "reflected_field_relative_error") if key in case]
    upper("closed_circular_sheet_fields", max(field_errors), 2e-10)
    upper("Faraday_Kerr_eigenchannel_angles_deg", max(case[key] for case in sheet["cases"] for key in ("rotation_error_deg", "ellipticity_error_deg") if key in case), 1e-9)
    upper("Onsager_reversal_Jones", sheet["Onsager_Jones_transpose_relative_error"], 2e-10)
    upper("g_zero_circular_degeneracy", sheet["g_zero_circular_power_difference"], 1e-12)
    upper("all_order_energy_conservation", max(case["energy_balance_absolute_error"] for case in energy["cases"]), 2e-10)
    upper("independent_material_absorption", max(case["absorption_absolute_error"] for case in energy["cases"]), 2e-10)
    nonzero = min(case["nonzero_order_power"] for case in energy["cases"])
    criteria.append({"name": "diffracted_orders_carry_nonzero_power", "measured": nonzero, "minimum": 1e-8, "passed": bool(nonzero > 1e-8)})
    upper("primitive_supercell_field_invariance", max(supercell[key] for key in ("dipole_relative_error", "zero_transmitted_relative_error", "zero_reflected_relative_error")), 2e-9)
    upper("primitive_supercell_power_invariance", supercell["total_power_absolute_error"], 2e-10)
    upper("folded_orders_cancel", supercell["spurious_folded_order_power"], 1e-16)
    paths = [Path(__file__).resolve()] + sorted((ROOT / "src" / "magnetooptics").glob("*.py"))
    return {"schema_version": "periodic-magnetooptics-acceptance-1.0", "passed": all(item["passed"] for item in criteria), "criteria": criteria,
            "scope": "Infinite rectangular coplanar electric-dipole sheet in a homogeneous lossless host; normal-incidence ports. Nonzero Bloch wavevector is tested only for the Green radiative identity. Accuracy is within the specified dipole model, not universal continuum Maxwell accuracy.",
            "lattice_references": lattice, "radiative_self_identity": radiative, "circular_sheet": sheet, "all_order_energy": energy, "supercell_invariance": supercell,
            "references": ["https://arxiv.org/abs/1703.00326", "https://doi.org/10.1103/PhysRevB.88.245109"],
            "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "blas_threads": 1},
            "elapsed_seconds": perf_counter() - started}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("use a fresh output directory")
    report = acceptance_report()
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    args.out.mkdir(parents=True)
    (args.out / "acceptance.json").write_bytes(data)
    print(json.dumps({"passed": report["passed"], "criteria": len(report["criteria"]), "seconds": report["elapsed_seconds"], "output": str(args.out / "acceptance.json")}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
