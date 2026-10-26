"""Measure periodic Ewald reference versus vectorization/displacement caching.

Default: scalar reference, vectorized without pair caching, and cached runs for
16/36 basis spheres, three repetitions; 64/144/256 cached runs with tighter,
altered-split verification.
This is a structured-supercell benchmark. Unstructured bases may have little
displacement reuse. Every timing is measured; no large-case timing is extrapolated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy
from threadpoolctl import threadpool_info, threadpool_limits

from magnetooptics.materials import gyrotropic_permittivity, sphere_polarizability
from magnetooptics.periodic_geometry import RectangularCell
from magnetooptics.periodic_interaction import PeriodicGreenOperator
from magnetooptics.periodic_observables import evaluate_ports, jones_matrices, polarization_ellipse
from magnetooptics.solver import CoupledDipoleSystem


def inputs(side):
    indices = np.indices((side, side)).reshape(2, -1).T
    spacing = np.array([36.0, 41.0])
    positions = np.column_stack((indices * spacing + [7.0, 11.0], np.zeros(len(indices))))
    cell = RectangularCell(side * spacing, positions, 6.0)
    # Distinct site response prevents this larger basis being only a relabelled
    # single-particle unit cell; positions still permit exact displacement reuse.
    diagonal = -2.5 + .15 * np.sin(.7 * indices[:, 0] + .9 * indices[:, 1]) + .5j
    coupling = .06 + .01j
    return cell, diagonal, coupling


def run_case(side, backend, *, tolerance=1e-10, eta_factor=1.0):
    cell, diagonal, coupling = inputs(side)
    wavelength, k = 620.0, 2 * np.pi / 620.0
    eta = eta_factor * np.sqrt(np.pi / cell.area_nm2)
    started = perf_counter()
    before = perf_counter()
    operator = PeriodicGreenOperator(cell.positions_nm, cell.periods_nm, k, tolerance=tolerance, backend=backend, ewald_eta_nm_inv=eta)
    setup_seconds = perf_counter() - before
    solver_seconds = material_seconds = port_seconds = 0.0
    dipoles, powers, jones, angles, solves = [], [], [], [], []
    max_energy_error = 0.0
    for sign in (1, -1):
        before = perf_counter()
        epsilon = np.array([gyrotropic_permittivity(value, coupling, [0, 0, sign]) for value in diagonal])
        alpha = sphere_polarizability(epsilon, 6.0, wavelength)
        system = CoupledDipoleSystem(operator, alpha, "dense")
        material_seconds += perf_counter() - before
        port_pair = []
        for polarization in ([1, 0, 0], [0, 1, 0]):
            incident = np.broadcast_to(polarization, (len(diagonal), 3)).astype(complex)
            before = perf_counter()
            result = system.solve(incident, rtol=1e-11)
            solver_seconds += perf_counter() - before
            before = perf_counter()
            ports = evaluate_ports(cell, result.dipoles, incident, result.local_field, wavelength)
            port_seconds += perf_counter() - before
            max_energy_error = max(max_energy_error, abs(ports["energy_balance_residual"]))
            dipoles.append(result.dipoles)
            powers.append([ports["total_transmission"], ports["total_reflection"], ports["absorption"]])
            solves.append({"magnetization_sign": sign, "polarization": "x" if polarization[0] else "y", "relative_residual": result.relative_residual,
                           "factorization_reused": result.metadata["factorization_reused"], "solve_seconds": result.metadata["solve_seconds"]})
            port_pair.append(ports)
        matrices = jones_matrices(*port_pair)
        jones.extend([matrices["transmission"], matrices["reflection"]])
        for port in ("zero_transmitted", "zero_reflected"):
            ellipse = polarization_ellipse(port_pair[0][port][:2])
            angles.append([ellipse["rotation_deg"], ellipse["ellipticity_deg"]])
    total_seconds = perf_counter() - started
    record = {
        "side": side, "basis_spheres": side**2, "backend": backend,
        "periods_nm": cell.periods_nm.tolist(), "wavelength_nm": wavelength,
        "tolerance": tolerance, "eta_factor": eta_factor,
        "setup_seconds": setup_seconds, "solve_four_rhs_seconds": solver_seconds,
        "material_preparation_seconds": material_seconds, "ports_seconds": port_seconds,
        "total_seconds": total_seconds, "max_energy_balance_absolute_error": max_energy_error,
        "operator": operator.metadata, "solves": solves,
        "powers_rows_Mplus_x_Mplus_y_Mminus_x_Mminus_y_columns_T_R_A": powers,
    }
    arrays = {"green": operator.matrix, "dipoles": np.asarray(dipoles), "powers": np.asarray(powers), "jones": np.asarray(jones), "angles": angles}
    return record, arrays


def compare(actual, reference):
    def relative(a, b):
        return float(np.linalg.norm(a - b) / np.linalg.norm(b))
    angle_errors = []
    for row, target in zip(actual["angles"], reference["angles"]):
        for index, (value, expected) in enumerate(zip(row, target)):
            if value is None or expected is None:
                if value is not expected:
                    raise ValueError("polarization angle changed between defined and undefined")
                continue
            angle_errors.append(abs((value - expected + 90) % 180 - 90) if index == 0 else abs(value - expected))
    metrics = {
        "green_relative_error": relative(actual["green"], reference["green"]),
        "max_dipole_relative_error": max(relative(a, b) for a, b in zip(actual["dipoles"], reference["dipoles"])),
        "max_T_R_A_absolute_error": float(np.max(np.abs(actual["powers"] - reference["powers"]))),
        "max_Jones_relative_error": max(relative(a, b) for a, b in zip(actual["jones"], reference["jones"])),
        "max_polarization_angle_absolute_error_deg": max(angle_errors, default=0.0),
    }
    limits = {"green_relative_error": 2e-9, "max_dipole_relative_error": 2e-9, "max_T_R_A_absolute_error": 2e-10, "max_Jones_relative_error": 2e-9, "max_polarization_angle_absolute_error_deg": 1e-7}
    return {"metrics": metrics, "limits": limits, "passed": all(np.isfinite(value) and value <= limits[name] for name, value in metrics.items())}


def summarize(records):
    fields = ("setup_seconds", "solve_four_rhs_seconds", "material_preparation_seconds", "ports_seconds", "total_seconds")
    return {key: {"median": statistics.median(row[key] for row in records), "minimum": min(row[key] for row in records), "maximum": max(row[key] for row in records)} for key in fields}


def benchmark(reference_sides=(4, 6), cached_sides=(8, 12, 16), repeats=3):
    started = perf_counter()
    paired, large = [], []
    with threadpool_limits(limits=1):
        run_case(1, "cached")  # Explicit warm-up, excluded from reported case timings.
        for side in reference_sides:
            baseline, vectorized, accelerated, comparisons = [], [], [], []
            for repeat in range(repeats):
                backends = ("reference", "vectorized", "cached")
                shift = repeat % len(backends)
                order = backends[shift:] + backends[:shift]
                pair = {backend: run_case(side, backend) for backend in order}
                baseline.append(pair["reference"][0])
                vectorized.append(pair["vectorized"][0])
                accelerated.append(pair["cached"][0])
                for target in ("reference", "vectorized"):
                    comparison = compare(pair["cached"][1], pair[target][1])
                    comparison.update({"repeat": repeat, "comparison": f"cached_vs_{target}"})
                    comparisons.append(comparison)
            reference_summary, vectorized_summary, cached_summary = summarize(baseline), summarize(vectorized), summarize(accelerated)
            paired.append({"basis_spheres": side**2, "reference_runs": baseline, "vectorized_runs": vectorized, "cached_runs": accelerated,
                           "reference_summary": reference_summary, "vectorized_summary": vectorized_summary, "cached_summary": cached_summary,
                           "median_setup_speedup_vs_scalar_reference": reference_summary["setup_seconds"]["median"] / cached_summary["setup_seconds"]["median"],
                           "median_total_speedup_vs_scalar_reference": reference_summary["total_seconds"]["median"] / cached_summary["total_seconds"]["median"],
                           "median_setup_speedup_vs_vectorized": vectorized_summary["setup_seconds"]["median"] / cached_summary["setup_seconds"]["median"],
                           "median_total_speedup_vs_vectorized": vectorized_summary["total_seconds"]["median"] / cached_summary["total_seconds"]["median"],
                           "comparisons": comparisons})
            print(f"paired basis={side**2}: passed={all(item['passed'] for item in comparisons)}", flush=True)
        for side in cached_sides:
            measured, arrays = run_case(side, "cached")
            verification, refined = run_case(side, "cached", tolerance=1e-12, eta_factor=1.25)
            comparison = compare(arrays, refined)
            large.append({"basis_spheres": side**2, "measured": measured, "verification": verification, "comparison": comparison,
                          "reference_scope": "same cached Ewald implementation with tighter requested cutoff convergence and a different splitting parameter; not an independent scalar reference"})
            print(f"cached basis={side**2}: passed={comparison['passed']}", flush=True)
        libraries = [{key: info.get(key) for key in ("internal_api", "prefix", "version", "num_threads")} for info in threadpool_info()]
    comparisons = [item for group in paired for item in group["comparisons"]] + [group["comparison"] for group in large]
    all_records = [item for group in paired for key in ("reference_runs", "vectorized_runs", "cached_runs") for item in group[key]] + [group[key] for group in large for key in ("measured", "verification")]
    energy_error = max(record["max_energy_balance_absolute_error"] for record in all_records)
    paths = [Path(__file__).resolve()] + sorted((ROOT / "src" / "magnetooptics").glob("*.py"))
    return {
        "schema_version": "periodic-magnetooptics-benchmark-1.0",
        "passed": bool(all(item["passed"] for item in comparisons) and energy_error <= 2e-10),
        "paired_reference_cached": paired, "larger_cached": large,
        "max_energy_balance_absolute_error": energy_error, "energy_acceptance_absolute_tolerance": 2e-10,
        "benchmark_inputs": {"reference_sides": list(reference_sides), "cached_sides": list(cached_sides), "repeats": repeats,
                             "spacing_nm": [36, 41], "offset_nm": [7, 11], "sphere_radius_nm": 6,
                             "epsilon_diagonal_formula": "-2.5+0.15*sin(0.7*ix+0.9*iy)+0.5j", "g": [.06, .01], "wavelength_nm": 620},
        "method": "Three matched-accuracy paths: scalar pair-by-pair Ewald reference; vectorized lattice evaluation without pair caching; vectorized evaluation with exact repeated-displacement caching. Speedups against the vectorized baseline isolate the cache benefit. Same physical dipole model, four linear solves per wavelength for x/y and +/-M. Dense LU reused for the second polarization at each M.",
        "timing_scope": "Setup includes matrix construction and cutoff convergence verification. Four-RHS solve includes two LU factorizations and four residual checks. Materials and diffraction ports timed separately. Total excludes geometry preparation, imports, JSON I/O, and warm-up; raw repetitions retained. One BLAS thread.",
        "limits": "Structured grids yield many repeated displacements. Unstructured positions can lose most cache benefit. Dense matrix remains O(N^2) storage and LU remains O(N^3); large-case scalar times are not extrapolated. Array/workspace memory estimates are not measured process peak RSS. Small relative errors do not bound continuum multipole error or a vanishing magneto-optical signal.",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "blas_threads": 1, "libraries": libraries},
        "elapsed_seconds": perf_counter() - started,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reference-sides", type=int, nargs="+", default=[4, 6])
    parser.add_argument("--cached-sides", type=int, nargs="+", default=[8, 12, 16])
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError("use a fresh output directory")
    if args.repeats < 1 or any(value < 1 for value in args.reference_sides + args.cached_sides):
        parser.error("sides and repeats must be positive integers")
    report = benchmark(args.reference_sides, args.cached_sides, args.repeats)
    data = (json.dumps(report, indent=2, allow_nan=False) + "\n").encode("utf-8")
    args.out.mkdir(parents=True)
    (args.out / "benchmark.json").write_bytes(data)
    print(json.dumps({"passed": report["passed"], "output": str(args.out / "benchmark.json"), "seconds": report["elapsed_seconds"]}, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
