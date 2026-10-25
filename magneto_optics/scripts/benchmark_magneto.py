"""Reproduce finite-array numerical accuracy and CPU performance evidence.

Run from any directory: python benchmark_magneto.py --out NEW_DIRECTORY.
Synthetic material parameters are a numerical benchmark, not a measured sample.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import sys
import threading
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import scipy
from scipy import fft
import psutil
from threadpoolctl import threadpool_limits, threadpool_info

from magnetooptics.interaction import DenseGreenOperator, FFTGreenOperator
from magnetooptics.solver import CoupledDipoleSystem
from magnetooptics.materials import gyrotropic_permittivity, sphere_polarizability
from magnetooptics.observables import circular_polarizations, plane_wave, cross_sections


class SampledRSS:
    """Process RSS sampled every 2 ms; short peaks can be missed."""
    def __init__(self):
        self.process = psutil.Process()
        self.stop = threading.Event()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline

    def __enter__(self):
        def sample():
            while not self.stop.wait(.002):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.worker = threading.Thread(target=sample, daemon=True)
        self.worker.start()
        return self

    def __exit__(self, *_):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.worker.join()


def inputs(shape):
    indices = np.indices(shape).reshape(3, -1).T
    positions = indices * 36.
    polarizations = circular_polarizations([0, 0, 1])
    fields = [plane_wave(positions, 500., [0, 0, 1], polarization) for polarization in polarizations]
    alphas = [sphere_polarizability(gyrotropic_permittivity(-2.5+.5j, .06+.01j, [0, 0, sign]), 6., 500.)
              for sign in (1, -1)]
    return indices, positions, fields, alphas


def single_run(shape, backend, *, rtol=1e-9):
    indices, positions, fields, alphas = inputs(shape)
    gc.collect()
    solutions, sections, details = [], [], []
    with SampledRSS() as memory:
        started = perf_counter()
        if backend == "fft_gmres":
            operator = FFTGreenOperator(indices, 36., 2*np.pi/500., max_memory_bytes=1024**3)
        else:
            operator = DenseGreenOperator(positions, 2*np.pi/500., max_memory_bytes=1024**3)
        operator_seconds = perf_counter() - started
        assembly_started = perf_counter()
        systems = [CoupledDipoleSystem(operator, alpha, "dense" if backend == "dense_lu" else "gmres")
                   for alpha in alphas]
        assembly_seconds = perf_counter() - assembly_started
        solve_seconds = 0.
        observable_seconds = 0.
        for magnetization, system in zip((1, -1), systems):
            for helicity, incident in zip((1, -1), fields):
                solve_start = perf_counter()
                result = system.solve(incident, rtol=rtol, restart=40, maxiter=200)
                solve_seconds += perf_counter() - solve_start
                observable_start = perf_counter()
                sections.append(cross_sections(result.dipoles, incident, result.local_field, operator.k_nm_inv))
                observable_seconds += perf_counter() - observable_start
                solutions.append(result.dipoles)
                details.append({"magnetization_z": magnetization, "helicity_index": helicity,
                                "iterations": result.iterations, "relative_residual": result.relative_residual,
                                "factorization_reused": result.metadata["factorization_reused"],
                                "solve_seconds": result.metadata["solve_seconds"]})
        total_seconds = perf_counter() - started
        operator_metadata = dict(operator.metadata)
        n = len(indices)
        if backend == "dense_lu":
            # Two magnetizations retain two LU factors while sharing one Green matrix.
            estimated = max(operator_metadata["estimated_peak_array_bytes"], 4*144*n*n+512*n)
        else:
            restart = min(40, 3*n)
            estimated = operator_metadata["estimated_peak_array_bytes"] + 16*3*n*(restart+8) + 16*(restart+1)**2
        record = {
            "backend": backend, "n_particles": n, "shape": list(shape),
            "centres_extent_nm": ((np.asarray(shape)-1)*36.).tolist(),
            "particle_envelope_extent_nm": ((np.asarray(shape)-1)*36.+12.).tolist(),
            "operator_seconds": operator_seconds, "system_assembly_seconds": assembly_seconds,
            "four_rhs_solve_seconds": solve_seconds, "observable_seconds": observable_seconds,
            "total_seconds": total_seconds, "rtol": rtol,
            "maximum_relative_residual": max(item["relative_residual"] for item in details),
            "iterations_total": sum(item["iterations"] for item in details),
            "rhs_details": details, "cross_sections": sections,
            "estimated_peak_owned_array_bytes": int(estimated), "operator": operator_metadata,
        }
    record.update(rss_baseline_bytes=memory.baseline, sampled_peak_rss_bytes=memory.peak,
                  sampled_peak_rss_increase_bytes=max(0, memory.peak-memory.baseline))
    return record, np.asarray(solutions), sections


def errors(actual_dipoles, actual_sections, reference_dipoles, reference_sections):
    actual_ext = np.asarray([s["extinction_nm2"] for s in actual_sections])
    reference_ext = np.asarray([s["extinction_nm2"] for s in reference_sections])
    scale = max(float(np.mean(np.abs(reference_ext))), np.finfo(float).tiny)
    odd = lambda ext: .5*((ext[0]-ext[1])-(ext[2]-ext[3]))
    per_section = {}
    for name in ("extinction_nm2", "absorption_nm2", "scattering_nm2"):
        actual = np.asarray([s[name] for s in actual_sections])
        reference = np.asarray([s[name] for s in reference_sections])
        per_section[name] = float(np.max(np.abs(actual-reference)/np.maximum(np.abs(reference), scale*1e-12)))
    difference = actual_dipoles-reference_dipoles
    dipole_error = np.linalg.norm(difference.reshape(4, -1), axis=1)/np.linalg.norm(reference_dipoles.reshape(4, -1), axis=1)
    result = {"max_relative_dipole_norm_error": float(np.max(dipole_error)),
              "max_relative_cross_section_error": per_section,
              "magnetic_odd_cd_error_over_mean_extinction": float(abs(odd(actual_ext)-odd(reference_ext))/scale),
              "reference_magnetic_odd_cd_over_mean_extinction": float(odd(reference_ext)/scale),
              "reference_mean_extinction_nm2": scale}
    result["within_0p1_percent_numerical_target"] = bool(max(
        result["max_relative_dipole_norm_error"], *per_section.values(),
        result["magnetic_odd_cd_error_over_mean_extinction"]) < .001)
    return result


def repeated(shape, backend, repeats):
    runs = []
    for _ in range(repeats):
        record, dipoles, sections = single_run(shape, backend)
        runs.append(record)
    timing_keys = ("operator_seconds", "system_assembly_seconds", "four_rhs_solve_seconds", "observable_seconds", "total_seconds")
    result = {"backend": backend, "repeat_count": repeats,
              "median_seconds": {key: statistics.median(run[key] for run in runs) for key in timing_keys},
              "runs": runs}
    return result, dipoles, sections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True, help="New output directory (must not exist).")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--quick", action="store_true", help="Reduced smoke benchmark; not full-size acceptance.")
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads must be positive")
    args.out.mkdir(parents=True, exist_ok=False)
    metadata = {
        "benchmark": "finite_gyrotropic_electric_dipole_arrays", "quick": args.quick,
        "material_kind": "synthetic passive tensor; not measured material data",
        "parameters": {"epsilon_diagonal": [-2.5, .5], "g": [.06, .01], "radius_nm": 6.,
                       "spacing_nm": 36., "vacuum_wavelength_nm": 500., "host_index": 1.},
        "four_rhs_order": ["+M,+helicity", "+M,-helicity", "-M,+helicity", "-M,-helicity"],
        "accuracy_target_fraction": .001, "threads_requested": args.threads,
        "memory_limit_bytes": 1024**3,
        "timing_scope": "warm import; includes Green setup, two material systems, four solves, scalar observables; excludes input generation, JSON I/O and tighter-tolerance refinement",
        "memory_scope": "owned-array conservative estimates and process RSS sampled every 2 ms; allocator caching can carry across cases and short peaks can be missed",
        "large_accuracy_scope": "tighter iterative tolerance comparison, not independent dense validation at large size",
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__,
                        "platform": platform.platform(), "processor": platform.processor(),
                        "physical_cpu_count": psutil.cpu_count(logical=False),
                        "logical_cpu_count": psutil.cpu_count(), "total_ram_bytes": psutil.virtual_memory().total,
                        "thread_environment": {key: os.environ.get(key) for key in
                            ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}},
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted([*ROOT.glob("src/magnetooptics/*.py"), Path(__file__).resolve(), ROOT/"pyproject.toml"])},
        "same_size_comparisons": [], "large_fft_cases": [],
    }
    def save():
        (args.out/"benchmark.json").write_text(json.dumps(metadata, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    with threadpool_limits(limits=args.threads), fft.set_workers(args.threads):
        metadata["environment"]["native_threadpools"] = threadpool_info()
        for backend in ("dense_lu", "dense_gmres", "fft_gmres"):
            single_run((2, 2, 1), backend)
        small = [(4, 4, 1), (8, 8, 1)] if args.quick else [(8, 8, 1), (16, 16, 1), (24, 24, 1)]
        for shape in small:
            entry = {"shape": list(shape), "n_particles": int(np.prod(shape)), "backends": {}}
            reference, reference_p, reference_c = repeated(shape, "dense_lu", 2 if args.quick else 3)
            entry["backends"]["dense_lu"] = reference
            for backend in ("dense_gmres", "fft_gmres"):
                record, p, c = repeated(shape, backend, 2 if args.quick else 3)
                record["accuracy_against_dense_lu"] = errors(p, c, reference_p, reference_c)
                entry["backends"][backend] = record
            entry["speedup_fft_over_dense_lu"] = reference["median_seconds"]["total_seconds"]/entry["backends"]["fft_gmres"]["median_seconds"]["total_seconds"]
            entry["speedup_fft_over_dense_gmres"] = entry["backends"]["dense_gmres"]["median_seconds"]["total_seconds"]/entry["backends"]["fft_gmres"]["median_seconds"]["total_seconds"]
            metadata["same_size_comparisons"].append(entry)
            save()
            print(json.dumps({"shape": shape, "fft_seconds": entry["backends"]["fft_gmres"]["median_seconds"]["total_seconds"],
                              "speedup_over_dense_lu": entry["speedup_fft_over_dense_lu"]}), flush=True)
        large = [(24, 24, 1)] if args.quick else [(64, 64, 1), (128, 128, 1), (24, 24, 24)]
        for shape in large:
            run, p, c = single_run(shape, "fft_gmres")
            refined, rp, rc = single_run(shape, "fft_gmres", rtol=1e-11)
            run["repeat_count"] = 1
            run["tolerance_refinement"] = {"reference_rtol": 1e-11, "reference_total_seconds": refined["total_seconds"],
                                           "reference_maximum_relative_residual": refined["maximum_relative_residual"],
                                           **errors(p, c, rp, rc)}
            metadata["large_fft_cases"].append(run)
            save()
            print(json.dumps({"shape": shape, "fft_seconds": run["total_seconds"],
                              "maximum_relative_residual": run["maximum_relative_residual"]}), flush=True)
    metadata["all_numerical_targets_passed"] = all(
        run["accuracy_against_dense_lu"]["within_0p1_percent_numerical_target"]
        for case in metadata["same_size_comparisons"] for backend, run in case["backends"].items() if backend != "dense_lu") and all(
        case["tolerance_refinement"]["within_0p1_percent_numerical_target"] for case in metadata["large_fft_cases"])
    save()
    if not metadata["all_numerical_targets_passed"]:
        raise SystemExit("At least one numerical accuracy target failed; inspect benchmark.json")
    print(f"Saved {args.out / 'benchmark.json'}", flush=True)


if __name__ == "__main__":
    main()
