#!/usr/bin/env python3
"""Run resolved finite MOF-5 PXRD, cache reuse, and optional accuracy checks.

Example: python scripts/benchmark_large_powder.py --cells 40 --threads 6 --out RUN
The output directory must be new. No instrumental broadening is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / 'src'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cells', type=int, default=40)
    parser.add_argument('--points', type=int, default=6001)
    parser.add_argument('--qmax', type=float, default=6.)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--bin-width', type=float, default=.04)
    parser.add_argument('--order', type=int, default=8)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--refine', action='store_true')
    parser.add_argument('--exact-q', type=float, nargs='*', default=[])
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.cells < 1 or args.points < 3 or args.repeats < 1:
        parser.error('cells/repeats must be positive and points >= 3')
    if (not math.isfinite(args.qmax)
            or args.qmax*(args.points-2)/(args.points-1) < .1):
        parser.error('q grid must contain at least two points at or above 0.1 inverse angstrom')
    if any(not math.isfinite(value) or not 0 <= value <= args.qmax for value in args.exact_q):
        parser.error('exact-q values must be finite and within the requested q range')
    args.out.mkdir(parents=True, exist_ok=False)
    import numpy as np
    from threadpoolctl import threadpool_limits
    from mofdebye.core import AtomCloud
    from mofdebye.crystal import read_cif
    from mofdebye.large_powder import LargePowderCache
    from mofdebye.runner import environment_record
    from validate_large_powder import compare_powder_curves

    threadpool_limits(limits=1)
    cif = PROJECT/'examples'/'real_mof'/'IRMOF-1.cif'
    crystal = read_cif(cif)
    mask = np.ones((args.cells,)*3, dtype=int)
    q = np.linspace(0, args.qmax, args.points)
    environment = environment_record()
    environment['logical_cpus'] = os.cpu_count()
    config = dict(schema_version='1.1', description='Resolved finite MOF-5 powder pattern; deterministic complete cells.',
                  scattering_case='pxrd', boundary='finite', guest_estimator='explicit',
                  backend='radial_moments', device='cpu',
                  host=dict(cif=str(cif.resolve()), cell_mask=mask.tolist()),
                  radial=dict(bin_width_A=args.bin_width, order=args.order, threads=args.threads),
                  sampling=dict(q=dict(min=0, max=args.qmax, points=args.points)))
    dump(args.out/'config.json', config)
    sources = {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted((PROJECT/'src'/'mofdebye').glob('*.py'))}
    sources['scripts/benchmark_large_powder.py'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    dump(args.out/'source_sha256.json', sources)
    # Warm compilation on a tiny geometry; retain this cost separately.
    start = time.perf_counter()
    warm = LargePowderCache(AtomCloud(('C',), [[0,0,0]]), np.eye(3)*5,
                           np.ones((2,2,2),dtype=int), np.linspace(0,6,61),
                           threads=args.threads, bin_width=args.bin_width, order=args.order)
    warm.iq()
    warmup_seconds = time.perf_counter()-start
    del warm
    start = time.perf_counter()
    cache = LargePowderCache(crystal.basis, crystal.cell_vectors_A, mask, q,
                            bin_width=args.bin_width, order=args.order, threads=args.threads)
    preparation_seconds = time.perf_counter()-start
    calls = []
    for _ in range(args.repeats):
        start = time.perf_counter()
        values, bound = cache.iq()
        calls.append(time.perf_counter()-start)
    np.savez_compressed(args.out/'pattern.npz', q=q, intensity=values, approximation_bound=bound)
    np.savetxt(args.out/'pattern.csv', np.column_stack((q, values, bound)), delimiter=',',
               header='q_A_inv,intensity_electron_squared,approximation_bound_excluding_roundoff', comments='')
    region = q >= .1
    peak = float(np.max(abs(values[region])))
    active = region & (abs(values) >= .01*peak)
    summary = dict(environment=environment, cif_sha256=hashlib.sha256(cif.read_bytes()).hexdigest(),
                   cells_per_edge=args.cells, represented_atoms=len(crystal.basis)*int(mask.sum()),
                   edge_nm=float(np.linalg.norm(crystal.cell_vectors_A[0])*args.cells/10),
                   q_points=len(q), q_step=float(q[1]-q[0]), q_max=float(q[-1]),
                   warmup_seconds=warmup_seconds, preparation_seconds=preparation_seconds,
                   reuse_seconds=calls, first_pattern_seconds=preparation_seconds+calls[0],
                   metadata=cache.metadata,
                   approximation_bound_peak_fraction=float(np.max(bound[region])/peak),
                   approximation_bound_significant_point_fraction=float(np.max(bound[active]/abs(values[active]))),
                   bound_caveat='Truncation and offset grouping only; floating-point error is validated empirically.',
                   target_fraction=.001, nonforward_q_min=.1, significant_point_floor_fraction=.01)
    try:
        import psutil
        summary['memory_after_main_pattern'] = psutil.Process().memory_info()._asdict()
    except ImportError:
        pass
    dump(args.out/'summary.json', summary)
    print(json.dumps({k:summary[k] for k in ('represented_atoms','edge_nm','first_pattern_seconds','reuse_seconds')}, indent=2), flush=True)
    if args.exact_q:
        requested = np.asarray(args.exact_q)
        indices = np.unique(np.argmin(abs(q[:,None]-requested[None,:]), axis=0))
        start = time.perf_counter()
        reference = cache.exact_selected(q[indices], merged=False)
        elapsed = time.perf_counter()-start
        errors = abs(values[indices]-reference)
        check = dict(q=q[indices].tolist(), reference=reference.tolist(), actual=values[indices].tolist(),
                     relative_error=(errors/np.maximum(abs(reference),np.finfo(float).tiny)).tolist(),
                     peak_scaled_error=(errors/peak).tolist(), seconds=elapsed,
                     reference_kind='Unbinned sinc sum, original unmerged canonical displacements; no radial interpolation.')
        summary['selected_exact_check'] = check
        np.savez_compressed(args.out/'selected_exact.npz', q=q[indices], reference=reference, actual=values[indices])
        dump(args.out/'summary.json', summary)
        print(json.dumps(check, indent=2), flush=True)
    if args.refine:
        start = time.perf_counter()
        refined = LargePowderCache(crystal.basis, crystal.cell_vectors_A, mask, q,
                                   bin_width=args.bin_width/2, order=min(12,args.order+2), threads=args.threads)
        refined_values, refined_bound = refined.iq()
        summary['refinement_seconds'] = time.perf_counter()-start
        summary['refinement'] = compare_powder_curves(q, values, refined_values, reference_kind='refinement')
        summary['refinement_metadata'] = refined.metadata
        np.savez_compressed(args.out/'refined.npz', q=q, intensity=refined_values, approximation_bound=refined_bound)
        dump(args.out/'summary.json', summary)
        print(json.dumps(summary['refinement'], indent=2), flush=True)


if __name__ == '__main__':
    main()
