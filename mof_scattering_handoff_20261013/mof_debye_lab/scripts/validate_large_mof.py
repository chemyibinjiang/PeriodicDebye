#!/usr/bin/env python3
"""Dense real-MOF comparison of radial moments with the exact lattice Debye sum.

Uses the same retained geometry and q points for both backends. Preparation and
evaluation times are separate; JIT compilation is warmed and reported before
timing. This script does not benchmark process startup, CIF parsing, or output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cells', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--points', type=int, default=601)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--bin-width', type=float, default=.04)
    parser.add_argument('--order', type=int, default=8)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output directory.')
    if min(args.cells) < 1 or args.points < 2 or args.repeats < 1:
        parser.error('Positive cell counts/repeats and at least two q points are required.')

    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from mofdebye.core import AtomCloud
    from mofdebye.crystal import read_cif
    from mofdebye.large_powder import LargePowderCache
    from mofdebye.lattice import LatticePairCache
    from mofdebye.runner import environment_record
    from validate_large_powder import compare_powder_curves

    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    args.out.mkdir(parents=True)
    q = np.linspace(0, 6, args.points)
    cif = PROJECT/'examples'/'real_mof'/'IRMOF-1.cif'
    provenance = json.loads((cif.parent/'provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256(cif.read_bytes()).hexdigest() == provenance['cif']['sha256']
    start = time.perf_counter()
    crystal = read_cif(cif)
    cif_seconds = time.perf_counter()-start

    warm_basis = AtomCloud(('C', 'O'), [[0., 0., 0.], [1.1, .3, .4]])
    start = time.perf_counter()
    warm_cache = LargePowderCache(warm_basis, np.eye(3)*4, np.ones((1, 1, 1)), q,
                                 bin_width=args.bin_width, order=args.order, threads=1, merge_decimals=12)
    warm_cache.iq()
    warmup_seconds = time.perf_counter()-start

    sources = [Path(__file__), PROJECT/'scripts'/'validate_large_powder.py', cif]
    sources += [PROJECT/'src'/'mofdebye'/name for name in
                ('large_powder.py', 'radial.py', 'lattice.py', 'core.py', 'crystal.py')]
    source_hashes = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sources}
    report = dict(command=sys.argv, environment=environment_record(), threads=1,
                  q_range_A_inv=[0., 6.], q_points=args.points, target_fraction=.001,
                  cif_seconds=cif_seconds, jit_warmup_seconds=warmup_seconds,
                  source_sha256=source_hashes, cases=[],
                  scope='Retained-cache CPU comparison. One preparation per backend and repeated evaluations. '
                        'CIF import, JIT warmup, process startup and I/O excluded from evaluation timings. '
                        'The exact reference uses LatticePairCache; no q=0 intensity normalization.')
    for cells in args.cells:
        mask = np.ones((cells, cells, cells), dtype=int)
        start = time.perf_counter()
        approximate = LargePowderCache(crystal.basis, crystal.cell_vectors_A, mask, q,
                                       bin_width=args.bin_width, order=args.order, threads=1,
                                       merge_decimals=12)
        approximate_prepare = time.perf_counter()-start
        approximate_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            actual, bound = approximate.iq()
            approximate_times.append(time.perf_counter()-start)
        start = time.perf_counter()
        exact = LatticePairCache(crystal.basis, crystal.cell_vectors_A, mask,
                                 cache_max_bytes=256*1024**2)
        exact_prepare = time.perf_counter()-start
        exact_times = []
        for _ in range(args.repeats):
            start = time.perf_counter()
            reference = exact.iq(q)
            exact_times.append(time.perf_counter()-start)
        comparison = compare_powder_curves(q, actual, reference)
        comparison['peaks_within_one_grid_step'] = bool(comparison['max_peak_q_shift_A_inv'] <= 6/(args.points-1)*1.000001)
        comparison['max_analytic_bound_over_nonforward_peak'] = float(
            np.max(bound[q >= .1])/comparison['nonforward_peak_scale'])
        case = dict(cells_per_edge=cells, conceptual_atoms=len(crystal.basis)*cells**3,
                    edge_nm=float(np.linalg.norm(crystal.cell_vectors_A[0])*cells/10),
                    q_points=args.points, approximate_prepare_seconds=approximate_prepare,
                    exact_prepare_seconds=exact_prepare, approximate_evaluation_seconds=approximate_times,
                    exact_evaluation_seconds=exact_times,
                    approximate_evaluation_median_seconds=statistics.median(approximate_times),
                    exact_evaluation_median_seconds=statistics.median(exact_times),
                    warm_evaluation_speedup=statistics.median(exact_times)/statistics.median(approximate_times),
                    first_curve_speedup=(exact_prepare+statistics.median(exact_times))/
                                        (approximate_prepare+statistics.median(approximate_times)),
                    approximate_metadata=approximate.metadata, exact_metadata=exact.metadata,
                    comparison=comparison)
        np.savez_compressed(args.out/f'n{cells}_curves.npz', q=q, actual=actual,
                            reference=reference, approximation_bound_excluding_roundoff=bound)
        report['cases'].append(case)
        (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
        print(json.dumps(dict(cells=cells, atoms=case['conceptual_atoms'],
                              exact_seconds=case['exact_evaluation_median_seconds'],
                              approximate_seconds=case['approximate_evaluation_median_seconds'],
                              speedup=case['warm_evaluation_speedup'],
                              scaled_error=comparison['max_abs_error_over_nonforward_peak'],
                              pointwise_error=comparison['max_relative_error_above_floor'],
                              passed=comparison['intensity_checks_passed'])), flush=True)
    if not all(case['comparison']['intensity_checks_passed'] and
               case['comparison']['peaks_within_one_grid_step'] for case in report['cases']):
        raise SystemExit('One or more accuracy checks failed; see saved raw curves and summary.')


if __name__ == '__main__':
    main()
