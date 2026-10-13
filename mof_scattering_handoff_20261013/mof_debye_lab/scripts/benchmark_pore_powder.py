#!/usr/bin/env python3
"""Synthetic 100 nm porous crystal with independent Au size/position distributions.

The hollow 424-carbon grid is a numerical fixture, not a chemically valid MOF.
All templates retain their fixed illustrative fcc lattice. Outputs include
standalone runner configurations and separate interpolation/MC uncertainties.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def prepare_inputs(output, cells=40, points=6001, threads=6, samples=4096, seed=20261013):
    import numpy as np
    from mofdebye.core import AtomCloud
    from mofdebye.fixtures import fcc_particle
    axis = (np.arange(8)-3.5)*3.125
    relative = np.stack(np.meshgrid(axis, axis, axis, indexing='ij'), axis=-1).reshape(-1, 3)
    relative = relative[np.linalg.norm(relative, axis=1)>8.5]
    basis = AtomCloud(('C',)*len(relative), relative+12.5,
                      'SYNTHETIC hollow cubic grid; not a chemically valid MOF')
    basis.write_xyz(output/'host_cell.xyz')
    particles = []
    for radius, weight in zip((3.2, 4.2, 5.1), (.5, .3, .2)):
        template = fcc_particle(radius)
        name = f'Au_{len(template)}.xyz'
        template.write_xyz(output/name)
        particles.append(dict(xyz=name, weight=weight))
    config = dict(schema_version='1.1',
        description='SYNTHETIC hollow 424-C grid with Au13/19/43 pores; not a real MOF or measured loading.',
        scattering_case='pxrd', boundary='finite', guest_estimator='analytic',
        backend='radial_moments', device='cpu',
        host=dict(xyz='host_cell.xyz', cell_vectors_A=(np.eye(3)*25).tolist(),
                  cell_mask=np.ones((cells,)*3, dtype=int).tolist()),
        pores=dict(centres_fractional=[[.5, .5, .5]]), particles=particles,
        population=dict(occupancy=.6, position_mode='uniform_ball', uniform_ball_radius_A=2.,
                        orientation='uniform', pore_radius_A=8., wall_clearance_A=.5),
        statistics=dict(seed=seed, n_samples=samples, n_batches=8),
        radial=dict(bin_width_A=.04, order=8, threads=threads),
        sampling=dict(q=dict(min=0., max=6., points=points)))
    dump(output/'analytic.json', config)
    sampled = json.loads(json.dumps(config))
    sampled['guest_estimator'] = 'moments'
    sampled['statistics']['moment_method'] = 'sampled'
    sampled['population']['uniform_ball_radius_A'] = 4.
    dump(output/'active_wall.json', sampled)
    return config, sampled


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cells', type=int, default=40)
    parser.add_argument('--points', type=int, default=6001)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--samples', type=int, default=4096)
    parser.add_argument('--seed', type=int, default=20261013)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--skip-sampled', action='store_true')
    args = parser.parse_args()
    if min(args.cells, args.threads)<1 or args.points<3 or args.samples<16 or args.samples%8 or args.seed<0:
        parser.error('Positive cells/threads, points>=3, samples>=16 divisible by eight, and a nonnegative seed are required.')
    args.out.mkdir(parents=True, exist_ok=False)
    configs = prepare_inputs(args.out, args.cells, args.points, args.threads, args.samples, args.seed)
    if args.prepare_only:
        print(str(args.out.resolve()))
        return
    import numpy as np
    from threadpoolctl import threadpool_limits
    from mofdebye.core import AtomCloud
    from mofdebye.large_powder import LargePowderCache
    from mofdebye.pore_powder import PorePowderCache
    from mofdebye.radial_guests import radial_guest_statistics
    from mofdebye.runner import _inputs, validate_config, write_result, environment_record
    threadpool_limits(limits=1)
    start = time.perf_counter()
    warm = LargePowderCache(AtomCloud(('C',), [[0, 0, 0]]), np.eye(3)*5,
        np.ones((2, 2, 2), dtype=int), np.linspace(0, 6, 61), threads=args.threads)
    warm.iq()
    warmup_seconds = time.perf_counter()-start
    del warm
    q = np.linspace(0, 6, args.points)
    populations = []
    input_seconds = []
    for config in configs:
        start = time.perf_counter()
        validate_config(config)
        _, basis, cell, mask, population, _, processing = _inputs(config, args.out)
        populations.append(population)
        input_seconds.append(time.perf_counter()-start)
    start = time.perf_counter()
    cache = PorePowderCache(basis, cell, mask, populations[0], q, threads=args.threads)
    preparation_seconds = time.perf_counter()-start
    parts = ('host', 'guest_internal', 'host_guest', 'guest_guest', 'total')
    summary = dict(model=configs[0]['description'], edge_nm=args.cells*2.5,
        input_preparation_seconds=input_seconds, warmup_seconds=warmup_seconds,
        geometry_preparation_seconds=preparation_seconds, geometry_metadata=cache.metadata,
        processing=processing, q_points=args.points, q_step_A_inv=float(q[1]),
        environment=environment_record(), distributions={})
    sources = list((PROJECT/'src'/'mofdebye').glob('*.py'))+[Path(__file__)]
    summary['source_sha256'] = {str(p.relative_to(PROJECT)):hashlib.sha256(p.read_bytes()).hexdigest()
                                for p in sorted(sources)}
    for name, population, method in zip(('analytic', 'active_wall'), populations, ('analytic', 'sampled')):
        if args.skip_sampled and method == 'sampled':
            continue
        start = time.perf_counter()
        statistics = radial_guest_statistics(population, q, method=method,
            n_samples=args.samples, n_batches=8, seed=args.seed)
        statistics_seconds = time.perf_counter()-start
        timings = []
        for _ in range(3):
            start = time.perf_counter()
            result = cache.iq(statistics)
            timings.append(time.perf_counter()-start)
        region = q>=.1
        peak = float(np.max(abs(result['total'][region])))
        significant = region & (abs(result['total'])>=.01*peak)
        record = dict(statistics_seconds=statistics_seconds, assembly_seconds=timings,
            first_pattern_seconds_excluding_warmup=preparation_seconds+statistics_seconds+timings[0],
            local_statistics=statistics.metadata,
            max_radial_bound_over_nonforward_peak=float(np.max(result['approximation_bound_excluding_roundoff'][region])/peak),
            max_radial_bound_fraction_at_significant_points=float(np.max(
                result['approximation_bound_excluding_roundoff'][significant]/abs(result['total'][significant]))),
            max_mc_sem_over_nonforward_peak=float(np.max(result['total_mc_sem'][region])/peak),
            max_mc_sem_fraction_at_significant_points=float(np.max(
                result['total_mc_sem'][significant]/abs(result['total'][significant]))),
            uncertainty_note='MC SEM estimates local integration uncertainty; not physical disorder or a strict error bound. '
                             'Radial bound excludes floating-point and MC error. Both omit q-grid undersampling.')
        summary['distributions'][name] = record
        tables = {'intensity':{'q':q, **{key:result[key] for key in parts}},
                  'numerical_error':{'q':q, 'approximation_bound_excluding_roundoff':result['approximation_bound_excluding_roundoff'],
                                     'total_mc_sem':result['total_mc_sem']}}
        write_result({'tables':tables, 'metadata':record}, args.out/name)
        np.savez_compressed(args.out/name/'curves.npz', q=q, **{key:result[key] for key in parts},
                            total_mc_sem=result['total_mc_sem'],
                            approximation_bound=result['approximation_bound_excluding_roundoff'])
        print(json.dumps({name:record}, indent=2), flush=True)
    try:
        import psutil
        summary['memory_after_calculations'] = psutil.Process().memory_info()._asdict()
    except ImportError:
        pass
    dump(args.out/'summary.json', summary)


if __name__ == '__main__':
    main()
