#!/usr/bin/env python3
"""Reproducible CPU MOF-5 workloads, isolated timeouts, profiles and raw outputs.

No finite-object result is substituted with a bulk or angular approximation.
Use --project to benchmark a preserved source checkout with the same interpreter.
The optional psutil package supplies sampled RSS and Windows peak working set.
"""
from __future__ import annotations

import argparse
import cProfile
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import pstats
import statistics
import subprocess
import sys
import threading
import time

PROJECT = Path(__file__).resolve().parents[1]


def cases():
    result = []
    for qmax in (3, 6, 10):
        for mode in ('periodic', 'bulk_powder'):
            result.append(dict(name=f'{mode}_q{qmax}', kind=mode, qmax=qmax,
                               points=401, repeats=3))
    for n in (4, 10, 20, 40):
        result.append(dict(name=f'finite_n{n}', kind='finite', cells_per_edge=n,
                           points=2049, qmax=6, repeats=3))
    for n in (1, 2, 4, 8):
        result.append(dict(name=f'powder_n{n}', kind='powder', cells_per_edge=n,
                           points=121, qmax=6, repeats=3))
    for samples in (512, 2048):
        result.append(dict(name=f'guest_samples{samples}', kind='guest',
                           points=257, qmax=6, samples=samples, repeats=3))
    return result


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def worker(args):
    # Imports occur after parent-provided thread limits and source selection.
    sys.path.insert(0, str(args.project.resolve()/'src'))
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits, threadpool_info
    from mofdebye import __version__
    from mofdebye.crystal import read_cif
    from mofdebye.runner import run_configuration, environment_record
    from mofdebye.moments import guest_moments
    from mofdebye.distributions import ParticleLibrary, PorePopulation
    from mofdebye.fixtures import fcc_particle

    torch.set_num_threads(args.threads)
    threadpool_limits(limits=args.threads)
    spec = json.loads(args.case_file.read_text(encoding='utf-8'))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    data = args.project.resolve()/'examples'/'real_mof'
    provenance = json.loads((data/'provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((data/'IRMOF-1.cif').read_bytes()).hexdigest() == provenance['cif']['sha256']
    # Warm metadata imports independently: reported time includes calculation and
    # CIF loading, excludes interpreter startup, output serialization and profiling.
    environment = environment_record()
    environment.update(logical_cpus=os.cpu_count(), threadpools=threadpool_info())
    t = time.perf_counter()
    crystal = read_cif(data/'IRMOF-1.cif')
    cif_seconds = time.perf_counter()-t
    assert len(crystal.basis) == 424
    q = np.linspace(0, spec['qmax'], spec['points'])
    direction = np.array([1., .27, .13])
    direction /= np.linalg.norm(direction)
    vectors = q[:, None]*direction
    config = dict(schema_version='1.1', guest_estimator='analytic', backend='reference',
                  device='cpu', host=dict(cif='IRMOF-1.cif',
                  cif_options=dict(occupancy_policy='require_full')))
    if spec['kind'] in ('finite', 'powder'):
        n = spec['cells_per_edge']
        config['host']['cell_mask'] = np.ones((n,n,n), dtype=int).tolist()
        config.update(boundary='finite', backend='lattice_grouped',
                      scattering_case='single_crystal_finite' if spec['kind']=='finite' else 'pxrd',
                      sampling=dict(q_vectors=vectors.tolist()) if spec['kind']=='finite' else dict(q=q.tolist()))
    elif spec['kind'] in ('periodic', 'bulk_powder'):
        config.update(boundary='bulk', scattering_case='single_crystal_periodic'
                      if spec['kind']=='periodic' else 'pxrd',
                      sampling=dict(reflection_q_max=spec['qmax'])
                      if spec['kind']=='periodic' else dict(q=q.tolist()),
                      angular=dict(n_mu=24, n_phi=48))
    else:
        # Algorithmic guest workload: these are synthetic sites, NOT established
        # MOF-5 pores or a physically fitted loading configuration.
        template = fcc_particle(radius_A=3.2)
        centres = np.array([[.2,.2,.2],[.2,.7,.7],[.7,.2,.7],[.7,.7,.2]])@crystal.cell_vectors_A
        population = PorePopulation(centres, ParticleLibrary((template,), [1.]),
                                    host=crystal.basis, occupancy=.7,
                                    displacement_sigma_A=.4, orientation='uniform')
        config = dict(description='Synthetic four-site Au13 sampling workload on MOF-5 cell; no physical pore claim.',
                      sites_A=centres.tolist(), template_atoms=len(template), occupancy=.7,
                      displacement_sigma_A=.4, n_samples=spec['samples'], seed=20261013)

    dump(out/'configuration.json', config)
    def evaluate():
        if spec['kind']=='guest':
            gm=guest_moments(population, vectors, method='sampled',
                             n_samples=spec['samples'], seed=20261013)
            return dict(tables=dict(moments=dict(mean_real=gm.mean.real, mean_imag=gm.mean.imag,
                        variance=gm.variance, mean_variance=gm.mean_variance)),
                        metadata=dict(moment_metadata=gm.metadata, actual_method=['sampled_local_guest_moments']))
        return run_configuration(config, data)

    # Sample process RSS during measured runs. Absolute peak includes interpreter
    # and libraries; the delta is incremental RSS, not a universal memory bound.
    stop = threading.Event()
    rss = []
    process = None
    try:
        import psutil
        process = psutil.Process()
        def monitor():
            while not stop.is_set():
                rss.append(process.memory_info().rss)
                stop.wait(.005)
        rss.append(process.memory_info().rss)
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    except ImportError:
        monitor_thread = None

    durations = []
    methods = []
    for repeat in range(spec['repeats']):
        started = time.perf_counter()
        result = evaluate()
        duration = time.perf_counter()-started
        durations.append(duration)
        methods.append(result['metadata']['actual_method'])
        # Save each completed timing immediately so timeout still leaves evidence.
        dump(out/'partial.json', dict(completed_repeats=len(durations), seconds=durations))
    stop.set()
    if monitor_thread is not None:
        monitor_thread.join()

    arrays = {}
    for table, columns in result['tables'].items():
        order = None
        if table=='reflections':
            order = np.lexsort((columns['l'], columns['k'], columns['h']))
        for name, values in columns.items():
            value=np.asarray(values)
            arrays[f'{table}__{name}']=value if order is None else value[order]
    np.savez_compressed(out/'outputs.npz', **arrays)
    dump(out/'metadata.json', result['metadata'])
    record = dict(status='completed', case=spec, version=__version__, environment=environment,
                  source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted((args.project/'src'/'mofdebye').glob('*.py'))},
                  source_cif_sha256=provenance['cif']['sha256'], cif_load_seconds=cif_seconds,
                  timings_seconds=durations, median_seconds=statistics.median(durations),
                  min_seconds=min(durations), max_seconds=max(durations),
                  timing_scope='Warm interpreter, complete run_configuration including CIF and metadata; excludes output files and profile. Guest case is guest_moments only after setup.',
                  actual_methods=methods,
                  table_rows={name:len(next(iter(table.values()))) for name,table in result['tables'].items()},
                  peak_sampled_rss_bytes=max(rss) if rss else None,
                  initial_rss_bytes=rss[0] if rss else None,
                  incremental_sampled_rss_bytes=max(rss)-rss[0] if rss else None,
                  windows_process_peak_wset_bytes=getattr(process.memory_info(),'peak_wset',None) if process else None,
                  memory_scope='5 ms sampled RSS over three runs; Windows process peak includes setup/imports. Not a hard memory bound.')
    if 'cells_per_edge' in spec:
        record.update(retained_cells=spec['cells_per_edge']**3,
                      conceptual_host_atoms=424*spec['cells_per_edge']**3,
                      edge_nm=25.832*spec['cells_per_edge']/10)
    dump(out/'timing.json', record)
    # Profile separately to avoid contaminating reported timing samples.
    if args.profile:
        profiler = cProfile.Profile()
        profiler.runcall(evaluate)
        profiler.dump_stats(str(out/'profile.pstats'))
        report = io.StringIO()
        pstats.Stats(profiler, stream=report).strip_dirs().sort_stats('cumulative').print_stats(45)
        (out/'profile.txt').write_text(report.getvalue(), encoding='utf-8')
    print(json.dumps(dict(case=spec['name'], median_seconds=record['median_seconds'],
                         rss_MiB=None if not rss else max(rss)/1024**2)), flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=PROJECT)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=120., help='Per worker budget including imports, repeats and optional profile.')
    parser.add_argument('--select', nargs='*', help='Case names; omit for all.')
    parser.add_argument('--repeats', type=int, help='Override repeat count for an extended workload.')
    parser.add_argument('--points', type=int, help='Override q-grid size; record the changed workload explicitly.')
    parser.add_argument('--profile', action='store_true')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--case-file', type=Path, help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.threads<1 or args.timeout<=0:
        parser.error('Positive thread count and timeout required.')
    if args.worker:
        return worker(args)
    selected=[c for c in cases() if args.select is None or c['name'] in args.select]
    if not selected or (args.select is not None and set(args.select)-{c['name'] for c in selected}):
        parser.error('Unknown/empty case selection.')
    if args.repeats is not None and args.repeats<1 or args.points is not None and args.points<2:
        parser.error('Require repeats >= 1 and points >= 2.')
    for spec in selected:
        if args.repeats is not None:
            spec['repeats']=args.repeats
        if args.points is not None:
            spec['points']=args.points
    args.out.mkdir(parents=True, exist_ok=True)
    # Refuse reuse: a killed worker must never inherit a prior successful timing
    # or NPZ result. Use a fresh destination for every benchmark campaign.
    occupied=[c['name'] for c in selected if (args.out/c['name']).exists()]
    if occupied:
        parser.error(f'Case directories already exist; choose a fresh --out: {occupied}')
    env=dict(os.environ)
    for name in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
        env[name]=str(args.threads)
    records=[]
    for spec in selected:
        output=args.out.resolve()/spec['name']
        output.mkdir(parents=True, exist_ok=True)
        case_path=output/'case.json'
        dump(case_path,spec)
        command=[sys.executable,str(Path(__file__).resolve()),'--worker','--project',str(args.project.resolve()),
                 '--out',str(output),'--threads',str(args.threads),'--case-file',str(case_path)]
        if args.profile:
            command.append('--profile')
        started=time.perf_counter()
        try:
            run=subprocess.run(command, env=env, capture_output=True, text=True, timeout=args.timeout)
            (output/'stdout.txt').write_text(run.stdout,encoding='utf-8')
            (output/'stderr.txt').write_text(run.stderr,encoding='utf-8')
            status='completed' if run.returncode==0 else 'failed'
            record=dict(case=spec['name'], status=status, returncode=run.returncode)
        except subprocess.TimeoutExpired as exc:
            record=dict(case=spec['name'],status='timeout', timeout_seconds=args.timeout)
            for name,value in [('stdout',exc.stdout),('stderr',exc.stderr)]:
                (output/f'{name}.txt').write_text(value.decode(errors='replace') if isinstance(value,bytes) else value or '',encoding='utf-8')
        record.update(worker_wall_seconds=time.perf_counter()-started, command=command)
        if (output/'timing.json').is_file():
            timing=json.loads((output/'timing.json').read_text(encoding='utf-8'))
            record.update(median_seconds=timing['median_seconds'], completed_timing_repeats=len(timing['timings_seconds']))
        records.append(record)
        dump(args.out/'run_summary.json',records)
        print(json.dumps(record),flush=True)
    if any(r['status']=='failed' for r in records):
        raise SystemExit(1)


if __name__=='__main__':
    main()
