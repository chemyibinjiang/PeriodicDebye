#!/usr/bin/env python3
"""Reproducible forward-kernel microbenchmark; not a full-package benchmark."""
from __future__ import annotations
import argparse, csv, json, os, platform, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import scipy, gemmi, torch
from mofdebye.core import direct_iq, upstream_kernel_iq
from mofdebye.fixtures import finite_host


def metrics(y, reference, q):
    result={}
    for label, select in [('all',q>=0), ('q_ge_1',q>=1), ('q_ge_3',q>=3)]:
        result[f'rel_max_error_{label}']=float(np.max(abs(y[select]-reference[select]))/np.max(abs(reference[select])))
    result['relative_l2_error']=float(np.linalg.norm(y-reference)/np.linalg.norm(reference))
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('results'))
    ap.add_argument('--sizes',type=int,nargs='+',default=[2,3,4])
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--threads',type=int,default=1)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(1)
    q=np.linspace(.15,8.,301)
    env=dict(python=sys.version,numpy=np.__version__,scipy=scipy.__version__,gemmi=gemmi.__version__,
             torch=torch.__version__,cuda_available=torch.cuda.is_available(),visible_cpu_count=os.cpu_count(),
             platform=platform.platform(),cpu_intraop_threads=1,pair_job_threads=args.threads,
             qmin=float(q.min()),qmax=float(q.max()),q_points=len(q),repeats=args.repeats,
             benchmark_scope='adapted standalone upstream forward kernels; NOT installed DebyeCalculator public API',
             normalisation='sum f_i^2 + 2 sum_i<j f_i f_j sinc(q*r_ij)',
             form_factors='Gemmi neutral IT92; common to all tested engines',
             force_grid=True,upstream_blob='89cd257ac681d3859457db7e8220cbc325c6953d')
    try:
        env['cpu_model']=next(x.split(':',1)[1].strip() for x in Path('/proc/cpuinfo').read_text().splitlines() if x.startswith('model name'))
    except (FileNotFoundError,StopIteration): pass
    (args.out/'environment.json').write_text(json.dumps(env,indent=2))
    # Warm-up imports and small tensor operations, excluded from timings.
    tiny,*_=finite_host((1,1,1),n_basis=12)
    for mode in ('direct','grid'):
        upstream_kernel_iq(tiny,q,method=mode,force_grid=True)
    rows=[];raw=[]
    for size in args.sizes:
        host,*_=finite_host((size,size,size),n_basis=24)
        start=time.perf_counter();ref=direct_iq(host,q);reference_seconds=time.perf_counter()-start
        print(f'N={len(host)} NumPy reference: {reference_seconds:.4f} s',flush=True)
        for method,precision in [('direct','float64'),('grid','float64'),('grid','float32')]:
            times=[]
            for rep in range(args.repeats):
                start=time.perf_counter()
                y,info=upstream_kernel_iq(host,q,method=method,precision=precision,
                       num_threads=args.threads,force_grid=True,return_info=True)
                times.append(time.perf_counter()-start)
            row=dict(n_atoms=len(host),q_points=len(q),method=f'upstream_{method}_{precision}',
                     seconds_median=float(np.median(times)),seconds_min=min(times),seconds_max=max(times),
                     numpy_reference_seconds=reference_seconds,**metrics(y,ref,q))
            rows.append(row);raw.append(dict(**row,times=times,kernel_info=info))
            print(f"  {row['method']}: {row['seconds_median']:.4f} s; high-q max error {row['rel_max_error_q_ge_3']:.3g}",flush=True)
        np.savez_compressed(args.out/f'benchmark_N{len(host)}.npz',q=q,reference=ref)
        (args.out/'benchmark_details.json').write_text(json.dumps(raw,indent=2))
    with (args.out/'benchmark.csv').open('w',newline='') as out:
        writer=csv.DictWriter(out,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows

if __name__=='__main__': main()
