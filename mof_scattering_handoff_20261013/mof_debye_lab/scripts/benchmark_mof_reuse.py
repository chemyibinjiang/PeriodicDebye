#!/usr/bin/env python3
"""Separate MOF geometry preparation from repeated exact q-grid evaluation."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(PROJECT/'src'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--cells',type=int,default=4)
    parser.add_argument('--points',type=int,nargs='+',default=[121,1201])
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--cache-mib',type=int,default=64,help='Retained radial cache budget, including integer counts.')
    args=parser.parse_args()
    if args.cells<1 or min(args.points)<2 or args.repeats<1 or args.cache_mib<0:
        parser.error('Positive cells/repeats, at least two q points, and nonnegative cache budget required.')
    if args.out.exists():
        parser.error('Use a fresh output directory.')
    import numpy as np
    import torch
    from threadpoolctl import threadpool_limits
    from mofdebye.crystal import read_cif
    from mofdebye.core import form_factors
    from mofdebye.lattice import LatticePairCache
    from mofdebye.directional import cell_shape_amplitude
    from mofdebye.runner import environment_record
    torch.set_num_threads(1)
    threadpool_limits(limits=1)
    args.out.mkdir(parents=True)
    data=PROJECT/'examples'/'real_mof'
    provenance=json.loads((data/'provenance.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((data/'IRMOF-1.cif').read_bytes()).hexdigest()==provenance['cif']['sha256']
    started=time.perf_counter()
    crystal=read_cif(data/'IRMOF-1.cif')
    cif_seconds=time.perf_counter()-started
    started=time.perf_counter()
    cache=LatticePairCache(crystal.basis,crystal.cell_vectors_A,np.ones((args.cells,)*3,dtype=int),
                           cache_max_bytes=args.cache_mib*1024**2)
    preparation_seconds=time.perf_counter()-started
    curves={}
    timings=[]
    for points in args.points:
        q=np.linspace(0,6,points)
        times=[]
        for _ in range(args.repeats):
            started=time.perf_counter()
            values=cache.iq(q)
            times.append(time.perf_counter()-started)
        factors=form_factors(crystal.basis.elements,[0])
        forward=(sum(factors[e][0] for e in crystal.basis.elements)*args.cells**3)**2
        np.testing.assert_allclose(values[0],forward,rtol=2e-12)
        curves[f'q_{points}']=q
        curves[f'intensity_{points}']=values
        timings.append(dict(points=points,seconds=times,median_seconds=statistics.median(times)))
    # A denser grid does not change the scattering value at common points.
    comparisons=[]
    coarse=min(args.points)
    for fine in args.points:
        if fine<=coarse or (fine-1)%(coarse-1):
            continue
        actual=curves[f'intensity_{fine}'][::((fine-1)//(coarse-1))]
        reference=curves[f'intensity_{coarse}']
        scaled=float(np.max(abs(actual-reference))/max(1.,np.max(abs(reference))))
        nonforward=float(np.max(abs(actual[1:]-reference[1:]))/max(1.,np.max(abs(reference[1:]))))
        assert scaled<2e-10 and nonforward<2e-10
        comparisons.append(dict(coarse_points=coarse,fine_points=fine,
                                scaled_error=scaled,nonforward_scaled_error=nonforward))
    # Reuse the imported cell for a 103.3 nm single-crystal line as well.
    q=np.linspace(0,6,2049)
    direction=np.array([1.,.27,.13]); direction/=np.linalg.norm(direction)
    vectors=q[:,None]*direction
    mask=np.ones((40,)*3,dtype=int)
    directional_times=[]
    for _ in range(args.repeats):
        started=time.perf_counter()
        amp=cell_shape_amplitude(crystal.basis,crystal.cell_vectors_A,mask,vectors)
        directional_times.append(time.perf_counter()-started)
    curves['directional_q_vectors']=vectors
    curves['directional_amplitude']=amp
    np.savez_compressed(args.out/'outputs.npz',**curves)
    report=dict(environment=environment_record(),threads=1,cells_per_edge=args.cells,
        conceptual_atoms=424*args.cells**3,edge_nm=2.5832*args.cells,
        cif_seconds=cif_seconds,cache_preparation_seconds=preparation_seconds,
        powder_evaluations=timings,cache_metadata=cache.metadata,grid_comparisons=comparisons,
        directional_cells_per_edge=40,directional_points=2049,
        directional_seconds=directional_times,
        directional_median_seconds=statistics.median(directional_times),
        scope='Warm kernels with cell/cache retained; excludes input import, cache construction and output I/O from evaluation timings. Exact Debye, no angular approximation.')
    (args.out/'summary.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    main()
