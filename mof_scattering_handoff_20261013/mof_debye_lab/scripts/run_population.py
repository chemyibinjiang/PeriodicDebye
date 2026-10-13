#!/usr/bin/env python3
"""Run the example or your own host/particle XYZ files and pore configuration."""
from __future__ import annotations
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
import torch
from mofdebye import __version__
from mofdebye.core import AtomCloud,direct_iq,upstream_kernel_iq
from mofdebye.distributions import ParticleLibrary,PorePopulation
from mofdebye.ensemble import analytic_ensemble,sampled_ensemble,accelerated_sampled_ensemble


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--config',type=Path,required=True)
    ap.add_argument('--mode',choices=['analytic','montecarlo'],default='analytic')
    ap.add_argument('--backend',choices=['reference','upstream-kernel'],default='upstream-kernel')
    ap.add_argument('--samples',type=int,default=512)
    ap.add_argument('--seed',type=int,default=20261013)
    ap.add_argument('--device',choices=['cpu','cuda'],default='cpu')
    ap.add_argument('--threads',type=int,default=1)
    ap.add_argument('--force-grid',action='store_true',help='Disable the upstream small-system direct fallback.')
    ap.add_argument('--out',type=Path,default=Path('population_output'))
    args=ap.parse_args()
    if args.threads < 1:
        ap.error('--threads must be positive.')
    if args.backend == 'reference' and args.device != 'cpu':
        ap.error('The reference backend executes on CPU; use --device cpu.')
    torch.set_num_threads(args.threads)
    cfg=json.loads(args.config.read_text(encoding='utf-8'));root=args.config.resolve().parent
    host=AtomCloud.read_xyz(root/cfg['host_xyz'])
    particles=tuple(AtomCloud.read_xyz(root/p['xyz']) for p in cfg['particles'])
    library=ParticleLibrary(particles,[p['weight'] for p in cfg['particles']])
    population=PorePopulation(cfg['pore_centres_A'],library,host=host,**cfg.get('population',{}))
    qs=cfg['q'];q=np.linspace(qs['min'],qs['max'],qs['points'])
    start=time.perf_counter()
    if args.mode=='analytic':
        # Validate assumptions BEFORE computing a possibly expensive host pattern.
        population.analytic_sigma()
        if args.backend=='reference':
            hh=direct_iq(host,q)
            actual_host_method='numpy_scipy_direct_debye'
        else:
            hh,host_info=upstream_kernel_iq(host,q,device=args.device,num_threads=args.threads,
                force_grid=args.force_grid,return_info=True)
            actual_host_method=host_info['actual_method']
        values=analytic_ensemble(host,population,q,host_iq=hh)
        result={'mean':values,'sem':{'total':np.zeros(len(q))},'actual_kernel_counts':{actual_host_method:1}}
    elif args.backend=='reference':
        result=sampled_ensemble(host,population,q,n_samples=args.samples,seed=args.seed)
    else:
        result=accelerated_sampled_ensemble(host,population,q,n_samples=args.samples,seed=args.seed,
                        device=args.device,num_threads=args.threads,force_grid=args.force_grid)
    elapsed=time.perf_counter()-start
    args.out.mkdir(parents=True,exist_ok=True)
    names=list(result['mean'])
    columns=[q]+[result['mean'][x] for x in names]+[result['sem']['total']]
    np.savetxt(args.out/'intensity.csv',np.column_stack(columns),delimiter=',',
               header='q_Ainv,'+','.join(names)+',total_MC_SEM',comments='')
    metadata=dict(config=cfg,mode=args.mode,backend=args.backend,device=args.device,seconds=elapsed,
                  scattering_case='pxrd',boundary='finite',guest_estimator=args.mode,
                  software_version=__version__,threads=args.threads,
                  host_atoms=len(host),particle_atom_counts=[len(t) for t in particles],
                  n_samples=None if args.mode=='analytic' else args.samples,seed=args.seed,
                  intensity_convention='full finite-system Debye intensity; neutral IT92 factors; no instrument corrections',
                  actual_kernel_counts=result.get('actual_kernel_counts'),
                  analytical_zero_SEM_means='no Monte Carlo sampling, NOT zero uncertainty in physical parameters')
    (args.out/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps(metadata,indent=2))

if __name__=='__main__':main()
