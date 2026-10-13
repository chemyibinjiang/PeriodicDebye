#!/usr/bin/env python3
"""Numerical/Monte Carlo validation and example generation, using a synthetic host."""
from __future__ import annotations
import argparse,csv,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from scipy.spatial.transform import Rotation
from mofdebye.core import direct_iq,cross_iq,concatenate,pair_decomposition,upstream_kernel_iq
from mofdebye.fixtures import finite_host,fcc_particle
from mofdebye.distributions import ParticleLibrary,PorePopulation,lognormal_catalogue_weights
from mofdebye.ensemble import analytic_ensemble,sampled_ensemble,COMPONENTS
from mofdebye.lattice import lattice_grouped_iq


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out',type=Path,default=Path('results'))
    ap.add_argument('--examples',type=Path,default=Path('examples'))
    ap.add_argument('--samples',type=int,default=8192)
    args=ap.parse_args();args.out.mkdir(parents=True,exist_ok=True);args.examples.mkdir(parents=True,exist_ok=True)
    q=np.linspace(.15,8.0,201)
    host,centres,*_=finite_host((2,1,1),n_basis=12)
    templates=(fcc_particle(3.2),fcc_particle(4.2))
    library=ParticleLibrary(templates,[.65,.35])
    pop=PorePopulation(centres,library,occupancy=[.6,.8],displacement_sigma_A=.45)
    start=time.perf_counter();prediction=analytic_ensemble(host,pop,q);analytic_seconds=time.perf_counter()-start
    start=time.perf_counter();mc=sampled_ensemble(host,pop,q,n_samples=args.samples,seed=20261013,
                                               verify_full_first=12,retain_samples=True)
    mc_seconds=time.perf_counter()-start
    difference=mc['mean']['total']-prediction['total']
    relative_l2=np.linalg.norm(difference)/np.linalg.norm(prediction['total'])
    z=np.abs(difference)/np.maximum(mc['sem']['total'],1e-12)
    plain_sum=prediction['host']+prediction['guest_internal']
    cross_importance=np.linalg.norm(prediction['total']-plain_sum)/np.linalg.norm(prediction['total'])
    columns=['q_Ainv']
    arrays=[q]
    for name in COMPONENTS:
        columns += [f'analytic_{name}',f'mc_mean_{name}',f'mc_sem_{name}']
        arrays.extend([prediction[name],mc['mean'][name],mc['sem'][name]])
    np.savetxt(args.out/'ensemble_validation.csv',np.column_stack(arrays),delimiter=',',header=','.join(columns),comments='')
    checkpoints=[n for n in [128,512,2048,8192] if n<=args.samples]
    if args.samples not in checkpoints: checkpoints.append(args.samples)
    convergence=[]
    total_index=COMPONENTS.index('total')
    for n in checkpoints:
        arr=mc['samples'][:n,total_index]
        mean=arr.mean(0);sem=arr.std(0,ddof=1)/np.sqrt(n)
        convergence.append(dict(n_samples=n,
            relative_l2_error=float(np.linalg.norm(mean-prediction['total'])/np.linalg.norm(prediction['total'])),
            max_abs_standardized_residual=float(np.max(abs(mean-prediction['total'])/np.maximum(sem,1e-12)))))
    with (args.out/'mc_convergence.csv').open('w',newline='') as out:
        w=csv.DictWriter(out,fieldnames=list(convergence[0]));w.writeheader();w.writerows(convergence)
    # A rigid guest has the same internal pattern but changed surrounding distances.
    guest=fcc_particle(4.2,axes=(1.2,.8,1.0))
    rotated=guest.moved(rotation=Rotation.from_euler('xyz',[31,17,11],degrees=True).as_matrix())
    shifted=guest.moved([1.3,-.8,.4])
    internal=direct_iq(guest,q)
    rigid_metrics={}
    for name,other in [('translated',shifted),('rotated',rotated)]:
        rigid_metrics[name]=dict(
            internal_relative_max_error=float(np.max(abs(direct_iq(other,q)-internal))/np.max(internal)),
            host_guest_relative_l2_change=float(np.linalg.norm(cross_iq(host,other,q)-cross_iq(host,guest,q))/np.linalg.norm(cross_iq(host,guest,q))))
    # All four contributions for one explicit configuration, independently verified.
    draws=pop.sample(np.random.default_rng(876))
    clouds=[d.cloud for d in draws]
    full=concatenate([host]+clouds,'SYNTHETIC host + sampled rigid guests')
    decomposition=pair_decomposition(host,clouds,q)
    direct=direct_iq(full,q)
    accelerated,info=upstream_kernel_iq(full,q,force_grid=True,return_info=True)
    # Separate irregular full-cell mask / non-orthogonal-cell check is also in pytest.
    h,_,basis,cell,mask=finite_host((3,3,3),n_basis=6)
    mask[0,0,:]=False;mask[2,2,2]=False
    irregular=concatenate([basis.moved(n@cell) for n in np.argwhere(mask)])
    t=time.perf_counter();iref=direct_iq(irregular,q);tref=time.perf_counter()-t
    t=time.perf_counter();igroup=lattice_grouped_iq(basis,cell,mask,q);tgroup=time.perf_counter()-t
    # Boundaries induce conditional position/orientation distributions while preserving chosen sizes.
    bounded_library=ParticleLibrary((fcc_particle(3.2),fcc_particle(4.2),fcc_particle(5.0)),[.5,.3,.2])
    bounded=PorePopulation(centres,bounded_library,occupancy=.7,
                displacement_covariance_A2=np.diag([.9**2,.45**2,.45**2]),mean_displacement_A=[1.4,0,0],
                orientation='wobble',wobble_std_deg=18.,pore_radius_A=6.2,wall_clearance_A=.2,
                host=host,min_host_distance_A=1.6)
    rng=np.random.default_rng(8080);records=[];counts=np.zeros(3,int);occ=np.zeros(2,int)
    max_radius=0.;bounded_samples=2000
    for sample in range(bounded_samples):
        sample_draws=bounded.sample(rng)
        if sample==0:
            concatenate([host]+[d.cloud for d in sample_draws]).write_xyz(args.examples/'bounded_snapshot.xyz')
        for d in sample_draws:
            rmax=float(np.linalg.norm(d.cloud.xyz-centres[d.pore_index],axis=1).max())
            max_radius=max(max_radius,rmax);counts[d.template_index]+=1;occ[d.pore_index]+=1
            records.append([sample,d.pore_index,d.template_index,len(d.cloud),*d.displacement_A,d.attempts,rmax])
    np.savetxt(args.out/'bounded_placements.csv',np.array(records),delimiter=',',
               header='configuration,pore,template,n_atoms,dx_A,dy_A,dz_A,attempts,max_atom_radius_A',comments='')
    # Calculate actual scattering for the bounded/biased model (no invalid analytical shortcut).
    start=time.perf_counter();bounded_mc=sampled_ensemble(host,bounded,q,n_samples=512,seed=7070,
                                                       verify_full_first=8);bounded_seconds=time.perf_counter()-start
    np.savetxt(args.out/'bounded_ensemble.csv',np.column_stack([q]+[bounded_mc['mean'][x] for x in COMPONENTS]+[bounded_mc['sem']['total']]),
               delimiter=',',header='q_Ainv,'+','.join(COMPONENTS)+',total_sem',comments='')
    summary={
        'model':'synthetic mixed C/O/Zn point cage lattice; NOT a chemically validated MOF',
        'host_atoms':len(host),'n_pores':len(centres),'particle_atom_counts':[len(x) for x in templates],
        'number_weights':library.weights.tolist(),'occupancy_probabilities':pop.occupancy.tolist(),
        'gaussian_sigma_per_coordinate_A':.45,'q_points':len(q),'q_range_Ainv':[float(q.min()),float(q.max())],
        'n_mc_samples':args.samples,'analytic_seconds':analytic_seconds,'mc_seconds':mc_seconds,
        'mc_relative_l2_error':float(relative_l2),'max_abs_standardized_residual':float(z.max()),
        'fraction_q_within_3_mc_standard_errors':float(np.mean(z<=3)),
        'naive_component_addition_relative_l2_error':float(cross_importance),
        'full_sum_vs_decomposition_relative_max_error':float(mc['max_full_sum_relative_error']),
        'upstream_grid_vs_reference_composite_relative_max_error':float(np.max(abs(accelerated-direct))/np.max(direct)),
        'composite_kernel_info':info,'rigid_particle_checks':rigid_metrics,
        'irregular_lattice_grouping':dict(n_atoms=len(irregular),relative_max_error=float(np.max(abs(igroup-iref))/np.max(iref)),
                                        direct_seconds=tref,grouped_seconds=tgroup),
        'bounded_sampling':dict(n_configurations=bounded_samples,n_particles=int(counts.sum()),
            requested_size_weights=bounded_library.weights.tolist(),sampled_size_weights=(counts/counts.sum()).tolist(),
            sampled_occupancy_probabilities=(occ/bounded_samples).tolist(),max_atom_radius_A=max_radius,allowed_radius_A=6.,
            mean_attempts=float(np.mean(np.array(records)[:,-2])),
            scattering_configurations=512,scattering_seconds=bounded_seconds,
            full_sum_relative_max_error=bounded_mc['max_full_sum_relative_error']),
        'limitations':['Finite-system validation, not an infinite-lattice implementation.',
             'Analytical formula uses independent pores, uniform SO(3) rotations, unbounded isotropic Gaussian shifts.',
             'Bounded example uses explicit Monte Carlo instead; no claim of identical distributions.',
             'No framework relaxation, anomalous terms, thermal/instrumental effects, pore finding, or inferred material parameters.']}
    (args.out/'validation_summary.json').write_text(json.dumps(summary,indent=2))
    host.write_xyz(args.examples/'synthetic_host.xyz')
    full.write_xyz(args.examples/'analytic_snapshot.xyz')
    for t in bounded_library.templates: t.write_xyz(args.examples/f'Au_{len(t)}.xyz')
    config=dict(description='Synthetic validation model, not a real MOF',host_xyz='synthetic_host.xyz',
        particles=[dict(xyz=f'Au_{len(t)}.xyz',weight=float(w)) for t,w in zip(templates,library.weights)],
        pore_centres_A=centres.tolist(),population=dict(occupancy=[.6,.8],displacement_sigma_A=.45,orientation='uniform'),
        q=dict(min=.15,max=8.,points=201))
    (args.examples/'analytic_validation.json').write_text(json.dumps(config,indent=2))
    bounded_config=dict(config)
    bounded_config['description']='Synthetic bounded, biased-position and wobbling-particle example'
    bounded_config['particles']=[dict(xyz=f'Au_{len(t)}.xyz',weight=float(w)) for t,w in zip(bounded_library.templates,bounded_library.weights)]
    bounded_config['population']=dict(occupancy=.7,displacement_covariance_A2=np.diag([.9**2,.45**2,.45**2]).tolist(),
        mean_displacement_A=[1.4,0,0],orientation='wobble',wobble_std_deg=18.,pore_radius_A=6.2,
        wall_clearance_A=.2,min_host_distance_A=1.6)
    (args.examples/'bounded_demo.json').write_text(json.dumps(bounded_config,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    assert z.max()<5.5, 'Unexpected Monte Carlo disagreement; inspect before relying on the analytic path.'
    assert max_radius<=6.+1e-12

if __name__=='__main__':main()
