import numpy as np
from mofdebye.fixtures import finite_host,fcc_particle
from mofdebye.distributions import ParticleLibrary,PorePopulation
from mofdebye.ensemble import sampled_ensemble,accelerated_sampled_ensemble


def test_same_random_configurations_external_kernel_vs_reference():
    host,centres,*_=finite_host((2,1,1),n_basis=12)
    pop=PorePopulation(centres,ParticleLibrary((fcc_particle(3.2),fcc_particle(4.2)),[.65,.35]),
                       occupancy=.7,displacement_sigma_A=.45,pore_radius_A=6.2)
    q=np.linspace(.15,8,51)
    reference=sampled_ensemble(host,pop,q,n_samples=32,seed=4451)
    accelerated=accelerated_sampled_ensemble(host,pop,q,n_samples=32,seed=4451,force_grid=True)
    assert accelerated['actual_kernel_counts']=={'grid':32}
    for statistic in ('mean','sem'):
        a,b=accelerated[statistic]['total'],reference[statistic]['total']
        assert np.max(np.abs(a-b))/np.max(np.abs(b)) < 1e-9
