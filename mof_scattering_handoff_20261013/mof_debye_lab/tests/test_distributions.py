from itertools import product
import numpy as np
import pytest
from scipy.integrate import quad
from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors
from mofdebye.fixtures import fcc_particle, finite_host
from mofdebye.distributions import ParticleLibrary, PorePopulation, lognormal_catalogue_weights
from mofdebye.ensemble import analytic_ensemble, sampled_ensemble


def test_random_rotation_pair_identity():
    for R,a in [(0.,2.),(4.,0.),(3.,1.5),(7.,4.2)]:
        for q in (0.,.2,1.,6.):
            value=quad(lambda u: .5*np.sinc(q*np.sqrt(max(R*R+a*a-2*R*a*u,0))/np.pi),-1,1,epsabs=1e-12)[0]
            np.testing.assert_allclose(value,np.sinc(q*R/np.pi)*np.sinc(q*a/np.pi),atol=1e-11)


def test_exact_occupancy_enumeration_keeps_p_not_p_squared():
    host=AtomCloud(('O','C'),[[0,1,0],[3,2,1]])
    guest=AtomCloud(('Au',),[[0,0,0]])
    centres=np.array([[5.,0,0],[12.,0,0]])
    probs=np.array([.35,.7])
    pop=PorePopulation(centres,ParticleLibrary((guest,),[1]),occupancy=probs)
    q=np.linspace(0,9,101)
    explicit=np.zeros(len(q))
    for occupied in product((0,1),repeat=2):
        weight=np.prod([probs[k] if x else 1-probs[k] for k,x in enumerate(occupied)])
        clouds=[host]+[guest.moved(centres[k]) for k,x in enumerate(occupied) if x]
        explicit+=weight*direct_iq(concatenate(clouds),q)
    prediction=analytic_ensemble(host,pop,q)
    np.testing.assert_allclose(prediction['total'],explicit,rtol=1e-12,atol=1e-8)


def test_qzero_size_occupancy_variance():
    host=AtomCloud(('C',),[[0,0,0]])
    lib=ParticleLibrary((fcc_particle(3.2),fcc_particle(4.2)),[.4,.6])
    p=np.array([.3,.8,.5]);q=np.array([0.])
    pop=PorePopulation([[0,0,0],[22,0,0],[44,0,0]],lib,occupancy=p,displacement_sigma_A=2.)
    f=form_factors(('C','Au'),q); Fh=f['C'][0]
    particle_F=np.array([len(t)*f['Au'][0] for t in lib.templates])
    m=lib.weights@particle_F;m2=lib.weights@particle_F**2
    expected=(Fh+p.sum()*m)**2+np.sum(p*m2-(p*m)**2)
    np.testing.assert_allclose(analytic_ensemble(host,pop,q)['total'][0],expected,rtol=1e-12)


def test_empty_population_equals_host():
    h,c,*_=finite_host((2,1,1),n_basis=12)
    pop=PorePopulation(c,ParticleLibrary((fcc_particle(3.2),),[1]),occupancy=0.)
    q=np.linspace(0,8,51)
    pred=analytic_ensemble(h,pop,q)
    np.testing.assert_allclose(pred['total'],direct_iq(h,q))
    assert not pop.sample(np.random.default_rng(1))


def test_hard_wall_and_size_probabilities():
    lib=ParticleLibrary((fcc_particle(3.2),fcc_particle(4.2)),[.7,.3])
    pop=PorePopulation([[0,0,0],[22,0,0]],lib,occupancy=.6,
                       displacement_sigma_A=1.0,mean_displacement_A=[1.4,0,0],pore_radius_A=5.5,
                       wall_clearance_A=.2)
    rng=np.random.default_rng(155);counts=np.zeros(2);occ=np.zeros(2)
    for _ in range(1600):
        for d in pop.sample(rng):
            assert np.linalg.norm(d.cloud.xyz-pop.centres[d.pore_index],axis=1).max() <= 5.3+1e-12
            counts[d.template_index]+=1;occ[d.pore_index]+=1
    np.testing.assert_allclose(occ/1600,.6,atol=.04)
    np.testing.assert_allclose(counts/counts.sum(),lib.weights,atol=.04)


def test_uniform_rotations_are_isotropic():
    lib=ParticleLibrary((fcc_particle(3.2),),[1])
    pop=PorePopulation([[0,0,0]],lib)
    rng=np.random.default_rng(29)
    vectors=np.array([pop._rotation(rng)[:,0] for _ in range(6000)])
    np.testing.assert_allclose(vectors.mean(0),0,atol=.03)
    np.testing.assert_allclose(vectors.T@vectors/len(vectors),np.eye(3)/3,atol=.02)


def test_gaussian_covariance():
    cov=np.diag([.2,.5,.9])
    pop=PorePopulation([[0,0,0]],ParticleLibrary((fcc_particle(3.2),),[1]),
                       displacement_covariance_A2=cov,mean_displacement_A=[1,2,3])
    rng=np.random.default_rng(99)
    x=np.array([pop._position(0,rng) for _ in range(12000)])
    np.testing.assert_allclose(x.mean(0),[1,2,3],atol=.03)
    np.testing.assert_allclose(np.cov(x.T),cov,atol=.03)

@pytest.mark.parametrize('params',[{'pore_radius_A':6.}, {'orientation':'fixed'},
                                    {'displacement_covariance_A2':np.diag([1.,2.,1.])},
                                    {'position_mode':'uniform_ball'}])
def test_analytical_path_rejects_incompatible_assumptions(params):
    pop=PorePopulation([[0,0,0]],ParticleLibrary((fcc_particle(3.2),),[1]),**params)
    with pytest.raises(ValueError): pop.analytic_sigma()


def test_lognormal_size_weights():
    w=lognormal_catalogue_weights([3.2,4.2,5.0],4.0,.25,lower_A=2.8,upper_A=5.5)
    assert np.all(w>0) and abs(w.sum()-1)<1e-14


def test_sampled_vs_analytic_with_mc_error():
    h,c,*_=finite_host((2,1,1),n_basis=12)
    pop=PorePopulation(c,ParticleLibrary((fcc_particle(3.2),fcc_particle(4.2)),[.65,.35]),
                       occupancy=[.6,.8],displacement_sigma_A=.45)
    q=np.linspace(.15,7.5,51)
    predicted=analytic_ensemble(h,pop,q)['total']
    mc=sampled_ensemble(h,pop,q,n_samples=1024,seed=9162)
    error=np.abs(mc['mean']['total']-predicted)
    assert np.max(error/np.maximum(mc['sem']['total'],1e-8)) < 5.5
    assert mc['max_full_sum_relative_error'] < 1e-11
