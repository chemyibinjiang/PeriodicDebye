"""Exact radial bulk shortcut; anisotropic sampled diffuse must stay angular."""
import numpy as np
import pytest

from mofdebye.core import AtomCloud, form_factors
from mofdebye.directional import angular_grid
from mofdebye.runner import run_configuration


def config(tmp_path, sampled=False):
    AtomCloud(('C',), [[0, 0, 0]]).write_xyz(tmp_path/'host.xyz')
    AtomCloud(('Au','Au'), [[-1,0,0],[1,0,0]]).write_xyz(tmp_path/'guest.xyz')
    return dict(schema_version='1.1', scattering_case='pxrd', boundary='bulk',
        guest_estimator='moments' if sampled else 'analytic', backend='reference',
        host=dict(xyz='host.xyz',cell_vectors_A=(np.eye(3)*8).tolist()),
        particles=[dict(xyz='guest.xyz',weight=1)], pores=dict(centres_A=[[2,3,4]]),
        population=dict(occupancy=.63, displacement_sigma_A=0.,
                        orientation='fixed' if sampled else 'uniform'),
        statistics=dict(seed=20261013,n_samples=128,moment_method='sampled'),
        sampling=dict(q=[0., .3, .7, 1.2]), angular=dict(n_mu=12,n_phi=24))


def test_runner_analytic_bulk_diffuse_matches_dimer_closed_form(tmp_path):
    c=config(tmp_path)
    c['population']['displacement_sigma_A']=.45
    result=run_configuration(c,tmp_path)
    q=np.array(c['sampling']['q'])
    f=form_factors(('Au',),q)['Au']
    mean=2*f*np.sinc(q/np.pi)*np.exp(-.5*(.45*q)**2)
    second=2*f*f*(1+np.sinc(2*q/np.pi))
    expected=.63*second-.63**2*mean**2
    np.testing.assert_allclose(result['tables']['diffuse']['diffuse'],expected,rtol=2e-13)
    assert result['metadata']['powder_diffuse_method']=='exact_isotropic_radial_variance'
    assert np.all(result['tables']['diffuse']['angular_mean_amplitude_variance']==0)


def test_runner_sampled_anisotropic_diffuse_retains_angular_average(tmp_path):
    c=config(tmp_path,sampled=True)
    result=run_configuration(c,tmp_path)
    q=np.array(c['sampling']['q'])
    directions,weights=angular_grid(**c['angular'])
    # Fixed Au dimer plus Bernoulli occupancy: draw sequence is the same for
    # every direction. Independent sample variance of occupancies multiplies
    # squared amplitude 4 f(q)^2 cos(q_x)^2, which is anisotropic.
    p_variance=result['tables']['diffuse']['diffuse'][0]/(4*form_factors(('Au',),[0])['Au'][0]**2)
    expected=p_variance*4*form_factors(('Au',),q)['Au']**2*(np.cos(q[:,None]*directions[:,0])**2@weights)
    np.testing.assert_allclose(result['tables']['diffuse']['diffuse'],expected,rtol=3e-13)
    assert 'powder_diffuse_method' not in result['metadata']


@pytest.mark.parametrize('value',[0,True,2.5])
def test_runner_analytic_shortcut_still_validates_angular_input(tmp_path,value):
    c=config(tmp_path)
    c['angular']['n_mu']=value
    with pytest.raises(ValueError,match='positive integer'):
        run_configuration(c,tmp_path)
