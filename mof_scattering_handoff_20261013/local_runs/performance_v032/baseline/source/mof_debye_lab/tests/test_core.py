import numpy as np
import pytest
import torch
from scipy.spatial.transform import Rotation
from mofdebye.core import (AtomCloud, concatenate, direct_iq, cross_iq, form_factors,
                          pair_decomposition, upstream_kernel_iq)
from mofdebye.fixtures import finite_host, fcc_particle
from mofdebye.lattice import shape_autocorrelation, lattice_grouped_iq

torch.set_num_threads(1)

@pytest.mark.parametrize('mode', ['direct', 'grid'])
def test_one_atom_full_normalisation(mode):
    a = AtomCloud(('Au',), [[1., 2., 3.]])
    q = np.array([0., .2, 1., 5., 12.])
    expected = form_factors(a.elements, q)['Au']**2
    got = upstream_kernel_iq(a, q, method=mode, force_grid=True)
    np.testing.assert_allclose(got, expected, rtol=2e-13)

@pytest.mark.parametrize('mode', ['direct', 'grid'])
def test_two_atom_closed_form(mode):
    a = AtomCloud(('Au', 'O'), [[0,0,0], [3.,0,0]])
    q = np.linspace(0., 30., 101); f = form_factors(a.elements, q)
    expected = f['Au']**2 + f['O']**2 + 2*f['Au']*f['O']*np.sinc(3*q/np.pi)
    np.testing.assert_allclose(direct_iq(a, q), expected, rtol=2e-14)
    np.testing.assert_allclose(upstream_kernel_iq(a, q, method=mode, force_grid=True), expected, rtol=1e-10)

@pytest.mark.parametrize('precision,tolerance', [('float64', 1e-9), ('float32', 3e-5)])
def test_grid_mixed_species_random_and_q_zero(precision, tolerance):
    rng = np.random.default_rng(912)
    a = AtomCloud(tuple(['C','O','Zn','Au']*33), rng.normal(size=(132,3))*7)
    q = np.linspace(0, 15, 201)
    reference = direct_iq(a, q)
    got, info = upstream_kernel_iq(a, q, precision=precision, force_grid=True, return_info=True)
    assert info['actual_method'] == 'grid'
    # Check high-Q separately: a giant forward peak must not hide errors.
    for select in (q>=0, q>=1, q>=5):
        assert np.max(abs(got[select]-reference[select]))/np.max(abs(reference[select])) < tolerance


def test_coincident_atoms_are_not_dropped():
    a=AtomCloud(('C','C','Au'), [[0,0,0], [0,0,0], [1,0,0]])
    q=np.linspace(0,8,51)
    np.testing.assert_allclose(upstream_kernel_iq(a,q,force_grid=True),direct_iq(a,q),rtol=1e-10)


def test_rigid_particle_invariance_and_cross_change():
    h,*_=finite_host((1,1,1), n_basis=12)
    g=fcc_particle(4.2, axes=(1.2,.8,1.0))
    moved=g.moved([1.1,-.7,.2],Rotation.from_euler('xyz',[31,17,11],degrees=True).as_matrix())
    q=np.linspace(.1,8,201)
    np.testing.assert_allclose(direct_iq(g,q),direct_iq(moved,q),rtol=1e-12,atol=1e-7)
    assert np.linalg.norm(cross_iq(h,g,q)-cross_iq(h,moved,q))/np.linalg.norm(cross_iq(h,g,q)) > 1e-3


def test_complete_pair_decomposition():
    h,c,*_=finite_host((2,1,1), n_basis=12)
    guests=[fcc_particle(3.2).moved(c[0]+[.2,0,0]), fcc_particle(4.2).moved(c[1]+[0,.3,0])]
    q=np.linspace(0,9,301)
    parts=pair_decomposition(h,guests,q)
    full=direct_iq(concatenate([h]+guests),q)
    np.testing.assert_allclose(parts['total'],full,rtol=1e-11,atol=1e-7)


def test_finite_fft_no_wraparound():
    mask=np.zeros((4,3,2),bool);mask[0,0,0]=True;mask[3,0,0]=True
    vectors,counts=shape_autocorrelation(mask)
    mapping={tuple(v):n for v,n in zip(vectors,counts)}
    assert mapping == {(0,0,0):2,(-3,0,0):1,(3,0,0):1}


def test_irregular_shape_triclinic_grouping():
    basis=AtomCloud(('C','O','Zn'), [[.1,.4,.3],[1.2,.5,.9],[.8,1.1,1.3]])
    cell=np.array([[6.,0,0],[1.1,5.2,0],[.3,.8,7.1]])
    rng=np.random.default_rng(100)
    mask=rng.random((4,3,2))>.25
    host=concatenate([basis.moved(n@cell) for n in np.argwhere(mask)])
    q=np.linspace(0,12,151)
    np.testing.assert_allclose(lattice_grouped_iq(basis,cell,mask,q),direct_iq(host,q),rtol=1e-11,atol=1e-7)


def test_xyz_roundtrip(tmp_path):
    a=fcc_particle(3.2);path=tmp_path/'a.xyz';a.write_xyz(path)
    b=AtomCloud.read_xyz(path)
    assert a.elements==b.elements
    np.testing.assert_allclose(a.xyz,b.xyz)

@pytest.mark.parametrize('q', [[-1], [float('nan')], [], [[1,2]]])
def test_bad_q_rejected(q):
    with pytest.raises(ValueError): direct_iq(fcc_particle(3.2),q)
