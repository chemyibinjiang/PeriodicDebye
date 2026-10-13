import itertools
import numpy as np
import pytest
from mofdebye.core import AtomCloud, concatenate, form_factors
from mofdebye.directional import amplitude, intensity, lattice_shape_amplitude
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.moments import GuestMoments, guest_moments
from mofdebye.periodic import (reciprocal_lattice, enumerate_reflections,
    bragg_reflections, powder_shells, powder_diffuse, finite_cell_ensemble,
    cell_guest_moments, validate_periodic_population)


def test_triclinic_reciprocal_complete_enumeration():
    cell = np.array([[4.,0,0],[3.7,2.1,0],[1.2,.8,3.3]])
    b = reciprocal_lattice(cell)
    np.testing.assert_allclose(cell @ b.T, 2*np.pi*np.eye(3), atol=1e-14)
    h, vectors, q = enumerate_reflections(cell, 3.)
    brute = {x for x in itertools.product(range(-5,6), repeat=3)
             if x != (0,0,0) and np.linalg.norm(np.array(x) @ b) <= 3.+1e-12}
    assert set(map(tuple,h)) == brute
    np.testing.assert_allclose(vectors, h @ b)
    assert np.all(np.diff(q) >= 0)


def test_bcc_extinctions_and_normalized_shell_multiplicity():
    a = 4.
    host = AtomCloud(('C','C'), [[0,0,0],[a/2]*3])
    reflection = bragg_reflections(host, np.eye(3)*a, 2.4)
    odd = np.sum(reflection['hkl'],axis=1) % 2 != 0
    assert np.max(reflection['f_squared'][odd]) < 1e-25
    np.testing.assert_allclose(reflection['diffuse_at_reflections'], 0.)
    np.testing.assert_allclose(reflection['bragg_weight'], (2*np.pi)**3/a**3*reflection['f_squared'])
    shells = powder_shells(reflection)
    assert list(shells['multiplicity']) == [6, 12]
    np.testing.assert_allclose(shells['radial_weight'],
        (2*np.pi)**3/a**3 * shells['sum_f_squared']/(4*np.pi*shells['q']**2))
    assert reflection['forward_weight'] > 0


def test_bernoulli_diffuse_and_independent_cells_exact_enumeration():
    cell = np.eye(3)*5
    host = AtomCloud(('O',), [[.1,.2,.3]])
    guest = AtomCloud(('Au',), [[0,0,0]])
    p = .3
    pop = PorePopulation([[1,1,1]], ParticleLibrary((guest,), [1]), occupancy=p)
    vectors = np.array([[0.,0,0],[.3,.2,.1],[1.1,.4,.2]])
    moments = guest_moments(pop, vectors)
    f = form_factors(('Au',), np.linalg.norm(vectors,axis=1))['Au']
    np.testing.assert_allclose(moments.variance,p*(1-p)*f**2,rtol=1e-13)
    mask = np.ones((2,1,1),int)
    expected = np.zeros(len(vectors))
    for flags in itertools.product((0,1),repeat=2):
        clouds = [host.moved([5*i,0,0]) for i in range(2)]
        clouds += [guest.moved([1+5*i,1,1]) for i,flag in enumerate(flags) if flag]
        weight = np.prod([p if flag else 1-p for flag in flags])
        expected += weight*intensity(concatenate(clouds), vectors)
    got = finite_cell_ensemble(host,cell,mask,vectors,moments)
    np.testing.assert_allclose(got['total'],expected,rtol=1e-13)
    # Repeating a single random cell replaces M*v by |S|^2*v and is a different model.
    repeated = abs(lattice_shape_amplitude(cell,mask,vectors))**2*(abs(amplitude(host,vectors)+moments.mean)**2+moments.variance)
    assert not np.allclose(repeated,expected)
    with pytest.raises(ValueError,match='Correlated'):
        finite_cell_ensemble(host,cell,mask,vectors,moments,independent_cells=False)
    q = np.array([0.,.5,1.])
    np.testing.assert_allclose(powder_diffuse(pop,q,n_mu=6,n_phi=12),
        p*(1-p)*form_factors(('Au',),q)['Au']**2,rtol=1e-13)


def test_finite_to_bulk_integrated_measure():
    # Integrate the finite Fejer kernel over the reciprocal primitive cell.
    # Its mass per cell is exactly (2*pi)^3/V; weighted continuous observables
    # converge to their value at G as all three crystal dimensions grow.
    cell = np.diag([4.,5.,6.]); volume = np.linalg.det(cell)
    grid = -np.pi + (np.arange(192)+.5)*2*np.pi/192
    errors = []
    for n in (2,4,8,16):
        kernel = (np.sin(n*grid/2)/np.sin(grid/2))**2/n
        np.testing.assert_allclose(kernel.mean(),1.,atol=2e-14)
        integral = (2*np.pi)**3/volume * kernel.mean()**3
        np.testing.assert_allclose(integral,(2*np.pi)**3/volume,rtol=1e-13)
        observable = np.mean(kernel*np.cos(grid))**3
        errors.append(abs(1-observable))
    assert np.all(np.diff(errors)<0)
    assert errors[-1] < .18


def test_sampling_bias_correction_uses_shape_not_cell_count():
    qv = np.array([[0.,0,0],[.2,.3,.4]])
    host = AtomCloud((),np.empty((0,3)))
    moments = GuestMoments(np.array([2+0j,2+0j]),np.array([3.,3.]),np.array([.5,.5]),{})
    mask = np.ones((3,1,1),int); cell=np.eye(3)*4
    result=finite_cell_ensemble(host,cell,mask,qv,moments)
    s2=abs(lattice_shape_amplitude(cell,mask,qv))**2
    np.testing.assert_allclose(result['total'],s2*(4-.5)+3*3)


def test_empty_reflection_table_and_search_guard():
    host=AtomCloud(('C',),[[0,0,0]])
    result=bragg_reflections(host,np.eye(3),.1)
    assert result['hkl'].shape==(0,3)
    assert len(powder_shells(result)['q'])==0
    with pytest.raises(ValueError,match='candidates'):
        enumerate_reflections(np.eye(3)*100,100)


def test_periodic_walls_validate_neighbor_images_and_require_cell():
    particle = AtomCloud(('Au',), [[0,0,0]])
    library = ParticleLibrary((particle,), [1])
    single = PorePopulation([[0,0,0]], library, pore_radius_A=2.1)
    with pytest.raises(ValueError, match='requires cell vectors'):
        cell_guest_moments(single, [[0,0,0]], method='sampled', n_samples=2)
    with pytest.raises(ValueError, match='Overlapping periodic'):
        validate_periodic_population(single, np.eye(3)*4)
    # These sites are separated in the reference cell but touch neighboring
    # cell faces, so checking only the reference-cell pair is insufficient.
    face_pair = PorePopulation([[0.2,0,0],[3.8,0,0]], library, pore_radius_A=0.3)
    with pytest.raises(ValueError, match='Overlapping periodic'):
        validate_periodic_population(face_pair, np.eye(3)*4)
    # The shortest lattice translation can require indices beyond [-1,1].
    skew = np.array([[1.,0,0],[5.1,0.2,0],[0,0,4]])
    small = PorePopulation([[0,0,0]], library, pore_radius_A=0.15)
    with pytest.raises(ValueError, match='Overlapping periodic'):
        validate_periodic_population(small, skew)
    valid = PorePopulation([[0,0,0]], library, pore_radius_A=1)
    got = cell_guest_moments(valid, [[0,0,0]], method='sampled', n_samples=2, cell=np.eye(3)*4)
    assert got.variance[0] == 0


def test_actual_finite_ensemble_converges_under_bulk_reciprocal_measure():
    # Constant cell moments isolate the lattice measure from atomic form-factor
    # variation. Integrate the actual implementation, including diffuse, over
    # one triclinic reciprocal primitive cell; do not compare delta peak heights.
    cell = np.array([[4.,0,0],[0.7,5.,0],[0.2,0.6,6.]])
    phases = -np.pi + (np.arange(16) + 0.5) * 2 * np.pi / 16
    phase_grid = np.stack(np.meshgrid(phases, phases, phases, indexing='ij'), axis=-1).reshape(-1, 3)
    vectors = phase_grid / (2 * np.pi) @ reciprocal_lattice(cell)
    moments = GuestMoments(np.ones(len(vectors), dtype=complex), np.full(len(vectors), 0.3),
                           np.zeros(len(vectors)), {'method': 'exact constant-cell fixture'})
    host = AtomCloud((), np.empty((0,3)))
    measure = (2 * np.pi)**3 / abs(np.linalg.det(cell))
    observable = np.prod(np.cos(phase_grid), axis=1)
    errors = []
    for size in (2, 4, 8):
        count = size**3
        result = finite_cell_ensemble(host, cell, np.ones((size,size,size), dtype=int), vectors, moments)
        np.testing.assert_allclose(measure * result['coherent'].mean() / count, measure, rtol=2e-13)
        np.testing.assert_allclose(result['diffuse'] / count, 0.3, rtol=2e-13)
        np.testing.assert_allclose(measure * result['total'].mean() / count, 1.3 * measure, rtol=2e-13)
        integrated_observable = (result['total'] * observable).mean() / count
        np.testing.assert_allclose(integrated_observable, (1 - 1 / size)**3, rtol=2e-13)
        errors.append(abs(1 - integrated_observable))
    assert np.all(np.diff(errors) < 0)
