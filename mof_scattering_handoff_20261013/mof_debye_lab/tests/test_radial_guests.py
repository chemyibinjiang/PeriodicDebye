"""Independent physical references for conditional isotropic guest statistics."""
import numpy as np
import pytest
from scipy.integrate import quad

from mofdebye.core import AtomCloud, form_factors
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.radial_guests import radial_guest_statistics


def population(template=None, **kwargs):
    template = template or AtomCloud(('Au',), [[0, 0, 0]])
    return PorePopulation([[0, 0, 0]], ParticleLibrary((template,), [1]), **kwargs)


def test_gaussian_monomer_exact_characteristic_and_conditional_occupancy():
    q = np.linspace(0, 6, 121)
    pop = population(occupancy=.13, displacement_sigma_A=.4, mean_displacement_A=[.2, -.3, .4])
    stats = radial_guest_statistics(pop, q)
    factor = form_factors(('Au',), q)['Au']
    np.testing.assert_allclose(stats.means[0, 0, 0], factor * np.exp(-.5*.4**2*q*q), rtol=2e-15)
    np.testing.assert_array_equal(stats.means[:, 0], stats.means[:, 1])
    np.testing.assert_allclose(stats.internal[0], factor**2, rtol=2e-15)
    np.testing.assert_array_equal(stats.centres, [[.2, -.3, .4]])
    np.testing.assert_array_equal(stats.occupancy, [.13])
    assert not np.any(stats.mean_bounds)
    assert not np.any(stats.internal_bounds)
    assert stats.metadata['batches'] == 1
    assert stats.metadata['accepted_placements'] == 0


def test_dimer_rotation_and_size_weights_have_closed_forms():
    q = np.linspace(0, 6, 151)
    gold = AtomCloud(('Au',), [[0, 0, 0]])
    carbon = AtomCloud(('C', 'C'), [[-.5, 0, 0], [.5, 0, 0]])
    pop = PorePopulation([[0, 0, 0], [5, 0, 0]], ParticleLibrary((gold, carbon), [.25, .75]), occupancy=[0, .6])
    stats = radial_guest_statistics(pop, q)
    factors = form_factors(('Au', 'C'), q)
    mean = .25*factors['Au'] + .75*2*factors['C']*np.sinc(.5*q/np.pi)
    internal = .25*factors['Au']**2 + .75*2*factors['C']**2*(1+np.sinc(q/np.pi))
    np.testing.assert_allclose(stats.means, np.broadcast_to(mean, (1, 2, 2, len(q))), rtol=2e-15)
    np.testing.assert_allclose(stats.internal, np.broadcast_to(internal, (2, len(q))), rtol=2e-15)


def test_uniform_ball_characteristic_near_zero_and_nonbinding_wall():
    q = np.r_[0, 1e-12, 1e-5, .1, 1, 3, 6]
    pop = population(position_mode='uniform_ball', uniform_ball_radius_A=.7,
                     pore_radius_A=1., wall_clearance_A=.1, mean_displacement_A=[.1, 0, 0])
    stats = radial_guest_statistics(pop, q)
    expected = np.array([3*quad(lambda r: r*r*np.sinc(x*r/np.pi), 0, .7,
                                epsabs=1e-14, epsrel=1e-13)[0]/.7**3 for x in q])
    factor = form_factors(('Au',), q)['Au']
    np.testing.assert_allclose(stats.means[0, 0, 0], factor*expected, rtol=4e-14, atol=2e-13)
    assert 'certified_nonbinding' in stats.metadata['spherical_wall']


def test_zero_variance_gaussian_wall_certificate_ignores_zero_weight_template():
    small = AtomCloud(('C',), [[.2, 0, 0]])
    huge = AtomCloud(('Au',), [[100, 0, 0]])
    pop = PorePopulation([[0, 0, 0]], ParticleLibrary((small, huge), [1, 0]),
                         mean_displacement_A=[.3, 0, 0], pore_radius_A=.6)
    stats = radial_guest_statistics(pop, np.linspace(0, 6, 101))
    assert stats.metadata['n_positive_weight_templates'] == 1
    assert stats.metadata['spherical_wall'].startswith('certified_nonbinding')


@pytest.mark.parametrize('kwargs', [
    dict(displacement_sigma_A=.01, pore_radius_A=100),
    dict(position_mode='uniform_ball', uniform_ball_radius_A=1, pore_radius_A=.5),
    dict(mean_displacement_A=[1, 0, 0], pore_radius_A=.5),
])
def test_binding_or_gaussian_tail_wall_never_uses_unconditioned_formula(kwargs):
    with pytest.raises(ValueError, match='wall'):
        radial_guest_statistics(population(**kwargs), np.linspace(0, 6, 101))


def test_truncated_gaussian_sampling_matches_independent_radial_quadrature():
    q = np.linspace(0, 6, 151)
    sigma, radius = .4, .5
    pop = population(displacement_sigma_A=sigma, pore_radius_A=radius)
    stats = radial_guest_statistics(pop, q, method='sampled', n_samples=512, n_batches=8, seed=819)
    normalizer = quad(lambda r: r*r*np.exp(-r*r/(2*sigma*sigma)), 0, radius, epsabs=1e-13)[0]
    selected = np.arange(0, len(q), 10)
    reference = np.array([quad(lambda r: r*r*np.exp(-r*r/(2*sigma*sigma))*np.sinc(x*r/np.pi),
                               0, radius, epsabs=1e-13)[0]/normalizer for x in q[selected]])
    factor = form_factors(('Au',), q[selected])['Au']
    batches = stats.means[:, :, 0, :].reshape(16, len(q))[:, selected]
    sem = batches.std(axis=0, ddof=1)/np.sqrt(len(batches))
    np.testing.assert_array_less(abs(batches.mean(axis=0)-factor*reference), 4*sem+1e-10)
    # A binding wall changes the distribution: the unbounded formula is not used.
    unbounded = factor*np.exp(-.5*sigma*sigma*q[selected]**2)
    assert np.max(abs(batches.mean(axis=0)-unbounded)) > 3
    assert stats.metadata['accepted_placements'] == 2*512
    assert stats.metadata['proposal_attempts'] > stats.metadata['accepted_placements']


def test_sampled_sizes_are_exactly_weighted_and_replicas_reproducible():
    q = np.linspace(0, 6, 101)
    small = AtomCloud(('Au',), [[0, 0, 0]])
    dimer = AtomCloud(('C', 'C'), [[-.2, 0, 0], [.2, 0, 0]])
    pop = PorePopulation([[0, 0, 0]], ParticleLibrary((small, dimer), [.3, .7]), occupancy=.2,
                         position_mode='uniform_ball', uniform_ball_radius_A=.7, pore_radius_A=.5)
    options = dict(method='sampled', n_samples=64, n_batches=4, seed=923)
    first = radial_guest_statistics(pop, q, **options)
    second = radial_guest_statistics(pop, q, **options)
    np.testing.assert_array_equal(first.means, second.means)
    np.testing.assert_array_equal(first.mean_bounds, second.mean_bounds)
    assert not np.array_equal(first.means[:, 0, :, 1:], first.means[:, 1, :, 1:])
    factors = form_factors(('Au', 'C'), q)
    forward = .3*factors['Au'][0]+.7*2*factors['C'][0]
    np.testing.assert_array_equal(first.means[..., 0], np.full((4, 2, 1), forward))
    expected_internal = .3*factors['Au']**2+.7*2*factors['C']**2*(1+np.sinc(.4*q/np.pi))
    np.testing.assert_allclose(first.internal[0], expected_internal, rtol=2e-15)
    assert first.metadata['accepted_placements'] == 2*64*2
    assert first.metadata['accepted_samples_per_template_pore_replica_batch'] == 16
    assert not first.means.flags.writeable
    assert not first.occupancy.flags.writeable
    assert pop.occupancy[0] == .2


@pytest.mark.parametrize('method', ['analytic', 'sampled'])
@pytest.mark.parametrize('kwargs', [dict(orientation='fixed'), dict(orientation='wobble'),
                                  dict(displacement_covariance_A2=np.diag([.1, .2, .3]))])
def test_asymmetric_proposals_rejected(method, kwargs):
    with pytest.raises(ValueError):
        radial_guest_statistics(population(**kwargs), np.linspace(0, 6, 101), method=method)


@pytest.mark.parametrize('method', ['analytic', 'sampled'])
def test_host_clearance_rejected(method):
    pop = population(host=AtomCloud(('C',), [[5, 0, 0]]), min_host_distance_A=.5)
    with pytest.raises(ValueError, match='Host-clearance'):
        radial_guest_statistics(pop, np.linspace(0, 6, 101), method=method)


@pytest.mark.parametrize('method', ['analytic', 'sampled'])
def test_external_boundary_rejected(method):
    boundary = {'kind': 'sphere', 'centre_A': [0, 0, 0], 'radius_A': 5.}
    pop = population(external_boundary=boundary, guest_retention='whole_particle')
    with pytest.raises(ValueError, match='External-boundary'):
        radial_guest_statistics(pop, np.linspace(0, 6, 101), method=method)


def test_uniform_ball_conditioning_preserves_each_pore_wall_radius():
    q = np.linspace(0, 6, 101)
    template = AtomCloud(('Au',), [[0, 0, 0]])
    radii = [.3, .5]
    pop = PorePopulation([[0, 0, 0], [5, 0, 0]], ParticleLibrary((template,), [1]),
                         position_mode='uniform_ball', uniform_ball_radius_A=.7, pore_radius_A=radii)
    stats = radial_guest_statistics(pop, q, method='sampled', n_samples=128, n_batches=4, seed=881)
    factor = form_factors(('Au',), q)['Au']
    for pore, radius in enumerate(radii):
        # Conditioning a uniform ball by a concentric smaller ball is itself
        # uniform in the smaller ball; independent radial quadrature supplies b.
        reference = np.array([3*quad(lambda r: r*r*np.sinc(x*r/np.pi), 0, radius)[0]/radius**3
                              for x in q])*factor
        batches = stats.means[:, :, pore].reshape(8, len(q))
        sem = batches.std(axis=0, ddof=1)/np.sqrt(8)
        np.testing.assert_array_less(abs(batches.mean(axis=0)-reference), 5*sem+1e-10)
    assert stats.means[:, :, 0, -1].mean() > stats.means[:, :, 1, -1].mean()
    assert stats.metadata['accepted_placements'] == 2*128*2


def test_sampled_nonzero_mean_shift_rejected():
    with pytest.raises(ValueError, match='zero mean'):
        radial_guest_statistics(population(mean_displacement_A=[.01, 0, 0]),
                                np.linspace(0, 6, 101), method='sampled')


@pytest.mark.parametrize('options', [dict(n_batches=1), dict(n_samples=5, n_batches=2),
                                    dict(n_samples=4, n_batches=4), dict(n_samples=True),
                                    dict(seed=-1), dict(seed=True), dict(method='unknown'),
                                    dict(order=3), dict(order=True), dict(bin_width=0),
                                    dict(bin_width=True)])
def test_sampling_controls_validated(options):
    arguments = dict(method='sampled')
    arguments.update(options)
    with pytest.raises(ValueError):
        radial_guest_statistics(population(), np.linspace(0, 6, 101), **arguments)


def test_impossible_fixed_size_remains_impossible_instead_of_size_reweighting():
    dimer = AtomCloud(('C', 'C'), [[-1, 0, 0], [1, 0, 0]])
    pop = population(dimer, pore_radius_A=.5, max_attempts=3)
    with pytest.raises(RuntimeError, match='No valid placement'):
        radial_guest_statistics(pop, np.linspace(0, 6, 101), method='sampled', n_samples=4, n_batches=2)


def test_custom_population_subclass_is_not_silently_cloned():
    class CorrelatedPopulation(PorePopulation):
        pass
    template = AtomCloud(('Au',), [[0, 0, 0]])
    pop = CorrelatedPopulation([[0, 0, 0]], ParticleLibrary((template,), [1]))
    with pytest.raises(ValueError, match='custom joint samplers'):
        radial_guest_statistics(pop, np.linspace(0, 6, 101))
