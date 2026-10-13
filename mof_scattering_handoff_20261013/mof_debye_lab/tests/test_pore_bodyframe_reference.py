"""Compare body-frame rejection to the original rotation/placement sampler.

For isotropic zero-mean displacement u and independent Haar R, v=R.T@u
is isotropic and independent of R. A spherical wall obeys
|R@x_i+u|=|x_i+v|, so conditioning leaves R Haar and moves the entire rejection
problem to the template frame. Off-centred, mixed-element asymmetric templates
are used here to detect assumptions that only hold for symmetric nanoparticles.
"""
import numpy as np
import pytest

from mofdebye.core import AtomCloud, form_factors
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye import radial_guests


def _radial_observations(radii, template, q):
    factors = form_factors(template.elements, q)
    per_atom = np.array([factors[element] for element in template.elements])
    return (np.sinc(radii[:, :, None]*q[None, None, :]/np.pi)
            * per_atom[None, :, :]).sum(axis=1)


@pytest.mark.parametrize('position_mode', ['uniform_ball', 'gaussian'])
def test_bodyframe_matches_original_sampler_for_asymmetric_offcentre_guest(position_mode, request):
    # Neither inversion-symmetric nor centred geometrically or by atomic mass.
    template = AtomCloud(('Au', 'C', 'O', 'Zn'),
        [[1.15, .15, -.05], [.2, .95, .1], [-.45, .2, .7], [.1, -.1, -.55]])
    centre = np.array([3.4, -1.7, 2.3])
    population = PorePopulation([centre], ParticleLibrary((template,), [1.]),
        occupancy=1., position_mode=position_mode, uniform_ball_radius_A=1.2,
        displacement_sigma_A=.65, pore_radius_A=1.6, wall_clearance_A=.1,
        orientation='uniform')
    count = 4096
    q = np.linspace(0, 6, 121)
    original_rng = np.random.default_rng(202610130)
    original_positions = np.empty((count, len(template), 3))
    original_attempts = 0
    for draw_index in range(count):
        draw = population.sample(original_rng)[0]
        original_positions[draw_index] = draw.cloud.xyz-centre
        original_attempts += draw.attempts
    original_radii = np.linalg.norm(original_positions, axis=2)
    fast_radii, fast_attempts = radial_guests._body_frame_radii(
        template, population, 0, count, np.random.default_rng(202610131))
    assert fast_radii.shape == original_radii.shape
    assert np.isfinite(fast_radii).all() and np.all(fast_radii >= 0)
    assert np.max(fast_radii) <= 1.5+1e-12
    assert np.max(original_radii) <= 1.5+1e-12
    assert fast_attempts >= count
    original = _radial_observations(original_radii, template, q)
    fast = _radial_observations(fast_radii, template, q)
    original_mean, fast_mean = original.mean(axis=0), fast.mean(axis=0)
    combined_sem = np.sqrt((original.var(axis=0, ddof=1)+fast.var(axis=0, ddof=1))/count)
    error = abs(fast_mean-original_mean)
    assert np.all(error[1:] <= 5*combined_sem[1:]+1e-10), float(np.max(error[1:]/combined_sem[1:]))
    np.testing.assert_allclose(fast_mean[0], original_mean[0], rtol=2e-13)
    # Attempt accounting should describe the same conditional acceptance law.
    original_acceptance, fast_acceptance = count/original_attempts, count/fast_attempts
    acceptance_sem = np.sqrt((original_acceptance**2*(1-original_acceptance)
                             +fast_acceptance**2*(1-fast_acceptance))/count)
    assert abs(original_acceptance-fast_acceptance) <= 5*acceptance_sem

    # Independently check that the original full directional amplitude has the
    # real radial mean used by the accelerated model. Its imaginary mean is zero.
    chosen = [6, 22, 46, 94]
    q_selected = q[chosen]
    factors = form_factors(template.elements, q_selected)
    per_atom = np.array([factors[element] for element in template.elements])
    amplitudes = (np.exp(1j*original_positions[:, :, 0, None]*q_selected[None, None, :])
                  *per_atom[None, :, :]).sum(axis=1)
    directional_mean = amplitudes.mean(axis=0)
    directional_variance = np.mean(abs(amplitudes-directional_mean)**2, axis=0)*count/(count-1)
    directional_sem = np.sqrt(directional_variance/count+fast.var(axis=0, ddof=1)[chosen]/count)
    assert np.all(abs(directional_mean-fast_mean[chosen]) <= 5*directional_sem)
    request.node.user_properties.extend([
        ('position_mode', position_mode), ('accepted_draws_per_sampler', count),
        ('maximum_radial_combined_standard_errors', float(np.max(error[1:]/combined_sem[1:]))),
        ('maximum_directional_combined_standard_errors', float(np.max(abs(directional_mean-fast_mean[chosen])/directional_sem))),
        ('original_acceptance_fraction', original_acceptance),
        ('bodyframe_acceptance_fraction', fast_acceptance),
    ])


def test_bodyframe_small_workspace_is_seed_reproducible():
    template = AtomCloud(('Au', 'O', 'C'), [[.9, .3, .1], [-.2, .7, .4], [.1, -.4, .6]])
    population = PorePopulation([[0., 0., 0.]], ParticleLibrary((template,), [1.]),
        position_mode='uniform_ball', uniform_ball_radius_A=1.1,
        pore_radius_A=1.4, orientation='uniform')
    # Verify a deliberately small workspace with the same seed and settings;
    # changing batch layout need only preserve the probability law.
    radii, attempts = radial_guests._body_frame_radii(template, population, 0, 256,
        np.random.default_rng(1234), max_atom_placement_pairs=96)
    repeated, repeated_attempts = radial_guests._body_frame_radii(template, population, 0, 256,
        np.random.default_rng(1234), max_atom_placement_pairs=96)
    np.testing.assert_array_equal(radii, repeated)
    assert attempts == repeated_attempts
    assert radii.shape == (256, 3)
    assert np.max(radii) <= 1.4+1e-12
