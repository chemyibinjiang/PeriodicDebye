"""Bounded proposal streams and rejection semantics of the exact radial sampler."""
import numpy as np
import pytest

from mofdebye.core import AtomCloud
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.radial_guests import _body_frame_radii


def _population(template, **kwargs):
    return PorePopulation([[0, 0, 0]], ParticleLibrary((template,), [1]), **kwargs)


class _ScriptedGaussian:
    """Known proposal order, with rejected padding for vectorized oversampling."""

    def __init__(self, x_values):
        self.x_values = list(x_values)
        self.position = 0
        self.requests = []

    def normal(self, size):
        self.requests.append(size)
        result = np.zeros(size)
        for row in range(size[0]):
            index = self.position + row
            result[row, 0] = self.x_values[index] if index < len(self.x_values) else 2.
        self.position += size[0]
        return result


@pytest.mark.parametrize('mode', ['gaussian', 'uniform_ball'])
def test_binding_wall_support_and_seed_reproducibility(mode):
    template = AtomCloud(('C', 'O', 'Zn'), [[.2, 0, 0], [0, .3, 0], [-.2, -.1, .1]])
    pop = _population(template, position_mode=mode, displacement_sigma_A=.6,
                      uniform_ball_radius_A=1.2, pore_radius_A=1., wall_clearance_A=.1)
    options = dict(max_atom_placement_pairs=9)
    first, attempts = _body_frame_radii(template, pop, 0, 100, np.random.default_rng(817), **options)
    second, repeat_attempts = _body_frame_radii(template, pop, 0, 100, np.random.default_rng(817), **options)
    np.testing.assert_array_equal(first, second)
    assert attempts == repeat_attempts
    assert attempts > 100
    assert first.shape == (100, 3)
    assert np.all(first <= .9)


@pytest.mark.parametrize('options', [
    dict(displacement_sigma_A=0.),
    dict(displacement_covariance_A2=np.eye(3)*-1e-13),
    dict(position_mode='uniform_ball', uniform_ball_radius_A=0.),
])
def test_zero_displacement_and_tolerated_negative_covariance_are_exact(options, monkeypatch):
    template = AtomCloud(('C', 'O'), [[0, 0, 0], [.3, .4, 0]])
    pop = _population(template, pore_radius_A=.5, **options)

    def forbidden_rigid_sampler(*args, **kwargs):
        raise AssertionError('The exact isotropic path must not construct rigid placements.')

    monkeypatch.setattr(PorePopulation, 'sample', forbidden_rigid_sampler)
    with np.errstate(invalid='raise'):
        radii, attempts = _body_frame_radii(template, pop, 0, 7, np.random.default_rng(4),
                                            max_atom_placement_pairs=4)
    np.testing.assert_array_equal(radii, np.tile([0., .5], (7, 1)))
    assert attempts == 7


def test_proposal_workspace_is_bounded_by_atom_placement_pairs():
    template = AtomCloud(('C', 'O', 'Zn'), [[0, 0, 0], [.2, 0, 0], [0, .3, 0]])
    pop = _population(template, displacement_sigma_A=1.)
    rng = _ScriptedGaussian([0.] * 7)
    radii, attempts = _body_frame_radii(template, pop, 0, 7, rng, max_atom_placement_pairs=6)
    assert all(shape[0] * len(template) <= 6 for shape in rng.requests)
    np.testing.assert_array_equal(radii, np.tile([0., .2, .3], (7, 1)))
    assert attempts == 7


def test_rejection_limit_resets_after_each_acceptance_across_chunks():
    template = AtomCloud(('C',), [[0, 0, 0]])
    pop = _population(template, displacement_sigma_A=1., pore_radius_A=1., max_attempts=2)
    radii, attempts = _body_frame_radii(template, pop, 0, 2, _ScriptedGaussian([2., .2, 2., .4]),
                                        max_atom_placement_pairs=1)
    np.testing.assert_array_equal(radii[:, 0], [.2, .4])
    assert attempts == 4


@pytest.mark.parametrize('pair_cap', [1, 32])
def test_limit_fails_before_later_acceptance_within_or_across_chunks(pair_cap):
    template = AtomCloud(('C',), [[0, 0, 0]])
    pop = _population(template, displacement_sigma_A=1., pore_radius_A=1., max_attempts=2)
    with pytest.raises(RuntimeError, match='after 2 attempts'):
        _body_frame_radii(template, pop, 0, 1, _ScriptedGaussian([2., 2., .2]),
                          max_atom_placement_pairs=pair_cap)


def test_unused_proposals_after_requested_acceptance_do_not_trigger_limit():
    template = AtomCloud(('C',), [[0, 0, 0]])
    pop = _population(template, displacement_sigma_A=1., pore_radius_A=1., max_attempts=2)
    rng = _ScriptedGaussian([.2])
    radii, attempts = _body_frame_radii(template, pop, 0, 1, rng, max_atom_placement_pairs=32)
    np.testing.assert_array_equal(radii, [[.2]])
    assert rng.requests == [(32, 3)]
    assert attempts == 1
