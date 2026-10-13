"""Changing physical pore geometry must re-run periodic overlap validation."""
from dataclasses import replace
import numpy as np
import pytest
from mofdebye.core import AtomCloud
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.pore_powder import PorePowderCache
from mofdebye.radial_guests import radial_guest_statistics


@pytest.mark.parametrize('change', ['radius', 'clearance', 'nominal_centre', 'missing_signature'])
def test_cache_reuse_rejects_changed_physical_wall_geometry(change):
    pytest.importorskip('numba')
    library = ParticleLibrary((AtomCloud(('Au',), [[0., 0., 0.]]),), [1])
    original = PorePopulation([[2., 2., 2.]], library, occupancy=.5, pore_radius_A=1.)
    q = np.linspace(0, 6, 61)
    cache = PorePowderCache(AtomCloud(('C',), [[0., 0., 0.]]), np.eye(3)*8,
                          np.ones((2, 2, 2), dtype=int), original, q)
    kwargs = dict(occupancy=.5, pore_radius_A=1.)
    centres = [[2., 2., 2.]]
    if change == 'radius':
        kwargs['pore_radius_A'] = 5.  # overlaps the neighbouring cell's pore
    elif change == 'clearance':
        kwargs['wall_clearance_A'] = .1
    elif change == 'nominal_centre':
        centres = [[2.1, 2., 2.]]
        kwargs['mean_displacement_A'] = [-.1, 0., 0.]
    updated = PorePopulation(centres, library, **kwargs)
    stats = radial_guest_statistics(updated, q)
    if change == 'missing_signature':
        metadata = dict(stats.metadata)
        metadata.pop('pore_geometry')
        stats = replace(stats, metadata=metadata)
    with pytest.raises(ValueError, match='pore centres|pore geometry'):
        cache.iq(stats)
    if change == 'radius':
        with pytest.raises(ValueError, match='overlap'):
            PorePowderCache(AtomCloud(('C',), [[0., 0., 0.]]), np.eye(3)*8,
                            np.ones((2, 2, 2), dtype=int), updated, q)


def test_pore_signature_is_a_copy_of_population_geometry():
    library = ParticleLibrary((AtomCloud(('Au',), [[0., 0., 0.]]),), [1])
    population = PorePopulation([[2., 2., 2.]], library, pore_radius_A=1.)
    stats = radial_guest_statistics(population, np.linspace(0, 6, 61))
    population.centres[0, 0] = 3.
    population.pore_radius[0] = 5.
    assert stats.metadata['pore_geometry']['centres_A'] == [[2., 2., 2.]]
    assert stats.metadata['pore_geometry']['pore_radius_A'] == [1.]
