import numpy as np
import pytest

from magnetooptics.geometry import ParticleGeometry, grid_geometry, from_config


def test_finite_array_dimensions_count_and_gap():
    g = grid_geometry([12, 9, 1], [30, 40, 50], 5, origin_nm=[-20, 2, 100])
    a = g.audit()
    assert a['particle_count'] == 108
    assert a['extent_nm'] == [340, 330, 10]
    assert a['minimum_surface_gap_nm'] == 20
    assert not g.positions_nm.flags.writeable


def test_variable_radii_gap_against_all_pairs():
    rng = np.random.default_rng(312)
    indices = np.indices((5, 4, 3)).reshape(3, -1).T
    radii = rng.uniform(0.1, 8, len(indices))
    g = ParticleGeometry(indices * 30, radii)
    d = np.linalg.norm(g.positions_nm[:, None] - g.positions_nm[None], axis=-1)
    gap = d - radii[:, None] - radii[None]
    np.fill_diagonal(gap, np.inf)
    assert g.audit()['minimum_surface_gap_nm'] == pytest.approx(gap.min())


@pytest.mark.parametrize('positions,radii', [([[0, 0, 0], [0, 0, 0]], 1), ([[0, 0, 0], [2, 0, 0]], 1), ([[0, 0, 0], [20, 0, 0]], [1, 20]), ([[np.nan, 0, 0]], 1)])
def test_bad_geometry_rejected(positions, radii):
    with pytest.raises(ValueError):
        ParticleGeometry(positions, radii)


def test_sparse_and_ellipsoid_masks():
    sparse = grid_geometry([10, 6, 4], 20, 2, active_indices=[[0, 0, 0], [9, 5, 3]])
    assert sparse.audit()['grid_shape'] == [10, 6, 4]
    ball = grid_geometry([7, 7, 7], 20, 2, mask='ellipsoid')
    assert 0 < len(ball.positions_nm) < 343
    centre = np.mean(ball.positions_nm, axis=0)
    assert np.allclose(centre, [60, 60, 60])


@pytest.mark.parametrize('shape', [[2.1, 3, 1], [True, True, True], [0, 1, 1], [1000, 1000, 1000]])
def test_grid_size_guards(shape):
    with pytest.raises(ValueError):
        grid_geometry(shape, 10, 1)


def test_typo_and_incompatible_fields_fail():
    with pytest.raises(ValueError, match='unknown'):
        from_config({'shpae': [2, 2, 1]})
    with pytest.raises(ValueError, match='explicit'):
        from_config({'kind': 'explicit', 'shape': [1, 1, 1], 'positions_nm': [[0, 0, 0]], 'radius_nm': 1})


@pytest.mark.parametrize('radius', [True, '3', [True, 3], 3+1j])
def test_radius_is_a_real_physical_number(radius):
    with pytest.raises(ValueError):
        ParticleGeometry([[0, 0, 0], [20, 0, 0]], radius)
