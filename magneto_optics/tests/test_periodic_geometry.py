import numpy as np
import pytest

from magnetooptics.periodic_geometry import RectangularCell


def test_single_site_audit_includes_its_own_periodic_images():
    cell = RectangularCell([90, 120], [[0, 0, 0]], 7)
    audit = cell.audit()
    assert cell.area_nm2 == 10800
    assert audit["minimum_center_distance_nm"] == 90
    assert audit["minimum_surface_gap_nm"] == 76
    assert audit["particle_count_per_cell"] == 1
    assert cell.plane_z_nm == 0


def test_minimum_image_gap_across_cell_edges_and_corners():
    edge = RectangularCell([100, 100], [[2, 40, 0], [99, 40, 0]], 1)
    assert edge.audit()["minimum_surface_gap_nm"] == 1
    corner = RectangularCell([12, 12], [[1, 1, 0], [11, 11, 0]], 1)
    assert corner.audit()["minimum_center_distance_nm"] == pytest.approx(np.sqrt(8))


def test_unequal_radii_gap_is_not_assumed_to_follow_nearest_centres():
    cell = RectangularCell([100, 100], [[0, 0, 0], [13, 0, 0], [35, 0, 0], [48, 0, 0]], [0.1, 10, 10, 0.1])
    assert cell.audit()["minimum_center_distance_nm"] == 13
    assert cell.audit()["minimum_surface_gap_nm"] == 2


@pytest.mark.parametrize("positions,radius", [
    ([[0, 0, 0]], 50),  # Contact with its own x/y image.
    ([[2, 30, 0], [99, 30, 0]], 1.5),  # Contact across the boundary.
    ([[2, 30, 0], [99, 30, 0]], 2),
    ([[0, 0, 0], [5, 0, 0]], [1, 5]),
])
def test_periodic_overlap_and_contact_are_rejected(positions, radius):
    with pytest.raises(ValueError, match="overlap|touch"):
        RectangularCell([100, 100], positions, radius)


@pytest.mark.parametrize("positions", [
    [[100, 0, 0]], [[0, 100, 0]], [[-1e-12, 0, 0]], [[0, 0, 1e-14]],
    [[0, 0, 0], [0, 0, 0]], [[np.nan, 0, 0]], [],
])
def test_invalid_positions_are_never_silently_wrapped_or_projected(positions):
    with pytest.raises(ValueError):
        RectangularCell([100, 100], positions, 1)


@pytest.mark.parametrize("periods", [[0, 100], [-100, 100], [np.inf, 100], [100], [100, True]])
def test_invalid_periods(periods):
    with pytest.raises(ValueError):
        RectangularCell(periods, [[0, 0, 0]], 1)


def test_geometry_owns_immutable_copies_and_periodic_translation_preserves_gap():
    positions = np.array([[2., 10, 0], [90, 80, 0]])
    radii = np.array([3., 4])
    periods = np.array([100., 100])
    cell = RectangularCell(periods, positions, radii)
    shifted = positions.copy()
    shifted[:, :2] = (shifted[:, :2] + [23, 41]) % periods
    translated = RectangularCell(periods, shifted, radii)
    assert translated.audit()["minimum_surface_gap_nm"] == pytest.approx(cell.audit()["minimum_surface_gap_nm"])
    positions[:] = 0
    radii[:] = 30
    periods[:] = 200
    assert cell.positions_nm[1, 0] == 90
    assert cell.radii_nm[0] == 3
    assert cell.periods_nm[0] == 100
    with pytest.raises(ValueError):
        cell.positions_nm[0, 0] = 5
