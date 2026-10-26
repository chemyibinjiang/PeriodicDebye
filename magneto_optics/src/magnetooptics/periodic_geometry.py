"""Coplanar spherical particles in a rectangular two-dimensional unit cell.

The physical array is infinite in x and y.  Each sphere is still represented
by one electric dipole, and the common plane is strictly z=0.  Positions are
required inside the supplied cell; no coordinate snapping or wrapping occurs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    if not isinstance(value, np.ndarray) and any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError(f"{name} must not contain booleans")
    result = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must be finite")
    return result


def validate_periods(periods_nm):
    periods = _real_array(periods_nm, "periods_nm")
    if periods.shape != (2,) or np.any(periods <= 0):
        raise ValueError("periods_nm must contain two positive rectangular-cell lengths")
    if not np.isfinite(periods[0] * periods[1]):
        raise ValueError("unit-cell area overflows; check physical lengths")
    return periods


@dataclass(frozen=True)
class RectangularCell:
    periods_nm: np.ndarray
    positions_nm: np.ndarray
    radii_nm: np.ndarray

    def __post_init__(self):
        periods = validate_periods(self.periods_nm)
        positions = _real_array(self.positions_nm, "positions_nm")
        if positions.ndim != 2 or positions.shape[1:] != (3,) or not len(positions):
            raise ValueError("positions_nm must be a nonempty (N,3) array")
        if np.any(positions[:, 2] != 0):
            raise ValueError("this periodic model requires every particle centre at z=0")
        if np.any(positions[:, :2] < 0) or np.any(positions[:, :2] >= periods):
            raise ValueError("particle x/y coordinates must lie in [0,Lx) and [0,Ly); no automatic wrapping")
        try:
            radii = np.array(np.broadcast_to(_real_array(self.radii_nm, "radii_nm"), (len(positions),)), copy=True)
        except ValueError as error:
            raise ValueError("radii_nm must be scalar or contain one radius per particle") from error
        if np.any(radii <= 0):
            raise ValueError("radii_nm must be positive")
        if len(np.unique(positions, axis=0)) != len(positions):
            raise ValueError("duplicate particle centres in periodic cell")
        for name, value in (("periods_nm", periods), ("positions_nm", positions), ("radii_nm", radii)):
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        self.audit()

    @property
    def area_nm2(self):
        return float(np.prod(self.periods_nm))

    @property
    def plane_z_nm(self):
        return 0.0

    def audit(self):
        """Check every potentially touching neighbour, including replica cells.

        For unequal radii the nearest centres need not give the smallest gap.
        Periodic KD-tree queries use a proven current-gap upper bound and keep
        candidate allocation bounded to one particle's neighbours at a time.
        """
        maximum_radius = float(self.radii_nm.max())
        minimum_distance = float(self.periods_nm.min())
        minimum_gap = minimum_distance - 2 * maximum_radius  # Self images.
        if len(self.positions_nm) > 1:
            tree = cKDTree(self.positions_nm[:, :2], boxsize=self.periods_nm)
            distances, neighbours = tree.query(self.positions_nm[:, :2], k=2)
            minimum_distance = min(minimum_distance, float(distances[:, 1].min()))
            gaps = distances[:, 1] - self.radii_nm - self.radii_nm[neighbours[:, 1]]
            minimum_gap = min(minimum_gap, float(gaps.min()))
            if not np.all(self.radii_nm == self.radii_nm[0]):
                for i, point in enumerate(self.positions_nm[:, :2]):
                    bound = self.radii_nm[i] + maximum_radius + max(minimum_gap, 0)
                    candidates = np.asarray(tree.query_ball_point(point, bound + 1e-10), dtype=int)
                    candidates = candidates[candidates != i]
                    if len(candidates):
                        difference = np.abs(self.positions_nm[candidates, :2] - point)
                        difference = np.minimum(difference, self.periods_nm - difference)
                        separation = np.linalg.norm(difference, axis=1)
                        minimum_gap = min(minimum_gap, float(np.min(separation - self.radii_nm[i] - self.radii_nm[candidates])))
        tolerance = 1e-10 * max(1.0, maximum_radius)
        if minimum_gap <= tolerance:
            raise ValueError("particles overlap or touch in the periodic array, including neighbouring-cell images")
        return {
            "particle_count_per_cell": len(self.positions_nm),
            "periods_nm": self.periods_nm.tolist(),
            "area_nm2": self.area_nm2,
            "plane_z_nm": 0.0,
            "minimum_center_distance_nm": minimum_distance,
            "minimum_surface_gap_nm": minimum_gap,
            "maximum_radius_nm": maximum_radius,
            "boundary": "infinite two-dimensional rectangular periodic array",
            "representation": "one electric dipole per homogeneous sphere",
            "incidence": "normal, from z<0 toward +z",
            "host": "identical homogeneous lossless medium above and below",
        }
