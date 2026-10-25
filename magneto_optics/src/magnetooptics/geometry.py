"""Finite spherical-particle geometries, in nm. A sphere is one electric dipole."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


def _real_array(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'fiu':
        raise ValueError(f'{name} must contain real numbers, not booleans or strings')
    if not isinstance(value, np.ndarray) and any(isinstance(x, (bool, np.bool_)) for x in np.asarray(value, dtype=object).flat):
        raise ValueError(f'{name} must not contain booleans')
    return np.array(raw, dtype=float, copy=True)


def _vector(value, name, *, positive=False):
    array = _real_array(value, name)
    if array.ndim == 0:
        array = np.repeat(array, 3)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f'{name} must be a finite scalar or length-three vector')
    if positive and np.any(array <= 0):
        raise ValueError(f'{name} must be positive')
    return array


@dataclass(frozen=True)
class ParticleGeometry:
    positions_nm: np.ndarray
    radii_nm: np.ndarray
    indices: np.ndarray | None = None
    spacing_nm: np.ndarray | None = None

    def __post_init__(self):
        pos = _real_array(self.positions_nm, 'positions_nm')
        if pos.ndim != 2 or pos.shape[1] != 3 or len(pos) == 0 or not np.isfinite(pos).all():
            raise ValueError('positions_nm must be nonempty finite (N,3) coordinates')
        radii = np.array(np.broadcast_to(_real_array(self.radii_nm, 'radii_nm'), (len(pos),)), copy=True)
        if not np.isfinite(radii).all() or np.any(radii <= 0):
            raise ValueError('radii_nm must be finite and positive')
        if len(np.unique(pos, axis=0)) != len(pos):
            raise ValueError('duplicate particle centres')
        indices = None
        spacing = None
        if self.indices is not None:
            raw = np.asarray(self.indices)
            if raw.shape != pos.shape or not np.isfinite(raw).all() or not np.equal(raw, np.floor(raw)).all():
                raise ValueError('indices must be integer (N,3) lattice sites')
            indices = np.array(raw, dtype=np.int64)
            spacing = _vector(self.spacing_nm, 'spacing_nm', positive=True)
            offset = pos - indices * spacing
            if not np.allclose(offset, offset[0], rtol=0, atol=1e-10):
                raise ValueError('positions do not match the supplied grid and translation')
        elif self.spacing_nm is not None:
            raise ValueError('spacing_nm requires grid indices')
        for name, value in [('positions_nm', pos), ('radii_nm', radii), ('indices', indices), ('spacing_nm', spacing)]:
            if value is not None:
                value.setflags(write=False)
            object.__setattr__(self, name, value)
        self.audit()  # Reject overlap before any electromagnetic allocation.

    def audit(self):
        n = len(self.positions_nm)
        minimum_distance = None
        minimum_gap = None
        if n > 1:
            tree = cKDTree(self.positions_nm)
            distances, neighbours = tree.query(self.positions_nm, k=2)
            minimum_distance = float(distances[:, 1].min())
            gaps = distances[:, 1] - self.radii_nm - self.radii_nm[neighbours[:, 1]]
            minimum_gap = float(gaps.min())
            if not np.all(self.radii_nm == self.radii_nm[0]):
                # Any pair improving this upper bound must lie in this search ball.
                # Query one site at a time to avoid an O(N^2) candidate allocation.
                maximum_radius = float(self.radii_nm.max())
                for i, position in enumerate(self.positions_nm):
                    bound = self.radii_nm[i] + maximum_radius + max(minimum_gap, 0)
                    candidates = np.asarray(tree.query_ball_point(position, bound + 1e-10), dtype=int)
                    candidates = candidates[candidates != i]
                    if len(candidates):
                        d = np.linalg.norm(self.positions_nm[candidates] - position, axis=1)
                        minimum_gap = min(minimum_gap, float(np.min(d - self.radii_nm[i] - self.radii_nm[candidates])))
            tolerance = 1e-10 * max(1.0, float(self.radii_nm.max()))
            if minimum_gap <= tolerance:
                raise ValueError('particles overlap or touch; separated spheres are required')
        lower = np.min(self.positions_nm - self.radii_nm[:, None], axis=0)
        upper = np.max(self.positions_nm + self.radii_nm[:, None], axis=0)
        return {
            'particle_count': n,
            'representation': 'one electric dipole per homogeneous sphere',
            'boundary': 'finite in an infinite homogeneous lossless host',
            'units': 'nm',
            'bounding_box_min_nm': lower.tolist(),
            'bounding_box_max_nm': upper.tolist(),
            'extent_nm': (upper - lower).tolist(),
            'minimum_center_distance_nm': minimum_distance,
            'minimum_surface_gap_nm': minimum_gap,
            'maximum_radius_nm': float(self.radii_nm.max()),
            'grid_shape': None if self.indices is None else (np.ptp(self.indices, axis=0) + 1).tolist(),
        }


def grid_geometry(shape, spacing_nm, radius_nm, *, mask='box', active_indices=None, origin_nm=(0, 0, 0), max_sites=1_000_000):
    raw = np.asarray(shape)
    if raw.shape != (3,) or raw.dtype.kind not in 'iu' or np.any(raw <= 0):
        raise ValueError('shape must be three positive integers')
    shape = tuple(int(x) for x in raw)
    if np.prod(shape, dtype=object) > max_sites:
        raise ValueError(f'grid bounding box exceeds max_sites={max_sites}')
    spacing = _vector(spacing_nm, 'spacing_nm', positive=True)
    origin = _vector(origin_nm, 'origin_nm')
    if mask not in ('box', 'ellipsoid'):
        raise ValueError('mask must be box or ellipsoid')
    if active_indices is None:
        indices = np.indices(shape).reshape(3, -1).T
        if mask == 'ellipsoid':
            scaled = (indices - (np.asarray(shape) - 1) / 2) / (np.asarray(shape) / 2)
            indices = indices[np.sum(scaled**2, axis=1) <= 1]
    else:
        if mask != 'box':
            raise ValueError('active_indices and an ellipsoid mask are mutually exclusive')
        raw_indices = np.asarray(active_indices)
        if raw_indices.ndim != 2 or raw_indices.shape[1] != 3 or raw_indices.dtype.kind not in 'iu':
            raise ValueError('active_indices must be integer (N,3) sites')
        indices = np.array(raw_indices, dtype=np.int64)
        if np.any(indices < 0) or np.any(indices >= np.asarray(shape)):
            raise ValueError('active_indices must be inside shape')
    return ParticleGeometry(origin + indices * spacing, radius_nm, indices, spacing)


def from_config(config):
    allowed = {'kind', 'shape', 'spacing_nm', 'radius_nm', 'mask', 'active_indices', 'origin_nm', 'positions_nm', 'material', 'site_materials'}
    if unknown := set(config) - allowed:
        raise ValueError(f'unknown geometry fields: {sorted(unknown)}')
    kind = config.get('kind', 'grid')
    if kind == 'grid':
        if 'positions_nm' in config:
            raise ValueError('positions_nm belongs to explicit geometry')
        return grid_geometry(config['shape'], config['spacing_nm'], config['radius_nm'],
            mask=config.get('mask', 'box'), active_indices=config.get('active_indices'), origin_nm=config.get('origin_nm', (0, 0, 0)))
    if kind == 'explicit':
        forbidden = {'shape', 'spacing_nm', 'mask', 'active_indices', 'origin_nm'} & set(config)
        if forbidden:
            raise ValueError(f'explicit geometry does not accept {sorted(forbidden)}')
        return ParticleGeometry(config['positions_nm'], config['radius_nm'])
    raise ValueError('geometry kind must be grid or explicit')
