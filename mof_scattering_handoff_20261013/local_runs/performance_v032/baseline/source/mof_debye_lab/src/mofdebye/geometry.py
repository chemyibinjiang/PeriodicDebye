"""Explicit finite boundaries and exact nearest-host distances under periodicity.

All distances are in angstrom and direct-lattice vectors are rows. Atom cuts
retain supplied atoms without adding, capping, reconstructing or relaxing them.
Periodic clearance uses lattice images; finite host exclusion must instead use
the actually retained finite atom cloud.
"""
from __future__ import annotations

from itertools import islice, product

import numpy as np
from scipy.spatial import cKDTree

from .core import AtomCloud, _positive_integer


def _cell(cell):
    # Keep this geometry layer independent of scattering/moment modules, which
    # themselves import the pore sampler using these boundary helpers.
    values = np.asarray(cell, dtype=float)
    if values.shape != (3, 3) or not np.isfinite(values).all() or abs(np.linalg.det(values)) < 1e-12:
        raise ValueError('cell must contain three finite, independent row vectors in angstrom.')
    return values


def _points(points):
    values = np.asarray(points, dtype=float)
    if values.ndim < 1 or values.shape[-1] != 3 or not np.isfinite(values).all():
        raise ValueError('Points must be finite Cartesian coordinates with final dimension 3.')
    return values


def _vector(specification, name):
    values = np.asarray(specification.get(name), dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        raise ValueError(f'{name} must contain three finite Cartesian coordinates.')
    return values


def _mask(mask):
    values = np.asarray(mask)
    if values.ndim != 3 or not values.size or not np.isin(values, (0, 1)).all() or not values.any():
        raise ValueError('Cell mask must be a nonempty 3D binary array with selected cells.')
    return values.astype(bool, copy=False)


def contains_cell_mask(points, cell, mask, tolerance_A=1e-10):
    """Test membership in the closed union of selected cell parallelepipeds.

    Cell ``(i,j,k)`` spans fractional coordinates ``[i,i+1] x [j,j+1]
    x [k,k+1]``. Checking both sides of a face preserves boundary points even
    when the cell on one side is absent. This boundary is a geometric envelope,
    not an atomistically reconstructed surface or an automatic pore model.
    """
    points = _points(points)
    cell = _cell(cell)
    selected = _mask(mask)
    if not np.isfinite(tolerance_A) or tolerance_A < 0:
        raise ValueError('Boundary tolerance must be finite and nonnegative.')
    inverse = np.linalg.inv(cell)
    fractional = points.reshape(-1, 3) @ inverse
    tolerance = tolerance_A * np.linalg.norm(inverse, axis=0)
    # A tolerance comparable to a whole cell has no useful containment meaning.
    if np.any(tolerance >= 0.25):
        raise ValueError('Boundary tolerance is too large for the cell plane spacing; reduce tolerance_A.')
    lower = np.floor(fractional - tolerance).astype(np.int64)
    upper = np.floor(fractional + tolerance).astype(np.int64)
    # With zero tolerance, exact integral points still belong to either face.
    lower -= ((tolerance == 0) & (fractional == lower)).astype(np.int64)
    inside = np.zeros(len(fractional), dtype=bool)
    for side in product((0, 1), repeat=3):
        indices = np.where(np.asarray(side), upper, lower)
        valid = np.all((indices >= 0) & (indices < np.asarray(selected.shape)), axis=1)
        rows = indices[valid]
        inside[valid] |= selected[rows[:, 0], rows[:, 1], rows[:, 2]]
    return inside.reshape(points.shape[:-1])


def contains_points(points, specification):
    """Closed sphere, box, complete-cell mask or intersection containment.

    Sphere fields are ``kind='sphere', centre_A, radius_A``; box fields are
    ``kind='box', min_A, max_A``. ``kind='cell_mask'`` takes
    ``cell_vectors_A, mask``. ``surface_policy`` is permitted for atom-cut
    specifications but is enforced by :func:`apply_atom_cut`. An intersection
    takes ``kind='intersection', boundaries=[specification, ...]``.
    """
    values = _points(points)
    if not isinstance(specification, dict):
        raise ValueError('Boundary specification must be an object.')
    kind = specification.get('kind')
    fields = {
        'sphere': {'kind', 'centre_A', 'radius_A', 'surface_policy'},
        'box': {'kind', 'min_A', 'max_A', 'surface_policy'},
        'cell_mask': {'kind', 'cell_vectors_A', 'mask'},
        'intersection': {'kind', 'boundaries'},
    }
    if kind not in fields:
        raise ValueError('Boundary kind must be sphere, box, cell_mask or intersection.')
    extra = set(specification) - fields[kind]
    if extra:
        raise ValueError(f'Unsupported {kind} boundary fields: {sorted(extra)}')
    if kind == 'intersection':
        boundaries = specification.get('boundaries')
        if not isinstance(boundaries, (list, tuple)) or not boundaries:
            raise ValueError('An intersection requires a nonempty list of boundaries.')
        retained = np.ones(values.shape[:-1], dtype=bool)
        for boundary in boundaries:
            retained &= contains_points(values, boundary)
        return retained
    if kind == 'sphere':
        centre = _vector(specification, 'centre_A')
        radius = specification.get('radius_A')
        if isinstance(radius, (bool, np.bool_)) or not isinstance(radius, (int, float, np.number)) or not np.isfinite(radius) or radius <= 0:
            raise ValueError('Sphere radius_A must be positive and finite.')
        return np.linalg.norm(values - centre, axis=-1) <= radius
    if kind == 'box':
        lower, upper = _vector(specification, 'min_A'), _vector(specification, 'max_A')
        if np.any(upper <= lower):
            raise ValueError('Box max_A must exceed min_A in every coordinate.')
        return np.all((values >= lower) & (values <= upper), axis=-1)
    return contains_cell_mask(values, specification.get('cell_vectors_A'), specification.get('mask'))


def apply_atom_cut(cloud: AtomCloud, specification):
    """Cut an explicitly materialized candidate cloud and record termination.

    The caller supplies enough candidate cells to cover the desired cut. No
    atoms outside those supplied cells are invented. The only supported
    termination policy is the explicitly requested ``unpassivated_atom_cut``.
    """
    if not isinstance(specification, dict) or specification.get('surface_policy') != 'unpassivated_atom_cut':
        raise ValueError('Atom cuts require surface_policy="unpassivated_atom_cut"; surface reconstruction and passivation are unsupported.')
    if specification.get('kind') not in ('sphere', 'box'):
        raise ValueError('Atom cuts require a Cartesian sphere or box boundary.')
    retained = contains_points(cloud.xyz, specification)
    if not retained.any():
        raise ValueError('Atom cut removed all host atoms; check the cut position, size and candidate cell_mask.')
    result = AtomCloud(tuple(element for element, keep in zip(cloud.elements, retained) if keep),
                       cloud.xyz[retained], f'{cloud.name}; unpassivated atom cut')
    metadata = {
        'surface_policy': 'unpassivated_atom_cut',
        'candidate_atom_count': len(cloud),
        'retained_atom_count': len(result),
        'removed_atom_count': len(cloud) - len(result),
        'termination': 'Supplied atoms retained by Cartesian containment; no capping, reconstruction, charge balancing or relaxation.',
        'candidate_coverage': 'Intersection with explicitly supplied candidate cells; no automatic padding.',
    }
    return result, metadata


class PeriodicHostTree:
    """Nearest periodic host atom with an exact, bounded-workspace image search.

    ``query(points, k=1)`` returns distances and indices in the supplied basis,
    matching the cKDTree interface used by the pore sampler. Both host and query
    coordinates are reduced modulo the lattice, so unbounded displacements do
    not require a guessed finite neighbor halo. A central-cell nearest distance
    gives an upper bound R. A possibly closer displacement d has fractional
    components bounded by R times the inverse-cell column norms. Enumerating
    that finite rectangular image set is rigorous for triclinic cells, including
    cases where the nearest image lies outside a 3-by-3-by-3 neighborhood.

    Image queries stream in ``query_chunk`` blocks. Excessive candidate counts
    fail explicitly instead of truncating distances or allocating the whole
    replicated structure. No cutoff approximation is used.
    """

    is_periodic = True

    def __init__(self, basis: AtomCloud, cell, max_images=250_000, query_chunk=4096):
        if len(basis) == 0:
            raise ValueError('Periodic host clearance requires a nonempty host basis.')
        self.cell = np.array(_cell(cell), copy=True)
        self.cell.setflags(write=False)
        self.inverse = np.linalg.inv(self.cell)
        self.inverse.setflags(write=False)
        self._dual_norms = np.linalg.norm(self.inverse, axis=0)
        self._tree = cKDTree((basis.xyz @ self.inverse % 1.) @ self.cell)
        self.max_images = _positive_integer(max_images, 'max_images')
        self.query_chunk = _positive_integer(query_chunk, 'query_chunk')
        self.metadata = {
            'method': 'exact_periodic_nearest_host',
            'basis_atom_count': len(basis),
            'image_search': 'Dynamic reciprocal-norm bound from an initial nearest distance; no fixed image halo.',
            'max_images_per_query': self.max_images,
            'query_chunk': self.query_chunk,
        }

    def query(self, points, k=1):
        if isinstance(k, (bool, np.bool_)) or k != 1:
            raise ValueError('PeriodicHostTree currently supports nearest-atom queries with k=1 only.')
        points = _points(points)
        shape = points.shape[:-1]
        fractional = (points.reshape(-1, 3) @ self.inverse) % 1.
        wrapped = fractional @ self.cell
        distances, indices = self._tree.query(wrapped, k=1)
        for row, (fraction, point) in enumerate(zip(fractional, wrapped)):
            bounds = distances[row] * self._dual_norms
            # Host fractional coordinates lie in [0,1); include harmless edge
            # images with a small roundoff margin so the search is conservative.
            lower = np.ceil(fraction - 1. - bounds - 1e-12).astype(np.int64)
            upper = np.floor(fraction + bounds + 1e-12).astype(np.int64)
            count = int(np.prod((upper - lower + 1).astype(object)))
            if count > self.max_images:
                raise ValueError(f'Periodic nearest-host search requires {count} images for one point; '
                                 'reduce cell skew or explicitly raise max_images. No image truncation was applied.')
            candidates = product(*(range(int(a), int(b) + 1) for a, b in zip(lower, upper)))
            while chunk := tuple(islice(candidates, self.query_chunk)):
                shifted = point - np.asarray(chunk, dtype=float) @ self.cell
                current_distances, current_indices = self._tree.query(shifted, k=1)
                best = int(np.argmin(current_distances))
                if current_distances[best] < distances[row]:
                    distances[row] = current_distances[best]
                    indices[row] = current_indices[best]
        distances, indices = distances.reshape(shape), indices.reshape(shape)
        if not shape:
            return float(distances), int(indices)
        return distances, indices
