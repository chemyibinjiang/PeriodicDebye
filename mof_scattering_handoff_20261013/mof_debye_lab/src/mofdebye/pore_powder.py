"""Scalable finite powder scattering from independent, isotropic pore guests.

Pore centres are labelled scattering sites, not atoms or fictitious elements.
The finite-cell geometry is reused with q-dependent occupied-pore amplitudes.
All four pair contributions are retained. Independent local sample replicas
give unbiased guest-guest products when one estimated mean is reused over many
cells; batch variability measures numerical integration uncertainty.
"""
from dataclasses import dataclass
from itertools import product
from time import perf_counter

import numpy as np

from .core import AtomCloud, form_factors, validate_q
from .distributions import PorePopulation
from .large_powder import LargePowderCache
from .radial_guests import RadialGuestStatistics, _pore_geometry_signature


_PARTS = ('host', 'guest_internal', 'host_guest', 'guest_guest')


@dataclass(frozen=True)
class _ScatteringSites:
    """Private geometry-only protocol; labels are explicitly not elements.

    LatticePairCache uses ``elements`` as grouping labels without atomic-factor
    lookup. Only that geometry protocol is reused here; LargePowderCache.iq and
    its element-based exact evaluator must never be called on these sites.
    """
    elements: tuple[str, ...]
    xyz: np.ndarray

    def __post_init__(self):
        labels = tuple(self.elements)
        xyz = np.array(self.xyz, dtype=float, copy=True)
        if xyz.shape != (len(labels), 3) or not np.isfinite(xyz).all():
            raise ValueError('Scattering-site labels and finite coordinates must match.')
        if any(not isinstance(label, str) or not label.startswith(('host:', 'pore:'))
               for label in labels):
            raise ValueError('Private scattering sites require host/pore channel labels.')
        xyz.setflags(write=False)
        object.__setattr__(self, 'elements', labels)
        object.__setattr__(self, 'xyz', xyz)

    def __len__(self):
        return len(self.elements)


def _validate_periodic_pore_separation(centres, radii, cell):
    """Reject sphere overlap with any lattice image, including the same site.

    Reciprocal-column norms bound every integer image that could overlap, also
    for skew cells. The search is conservative for a finite mask: its repeated
    pore model must be geometrically compatible before a crystallite is cut.
    """
    matrix = np.asarray(cell, dtype=float)
    if (matrix.shape != (3, 3) or not np.isfinite(matrix).all()
            or abs(np.linalg.det(matrix)) < 1e-12):
        raise ValueError('cell must be a nonsingular finite 3-by-3 matrix.')
    inverse = np.linalg.inv(matrix)
    for a in range(len(centres)):
        for b in range(a):
            fraction = (centres[a] - centres[b]) @ inverse
            residual = (fraction - np.rint(fraction)) @ matrix
            if np.linalg.norm(residual) <= 1e-10:
                raise ValueError('Duplicate periodically equivalent pore centres are not independent sites.')
    if radii is None:
        return
    reciprocal_norms = np.linalg.norm(inverse, axis=0)
    for a in range(len(centres)):
        for b in range(a + 1):
            displacement = centres[a] - centres[b]
            fractional = displacement @ inverse
            radius = float(radii[a] + radii[b])
            extent = radius * reciprocal_norms
            lower = np.ceil(-fractional - extent - 1e-12).astype(np.int64)
            upper = np.floor(-fractional + extent + 1e-12).astype(np.int64)
            candidates = int(np.prod((upper - lower + 1).astype(object)))
            if candidates > 250_000:
                raise ValueError('Periodic pore-overlap validation exceeds 250000 images; '
                                 'use a better-conditioned cell or smaller pore radii.')
            for image in product(*(range(int(lo), int(hi) + 1) for lo, hi in zip(lower, upper))):
                if a == b and image == (0, 0, 0):
                    continue
                distance = np.linalg.norm(displacement + np.asarray(image) @ matrix)
                if distance < radius - 1e-10:
                    raise ValueError('Hard-wall pores overlap under a lattice translation; '
                                     'independent pore states need an exclusion/correlation model.')


def _product_bound(a, b, a_bound, b_bound):
    return np.abs(a) * b_bound + np.abs(b) * a_bound + a_bound * b_bound


class PorePowderCache:
    """Reusable finite host/pore geometry with independent-cell guest disorder.

    ``population`` specifies one cell's pore sites and occupancies. Every pore
    in every retained cell has an independent physical state. The supplied
    radial statistics describe conditional occupied-pore amplitudes; this class
    applies whole-particle occupancy exactly once to internal scattering.

    Only isotropic local guest distributions represented by
    ``radial_guest_statistics`` are supported. Geometry and occupancy are copied
    at construction. The returned approximation bound excludes Monte Carlo
    integration uncertainty and floating-point roundoff; SEM is separate.
    """

    def __init__(self, host_basis, cell, mask, population, q, **radial_options):
        started = perf_counter()
        if not isinstance(host_basis, AtomCloud):
            raise TypeError('host_basis must be an AtomCloud with real chemical elements.')
        if type(population) is not PorePopulation:
            raise TypeError('population must be an independent-pore PorePopulation.')
        self.q = np.array(validate_q(q), copy=True)
        self.centres = np.array(population.centres + population.mean_displacement, copy=True)
        self.occupancy = np.array(population.occupancy, copy=True)
        self._pore_geometry = _pore_geometry_signature(population)
        if (self.occupancy.shape != (len(self.centres),)
                or not np.isfinite(self.occupancy).all()
                or np.any((self.occupancy < 0) | (self.occupancy > 1))):
            raise ValueError('Per-pore occupancy must remain finite and between zero and one.')
        for array in (self.q, self.centres, self.occupancy):
            array.setflags(write=False)
        # Physical pore walls remain centred on the declared pore, even when
        # an analytically supported mean displacement shifts its scattering site.
        _validate_periodic_pore_separation(population.centres, population.pore_radius, cell)
        self.host_elements = tuple(host_basis.elements)
        self.host_atoms_per_cell = len(host_basis)
        self.pores_per_cell = len(self.centres)
        labels = tuple(f'host:{element}' for element in host_basis.elements)
        self._pore_channels = tuple(f'pore:{index}' for index in range(self.pores_per_cell))
        self._pore_index = {channel: index for index, channel in enumerate(self._pore_channels)}
        sites = _ScatteringSites(labels + self._pore_channels,
                                 np.concatenate((host_basis.xyz, self.centres)))
        self._cache = LargePowderCache(sites, cell, mask, self.q, **radial_options)
        self.n_cells = self._cache.n_cells
        self.expected_guest_atoms = (self.n_cells * float(self.occupancy.sum())
                                     * float(population.library.mean_atom_count))
        self.expected_guest_particles = self.n_cells * float(self.occupancy.sum())
        self._guest_atom_count_source = 'construction_population_before_first_evaluation'
        self._geometry_values = {}
        self._geometry_bounds = {}
        transform_started = perf_counter()
        for pair, histogram in self._cache._histograms.items():
            values, bound = histogram.evaluate()
            bound = bound + self.q * 0.5 * self._cache._merge_weighted_errors[pair]
            values.setflags(write=False)
            bound.setflags(write=False)
            self._geometry_values[pair] = values
            self._geometry_bounds[pair] = bound
        self.geometry_transform_seconds = perf_counter() - transform_started
        self.preprocessing_seconds = perf_counter() - started
        self.evaluation_count = 0
        self.last_evaluation_seconds = None
        self.last_statistics_metadata = None
        self.last_max_approximation_bound = None

    def _validate_statistics(self, statistics):
        if not isinstance(statistics, RadialGuestStatistics):
            raise TypeError('guest_stats must be RadialGuestStatistics.')
        if not np.array_equal(statistics.q, self.q):
            raise ValueError('Guest statistics q grid must match the geometry cache exactly.')
        if not np.array_equal(statistics.centres, self.centres):
            raise ValueError('Guest statistics pore centres must match the geometry cache exactly.')
        if not np.array_equal(statistics.occupancy, self.occupancy):
            raise ValueError('Guest statistics occupancies must match the cache population.')
        if statistics.metadata.get('pore_geometry') != self._pore_geometry:
            raise ValueError('Guest statistics must preserve declared pore centres, wall radii '
                             'and clearance; construct a new cache for changed pore geometry.')
        if any(np.iscomplexobj(values) for values in
               (statistics.means, statistics.mean_bounds, statistics.internal, statistics.internal_bounds)):
            raise ValueError('Isotropic radial guest statistics and bounds must be real-valued.')
        means = np.asarray(statistics.means, dtype=float)
        mean_bounds = np.asarray(statistics.mean_bounds, dtype=float)
        internal = np.asarray(statistics.internal, dtype=float)
        internal_bounds = np.asarray(statistics.internal_bounds, dtype=float)
        tail_shape = (2, self.pores_per_cell, len(self.q))
        if means.ndim != 4 or means.shape[1:] != tail_shape or means.shape[0] == 0:
            raise ValueError('Conditional guest means must have shape (batches,2,pores,q).')
        if mean_bounds.shape != means.shape:
            raise ValueError('Guest mean bounds must match guest means.')
        if internal.shape != tail_shape[1:] or internal_bounds.shape != internal.shape:
            raise ValueError('Guest internal intensities and bounds must have shape (pores,q).')
        for array in (means, mean_bounds, internal, internal_bounds):
            if not np.isfinite(array).all():
                raise ValueError('Guest statistics and bounds must be finite.')
        internal_scale = np.maximum(1.0, np.max(np.abs(internal), axis=1, keepdims=True))
        if np.any(internal < -1e-10 * internal_scale):
            raise ValueError('Conditional internal guest intensities must be nonnegative '
                             'within floating-point tolerance.')
        if np.any(mean_bounds < 0) or np.any(internal_bounds < 0):
            raise ValueError('Guest approximation bounds must be nonnegative.')
        method = statistics.metadata.get('method')
        if method == 'analytic':
            if means.shape[0] != 1 or not np.array_equal(means[:, 0], means[:, 1]):
                raise ValueError('Analytic statistics require one batch and identical replicas.')
        elif method == 'sampled':
            if means.shape[0] < 2:
                raise ValueError('Sampled statistics require at least two independent batches for SEM.')
        else:
            raise ValueError('Guest statistics method must be analytic or sampled.')
        mean_atom_count = statistics.metadata.get('conditional_mean_atom_count')
        if mean_atom_count is not None:
            if (isinstance(mean_atom_count, (bool, np.bool_))
                    or not isinstance(mean_atom_count, (int, float, np.integer, np.floating))
                    or not np.isfinite(mean_atom_count) or mean_atom_count < 0):
                raise ValueError('Conditional mean atom count must be finite and nonnegative when supplied.')
            mean_atom_count = float(mean_atom_count)
        return means, mean_bounds, internal, internal_bounds, mean_atom_count

    def iq(self, guest_stats):
        """Return four contributions, their sum, Monte Carlo SEM and error bounds.

        In each batch A and B are independent estimates of conditional occupied
        amplitudes. Unlike pores use (A_a B_b+B_a A_b)/2; the same pore channel
        uses A_a B_a. This removes the reused sample-mean-squared bias without
        conflating numerical mean uncertainty with physical disorder variance.
        """
        started = perf_counter()
        means, mean_bounds, internal, internal_bounds, mean_atom_count = self._validate_statistics(guest_stats)
        batches = len(means)
        shape = (batches, len(self.q))
        parts = {name: np.zeros(shape) for name in _PARTS}
        bounds = {name: np.zeros(shape) for name in _PARTS}
        factors = form_factors(self.host_elements, self.q)
        mu = means * self.occupancy[None, None, :, None]
        mu_bounds = mean_bounds * self.occupancy[None, None, :, None]

        for pair, geometry in self._geometry_values.items():
            a, b = pair
            geometry_bound = self._geometry_bounds[pair]
            a_host, b_host = a.startswith('host:'), b.startswith('host:')
            if a_host and b_host:
                coefficient = (1 if a == b else 2) * factors[a[5:]] * factors[b[5:]]
                parts['host'] += coefficient * geometry
                bounds['host'] += np.abs(coefficient) * geometry_bound
                continue
            if a_host or b_host:
                host, pore = (a, b) if a_host else (b, a)
                index = self._pore_index[pore]
                # The cross-pair factor two cancels the replica-mean half.
                coefficient = factors[host[5:]] * (mu[:, 0, index] + mu[:, 1, index])
                coefficient_bound = np.abs(factors[host[5:]]) * (
                    mu_bounds[:, 0, index] + mu_bounds[:, 1, index])
                name = 'host_guest'
            else:
                ia, ib = self._pore_index[a], self._pore_index[b]
                coefficient = mu[:, 0, ia] * mu[:, 1, ib]
                coefficient_bound = _product_bound(mu[:, 0, ia], mu[:, 1, ib],
                                                   mu_bounds[:, 0, ia], mu_bounds[:, 1, ib])
                if ia != ib:
                    coefficient += mu[:, 1, ia] * mu[:, 0, ib]
                    coefficient_bound += _product_bound(mu[:, 1, ia], mu[:, 0, ib],
                                                        mu_bounds[:, 1, ia], mu_bounds[:, 0, ib])
                name = 'guest_guest'
            parts[name] += coefficient * geometry
            bounds[name] += (np.abs(coefficient) * geometry_bound
                             + coefficient_bound * (np.abs(geometry) + geometry_bound))

        # Whole-particle internal scattering has p, not p squared. Remove only
        # the same physical pore's coherent self; intercell same-type pairs stay.
        parts['guest_internal'][:] = self.n_cells * np.sum(
            self.occupancy[:, None] * internal, axis=0)
        bounds['guest_internal'][:] = self.n_cells * np.sum(
            self.occupancy[:, None] * internal_bounds, axis=0)
        parts['guest_guest'] -= self.n_cells * np.sum(mu[:, 0] * mu[:, 1], axis=1)
        bounds['guest_guest'] += self.n_cells * np.sum(
            _product_bound(mu[:, 0], mu[:, 1], mu_bounds[:, 0], mu_bounds[:, 1]), axis=1)
        # Exact host forward amplitude avoids accumulation-order differences.
        forward = self.n_cells * sum(factors[element][0] for element in self.host_elements)
        parts['host'][:, 0] = forward * forward
        total_batches = sum(parts.values(), start=np.zeros(shape))
        bound_batches = sum(bounds.values(), start=np.zeros(shape))
        output = {name: array.mean(axis=0) for name, array in parts.items()}
        output['total'] = total_batches.mean(axis=0)
        output['mc_sem'] = {
            name: (array.std(axis=0, ddof=1) / np.sqrt(batches)
                   if batches > 1 else np.zeros(len(self.q)))
            for name, array in {**parts, 'total': total_batches}.items()
        }
        output['total_mc_sem'] = output['mc_sem']['total']
        output['component_approximation_bounds'] = {
            name: array.mean(axis=0) for name, array in bounds.items()
        }
        output['approximation_bound_excluding_roundoff'] = bound_batches.mean(axis=0)
        self.last_evaluation_seconds = perf_counter() - started
        self.evaluation_count += 1
        self.last_statistics_metadata = dict(guest_stats.metadata)
        self.expected_guest_atoms = (None if mean_atom_count is None
                                     else self.expected_guest_particles * mean_atom_count)
        self._guest_atom_count_source = ('unavailable_in_supplied_statistics' if mean_atom_count is None
                                         else 'current_guest_statistics')
        self.last_max_approximation_bound = float(output['approximation_bound_excluding_roundoff'].max())
        output['metadata'] = self.metadata
        return output

    @property
    def metadata(self):
        geometry = dict(self._cache.metadata)
        geometry.pop('represented_atoms', None)
        geometry.pop('n_basis_atoms', None)
        geometry['species_pairs'] = [
            {('channels' if key == 'elements' else key): value for key, value in pair.items()}
            for pair in geometry['species_pairs']
        ]
        return {
            **geometry,
            'method': 'finite_full_cell_independent_pore_radial_moments',
            'geometry_scattering_sites_per_cell': self.host_atoms_per_cell + self.pores_per_cell,
            'host_atoms_per_cell': self.host_atoms_per_cell,
            'physical_host_atoms': self.n_cells * self.host_atoms_per_cell,
            'pores_per_cell': self.pores_per_cell,
            'physical_pores': self.n_cells * self.pores_per_cell,
            'expected_guest_particles': self.expected_guest_particles,
            'expected_guest_atoms': self.expected_guest_atoms,
            'guest_atom_count_source': self._guest_atom_count_source,
            'pore_channels': 'geometry-only labels; not atoms or artificial elements',
            'physical_disorder': 'independent pore states in every retained cell',
            'sampling_products': 'two independent conditional-mean replicas per batch',
            'uncertainty': 'batch SEM is numerical integration error, not physical population uncertainty',
            'approximation_bound': 'radial Taylor and displacement merging plus propagated local-mean bounds; '
                                   'excludes Monte Carlo error and floating-point roundoff',
            'geometry_curve_bytes': sum(array.nbytes for array in self._geometry_values.values())
                                    + sum(array.nbytes for array in self._geometry_bounds.values()),
            'geometry_transform_seconds': self.geometry_transform_seconds,
            'preprocessing_seconds': self.preprocessing_seconds,
            'last_evaluation_seconds': self.last_evaluation_seconds,
            'evaluation_count': self.evaluation_count,
            'guest_statistics': self.last_statistics_metadata,
            'max_absolute_approximation_bound_electron2': self.last_max_approximation_bound,
        }
