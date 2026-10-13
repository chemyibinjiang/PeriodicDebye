"""Bounded radial-moment approximation to finite, complete-cell powder sums.

Geometry is the same nonperiodic cell-pair model as ``LatticePairCache``.
Numba is optional and imported only when this backend is constructed. Distances
are streamed into private histograms; no full distance array is materialized.
The reported bound covers Taylor truncation and optional displacement merging,
not floating-point arithmetic. This is not an infinite-crystal peak model.
"""
from contextlib import contextmanager
from functools import lru_cache
import math
from time import perf_counter

import numpy as np

from .core import _positive_integer, form_factors, validate_q
from .lattice import LatticePairCache


_MAX_PRIVATE_HISTOGRAM_BYTES = 256 * 1024**2


@lru_cache(maxsize=1)
def _compiled_kernels():
    try:
        import numba
    except ImportError as error:
        raise ImportError('LargePowderCache requires the optional numba dependency; '
                          'install mof-debye-lab[fast].') from error

    @numba.njit(parallel=True, cache=True)
    def histogram(vectors, translations_count, offsets, offsets_count,
                  width, order, bins, partitions):
        moments = np.zeros((partitions, order + 1, bins), dtype=np.float64)
        remainder = np.zeros(partitions, dtype=np.float64)
        zero = np.zeros(partitions, dtype=np.int64)
        total_count = np.zeros(partitions, dtype=np.int64)
        n_translations = len(vectors)
        entries = len(offsets) * n_translations
        for part in numba.prange(partitions):
            start = entries * part // partitions
            stop = entries * (part + 1) // partitions
            remainder_sum = 0.0
            zero_count = np.int64(0)
            pair_count = np.int64(0)
            for entry in range(start, stop):
                offset_index = entry // n_translations
                translation_index = entry % n_translations
                dx = vectors[translation_index, 0] + offsets[offset_index, 0]
                dy = vectors[translation_index, 1] + offsets[offset_index, 1]
                dz = vectors[translation_index, 2] + offsets[offset_index, 2]
                radius = math.sqrt(dx * dx + dy * dy + dz * dz)
                weight = translations_count[translation_index] * offsets_count[offset_index]
                pair_count += weight
                if radius == 0.0:
                    zero_count += weight
                    continue
                index = int(math.floor(radius / width + 0.5))
                delta = (radius - index * width) / width
                term = weight / radius
                for degree in range(order + 1):
                    moments[part, degree, index] += term
                    term *= delta
                remainder_sum += abs(term)
            remainder[part] = remainder_sum
            zero[part] = zero_count
            total_count[part] = pair_count
        return moments, remainder, zero, total_count

    @numba.njit(parallel=True, cache=True)
    def exact(vectors, translations_count, offsets, offsets_count, q, partitions):
        sums = np.zeros((partitions, len(q)), dtype=np.float64)
        n_translations = len(vectors)
        entries = len(offsets) * n_translations
        for part in numba.prange(partitions):
            corrections = np.zeros(len(q), dtype=np.float64)
            start = entries * part // partitions
            stop = entries * (part + 1) // partitions
            for entry in range(start, stop):
                offset_index = entry // n_translations
                translation_index = entry % n_translations
                dx = vectors[translation_index, 0] + offsets[offset_index, 0]
                dy = vectors[translation_index, 1] + offsets[offset_index, 1]
                dz = vectors[translation_index, 2] + offsets[offset_index, 2]
                radius = math.sqrt(dx * dx + dy * dy + dz * dz)
                weight = translations_count[translation_index] * offsets_count[offset_index]
                for iq in range(len(q)):
                    argument = q[iq] * radius
                    value = float(weight) if argument == 0.0 else weight * math.sin(argument) / argument
                    adjusted = value - corrections[iq]
                    updated = sums[part, iq] + adjusted
                    corrections[iq] = (updated - sums[part, iq]) - adjusted
                    sums[part, iq] = updated
        return sums

    return numba, histogram, exact


@contextmanager
def _thread_count(numba, threads):
    previous = numba.get_num_threads()
    numba.set_num_threads(threads)
    try:
        yield
    finally:
        numba.set_num_threads(previous)


def _merge_offsets(offsets, weights, decimals):
    """Merge only grouping keys; retain an actual input vector per group.

    Return the weighted displacement error as well as its maximum. The weighted
    error propagates through the global sinc derivative bound q/2, without any
    assumption about the intensity being large or about the crystal symmetry.
    """
    if decimals is None or len(offsets) == 0:
        return offsets, weights, 0.0, 0.0
    _, first, inverse = np.unique(np.round(offsets, decimals=decimals), axis=0,
                                  return_index=True, return_inverse=True)
    representatives = offsets[first].copy()
    counts = np.zeros(len(first), dtype=np.int64)
    np.add.at(counts, inverse, weights)
    differences = np.linalg.norm(offsets - representatives[inverse], axis=1)
    weighted_error = float(np.dot(weights.astype(float), differences))
    return representatives, counts, weighted_error, float(differences.max(initial=0.0))


class LargePowderCache:
    """Prepare finite-powder geometry once and evaluate a resolved uniform q grid.

    ``iq()`` returns ``(intensity, absolute_approximation_bound)`` in electron^2.
    The bound includes radial Taylor truncation and displacement-key merging,
    but excludes roundoff. ``merge_decimals=None`` disables displacement merging.
    All cells must contain the same supplied basis. Guests, disorder and atom
    cutting require their own ensemble or explicit-coordinate models.

    Histograms use independent work partitions, reduced in a fixed order. There
    are no parallel writes to a shared histogram and no fastmath arithmetic.
    Private histogram payload is capped at 256 MiB by reducing the number of
    partitions; retained histograms, FFT workspace and Python are additional.
    """

    def __init__(self, basis, cell, mask, q, bin_width=0.04, order=8,
                 threads=1, merge_decimals=12):
        from .radial import RadialMomentHistogram

        started = perf_counter()
        self.q = np.array(validate_q(q), copy=True)
        self.q.setflags(write=False)
        self.threads = _positive_integer(threads, 'threads')
        if merge_decimals is not None:
            if (isinstance(merge_decimals, (bool, np.bool_))
                    or not isinstance(merge_decimals, (int, np.integer))
                    or not 0 <= merge_decimals <= 15):
                raise ValueError('merge_decimals must be None or an integer from 0 through 15.')
            merge_decimals = int(merge_decimals)
        self.merge_decimals = merge_decimals
        self._geometry = LatticePairCache(basis, cell, mask, cache_max_bytes=0)
        geometry = self._geometry
        self.n_cells = geometry.n_cells
        self.n_basis_atoms = len(basis)
        self.n_atoms = self.n_cells * self.n_basis_atoms
        if self.n_atoms**2 > np.iinfo(np.int64).max:
            raise ValueError('represented atom-pair count exceeds the int64 counting range.')
        largest_vector = float(np.linalg.norm(geometry.vectors, axis=1).max(initial=0.0))
        largest_offset = max((float(np.linalg.norm(values, axis=1).max(initial=0.0))
                              for values in geometry._offsets.values()), default=0.0)
        # Triangle bound plus an outward margin for all norm/sum operations.
        triangle = largest_vector + largest_offset
        radius_limit = np.nextafter(triangle + 32 * np.finfo(float).eps * max(1., triangle), np.inf)
        template = RadialMomentHistogram(self.q, radius_limit,
                                         bin_width=bin_width, order=order)
        bytes_per_partition = template.moments.nbytes
        self._grid_metadata = template.metadata
        del template
        if bytes_per_partition > _MAX_PRIVATE_HISTOGRAM_BYTES:
            raise ValueError('one radial histogram exceeds the 256 MiB workspace cap; '
                             'increase bin_width or reduce order.')
        self.partitions = min(self.threads,
                              max(1, _MAX_PRIVATE_HISTOGRAM_BYTES // bytes_per_partition))
        self.private_histogram_bytes = self.partitions * bytes_per_partition
        numba, build, _ = _compiled_kernels()
        if self.threads > numba.config.NUMBA_NUM_THREADS:
            raise ValueError(f'threads exceeds the Numba limit of {numba.config.NUMBA_NUM_THREADS}.')
        self._histograms = {}
        self._merged_offsets = {}
        self._merge_weighted_errors = {}
        self.max_merged_displacement_A = 0.0
        self.grouped_basis_displacements = 0
        self.pair_records = []
        with _thread_count(numba, self.threads):
            for pair in geometry._pairs:
                a, b = pair
                offsets, counts, weighted_error, max_error = _merge_offsets(
                    geometry._offsets[pair], geometry._offset_counts[pair], merge_decimals)
                offsets.setflags(write=False)
                counts.setflags(write=False)
                self._merged_offsets[pair] = offsets, counts
                self._merge_weighted_errors[pair] = weighted_error * self.n_cells**2
                self.max_merged_displacement_A = max(self.max_merged_displacement_A, max_error)
                self.grouped_basis_displacements += len(offsets)
                histogram = RadialMomentHistogram(self.q, radius_limit,
                                                   bin_width=bin_width, order=order)
                moments, remainder, zero, total = build(
                    geometry.vectors, geometry.multiplicities, offsets, counts,
                    histogram.dr, histogram.order, histogram.n_bins, self.partitions)
                total_count = int(total.sum(dtype=np.int64))
                expected_count = (len(geometry._positions[a]) * len(geometry._positions[b])
                                  * self.n_cells**2)
                if total_count != expected_count:
                    raise ArithmeticError('finite-cell species pair-count identity failed.')
                combined = moments.sum(axis=0)
                histogram.load_moments(
                    combined, total_count, int(zero.sum(dtype=np.int64)),
                    float(combined[0].sum()), float(remainder.sum()))
                histogram.moments.setflags(write=False)
                self._histograms[pair] = histogram
                self.pair_records.append({
                    'elements': [a, b], 'basis_offsets': len(offsets),
                    'pair_count': total_count, 'zero_distance_count': int(zero.sum(dtype=np.int64)),
                    'distance_entries': len(offsets) * len(geometry.vectors),
                    'max_merged_displacement_A': max_error,
                    'weighted_displacement_error_A': self._merge_weighted_errors[pair],
                })
                del moments, combined, remainder, zero, total
        self.preprocessing_seconds = perf_counter() - started
        self.last_evaluation_seconds = None
        self.evaluation_count = 0
        self.last_exact_selected_seconds = None
        self.last_max_approximation_bound = None
        self.last_max_taylor_bound = None
        self.last_max_grouping_bound = None

    def iq(self):
        """Return finite powder intensity and absolute approximation bound."""
        started = perf_counter()
        factors = form_factors(self._geometry.elements, self.q)
        values = np.zeros(len(self.q))
        taylor_bound = np.zeros(len(self.q))
        grouping_bound = np.zeros(len(self.q))
        for (a, b), histogram in self._histograms.items():
            multiplier = (1 if a == b else 2) * factors[a] * factors[b]
            pair_values, pair_bound = histogram.evaluate()
            values += multiplier * pair_values
            taylor_bound += np.abs(multiplier) * pair_bound
            grouping_bound += (np.abs(multiplier) * self.q * 0.5
                               * self._merge_weighted_errors[a, b])
        # Use the direct forward-amplitude identity, independent of pair order.
        forward = self.n_cells * sum(factors[element][0] for element in self._geometry.elements)
        values[0] = forward * forward
        bound = taylor_bound + grouping_bound
        self.last_evaluation_seconds = perf_counter() - started
        self.evaluation_count += 1
        self.last_max_approximation_bound = float(bound.max(initial=0.0))
        self.last_max_taylor_bound = float(taylor_bound.max(initial=0.0))
        self.last_max_grouping_bound = float(grouping_bound.max(initial=0.0))
        return values, bound

    def exact_selected(self, q_values, merged=False):
        """Compiled compensated Debye sums at selected q values.

        This performs no radial binning. The default uses all original exact
        float64 displacement groups, including those merged for the fast path.
        ``merged=True`` is useful for isolating histogram error and still uses
        the original grouping bound. This is a validation path, not a guarantee
        that large dense-grid exact calculations are inexpensive.
        """
        if not isinstance(merged, (bool, np.bool_)):
            raise ValueError('merged must be a boolean.')
        started = perf_counter()
        q = validate_q(q_values)
        numba, _, exact = _compiled_kernels()
        factors = form_factors(self._geometry.elements, q)
        values = np.zeros(len(q))
        with _thread_count(numba, self.threads):
            for a, b in self._geometry._pairs:
                if merged:
                    offsets, counts = self._merged_offsets[a, b]
                else:
                    offsets = self._geometry._offsets[a, b]
                    counts = self._geometry._offset_counts[a, b]
                pair_values = exact(self._geometry.vectors, self._geometry.multiplicities,
                                    offsets, counts, q, self.partitions).sum(axis=0)
                values += (1 if a == b else 2) * factors[a] * factors[b] * pair_values
        if np.any(q == 0):
            ff_zero = form_factors(self._geometry.elements, [0.0])
            forward = self.n_cells * sum(ff_zero[element][0] for element in self._geometry.elements)
            values[q == 0] = forward * forward
        self.last_exact_selected_seconds = perf_counter() - started
        return values

    @property
    def metadata(self):
        geometry = self._geometry
        return {
            **self._grid_metadata,
            'method': 'finite_full_cell_radial_moments',
            'numerical_backend': 'numba_private_histograms_scipy_real_fft',
            'n_cells': self.n_cells, 'n_basis_atoms': self.n_basis_atoms,
            'represented_atoms': self.n_atoms, 'n_translations': len(geometry.vectors),
            'original_basis_displacements': geometry.grouped_basis_displacements,
            'grouped_basis_displacements': self.grouped_basis_displacements,
            'displacement_symmetry': geometry.displacement_symmetry,
            'original_distance_entries': geometry.distance_entries,
            'distance_entries': self.grouped_basis_displacements * len(geometry.vectors),
            'merge_decimals': self.merge_decimals,
            'displacement_merge': 'rounded_keys_actual_input_representatives',
            'max_merged_displacement_A': self.max_merged_displacement_A,
            'threads': self.threads, 'histogram_partitions': self.partitions,
            'histogram_private_workspace_bytes': self.private_histogram_bytes,
            'histogram_bytes': sum(histogram.moments.nbytes for histogram in self._histograms.values()),
            'memory_scope': 'private histogram payload capped at 256 MiB; retained histograms, '
                            'shape FFT, basis offsets, transform workspace and Python are additional',
            'species_pairs': self.pair_records,
            'preprocessing_seconds': self.preprocessing_seconds,
            'last_evaluation_seconds': self.last_evaluation_seconds,
            'evaluation_count': self.evaluation_count,
            'last_exact_selected_seconds': self.last_exact_selected_seconds,
            'max_absolute_approximation_bound_electron2': self.last_max_approximation_bound,
            'max_absolute_taylor_bound_electron2': self.last_max_taylor_bound,
            'max_absolute_grouping_bound_electron2': self.last_max_grouping_bound,
            'approximation_bound': 'pointwise absolute Taylor plus displacement-merge bound; '
                                   'excludes floating-point accumulation and FFT roundoff',
            'fastmath': False,
        }
