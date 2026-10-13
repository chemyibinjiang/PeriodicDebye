"""Independent prototype: FFT finite-shape autocorrelation and lattice-grouped Debye.
This is new project code, NOT a feature attributed to the tested external kernel.
It preserves complete identical unit cells; arbitrary atom-level surface cutting
and surface relaxation require separate treatment.
"""
from time import perf_counter
import numpy as np
from scipy.signal import fftconvolve
from .core import form_factors, weighted_sinc_sum, validate_q, _positive_integer

def shape_autocorrelation(mask):
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.all(np.isin(mask, (0, 1))) or not np.any(mask):
        raise ValueError('mask must be a nonempty three-dimensional binary array.')
    # Full LINEAR correlation: fftconvolve pads internally; there is no periodic wrap.
    floating = fftconvolve(mask.astype(float), mask[::-1, ::-1, ::-1].astype(float), mode='full')
    counts = np.rint(floating).astype(np.int64)
    if np.max(np.abs(floating-counts)) > 1e-5 or np.any(counts < 0):
        raise ArithmeticError('FFT correlation rounding did not produce valid pair counts.')
    if counts.sum() != int(mask.sum())**2 or counts[tuple(np.array(mask.shape)-1)] != mask.sum():
        raise ArithmeticError('Finite-shape pair-count identities failed.')
    indices = np.argwhere(counts > 0)
    translations = indices - (np.array(mask.shape) - 1)
    return translations, counts[tuple(indices.T)]


def _aggregate_radii(distances, weights):
    """Combine only exactly equal float64 radii; preserve integer pair counts."""
    radii, inverse = np.unique(distances, return_inverse=True)
    counts = np.zeros(len(radii), dtype=np.int64)
    np.add.at(counts, inverse, weights)
    return radii, counts


class LatticePairCache:
    """Reusable full-cell translation counts and optional pair-distance cache.

    ``A`` has lattice vectors as rows and basis coordinates are Cartesian.
    Construction computes the linear FFT shape autocorrelation once and groups
    exactly equal basis displacements and radial distances. Candidate cache
    blocks are retained only while their radii AND integer weights fit
    ``cache_max_bytes``; completed species pairs are combined exactly once.
    A budget overflow discards the candidate cache and streams bounded blocks
    in ``iq``. No radius is rounded, binned or interpolated.

    The budget caps retained cache-array payload, including weights. Temporary
    concatenation/sorting arrays can use a small multiple of that payload plus
    one distance block. It excludes shape FFT arrays, the O(basis_atoms**2)
    displacement library, input/output, factors and Python overhead; it is not
    a bound on total process memory.

    This exact rearrangement includes self terms, host interference and every
    retained complete cell. It does not describe independent random occupations
    in different cells or atomically cut surfaces. Cache geometry is copied.
    """
    def __init__(self, basis, cell_vectors_A, mask, cache_max_bytes=64 * 1024**2,
                 translation_chunk=8192):
        started = perf_counter()
        if (isinstance(cache_max_bytes, (bool, np.bool_))
                or not isinstance(cache_max_bytes, (int, np.integer)) or cache_max_bytes < 0):
            raise ValueError('cache_max_bytes must be a nonnegative integer.')
        self.cache_max_bytes = int(cache_max_bytes)
        self.translation_chunk = _positive_integer(translation_chunk, 'translation_chunk')
        cell = np.array(cell_vectors_A, dtype=float, copy=True)
        if cell.shape != (3, 3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-12:
            raise ValueError('cell_vectors_A must be a nonsingular finite 3-by-3 matrix.')
        self.translations, self.multiplicities = shape_autocorrelation(mask)
        self.vectors = self.translations @ cell
        self.n_cells = int(np.asarray(mask).sum())
        self.n_basis_atoms = len(basis)
        self.elements = tuple(basis.elements)
        labels = np.asarray(basis.elements)
        self.species = sorted(set(self.elements))
        self._positions = {el: np.array(basis.xyz[labels == el], copy=True) for el in self.species}
        self._pairs = [(a, b) for k, a in enumerate(self.species) for b in self.species[k:]]
        shape = np.asarray(mask).shape
        diagonal = np.diag(cell)
        # Use no geometric tolerance: this stronger symmetry is valid only for
        # an exactly axis-aligned cubic metric and a completely filled cube.
        cube_symmetry = (len(set(shape)) == 1 and np.all(mask)
                         and np.array_equal(cell, np.diag(diagonal))
                         and np.all(np.abs(diagonal) == abs(diagonal[0])))
        self.displacement_symmetry = 'cubic_signed_permutations' if cube_symmetry else 'inversion'
        self._offsets, self._offset_counts = {}, {}
        self.raw_basis_pairs = 0
        for a, b in self._pairs:
            xa, xb = self._positions[a], self._positions[b]
            self.raw_basis_pairs += len(xa) * len(xb)
            if a == b:
                first, second = np.triu_indices(len(xa))
                offsets = xa[first] - xa[second]
                pair_counts = np.where(first == second, 1, 2)
            else:
                offsets = (xa[:, None, :] - xb[None, :, :]).reshape(-1, 3)
                pair_counts = np.ones(len(offsets), dtype=np.int64)
            # C(T) == C(-T), hence S(d) == S(-d) for every finite shape.
            # A complete cubic shape additionally has all signed-permutation
            # symmetries. Neither path rounds coordinates or bins distances.
            if cube_symmetry:
                offsets = np.sort(np.abs(offsets), axis=1)
            else:
                leading = np.argmax(offsets != 0, axis=1)
                negative = offsets[np.arange(len(offsets)), leading] < 0
                offsets[negative] *= -1
            unique, inverse = np.unique(offsets, axis=0, return_inverse=True)
            counts = np.zeros(len(unique), dtype=np.int64)
            np.add.at(counts, inverse, pair_counts)
            self._offsets[a, b], self._offset_counts[a, b] = unique, counts
        self.grouped_basis_displacements = sum(map(len, self._offsets.values()))
        self.distance_entries = self.grouped_basis_displacements * len(self.vectors)
        self.estimated_distance_cache_bytes = self.distance_entries * np.dtype(np.float64).itemsize
        self._distances = None
        self.distance_cache_bytes = 0
        self.cache_build_distance_entries = 0
        self.cache_candidate_payload_peak_bytes = 0
        if self.cache_max_bytes:
            self._distances = {}
            for a, b in self._pairs:
                radius_blocks, count_blocks = [], []
                pair_bytes = 0
                for start, stop, offset_index, translation_index in self._blocks(a, b):
                    distances = np.linalg.norm(
                        self.vectors[translation_index] + self._offsets[a, b][offset_index], axis=1)
                    weights = self.multiplicities[translation_index] * self._offset_counts[a, b][offset_index]
                    radii, counts = _aggregate_radii(distances, weights)
                    self.cache_build_distance_entries += stop - start
                    block_bytes = radii.nbytes + counts.nbytes
                    retained_bytes = self.distance_cache_bytes + pair_bytes + block_bytes
                    if retained_bytes > self.cache_max_bytes:
                        # Do not retain a partial geometry cache or a full raw
                        # distance library after the explicit budget is reached.
                        self._distances = None
                        self.distance_cache_bytes = 0
                        break
                    radius_blocks.append(radii)
                    count_blocks.append(counts)
                    pair_bytes += block_bytes
                    self.cache_candidate_payload_peak_bytes = max(
                        self.cache_candidate_payload_peak_bytes, retained_bytes)
                if self._distances is None:
                    break
                # Combine identical radii across different bounded blocks as
                # well. Sorting workspace is transient, not retained payload.
                radii, counts = _aggregate_radii(np.concatenate(radius_blocks),
                                                np.concatenate(count_blocks))
                radii.setflags(write=False)
                counts.setflags(write=False)
                self._distances[a, b] = radii, counts
                self.distance_cache_bytes += radii.nbytes + counts.nbytes
            # Drop any staged blocks, including the failed-cache path.
            if self._pairs:
                del radius_blocks, count_blocks
        for array in (self.translations, self.multiplicities, self.vectors,
                      *self._positions.values(), *self._offsets.values(), *self._offset_counts.values()):
            array.setflags(write=False)
        self.preprocessing_seconds = perf_counter() - started
        self.evaluation_count = 0
        self.last_evaluation_seconds = None

    def _blocks(self, a, b):
        """Bound both distance and index workspaces across basis and translations."""
        n_translations = len(self.vectors)
        total = len(self._offsets[a, b]) * n_translations
        for start in range(0, total, self.translation_chunk):
            stop = min(start + self.translation_chunk, total)
            index = np.arange(start, stop)
            yield start, stop, index // n_translations, index % n_translations

    @property
    def metadata(self):
        return {
            'method': 'full_cell_lattice_grouped',
            'n_cells': self.n_cells,
            'n_basis_atoms': self.n_basis_atoms,
            'n_translations': len(self.vectors),
            'raw_basis_pairs': self.raw_basis_pairs,
            'grouped_basis_displacements': self.grouped_basis_displacements,
            'displacement_symmetry': self.displacement_symmetry,
            'displacement_grouping': 'exact_float_equality_no_distance_binning',
            'basis_displacement_bytes': sum(value.nbytes for value in self._offsets.values())
                                        + sum(value.nbytes for value in self._offset_counts.values()),
            'distance_entries': self.distance_entries,
            'ungrouped_distance_entries': self.raw_basis_pairs * len(self.vectors),
            'estimated_distance_cache_bytes': self.estimated_distance_cache_bytes,
            'estimated_uncompressed_radius_weight_bytes': 2 * self.estimated_distance_cache_bytes,
            'distance_cache_bytes': self.distance_cache_bytes,
            'radial_cache_entries': sum(len(pair[0]) for pair in self._distances.values())
                                    if self._distances is not None else 0,
            'radial_aggregation': 'exact_float64_radius_equality_integer_weights',
            'cache_build_distance_entries': self.cache_build_distance_entries,
            'cache_candidate_payload_peak_bytes': self.cache_candidate_payload_peak_bytes,
            'cache_max_bytes': self.cache_max_bytes,
            'distance_storage': 'cached' if self._distances is not None else 'streamed',
            'translation_chunk': self.translation_chunk,
            'preprocessing_seconds': self.preprocessing_seconds,
            'evaluation_count': self.evaluation_count,
            'last_evaluation_seconds': self.last_evaluation_seconds,
            'memory_scope': 'cache budget includes retained radii and integer weights; temporary sorting arrays, '
                            'basis displacements, FFT shape and total process memory are not capped',
        }

    def iq(self, q, max_matrix_entries=500_000):
        """Evaluate exact finite powder intensity, reusing q-independent geometry."""
        started = perf_counter()
        q = validate_q(q)
        max_matrix_entries = _positive_integer(max_matrix_entries, 'max_matrix_entries')
        factors = form_factors(self.elements, q)
        result = np.zeros(len(q))
        for a, b in self._pairs:
            # Same-species offset counts include both orders and all self terms.
            # Unlike-species reverse pairs match after summing both +/-T.
            multiplier = (1 if a == b else 2) * factors[a] * factors[b]
            if self._distances is not None:
                radii, weights = self._distances[a, b]
                result += multiplier * weighted_sinc_sum(
                    radii, q, weights, max_matrix_entries=max_matrix_entries)
                continue
            for start, stop, offset_index, translation_index in self._blocks(a, b):
                distances = np.linalg.norm(
                    self.vectors[translation_index] + self._offsets[a, b][offset_index], axis=1)
                weights = self.multiplicities[translation_index] * self._offset_counts[a, b][offset_index]
                radii, weights = _aggregate_radii(distances, weights)
                result += multiplier * weighted_sinc_sum(
                    radii, q, weights, max_matrix_entries=max_matrix_entries)
        self.evaluation_count += 1
        self.last_evaluation_seconds = perf_counter() - started
        return result


def lattice_grouped_iq(basis, cell_vectors_A, mask, q):
    """Exact finite full-cell powder sum; retain LatticePairCache for repeated q grids.

    The one-shot compatibility API retains shape counts and streams distances,
    avoiding a distance library which would be discarded immediately afterward.
    """
    q = validate_q(q)
    return LatticePairCache(basis, cell_vectors_A, mask, cache_max_bytes=0).iq(q)
