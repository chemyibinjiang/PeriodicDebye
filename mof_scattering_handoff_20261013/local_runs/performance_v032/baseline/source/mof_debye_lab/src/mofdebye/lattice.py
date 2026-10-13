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

class LatticePairCache:
    """Reusable full-cell translation counts and optional pair-distance cache.

    ``A`` has lattice vectors as rows and basis coordinates are Cartesian.
    Construction computes the linear FFT shape autocorrelation once. All basis
    pair/translation distances are retained only if they fit ``cache_max_bytes``;
    otherwise distances are recomputed in bounded translation blocks in ``iq``.
    The distance budget does not include shape FFT arrays, input/output storage,
    form factors or Python overhead, and is not a total-memory guarantee.

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
        self.distance_entries = sum(len(self._positions[a]) * len(self._positions[b])
                                    for a, b in self._pairs) * len(self.vectors)
        self.estimated_distance_cache_bytes = self.distance_entries * np.dtype(np.float64).itemsize
        self._distances = None
        if self.estimated_distance_cache_bytes <= self.cache_max_bytes:
            self._distances = {}
            for a, b in self._pairs:
                xa, xb = self._positions[a], self._positions[b]
                distances = np.empty((len(xa) * len(xb), len(self.vectors)), dtype=np.float64)
                row = 0
                for ra in xa:
                    for rb in xb:
                        for start in range(0, len(self.vectors), self.translation_chunk):
                            stop = min(start + self.translation_chunk, len(self.vectors))
                            distances[row, start:stop] = np.linalg.norm(
                                self.vectors[start:stop] + ra - rb, axis=1)
                        row += 1
                distances.setflags(write=False)
                self._distances[a, b] = distances
        for array in (self.translations, self.multiplicities, self.vectors, *self._positions.values()):
            array.setflags(write=False)
        self.preprocessing_seconds = perf_counter() - started
        self.evaluation_count = 0
        self.last_evaluation_seconds = None

    @property
    def metadata(self):
        return {
            'method': 'full_cell_lattice_grouped',
            'n_cells': self.n_cells,
            'n_basis_atoms': self.n_basis_atoms,
            'n_translations': len(self.vectors),
            'distance_entries': self.distance_entries,
            'estimated_distance_cache_bytes': self.estimated_distance_cache_bytes,
            'distance_cache_bytes': self.estimated_distance_cache_bytes if self._distances is not None else 0,
            'cache_max_bytes': self.cache_max_bytes,
            'distance_storage': 'cached' if self._distances is not None else 'streamed',
            'translation_chunk': self.translation_chunk,
            'preprocessing_seconds': self.preprocessing_seconds,
            'evaluation_count': self.evaluation_count,
            'last_evaluation_seconds': self.last_evaluation_seconds,
            'memory_scope': 'distance cache budget; FFT shape and total process memory not capped',
        }

    def iq(self, q, max_matrix_entries=500_000):
        """Evaluate exact finite powder intensity, reusing q-independent geometry."""
        started = perf_counter()
        q = validate_q(q)
        max_matrix_entries = _positive_integer(max_matrix_entries, 'max_matrix_entries')
        factors = form_factors(self.elements, q)
        result = np.zeros(len(q))
        for a, b in self._pairs:
            # Same-species ordered basis pairs include the self terms at T=0.
            # Unlike-species reverse pairs match after summing both +/-T.
            multiplier = (1 if a == b else 2) * factors[a] * factors[b]
            if self._distances is not None:
                for distances in self._distances[a, b]:
                    result += multiplier * weighted_sinc_sum(
                        distances, q, self.multiplicities, max_matrix_entries=max_matrix_entries)
            else:
                for ra in self._positions[a]:
                    for rb in self._positions[b]:
                        for start in range(0, len(self.vectors), self.translation_chunk):
                            stop = min(start + self.translation_chunk, len(self.vectors))
                            distances = np.linalg.norm(self.vectors[start:stop] + ra - rb, axis=1)
                            result += multiplier * weighted_sinc_sum(
                                distances, q, self.multiplicities[start:stop],
                                max_matrix_entries=max_matrix_entries)
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
