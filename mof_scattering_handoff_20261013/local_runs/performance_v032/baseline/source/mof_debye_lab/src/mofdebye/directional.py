"""Finite directional scattering and convergent whole-object powder quadrature.

Positions are Cartesian angstrom and q vectors are inverse angstrom. Amplitudes
are in electrons and intensities in electron squared, with no normalization by
atom/cell count. Neutral real IT92 factors match the independent Debye reference.
These reciprocal-space evaluations are not detector/Ewald-sphere projections.
"""
from __future__ import annotations

import numpy as np

from .core import AtomCloud, form_factors, validate_q


def validate_q_vectors(q_vectors):
    """Return a nonempty finite float64 array of Cartesian row q vectors."""
    vectors = np.asarray(q_vectors, dtype=np.float64)
    if (vectors.ndim != 2 or vectors.shape[1] != 3 or len(vectors) == 0
            or not np.isfinite(vectors).all()):
        raise ValueError('q_vectors must be a nonempty finite Q-by-3 Cartesian array.')
    return vectors


def _positive_int(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def amplitude(cloud: AtomCloud, q_vectors, atom_chunk=256, q_chunk=256):
    """Compute sum_i f_i(|q|) exp(i q.r_i), including an empty-cloud limit.

    The atom/q phase and factor workspace is tiled: it never contains an
    all-atoms by all-q array. Besides input/output storage, the largest phase
    array has at most ``atom_chunk * q_chunk`` complex entries. Factors are
    evaluated only for the current q tile. Particle coordinates remain unwrapped.
    """
    vectors = validate_q_vectors(q_vectors)
    atom_chunk = _positive_int(atom_chunk, 'atom_chunk')
    q_chunk = _positive_int(q_chunk, 'q_chunk')
    result = np.zeros(len(vectors), dtype=np.complex128)
    if not len(cloud):
        return result
    labels = np.asarray(cloud.elements)
    for q_start in range(0, len(vectors), q_chunk):
        q_stop = min(q_start + q_chunk, len(vectors))
        tile = vectors[q_start:q_stop]
        factors = form_factors(cloud.elements, np.linalg.norm(tile, axis=1))
        tile_sum = np.zeros(len(tile), dtype=np.complex128)
        for atom_start in range(0, len(cloud), atom_chunk):
            atom_stop = min(atom_start + atom_chunk, len(cloud))
            phase = np.exp(1j * (tile @ cloud.xyz[atom_start:atom_stop].T))
            atom_labels = labels[atom_start:atom_stop]
            for element in np.unique(atom_labels):
                tile_sum += factors[element] * phase[:, atom_labels == element].sum(axis=1)
        result[q_start:q_stop] = tile_sum
    return result


def intensity(cloud: AtomCloud, q_vectors, atom_chunk=256, q_chunk=256):
    """Finite, unnormalised |A(q)| squared in electron squared."""
    return np.abs(amplitude(cloud, q_vectors, atom_chunk=atom_chunk, q_chunk=q_chunk)) ** 2


def amplitude_components(host: AtomCloud, guests, q_vectors, atom_chunk=256, q_chunk=256):
    """Retain host, particle-internal, host-guest and between-guest terms.

    ``guests`` is an iterable of complete explicit particle configurations in
    the host's Cartesian frame. Cross terms may be negative. This describes a
    single realization, not an ensemble mean or a periodically replicated cell.
    """
    vectors = validate_q_vectors(q_vectors)
    host_a = amplitude(host, vectors, atom_chunk=atom_chunk, q_chunk=q_chunk)
    guest_a = np.zeros(len(vectors), dtype=np.complex128)
    internal = np.zeros(len(vectors), dtype=np.float64)
    for guest in guests:
        current = amplitude(guest, vectors, atom_chunk=atom_chunk, q_chunk=q_chunk)
        guest_a += current
        internal += np.abs(current) ** 2
    total_a = host_a + guest_a
    return {
        'host': np.abs(host_a) ** 2,
        'guest_internal': internal,
        'host_guest': 2 * np.real(host_a.conj() * guest_a),
        'guest_guest': np.abs(guest_a) ** 2 - internal,
        'total': np.abs(total_a) ** 2,
        'host_amplitude': host_a,
        'guest_amplitude': guest_a,
        'amplitude': total_a,
    }


def angular_grid(n_mu=24, n_phi=48):
    """Gauss-Legendre cos(theta) x periodic phi quadrature, weights sum to one.

    The resulting average is (1/4pi) integral over the unit sphere. This is a
    deterministic grid, not a convergence guarantee: increase both resolutions
    with q times object extent, and compare successive refinements.
    """
    n_mu = _positive_int(n_mu, 'n_mu')
    n_phi = _positive_int(n_phi, 'n_phi')
    mu, mu_weights = np.polynomial.legendre.leggauss(n_mu)
    phi = 2 * np.pi * np.arange(n_phi) / n_phi
    radius = np.sqrt(np.maximum(0., 1 - mu ** 2))
    directions = np.stack([
        radius[:, None] * np.cos(phi)[None, :],
        radius[:, None] * np.sin(phi)[None, :],
        np.broadcast_to(mu[:, None], (n_mu, n_phi)),
    ], axis=-1).reshape(-1, 3)
    weights = np.repeat(mu_weights / (2 * n_phi), n_phi)
    return directions, weights


def powder_average(evaluator, q, n_mu=24, n_phi=48, q_chunk=8):
    """Average a directional intensity evaluator over the whole object's SO(3).

    The evaluator accepts a Q-by-3 vector array and returns Q finite real
    intensities. At most ``q_chunk * n_mu * n_phi`` q vectors are passed at once.
    Both the cell and shape amplitudes must be combined *before* averaging.
    Finite-sample unbiased moment estimators may be negative, so their signs are
    preserved; this routine never clips results into physical nonnegativity.
    """
    q = validate_q(q)
    q_chunk = _positive_int(q_chunk, 'q_chunk')
    directions, weights = angular_grid(n_mu, n_phi)
    result = np.empty(len(q), dtype=np.float64)
    for start in range(0, len(q), q_chunk):
        stop = min(start + q_chunk, len(q))
        vectors = (q[start:stop, None, None] * directions[None, :, :]).reshape(-1, 3)
        values = np.asarray(evaluator(vectors))
        if (values.shape != (len(vectors),) or np.iscomplexobj(values)
                or not np.isfinite(values).all()):
            raise ValueError('The directional evaluator must return one finite real intensity per q vector.')
        result[start:stop] = np.asarray(values, dtype=np.float64).reshape(stop - start, -1) @ weights
    return result


def lattice_shape_amplitude(cell_vectors_A, mask, q_vectors, cell_chunk=256, q_chunk=256):
    """Compute sum_n w_n exp(i q.(n @ A)) for a binary full-cell mask.

    Lattice vectors are rows of A, so triclinic translations use ``n @ A``.
    The mask origin is integer index (0,0,0). This is a finite shape, with no
    cyclic wrapping or minimum-image convention. Workspace is tiled over cells
    and q vectors. It does not describe atomically cut/reconstructed surfaces.
    """
    vectors = validate_q_vectors(q_vectors)
    cell_chunk = _positive_int(cell_chunk, 'cell_chunk')
    q_chunk = _positive_int(q_chunk, 'q_chunk')
    cell = np.asarray(cell_vectors_A, dtype=np.float64)
    if (cell.shape != (3, 3) or not np.isfinite(cell).all()
            or abs(np.linalg.det(cell)) < 1e-12):
        raise ValueError('cell_vectors_A must be a nonsingular finite 3-by-3 matrix.')
    mask = np.asarray(mask)
    if mask.ndim != 3 or not np.all(np.isin(mask, (0, 1))) or not np.any(mask):
        raise ValueError('mask must be a nonempty three-dimensional binary array.')
    indices = np.argwhere(mask)
    result = np.zeros(len(vectors), dtype=np.complex128)
    for q_start in range(0, len(vectors), q_chunk):
        tile = vectors[q_start:q_start + q_chunk]
        tile_sum = np.zeros(len(tile), dtype=np.complex128)
        for start in range(0, len(indices), cell_chunk):
            positions = indices[start:start + cell_chunk] @ cell
            tile_sum += np.exp(1j * (tile @ positions.T)).sum(axis=1)
        result[q_start:q_start + len(tile)] = tile_sum
    return result


def cell_shape_amplitude(basis: AtomCloud, cell_vectors_A, mask, q_vectors,
                         atom_chunk=256, q_chunk=256, cell_chunk=256):
    """Exact F_cell(q) * S_shape(q) for complete identical retained cells."""
    vectors = validate_q_vectors(q_vectors)
    cell_a = amplitude(basis, vectors, atom_chunk=atom_chunk, q_chunk=q_chunk)
    shape_a = lattice_shape_amplitude(cell_vectors_A, mask, vectors,
                                      cell_chunk=cell_chunk, q_chunk=q_chunk)
    return cell_a * shape_a
