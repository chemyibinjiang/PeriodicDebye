"""Full, unnormalised elastic Debye intensities in electron^2.
Positions: angstrom. q: inverse angstrom. No absorption, anomalous factors,
thermal displacement, instrumental broadening, or reciprocal-space cutoff.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from itertools import combinations
import math
import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import pdist, cdist
import gemmi

FloatArray = NDArray[np.float64]

@dataclass(frozen=True)
class AtomCloud:
    elements: tuple[str, ...]
    xyz: FloatArray
    name: str = ''

    def __post_init__(self):
        elements = tuple(self.elements)
        xyz = np.array(self.xyz, dtype=np.float64, copy=True).reshape((-1, 3))
        if len(elements) != len(xyz) or not np.isfinite(xyz).all():
            raise ValueError('Elements and finite N-by-3 coordinates must match.')
        for el in set(elements):
            if gemmi.Element(el).atomic_number == 0:
                raise ValueError(f'Unrecognised chemical element: {el!r}')
        xyz.setflags(write=False)
        object.__setattr__(self, 'elements', elements)
        object.__setattr__(self, 'xyz', xyz)

    def __len__(self):
        return len(self.elements)

    def moved(self, translation=(0., 0., 0.), rotation=None):
        r = np.eye(3) if rotation is None else np.asarray(rotation, dtype=float)
        if r.shape != (3, 3) or not np.allclose(r.T @ r, np.eye(3), atol=1e-10) or np.linalg.det(r) < 0:
            raise ValueError('rotation must be a proper orthogonal 3-by-3 matrix.')
        t = np.asarray(translation, dtype=float)
        if t.shape != (3,):
            raise ValueError('translation must have 3 coordinates.')
        return AtomCloud(self.elements, self.xyz @ r.T + t, self.name)

    def write_xyz(self, path):
        path = Path(path)
        with path.open('w') as out:
            out.write(f'{len(self)}\n{self.name}; coordinates in angstrom\n')
            for el, r in zip(self.elements, self.xyz):
                out.write(f'{el} {r[0]:.12g} {r[1]:.12g} {r[2]:.12g}\n')

    @classmethod
    def read_xyz(cls, path):
        lines = Path(path).read_text().splitlines()
        if len(lines) < 2:
            raise ValueError('XYZ requires an atom-count line and a comment line.')
        n = int(lines[0]); rows = [line.split() for line in lines[2:] if line.strip()]
        if n < 0 or len(rows) != n or any(len(row) < 4 for row in rows):
            raise ValueError('XYZ atom count/coordinate rows are inconsistent.')
        return cls(tuple(row[0] for row in rows),
                   np.array([[float(x) for x in row[1:4]] for row in rows]).reshape(-1, 3), lines[1])

def concatenate(clouds, name='combined'):
    clouds = list(clouds)
    if not clouds:
        return AtomCloud((), np.empty((0, 3)), name)
    return AtomCloud(tuple(el for cloud in clouds for el in cloud.elements),
                     np.concatenate([c.xyz for c in clouds]), name)

def validate_q(q):
    q = np.asarray(q, dtype=np.float64)
    if q.ndim != 1 or q.size == 0 or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('q must be a nonempty finite, nonnegative 1D array in inverse angstrom.')
    return q

def form_factors(elements, q):
    """Neutral-atom IT92 f0(q), using Gemmi tables; same factors in all engines.
    The table approximations need not give exactly the integer atomic number at q=0.
    """
    q = validate_q(q)
    out = {}
    ss = (q / (4 * np.pi)) ** 2
    for el in sorted(set(elements)):
        atom = gemmi.Element(el)
        if atom.atomic_number == 0:
            raise ValueError(f'Unknown element {el}.')
        coeff = atom.it92
        if coeff is None:
            raise ValueError(f'No neutral IT92 scattering factors for {el}.')
        a, b = np.asarray(coeff.a, dtype=float), np.asarray(coeff.b, dtype=float)
        out[el] = np.sum(a[:, None] * np.exp(-b[:, None] * ss), axis=0) + coeff.c
    return out

def _positive_integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


def weighted_sinc_sum(distances, q, weights=None, max_matrix_entries=500_000):
    """Evaluate sum_k weights[k] sinc(q*d[k]) with tiled distance/q workspace.

    The limit counts entries of each distance-by-q array, not total process
    memory: inputs, output, factors and temporary arithmetic also use memory.
    """
    max_matrix_entries = _positive_integer(max_matrix_entries, 'max_matrix_entries')
    d = np.asarray(distances, dtype=float).ravel()
    q = validate_q(q)
    w = None if weights is None else np.asarray(weights, dtype=float).ravel()
    if w is not None and w.shape != d.shape:
        raise ValueError('weights must match distances.')
    out = np.zeros(q.size)
    q_chunk = min(len(q), max_matrix_entries)
    for q_start in range(0, len(q), q_chunk):
        q_stop = min(q_start + q_chunk, len(q))
        chunk = max(1, max_matrix_entries // (q_stop - q_start))
        for start in range(0, len(d), chunk):
            stop = min(start + chunk, len(d))
            values = np.sinc(d[start:stop, None] * q[None, q_start:q_stop] / np.pi)
            if w is None:
                out[q_start:q_stop] += values.sum(axis=0)
            else:
                out[q_start:q_stop] += (values * w[start:stop, None]).sum(axis=0)
    return out

def _distance_blocks(xa, xb=None, atom_chunk=256):
    """Yield every unordered same-set or every cross-set distance once."""
    for start in range(0, len(xa), atom_chunk):
        a = xa[start:start + atom_chunk]
        if xb is None:
            yield pdist(a)
            for other in range(start + atom_chunk, len(xa), atom_chunk):
                yield cdist(a, xa[other:other + atom_chunk]).ravel()
        else:
            for other in range(0, len(xb), atom_chunk):
                yield cdist(a, xb[other:other + atom_chunk]).ravel()


def direct_iq(cloud: AtomCloud, q, atom_chunk=256, max_matrix_entries=500_000):
    """Independent self + twice-unordered-pairs Debye reference.

    Both distance construction and sinc evaluation are tiled. No full N-by-N
    distance matrix or full condensed pair list is retained.
    """
    q = validate_q(q)
    atom_chunk = _positive_integer(atom_chunk, 'atom_chunk')
    max_matrix_entries = _positive_integer(max_matrix_entries, 'max_matrix_entries')
    factors = form_factors(cloud.elements, q)
    labels = np.asarray(cloud.elements)
    species = list(factors)
    out = np.zeros(len(q))
    for i, a in enumerate(species):
        xa = cloud.xyz[labels == a]
        out += len(xa) * factors[a] ** 2
        for distances in _distance_blocks(xa, atom_chunk=atom_chunk):
            out += 2 * factors[a] ** 2 * weighted_sinc_sum(
                distances, q, max_matrix_entries=max_matrix_entries)
        for b in species[i + 1:]:
            xb = cloud.xyz[labels == b]
            for distances in _distance_blocks(xa, xb, atom_chunk=atom_chunk):
                out += 2 * factors[a] * factors[b] * weighted_sinc_sum(
                    distances, q, max_matrix_entries=max_matrix_entries)
    return out

def cross_iq(a: AtomCloud, b: AtomCloud, q, atom_chunk=256, max_matrix_entries=500_000):
    """Full host/guest interference with bounded atom-block distance arrays."""
    q = validate_q(q)
    atom_chunk = _positive_integer(atom_chunk, 'atom_chunk')
    max_matrix_entries = _positive_integer(max_matrix_entries, 'max_matrix_entries')
    f = form_factors(a.elements + b.elements, q)
    la, lb = np.asarray(a.elements), np.asarray(b.elements)
    out = np.zeros(len(q))
    for ea in sorted(set(a.elements)):
        for eb in sorted(set(b.elements)):
            for distances in _distance_blocks(a.xyz[la == ea], b.xyz[lb == eb], atom_chunk=atom_chunk):
                out += 2 * f[ea] * f[eb] * weighted_sinc_sum(
                    distances, q, max_matrix_entries=max_matrix_entries)
    return out

def pair_decomposition(host: AtomCloud, guests: list[AtomCloud], q, host_iq=None):
    q = validate_q(q)
    hh = direct_iq(host, q) if host_iq is None else np.asarray(host_iq)
    intra = sum((direct_iq(g, q) for g in guests), start=np.zeros(len(q)))
    hg = sum((cross_iq(host, g, q) for g in guests), start=np.zeros(len(q)))
    gg = sum((cross_iq(a, b, q) for a, b in combinations(guests, 2)), start=np.zeros(len(q)))
    return dict(host=hh, guest_internal=intra, host_guest=hg, guest_guest=gg,
                total=hh + intra + hg + gg)

def upstream_kernel_iq(cloud, q, method='grid', precision='float64', num_threads=1,
                       batch_size=None, force_grid=False, device='cpu', return_info=False,
                       max_pair_q_entries=500_000):
    """Standalone harness around adapted DebyeCalculator forward source.
    NOT its public API. Uses the source's distance-grid rule and job construction.
    force_grid=True disables the small-system direct fallback for validation.
    Direct and fallback pair-by-q arrays are capped by max_pair_q_entries through
    pair batches and q tiling. This does not bound complete grid histograms, grid
    job lists or total process/device memory. The unmodified vendor direct kernel
    needs at least one full species row; raise if that alone exceeds the cap.
    """
    import torch
    from .vendor.debyecalculator_kernel import ForwardKernel, _Grid
    q = validate_q(q)
    if method not in ('grid', 'direct') or precision not in ('float32', 'float64'):
        raise ValueError('Invalid method or precision.')
    num_threads = _positive_integer(num_threads, 'num_threads')
    max_pair_q_entries = _positive_integer(max_pair_q_entries, 'max_pair_q_entries')
    if batch_size is not None:
        batch_size = _positive_integer(batch_size, 'batch_size')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is not available; no silent CPU fallback.')
    if len(cloud) == 0:
        result, info = np.zeros(len(q)), {'actual_method': 'empty', 'n_atoms': 0}
        return (result, info) if return_info else result
    dtype = getattr(torch, precision)
    engine = ForwardKernel(device=device, dtype=dtype, pair_sum=method, batch_size=batch_size)
    f = form_factors(cloud.elements, q)
    species = list(f)
    element_ids = np.array([species.index(e) for e in cloud.elements])
    xyz = torch.tensor(np.array(cloud.xyz), dtype=dtype, device=device)
    qt = torch.tensor(q, dtype=dtype, device=device)
    ids = torch.tensor(element_ids, dtype=torch.long, device=device)
    ff = torch.tensor(np.stack([f[e] for e in species]), dtype=dtype, device=device)
    occ = torch.ones(len(cloud), dtype=dtype, device=device)
    indices = [torch.nonzero(ids == i).squeeze(-1) for i in range(len(species))]
    pairs = [(a, b) for a in range(len(species)) for b in range(a, len(species))]
    n_pairs = len(cloud) * (len(cloud) - 1) // 2
    step_factor = 0.03 if dtype == torch.float64 else 0.12
    dr = 2.0 ** math.floor(math.log2(step_factor / max(float(q.max()), 1e-3)))
    centered = xyz - xyz.mean(dim=0)
    d_max = 2 * torch.linalg.norm(centered, dim=1).max().item()
    n_nodes = math.ceil(d_max / dr) + 5
    info = dict(n_atoms=len(cloud), n_pairs=n_pairs, distance_grid_A=dr,
                n_nodes=n_nodes, precision=precision, num_threads_requested=num_threads,
                device=device, upstream_blob='89cd257ac681d3859457db7e8220cbc325c6953d',
                direct_pair_q_entries_limit=max_pair_q_entries,
                grid_memory_is_capped=False)
    with torch.no_grad():
        if method == 'direct' or (not force_grid and n_pairs <= n_nodes):
            largest_species = max(index.numel() for index in indices)
            if largest_species > max_pair_q_entries:
                raise ValueError('Direct kernel workspace cap cannot fit one species row; '
                                 'increase max_pair_q_entries or use the tiled reference backend.')
            q_chunk = min(len(q), max(1, max_pair_q_entries // largest_species))
            # A grid request that falls back must use a direct-sized batch.
            engine.pair_sum = 'direct'
            engine.batch_size = min(batch_size or 10_000, max_pair_q_entries // q_chunk)
            pair_iq = torch.empty(len(q), dtype=dtype, device=device)
            for start in range(0, len(q), q_chunk):
                stop = min(start + q_chunk, len(q))
                pair_iq[start:stop] = engine._direct_pair_iq(
                    xyz, occ, indices, pairs, ff[:, start:stop], qt[start:stop])
            info['actual_method'] = 'direct' if method == 'direct' else 'direct_fallback'
            info['num_threads_jobs'] = 1
            info['direct_q_chunk'] = q_chunk
            info['direct_pair_batch_size'] = engine.batch_size
        else:
            n_threads = min(num_threads, max(1, n_pairs // 2**16)) if device == 'cpu' else 1
            pairs_per_job = max(1, engine._effective_batch_size() // n_threads)
            jobs = []
            for pair_number, (a, b) in enumerate(pairs):
                n_a, n_b = indices[a].numel(), indices[b].numel()
                rows_per_batch = max(1, pairs_per_job // max(1, n_b))
                for start in range(0, n_a, rows_per_batch):
                    if a == b and start + 1 >= n_b:
                        break
                    jobs.append((pair_number, a, b, start, min(start + rows_per_batch, n_a)))
            grid = _Grid(jobs, indices, len(pairs), n_nodes, dr, torch.float64, qt, n_threads)
            sums = engine._grid_pair_sums(xyz, occ, grid)
            pair_iq = sum((ff[a] * ff[b] * sums[k] for k, (a, b) in enumerate(pairs)),
                          start=torch.zeros(len(q), dtype=dtype, device=device))
            info['actual_method'] = 'grid'
            info['num_threads_jobs'] = n_threads
        # Deliberately use the conventional full Debye normalisation.
        self_iq = sum((len(indices[k]) * ff[k] ** 2 for k in range(len(species))),
                       start=torch.zeros(len(q), dtype=dtype, device=device))
        result = (2 * pair_iq + self_iq).detach().cpu().numpy().astype(float)
    return (result, info) if return_info else result
