"""MODIFIED extract of DebyeCalculator's forward kernels (Apache-2.0).
See NOTICE.txt for attribution, upstream blob hash, and the scope of changes.
This is a standalone kernel harness, not the installed DebyeCalculator package.
"""
from __future__ import annotations
import contextlib
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import torch

_THREADS_LOCK = threading.Lock()
_THREADS_STATE = {'active': 0, 'saved': None}

@contextlib.contextmanager
def _single_intra_op_thread():
    with _THREADS_LOCK:
        if _THREADS_STATE['active'] == 0:
            _THREADS_STATE['saved'] = torch.get_num_threads()
            torch.set_num_threads(1)
        _THREADS_STATE['active'] += 1
    try:
        yield
    finally:
        with _THREADS_LOCK:
            _THREADS_STATE['active'] -= 1
            if _THREADS_STATE['active'] == 0:
                torch.set_num_threads(_THREADS_STATE['saved'])

@dataclass
class _Grid:
    jobs: list
    atom_indices: list
    n_element_pairs: int
    n_nodes: int
    dr: float
    acc_dtype: torch.dtype
    q: torch.Tensor
    n_threads: int

class ForwardKernel:
    """Forward arithmetic adapted from the inspected upstream source."""
    def __init__(self, device='cpu', dtype=torch.float64, pair_sum='grid', batch_size=None):
        self.device = device
        self.dtype = dtype
        self.pair_sum = pair_sum
        self.batch_size = batch_size
        self.rthres = 0.0

    def _effective_batch_size(self):
        if self.batch_size:
            return int(self.batch_size)
        return 3_000_000 if self.pair_sum == 'grid' else 10_000

    def _spread_pairs(self, jobs, xyz, occupancy, atom_indices,
                      n_element_pairs, n_nodes, dr, acc_dtype, progress=None):
        hist = torch.zeros((n_element_pairs, n_nodes, 4), device=self.device, dtype=acc_dtype)
        batch_moments = torch.zeros((4, n_nodes), device=self.device, dtype=self.dtype)
        uniform_occupancy = bool(torch.all(occupancy == 1))
        for pair_number, a, b, start, stop in jobs:
            idx_a, idx_b = atom_indices[a], atom_indices[b]
            n_b = idx_b.numel()
            col_start = start + 1 if a == b else 0
            n_rows, n_cols = stop - start, n_b - col_start
            d = torch.cdist(xyz[idx_a[start:stop]], xyz[idx_b[col_start:]],
                            compute_mode='donot_use_mm_for_euclid_dist')
            w = None
            if not uniform_occupancy:
                w = occupancy[idx_a[start:stop]].unsqueeze(-1) * occupancy[idx_b[col_start:]].unsqueeze(0)
            if self.rthres > 0:
                above = (d >= self.rthres).to(self.dtype)
                w = above if w is None else w * above
            u = d.reshape(-1).mul_(1 / dr)
            bins = torch.floor(u)
            moments = torch.empty((4, u.numel()), device=self.device, dtype=self.dtype)
            torch.sub(u, bins, out=moments[1])
            torch.mul(moments[1], moments[1], out=moments[2])
            torch.mul(moments[2], moments[1], out=moments[3])
            if w is None:
                moments[0].fill_(1.0)
            else:
                w = w.reshape(-1)
                moments[0].copy_(w)
                moments[1:].mul_(w)
            if a == b:
                n_square = min(n_rows, n_cols)
                rows = torch.arange(n_rows, device=self.device).unsqueeze(-1)
                cols = torch.arange(n_square, device=self.device).unsqueeze(0)
                moments.view(4, n_rows, n_cols)[:, :, :n_square].masked_fill_(cols < rows, 0.0)
            batch_moments.zero_()
            batch_moments.index_add_(1, bins.long(), moments)
            hist[pair_number] += batch_moments.T
            if progress is not None:
                progress.step()
        return hist

    def _run_jobs(self, function, jobs, n_threads):
        def run(subset):
            with torch.no_grad():
                return function(subset)
        n_workers = min(n_threads, len(jobs))
        if n_workers <= 1:
            return [run(jobs)]
        with _single_intra_op_thread():
            with ThreadPoolExecutor(n_workers) as executor:
                return list(executor.map(lambda worker: run(jobs[worker::n_workers]), range(n_workers)))

    def _grid_pair_sums(self, xyz, occupancy, grid):
        spread_args = (xyz, occupancy, grid.atom_indices, grid.n_element_pairs,
                       grid.n_nodes, grid.dr, grid.acc_dtype, getattr(grid, 'progress', None))
        hists = self._run_jobs(lambda jobs: self._spread_pairs(jobs, *spread_args), grid.jobs, grid.n_threads)
        hist = torch.stack(hists).sum(0)
        lagrange_coefficients = torch.tensor([
            [0.0, -1 / 3, 1 / 2, -1 / 6],
            [1.0, -1 / 2, -1.0, 1 / 2],
            [0.0, 1.0, 1 / 2, -1 / 2],
            [0.0, -1 / 6, 0.0, 1 / 6],
        ], device=self.device, dtype=grid.acc_dtype)
        bin_weights = hist @ lagrange_coefficients.T
        node_weights = torch.zeros((grid.n_element_pairs, grid.n_nodes + 3), device=self.device, dtype=grid.acc_dtype)
        for j in range(4):
            node_weights[:, j:j + grid.n_nodes] += bin_weights[:, :, j]
        return self._node_sinc_sums(node_weights, grid.dr, grid.q).to(self.dtype)

    def _direct_pair_iq(self, xyz, occupancy, atom_indices, element_pairs, form_factors, q):
        pair_iq = torch.zeros(q.numel(), device=self.device, dtype=self.dtype)
        batch_size = self._effective_batch_size()
        for a, b in element_pairs:
            idx_a, idx_b = atom_indices[a], atom_indices[b]
            n_b = idx_b.numel()
            rows_per_batch = max(1, batch_size // max(1, n_b))
            sinc_sum = torch.zeros(q.numel(), device=self.device, dtype=self.dtype)
            for start in range(0, idx_a.numel(), rows_per_batch):
                stop = min(start + rows_per_batch, idx_a.numel())
                d = torch.cdist(xyz[idx_a[start:stop]], xyz[idx_b],
                                compute_mode='donot_use_mm_for_euclid_dist')
                w = occupancy[idx_a[start:stop]].unsqueeze(-1) * occupancy[idx_b].unsqueeze(0)
                valid = d >= self.rthres
                if a == b:
                    rows = torch.arange(start, stop, device=self.device).unsqueeze(-1)
                    valid &= torch.arange(n_b, device=self.device).unsqueeze(0) > rows
                d, w = d[valid], w[valid]
                sinc_sum = sinc_sum + w @ torch.sinc(d.unsqueeze(-1) * q.unsqueeze(0) / math.pi)
            pair_iq = pair_iq + form_factors[a] * form_factors[b] * sinc_sum
        return pair_iq

    def _node_sinc_sums(self, node_weights, dr, q):
        acc_dtype = node_weights.dtype
        n_pairs, n_nodes = node_weights.shape
        q = q.to(acc_dtype)
        B = max(1, math.ceil(math.sqrt(n_nodes)))
        C = math.ceil(n_nodes / B)
        r = (torch.arange(C * B, device=self.device, dtype=acc_dtype) - 1) * dr
        r_safe = torch.where(r == 0, torch.ones_like(r), r)
        weights = torch.zeros((n_pairs, C * B), device=self.device, dtype=acc_dtype)
        weights[:, :n_nodes] = node_weights
        v = torch.where(r == 0, torch.zeros_like(weights), weights / r_safe).view(n_pairs, C, B)
        r_fine = (torch.arange(B, device=self.device, dtype=acc_dtype) * dr).unsqueeze(-1) * q
        cos_fine, sin_fine = torch.cos(r_fine), torch.sin(r_fine)
        sums = torch.zeros((n_pairs, q.numel()), device=self.device, dtype=acc_dtype)
        blocks_per_chunk = max(1, self._effective_batch_size() // max(1, n_pairs * q.numel()))
        for start in range(0, C, blocks_per_chunk):
            stop = min(start + blocks_per_chunk, C)
            r_coarse = ((torch.arange(start, stop, device=self.device, dtype=acc_dtype) * B - 1) * dr).unsqueeze(-1) * q
            v_chunk = v[:, start:stop]
            sums += (torch.sin(r_coarse) * (v_chunk @ cos_fine)).sum(1)
            sums += (torch.cos(r_coarse) * (v_chunk @ sin_fine)).sum(1)
        total = node_weights.sum(1, keepdim=True)
        q_safe = torch.where(q == 0, torch.ones_like(q), q)
        at_zero = weights[:, (r == 0).nonzero().squeeze(-1)].sum(1, keepdim=True)
        return torch.where(q == 0, total, sums / q_safe + at_zero)
