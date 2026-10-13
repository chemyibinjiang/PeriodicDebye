# Resolved finite PXRD near 100 nm

The optional `radial_moments` CPU backend calculates the finite Debye powder
intensity of a deterministic basis repeated through a binary cell mask. It
includes self scattering and every retained cell-pair interference term. The
40-by-40-by-40 MOF-5 example represents 27,136,000 atoms in a 103.328 nm cube.
No atom array of that size is materialized. No instrumental broadening is added.

## Run the real-MOF example

From this directory, install the optional compiled backend and run:

```powershell
..\.venv\Scripts\python.exe -m pip install -e ".[fast]"
..\.venv\Scripts\python.exe scripts/benchmark_large_powder.py --cells 40 --threads 6 --out ../local_runs/my_100nm_pxrd
```

The output directory must be new. This writes a reusable `config.json`, dense
`pattern.csv` and `pattern.npz`, source hashes, input checksum, timing and memory
metadata. The default q grid is 0--6 inverse angstrom in 0.001 steps (6,001
points). This gives roughly six samples over the scale 2*pi/L for the target
cube. The q grid still needs refinement when precise peak positions or
integrated peak areas are required.

For the denser example (60,001 points), add `--points 60001`. The measured cost
was 8.79 seconds on six CPU threads after kernel warm-up. The 6,001-point grid
underestimates the strongest sampled peak height by 2.125% relative to this
finer grid, despite excellent numerical accuracy at shared q coordinates.
See [the measured report](../LARGE_PXRD_REPORT.md) for the distinction between
backend error and q-grid resolution.

Run the saved configuration through the standard interface:

```powershell
..\.venv\Scripts\python.exe scripts/run_scattering.py --config ../local_runs/my_100nm_pxrd/config.json --out ../local_runs/my_100nm_pxrd_runner
```

The configuration uses `backend: "radial_moments"` and optional `radial` settings
`bin_width_A: 0.04`, `order: 8`, `threads: 6`, `merge_decimals: 12`.
The library default thread count is one. `merge_decimals: null` disables
displacement merging. The radial backend requires a uniform q grid starting
at zero with at least two points. The supported range is checked against the
radial FFT grid. Version 0.4.1 also supports independent isotropic guest
populations through [PORE_DISTRIBUTION_GUIDE.md](PORE_DISTRIBUTION_GUIDE.md).
Atom-level surface cuts remain unsupported by this backend.

## Approximation and accuracy

The finite shape autocorrelation supplies integer cell-pair multiplicities.
Species-resolved displacement vectors are streamed into nearest-center radial
bins. Each bin retains Taylor moments through the requested order, so it does
not simply replace every distance by the bin center. FFTs evaluate the radial
sine sums on the requested q grid. Zero-distance terms and the forward pair
count are handled separately.

Optional displacement merging groups rounded keys but uses an actual input
vector as each representative. Its error contribution uses the measured
weighted displacement difference and the global sinc derivative bound q/2.
The Taylor truncation bound and this displacement bound are combined into an
absolute per-q intensity bound. **Floating-point accumulation and transform
roundoff are excluded from that analytical bound.** Independent references and
refinement checks test their combined numerical effect. Negative computed
values are retained rather than clipped.

The initial 0.1% acceptance target is assessed on q >= 0.1 inverse angstrom:

- Maximum absolute error divided by the largest non-forward reference intensity.
- Pointwise relative error wherever the reference is at least 1% of that peak.
- Integrated absolute error, total integrated intensity, and individual resolved
  peak areas. Peak positions are compared on the stated q grid.

The huge forward intensity is excluded from the normalization. Near intensity
zeros an absolute peak-scaled error is reported; a universal relative-error
claim at exact zeros would be meaningless. These are numerical accuracy checks
for the stated structural and scattering model, not experimental accuracy.

Add `--refine` to halve the radial bin width and increase the Taylor order.
Add `--exact-q 0.344 1.0 6.0` to compare selected nearest grid points to unbinned
Debye sums over the original, unmerged displacements. Exact checks can take
substantially longer than preparing the whole approximate pattern. Refinement
is labelled separately from exact-reference validation.

The cache can be retained for repeated evaluation:

```python
import numpy as np
from mofdebye.crystal import read_cif
from mofdebye.large_powder import LargePowderCache

crystal = read_cif('examples/real_mof/IRMOF-1.cif')
cache = LargePowderCache(crystal.basis, crystal.cell_vectors_A,
    np.ones((40,40,40), dtype=int), np.linspace(0,6,6001), threads=6)
intensity, absolute_approximation_bound = cache.iq()
print(cache.metadata)
```

Preparation and repeated transforms are timed separately. Compilation is warmed
on a tiny geometry and its cost is reported separately. Prior exact timings used
one CPU thread; compare thread counts and q grids before interpreting speedups.
