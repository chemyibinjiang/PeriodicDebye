# Reproduce the MOF performance study

The measured results, accuracy comparisons and remaining bottlenecks are in
[PERFORMANCE_REPORT.md](../PERFORMANCE_REPORT.md). The source is the bundled,
checksum-verified MOF-5/IRMOF-1 model: 424 atoms in a 25.832 A conventional cell.
Complete-cell cubes are geometric crystallites, with no reconstructed surface.
The guest microbenchmarks use explicitly synthetic Au13 sites; they do not
establish physical MOF loading or pore locations.

## Run the workloads

From `mof_debye_lab/`, use a fresh output directory for each invocation:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_mof_modes.py --out ../local_runs/my_benchmark --timeout 300 --profile
```

The driver isolates each case in a subprocess, fixes the CPU thread count to
one by default, repeats each calculation three times, and saves raw output
arrays, configuration, source hashes, environment, timings and memory records.
`--profile` adds a separate cProfile run, excluded from timing samples. The
worker timeout includes imports, all repeats and the optional profile; it is
not a single-calculation time limit. Partial timing records are retained on
timeout, but are not silently promoted to completed benchmarks.

Select a smaller subset or a denser q grid:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_mof_modes.py --out ../local_runs/my_finite --select finite_n4 finite_n10 finite_n20 finite_n40
..\.venv\Scripts\python.exe scripts/benchmark_mof_modes.py --out ../local_runs/my_dense_powder --select powder_n4 --points 1201 --repeats 1 --timeout 300
```

Finite directional cases use 2,049 q vectors on a generic off-axis line through
6 inverse angstrom. Cube edges run from 10.3 to 103.3 nm, representing 27,136
to 27,136,000 atoms without materializing every atom. Finite powder defaults
to 121 scalar q points over 0--6 inverse angstrom for 1, 2, 4 and 8 cells per
edge (2.6--20.7 nm). That coarse grid is a timing workload, not a guarantee that
every narrow peak is resolved. Increase points for resolved curves; the exact
Debye value at each requested q does not depend on neighboring grid points.

Periodic and bulk powder cases cover reciprocal cutoffs of 3, 6 and 10 inverse
angstrom. An ideal infinite crystal has no finite atom count or edge length:
work scales with cell basis and reflection count. Bulk powder shell weights
and finite powder intensities have different measures and are never substituted
for one another in the benchmark.

To benchmark the preserved v0.3.1 source, add:

```powershell
--project ../local_runs/performance_v032/baseline/source/mof_debye_lab
```

Both source versions run with the same Python and dependencies. Warm wall time
includes CIF loading, calculation, and metadata construction, and excludes
interpreter startup, file output and profiling. The guest microbenchmarks are
the exception: they time `guest_moments` only after setup, excluding CIF loading,
population construction and metadata. RSS is sampled every 5 ms; the
Windows peak working set is also recorded. Absolute memory includes Python,
NumPy, SciPy and PyTorch imports. Repeated-call peaks can include simultaneous
old and new result arrays. Neither measurement is a hard memory guarantee.

## Accuracy-preserving changes

- Rectangular full-cell single crystals use three stable geometric sums. The
  arbitrary-mask implementation remains available for irregular shapes.
- Reflection enumeration uses bounded candidate batches. Shell aggregation
  preserves its original first-member tolerance rule and all multiplicities.
- Finite powder groups exactly equal displacements and radii with integer pair
  counts. Inversion symmetry applies to every finite mask. The stronger cubic
  symmetry is used only for an exactly diagonal cubic cell and full cubic mask.
  There is no distance binning, coordinate rounding, or interpolation.
- Sampled guest amplitudes reuse form factors and element masks while retaining
  the sampler, seed, rejection rules and statistical accumulation order.
- Bulk analytic diffuse scattering uses its exact radial variance. Finite
  interference and anisotropic sampled diffuse still require directional work.

The original reference Debye and explicit directional implementations remain
independent numerical checks. Tests include near-Bragg points, extinction zeros,
skew cells, incomplete shapes, mixed species, conditioned sampling and exact
physical solutions. Tiny changes from reordered float64 sums are measured in
the benchmark comparison; no accuracy threshold is relaxed for timing.

## Reuse geometry for repeated evaluations

The configuration runner intentionally loads a fresh input and geometry each
time. A repeated fit can retain the imported cell and, for finite powder, a
`LatticePairCache`:

```python
import numpy as np
from mofdebye.crystal import read_cif
from mofdebye.lattice import LatticePairCache

crystal = read_cif('examples/real_mof/IRMOF-1.cif')
cache = LatticePairCache(crystal.basis, crystal.cell_vectors_A,
                         np.ones((4, 4, 4), dtype=int))
first = cache.iq(np.linspace(0, 6, 121))
refined = cache.iq(np.linspace(0, 6, 1201))
print(cache.metadata)
```

Run the separate reuse and dense-grid experiment with:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_mof_reuse.py --out ../local_runs/my_reuse --cells 4 --points 121 1201
..\.venv\Scripts\python.exe scripts/benchmark_mof_reuse.py --out ../local_runs/my_larger_cache --cells 8 --cache-mib 256 --points 121 601 --repeats 1
```

`--cache-mib` changes the retained-data budget in the Python cache API; the
configuration runner retains its 64 MiB default. The final library can be
smaller than intermediate candidate blocks, so inspect cache metadata rather
than assuming a given mask is cached. `--repeats 1` is a single measurement.

Preprocessing and evaluation times are recorded separately in cache metadata.
Its byte budget applies to retained radial data; basis-displacement grouping,
shape FFTs and transient sorting workspace also need memory. Arbitrary atom cuts
break complete-cell factorization and still use explicit-coordinate backends.
GPU performance and instrument broadening are outside this CPU study.
The subsequent approximate finite powder implementation is documented separately
in [LARGE_PXRD_GUIDE.md](LARGE_PXRD_GUIDE.md); the exact v0.3.2 results above remain
the preserved baseline.
