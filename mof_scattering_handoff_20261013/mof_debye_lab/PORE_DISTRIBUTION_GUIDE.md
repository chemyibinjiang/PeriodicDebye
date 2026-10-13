# Finite PXRD with independent pore distributions

The `radial_moments` powder backend can combine a deterministic, repeated host
basis with independent guests in every retained pore. It retains finite crystal
shape, host interference, intraparticle scattering, and scattering between
different guests. Pore occupancy, particle size, orientation and displacement
describe a probability distribution; a randomly filled unit cell is never
repeated as a single realization.

The current runnable example is a **synthetic hollow carbon grid**, with
illustrative Au13/Au19/Au43 templates. It is a numerical fixture, not a chemically
valid MOF, measured nanoparticle population, or validated pore-loading model.
The CIF host workflow remains available, but a real loading model needs valid
guest templates and pore sites supplied by the user.

## Run the small example

From this `mof_debye_lab` directory, using the handoff environment:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_pore_powder.py --prepare-only --cells 2 --threads 1 --out ../local_runs/pore_quick_example
..\.venv\Scripts\python.exe scripts/run_scattering.py --config ../local_runs/pore_quick_example/analytic.json --out ../local_runs/pore_quick_analytic
..\.venv\Scripts\python.exe scripts/run_scattering.py --config ../local_runs/pore_quick_example/active_wall.json --out ../local_runs/pore_quick_active_wall
```

Use a fresh preparation directory. The preparation command writes the host and
guest XYZ files plus two complete runner configurations. The first uses an
analytic distribution with certified nonbinding walls; the second increases
the displacement support so that spherical-wall rejection changes the joint
position/orientation distribution. The latter uses sampled local statistics.

For the 100 nm synthetic benchmark, use the same script with `--cells 40`
and omit `--prepare-only`. It uses 6,001 q points from 0 to 6 inverse angstrom,
reports compilation warmup separately, and separates geometry preparation,
local-statistics integration, and final intensity assembly. The optional fast
dependency is installed with `python -m pip install -e ".[fast]"`.

## Configuration

The host contains one cell's atomic basis. `host.cell_mask` specifies retained
complete cells. Pore centres are **one unit cell's sites**; the runner keeps them
implicit instead of expanding all host atoms or pore sites.

```json
{
  "schema_version": "1.1",
  "scattering_case": "pxrd",
  "boundary": "finite",
  "guest_estimator": "analytic",
  "backend": "radial_moments",
  "device": "cpu",
  "host": {
    "xyz": "host_cell.xyz",
    "cell_vectors_A": [[25, 0, 0], [0, 25, 0], [0, 0, 25]],
    "cell_mask": [[[1]]]
  },
  "pores": {"centres_fractional": [[0.5, 0.5, 0.5]]},
  "particles": [
    {"xyz": "Au_13.xyz", "weight": 0.5},
    {"xyz": "Au_19.xyz", "weight": 0.3},
    {"xyz": "Au_43.xyz", "weight": 0.2}
  ],
  "population": {
    "occupancy": 0.6,
    "position_mode": "uniform_ball",
    "uniform_ball_radius_A": 2.0,
    "orientation": "uniform",
    "pore_radius_A": 8.0,
    "wall_clearance_A": 0.5
  },
  "statistics": {"seed": 20261013, "n_samples": 4096, "n_batches": 8},
  "radial": {"bin_width_A": 0.04, "order": 8, "threads": 6},
  "sampling": {"q": {"min": 0, "max": 6, "points": 6001}}
}
```

For binding spherical walls, set `guest_estimator` to `"moments"` and
`statistics.moment_method` to `"sampled"`. Size probabilities are per occupied
particle; they are normalized once, then applied exactly. They are never
reweighted by different template acceptance rates. Whole-particle occupancy
multiplies its internal scattering by p, not p squared.

## Supported local distributions

All supported guests have Haar-uniform rigid orientation and isotropic position
proposals. The displacement and orientation become correlated when a wall
rejects placements; the sampled path preserves that conditioning.

| Local model | Analytic statistics | Sampled statistics |
|---|---|---|
| Isotropic Gaussian, no wall | Supported | Supported with zero mean shift |
| Uniform displacement in a solid ball, no wall | Supported | Supported with zero mean shift |
| Uniform ball with spherical wall | Only when the wall is certified nonbinding | Supported with zero mean shift |
| Zero-variance Gaussian with spherical wall | Only when every rotated template fits | Supported with zero mean shift |
| Nonzero-variance Gaussian with spherical wall | Rejected, even for a very large wall | Supported with zero mean shift |

For a uniform-ball proposal, the sufficient analytic certificate is

\[
|\bar u_k| + R_{\rm ball} + \max_{t:w_t>0,i}|x_{ti}|
\le R_{{\rm pore},k}-c_{\rm wall}.
\]

Gaussian tails have unbounded support when the variance is nonzero, so the same
certificate cannot justify ignoring a Gaussian hard wall. Analytic nonzero mean
shifts are incorporated into the scattering-site centres. The sampled path
requires zero mean shift so the conditioned local distribution stays isotropic.

Fixed or biased orientations, anisotropic Gaussian proposals, host-clearance
rejection, external-boundary clipping, arbitrary correlated samplers, and
atomically cut surfaces are not supported by this backend. Use the existing
directional/explicit methods for those models. Hard-wall pore spheres must be
disjoint, including neighbouring cell images. A non-overlap check between guest
atoms and the host is not performed by this fast model.

## Python API and cache reuse

```python
import numpy as np
from mofdebye.pore_powder import PorePowderCache
from mofdebye.radial_guests import radial_guest_statistics

q = np.linspace(0.0, 6.0, 6001)
cache = PorePowderCache(
    host_basis, cell_vectors_A, cell_mask, population, q,
    bin_width=0.04, order=8, threads=6,
)
statistics = radial_guest_statistics(population, q, method="analytic")
result = cache.iq(statistics)

# Change a supported local size/position distribution and reuse the geometry.
# Keep q, declared/effective centres, pore walls, and occupancies identical.
sampled = radial_guest_statistics(
    updated_population, q, method="sampled",
    n_samples=4096, n_batches=8, seed=20261013,
)
updated_result = cache.iq(sampled)
```

The cache can be reused when template shapes, size probabilities or supported
local displacement distributions change. Effective centres equal declared pore
centres plus the mean displacement. Keep the q grid, declared and effective pore
centres, occupancies, pore-wall radii, and wall clearance fixed during reuse.
Changing those settings requires a new cache and its physical geometry checks.
In particular, increasing a hard-wall radius must not bypass the validation
against neighbouring pore images. Statistics carry the corresponding geometry
description so mismatches fail explicitly. This reuse does not make a physically
incompatible updated template or pore distribution valid; local distribution
checks still apply.

The scalar q grid starts at zero and is uniform. The present FFT implementation
requires the maximum geometric radius to be less than pi/dq, and explicitly
rejects undersampled or impractically large transform grids. This numerical
guard does not include instrumental broadening or establish experimental
resolution.

## Quantities and uncertainty

`RadialGuestStatistics.means` has shape `(B, 2, K, Q)`: B independent batches,
two independent mean replicas, K pore types, and Q scalar q points. Means are
conditional on occupancy. `internal` has shape `(K, Q)` and is the exact
number-weighted template Debye intensity, invariant under rigid motion.
Analytic statistics use B=1 with identical replicas.

For sampling, `n_samples` is the **total accepted placements per template, per
pore type, per replica across all batches**. It must be divisible by
`n_batches`, with at least two batches and two accepted placements per batch.
Thus 4,096 samples and eight batches means 512 accepted placements per batch;
total accepted placements are `2 * n_samples * K * positive_weight_templates`.
Occupancy is applied analytically and each size is integrated separately.

Isotropy also allows the local integration to be performed in the particle's
body frame: with a rigid rotation R and displacement u, set v=R-transpose u.
Then atom radii and wall acceptance depend on `v + x_i`. Averaging the remaining
global rotation gives `sinc(q*|v+x_i|)`. This preserves the joint conditioned
distribution and avoids redundant rotation work without discarding host–guest
interference. It does not replace internal particle intensity by squared mean
amplitude.

The result contains `host`, `guest_internal`, `host_guest`, `guest_guest`, and
`total`, all in electron squared. Cross terms may be negative. Independent
replicas form unbiased products of estimated means; squaring one noisy reused
mean would introduce a bias. Same-physical-pore coherent self terms are removed
before adding the occupancy-weighted full internal term.

The runner writes three uncertainty columns in `numerical_error.csv`:

- `q`: the same reciprocal-space grid as the intensity table.
- `approximation_bound_excluding_roundoff`: propagated local and geometry radial
  interpolation/displacement-grouping bound. It excludes floating-point error,
  Monte Carlo error, and q-grid discretization error.
- `total_mc_sem`: estimated standard error of the mean intensity from independent
  batch patterns. This is numerical integration uncertainty, not physical
  occupancy variance, a strict error bound, or an experimental uncertainty.

The Python result additionally exposes component SEMs and approximation bounds.
An analytic model has zero Monte Carlo SEM because it uses no random integration;
that does not imply zero interpolation or floating-point error. Forward
conditional amplitudes are evaluated exactly from size weights and atomic
factors, so stochastic numerical noise is absent at q=0.

A 0.1% target needs separate checks of interpolation error and sampling
convergence in the relevant non-forward intensity region. Do not normalize all
errors by the forward beam. Refine radial settings against independent exact
references where feasible; increase accepted samples and compare independent
seeds when SEM is too large. Peak positions and areas also depend on q-grid
resolution. Monte Carlo SEM alone is not a universal 0.1% accuracy certificate.

The regression tests and independent validation script are
`tests/test_radial_guests.py`, `tests/test_pore_powder.py`,
`tests/test_pore_powder_physics.py`, `tests/test_pore_powder_runner.py`, and
`scripts/validate_pore_powder.py`.

The [measured report](../PORE_DISTRIBUTION_REPORT.md) records 980 passing tests,
the 100 nm synthetic runs, and independent sampler checks. Use `--samples` to
increase local integration work and `--seed` to make an independent comparison;
the final demonstrated budget is `--samples 1048576 --seed 20261014`.
