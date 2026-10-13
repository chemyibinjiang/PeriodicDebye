# Three-mode forward simulator, version 0.3

The implemented core is kinematic elastic X-ray scattering from a fixed atomic
host and rigid nanoparticle guests. The core validation fixtures are synthetic;
[R07_GUIDE.md](R07_GUIDE.md) adds CIF input and a sourced IRMOF-1 example. Lengths
are angstrom, reciprocal coordinates are inverse angstrom, and lattice vectors
are rows. There is no material fit or detector simulation.

## Run

From `mof_debye_lab/` in this checkout, use the isolated interpreter explicitly:

```powershell
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/finite_single_crystal.json --out ../local_runs/examples/finite_single_crystal
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/periodic_single_crystal.json --out ../local_runs/examples/periodic_single_crystal
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/pxrd_finite.json --out ../local_runs/examples/pxrd_finite
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/pxrd_bulk.json --out ../local_runs/examples/pxrd_bulk
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/modes/finite_bounded_moments.json --out ../local_runs/examples/finite_bounded_moments
```

For a fresh environment, create a Python >=3.10 venv and install `.[test]`.
The optional verified installed backend uses `.[external]` (DebyeCalculator
1.0.14). After installing this project, `mofdebye --config ... --out ...` is the
same entry point. The development venv inherits existing dependency installations
read-only to avoid duplicating PyTorch; no system package was changed.

## Configuration contract

`schema_version` is `"1.0"` for the original XYZ contract or `"1.1"` for CIF and
surface extensions. [R07_GUIDE.md](R07_GUIDE.md) defines the additional fields.
`mofdebye.runner.validate_config` and its input loader enforce the contract and
reject unsupported fields/combinations.

| Field | Values and meaning |
|---|---|
| `scattering_case` | `single_crystal_finite`, `single_crystal_periodic`, `pxrd` |
| `boundary` | `finite` or `bulk`; must agree with single-crystal case |
| `guest_estimator` | `explicit` for one seeded realization; `analytic` for the restricted exact expectation; `moments` for local moment integration |
| `backend` | `reference`, `adapted_kernel`, `installed_package`, `lattice_grouped` |
| `device` | CPU for all reference/lattice work; accelerated adapters can explicitly request CUDA if available |
| `host.xyz` | Explicit finite atoms, or a single-cell basis if a cell mask/bulk boundary is supplied |
| `host.cell_vectors_A` | Nonsingular 3x3 matrix with direct lattice vectors as rows |
| `host.cell_mask` | Nonempty 3D binary mask of complete identical retained cells; finite only |
| `particles` | List of `{xyz, weight}`; number probabilities per occupied pore, normalized internally |
| `pores` | Exactly one of `centres_A` or `centres_fractional`; in the supplied host/cell frame |
| `population` | Existing `PorePopulation` parameters: occupancy, size library, displacement, orientation, confinement |
| `sampling` | PXRD: `q` list or `{min,max,points}`; finite directional: `q_vectors` or `hkl`; periodic: `reflection_q_max`, optional `reflection_q_min`, optional directional diffuse samples |
| `statistics` | `seed`, `n_samples` (>=2 per pore), `moment_method` (`analytic` or `sampled`, default sampled for `moments`) |
| `angular` | `n_mu`, `n_phi` for Gauss-Legendre x periodic azimuth powder integration |

Fractional row coordinates map through `x @ A`; reciprocal rows are
`B = 2*pi*inv(A).T`, and `q = hkl @ B`. Directional `hkl` coordinates may be
noninteger to inspect fringes and diffuse scattering. Reflection enumeration
itself uses integer hkl. Particle XYZ origins are their rotation centres;
templates are never silently recentered. With a finite mask, cell host atoms and
pore sites are replicated; each pore is independently populated. Particles remain
rigid and unwrapped across cell faces. Guests belonging to retained cell sites
are kept whole, even if they extend beyond the external host boundary; this is
a pore-site retention rule, not automatic whole-crystallite containment.

The analytic distribution is a common discrete template library, Haar-uniform
relative rotations and unbounded isotropic Gaussian centre jitter. Confinement,
anisotropic jitter or orientation bias use sampled moments or explicit draws.
Rejection keeps the already selected size fixed. Bulk hard-wall spheres are
checked against periodic images. Schema 1.1 supports periodic host-clearance
through an exact bounded lattice-image search. Correlated pore/cell configuration
fields fail validation. `exact_guest_moments` in the Python API accepts explicitly
enumerated joint within-cell guest states; `finite_cell_ensemble` can use those
moments with independent cells.

## Backend capabilities

| Backend | Finite PXRD | Finite directional | Bulk |
|---|---|---|---|
| Reference | Explicit, restricted analytic, or angular moment average | Explicit or moments | Bragg/diffuse moments and bulk powder |
| Adapted kernel | Explicit or analytical host term, common IT92 factors | Unsupported | Unsupported |
| Installed package | Explicit configurations only; native WK factors | Unsupported | Unsupported |
| Lattice grouped | Complete-cell host, optionally restricted analytic guest terms | Complete-cell deterministic host factorization | Unsupported |

The installed DebyeCalculator 1.0.14 public API is **batched direct**, with no
grid option. Its native intensity is multiplied by two to match conventional
full Debye normalization; this was checked from its independent tabulated
Waasmaier-Kirfel factors. These factors differ from Gemmi IT92, so that backend
cannot be silently mixed into analytical IT92 guest terms. The separate adapted
forward-kernel backend has direct/grid paths and reports the actual fallback.

`LatticePairCache` reuses linear FFT shape counts and budget-controlled pair
distances across q grids. Its metadata separates preprocessing and evaluation.
The reference now tiles pair distances and sinc workspace. Directional kernels
tile atoms and q vectors. The adapted grid histogram and actual installed
package can still have substantial allocations; they are not globally bounded
memory engines. Sparse masks retain a dense bounding-grid FFT cost.

## Output and normalization

- Finite `intensity.csv` contains unnormalized electron-squared intensity. A
  reciprocal slice/line describes reciprocal space, not an Ewald-sphere image.
- Periodic `reflections.csv` has integer hkl, Cartesian q, `f_squared`, and
  `bragg_weight = (2*pi)^3/V * f_squared`. Weights multiply three-dimensional
  reciprocal delta functions per cell; they are not finite peak heights.
- Bulk `diffuse.csv` is a continuous per-cell intensity, distinct from numerical
  sampling uncertainty. It can vary with direction and q.
- Bulk powder `powder_shells.csv` contains radial delta weights, summing every
  reciprocal vector once and applying `1/(4*pi*|G|^2)`. Multiplicity is already
  included. These are not experimental integrated line intensities. The forward
  G=0 weight stays separate in metadata; it is never divided by zero.
- `metadata.json` records units, actual method/device, package versions, input
  hashes, seed, physical assumptions and normalization. No instrument convolution
  or duplicate finite-size broadening is applied.

Moments separate physical disorder variance from numerical variance of the
estimated complex mean amplitude. Squared sampled means are bias-corrected.
Negative finite-sample coherent estimates near extinctions are retained rather
than clipped. Directional outputs report mean-amplitude RMS error; powder
outputs report its angular mean variance. Neither is an intensity SEM or
physical-parameter uncertainty. Increase sample count and independently refine
both angular orders to establish convergence; fixed example quadrature orders
are not general accuracy guarantees for larger objects/higher q.

## Compatibility and validation

The original `scripts/run_population.py --mode analytic|montecarlo` remains a
finite PXRD compatibility interface. Its mode is only an estimator choice.
It now records physical case/boundary and actual analytical host fallback and
rejects a false CUDA label on the CPU reference backend.

```powershell
..\.venv\Scripts\python.exe -m pytest -q
..\.venv\Scripts\python.exe scripts/validate_modes.py --out ../local_runs/validation_v02
..\.venv\Scripts\python.exe scripts/test_installed_debyecalculator.py --device cpu --out ../local_runs/installed/cpu_validation.json
```

Archived `results/`, `logs/`, and handoff `qa/` remain baseline evidence. Current
source intentionally differs from the original manifests; those manifests remain
historical records. New evidence is under `local_runs/` and a separate development
manifest records the delivered source. The original ZIP is the untouched baseline.

## Explicitly outside the delivered core

Automatic pore finding, chemical surface passivation/reconstruction,
host relaxation, correlations across cells, general interparticle exclusion,
multiple particles per pore, thermal/anomalous corrections, instrument profiles
and detector geometry remain unsupported. Real use needs a sourced CIF, explicit
atom set or cell basis, validated pore coordinates, particle templates, selected
probability laws and accuracy/memory targets. Synthetic fixtures suffice for
forward-model validation and must not be relabelled as material predictions.
