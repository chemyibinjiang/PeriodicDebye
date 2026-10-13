# 03 — Code status and handoff audit

All paths in this section are relative to `mof_debye_lab/`. This is an inspection of the supplied source and a fresh baseline test, not a completed refactor.

## Existing implementation

| Location | Implemented responsibilities |
|---|---|
| `src/mofdebye/core.py` | `AtomCloud`, XYZ I/O, common Gemmi neutral IT92 form factors, `direct_iq`, `cross_iq`, `pair_decomposition`, adapted-kernel wrapper `upstream_kernel_iq` |
| `src/mofdebye/distributions.py` | `ParticleLibrary`, lognormal catalogue weights, `PorePopulation`, rigid placement, occupancy/size sampling, Gaussian/uniform-ball positions, uniform/fixed/wobble rotations, optional rejection |
| `src/mofdebye/ensemble.py` | Restricted `analytic_ensemble`, component-resolved `sampled_ensemble`, total-only streaming `accelerated_sampled_ensemble` |
| `src/mofdebye/lattice.py` | Linear FFT `shape_autocorrelation`, full-cell `lattice_grouped_iq`; supports nonorthogonal cells |
| `src/mofdebye/fixtures.py` | Synthetic host, basis and finite fcc particle builders; not chemical MOF validation |
| `src/mofdebye/vendor/` | Adapted upstream forward kernels and their notice/license; not the installed package |
| `scripts/run_population.py` | Existing finite-powder JSON/XYZ CLI |
| `scripts/run_benchmarks.py`, `scripts/run_validation.py` | Reproducible historical benchmark/validation drivers; output destinations are configurable |
| `scripts/test_installed_debyecalculator.py` | Optional public-API test scaffold; **never runtime-validated in the baseline or this handoff** |
| `tests/` | 30 parameterized cases covering numerics, decomposition, statistics, rotations, confinement and lattice masks |

## What the archive does not implement

There is no directional complex-amplitude engine, reciprocal-lattice reflection enumerator, infinite-lattice Bragg/diffuse solver, bulk powder projection, complete three-mode dispatcher, automatic CIF symmetry expansion, pore finder, real-MOF surface model, host relaxation, correlated-pore sampler or instrument model. Do not present existing functions as these missing capabilities.

## Important implementation hazards to address

### 1. CLI mode and output metadata

`--mode analytic|montecarlo` is an estimator choice; all existing paths calculate finite powder scattering. Add explicit physics-case and boundary fields while maintaining a documented backward-compatible mapping. Analytical CLI metadata currently does not record the actual host-kernel fallback, whereas the accelerated Monte Carlo path counts grid/direct fallbacks. Record actual—not merely requested—method/device in all new outputs.

### 2. Optional external package dependency is unverified

`pyproject.toml` has an `external` extra pinned to `debyecalculator==1.1.1`. Do not assume this release or its exact API supports all scaffold parameters. Inspect local installation metadata and signatures. Pin a verified release or repository commit with a recorded reason; do not silently substitute whatever source happens to install.

The optional API check calibrates its independent sum from one-atom native outputs. This can check consistency in the package's native scale, but can conceal an overall normalization factor. Add an explicit scale check against known native form factors and one-/two-atom conventional intensities. Its constructor request for grid mode is also not proof the grid kernel ran; upstream can fall back to direct for small systems. Detect the actual path.

### 3. Benchmark scope and computational scaling

The grid kernel still visits all atomic pairs. Its benefit comes from reducing repeated q-dependent sinc work, not automatic crystallographic reduction. The source-kernel wrapper receives common Gemmi factors; the real package may use different factor tables and output scale. Separate engine arithmetic error from factor-model differences.

GPU timing must synchronize before/after measurement and separate setup/transfers from warmed computation. Record precision and float64 performance rather than assuming a consumer GPU is uniformly faster. No GPU result exists in the archive.

### 4. Memory and caching

`weighted_sinc_sum` chunks its q-by-distance workspace, but reference `pdist`/`cdist` still allocate whole pair-distance arrays. Do not call this a fully bounded-memory large-N engine.

`sampled_ensemble` retains an `n_samples x 5 x n_q` array even when sample retention is not requested; it does cache host and isolated-particle patterns. `accelerated_sampled_ensemble` uses streaming statistics, but concatenates the whole system and recomputes all host pairs for every draw. A useful optimization is accelerated component/cross-histogram evaluation with fixed terms cached, followed by local amplitude moments for independent pores.

`lattice_grouped_iq` is a separate exact full-cell prototype, not integrated into the CLI. It loops over basis pairs and all retained translations. Reuse pair databases across q grids/parameter sweeps where appropriate; benchmark construction separately from later evaluation.

### 5. Distribution semantics

The current library and covariance are common to all pores; occupancies and mean offsets can vary. Pore-specific libraries/covariances would be a new feature. The particle origin is caller-defined and not silently recentered.

Spherical confinement checks atomic centres plus a scalar clearance, not element-specific atomic radii or a chemically validated pore surface. Overlapping hard-wall pore spheres are rejected. Without those walls, the model does not impose general particle–particle exclusion. Host-clearance KD-tree checks use only the explicit finite host; a periodic implementation needs correctly expanded host neighbours.

Rejection preserves selected size; if no placement succeeds it raises, rather than silently changing occupancy/size weights. This is a conditional geometric model and must be documented as such. Random orientations use Haar-uniform SO(3); the wobble option is a Gaussian rotation-vector perturbation, not a matrix-Fisher law.

### 6. Results are raw elastic finite-system intensities

No anomalous factors, thermal displacement, absorption, polarization/Lorentz, background or resolution convolution is present. All executed summation comparisons use the same neutral IT92 factors. Numerical errors near 1e-11 are algorithm-consistency measures, not material prediction accuracy.

The only structure fixtures are synthetic. The stored 8,192-sample experiment supports the restricted finite analytical formula; it does not establish bulk convergence, physical parameter identifiability, or universal importance of cross terms.

## Fresh handoff check

The unchanged 30-case suite passed, and both a restricted analytical CLI run and 64-sample bounded forced-grid run executed. New outputs are under `../qa/`; original `results/` and `logs/` are untouched. Original archive/member hashes are recorded in `../HANDOFF_METADATA.json` and `../qa/original_archive_inventory.json`.
