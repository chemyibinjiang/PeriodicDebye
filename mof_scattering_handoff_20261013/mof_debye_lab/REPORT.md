# First-run report: accelerated Debye kernels and pore-confined particles

**All results below were executed in the sandbox. 30 tests passed.**

## Scope

CPU-only microbenchmarks of an adapted DebyeCalculator forward-kernel extract, plus a new probability-distribution layer. This is not a complete package-installation benchmark. DEBUSSY, AES-Debye, the installed DebyeCalculator public API and GPU execution were not tested. The host geometry is synthetic, not a real MOF.

## 1. CPU forward-kernel benchmark

301 q points from 0.15 to 8.0 inverse angstrom; one CPU thread; medians of three repetitions; common neutral IT92 factors. Grid mode was forced so the small-system automatic direct fallback did not mask which algorithm was timed. No peak truncation was introduced.

| Atoms | Direct float64 (s) | Grid float64 (s) | Speedup | Grid high-q relative max error |
|---:|---:|---:|---:|---:|
| 192 | 0.0914 | 0.0133 | 6.9x | 6.91e-12 |
| 648 | 0.8074 | 0.0215 | 37.5x | 9.04e-12 |
| 1,536 | 4.3551 | 0.0525 | 83.0x | 1.18e-11 |

High-q error is max absolute difference divided by the maximum reference intensity **within q >= 3 inverse angstrom**. It is not a pointwise relative error at every minimum. The independent reference is in core.py. Float32 results, individual repeats and input settings are also saved in results/.

The grid kernel still enumerates atom pairs. This test demonstrates a reduction in repeated q-dependent work; it does not establish linear scaling with atom number or crystallographic symmetry acceleration. The full-cell lattice prototype is a separate algorithm.

![CPU benchmark](results/benchmark_runtime.png)

## 2. Independent-pore statistical average

The example has 24 host atoms, two pores, Au templates with [13, 19] atoms, number weights [0.65, 0.35], occupancy probabilities [0.6, 0.8], and a 0.45 A Gaussian standard deviation per coordinate. Particle orientations are Haar-uniform. The analytical test uses an unbounded Gaussian, not hard-wall rejection.

| Check | Executed result |
|---|---|
| Explicit finite configurations | 8,192 |
| Analytical evaluation | 0.0039 s |
| Reference Monte Carlo evaluation | 17.58 s |
| Relative L2 difference, MC mean vs analytical mean | 0.565% |
| Largest absolute residual / Monte Carlo SEM | 1.479 |
| q points within three Monte Carlo SEM | 100.0% |
| Full atom sum vs four-term decomposition, relative max error | 4.01e-15 |
| Accelerated grid vs reference for a composite snapshot | 2.87e-12 |

SEM here is uncertainty from a finite number of random configurations. It is not experimental uncertainty. The analytical/Monte Carlo timings compare different ways to average the same specified model, not different hardware or equivalent engine workloads.

![Probability average check](results/ensemble_validation.png)

Adding only isolated host and particle patterns gave a 6.97% relative L2 difference from the full average **in this synthetic example**. That is not a general error estimate for real MOFs. It shows that the implemented cross terms have an observable numerical effect.

## 3. Rigid-particle checks

| Operation | Change in isolated particle pattern | Change in host–particle term |
|---|---:|---:|
| translated | 2.79e-16 (relative max) | 7.128% (relative L2) |
| rotated | 1.74e-16 (relative max) | 1.577% (relative L2) |

## 4. Confinement-conditioned distribution

A separate model used nonzero centre offsets, anisotropic Gaussian displacements, orientational wobble, and a spherical hard wall. It deliberately used explicit sampling, not the incompatible unbounded-isotropic analytical formula.

In 2,000 configurations, 2,832 particles were placed. Requested size fractions were [0.50, 0.30, 0.20]; the sampled fractions were ['0.5025', '0.2934', '0.2041']. No atom exceeded the allowed 6.0 A radius (largest observed: 5.999745 A). The average placement needed 1.722 attempts. A 512-configuration reference scattering calculation was also executed.

An end-to-end CLI run separately passed 64 bounded random configurations through the accelerated grid kernel. The automated tests compare reference and accelerated ensemble means and SEMs on identical random draws.

![Conditional displacement distributions](results/bounded_displacement.png)

## 5. Finite crystal periodicity prototype

The irregular full-cell-mask test had 138 atoms and agreed with direct Debye evaluation to 7.92e-16 relative maximum error. A separate automated test used a nonorthogonal lattice. The correlation count at zero translation equals the cell count; the sum of counts equals the square of the cell count. A two-cell boundary test rules out accidental periodic wraparound. This is still a small prototype, not a mature arbitrary-surface engine.

## 6. Limitations before use on real MOF data

No real CIF, pore extraction, chemical structure validation, atomic-level surface termination, host relaxation, correlations between pores, multiple independent particles within one pore, or infinite-lattice Bragg/diffuse solver is included. Instrumental and anomalous-scattering corrections are also absent. GPU performance and full-package installation remain untested. The supplied numerical checks establish consistency for the implemented models, not identifiability of distribution parameters from experimental PXRD.

## Reproduction record

Settings and software versions: results/environment.json. Raw timings: results/benchmark.csv and benchmark_details.json. Validation statistics: results/validation_summary.json. Test log: logs/pytest.txt. Probability-convergence data: results/mc_convergence.csv. Individual accepted bounded placements: results/bounded_placements.csv. Input coordinates and JSON settings: examples/.
