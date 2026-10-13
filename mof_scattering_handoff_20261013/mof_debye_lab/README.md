# MOF scattering, version 0.4.1

New to scattering? Start with the [step-by-step Debye tutorial](../../README.md):
waves, two atoms, all three modes, and nanoparticles in pores, with explicit equations.

Finite single-crystal, periodic single-crystal Bragg/diffuse, and finite/bulk
PXRD are implemented. Start with [MODE_GUIDE.md](MODE_GUIDE.md) for current
commands, configuration, output conventions and supported combinations.
See [R07_GUIDE.md](R07_GUIDE.md) for CIF symmetry expansion, pore coordinates,
finite cuts and guest containment, including the runnable sourced IRMOF-1 example.
Development results and reports are in `../local_runs/`, `../R07_REPORT.md`
and the preserved v0.2 `../COMPLETION_REPORT.md`.

For correctness checks against known equations, run the
[eight physical examples](examples/physical/README.md). The expanded independent
test evidence is described in `../PHYSICAL_VALIDATION_REPORT.md`.

For measured MOF-size scaling and exact optimizations, see
[PERFORMANCE_GUIDE.md](PERFORMANCE_GUIDE.md) and
[the performance report](../PERFORMANCE_REPORT.md).

For the optional finite PXRD approximation targeting 100 nm crystals and a
0.1% numerical intensity tolerance, see [LARGE_PXRD_GUIDE.md](LARGE_PXRD_GUIDE.md).
It retains finite Debye interference and reports its approximation bounds.
For scalable independent nanoparticle populations in spherical pores, see
[PORE_DISTRIBUTION_GUIDE.md](PORE_DISTRIBUTION_GUIDE.md). The synthetic examples
include occupancy, Au size mixtures, Haar rotations and confined positions,
with separate radial approximation bounds and sampling uncertainties.

The following text is the preserved **historical v0.1 report**. Its old status
statements and installation instructions describe the original archived run.

# MOF–nanoparticle Debye lab — first working prototype

**Status: 30 automated tests passed. All included results were computed on CPU.**

This project tests accelerated forward Debye arithmetic and models a probability
distribution of rigid nanoparticles around known pore centres. It keeps host–host,
within-particle, host–particle and between-particle atom pairs.

## What actually ran, and what did not

| Component | Status |
|---|---|
| Independent NumPy/SciPy full Debye reference | Executed and tested |
| DebyeCalculator direct/grid **forward-kernel extract** | Adapted from inspected upstream source; executed and benchmarked |
| Full installed DebyeCalculator package / public API | **Not executed**: pip and binary downloads failed in this sandbox |
| Full-package test script | Included, syntactically checked, **not runtime-validated here** |
| DEBUSSY / AES-Debye | Not benchmarked in this round |
| CUDA / GPU | Not tested: no CUDA hardware in this sandbox |
| Pore-occupancy, size, displacement and rotation sampling | Implemented and tested |
| Analytical independent-pore average | Tested against 8,192 explicitly sampled finite configurations |
| Sampled configurations evaluated with the accelerated kernel | Executed; checked against the reference using identical random draws |
| Finite-shape FFT autocorrelation / lattice grouping | Separate new prototype, tested; not attributed to DebyeCalculator |

The upstream computational source was retrieved through GitHub, rather than pip.
The adapted file is `src/mofdebye/vendor/debyecalculator_kernel.py`; its notice,
license, upstream file hash and modifications are recorded alongside it. **These
are kernel-level tests, not evidence of a complete installation or a benchmark
of the unmodified package.** The published package uses its own form-factor tables
and public-API conventions, which are not covered by the extracted-kernel tests.

No local development workspace was modified: the connection to that workspace
failed. This archive was created in a separate sandbox.

## Model used in the demonstrations

The host is a **synthetic C/O/Zn point-cage lattice**, not a chemically validated
MOF. No experimental CIF was supplied. Au particles are illustrative fixed-lattice
fcc cuts with 13, 19 or 43 atoms. The lattice parameter, pore sizes and probability
parameters are test inputs, not fitted or inferred material properties.

Positions are in angstrom; q is in inverse angstrom. The convention is

`I(q) = sum_i f_i(q)^2 + 2 sum_(i<j) f_i(q) f_j(q) sinc(q r_ij)`.

All executed engines receive the **same neutral-atom IT92 form factors from
Gemmi**. Thus numerical differences assess the summation, not differences between
scattering-factor parametrizations. The functions return unnormalised finite-
system intensity, without anomalous scattering, temperature factors, absorption,
polarization/Lorentz factors, instrument broadening, or a background model.
Numerical agreement at 1e-11 does not imply that level of physical accuracy.

## Quick start

Run these commands from the extracted `mof_debye_lab` folder, in a Python environment
with PyTorch installed for the intended CPU/GPU device:

```bash
python -m pip install -e ".[test]"
python -m pytest -q

# Exact statistical average under the restricted analytical assumptions.
python scripts/run_population.py --config examples/analytic_validation.json --mode analytic --out my_analytic_run

# Bounded positions and preferred orientations: sample configurations explicitly.
python scripts/run_population.py --config examples/bounded_demo.json --mode montecarlo --backend upstream-kernel --samples 512 --out my_bounded_run
```

The CSV includes q, intensity and the Monte Carlo standard error where applicable.
A zero analytical SEM means no Monte Carlo integration was performed, **not** zero
uncertainty about the material. By default the kernel follows the upstream
small-system direct fallback. `--force-grid` disables this fallback for testing.
`--device cuda` is accepted, but intentionally raises an error when CUDA is absent;
it never silently labels a CPU run as GPU work.

To repeat the benchmark and validation:

```bash
python scripts/run_benchmarks.py --sizes 2 3 4 --repeats 3
python scripts/run_validation.py --samples 8192
```

For strict timing comparisons, use one CPU intra-op thread and set
`OPENBLAS_NUM_THREADS=1` and `OMP_NUM_THREADS=1` before starting Python, as in the
recorded runs. Exact versions and CPU details are in `results/environment.json`.

## The particle probability model

For each pore k, the code first samples whether it is occupied, then selects a
particle template by number-weighted probability. It then draws its displacement
and rotation, and places every atom rigidly:

`r_(k,a) = pore_centre_k + displacement_k + rotation_k @ template_atom_a`.

| Variable | Implemented choices |
|---|---|
| Occupancy | Independent Bernoulli probability, individually specified per pore |
| Size / atomic shape | A discrete library of arbitrary atomic XYZ templates; helper for lognormal radius-bin weights |
| Centre position | Gaussian with a 3D covariance and optional pore-specific offset, or uniform volume in a ball |
| Relative orientation | Haar-uniform SO(3), fixed orientation, or Gaussian rotation-vector wobble around a reference |
| Confinement | Optional spherical hard wall and minimum distance to explicitly supplied host atoms |

The size weights are probabilities **per occupied pore / per particle**, not mass
fractions or intensity fractions. Different-sized particles keep their correct
atom and atom-pair counts. The code does not imitate size changes by stretching a
fixed lattice.

Confinement rejection occurs **after occupancy and size have been selected**.
It resamples position and orientation at the same size until they fit. This
preserves the requested size probabilities while introducing physically necessary
size–position–orientation coupling through geometry. Impossible placements raise
an error instead of silently removing large particles. This is a specified
conditional geometric distribution, not a thermodynamic equilibrium model.

The hard-wall version requires disjoint pore spheres. Inter-pore exclusion,
correlated occupancy and cooperative alignment are not included. Uniform Euler
angles are not used: uniform rotations come from SciPy's SO(3) rotation sampler.
The wobble distribution is explicitly Gaussian in the rotation vector; it is not
claimed to be a matrix-Fisher distribution.

## Statistical acceleration within the Debye framework

For independent pores, uniformly random rigid-particle orientations and
independent **unbounded isotropic Gaussian** centre displacements, orientation
and position averages can be integrated analytically. The particle's own Debye
pattern is unchanged by rigid rotation/translation. Host–particle and interparticle
contributions are averaged using the sinc distance identity.

Let `P(q)` be the number-weighted mean of the isolated template intensities,
`B(q) = average_template sum_a f_a(q) sinc(q |x_a|)`, and
`D(q) = exp(-q^2 sigma^2/2)`. With mean centres `c_k` (including specified offsets):

```
Host:          I_host(q)
Within guests: sum_k p_k * P(q)
Host–guest:    2 B(q) D(q) sum_k p_k sum_i f_i(q) sinc(q |r_i-c_k|)
Guest–guest:   2 B(q)^2 D(q)^2 sum_(k<l) p_k p_l sinc(q |c_k-c_l|)
```

All four terms are retained. The internal term is weighted by **p**, not p^2:
once a pore is occupied, its particle's atoms coexist. Treating every guest atom
as an independent fractional occupancy would lose this correlation.

The analytical function refuses hard-wall-conditioned positions, anisotropic
Gaussian displacements, and nonuniform relative orientations. Those models use
the explicit sampler. In particular, clipping a Gaussian at a pore wall and then
using the unbounded Gaussian formula would be inconsistent.

This version computes **finite hosts and finite sets of pores**. It does not yet
implement the ideal infinite-crystal Bragg/diffuse distributional limit. Nor does
it assert that a PXRD pattern uniquely identifies all the distribution parameters.

## Main files

| File | Purpose |
|---|---|
| `src/mofdebye/core.py` | Atomic coordinates, common factors, direct reference, interference terms, upstream-kernel wrapper |
| `src/mofdebye/distributions.py` | Template probabilities, pore placement, confinement and orientation sampling |
| `src/mofdebye/ensemble.py` | Analytical mean, reference Monte Carlo, accelerated-kernel Monte Carlo |
| `src/mofdebye/lattice.py` | New full-cell finite-shape autocorrelation and lattice-grouped Debye prototype |
| `scripts/run_population.py` | End-to-end CLI using JSON plus explicit XYZ files |
| `scripts/test_installed_debyecalculator.py` | Optional actual-public-API check; not executed here |
| `tests/` | The 30 numerical/physical-consistency checks |
| `REPORT.md` | Executed results, errors, timings and interpretation |
| `results/` | Raw CSVs, JSON metadata, convergence data, placements and plots |
| `examples/` | Example input settings, host/particle XYZ files and sampled snapshots |

The lattice-grouping prototype keeps identical **complete unit cells** under a
binary finite-shape mask, including nonorthogonal cells. Its FFT is a linear,
zero-padded correlation, not a periodic wrap. Atom-level surface cutting, missing
ligands and relaxation require extensions; they are not silently approximated by
that module.

## Using real coordinates

Replace the example host XYZ with an explicit finite MOF atom set. Supply your own
particle XYZ templates and pore-centre coordinates in the same Cartesian frame.
Each template's origin is its intended rotation/placement centre; the program
does not silently recenter it. Then change the configuration JSON probabilities
and geometric constraints. Do not use a fractional-occupancy average guest as a
substitute for the sampled whole-particle occupancy model.

This prototype does not find pores or expand a crystallographic CIF automatically.
That preprocessing step must establish the intended finite boundary, symmetry
expansion, disorder treatment and pore centres first.

## Actual installed-package check on a connected machine

```bash
python -m pip install -e ".[external]"
python scripts/test_installed_debyecalculator.py --device cpu
```

This script imports the real `debyecalculator` package, not the vendored extract.
It compares direct and grid public-API results and an independent pair sum calibrated
with the package's one-atom outputs. It is included as the next installation test,
but has **not** been executed in this sandbox. Its timings are not included in the
reported source-kernel benchmark.

## References and provenance

DebyeCalculator source and documentation:
https://github.com/FrederikLizakJohansen/DebyeCalculator

Inspected computational file Git blob SHA-1:
`89cd257ac681d3859457db7e8220cbc325c6953d`

Johansen et al., JOSS 9 (2024), 6024:
https://joss.theoj.org/papers/10.21105/joss.06024

Gemmi scattering-factor documentation:
https://gemmi.readthedocs.io/en/latest/scattering.html

SciPy rotation sampling documentation:
https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html

The external kernel is Apache-2.0 licensed. See the included license and vendor
notice for attribution and explicit modifications. No third-party wheels or font
files are bundled.
