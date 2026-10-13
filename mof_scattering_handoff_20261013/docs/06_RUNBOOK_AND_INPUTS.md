# 06 — Local runbook and material inputs

## Verify before editing

From the handoff root:

```bash
python tools/verify_manifest.py
```

This uses only Python's standard library. It checks all listed files; hashes will intentionally change after development. Keep an untouched copy or commit before editing.

## Linux/macOS isolated baseline

From `mof_debye_lab/`:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[test]'
python -m pytest -q
```

Use a PyTorch build appropriate to the machine. Do not assume the recorded CPU build provides CUDA. The recorded `requirements-tested.txt` is an environment snapshot, not a universal wheel-availability guarantee.

## Windows PowerShell isolated baseline

From `mof_debye_lab/`, activation is optional when using the environment interpreter explicitly:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e ".[test]"
.\.venv\Scripts\python.exe -m pytest -q
```

Substitute that interpreter for `python` in the commands below. Record CPU/CUDA availability in the actual process used.

## Existing, runnable finite-powder examples

```bash
python scripts/run_population.py --config examples/analytic_validation.json --mode analytic --backend upstream-kernel --out ../local_runs/analytic
python scripts/run_population.py --config examples/bounded_demo.json --mode montecarlo --backend upstream-kernel --samples 512 --force-grid --out ../local_runs/bounded
```

The analytical case has no Monte Carlo integration SEM; the zero field does not mean zero uncertainty in material parameters. The forced-grid option prevents a small-system direct fallback from masking which algorithm is being exercised. CPU is the current validated device.

The following runs write to new destinations, avoiding the archived results and example files:

```bash
python scripts/run_benchmarks.py --sizes 2 3 4 --repeats 3 --out ../local_runs/benchmark
python scripts/run_validation.py --samples 8192 --out ../local_runs/validation --examples ../local_runs/validation_examples
```

For comparable CPU timings set `OPENBLAS_NUM_THREADS=1`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` before Python starts, and record Torch thread settings. In PowerShell use `$env:OMP_NUM_THREADS="1"` and analogous assignments.

The optional real-package check is in `scripts/test_installed_debyecalculator.py`. First audit the unverified `external` extra/version and script API assumptions described in document 03. After resolving the installation, run that script with `--device cpu` and a new `--out` path. Do not count vendored tests as successful public-API tests.

The **proposed** three-case dispatcher is not implemented in v0.1. No `single_crystal_*` CLI command is claimed to be runnable yet.

## Current JSON input contract

The runnable example JSON files contain `host_xyz`, `particles` (each with `xyz` and `weight`), `pore_centres_A`, `population`, and `q` with `min`, `max`, `points`. Paths are resolved relative to the JSON file. Coordinates are Cartesian angstrom. Templates use their existing origin as the rotation/placement centre.

`population` is passed to `PorePopulation`. Review its signature rather than inventing keys. It currently supports independent occupancy, a common template library, isotropic sigma or a common 3x3 covariance, per-pore mean offsets, Gaussian/uniform-ball position modes, uniform/fixed/wobble orientation, optional spherical pore radius/clearance, and a scalar minimum distance to explicit host atoms. Analytical mode enforces its narrower assumptions.

## Inputs needed for a real MOF application

| Input | What must be stated |
|---|---|
| Host CIF or explicit finite XYZ | Source, symmetry/cell convention, solvent/disorder/occupancy treatment; XYZ must already be the intended finite atom set |
| Pore centres and geometry | Cartesian/fractional frame, site labels, symmetry-equivalent sites and clearance definition; not automatically unit-cell centres |
| Finite host boundary | Full-cell mask versus atom-level cut; shape axes, size and surface/termination policy |
| Guest templates | Elements and atomic coordinates for each size/shape, lattice/model provenance and origin |
| Probability law | Occupied-pore number fractions, centre offsets/covariance, relative orientations, confinement and any dependencies |
| Measurement | Which of the three cases, scalar/vector q range, finite/bulk choice, and instrument settings only when required |
| Validation/performance target | Available CPU/GPU, approximate atom/cell counts, accuracy and memory constraints |

None of these material-specific quantities should be inferred from the illustrative synthetic fixtures. Continue generic implementation without waiting for real data. When real data arrives, validate it and create a versioned example with provenance.

## Expected output from the local agent

Return source and tests, three runnable mode examples, environment records, a benchmark/error table backed by raw files, and a support matrix. Explain what remains synthetic, unimplemented or untested. Keep per-configuration intensity distinct from ensemble mean, raw intensity distinct from instrument profile, and statistical sampling SEM distinct from physical uncertainty.
