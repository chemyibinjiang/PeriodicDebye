# 04 — Prioritized implementation roadmap

These are recommended development tasks, not claims of work already completed. Implement small, tested increments. Keep the simple reference independent of accelerated adapters.

## R00 — Reproduce and preserve the baseline

Create an isolated environment, run the current tests and two example CLIs, record versions/hardware, and retain historical outputs. Establish a local branch or baseline copy before modifications. Use `tools/verify_manifest.py` before editing. Handoff baseline checks already passed, but the local environment must be checked too.

**Done when:** local tests/logs and example outputs are saved; source provenance remains intact.

## R01 — Establish an actual accelerated package adapter

Verify and install a suitable DebyeCalculator release/commit locally and test its public API. Resolve the optional version pin explicitly. Calibrate scale and factor conventions, detect direct fallback, and compare CPU direct/grid with the independent reference. Test CUDA only when available. Evaluate DEBUSSY/CLAUDE next for crystal-aware workflows; Debyer/AES-Debye are secondary candidates, not prerequisites for every branch.

**Done when:** at least one installed accelerated adapter is genuinely exercised with raw error/timing records, or an external installation failure is documented while the validated vendored fallback remains usable. GPU absence is reported, not disguised.

## R02 — Introduce the explicit three-mode contract

Separate `scattering_case`, `boundary`, `guest_estimator`, `backend`, and q versus q-vector input. Preserve the old finite-powder CLI as a compatibility path. Store a versioned configuration schema and metadata. Reject incompatible combinations at input validation.

Proposed conceptual configuration—not existing executable syntax:

```text
scattering_case: pxrd | single_crystal_finite | single_crystal_periodic
boundary: finite | bulk
host: explicit_xyz or crystallographic_structure + shape specification
pores: explicit centres + coordinate convention + optional geometry
population: occupancy + particle library + joint placement distribution
guest_estimator: explicit | analytic | moments
backend: reference | adapted_kernel | installed_package | lattice_grouped
sampling: scalar_q OR cartesian_q_vectors OR reciprocal_hkl
```

Backend applicability must be a capability table, not an assumption every engine supports every case.

**Done when:** tests reject directional requests to scalar Debye, label current results finite PXRD, and preserve original examples.

## R03 — Finite-single-crystal reference and full-cell acceleration

Implement tiled `A(q-vector)` and `|A|²`; add host/guest amplitude components and per-configuration output. Implement `F_cell*S_shape` for identical complete cells and verify it against explicit atoms. Provide one reciprocal-space slice and a line through a reflection with finite-size fringes. Label these reciprocal outputs, not simulated detector images.

**Done when:** rotation covariance, translation invariance of total intensity, cell-factorization, finite-width scaling and spherical-average equivalence to Debye pass.

## R04 — Shared independent-pore moment engine

Implement local mean amplitude and second moment at vector q, including whole-particle occupancy. Support the actual joint distribution, with analytic moments where justified and quadrature/Monte Carlo otherwise. Sum finite-host mean intensity using the moment identity, retaining all cross terms. Cache template/fixed-host work. Expose mean-amplitude integration uncertainty separately from physical diffuse variance.

**Done when:** exact discrete-state enumeration, p-not-p² checks, unrestricted local sampled distributions and angular averaging recover the existing powder reference. Sampling bias/convergence for reused cell moments is documented and tested.

## R05 — Periodic single-crystal Bragg and diffuse outputs

Add triclinic reciprocal-lattice enumeration, cell amplitudes, and independent-cell moment assembly. Return reflection weights and a separate continuous diffuse evaluator per cell. Support multiple pore sites in a cell with declared independence or explicitly enumerated within-cell joint states. Do not implement bulk by repeating one random occupied cell or wrapping finite Debye distances.

**Done when:** deterministic-cell and occupancy-disorder benchmarks pass, reflection normalization is documented, and growing finite crystals converge to the chosen integrated Bragg/diffuse convention. Correlated cells must be explicitly unsupported until a covariance model exists.

## R06 — PXRD over finite and bulk structures

Keep fast finite Debye as the default random-powder evaluator. Add a general angular-average reference for finite directional moments. Implement bulk reflection-shell grouping, correct reciprocal/powder weights and diffuse angular averaging. Preserve coherent host–guest effects. Finite-size and bulk outputs need distinct metadata.

**Done when:** all three user-facing cases have small runnable examples, and powder-versus-directional and finite-to-bulk crosschecks pass under a stated intensity measure.

## R07 — CIF, pore and surface input layer

Add an explicit CIF symmetry-expansion path and duplicate-site handling. State treatment of partial occupancy and crystallographic disorder; an average atom occupancy is not a whole-particle occupancy model. Allow user-supplied pore centres in fractional or Cartesian coordinates. Validate centres/geometry; do not equate unit-cell centres with real MOF pores by default.

Build finite shapes either from complete-cell masks or explicit atom cuts with a documented termination policy. Keep guest placement coherent near external surfaces and periodic faces; define whether guest retention uses centre containment or whole-particle containment. Check host exclusion against required periodic neighbours. Automatic pore finding can be an optional later module, not a hidden prerequisite.

**Done when:** at least one documented nonorthogonal crystallographic fixture expands consistently, and missing real-material inputs produce actionable errors rather than fabricated structures.

## R08 — Integrate periodicity reduction and ensemble caching

Connect the full-cell lattice prototype to the mode dispatcher. Cache shape counts, cell-pair distances/histograms, species factors and template statistics at appropriate boundaries. Avoid recomputing invariant host scattering every Monte Carlo draw. Add basis-specific occupancy masks or direct residual handling before claiming arbitrary atomically cut surfaces. Select sparse versus dense shape representations based on actual memory use.

**Done when:** an atom-count/basis-size/shape benchmark separates preprocessing from repeat evaluation and confirms accuracy and peak memory; no hidden pair-distance truncation is used.

## R09 — Broader pore models (second-stage extension)

Add pore-specific template libraries, anisotropic pore shapes and efficient constrained sampling as needed. Preserve number-weighted size semantics and report accepted joint distributions. Correlated occupancy, orientational domains, particles spanning multiple pores and host relaxation require separate physical models and tests; do not silently fold them into independent-pore formulas.

**Done when:** each enabled option has an explicit distribution, a sampler validation and a scattering consistency check. Unsupported options fail clearly.

## R10 — Experimental/profile layer (after raw physics)

Add wavelength/2theta conversion and explicitly normalized resolution convolution while retaining raw results. Keep finite-size interference separate from instrument width; do not count size broadening twice. Additional polarization/absorption/Lorentz/thermal/anomalous features need convention-specific tests. Detector Ewald-sphere projection is separate from a reciprocal slice.

**Done when:** unit-area tests and coordinate conventions pass, and required settings are explicit.

## R11 — Reproducible completion report and delivery

Provide source, versioned three-mode examples, environment records, test logs, raw CPU/GPU benchmarks actually performed, and a support matrix. Highlight boundaries of the physical model. Include an actual-material walkthrough only when a sourced structure and pore definition are available. No promise of unique inverse recovery from PXRD is warranted by forward-code tests alone.

**Minimum scientifically useful completion:** R00–R06 plus a documented explicit-XYZ/pore input path and reproducibility report. R07–R10 expand material realism and performance; unsupported extensions must remain visible rather than block validated core delivery.
