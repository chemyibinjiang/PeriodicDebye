# Prompt for the local coding agent

Continue this MOF–nanoparticle scattering project from the attached working v0.1 code. Implement and test the missing capabilities; do not stop at another design document. Work only in this extracted project unless I explicitly authorize another workspace.

Read `AGENTS.md`, `docs/01_DISCUSSION_AND_DECISIONS.md`, `docs/02_PHYSICS_AND_MODE_SPEC.md`, and `docs/03_CODE_STATUS_AND_AUDIT.md` first. Then use `TASKS.json`, the roadmap, and validation plan as your execution guide.

## Scientific objective

We want accelerated scattering for arbitrarily shaped finite MOF nanocrystals, plus nanoparticles distributed around known MOF pore centres. Guest occupancy, size/atomic structure, displacement and orientation must be configurable. Retain host–guest and guest–guest interference, not just an incoherent sum of two powder spectra.

**Keep three user-facing cases separate:**

1. Periodic/bulk single crystal: direction-resolved reciprocal-space Bragg weights and disorder diffuse intensity.
2. PXRD: whole-crystallite orientational averaging; support finite crystallites first and a separately documented bulk limit.
3. Finite single crystal: direction-resolved complex-amplitude scattering, retaining the finite shape and size.

Guest orientation relative to the host, whole-crystallite powder averaging, finite/bulk boundary conditions, and statistical averaging are separate operations. The existing `--mode analytic|montecarlo` is only an estimator choice and must not be mistaken for a scattering mode.

## Starting point and order

First reproduce the 30 current tests and existing example CLIs in an isolated local environment. Preserve their results and the vendor attribution. The tested accelerated code is an **adapted DebyeCalculator kernel extract**, not a complete package installation. Verify a real installed accelerated package through its actual public API, recording version/source, normalization, form factors, direct fallback and available hardware. Resolve the unverified optional dependency pin/API contract instead of assuming it works.

Next implement a tiled finite-single-crystal amplitude reference and validate its angular average against the current Debye reference. Add the shared guest-amplitude-moment layer, then periodic single-crystal Bragg/diffuse calculations and bulk powder projection with explicit normalization. Integrate the already tested finite-cell shape grouping into an acceleration path with caching. Improve stochastic caching without losing interference.

Use the detailed roadmap for CIF expansion, pore inputs, surface boundaries, adapters, performance and instrument-output work. Continue on synthetic fixtures when real material inputs are absent; label them synthetic and do not invent a MOF, pore geometry, equilibrium distribution or experimental fit. Multiple particles within a pore, correlated pores, host relaxation and full detector simulation are extensions, not prerequisites for validating the independent-pore core.

## Required result

Deliver runnable code, versioned configuration examples for the three cases, tests with actual logs, raw benchmark/error data, and a concise completion report identifying implemented, tested, unsupported and untested capabilities. Record actual CPU/GPU execution; a missing GPU is an untested backend, not a successful GPU benchmark. Unsupported model combinations must raise clear errors rather than silently becoming a different physical problem.

Use an explicitly chosen small synthetic structure across modes so the physical equivalence can be tested. Add real structures only with provenance and stated occupancy/surface conventions. Keep raw physical intensity distinct from any instrument-convolved profile.

Do not overwrite the archived baseline results without saving them. Preserve Apache-2.0 notices. Do not upload private material files or credentials, change unrelated repositories, or silently install system-wide dependencies. Report unresolved external installation problems without treating them as scientific failures.
