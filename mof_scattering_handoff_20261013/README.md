# MOF–nanoparticle scattering handoff

For a beginner's introduction to the mathematics, read the
[Debye tutorial in the repository README](../README.md).

Start with [START_HERE.md](START_HERE.md). Give the local coding agent [LOCAL_AGENT_PROMPT.md](LOCAL_AGENT_PROMPT.md); keep this entire folder available to it.

The implemented v0.4.1 code is in `mof_debye_lab/`. Read
`mof_debye_lab/MODE_GUIDE.md`, `mof_debye_lab/R07_GUIDE.md` and `R07_REPORT.md`
for current behavior. `COMPLETION_REPORT.md` preserves the v0.2 delivery record.
`PHYSICAL_VALIDATION_REPORT.md` documents the subsequent correctness campaign
and runnable examples with known analytical solutions, before performance work.
`PERFORMANCE_REPORT.md` records the subsequent real-MOF CPU benchmarks, exact
optimizations and before/after accuracy comparisons.
The optional finite powder approximation for 100 nm crystals is described in
`mof_debye_lab/LARGE_PXRD_GUIDE.md`, with separate measurements and validation in
`LARGE_PXRD_REPORT.md`.
Independent pore occupancy, particle sizes, random orientations and spherical
confinement now use the same finite powder geometry. See
`mof_debye_lab/PORE_DISTRIBUTION_GUIDE.md` and `PORE_DISTRIBUTION_REPORT.md`.
The original handoff specifications and manifests describe v0.1; intentional
source changes are recorded separately. The original ZIP, historical results,
logs and `qa/` preserve baseline evidence. New runs are in `local_runs/`.
