# MOF + nanoparticles: local-agent handoff

**Prepared 13 October 2026.** This is a development handoff, not a claim that all scattering modes are implemented.

## What to do first

Give your local coding agent **`LOCAL_AGENT_PROMPT.md`**, with this entire folder available to it. The agent should read `AGENTS.md`, inspect the existing project, reproduce its tests, and work through `TASKS.json` and `docs/04_IMPLEMENTATION_ROADMAP.md`.

The runnable v0.1 project is in **`mof_debye_lab/`**. Its 62 files have been preserved byte-for-byte from the supplied archive, including source, examples, historical logs, raw data, plots, licenses, and its original manifest. The new instructions live outside that directory. There is no need to start a new scattering package from scratch.

## The three cases must remain distinct

| User-facing case | Quantity required | Status in v0.1 |
|---|---|---|
| Periodic single crystal | Direction-resolved Bragg weights and, for disorder, diffuse intensity in reciprocal space | **Not implemented** |
| PXRD | Whole-crystallite powder average; distinguish finite crystallites from the bulk limit | **Finite random-powder calculation implemented**; bulk powder not implemented |
| Finite single crystal | Direction-resolved finite-object intensity, retaining shape interference and broadening | **Not implemented** |

The existing CLI's `--mode analytic` and `--mode montecarlo` select **how to average guest configurations**, not which of these scattering cases is calculated. Both currently return finite-system powder Debye intensity. This distinction is the highest-priority architecture correction.

## What already works

There is an independent full-Debye reference, an adapted accelerated DebyeCalculator forward-kernel extract, a rigid-particle probability sampler, a restricted analytical powder ensemble average, and a separate complete-unit-cell FFT/lattice-grouping prototype. Host–host, within-particle, host–particle and between-particle contributions are retained.

All numerical examples use a **synthetic C/O/Zn host**, not a chemically validated MOF, and illustrative Au templates. A real MOF CIF, validated pore centres, experimental distribution parameters and instrument settings have not been supplied.

## Verification during this handoff

- The original 61-entry internal file manifest passed, and all 62 files match the original ZIP members.
- The unchanged project passed **30 tests in 3.46 s** on CPU in this handoff session (`qa/baseline_pytest.txt`, `qa/baseline_pytest.xml`).
- The analytical example and a **64-configuration, forced-grid bounded-pore CLI example** both executed successfully (`qa/smoke_*`).
- These are fresh baseline checks, **not** a new large benchmark, full-package installation test or GPU test.

Historical performance measurements are preserved in `mof_debye_lab/REPORT.md` and `results/`. They describe adapted-kernel CPU microbenchmarks. In particular, the reported 83-fold speedup at 1,536 atoms is not a GPU result and does not establish crystal-periodicity acceleration.

## Contents

| File/folder | Use |
|---|---|
| `LOCAL_AGENT_PROMPT.md` | Ready-to-use instruction for the local coding agent |
| `AGENTS.md` | Development rules and non-negotiable scientific constraints |
| `docs/01_DISCUSSION_AND_DECISIONS.md` | Condensed scientific discussion and user requirements |
| `docs/02_PHYSICS_AND_MODE_SPEC.md` | Equations, normalization and the three-mode contract |
| `docs/03_CODE_STATUS_AND_AUDIT.md` | Existing functions, actual coverage, and implementation hazards |
| `docs/04_IMPLEMENTATION_ROADMAP.md` | Prioritized development sequence and completion criteria |
| `docs/05_VALIDATION_PLAN.md` | Cross-mode, statistical and performance acceptance tests |
| `docs/06_RUNBOOK_AND_INPUTS.md` | Reproduction commands, local setup, and missing material inputs |
| `docs/07_SOURCES_AND_PROVENANCE.md` | Recorded references, source attribution and evidence boundaries |
| `TASKS.json` | Machine-readable task list; proposals are not marked implemented |
| `HANDOFF_METADATA.json` | Original archive hash, scope, environment and recheck outcomes |
| `qa/` | Fresh test logs and smoke-test outputs |
| `mof_debye_lab/` | Complete, unchanged v0.1 source and results |
| `tools/verify_manifest.py` | Standard-library-only integrity check |

From this root, `python tools/verify_manifest.py` checks the new outer manifest. From `mof_debye_lab/`, the existing example CLIs remain runnable as documented. The proposed three-mode interface is **not yet present**.
