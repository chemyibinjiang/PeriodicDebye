# Agent instructions for this handoff

## Scope

The user requests a handoff so a local coding agent can complete the codes. The `mof_debye_lab/` tree is the working v0.1 baseline, not the final implementation. This root adds the scientific specification, work plan and fresh verification. Work within this project, using an isolated environment and version control or a preserved baseline copy.

## Non-negotiable model rules

- Keep periodic single crystal, PXRD and finite single crystal as distinct user-facing cases. Internally separate measurement/orientation, boundary condition, guest estimator and numerical backend.
- Standard Debye `sinc(qr)` is a whole-object random-powder average. Do not feed it a vector q and label it a single-crystal calculation.
- A guest rotates rigidly; uniformly random relative orientation does not delete host–guest interference. It is distinct from rotating the entire host–guest object.
- Retain all four finite powder pair groups. Cross terms can be negative; total physical intensity must be nonnegative within numerical tolerance.
- Whole-particle occupancy weights internal guest scattering by p, not p squared. Size weights are per occupied pore, not mass or intensity fractions.
- Confinement changes the joint placement distribution. Preserve the selected size during rejection under the current model; never apply the unbounded-Gaussian analytical formula to rejected/clipped samples.
- A periodic random ensemble is not a single randomly filled cell repeated forever. The latter imposes artificial long-range correlations.
- Infinite-crystal Bragg scattering is distributional; use reflection weights and a separately normalized continuous diffuse component. Do not report an arbitrary finite number as an ideal peak height.
- Shape FFT correlations must be linearly zero-padded. Finite boundaries are not minimum-image periodic boundaries.
- Full-cell shape grouping is not an atomically terminated or reconstructed surface model.

## Evidence and implementation discipline

Reproduce tests before refactoring. Keep a simple independent reference. Use one-/two-atom checks, q=0 limits, mixed species, nonorthogonal cells and angular-average equivalence. Record units, normalization, estimator assumptions, seed, version, device and actual method in outputs. Detect direct fallbacks. Bound peak memory, not just sinc workspace. Treat Monte Carlo SEM as numerical integration uncertainty, not physical parameter uncertainty.

The historical CPU speedups apply only to the adapted forward kernels. No full-package, DEBUSSY, AES-Debye or GPU execution is established by those results. Inspect and verify external APIs rather than assuming the bundled optional version pin is valid. Keep vendor licenses and notices; do not rebrand imported algorithms as new project code.

Preserve `mof_debye_lab/results/`, `mof_debye_lab/logs/`, and this handoff's `qa/` as baseline evidence. Write new runs to `local_runs/` or a versioned destination. Regenerate checksums only after documenting intentional source changes. New schemas and module names in the handoff are proposals, not existing APIs.

Do not require a real CIF to finish synthetic forward-model tests. Do not fabricate missing material data. Add a clear input contract for a future real MOF. Keep unsupported correlations, relaxations and instrument physics explicit.
