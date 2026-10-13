# 05 — Validation and acceptance tests

The 30 original tests are baseline requirements, not full coverage for the three-mode target. Add tests before optimizing. Use deterministic small fixtures for algebraic checks and convergence-controlled tolerances for quadrature/Monte Carlo. Do not use a large forward-scattering peak to hide high-q errors.

## Core algebra and normalization

| Test | Required behaviour |
|---|---|
| One atom | Finite intensity equals f(q)^2, independently of position; explicitly detect native package scale factors |
| Two unlike atoms | Equals f1²+f2²+2 f1 f2 sinc(qr) after powder averaging |
| q=0 | Deterministic finite intensity is `(sum f_i(0))²`; do not assume fitted f(0) equals exactly integer Z |
| Empty host/pores | Valid zero/host-only/guest-only limits; no fabricated particles |
| Decomposition | Complete sum equals host + internal guests + host–guest + between-guests |
| Mixed species | Correct species factors and no double counting |
| Coincident coordinates | Algebraic pair contribution retained; geometry validation may flag it separately |
| Complex-factor extension | When enabled, conjugation and real cross terms tested; current real-factor scope explicit |

For deterministic float64 references, begin with scale-aware relative tolerances around 1e-10 to 1e-9 and nonzero absolute tolerances for extinction/minimum regions, tightening for exactly evaluable small cases. These are suggested targets, not universal guarantees. The baseline's float32 grid test allows scaled errors below 3e-5. Report bandwise scaled maximum error, relative L2 error, and absolute residual near weak features separately.

## Finite directional versus powder

1. Compute vector-q amplitudes for an asymmetric mixed-element finite structure.
2. Average its directional intensity over a converged angular grid at fixed |q|. Gauss–Legendre in cos(theta) plus periodic azimuth quadrature is a useful transparent reference.
3. Compare with the independent Debye sum using exactly the same factors and configuration.
4. Double angular resolution until the discrepancy stabilizes below the chosen numerical target. Test several q*extent regimes; coarse-grid apparent agreement is insufficient.

Translate the entire object: directional intensity must be invariant. Rotate the object: its reciprocal pattern must rotate covariantly, while powder intensity remains invariant. Rotate only a guest: its isolated powder pattern remains invariant, but directional intensity and cross terms may change.

## Finite shape and lattice

Verify full-cell amplitude factorization against explicit atoms, mask autocorrelation against direct integer counts, and lattice-grouped powder against direct Debye. Include an irregular mask and a triclinic cell. Check zero translation, count sum, inversion symmetry and absence of cyclic wraparound.

For a 1D test chain use the exact geometric sum as a directional reference, including its removable singularities. In a sequence of larger finite crystals, verify peak widths shrink with linear size, peak heights and integrated weights follow the declared normalization, and anisotropic shapes create anisotropic broadening. Use integrated windows/convolution, not pointwise comparisons with an ideal delta peak.

Atom-level cut tests must compare the exact retained atom set. Passing full-cell tests does not validate surface chemistry.

## Probability distributions and moments

- Occupancy: measured independent Bernoulli frequencies and covariance agree with the specification.
- Size: frequencies conditional on occupancy agree with number weights; larger particles retain their atom-pair counts.
- Rotation: Haar samples have isotropic first/second axis moments; do not use independently uniform Euler angles.
- Position: unbounded samples recover prescribed means/covariances. Bounded samples are checked against the **conditioned** distribution, not the pre-rejection covariance.
- Confinement: every atom passes the declared wall/host clearance; impossible placements raise rather than changing size weights or removing occupied pores.
- Within-particle occupancy: exact enumeration shows p weighting, not p².
- Moments: enumerate a small finite list of full guest states to obtain exact mu and nu and compare with full configuration enumeration. Check nu minus |mu|² nonnegativity up to roundoff.
- Correlation counterexample: two perfectly correlated occupancies must differ from the independent-pore result. A model requiring such correlations must not silently use the independent formula.

For Monte Carlo versus analytical averages, compare error to the measured SEM and inspect sample-count scaling over independent seeds. A multi-q maximum-z threshold is a diagnostic, not a calibrated independent 3-sigma test because q samples are correlated. Do not confuse the uncertainty of estimated mean amplitudes with physical disorder variance. Include a test of finite-sample bias when shared moments are reused across many cells.

The baseline 8,192-configuration run yielded a 0.565% relative L2 difference and a largest absolute residual/SEM of 1.479. Reproducing the model is required; reproducing exactly those runtime values is not.

## Periodic and bulk-powder tests

Use a simple lattice with a known basis and extinctions, then a nonorthogonal lattice. Verify `A @ B.T = 2*pi*I`. Test complete deterministic occupancy (zero disorder variance), p=0 (host only), and single-atom Bernoulli guests (variance p(1-p)|f|²).

Demonstrate why repeating one random cell differs from independently disordered cells. Check finite mean intensities against the cell-moment expression. Compare integrated, per-cell normalized finite patterns with bulk reflection weights plus diffuse intensity. Handle forward scattering separately. Verify reciprocal-vector multiplicities and radial shell weighting before adding instrument corrections.

For the independent-cell model, diffuse intensity can vary with q and direction; do not test against a flat-background assumption.

## Benchmarks that support useful claims

Record hardware model, OS, Python/package versions or commits, CPU thread settings, device, dtype, warmups, repetitions, synchronization, actual direct/grid path, input hashes and q sampling. Separate model construction, pair-library construction, host/template preprocessing, individual evaluation, probability sampling, and transfers. Save all repeats, medians, errors and peak memory.

Compare the same form factors or separately quantify their differences. Include high-q/weak-intensity regions and multiple atom counts, cell basis sizes, aspect ratios and mask sparsities. Larger q ranges and guest distributions are separate workloads. Avoid mandatory giant N² reference runs when exact small cases and controlled medium-size checks suffice.

An external-package constructor request for `grid` is not proof it executed that kernel. A CUDA option is not evidence of a GPU run. Missing devices or packages must be labelled untested/unavailable for that run.

## Delivery gates

Keep all original tests passing. Provide at least one runnable example for each named scattering case and cross-mode equivalence evidence. Provide unit/convention metadata and a support matrix. Unsupported assumptions must raise explicit errors. Preserve raw outputs alongside plots; synthetic fixtures must not be relabelled as real MOFs.
