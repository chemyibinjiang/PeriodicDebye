# Independent nanoparticle pore distributions, v0.4.1

This extends the finite radial-moment PXRD backend to independent pore
occupancies, discrete number-weighted particle size distributions, Haar-uniform
particle orientations and confined positions. Every physical pore in every
retained cell has its own independent state. The host basis, pore basis and
finite-cell pair counts remain implicit; a randomly populated cell is never
repeated as though it represented independent disorder.

The examples are synthetic by request. They establish numerical and probability
model correctness, not actual MOF pore locations, adsorption energetics or
nanoparticle loading inferred from experiments. Existing real-MOF host-only
measurements remain in `LARGE_PXRD_REPORT.md`.

## Distribution and pair accounting

For pore class a, p_a is occupation probability, b_a(q) is conditional mean
occupied-particle amplitude, and J_a(q) is the conditional number-weighted
internal Debye intensity. K_ab(q) sums sinc(qr) over all retained pairs of cell
translations and the two basis sites. With N retained cells:

```
host-particle = 2 sum_(host i, pore a) f_i p_a b_a K_ia
within-particle = N sum_a p_a J_a
between-particles = sum_(a,b) p_a p_b b_a b_b K_ab
                    - N sum_a (p_a b_a)^2
total = host-host + host-particle + within-particle + between-particles
```

The subtraction removes only the same physical pore. Different cells of the
same pore class still interfere. Internal scattering carries p, while distinct
pores carry the product of probabilities. Negative interference contributions
are retained. In particular, forward intensity includes the variance of random
occupation and size, rather than simply squaring expected electron count.

Analytic local moments support isotropic Gaussian positions without binding
walls, or uniform solid-ball positions whose complete particle support fits
inside the wall for every orientation. A triangle support bound certifies the
latter case. An active spherical wall uses conditional local sampling; the
chosen particle size is held fixed during rejection. Internal rigid-particle
Debye intensities and number weights are integrated exactly.

Two independently sampled mean replicas form each batch's particle-particle
products. Squaring one estimated mean would introduce a numerical bias that
could be amplified by the large number of cell pairs. Batch SEM estimates local
integration uncertainty and is separate from physical disorder variance.

## Independent correctness evidence

`scripts/validate_pore_powder.py` constructs an integer-shell reference for a
simple-cubic one-C host with body-centred Au monomer/dimer mixtures. Occupancy,
Haar rotation and uniform-ball position moments have closed solutions. The
40-by-40-by-40 reference is 103.328 nm across and contains 6,001 q coordinates
from 0 to 6 inverse angstrom.

The final full-pattern comparison in `local_runs/pore_distribution_v041/integer_shell_final/`
gives a maximum significant-intensity relative error of **2.266e-13**, a maximum
error divided by the non-forward total peak of **1.905e-14**, unchanged positions
for all **57 detected peaks**, and a maximum peak-area relative error of
**3.023e-14**. These easily satisfy the 0.1% numerical target for this model.

Independent tests additionally cover:

- Empty, partially occupied and fully occupied pores; size mixtures; multiple
  pore classes with different occupancies; deterministic shifts of pore means.
- Exact q=0 population variance, all four pair groups and negative interference.
- An active monomer wall, whose accepted positions are exactly a smaller uniform
  ball; uncertainty is checked against independent radial quadrature.
- An active dimer wall, whose accepted body-frame positions form the intersection
  of two spheres. Independent lens quadrature distinguishes the correct coupled
  mean from the incorrect product of separate position and orientation means.
- Periodic pore overlap, duplicate sites, incompatible size/wall inputs,
  unsupported anisotropy and surface conditioning, reproducibility, and export.
- A 40-cubed runner input that prohibits full host/guest atom expansion.

## Runnable 100 nm example

The synthetic 25 A cell has 424 carbon sites on a hollow cubic grid. Its central
pore radius is 8 A, with an additional 0.5 A atom-centre clearance. Particles
are fixed-lattice Au13, Au19 and Au43 templates with number weights 0.5, 0.3 and
0.2. Occupancy is 0.6. A 40-cubed mask therefore represents **27,136,000 host
atoms**, **64,000 independent pores**, **38,400 expected particles** and
**798,720 expected Au atoms**.

The 2 A centre ball guarantees that every atom fits. The 4 A proposal ball has
active wall rejection and couples position to orientation. Both retain all
host-particle and particle-particle interference. This synthetic host has more
geometric symmetry than MOF-5; its timing must not be applied to arbitrary real
MOF structures.

From `mof_debye_lab/`, run:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_pore_powder.py --cells 40 --threads 6 --out ../local_runs/my_pores_100nm
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/pore_distribution/active_wall.json --out ../local_runs/my_small_pore_example
..\.venv\Scripts\python.exe scripts/validate_pore_powder.py --cells 40 --threads 6 --out ../local_runs/my_pore_reference
```

Use new output directories for benchmark and validation runs. Saved runner
configurations, CSV components, numerical error tables, compressed arrays,
environment details and source checksums are retained. The supplied small
examples use two cells per edge; the benchmark generates the large masks.

## Initial sampling bottleneck

Initial measurements used the original rotation-based `PorePopulation.sample`
inside each local integration batch. On the existing i5-12400 CPU, six geometry
threads, 6,001 q points, after kernel warm-up:

| Distribution / sample budget | Geometry + local statistics + assembly | Largest SEM / significant intensity |
|---|---:|---:|
| Analytic confined centre ball | 0.714 s | 0 |
| Active wall, 4,096 samples | 5.057 s | 0.226% |
| Active wall, 262,144 samples | 192.070 s | 0.0473% |

Samples count accepted placements per template, pore class and replica across
eight batches. There are two independent replicas, so the last run accepted
1,572,864 local placements and examined 3,090,793 proposals. Local statistics
alone took 191.476 s; final intensity assembly took about 0.003 s. Initial and
refined runs are preserved in `target_n40/` and `converged_n40/`. This separates
local distribution integration from the cost of representing a large crystal.

## Exact change of coordinates for local sampling

For isotropic zero-mean displacement u and independent Haar rotation R, define
v = R-transpose u. The distribution of v is the same isotropic proposal and is
independent of R. Every atom-wall condition becomes
`|R x_i + u| = |x_i + v| <= wall_radius`. The radial mean can therefore be sampled
using translations in the fixed particle frame, with no sampled rotation or
per-placement AtomCloud construction. The wall still conditions the joint
physical distribution; no separability approximation is introduced.

Production sampling uses this change of coordinates with bounded proposal
chunks. The general `PorePopulation.sample` remains the independent reference
and the route for explicit spatial configurations. Seeds remain reproducible
within each algorithm, but optimized draws are not bitwise the same old draws.

The asymmetric, off-centred mixed Au/C/O/Zn reference compares 4,096 draws from
each sampler. Maximum radial discrepancies are 0.982 combined SEM for the
uniform-ball proposal and 1.623 for the Gaussian proposal. Directional means,
wall support, rejection accounting and reproducibility also pass. This speeds
up the estimator without changing its probability law or intrinsic variance.

## Final measured performance and uncertainty

| Production calculation | Geometry + local statistics + assembly | Largest SEM / significant intensity |
|---|---:|---:|
| Analytic confined distribution | 0.697 s | 0 |
| Active wall, 262,144 samples | 9.395 s | 0.0419% |
| Active wall, 1,048,576 samples; independent seed | 31.678 s | 0.0254% |

These times exclude process startup, imports, input/output and separately
recorded kernel warm-up. At the same accepted-sample budget, body-frame local
integration is **21.76 times faster** (191.476 to 8.798 s), and the complete
calculation is **20.44 times faster** (192.070 to 9.395 s). Final geometry
preparation costs 0.573 s; the analytic local statistics cost 0.122 s. Assembling
an intensity from retained geometry and local statistics costs a few milliseconds.

For accuracy reporting, the non-forward region is q >= 0.1 inverse angstrom;
significant points have at least 1% of that region's maximum total intensity.
In the final active-wall run, the largest estimated **three SEM is 0.07635%**
across 2,146 significant coordinates. This is below the initial 0.1% integration
uncertainty target for this synthetic example, but it is not a strict or
simultaneous error certificate. The radial bound at those coordinates is at
most 1.95e-13 of intensity, excluding roundoff.

Refining the optimized calculation from 262,144 to 1,048,576 samples with an
independent seed changes significant intensities by at most **0.01351%**.
The full non-forward comparison is within 1.805 combined SEM after a stated
roundoff floor. Sampling refinement is not an independent exact solution;
the closed-form and geometric-integral tests above supply that separate evidence.

The final process peak working set was **841.9 MiB**. Proposal workspace is
bounded, while the accepted radial array grows with samples per batch and
template atom count. The dominant remaining cost is local integration, not
enumerating a 27-million-atom crystallite. Streaming accepted radii directly
into combined species histograms is a possible next memory/transform optimization.

Measurements and checksums are in `bodyframe_n40/`, `final_n40/` and
`sampling_and_speed_comparison.json` under the new run directory. The final
plot is [available as PNG](local_runs/pore_distribution_v041/final_n40/pore_distribution.png)
and [PDF](local_runs/pore_distribution_v041/final_n40/pore_distribution.pdf).

Reproduce the final sampling budget with:

```powershell
..\.venv\Scripts\python.exe scripts/benchmark_pore_powder.py --cells 40 --threads 6 --samples 1048576 --seed 20261014 --out ../local_runs/my_pores_refined
```

## Final regression and runnable checks

**980 tests pass**, with no failures or skips: 123 more than v0.4.0. The full
suite completed in 32.15 s. The 236 warnings are 230 existing JUnit property
warnings and six existing optional-package deprecation warnings. XML and text
logs are in `local_runs/pore_distribution_v041/pytest_final.*`.

Both shipped pore configurations execute through the standard CLI. The 100 nm
analytic configuration also executes through the CLI, preserving the implicit
host/pore representation and all four exported pair contributions. Its output
is bitwise identical to the benchmark's analytic arrays, as recorded in
`runner_comparison.json`. Cache reuse
tests prohibit changing declared pore centres, wall radii or clearance without
constructing a new cache and repeating periodic pore-overlap validation.

## Accuracy scope and unsupported models

The radial approximation bound excludes floating-point roundoff and Monte Carlo
integration error. SEM is a statistical estimate, not a strict error bound.
Neither addresses q-grid undersampling: 0.001 inverse-angstrom spacing can miss
peak maxima even when intensity at every requested coordinate is accurate.
Use a finer grid for precise peak heights or areas.

Fast radial guests require an isotropic occupied-pore mean: Haar orientations,
isotropic zero-mean sampled proposals and spherical walls. Active wall sampling
rejects nonzero mean shifts. Analytic supported shifts enter the coherent pore
geometry. Fixed/wobble orientations, anisotropic Gaussian positions, atom-based
host clearance, conditioning on the crystallite exterior, overlapping pore
walls and correlated occupation require the more general existing methods or
further algorithms. Complete-cell masks retain whole guests belonging to cell
sites; this is not automatic external-surface containment.

The size library is presently shared across pore classes; occupancies and wall
radii may differ by class. Reusing geometry with a new library or new number
weights updates expected guest-atom metadata from current statistics. q,
declared and effective centres, cell mask, occupancy and physical wall geometry
must match the cache contract.

The original ZIP, historical results, logs and QA are preserved. A verified
v0.4.0 manifest and source snapshot precede this work in
`local_runs/pore_distribution_v041/baseline/`. Intentional changes and the final
validation record are tracked in `SOURCE_AUDIT.json` and the development manifest.
