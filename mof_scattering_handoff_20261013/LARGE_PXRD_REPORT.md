# Finite MOF powder near 100 nm — v0.4.0

The optional `radial_moments` backend computes a resolved finite powder pattern
for a 103.328 nm MOF-5 cube representing 27,136,000 atoms. The numerical model
is the whole-object Debye orientational average, with complete-cell boundaries,
all self terms and finite interference. This is a powder calculation, not the
previous fixed-orientation reciprocal-space line or an infinite-crystal peak list.

The initial accuracy target is 0.1%. Independent physical references, real-MOF
exact calculations, selected full-size unbinned sums, and radial refinement
all pass the stated numerical checks. This is validation of the stated model,
not a claim of experimental agreement or a universal floating-point guarantee.

## Workload and method

The input is the unchanged, sourced RASPA IRMOF-1 model: 424 atoms in a 25.832 A
conventional cell. Its provenance and checksum are in
`mof_debye_lab/examples/real_mof/provenance.json`. A 40^3 binary cell mask forms
the crystallite. No 27-million-atom coordinate array is created. The model has
no guests, relaxed surfaces, thermal disorder or instrumental broadening.

Preparation uses the exact linear shape autocorrelation and species-resolved
basis displacement multiplicities. Nearly identical displacement groups use
rounded keys and actual original coordinate representatives; their measured
perturbation enters an error bound. This reduces 11,339 displacement groups to
1,526, requiring 752,377,514 radius evaluations. The unmerged groups are kept
for independent unbinned validation at selected q values.

Distances are streamed into nearest-center radial bins with moments through
degree 8; the requested maximum bin width is 0.04 A. The sine transform is
evaluated by FFT on a uniform q grid. This retains sub-bin distance information
and finite interference rather than approximating the powder by broadened
infinite-crystal reflections. Six private Numba histograms avoid parallel write
races. Computation is float64 without fastmath.

The per-q absolute approximation bound includes Taylor truncation and the
measured displacement perturbation using |d sinc(qr)/dr| <= q/2. It explicitly
excludes floating-point accumulation and transform roundoff. Independent
comparisons below test the combined numerical result. Intensities are not clipped.

## Validation

The full suite passes **857 tests** (122 more than v0.3.2). The recorded run
took 33.82 seconds. Its 230 JUnit `record_property` compatibility warnings and
six existing optional-package deprecation warnings do not indicate failures.
See `local_runs/pxrd_100nm/pytest.txt` and `pytest.xml`.

Comparisons exclude q < 0.1 inverse angstrom from peak normalization. The 0.1%
criteria include maximum absolute error relative to the non-forward peak,
pointwise relative error above a 1%-of-peak intensity floor, integrated absolute
error, total area and individual peak areas. Near zeros the absolute metric
remains meaningful. Peak positions are compared on the stated discrete grid.

| Independent reference | Grid | Worst non-forward peak-scaled error | Worst significant-point relative error |
|---|---:|---:|---:|
| Real MOF-5, 1/2/4 cells per edge, exact lattice Debye | 601 points each | 1.64e-15 | 4.16e-14 |
| 103.328 nm monatomic cube, exact integer-radius shell multiplicities | 6,001 points | 2.65e-14 | 1.83e-12 |

These references are independent of the radial histogram/FFT. The large
monatomic oracle also avoids the production shape cache and basis symmetry
grouping. Its 201 detected peak positions match exactly on the grid; the worst
peak-area error is 3.58e-13. Small explicit atom-pair tests additionally cover
mixed Zn/C/O/H, skew and left-handed cells, missing cells, coincident atoms,
rotation invariance, thread counts, exact forward identities and nontrivial
approximation bounds.

On the full 27.136-million-atom MOF, original unmerged displacements and direct
compensated sinc sums at q = 0.344, 1, 3 and 6 agree with relative errors no
larger than 4.43e-12. Those four reference values took 79.0 seconds. Halving
the radial bin width and increasing the degree from 8 to 10 gives a maximum
relative discrepancy of 9.39e-11 across **all** 5,901 non-forward samples;
the significant-point metric is 1.44e-12. The approximation bound, excluding
roundoff, is at most 1.82e-6 of the computed intensity across these samples,
well below the 1e-3 target. Refinement is convergence evidence, not an exact
reference. Additional final-source unbinned checks at the strongest sampled peak
(q=0.486) and weakest sampled intensity (q=0.912) give relative errors of
8.92e-15 and 5.01e-11, respectively. These six full-size checks include both
strong diffraction and weak background regions.

## Measured performance

CPU: Intel i5-12400, 12 logical processors, no GPU. Timings exclude interpreter
startup, CIF loading, output serialization and kernel warm-up, which is recorded
separately and uses the Numba disk cache when available. Full first-ever JIT
compilation cost is not established by those warm-up measurements.
Each retained-cache transform was measured three times. A geometry
preparation is a single measurement; these are not three independently built
crystals. Memory includes the Python process and libraries.

For the full 103.328 nm crystal on six threads, the first 6,001-point pattern
took **4.91 seconds** in the initial run and **5.53 seconds** after final memory
and input hardening. Retained-histogram transforms took about **0.21 seconds**.
The final peak working set was **405.5 MiB**. About 96% of first-pattern time
remains in preparing geometry and accumulating radial moments.
Repeated transforms evaluate the same fixed geometry and q grid.

The following comparison uses **one thread and the same 601-point q grid** for
both methods, with three evaluations per retained cache:

| MOF cube edge | Represented atoms | Exact evaluation | Radial evaluation | Speedup including preparation |
|---|---:|---:|---:|---:|
| 2.5832 nm | 424 | 0.04160 s | 0.01568 s | 1.30x |
| 5.1664 nm | 3,392 | 0.58998 s | 0.01372 s | 9.08x |
| 10.3328 nm | 27,136 | 4.0165 s | 0.01827 s | 43.94x |

The last retained-cache evaluation is 219.9x faster. This differs from the
43.94x first-curve speedup because geometry preparation is included only in
the latter. A full dense exact 103 nm MOF pattern was not timed; no extrapolated
speedup is presented as a measurement.

## Reproduction and limits

Install `mof-debye-lab[fast]` for the optional Numba dependency. Exact backends
remain available. The supported fast-backend domain is a deterministic basis,
complete-cell binary mask, and a uniform q grid starting at zero. Guest
populations and atomically cut surfaces are rejected explicitly.

Commands and the reusable Python API are in
[LARGE_PXRD_GUIDE.md](mof_debye_lab/LARGE_PXRD_GUIDE.md). Raw arrays, source hashes,
checksums, environments and timings are in `local_runs/pxrd_100nm/`:

- `small_exact/`: matched exact/approximate MOF comparisons.
- `integer_cube_validation/`: independent 100 nm physical reference.
- `target_n40/`: initial full MOF, four exact points and radial refinement.
- `final_n40/`: final source, additional peak/valley checks and refinement.
- `dense_q_n40/`: 60,001-point q-grid refinement and a denser plotted pattern.
- `runner_n40/`: standard configuration CLI run, bitwise equal to the API result.

The 0.001 inverse-angstrom grid gives about six points over 2*pi/L. Numerical
agreement at its sample points does not establish 0.1% accuracy in an unsampled
peak maximum. In fact, its largest sampled peak is **2.125% below** the maximum
sampled on a ten-times-finer grid. That is q-grid undersampling, not a radial
interpolation error.

The finer **60,001-point** pattern uses dq=0.0001 inverse angstrom and took
**8.79 seconds** for preparation plus the first transform on six CPU threads;
retained transforms had a 3.70-second median and peak working set was 644.7 MiB.
The strongest sampled peak moves from q=0.486 to 0.4865. At shared q coordinates,
the two grids agree to 8.79e-10 relative across the entire non-forward range.
These sampling results are saved in `q_sampling_comparison.json`. A 0.1% claim
for continuous peak maxima or arbitrary integrated windows still requires a
separate q-grid convergence criterion; the backend target concerns intensity
evaluation at requested q values.

The v0.3.2 source snapshot and its verified 861-file manifest are preserved
under `local_runs/pxrd_100nm/baseline/`. Original vendor code, historical results,
logs, `qa/`, and the original ZIP remain preserved.
