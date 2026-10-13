# 01 — Discussion summary and decisions

## Original questions

The user asked how to accelerate PXRD simulation of a finite MOF crystal of arbitrary nanoparticle morphology, currently evaluated through the Debye scattering equation. They wondered whether DDSCAT's spatial-repetition/FFT ideas could help. They also asked whether an infinite periodic MOF filled with nanoparticles having size, orientation and position distributions admits an analytical scattering solution.

The discussion progressed from algorithms and existing packages to a distance-based explanation, then to a tested first implementation. The user finally emphasized: **single-crystal mode, PXRD mode, and finite-size single-crystal mode must be kept separate.**

## Conclusions retained for development

### 1. Separate internal structure, outer shape and guest statistics

A periodic internal MOF structure can be exploited even when its external boundary is irregular. With identical complete cells, an atomic displacement separates into a lattice translation plus a basis-atom displacement. Equivalent lattice displacements can be counted once and reused. A finite binary cell mask's autocorrelation supplies those counts; a zero-padded FFT computes the linear correlation.

This is conceptually more relevant than transplanting DDSCAT's electromagnetic multiple-scattering solver. The requested model is kinematic X-ray scattering from known atoms, not a coupled-dipole polarization problem. Existing crystallographic Debye methods are a more direct starting point.

The complete-cell reduction is exact for its stated geometry. It does not automatically handle a cut through a large MOF cell, missing linkers, changed surface termination or relaxed positions. Those need an explicit extension or direct finite-coordinate fallback.

### 2. Use existing Debye engines, but compare the right methods

DEBUSSY/CLAUDE was identified as a candidate for crystal-aware distance libraries and size/shape populations. DebyeCalculator provides a practical Python/GPU-oriented route for explicit finite coordinates. Debyer provides a separate histogram-based candidate; AES-Debye was identified as another accelerated candidate. Their current releases, installation details and actual features should be verified locally.

Only the **adapted DebyeCalculator forward kernels** were actually run in the supplied prototype. A separate new lattice-grouping prototype was tested too. No claim of a successful full-package installation, CUDA run or DEBUSSY/AES-Debye benchmark should be inherited from the discussion.

### 3. In the Debye framework there are four pair categories

For many guests, finite powder intensity contains host–host pairs, pairs within each particle, host–particle pairs, and pairs between distinct particles. Merely adding a MOF powder spectrum to isolated-particle spectra omits the last two categories.

Moving or rotating a rigid particle leaves its internal pair distances—and therefore its own powder pattern—unchanged. It changes pair distances to the host and to other particles, so interference can change. Size changes alter internal pair counts and generally the particle's own spectrum.

A statistical model averages **pair contributions**, not pair distances. In general, the average of `sinc(qr)` is not `sinc(q times average r)`. Random relative orientation does not normally make all cross terms vanish because particles remain associated with pore locations.

### 4. Analytical averages are conditional on the distribution

For independent pores, uniformly random rigid-particle orientations, a discrete size library, and independent unbounded isotropic Gaussian centre displacements, the finite powder average has a closed expression using single-particle Debye intensities and radial mean amplitudes. It was validated against explicit sampled configurations.

For anisotropic, biased or hard-wall-conditioned placements, the current general route samples complete configurations. The model preserves selected size during rejection, so it specifies a distribution of positions/orientations conditional on size. This is a geometric probability model, not a derived thermal equilibrium model.

For periodic bulk ensembles, average amplitudes determine Bragg scattering; amplitude covariance determines additional diffuse scattering. Arbitrary distributions may require numerical local moments even when the infinite lattice sum is handled analytically. An exact expectation formula is not a promise of elementary closed-form integrals for every morphology/distribution.

### 5. Scattering mode is not estimator mode

The physical axes are: fixed orientation or powder averaging; finite boundary or bulk limit; guest distribution; and numerical evaluator. Preserve the user's three named entry points while sharing the underlying structural/statistical objects.

A reciprocal-space slice is not automatically a detector image. A bulk reflection table is not a finite-size profile. A finite Debye curve is not a single-crystal pattern. Label every output accordingly.

## Not yet supplied or decided

No real MOF structure or validated pore list has been supplied. There is no selected nanoparticle material beyond illustrative Au, no experimentally constrained size/position/orientation distribution, no instrument wavelength/resolution model, and no target atom-count/runtime requirement. Finite masks and atomic-level termination policies still require explicit choices. Continue implementation on known synthetic fixtures and make these future inputs configurable, rather than guessing them.
