# Physical examples with known solutions

These generated examples test scattering against explicit equations. They are synthetic geometries for numerical validation; they do not represent a proposed chemical structure, synthesis, or experimental measurement. The BCC and FCC examples also exercise CIF symmetry expansion before evaluating scattering.

From the `mof_debye_lab` directory, run:

```powershell
& '../.venv/Scripts/python.exe' scripts/validate_physical_examples.py --out ../local_runs/physical_examples
```

Add `--no-plot` to generate numerical results without figures. The driver writes the generated XYZ/CIF inputs and configuration JSON, per-case CSV comparisons, equations and error summaries in JSON, and two PNG/PDF figures: `physical_geometries` shows the inputs and `physical_examples` overlays calculations and exact equations. The random case uses a recorded seed and 8,192 draws. The numerical checks run independently of plotting.

Positions and distances are in angstrom; wavevectors are in inverse angstrom and include the factor of `2*pi`. Unnormalized intensity is in electron squared. Here `f_C(q)` and `f_Au(q)` are real neutral-atom IT92 form factors. The implementations share the atomic-factor table; these comparisons independently test geometry, interference, occupancy, and averaging, rather than the accuracy of that table. Define `sinc(x) = sin(x)/x`, with `sinc(0) = 1`.

## 1. One carbon atom

For one atom at any position, every direction and the powder average obey

```text
I(q) = f_C(q)^2.
```

Moving the atom changes only its amplitude phase. This case detects a missing self term, an incorrect intensity normalization, or unintended dependence on direction.

## 2. Two carbon atoms: direction versus powder

Place two atoms `d = 2.5 A` apart. For a wavevector parallel to their separation,

```text
I_parallel(q) = 2 f_C(q)^2 [1 + cos(q d)].
I_transverse(q) = 4 f_C(q)^2.
I_powder(q) = 2 f_C(q)^2 [1 + sinc(q d)].
```

The powder equation follows by averaging the cosine over all directions of the entire two-atom object. This case detects substituting a Debye powder average for directional scattering, losing the cross term, or using the wrong Fourier convention. All three agree at `q = 0`.

## 3. A finite nine-atom chain

Place nine carbon atoms on a line with spacing `d = 2.1 A`. Along the chain,

```text
I_parallel(q) / f_C(q)^2 = [sin(N q d / 2) / sin(q d / 2)]^2,  N = 9.
```

At `q d = 2 pi m`, evaluate the removable singularity by its limit `N^2 = 81`. The independently derived powder result counts pairs at each distinct distance:

```text
I_powder(q) / f_C(q)^2 = N + 2 sum_{m=1}^{N-1} (N-m) sinc(q m d).
```

This case tests finite-size interference, pair multiplicity, the factor of two for unordered pairs, and stable evaluation at exact diffraction maxima. The finite chain has a continuous intensity curve with finite peaks.

## 4. BCC centering and forbidden reflections

An `I m -3 m` CIF with a cubic `a = 4 A` cell and one carbon seed at the origin expands to two atoms. At a reciprocal-lattice vector `q = (2 pi/a)(h,k,l)`, its unit-cell structure factor is

```text
F_hkl / f_C(q) = 1 + (-1)^(h+k+l).
|F_hkl|^2 / f_C(q)^2 = 4 if h+k+l is even, and 0 otherwise.
```

This tests the imported multiplicity, body-centering translation, reciprocal-vector units, and destructive interference at forbidden reflections.

## 5. FCC centering and bulk powder weights

An `F m -3 m` CIF with `a = 4 A` and one carbon seed at the origin expands to four atoms. Its structure factor obeys

```text
F_hkl / f_C(q) = 1 + (-1)^(h+k) + (-1)^(h+l) + (-1)^(k+l).
|F_hkl|^2 / f_C(q)^2 = 16 if h,k,l are all even or all odd, and 0 otherwise.
```

For a cell volume `V`, the bulk single-crystal reflection weight multiplying `delta^3(q-G)` is

```text
W_G = (2 pi)^3 |F_G|^2 / V.
```

The powder radial weight multiplying `delta(q-|G|)` is the sum over all reciprocal vectors in that shell of `W_G / (4 pi |G|^2)`. This convention satisfies the three-dimensional measure with radial volume element `4 pi q^2 dq`. The forward reflection at `q = 0` is kept separately. These distributional weights are not peak heights; no instrumental broadening is assumed. The example detects lost shell multiplicities and missing volume or radial-Jacobian factors as well as face-centering errors.

## 6. Random occupancy of one gold guest

Keep a carbon host fixed and occupy one gold guest site with probability `p`. At `q = 0`, positions do not affect the answer:

```text
E[I(0)] = f_C(0)^2 + p f_Au(0)^2 + 2 p f_C(0) f_Au(0).
```

The guest self term is proportional to `p`. Squaring the mean amplitude alone would incorrectly replace that term by `p^2 f_Au^2`; the omitted contribution is `p(1-p) f_Au^2`. Sweeping `p` tests the empty and fully occupied limits and the distinction between mean intensity and squared mean amplitude.

## 7. A Gaussian-displaced gold guest

Use `p = 0.37`, mean position `r = [1.2, 0.5, 0.8] A`, and isotropic Gaussian displacement with component standard deviation `sigma = 0.45 A`. With `q = |q_vector|`,

```text
mean_guest(q_vector) = p f_Au(q) exp(-sigma^2 q^2/2) exp(i q_vector.r).
diffuse(q) = f_Au(q)^2 [p - p^2 exp(-sigma^2 q^2)].
E[I(q_vector)] = |f_C(q) + mean_guest(q_vector)|^2 + diffuse(q).
```

For independent cells, the periodic coherent part has the same mean cell amplitude and the Bragg prefactor `(2 pi)^3/V`; its continuous diffuse contribution is the variance above per cell. This case tests the characteristic-function exponent, host-guest interference, occupancy variance, and the different normalizations of finite intensity and bulk Bragg measures.

## 8. A uniformly displaced gold guest inside a ball

Use `p = 0.65`, mean position `r = [2, 1, 0.5] A`, and a displacement uniformly distributed throughout a ball of radius `R = 1.25 A`. Its characteristic function is

```text
x = q R.
phi(q) = 3 [sin(x) - x cos(x)] / x^3, with phi(0) = 1.
mean_guest(q_vector) = p f_Au(q) phi(q) exp(i q_vector.r).
diffuse(q) = f_Au(q)^2 [p - p^2 phi(q)^2].
E[I(q_vector)] = f_C(q)^2 + p f_Au(q)^2
                + 2 p f_C(q) f_Au(q) phi(q) cos(q_vector.r).
```

The sampled moments are compared with these exact expressions using 8,192 draws and an independently derived intensity integration uncertainty. This checks uniform sampling in volume, bounded-displacement interference, moment estimation, and stochastic error accounting. Monte Carlo uncertainty describes numerical integration error; it is not uncertainty in material parameters. A finite sample need not match the exact curve to floating-point precision.

## Reading the results

Use the saved CSVs and JSON errors to inspect numerical agreement; plots make interference and extinction patterns easier to see. Exact cases should agree within their declared numerical tolerances. The sampled case should agree within its declared statistical criterion. These checks establish correctness for the stated assumptions and geometries; they do not establish a performance result or add chemical relaxation, thermal motion, or instrument physics.
