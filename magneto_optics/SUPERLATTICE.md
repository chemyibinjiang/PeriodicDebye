# Which material, and can a superlattice replace the slab?

The earlier 45-degree slab was an **invented, lossless gyrotropic dielectric**:
at 600 nm its diagonal permittivity was 2.5, its magnetic coupling was 1.5,
and its thickness was 150 nm. Those numbers construct an exact Maxwell benchmark.
They are not optical constants of a measured material.

Version 0.4.0 adds a **layered magnetophotonic superlattice**. The geometry now
enhances a much smaller synthetic coupling. This is a laterally uniform 1D
photonic crystal with a magnetic defect layer; it is distinct from the existing
2D particle-sheet model. Both are useful superstructures, but their solvers and
physical approximations differ.

## A concrete structure

From left to right, the example is

```text
air -> (H L)^5 -> MO -> (L H)^5 -> air gap -> ordinary PEC mirror
                       M = +z                  return travels -z
```

There are 21 material layers, about 1.817 micrometres thick in total.
H and L are transparent dielectric layers with synthetic indices 2.3 and 1.45.
At the design wavelength of 600 nm their quarter-wave thicknesses are

$$
d_H=\frac{600}{4(2.3)}=65.2174\ {\rm nm},\qquad
d_L=\frac{600}{4(1.45)}=103.4483\ {\rm nm}.
$$

The central MO layer has $\epsilon_d=5.29$, $g\simeq0.02904194$, and
$d_{\rm MO}\simeq130.4343$ nm. The air gap is about 50.0010 nm.
The exact values and every layer are saved in
[the input file](examples/superlattice/bragg_cavity.json).
Here $g/\epsilon_d\simeq0.00549$, compared with 0.6 in the previous slab benchmark.
These are still synthetic parameters, not a fit to a garnet sample.

Think of the Bragg layers as mirrors that hold light near the magnetic layer.
The light interacts with that layer repeatedly, so a small magnetic response can
produce a large polarization change near a resonance. This enhancement also makes
the result sensitive to wavelength, absorption, dimensions and mirror position.

## What Maxwell's equations calculate

We use $e^{-i\omega t}$, $\mu_r=1$, normal incidence, and the same real positive
host index on both sides. Magnetization stays along the laboratory $z$ axis on
both propagation legs. Every layer has

$$
\epsilon_j=\begin{pmatrix}
\epsilon_{d,j}&i g_j M&0\\
-i g_j M&\epsilon_{d,j}&0\\
0&0&\epsilon_{d,j}
\end{pmatrix},\qquad
n_{j,\pm}=\sqrt{\epsilon_{d,j}\mp g_j M}.
$$

The two fixed-laboratory circular fields
$\mathbf e_\pm=(\hat x\pm i\hat y)/\sqrt2$ travel independently through the stack.
In each layer their fields obey

$$
E_{j,s}(z)=a_{j,s}e^{i k_0 n_{j,s}z}
          +b_{j,s}e^{-i k_0 n_{j,s}z},\quad s\in\{+,-\}.
$$

Continuity of tangential electric and magnetic fields determines all reflection
and transmission amplitudes. The production solver combines scattering matrices
using decaying propagation factors, avoiding growing transfer-matrix factors in
thick absorbing stacks. It keeps the two incidence directions separate:

$$
\begin{pmatrix}E_{L,\rm out}\\E_{R,\rm out}\end{pmatrix}
=\begin{pmatrix}R_L&T_{RL}\\T_{LR}&R_R\end{pmatrix}
\begin{pmatrix}E_{L,\rm in}\\E_{R,\rm in}\end{pmatrix}.
$$

For an asymmetric stack $R_L$ and $R_R$ generally differ. A mirror behind the
right port must feed back through **$R_R$**. With gap $\ell$ and isotropic mirror
amplitude $r_m$, define $D=r_m e^{2ik_h\ell}I$. Then

$$
F=(I-R_R D)^{-1}T_{LR},\qquad
J_{\rm return}=R_L+T_{RL}D(I-R_R D)^{-1}T_{LR}.
$$

This includes prompt reflection and every cavity return. The first return path
alone is $J_{\rm first}=T_{RL}DT_{LR}$. These are different fields; coherent
path powers cannot be added independently.

For an $x$-polarized input, the transmitted field is

$$
E_x=\frac{t_++t_-}{2},\qquad E_y=\frac{i(t_+-t_-)}{2}.
$$

Equal $|t_+|$ and $|t_-|$ give a line rather than an ellipse. Its rotation is
$\theta=\arg(t_-/t_+)/2$ modulo 180 degrees. The design script solves both
conditions: equal amplitudes and a 90-degree phase difference. It then tunes
the mirror gap to suppress the returned $x$ component. The solver never inserts
a prescribed rotation into the fields.

## What the example demonstrates

At 600 nm the lossless synthetic stack transmits linear polarization at 45 degrees
with approximately **50.004% transmission**. Its first double-pass path has a
90-degree axis and approximately 25.004% power. With the specified mirror gap,
the full coherent returned field has a 90-degree axis and unit power.
The larger full-return power comes from the other coherent paths in the lossless
cavity. It is not evidence of unit single-pass transmission.

The continuously driven field between the stack and mirror is $F E_{\rm in}$;
it is generally different from the isolated-stack transmitted field
$T_{LR}E_{\rm in}$. Therefore the plot labels these as separate experiments,
rather than drawing a misleading sequence of isolated-stack fields along a
loaded cavity. An ordinary isotropic mirror adds no laboratory ellipse rotation.

The demo also adds $+0.001i$ to the MO diagonal permittivity, **without retuning**.
It reports rotation, ellipticity, absorption, transmitted power and returned
analyzer power over 590–610 nm. This loss is a synthetic sensitivity test,
not a measured estimate. The material model has no fitted dispersion, so this
narrow spectral calculation does not establish broadband material feasibility.

| Case at 600 nm | Transmitted axis | Transmission | Full returned axis | Returned power | Returned ellipticity |
|---|---:|---:|---:|---:|---:|
| Ideal synthetic stack | 45.000 degrees | 50.004% | 90.000 degrees | 100% | 0 degrees |
| Same stack with added MO loss | 44.030 degrees | 48.311% | 90.068 degrees | 87.719% | 3.424 degrees |

Axes are equivalent modulo 180 degrees; raw data can report -89.932 degrees for
the last case. The same bare MO film without Bragg layers rotates only 0.6756
degrees, about 66.6 times less than the ideal stack. But at 600.1 nm the ideal
full-return axis is already about 72.72 degrees. The report therefore includes
0.005 nm spectral sampling near the resonance and a separate mirror-gap sweep
of +/-5 nm. This is a narrow resonance demonstration, not a robust broadband
45-degree device.

## Real materials and the 2D metasurface route

Garnets provide experimental starting points for replacing the synthetic data:

| Architecture | Published experimental evidence |
|---|---|
| Alternating BIG/YIG layers with a defect; silver mirror for reflection | A 1.5 micrometre structure gave peak transmission rotation 5.3 degrees at 748 nm and reflection rotation 18 degrees at 733 nm. [Kahl and Grishin, Physical Review B (2005)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.71.205110) |
| Ce:YIG layer between Bragg mirrors | Rotation of -2.92 degrees at 1570 nm, approximately 30 times the film response. [Yoshimoto et al., Optics Express (2016)](https://opg.optica.org/oe/abstract.cfm?uri=oe-24-8-8746) |
| Silicon nanoresonator array with a Ce:YIG film | Approximately fourfold enhancement of Faraday response and magneto-optical figure of merit at 1370 nm. [Xia et al., ACS Photonics (2022)](https://pubs.acs.org/doi/10.1021/acsphotonics.1c01692) |

These demonstrate useful architectures, not the 45-degree performance of our
synthetic fixture. Real-device design needs wavelength-dependent complex
$\epsilon_d$ and $g$ for a specified composition, film quality and magnetization,
plus substrate and mirror optical data. The multilayer API accepts complex
permittivities at each wavelength; this release does not claim a calibrated
real-material design or provide a measured-material database.

A 2D metasurface can also enhance Faraday rotation. A credible next design is
dielectric nanoresonators coupled to a magneto-optical garnet film. The existing
one-electric-dipole-per-sphere sheet cannot resolve its substrate, magnetic Mie
modes, or mirror near fields. Those require a layered-medium volume/multipole
Maxwell solver and convergence checks. A purely reciprocal geometric rotator
does not automatically reproduce the nonreciprocal double-pass effect.

## Run and validate

From `magneto_optics`, using the existing environment:

```powershell
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe scripts/demo_superlattice_mirror.py --out runs/superlattice
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe -m pytest -q
```

To reproduce the synthetic parameter fit into a new file:

```powershell
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe scripts/design_superlattice.py --out runs/refitted_superlattice.json
```

The demo produces complete input/provenance, complex Jones matrices, numerical
checks, ideal/lossy CSV spectra, and PNG/SVG plots. It accepts a `--config` path
to a fixture of this architecture; its acceptance criteria remain 45-degree
linear transmission and 90-degree full return at the design wavelength.
Use `layer_stack_scattering` and `stack_mirror_return` directly for other designs.

Independent tests solve the electric/magnetic boundary equations directly and
integrate volume Joule heating. They check both incidence sides, asymmetric
stacks with a mirror, magnetic reversal, zero coupling, layer splitting,
passivity and thick absorbing stacks. These tests check the Maxwell solution;
they cannot validate invented constitutive data against a real material.

On 2026-10-26, **439 tests passed**, including 75 new multilayer tests.
Six independent cases use the shipped 21-layer fixture at 599.9, 600 and
600.1 nm, with and without the added loss. The demonstration scans 581
wavelengths per case and 201 mirror gaps per case; the recorded numerical
calculation took 2.88 seconds on the local i5-12400, excluding plotting and
interpreter startup. This is one measured run, not a general speed guarantee.
The largest lossless energy-closure error across the saved spectra was
$4.71\times10^{-13}$. Parameter fitting reproduced the saved input exactly.

Saved results: [superlattice report](reports/2026-10-26-superlattice/result.json)
and [figure](reports/2026-10-26-superlattice/overview.png),
[sensitivity plots](reports/2026-10-26-superlattice/sensitivity.png), and
[full test log](reports/2026-10-26-superlattice/pytest.txt).
