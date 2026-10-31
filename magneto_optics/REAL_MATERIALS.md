# Real materials and fabrication

The synthetic benchmarks established that the Maxwell solver and double-pass
conventions work. A fabricated device must instead start with a specified
material and process. We now include the published cobalt-ferrite composite
parameters and the authors' numerical Ce:YIG dataset. Subsequent design
must keep the material data fixed and optimize geometry within fabrication limits.

The user selected **Si3N4 patterning plus a cobalt-ferrite nanoparticle composite
coating** for the first prototype. Two-photon 3D printing remains a possible later
route. The composite's provisional design band is 1550–1570 nm; the actual laser
and detector have not yet been specified.

## Selected composite material

The [composite input](material_data/royer_2020/cofe2o4_silica_1550_1570nm.json)
uses the effective properties reported by
[Royer et al., ACS Omega (2020)](https://doi.org/10.1021/acsomega.9b03728):
CoFe2O4 nanoparticles at about 15% volume fraction in a silica sol–gel matrix.
Its measured index near 1550 nm is $1.69+0.006i$. The publication gives the
off-diagonal permittivity element as $0.0015(1.2-2.5i)$.

Interpreting that reported element as $\epsilon_{xy}$ in its source field
reference, our convention gives

$$
\epsilon_d=(1.69+0.006i)^2=2.856064+0.02028i,
\qquad g=-i\epsilon_{xy}=-0.00375-0.0018i.
$$

The loss eigenvalues are 0.01848, 0.02028 and 0.02208, all positive. We retain
the measured absorption and the reported complex magnetic coefficient. Absolute
field/viewing conventions must still be calibrated experimentally; reversing
magnetization reverses $g$.

The two identical table endpoints at 1550 and 1570 nm explicitly encode a
**local constant approximation**, not two independent measurements or a measured
dispersion curve. The existing runner forbids extrapolation. The index was
measured outside the patterned area, and the magnetic coefficient was reported
for the loading fraction; this is not a measured tensor for every fabricated slit.
The input is neither pure CoFe2O4 nor a generic printing resin.

The [experimental geometry reference](examples/fabrication/royer_2020_reference.json)
records period, width, depth, coating treatment, and missing device inputs. It
is a source-based starting point, not an optimized mask or a released fabrication
recipe. The paper demonstrated enhancement, not 45-degree transmission and
90-degree mirror return.

## Additional measured garnet reference

[Onbasli et al., Scientific Reports 6, 23640 (2016)](https://doi.org/10.1038/srep23640)
report optical, Faraday and Kerr measurements and fitted complete complex
permittivity tensors for 80 nm Ce1Y2Fe5O12 films on GGG substrates. The original
[supplementary XLS](material_data/onbasli_2016/source.xls) and
[source metadata](material_data/onbasli_2016/source_metadata.json) are included
with attribution under CC BY 4.0. The XLS is unmodified; its SHA256 is
`ff770dd764391e77ce61d650abb2c783d600f2227e1173d1d1c191e5278fd9ce`.

We select the **GGG(110) sample**. This identifies the sample/substrate orientation,
not a different chemical formula. These fitted properties apply to that paper's
film and magnetic state; a generic Ce:YIG film or nanoparticle suspension is not
automatically equivalent. The source's film was grown by PLD at 615 degrees C.
Its easy axis was in-plane and perpendicular saturation was about 5 kOe
(approximately 0.5 T applied field in vacuum). Lithography alone does not create
this film or establish the required magnetic state.

The publication's equation (1) writes

$$
\epsilon_{xy}=-i\epsilon_2,\quad\epsilon_{yx}=+i\epsilon_2.
$$

Our fixed-laboratory convention is $\epsilon_{xy}=+igM$, so at $M=+1$ the
conversion is **$g=-\epsilon_2$**. Positive imaginary diagonal permittivity is
interpreted as absorption with our $e^{-i\omega t}$ convention. The paper does
not explicitly state a harmonic time convention in its prose; the mapping uses
its written tensor and the positive-loss optical data. Absolute experimental
rotation signs also require matching the laboratory magnetic-field and viewing
conventions.

At 1550 nm, linear interpolation in photon energy gives approximately

$$
\epsilon_d=5.134728183+0.001450092835i,\qquad
g=-0.007302259641-0.000596176752i.
$$

Both the optical absorption and magnetic circular dichroism are retained.
The importer uses $E[\mathrm{eV}]=1239.841984/\lambda[\mathrm{nm}]$, converts the
original fits to a 1 nm wavelength table, and records the source hash and method.
The exported file uses the existing `gyrotropic_table` format, including the
magnetization and provenance fields; it can be supplied to existing material
evaluation routines. Runtime interpolation between those exported nodes is
linear in wavelength, as in the existing runner. The table is limited to
1300–1600 nm and extrapolation is forbidden.

The current import retains the supplied complex epsilon1 fit. It does not replace
absorption with zero or combine the best quantities from different samples.
The paper also supplies a separate infrared extinction-coefficient fit; for the
(110) sample at 1550 nm it closely agrees with the epsilon1-derived value.
There is an apparent discrepancy between the source's table of optical losses
and the numerical dataset: the (110) tensor implies roughly 113 dB/cm at 1550 nm,
whereas Table 1 lists 11.2 dB/cm. This is recorded as a source inconsistency,
not silently corrected. Device loss predictions need confirmation on the actual
film before fabrication decisions are based on them.

## Why a measured table still needs checks

A passive material cannot amplify light. For this axial tensor its loss
eigenvalues are

$$
\operatorname{Im}\epsilon_d,\quad
\operatorname{Im}(\epsilon_d-g),\quad
\operatorname{Im}(\epsilon_d+g).
$$

They must all be nonnegative, equivalently
$\operatorname{Im}\epsilon_d\geq|\operatorname{Im}g|$.
The imported (110) data pass in the selected band. In contrast, the publication's
low-loss (111) fit violates this condition at 251 of the 301 requested wavelengths.
For example, at 1550 nm its diagonal loss is about $7.10\times10^{-5}$ while
the magnitude of its off-diagonal loss is about $9.63\times10^{-5}$.
This is an inconsistency of the independently fitted data for use as a passive
tensor, not evidence that the physical film is an amplifier. The importer rejects
it rather than clipping the data or inventing a corrected loss. A correction
would require a documented constrained fit with experimental uncertainties.

## Structures compatible with the stated processes

**Reference alternative: lithographic dielectric resonators with a garnet film.**
An experimentally demonstrated example is a-Si disks above a Ce:YIG/YIG film
on quartz. [Xia et al., ACS Photonics (2022)](https://doi.org/10.1021/acsphotonics.1c01692)
report a 750 nm period, 230–250 nm disk radius and 170 nm disk height, using
electron-beam lithography and dry etching. The work demonstrates enhancement
near 1370 nm, not a 45-degree single pass. These dimensions are a fabrication
precedent, not our optimized mask. Its film stack differs from the measured
GGG(110) material sample above; substituting that dataset is a modeling
assumption until the intended film is characterized.

**Selected route: Si3N4 grating filled with a magnetic nanoparticle composite.**
[Royer et al., ACS Omega (2020)](https://doi.org/10.1021/acsomega.9b03728) fabricated
a 966 nm period grating with 608 nm wide Si3N4 strips, initially 593 nm thick,
on BK7. The approximately 358 nm gaps were filled by dip-coating a silica sol–gel
containing CoFe2O4 nanoparticles, followed by 100 degrees C treatment for one
hour. The reported composite has about 15% particle volume fraction and measured
$n\simeq1.69+0.006i$ near 1550 nm. This is a real material/process alternative
with enhanced magneto-optical response. The adopted tensor mapping is recorded
above and in the material input. The template was made using electron-beam lithography and
reactive-ion etching. Capability to resolve these gaps and etch these films
must be established for the user's particular lithography equipment.

**Two-photon printing remains a separate development route.**
It can form a scaffold or resonator, potentially followed by magnetic coating or
infiltration. It cannot be assumed to print Ce:YIG or to reproduce the tensor of
a dense crystalline film. A loaded or coated printed structure needs measured
optical loss, magnetic response, dispersion, loading and feature fidelity.
This proposed adaptation is not a reproduction of either experiment above.

## Next simulation acceptance

Version 0.5.0 adds a validated one-dimensional tensor RCWA solver for rectangular
Si3N4/composite gratings, with a substrate and full retained-order mirror feedback.
The [first geometry scan](GRATING_SCAN.md) includes independently checked Maxwell
boundaries and volume heating, harmonic refinement, magnetic reversal, spectral
averaging and dimensional sensitivity. It yields a roughly 1.12-degree candidate
at 1560 nm, not a completed fabrication-ready 45-degree device. The older
single-dipole sphere model is not used to approximate these patterned films.
The planar stack solver remains the uniform-film reference; a two-dimensional
post lattice, angular averaging and measured coating/mirror profiles are still
outside this new one-dimensional study.

Acceptance must include forward and returned Jones matrices, transmitted and
returned power, ellipticity, magnetic reversal, material heating, convergence,
finite spectral bandwidth and dimensional tolerance. The mirror and substrate
must be modeled explicitly. The design variables are physical dimensions and
accessible materials; measured $g$ is not a free parameter for forcing 45 degrees.

From `magneto_optics`, regenerate the imported table into a new path:

```powershell
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe -m pip install -e ".[data,test]"
..\mof_scattering_handoff_20261013\.venv\Scripts\python.exe scripts/import_onbasli_material.py --out runs/ceyig_110.json
```

The validated exported table is saved alongside the source XLS. Import checks
and the full test log are saved under `reports/2026-10-26-real-material/`.

Validation on 2026-10-26: **450 tests passed**, including 11 new literature-data
tests. These verify source identity, independent tensor conversion, retention of
both complex components, magnetic reversal, passivity and spectral limits. The
composite's 1550–1570 nm constant approximation is explicitly checked; the garnet
import also checks the union of original energy knots so sparse output cannot
conceal non-passive segments. These tests validate data handling and consistency,
not the user's eventual fabricated sample or a full grating/mirror design.
