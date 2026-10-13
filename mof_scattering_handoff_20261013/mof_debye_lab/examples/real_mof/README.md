# IRMOF-1 (MOF-5) CIF example

These configurations run the three scattering cases on a sourced real-MOF
reference model. The unmodified CIF comes from the
[RASPA2 maintainers](https://github.com/iRASPA/RASPA2/blob/6498ab1eec9c8e0f063dcd0d71dd7add372c529b/structures/mofs/cif/IRMOF-1.cif)
at the pinned commit recorded in `provenance.json`. Its upstream MIT notice is
preserved in `RASPA2-COPYING.txt`. No network access is needed to run the examples.

The file identifies its creation method as RASPA-1.0 and cites Eddaoudi et al.,
*Science* **295**, 469-472 (2002),
[DOI: 10.1126/science.1067208](https://doi.org/10.1126/science.1067208).
It is a simulation reference structure of a published material. The source does
not establish an experimental deposition identifier, a specific sample, or an
optimization method. RASPA also labels its supplied structures as examples that
require validation for a new application.

## Reproduce

From `mof_debye_lab/`, after installing this project in the development environment:

```powershell
..\.venv\Scripts\python.exe scripts/run_real_mof_example.py --out ../local_runs/r07_real_mof
```

This single command verifies the source checksum and expanded composition, runs
all five configurations, and saves raw CSV tables and JSON metadata, an expanded
unit-cell XYZ, a run summary, and PNG/PDF plots. Finite and bulk powder outputs
have separate axes and units. Use `--no-plot` when matplotlib is unavailable.
The bundled data suffice; the driver never downloads a structure. The focused
regression tests run with `python -m pytest tests/test_real_mof.py -q`.

To run an individual configuration:

```powershell
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/real_mof/periodic_single_crystal.json --out ../local_runs/r07_real_mof/periodic_single_crystal
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/real_mof/finite_single_crystal.json --out ../local_runs/r07_real_mof/finite_single_crystal
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/real_mof/pxrd_finite.json --out ../local_runs/r07_real_mof/pxrd_finite
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/real_mof/pxrd_bulk.json --out ../local_runs/r07_real_mof/pxrd_bulk
..\.venv\Scripts\python.exe scripts/run_scattering.py --config examples/real_mof/pxrd_atom_cut.json --out ../local_runs/r07_real_mof/pxrd_atom_cut
```

With another environment, substitute its interpreter for the path above. An
installed `mofdebye --config ... --out ...` command is equivalent.

| Configuration | Host boundary and calculation | Output |
|---|---|---|
| `periodic_single_crystal.json` | Infinite ordered conventional cell; reflections through 3 inverse angstrom | Bragg weights and zero static diffuse scattering |
| `finite_single_crystal.json` | Two complete cells along a, one along b/c; factorized directional amplitudes | A reciprocal h00 line including finite-shape interference |
| `pxrd_finite.json` | One complete cell, explicit pair Debye calculation | Finite-object powder intensity |
| `pxrd_bulk.json` | Infinite crystal with reflection-shell grouping | Radial powder delta weights and zero static diffuse scattering |
| `pxrd_atom_cut.json` | Atoms inside a 12 angstrom sphere centred at (12.916, 12.916, 12.916) angstrom, intersected with one complete cell | Finite powder intensity for an explicitly unpassivated geometric cut |

The finite powder curve and bulk shell weights have different measures and
normalizations. They must not be overlaid as equivalent peak heights. Finite
surfaces are the result of the declared complete-cell selection. Broken boundary
bonds are not repaired or passivated; the model is not a surface-chemistry
calculation. The small cell counts keep this example quick, rather than represent
an experimentally measured crystallite size. The additional atom-cut example
uses `surface_policy: "unpassivated_atom_cut"` and changes the composition at the
surface; it does not represent a chemically balanced terminated particle. All
configurations select schema version 1.1 for the CIF input features.

## Expansion and material treatment

The F m -3 m conventional cubic cell has a = b = c = 25.832 angstrom. Its seven
asymmetric-unit records and 192 explicit symmetry operations must yield 424
unique atoms:

| Element | Atoms per conventional cell |
|---|---:|
| C | 192 |
| H | 96 |
| O | 104 |
| Zn | 32 |

This is eight Zn4O(BDC)3 formula units, with BDC = C8H4O4. Special-position
multiplicities by source label are Zn1:32, O1:8, O2:96, C1:48, C2:48, C3:96,
and H1:96. Both Gemmi 0.7.5 and an independent ASE 3.22.1 import gave the same
species-resolved fractional coordinate set to eight decimal places before the
new project importer was used. The CIF SHA-256 and crosscheck details are in
`provenance.json`.

All source H atoms are retained; no solvent, atoms, or labels are removed. The
source omits occupancy and disorder columns, so every record has the CIF default
occupancy of one. The example requests `require_full` occupancy explicitly.
Thermal motion, relaxation, defects, anomalous scattering and instrument effects
are absent. Source charges are not used to modify the neutral IT92 factors.

No pore centres or nanoparticle populations are supplied by this source. These
examples therefore calculate the empty framework. They do not insert a guest at
the unit-cell centre or infer a loading law. To model a guest-filled MOF, supply
independently justified `pores.centres_fractional` or `pores.centres_A`, guest XYZ
templates and `population` parameters using the main input guide. Fractional
coordinates then use the cell imported from this CIF. The input tests exercise
that path separately with declared synthetic pore geometries.
