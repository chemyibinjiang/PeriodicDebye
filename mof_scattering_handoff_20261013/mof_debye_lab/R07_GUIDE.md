# CIF, pore and finite-boundary inputs (v0.3 / schema 1.1)

Schema `1.1` adds crystallographic CIF input, optional pore-site symmetry
expansion, explicit atom cuts and guest containment, and periodic host clearance.
Schema `1.0` examples retain their previous behavior. The three physical cases
and intensity conventions in [MODE_GUIDE.md](MODE_GUIDE.md) still apply.

## A sourced real-MOF example

The bundled [IRMOF-1/MOF-5 example](examples/real_mof/README.md) uses an unmodified
CIF from a pinned RASPA2 revision, with its MIT notice and SHA-256 provenance.
It is a simulation reference structure for a published material, not a claim
to an experimental CIF deposition or a measured sample. Seven asymmetric-unit
sites and 192 F m -3 m operations expand to 424 unique atoms: C192 H96 O104 Zn32.
All source hydrogen atoms are retained; no solvent or partial sites are removed.

From `mof_debye_lab/`:

```powershell
..\.venv\Scripts\python.exe scripts/run_real_mof_example.py --out ../local_runs/r07_real_mof
```

This runs finite and periodic single-crystal, finite and bulk PXRD, and an
explicitly unpassivated spherical atom cut. Each can also run separately through
`scripts/run_scattering.py` or the installed `mofdebye` command. No download is
needed. The demonstration intentionally computes the empty framework: the source
does not supply nanoparticle pore sites or guest distributions. The input tests
connect CIF hosts to declared synthetic pore/guest models without presenting
those assumptions as measurements on this MOF.

## CIF source and preprocessing choices

Choose exactly one `host.cif` or `host.xyz`. A CIF supplies its own lattice; a
simultaneous `host.cell_vectors_A` override is rejected. All source-file paths are
relative to the configuration file. For example:

```json
{
  "schema_version": "1.1",
  "scattering_case": "pxrd",
  "boundary": "finite",
  "guest_estimator": "explicit",
  "backend": "reference",
  "host": {
    "cif": "IRMOF-1.cif",
    "cif_options": {
      "occupancy_policy": "require_full",
      "exclude_labels": [],
      "exclude_elements": [],
      "duplicate_policy": "merge_same_species",
      "duplicate_tolerance_A": 0.00001
    },
    "cell_mask": [[[1]], [[1]]]
  },
  "sampling": {"q": {"min": 0.0, "max": 3.0, "points": 81}}
}
```

The Python entry point is `mofdebye.crystal.read_cif(path, ...)`, returning a
`CrystalStructure` with `basis`, `cell_vectors_A`, `fractional`, `symops`, and
JSON-compatible `metadata`. `cif_options.block` selects a data block explicitly
when the CIF contains multiple structural blocks. Missing or inconsistent
coordinates, cells or symmetry declarations fail with actionable errors.

The importer uses Gemmi parsing, expands crystallographic operators, wraps
fractional coordinates into `[0,1)`, and deduplicates periodic sites using
Cartesian distances with the full cell metric. This handles special positions,
face equivalences and nonorthogonal cells. Same-species overlapping source
records merge with recorded provenance by default; `duplicate_policy: "error"`
rejects overlaps from distinct source records. Special-position symmetry images
are always deduplicated. Different species at the same position are rejected.

When the CIF declares a site multiplicity, it must agree with that site's unique
expanded orbit. Rounded special-position coordinates can require a carefully
verified `duplicate_tolerance_A`; the importer does not snap coordinates or
silently override an inconsistent multiplicity. With no declared multiplicity,
the supplied coordinates and explicit duplicate tolerance define the orbit.

| Choice | Defined behavior |
|---|---|
| Default `require_full` | Only fully occupied, resolved sites become unweighted host atoms; unsupported fractional occupancy or disorder requires an explicit preprocessing choice |
| `drop_partial` | Explicitly discard partial sites, record the removed source records, then expand the retained structure; never promote fractional occupancy to a full atom |
| `exclude_labels` | Explicit source-site removal, useful for a user-identified solvent/disorder component; applied before occupancy validation |
| `exclude_elements` | Explicit element removal, e.g. H; default retains every element |
| Hydrogen | Preserve supplied positions; do not add, optimize or guess missing H |
| Solvent | No automatic solvent detection or deletion |
| ADP/thermal/charge fields | Record conventions; forward calculation remains fixed neutral atoms without thermal or anomalous corrections |

Partial host occupancy is not equivalent to Bernoulli whole-particle guest
occupancy. This release does not construct host-disorder ensembles or use
fractional atom weights. Dropping or excluding sites creates a different host
model, and its recorded composition must be reviewed accordingly. Unknown
nonstandard symmetry settings must be clarified in the CIF rather than silently
treated as P1.

Explicit operators, Hall symbols and H-M symbols are checked for consistency,
group closure and compatibility with the cell metric. An IT number alone uses
Gemmi's reference setting, recorded as `IT_number_reference_setting`; it cannot
infer a nonstandard axis setting. Ambiguous origin choices are rejected and
require explicit operators or an H-M origin suffix.

## Pores in the imported coordinate frame

Existing `pores.centres_fractional` coordinates now use the CIF lattice directly;
`pores.centres_A` remain Cartesian angstrom. Neither is inferred from unit-cell
centres. The importer does not find pores.

`pores.expand_symmetry: true` optionally expands supplied pore seeds using the
CIF operators and removes duplicate periodic centres. It requires `host.cif`.
Expansion changes **centres only**: shared mean displacement, covariance and
orientation parameters remain in the common Cartesian host frame. It does not
rotate an anisotropic probability law with each symmetry operation. Expanded
seeds require scalar occupancy/radius and a single shared mean-displacement
vector; supply a fully expanded explicit list for per-site arrays. Without the
flag, input pore centres are taken as the complete provided site list. Schema
1.1 rejects duplicate sites within 0.00001 angstrom, including periodic-equivalent
centres when defining a repeated cell basis. Finite explicit XYZ inputs without
a cell mask may declare separate sites in different cells.

Explicit pore lists preserve the supplied coordinate frame, including unwrapped
coordinates; only symmetry expansion wraps nominal centres into `[0,1)`. For a
finite cell mask, choose the site basis deliberately: an out-of-cell site moves
with each retained cell and may lie beyond its geometric envelope. `centre` or
`whole_particle` retention filters those nominal sites; `site` keeps them.

With a finite `cell_mask`, host cells and pore sites are replicated together.
Occupancy/radius arrays and per-site mean offsets are replicated consistently.
Metadata records input/expanded/retained pore counts and final centres. Fractional
row coordinates use `x @ A`, and reciprocal coordinates use `hkl @
(2*pi*inv(A).T)`, including nonorthogonal lattices.

## Complete cells versus atom cuts

For finite CIF input, omitting `cell_mask` selects one complete conventional
cell. A supplied mask selects the existing arbitrary union of complete cells.
Complete-cell backend factorization remains valid and can avoid atom expansion.

`host.atom_cut` additionally filters the explicitly materialized candidate atoms
by a Cartesian sphere or axis-aligned box. For a sphere:

```json
"atom_cut": {
  "kind": "sphere",
  "centre_A": [12.916, 12.916, 12.916],
  "radius_A": 12.0,
  "surface_policy": "unpassivated_atom_cut"
}
```

A box uses `kind: "box"`, `min_A` and `max_A` with the same mandatory surface
policy. The effective object is the intersection with the supplied candidate
cells/XYZ; there is no automatic padding or atom generation outside those
candidates. No surface bonds are repaired, no atoms are capped, and no charge
balance, reconstruction or relaxation is inferred. Metadata records candidate,
retained and removed atom counts. Lattice-grouped acceleration is rejected for
atom cuts; the explicit-coordinate backends handle the actual remaining atoms.

## Guest containment and host exclusion

`host.guest_retention` has three meanings:

- `site` (legacy default): keep complete particles associated with retained pore
  sites, without conditioning their sampled positions on the external surface.
- `centre`: filter nominal pore sites to the external geometry, then condition
  each placement on its displaced particle centre remaining inside.
- `whole_particle`: filter the same nominal sites, then require every atom of
  the rigid guest to remain inside the external geometry.

Atom cuts with guests require an explicit `centre` or `whole_particle` rule.
For masked cuts, external containment uses the intersection of the cut and the
closed cell-mask volume. Rejection preserves the already selected occupancy and
template size; impossible placements raise rather than deleting large particles.
`population.max_attempts` must be a positive integer; fractional and boolean
limits are rejected instead of silently changing the rejection budget.
If no nominal pore sites survive, the output is host-only and records zero
retained pores. Centre/whole-particle conditioning changes the placement law,
so it uses explicit draws or sampled moments; the unbounded-Gaussian analytic
formula is rejected.

Finite `population.min_host_distance_A` checks only the retained finite host.
Bulk schema 1.1 uses `PeriodicHostTree`: query and host positions are reduced
modulo the lattice, and a rigorously bounded image search finds the nearest atom.
This covers guests across cell faces and unbounded displacements without a
guessed neighbor halo; skew cells can require images beyond adjacent cells.
Excessive image searches raise an explicit limit error, never truncate the
exclusion distance. Clearance is between atomic centres using the supplied
scalar threshold, not a force field or element-specific radius model.

All guests remain internally unwrapped and rigid. Periodic hard-wall pore
spheres still must be disjoint across lattice images. General interparticle
exclusion, correlated occupations, multiple particles per pore and energetic
equilibrium sampling remain separate extensions.

## Evidence and provenance

`metadata.json` includes the source hash, selected CIF block, lattice/symmetry
choices, asymmetric-unit and expanded counts, exclusions/occupancy policy, and
surface/pore processing. The real-MOF example additionally records pinned source
URLs, license and independent Gemmi/ASE expansion checks in `provenance.json`.
Original v0.1 and v0.2 results remain preserved; R07 runs use new `local_runs/`
destinations. The v0.2 snapshot is under `local_runs/r07_baseline/`.
