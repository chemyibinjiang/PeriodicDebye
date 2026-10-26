# Implementation and provenance

The finite and periodic electromagnetic solvers in this folder are new source for this
repository, distributed under the accompanying Apache License 2.0. It uses
NumPy and SciPy numerical routines; no third-party solver source was copied.
The electromagnetic conventions and literature are cited in README.md and
PERIODIC_TUTORIAL.md. The periodic Ewald derivation follows the cited literature;
the implementation, tests, and synthetic examples are original repository code.

The original X-ray implementation, external-kernel notices, archived datasets,
and manifests remain in ../mof_scattering_handoff_20261013/.

All supplied magneto-optical material tables are explicitly synthetic fixtures.
They must not be described as measured optical constants of any real material.
