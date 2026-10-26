# Implementation and provenance

The finite, periodic, and homogeneous slab/mirror solvers in this folder are new source for this
repository, distributed under the accompanying Apache License 2.0. It uses
NumPy and SciPy numerical routines; no third-party solver source was copied.
The electromagnetic conventions and literature are cited in README.md and
PERIODIC_TUTORIAL.md and FARADAY_MIRROR.md. The periodic Ewald derivation follows the cited literature;
the implementation, tests, and synthetic examples are original repository code.

The original X-ray implementation, external-kernel notices, archived datasets,
and manifests remain in ../mof_scattering_handoff_20261013/.

The original examples remain explicitly synthetic fixtures and must not be
described as measured optical constants. The data in material_data/onbasli_2016/
are a separate exception: the authors' supplementary dataset and an explicitly
documented numerical conversion of it, from M. C. Onbasli et al., Scientific
Reports 6, 23640 (2016), DOI 10.1038/srep23640. These data are attributed to the
authors and distributed under CC BY 4.0, https://creativecommons.org/licenses/by/4.0/,
as specified by the publication. Their original license is retained; the
repository's Apache license does not replace it. See source_metadata.json for
the original URL and checksum, and REAL_MATERIALS.md for conversion and limits.

material_data/royer_2020/ records numerical facts from F. Royer et al., ACS Omega
5, 2886-2892 (2020), DOI 10.1021/acsomega.9b03728, and our explicitly documented
tensor conversion/local constant approximation. The article is available under
ACS AuthorChoice terms (non-commercial copying/redistribution); no article text
or figures are bundled. The provenance notes and geometry record are original
summaries and do not imply validation of a fabrication-ready device.
