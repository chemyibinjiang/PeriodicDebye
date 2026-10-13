# 07 — Sources and provenance

This handoff summarizes the supplied conversation and local archive. External sites were **not re-browsed for this packaging step**; the references below are recorded sources for the local agent to consult and verify. Do not treat an old URL, discussed capability or version pin as proof of the current software state.

## Source archive and unchanged baseline

Input: `mof_debye_lab_v0.1.zip`. The original SHA-256, every ZIP member hash and the new baseline-check environment are recorded in `HANDOFF_METADATA.json` and `qa/original_archive_inventory.json`. The `mof_debye_lab/` directory preserves the original 62 files unchanged. The internal `MANIFEST.sha256` covers 61 files, excluding itself.

Historical numerical evidence: `mof_debye_lab/REPORT.md`, `results/benchmark.csv`, `results/benchmark_details.json`, `results/validation_summary.json`, `results/environment.json`, and `logs/pytest.txt`. Fresh recheck evidence: `qa/baseline_pytest.*`, `qa/smoke_analytic/`, and `qa/smoke_bounded/`. The complete 8,192-sample historical validation and speed benchmark were not rerun during handoff; the original test suite and two small example CLIs were.

## Adapted accelerated source

- DebyeCalculator repository: https://github.com/FrederikLizakJohansen/DebyeCalculator
- Inspected file: `debyecalculator/debye_calculator.py`
- Recorded **file Git blob SHA-1**, not a repository commit: `89cd257ac681d3859457db7e8220cbc325c6953d`
- JOSS paper: https://joss.theoj.org/papers/10.21105/joss.06024
- Local attribution: `mof_debye_lab/src/mofdebye/vendor/NOTICE.txt`
- Local upstream license: `mof_debye_lab/src/mofdebye/vendor/LICENSE-Apache-2.0.txt`

The forward kernels were extracted/adapted; GUI, readers, native factor tables, autograd/backward routines and progress machinery were not retained. The wrapper uses conventional full-Debye normalization and common Gemmi factors. This is not a byte-identical installed package. Preserve and update notices for modifications.

## Other recorded software candidates

- DEBUSSY project: https://debyeusersystem.github.io/
- DEBUSSY 2010 paper copy discussed previously: https://www.psi.ch/sites/default/files/import/sls/ms/Station1_ResearchEN/aj5166.pdf
- DEBUSSY later software article: https://journals.iucr.org/paper?to5122=
- Debyer documentation: https://debyer.readthedocs.io/en/latest/
- AES-Debye article discussed previously: https://journals.iucr.org/j/issues/2026/05/00/po5180/
- DDSCAT repository: https://github.com/DDSCAT/current

These candidates were discussed, **not executed** in the source archive. DDSCAT is conceptual background for translation/FFT reuse, not the chosen scattering solver. Verify exact geometry support before assuming any package automatically accelerates an arbitrary full-cell or atom-cut morphology.

## Mathematical and dependency references

- Gemmi scattering factors: https://gemmi.readthedocs.io/en/latest/scattering.html
- SciPy Haar-uniform rotation sampler: https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.random.html
- Statistical crystalline-disorder reference used in discussion: https://journals.iucr.org/d/issues/2015/08/00/rr5095/index.html
- Crystal pair-vector Debye method discussed: https://journals.iucr.org/paper?sh5094=
- Whole-PDF / finite-shape discussion: https://journals.iucr.org/m/issues/2021/02/00/fc5048/index.html

The equations in document 02 state the selected Fourier, powder and intensity conventions explicitly. Their implementation must be verified by small exact models and cross-mode tests rather than solely by appealing to a reference. No third-party wheels, credentials, private workspace files or fonts are included.
