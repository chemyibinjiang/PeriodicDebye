"""Independent checks of source data, tensor conventions and spectral limits."""

import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import xlrd
from numpy.testing import assert_allclose

from magnetooptics.runner import material_tensor


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "material_data"
SOURCE = DATA / "onbasli_2016" / "source.xls"
AUTHOR_SHA256 = "ff770dd764391e77ce61d650abb2c783d600f2227e1173d1d1c191e5278fd9ce"
_module_spec = importlib.util.spec_from_file_location(
    "onbasli_importer", ROOT / "scripts" / "import_onbasli_material.py"
)
importer = importlib.util.module_from_spec(_module_spec)
_module_spec.loader.exec_module(importer)


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _source_value(book, sheet_name, wavelength_nm, column=3):
    """Explicit two-point interpolation of author cells, independent of importer."""
    sheet = book.sheet_by_name(sheet_name)
    samples = sorted(
        (float(sheet.cell_value(row, column)),
         complex(sheet.cell_value(row, column + 1), sheet.cell_value(row, column + 2)))
        for row in range(2, sheet.nrows)
        if sheet.cell_value(row, column) != ""
    )
    energy = 1239.841984 / wavelength_nm
    lower = max((item for item in samples if item[0] <= energy), key=lambda item: item[0])
    upper = min((item for item in samples if item[0] >= energy), key=lambda item: item[0])
    if lower[0] == upper[0]:
        return lower[1]
    fraction = (energy - lower[0]) / (upper[0] - lower[0])
    return lower[1] * (1 - fraction) + upper[1] * fraction


@pytest.fixture(scope="module")
def garnet():
    return importer.build_table(SOURCE, "110")


def test_author_workbook_is_unmodified_and_metadata_identifies_it():
    source = SOURCE.read_bytes()
    metadata = _read_json(SOURCE.with_name("source_metadata.json"))
    assert hashlib.sha256(source).hexdigest() == AUTHOR_SHA256
    assert metadata["sha256"] == AUTHOR_SHA256
    assert len(source) == metadata["bytes"] == 1071616


def test_imported_tensor_uses_author_cells_energy_interpolation_and_equation_one_sign():
    wavelengths = [1300., 1457.25, 1550., 1600.]
    table = importer.build_table(SOURCE, "110", wavelengths)
    book = xlrd.open_workbook(str(SOURCE))
    for wavelength in wavelengths:
        diagonal = _source_value(book, "Epsilon1 CeYIG films", wavelength)
        epsilon2 = _source_value(book, "Epsilon2", wavelength)
        # Article Eq. (1): epsilon_xy=-i*epsilon2, while this repo uses +i*g.
        expected = diagonal * np.eye(3, dtype=complex)
        expected[0, 1], expected[1, 0] = -1j * epsilon2, 1j * epsilon2
        assert_allclose(material_tensor(table, wavelength), expected, rtol=2e-14, atol=2e-16)


def test_garnet_1550_retains_both_complex_tensor_components(garnet):
    actual = material_tensor(garnet, 1550.)
    assert_allclose(actual[0, 0], 5.134728182974142 + .0014500928318342177j,
                    rtol=0, atol=2e-14)
    assert_allclose(actual[0, 1] / 1j, -.007302259631641059 - .0005961767502381412j,
                    rtol=0, atol=2e-16)


def test_exported_garnet_and_all_runtime_knots_are_passive_without_loss_clipping(garnet):
    stored = _read_json(SOURCE.with_name("ceyig_110_1300_1600nm.json"))
    for key in ("wavelength_nm", "epsilon_diagonal", "g"):
        assert_allclose(stored[key], garnet[key], rtol=0, atol=0)
    losses = []
    for wavelength in garnet["wavelength_nm"]:
        epsilon = material_tensor(garnet, wavelength)
        losses.append(np.linalg.eigvalsh((epsilon - epsilon.conj().T) / (2j))[0])
    assert len(losses) == 301
    assert min(losses) > 0
    assert_allclose(garnet["provenance"]["minimum_loss_eigenvalue"],
                    .000572700193465177, rtol=1e-12, atol=1e-16)
    assert garnet["provenance"]["passivity_audit_knot_count"] == 324


def test_raw_111_fit_is_rejected_instead_of_repaired_even_with_sparse_output():
    with pytest.raises(ValueError, match="passivity.*Raw data were not clipped"):
        importer.build_table(SOURCE, "111", [1300., 1600.])


def test_author_import_rejects_extrapolation():
    with pytest.raises(ValueError, match="Extrapolation forbidden"):
        importer.build_table(SOURCE, "110", [1550., 10000.])


def test_import_rejects_unidentified_workbook_before_parsing(tmp_path):
    changed = tmp_path / "changed.xls"
    changed.write_bytes(SOURCE.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        importer.build_table(changed)


def test_garnet_runtime_does_not_extend_author_export_window(garnet):
    for wavelength in [1299.999, 1600.001]:
        with pytest.raises(ValueError, match="extrapolation"):
            material_tensor(garnet, wavelength)


def test_royer_reported_index_and_offdiagonal_convert_to_adopted_tensor():
    table = _read_json(DATA / "royer_2020" / "cofe2o4_silica_1550_1570nm.json")
    # Royer Methods values; sign interpretation is explicitly conditional in provenance.
    diagonal = (1.69 + .006j) ** 2
    reported_xy = .0015 * (1.2 - 2.5j)
    expected = diagonal * np.eye(3, dtype=complex)
    expected[0, 1], expected[1, 0] = reported_xy, -reported_xy
    for wavelength in [1550., 1560., 1570.]:
        assert_allclose(material_tensor(table, wavelength), expected, rtol=2e-15, atol=1e-17)
    assert_allclose(np.asarray(table["g"]), [[-.00375, -.0018]] * 2, rtol=0, atol=1e-18)
    assert "local constant approximation" in table["provenance"]["spectral_policy"]
    assert "not a measured dispersion curve" in table["provenance"]["spectral_policy"]


def test_royer_reversal_transposes_tensor_and_preserves_positive_loss_spectrum():
    table = _read_json(DATA / "royer_2020" / "cofe2o4_silica_1550_1570nm.json")
    plus = material_tensor(table, 1560., 1)
    minus = material_tensor(table, 1560., -1)
    assert_allclose(minus, plus.T, rtol=0, atol=0)
    for tensor in (plus, minus):
        loss = (tensor - tensor.conj().T) / (2j)
        assert_allclose(np.linalg.eigvalsh(loss), [.01848, .02028, .02208],
                        rtol=1e-14, atol=1e-17)


def test_royer_local_approximation_rejects_out_of_range_runtime_queries():
    table = _read_json(DATA / "royer_2020" / "cofe2o4_silica_1550_1570nm.json")
    for wavelength in [1549.999, 1570.001]:
        with pytest.raises(ValueError, match="extrapolation"):
            material_tensor(table, wavelength)
