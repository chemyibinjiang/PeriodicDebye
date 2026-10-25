"""End-to-end input, geometry, spectral and provenance acceptance tests."""

import copy
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

import magnetooptics.runner as runner
from magnetooptics.geometry import ParticleGeometry, from_config
from magnetooptics.runner import material_tensor, simulate, validate_config, write_run


def _config(backend="dense"):
    return {
        "schema_version": "magnetooptics-1.0",
        "description": "Synthetic sparse test, no experimental material claim",
        "geometry": {
            "kind": "grid", "shape": [4, 3, 2], "spacing_nm": [33, 41, 37],
            "active_indices": [[0, 0, 0], [3, 0, 1], [1, 2, 0], [0, 1, 1]],
            "radius_nm": [4, 5, 6, 7],
            "site_materials": ["magnetic", "dielectric", "magnetic", "dielectric"],
            "origin_nm": [-7.3, 12.5, 3.1],
        },
        "materials": {
            "magnetic": {
                "model": "gyrotropic_table", "wavelength_nm": [450, 700],
                "epsilon_diagonal": [[3, 0.2], [2.4, 0.3]],
                "g": [[0.3, 0.02], [0.4, 0.03]],
                "magnetization": [0.2, 0.3, 0.8], "provenance": "synthetic test",
            },
            "dielectric": {"epsilon_diagonal": [2.2, 0.1], "g": 0},
        },
        "wavelength_nm": [480, 550, 650], "host_index": 1.2,
        "incidence": {"direction": [1, -2, 3], "reference_axis": [1, 0, 0]},
        "solver": {"backend": backend, "rtol": 1e-12}, "threads": 1,
    }


def _assert_spectral_rows_equal(actual, reference, rtol=2e-9, atol=2e-11):
    assert len(actual) == len(reference)
    for result, expected in zip(actual, reference):
        assert result.keys() == expected.keys()
        for name, value in expected.items():
            if name == "max_relative_residual":
                assert result[name] < 2e-12
            elif value is None:
                assert result[name] is None
            else:
                assert_allclose(result[name], value, rtol=rtol, atol=atol, err_msg=name)


@pytest.mark.parametrize("backend", ["fft", "dense_gmres"])
def test_full_spectrum_matches_dense_for_all_magnetic_and_helicity_channels(backend):
    config = _config()
    untouched = copy.deepcopy(config)
    reference, dense_metadata, dense_geometry = simulate(config)
    assert config == untouched
    config["solver"]["backend"] = backend
    actual, metadata, geometry = simulate(config)
    _assert_spectral_rows_equal(actual, reference)
    assert_allclose(geometry.positions_nm, dense_geometry.positions_nm)
    assert_allclose(geometry.radii_nm, [4, 5, 6, 7])
    assert metadata["geometry"]["particle_count"] == 4
    assert metadata["geometry"]["grid_shape"] == [4, 3, 2]
    assert metadata["geometry"] == dense_metadata["geometry"]
    for record in metadata["records"]:
        assert set(record["solves"]) == {"Mplus_plus", "Mplus_minus", "Mminus_plus", "Mminus_minus"}
        assert all(case["relative_residual"] < 2e-12 for case in record["solves"].values())
    for row in actual:
        positive = row["extinction_nm2_Mplus_plus"] - row["extinction_nm2_Mplus_minus"]
        negative = row["extinction_nm2_Mminus_plus"] - row["extinction_nm2_Mminus_minus"]
        assert_allclose(row["magnetic_cd_extinction_nm2"], (positive - negative) / 2)
        assert_allclose(row["even_cd_extinction_nm2"], (positive + negative) / 2)


@pytest.mark.parametrize("backend", ["dense", "fft"])
def test_translation_preserves_full_spectrum_and_forward_polarization(backend):
    original = _config(backend)
    translated = copy.deepcopy(original)
    shift = np.array([211.0, -137.0, 89.0])
    translated["geometry"]["origin_nm"] = (np.asarray(original["geometry"]["origin_nm"]) + shift).tolist()
    reference, _, old_geometry = simulate(original)
    actual, _, new_geometry = simulate(translated)
    assert_allclose(new_geometry.positions_nm - old_geometry.positions_nm, np.broadcast_to(shift, (4, 3)))
    _assert_spectral_rows_equal(actual, reference, rtol=3e-10, atol=2e-11)


def test_off_grid_explicit_geometry_is_preserved_and_fft_rejects_snapping():
    config = _config()
    positions = from_config(config["geometry"]).positions_nm.copy()
    positions[2] += [0.37, -0.18, 0.29]
    config["geometry"] = {
        "kind": "explicit", "positions_nm": positions.tolist(),
        "radius_nm": [4, 5, 6, 7],
        "site_materials": ["magnetic", "dielectric", "magnetic", "dielectric"],
    }
    rows, metadata, geometry = simulate(config)
    assert_allclose(geometry.positions_nm, positions, rtol=0, atol=0)
    assert metadata["geometry"]["grid_shape"] is None
    assert all(row["mean_extinction_nm2"] > 0 for row in rows)
    config["solver"]["backend"] = "fft"
    with pytest.raises(ValueError, match="no automatic snapping"):
        validate_config(config)


def test_variable_radius_audit_finds_minimum_gap_outside_nearest_centres():
    positions = np.array([[-13, 0, 0], [0, 0, 0], [22, 0, 0], [35, 0, 0]])
    radii = np.array([0.1, 10, 10, 0.1])
    geometry = ParticleGeometry(positions, radii)
    audit = geometry.audit()
    assert audit["minimum_center_distance_nm"] == 13
    assert audit["minimum_surface_gap_nm"] == 2
    assert_allclose(audit["extent_nm"], [48.2, 20, 20])
    overlapping = np.array([[-11, 0, 0], [0, 0, 0], [19, 0, 0], [30, 0, 0]])
    with pytest.raises(ValueError, match="overlap"):
        ParticleGeometry(overlapping, radii)


@pytest.mark.parametrize("backend", ["dense", "fft", "dense_gmres"])
def test_zero_contrast_has_zero_spectrum_and_undefined_normalized_observables(backend):
    config = _config(backend)
    config["materials"] = {"invisible": {"epsilon_diagonal": config["host_index"]**2, "g": 0}}
    config["geometry"].pop("site_materials")
    config["geometry"]["material"] = "invisible"
    rows, metadata, _ = simulate(config)
    for row in rows:
        for name, value in row.items():
            if name == "wavelength_nm":
                continue
            if name.startswith("forward_scattered_") or name == "magnetic_cd_normalized":
                assert value is None
            else:
                assert value == 0
    for record in metadata["records"]:
        assert all(case["iterations"] == 0 for case in record["solves"].values())


def test_material_interpolation_uses_all_complex_components_without_extrapolation():
    spec = {
        "model": "gyrotropic_table", "wavelength_nm": [400, 600],
        "epsilon_diagonal": [[2, 0.2], [4, 0.4]],
        "g": [[0.1, 0.01], [0.3, 0.03]], "magnetization": [0, 0, 0.5],
    }
    expected = (3 + 0.3j) * np.eye(3, dtype=complex)
    expected[0, 1] = 1j * (0.2 + 0.02j) * 0.5
    expected[1, 0] = -expected[0, 1]
    assert_allclose(material_tensor(spec, 500), expected, rtol=1e-14, atol=1e-15)
    assert_allclose(material_tensor(spec, 500, -1), expected.T, rtol=1e-14, atol=1e-15)
    for wavelength in [399.999, 600.001]:
        with pytest.raises(ValueError, match="extrapolation"):
            material_tensor(spec, wavelength)


@pytest.mark.parametrize("spec", [
    {"epsilon_diagonal": [3, -0.1], "g": 0},
    {"epsilon_diagonal": [3, 0.01], "g": [0.2, 0.1]},
    {"model": "gyrotropic_table", "wavelength_nm": [400, 600], "epsilon_diagonal": [[3, 0.01], [3, 0.01]], "g": [[0.2, 0.1], [0.2, 0.1]]},
])
def test_active_materials_rejected_by_hermitian_loss_not_diagonal_imaginary_part(spec):
    with pytest.raises(ValueError, match="active|positive semidefinite"):
        material_tensor(spec, 500)


def test_passivity_checks_even_unsampled_material_table_knots():
    config = _config()
    config["wavelength_nm"] = [550]
    config["materials"]["magnetic"] = {
        "model": "gyrotropic_table", "wavelength_nm": [400, 700],
        "epsilon_diagonal": [[3, -1], [3, 2]], "g": [0, 0],
    }
    # Interpolation at 550 is passive, but the supplied physical table is not.
    with pytest.raises(ValueError, match="active|positive semidefinite"):
        validate_config(config)


@pytest.mark.parametrize("section,field", [
    (None, "wavelenght_nm"), ("geometry", "raduis_nm"),
    ("solver", "rto1"), ("incidence", "directions"),
    ("materials", "epsilon_diag"),
])
def test_unknown_fields_never_silently_change_the_physical_problem(section, field):
    config = _config()
    target = config if section is None else config[section]
    if section == "materials":
        target = target["magnetic"]
    target[field] = 1
    with pytest.raises(ValueError, match="unknown"):
        validate_config(config)


@pytest.mark.parametrize("path,value", [
    (("schema_version",), "1.0"),
    (("wavelength_nm",), []), (("wavelength_nm",), [550, 500]),
    (("wavelength_nm",), [550, 550]), (("wavelength_nm",), [0]),
    (("wavelength_nm",), [np.nan]),
    (("host_index",), 0), (("host_index",), True), (("host_index",), np.inf),
    (("threads",), 0), (("threads",), True),
    (("solver", "rtol"), 0), (("solver", "rtol"), 1),
    (("solver", "restart"), 1.5), (("solver", "maxiter"), True),
    (("solver", "max_memory_mib"), -1), (("solver", "backend"), "snap_fft"),
    (("geometry", "site_materials"), ["magnetic"]),
    (("geometry", "site_materials"), ["missing"] * 4),
    (("incidence", "direction"), [0, 0, 0]),
])
def test_invalid_config_rejected_before_simulation(path, value):
    config = _config()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        validate_config(config)


def test_conflicting_material_assignments_and_geometry_controls_rejected():
    config = _config()
    config["geometry"]["material"] = "magnetic"
    with pytest.raises(ValueError, match="not both"):
        validate_config(config)
    config = _config()
    config["geometry"]["mask"] = "ellipsoid"
    with pytest.raises(ValueError, match="mutually exclusive"):
        validate_config(config)


def test_write_run_artifacts_round_trip_and_hash_actual_saved_bytes(tmp_path):
    config = _config("fft")
    config["description"] = "磁光结构 provenance round-trip"
    output = tmp_path / "run"
    returned = write_run(config, output)
    expected_files = {"config.json", "geometry.npz", "metadata.json", "spectrum.csv"}
    assert {path.name for path in output.iterdir()} == expected_files
    assert json.loads((output / "config.json").read_text(encoding="utf-8")) == config
    metadata = json.loads((output / "metadata.json").read_text(encoding="utf-8"))
    assert metadata == returned
    assert metadata["configuration_sha256"] == hashlib.sha256((output / "config.json").read_bytes()).hexdigest()
    package_dir = Path(runner.__file__).resolve().parent
    assert set(metadata["source_sha256"]) == {path.name for path in package_dir.glob("*.py")}
    for name, expected in metadata["source_sha256"].items():
        assert hashlib.sha256((package_dir / name).read_bytes()).hexdigest() == expected
    with np.load(output / "geometry.npz") as saved:
        expected = from_config(config["geometry"])
        assert_allclose(saved["positions_nm"], expected.positions_nm, rtol=0, atol=0)
        assert_allclose(saved["radii_nm"], expected.radii_nm, rtol=0, atol=0)
    with (output / "spectrum.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    reference, _, _ = simulate(config)
    assert len(rows) == len(reference)
    for row, expected in zip(rows, reference):
        for name, value in expected.items():
            if value is None:
                assert row[name] == ""
            else:
                assert float(row[name]) == value


def test_output_directory_never_overwritten_and_invalid_run_creates_no_output(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    marker = output / "user.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError):
        write_run(_config(), output)
    assert marker.read_text(encoding="utf-8") == "keep"
    config = _config()
    config["solver"]["rtol"] = -1
    with pytest.raises(ValueError):
        write_run(config, tmp_path / "invalid")
    assert not (tmp_path / "invalid").exists()


@pytest.mark.parametrize("duplicate", ["schema_version", "backend"])
def test_cli_rejects_duplicate_json_keys_even_when_last_value_is_valid(tmp_path, duplicate):
    serialized = json.dumps(_config())
    old = f'"{duplicate}":'
    serialized = serialized.replace(old, f'"{duplicate}": "silently overwritten", {old}', 1)
    input_path = tmp_path / "duplicate.json"
    input_path.write_text(serialized, encoding="utf-8")
    output = tmp_path / "output"
    with pytest.raises(ValueError, match="duplicate"):
        runner.main(["--config", str(input_path), "--out", str(output)])
    assert not output.exists()


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_non_json_numbers_in_metadata_rejected_before_creating_artifacts(tmp_path, nonfinite):
    config = _config()
    config["description"] = nonfinite
    output = tmp_path / "output"
    with pytest.raises(ValueError):
        write_run(config, output)
    assert not output.exists()
