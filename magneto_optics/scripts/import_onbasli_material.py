"""Reproduce a passive Ce:YIG material table from Onbasli et al.'s author XLS.

Install the optional data dependency with: pip install -e 'magneto_optics[data]'.
No dependencies are installed by this script. Raw fitted losses are retained;
an active interpolated tensor is rejected, never clipped or refitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "material_data" / "onbasli_2016" / "source.xls"
METADATA = SOURCE.with_name("source_metadata.json")
EXPECTED_SHA256 = "ff770dd764391e77ce61d650abb2c783d600f2227e1173d1d1c191e5278fd9ce"
HC_EV_NM = 1239.841984


def _source_block(book, sheet_name, column, orientation, low, high):
    sheet = book.sheet_by_name(sheet_name)
    if orientation not in str(sheet.cell_value(0, column)):
        raise ValueError(f"Unexpected orientation header in {sheet_name}")
    rows = []
    for row in range(2, sheet.nrows):
        values = sheet.row_values(row, column, column+3)
        if all(value == "" for value in values):
            continue
        if len(values) != 3 or any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in values):
            raise ValueError(f"Non-numeric source record in {sheet_name}, Excel row {row+1}")
        rows.append(values)
    values = np.asarray(rows, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all() or np.any(values[:, 0] <= 0):
        raise ValueError(f"Invalid numeric source block in {sheet_name}")
    values = values[np.argsort(values[:, 0], kind="stable")]
    energy = values[:, 0]
    if low < energy[0] or high > energy[-1]:
        raise ValueError(f"Extrapolation forbidden: {sheet_name} covers {energy[0]:.8g}..{energy[-1]:.8g} eV")
    # Preserve bracketing knots. Unused out-of-band duplicate fits are not averaged.
    lower = energy[np.searchsorted(energy, low, side="right")-1]
    upper = energy[np.searchsorted(energy, high, side="left")]
    selected = values[(energy >= lower) & (energy <= upper)]
    energies, first, counts = np.unique(selected[:, 0], return_index=True, return_counts=True)
    for start, count in zip(first, counts):
        if not np.all(selected[start:start+count, 1:] == selected[start, 1:]):
            raise ValueError(f"Conflicting duplicate energy {selected[start, 0]} eV in used {sheet_name} interval")
    return energies, selected[first, 1]+1j*selected[first, 2], {
        "sheet": sheet_name, "columns_zero_based": [column, column+1, column+2],
        "source_rows": len(values), "retained_knots": len(energies),
        "identical_duplicates_removed_in_used_interval": int(len(selected)-len(energies)),
        "source_energy_domain_ev": [float(energy[0]), float(energy[-1])],
        "used_bracketing_energy_domain_ev": [float(lower), float(upper)],
    }


def build_table(source_path=SOURCE, orientation="110", wavelength_nm=None):
    """Return a runner-compatible material dict; wavelengths must increase strictly."""
    if orientation not in ("100", "110", "111"):
        raise ValueError("orientation must be '100', '110', or '111'")
    raw = np.asarray(np.arange(1300., 1601.) if wavelength_nm is None else wavelength_nm)
    if raw.dtype.kind not in "fiu" or (not isinstance(wavelength_nm, np.ndarray) and wavelength_nm is not None
            and any(isinstance(v, (bool, np.bool_)) for v in np.asarray(wavelength_nm, dtype=object).flat)):
        raise ValueError("wavelength_nm must contain real numbers, not booleans or strings")
    wavelengths = np.asarray(raw, dtype=float)
    if wavelengths.ndim != 1 or len(wavelengths) < 2 or not np.isfinite(wavelengths).all() or np.any(wavelengths <= 0) or np.any(np.diff(wavelengths) <= 0):
        raise ValueError("wavelength_nm must have at least two finite positive strictly increasing knots")
    source = Path(source_path).read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    if digest != EXPECTED_SHA256 or metadata.get("sha256") != EXPECTED_SHA256:
        raise ValueError("Author XLS SHA-256 mismatch; refusing changed or unidentified source data")
    try:
        import xlrd
    except ImportError as error:
        raise RuntimeError("Reading author XLS requires xlrd; install the optional data extra: pip install -e 'magneto_optics[data]'") from error
    book = xlrd.open_workbook(file_contents=source)
    query = HC_EV_NM/wavelengths
    column = {"100": 0, "110": 3, "111": 6}[orientation]
    blocks = [_source_block(book, name, column, orientation, float(query.min()), float(query.max()))
              for name in ("Epsilon1 CeYIG films", "Epsilon2")]
    interpolate = lambda energy: (np.interp(energy, blocks[0][0], blocks[0][1]),
                                  -np.interp(energy, blocks[1][0], blocks[1][1]))
    # A union of both source grids checks every linear segment, not just the 1 nm output samples.
    audit = np.unique(np.concatenate((query, *[b[0][(b[0] >= query.min()) & (b[0] <= query.max())] for b in blocks])))
    diagonal_audit, g_audit = interpolate(audit)
    minimum_loss = diagonal_audit.imag-np.abs(g_audit.imag)
    if np.any(minimum_loss < 0):
        bad = int(np.argmin(minimum_loss))
        raise ValueError(f"Ce:YIG({orientation}) fitted tensor violates passivity near {HC_EV_NM/audit[bad]:.6f} nm: "
                         f"minimum Hermitian loss eigenvalue={minimum_loss[bad]:.9g}. Raw data were not clipped or refitted; no table exported.")
    diagonal, coupling = interpolate(query)
    encode = lambda value: np.column_stack((value.real, value.imag)).tolist()
    return {"model": "gyrotropic_table", "label": f"Ce:YIG on GGG({orientation}); Onbasli et al. 2016 author fitted tensor",
            "wavelength_nm": wavelengths.tolist(), "epsilon_diagonal": encode(diagonal), "g": encode(coupling),
            "magnetization": [0., 0., 1.], "provenance": {
                **metadata, "source_sha256": digest, "substrate_orientation": orientation,
                "source_blocks": [block[2] for block in blocks], "hc_ev_nm": HC_EV_NM,
                "interpolation": "Linear complex-component interpolation in source photon energy E=hc/lambda; no extrapolation. Runner interpolation between exported knots is in wavelength.",
                "sign_mapping": "Publication epsilon_xy=-i*epsilon2; repository epsilon_xy=+i*g for m_z=+1; g=-epsilon2.",
                "signed_magnetization_reference": "m_z=+1 means the source fitted tensor orientation/sign; reversing m_z reverses g. No absolute laboratory field-axis calibration is inferred.",
                "harmonic_convention": "Source positive Im(epsilon1) is interpreted as passive exp(-i omega t); article does not explicitly state its harmonic convention.",
                "loss_policy": "Raw fitted epsilon1 and epsilon2 retained, with no clipping, added loss, averaging of conflicting knots, or refitting from other crystal orientations.",
                "passivity_audit_knot_count": len(audit), "minimum_loss_eigenvalue": float(minimum_loss.min()),
            }}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=SOURCE)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--orientation", choices=("110", "111", "100"), default="110")
    args = parser.parse_args()
    table = build_table(args.source, args.orientation)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8", newline="\n") as output:
        output.write(json.dumps(table, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
    print(f"Exported {len(table['wavelength_nm'])} passive knots to {args.out}")


if __name__ == "__main__":
    main()
