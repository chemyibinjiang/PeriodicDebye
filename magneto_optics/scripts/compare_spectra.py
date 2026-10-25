"""Compare saved spectra with a nonzero extinction scale for vanishing CD."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def read(path):
    with path.open(encoding='utf-8', newline='') as handle:
        rows = list(csv.DictReader(handle))
    return {key: np.array([float(row[key]) if row[key] else np.nan for row in rows]) for key in rows[0]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError('use a new comparison output')
    actual, reference = read(args.actual), read(args.reference)
    if actual.keys() != reference.keys() or not np.array_equal(actual['wavelength_nm'], reference['wavelength_nm']):
        raise ValueError('spectrum columns and wavelength grids must agree')
    scale = float(np.max(np.abs(reference['mean_extinction_nm2'])))
    if not scale > 0:
        raise ValueError('comparison needs a nonzero extinction scale')
    errors = {}
    relative = {}
    for key, values in reference.items():
        if key.startswith(('extinction_nm2', 'absorption_nm2', 'scattering_nm2')):
            errors[key] = float(np.max(abs(actual[key] - values)) / scale)
            mask = abs(values) >= .01 * scale
            if np.any(mask):
                relative[key] = float(np.max(abs(actual[key][mask] - values[mask]) / abs(values[mask])))
    cd_errors = {key: float(np.max(abs(actual[key] - reference[key])) / scale)
                 for key in ['magnetic_cd_extinction_nm2', 'even_cd_extinction_nm2']}
    normalized = float(np.nanmax(abs(actual['magnetic_cd_normalized'] - reference['magnetic_cd_normalized'])))
    maximum = max([*errors.values(), *relative.values(), *cd_errors.values(), normalized])
    result = {
        'actual': str(args.actual), 'reference': str(args.reference),
        'wavelength_count': len(reference['wavelength_nm']),
        'intensity_scale': 'maximum mean extinction of the reference four channels',
        'cross_section_error_over_peak_extinction': errors,
        'relative_cross_section_error_where_at_least_1percent_peak_extinction': relative,
        'cd_absolute_error_over_peak_extinction': cd_errors,
        'normalized_magnetic_cd_max_absolute_difference': normalized,
        'maximum_reported_error': maximum,
        'target_fraction': .001, 'passed': maximum <= .001,
        'scope': 'finite particle-CDA numerical comparison; near-zero CD uses an absolute extinction scale; no model/multipole or between-wavelength guarantee',
    }
    args.out.write_bytes((json.dumps(result, indent=2, allow_nan=False) + '\n').encode('utf-8'))
    print(json.dumps({'passed': result['passed'], 'maximum_reported_error': maximum, 'wavelengths': result['wavelength_count']}))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
