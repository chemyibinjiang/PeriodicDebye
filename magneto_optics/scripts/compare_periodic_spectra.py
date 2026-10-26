"""Compare complete periodic spectra using finite absolute errors at signal zeros."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_rows(path):
    with (path / 'spectrum.csv').open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def compare(actual_path, reference_path):
    actual, reference = read_rows(actual_path), read_rows(reference_path)
    if len(actual) != len(reference) or actual[0].keys() != reference[0].keys():
        raise ValueError('spectra must have the same rows and columns')
    power_error = angle_error = 0.
    nulls_match = True
    for row, expected in zip(actual, reference):
        if row['wavelength_nm'] != expected['wavelength_nm'] or row['propagating_orders_per_side'] != expected['propagating_orders_per_side']:
            raise ValueError('spectra have different wavelengths or open orders')
        for key, value in expected.items():
            if key in ('wavelength_nm', 'propagating_orders_per_side', 'max_relative_residual'):
                continue
            if not value or not row[key]:
                nulls_match &= value == row[key]
                continue
            difference = float(row[key]) - float(value)
            if 'rotation_deg' in key:
                period = 90. if 'magnetic_odd' in key else 180.
                difference = (difference + period / 2) % period - period / 2
            if '_deg' in key:
                angle_error = max(angle_error, abs(difference))
            else:
                power_error = max(power_error, abs(difference))
    metadata = [json.loads((path / 'metadata.json').read_text(encoding='utf-8')) for path in (actual_path, reference_path)]
    jones_absolute = 0.
    for record, expected in zip(metadata[0]['records'], metadata[1]['records']):
        for sign in ('Mplus', 'Mminus'):
            for name in ('transmission', 'reflection'):
                values = np.asarray(record['jones'][sign][name])
                targets = np.asarray(expected['jones'][sign][name])
                difference = values - targets
                jones_absolute = max(jones_absolute, float(np.max(np.hypot(difference[..., 0], difference[..., 1]))))
    result = {'wavelength_count': len(actual), 'actual': str(actual_path), 'reference': str(reference_path),
              'max_absolute_dimensionless_output_error': power_error,
              'max_absolute_angle_error_deg': angle_error, 'max_absolute_jones_error': jones_absolute,
              'undefined_values_match': bool(nulls_match),
              'metric': 'Absolute power/fraction and Jones errors retain meaning at zero signals. Rotation differences are wrapped modulo180deg (magnetic-odd modulo90deg).',
              'limits': {'dimensionless': 1e-8, 'jones': 1e-8, 'angle_deg': 1e-6}}
    result['passed'] = bool(nulls_match and power_error < 1e-8 and jones_absolute < 1e-8 and angle_error < 1e-6)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError('use a new comparison output file')
    result = compare(args.actual, args.reference)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    if not result['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
