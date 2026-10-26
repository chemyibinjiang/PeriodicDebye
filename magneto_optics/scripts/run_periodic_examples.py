"""Run all committed synthetic periodic examples and record actual CLI timings."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    records = []
    for name in ('square_sheet', 'two_site_sheet', 'diffractive_sheet', 'lossless_sheet'):
        config = ROOT / 'examples' / 'periodic' / f'{name}.json'
        destination = args.out / name
        started = perf_counter()
        completed = subprocess.run([sys.executable, str(ROOT / 'scripts' / 'run_periodic.py'),
                                    '--config', str(config), '--out', str(destination)],
                                   capture_output=True, text=True, encoding='utf-8', check=True)
        wall_seconds = perf_counter() - started
        metadata = json.loads((destination / 'metadata.json').read_text(encoding='utf-8'))
        records.append({'example': name, 'end_to_end_seconds': wall_seconds,
                        'calculation_seconds': metadata['elapsed_seconds'],
                        'wavelength_count': len(metadata['records']),
                        'linear_system_count': 4 * len(metadata['records']),
                        'basis_spheres': metadata['geometry']['particle_count_per_cell'],
                        'max_energy_balance_residual': metadata['max_energy_balance_residual'],
                        'input_sha256': hashlib.sha256(config.read_bytes()).hexdigest()})
        (args.out / f'{name}.log').write_text(completed.stdout, encoding='utf-8')
        print(json.dumps(records[-1]), flush=True)
    report = {'python_executable': sys.executable, 'timing_scope': 'Single sequential CLI run per example. End-to-end includes Python/imports, validation and file writing. Calculation excludes startup and output writing. Four x/y and +/-M linear systems per wavelength; circular channels use linearity.',
              'runs': records}
    (args.out / 'timings.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
