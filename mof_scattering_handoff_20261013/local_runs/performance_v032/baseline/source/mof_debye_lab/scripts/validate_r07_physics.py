"""Run independent R07 physics checks and export equations/numerical evidence.

Run from the project root with:
    python scripts/validate_r07_physics.py --out ../local_runs/r07_thorough/physics

This validation concerns correctness, not performance. It writes only the
requested output directory, leaving historical baseline records untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    output = args.out.resolve()
    output.mkdir(parents=True, exist_ok=True)
    xml_path = output / 'pytest.xml'
    command = [sys.executable, '-m', 'pytest', 'tests/test_physics_stress.py', '-q',
               '-o', 'junit_family=xunit1', f'--junitxml={xml_path}']
    process = subprocess.run(command, cwd=project, capture_output=True, text=True,
                             encoding='utf-8', env={**os.environ, 'PYTHONIOENCODING': 'utf-8'})
    transcript = process.stdout + process.stderr
    (output / 'pytest.txt').write_text(transcript, encoding='utf-8')
    print(transcript, end='')
    records = []
    if xml_path.is_file():
        for case in ET.parse(xml_path).getroot().iter('testcase'):
            record = {'test': case.attrib['name'],
                      'passed': case.find('failure') is None and case.find('error') is None
                                and case.find('skipped') is None}
            for prop in case.findall('./properties/property'):
                if prop.attrib.get('name') == 'physics_record':
                    record.update(json.loads(prop.attrib['value']))
            records.append(record)
    report = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Independent numerical physics validation; no performance claims.',
        passed=process.returncode == 0 and len(records) == 13 and all(r['passed'] for r in records),
        test_count=len(records), command=command, working_directory=str(project),
        numerical_conventions={
            'scattering': 'Neutral real IT92 elastic factors, positions angstrom, q inverse angstrom.',
            'scaled_error': 'max(abs(actual-expected))/max(1,max(abs(expected))).',
            'sampled_mean_error': 'Complex amplitude error divided by sqrt(exact physical variance/sample count).',
            'conditioning': 'Template selected first; position conditioned separately at fixed template, preserving number weights.',
            'limits': 'Synthetic geometric fixtures validate the model equations, not a force field, physical MOF loading, or experimental accuracy.'},
        cases=records)
    (output / 'physics_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'test_count': len(records),
                      'report': str(output / 'physics_report.json')}, indent=2))
    return 0 if report['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
