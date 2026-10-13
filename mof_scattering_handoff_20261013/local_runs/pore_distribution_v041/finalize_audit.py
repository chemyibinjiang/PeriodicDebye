"""Record this development change after checking preserved historical evidence."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[1]
PROJECT = ROOT/'mof_debye_lab'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def manifest(path):
    return {line.split('  ', 1)[1]:line.split('  ', 1)[0]
            for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def main():
    previous = manifest(RUN/'baseline'/'DEVELOPMENT_MANIFEST_v040.sha256')
    allowed = {'README.md', 'SOURCE_AUDIT.json', 'mof_debye_lab/README.md',
        'mof_debye_lab/MODE_GUIDE.md', 'mof_debye_lab/LARGE_PXRD_GUIDE.md',
        'mof_debye_lab/pyproject.toml', 'mof_debye_lab/src/mofdebye/__init__.py',
        'mof_debye_lab/src/mofdebye/runner.py'}
    changed = []
    for relative, digest in previous.items():
        actual = sha(ROOT/relative)
        if actual != digest:
            if relative not in allowed:
                raise RuntimeError(f'Unexpected modification of historical file: {relative}')
            if relative != 'SOURCE_AUDIT.json':
                changed.append(dict(path=relative, v040_sha256=digest, current_sha256=actual))
    audit = json.loads((ROOT/'SOURCE_AUDIT.json').read_text(encoding='utf-8'))
    original = manifest(ROOT/'MANIFEST.sha256')
    for relative in audit['preserved_evidence_files']:
        if sha(ROOT/relative) != original[relative]:
            raise RuntimeError(f'Original protected evidence changed: {relative}')
    archive = ROOT.parent/'MOF_Scattering_Local_Agent_Handoff_20261013.zip'
    assert sha(archive) == audit['baseline_archive_sha256']
    suites = list(ET.parse(RUN/'pytest_final.xml').getroot().iter('testsuite'))
    totals = {key:sum(int(s.attrib.get(key, 0)) for s in suites)
              for key in ('tests', 'failures', 'errors', 'skipped')}
    assert totals['failures'] == totals['errors'] == totals['skipped'] == 0, totals
    validation = dict(report='PORE_DISTRIBUTION_REPORT.md', source_version='0.4.1',
        tests_passed=totals['tests'], tests_skipped=totals['skipped'],
        new_cases_since_v040=totals['tests']-857,
        logs='local_runs/pore_distribution_v041/',
        archived_original_evidence_unchanged=len(audit['preserved_evidence_files']),
        benchmark_device='cpu', benchmark_geometry_threads=6,
        scope='Independent finite powder pore distributions, exact analytic and sampled spherical confinement; '
              'body-frame sampling equivalence; local integration SEM separate from radial approximation bounds.')
    if audit.get('validation', {}).get('source_version') != '0.4.1':
        audit['validation_v040'] = audit['validation']
    audit['validation'] = validation
    audit['scope'] = 'v0.4.1 independent pore distributions for finite radial PXRD, synthetic physical references and 100 nm runs'
    audit['changed_since_v040_excluding_this_audit'] = changed
    excluded = {'__pycache__', '.pytest_cache'}
    project_files = {p.relative_to(ROOT).as_posix() for p in PROJECT.rglob('*')
                     if p.is_file() and not (set(p.parts)&excluded)
                     and not any(part.endswith('.egg-info') for part in p.parts)}
    audit['new_project_files_since_v040'] = sorted(project_files-set(previous))
    audit['v040_source_snapshot'] = 'local_runs/pore_distribution_v041/baseline/v040_source_snapshot.zip'
    audit['v040_source_snapshot_sha256'] = sha(ROOT/audit['v040_source_snapshot'])
    audit['v040_development_manifest'] = 'local_runs/pore_distribution_v041/baseline/DEVELOPMENT_MANIFEST_v040.sha256'
    audit['v040_verified_manifest_files'] = len(previous)
    def refresh(value):
        if isinstance(value, dict):
            if 'current_sha256' in value and 'path' in value:
                value['current_sha256'] = sha(ROOT/value['path'])
            for item in value.values():
                refresh(item)
        elif isinstance(value, list):
            for item in value:
                refresh(item)
    refresh(audit)
    (ROOT/'SOURCE_AUDIT.json').write_text(json.dumps(audit, indent=2)+'\n', encoding='utf-8')
    (RUN/'validation_summary.json').write_text(json.dumps(validation, indent=2)+'\n', encoding='utf-8')
    paths = set(previous)|project_files|{'PORE_DISTRIBUTION_REPORT.md'}
    paths.update(p.relative_to(ROOT).as_posix() for p in RUN.rglob('*')
                 if p.is_file() and not (set(p.parts)&excluded))
    paths.discard('DEVELOPMENT_MANIFEST.sha256')
    lines = [f'{sha(ROOT/relative)}  {relative}' for relative in sorted(paths)]
    (ROOT/'DEVELOPMENT_MANIFEST.sha256').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(dict(manifest_files=len(paths), validation=validation,
                         changed_historical_files=changed), indent=2))


if __name__ == '__main__':
    main()
