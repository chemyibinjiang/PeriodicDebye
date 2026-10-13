#!/usr/bin/env python3
"""Compare completed MOF benchmark measurements and independently check arrays.

Incomplete workers retain their measured samples as evidence, but receive no
median or speedup. Extra directories select newer runs explicitly without
overwriting evidence. Numerical comparisons use raw NPZ arrays, never plots.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics

import numpy as np


def read_json(path, default=None):
    return json.loads(path.read_text(encoding='utf-8')) if path.is_file() else default


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def load_campaigns(paths):
    selected = {}
    for root in paths:
        root = root.resolve()
        if not root.is_dir():
            raise ValueError(f'Campaign directory does not exist: {root}')
        workers = {r['case']: r for r in read_json(root / 'run_summary.json', [])}
        for directory in sorted(p for p in root.iterdir() if p.is_dir()):
            spec = read_json(directory / 'case.json')
            if spec is None:
                continue
            name = spec['name']
            timing = read_json(directory / 'timing.json')
            partial = read_json(directory / 'partial.json', {})
            worker = workers.get(name, {})
            complete = bool(timing and timing.get('status') == 'completed')
            if complete:
                samples = timing['timings_seconds']
                complete = (len(samples) == spec['repeats'] and bool(samples)
                            and np.isfinite(samples).all() and min(samples) > 0
                            and np.isclose(statistics.median(samples), timing['median_seconds'],
                                           rtol=1e-14, atol=0.))
            run = dict(path=str(directory), case=spec,
                       timing_status='completed' if complete else 'incomplete',
                       worker_status=worker.get('status', 'not_recorded'),
                       worker_timeout_seconds=worker.get('timeout_seconds'),
                       partial_samples_seconds=partial.get('seconds', []),
                       partial_completed_repeats=partial.get('completed_repeats', 0),
                       timing=timing if complete else None,
                       has_raw_outputs=(directory / 'outputs.npz').is_file(),
                       configuration=read_json(directory / 'configuration.json'),
                       supersedes=[])
            if name in selected:
                run['supersedes'] = selected[name]['supersedes'] + [selected[name]['path']]
            selected[name] = run
    return selected


def nonforward_mask(arrays, table, length, spec):
    key = table + '__q'
    if key in arrays and arrays[key].shape == (length,):
        return arrays[key] > 0, 'stored radial q'
    keys = [table + '__' + component for component in ('qx', 'qy', 'qz')]
    if all(key in arrays and arrays[key].shape == (length,) for key in keys):
        return np.any(np.stack([arrays[key] for key in keys]) != 0, axis=0), 'stored q vectors'
    if table == 'moments' and length == spec.get('points'):
        return np.linspace(0., spec['qmax'], length) > 0, 'benchmark case radial grid'
    return None, None


def array_errors(before_path, after_path, spec, tolerance, relative_tolerance):
    columns = {}
    with np.load(before_path, allow_pickle=False) as before, np.load(after_path, allow_pickle=False) as after:
        missing_after = sorted(set(before.files) - set(after.files))
        missing_before = sorted(set(after.files) - set(before.files))
        for key in sorted(set(before.files) & set(after.files)):
            reference, actual = before[key], after[key]
            table, column = key.split('__', 1)
            integer = (column in ('h', 'k', 'l', 'multiplicity')
                       or reference.dtype.kind in 'iub' or actual.dtype.kind in 'iub')
            record = dict(shape_before=list(reference.shape), shape_after=list(actual.shape),
                          dtype_before=str(reference.dtype), dtype_after=str(actual.dtype),
                          exact_required=integer, passed=False)
            columns[key] = record
            if reference.shape != actual.shape:
                record['reason'] = 'shape mismatch'
                continue
            if reference.dtype.kind not in 'biufc' or actual.dtype.kind not in 'biufc':
                record['passed'] = bool(np.array_equal(reference, actual))
                record['reason'] = 'non-numeric column; exact equality required'
                continue
            if not np.isfinite(reference).all() or not np.isfinite(actual).all():
                record['reason'] = 'non-finite raw output'
                continue
            difference = np.abs(actual.astype(np.complex128) - reference.astype(np.complex128))
            peak = float(np.max(np.abs(reference), initial=0.))
            max_error = float(np.max(difference, initial=0.))
            scaled_error = max_error / max(1., peak)
            record.update(max_abs_error=max_error, reference_peak=peak,
                          scaled_error=scaled_error,
                          exact_equal=bool(np.array_equal(reference, actual)),
                          passed=bool(np.array_equal(reference, actual)) if integer else scaled_error <= tolerance)
            coordinate = column in ('q', 'qx', 'qy', 'qz', 'h', 'k', 'l', 'multiplicity')
            amplitude = ('amplitude' in column and 'variance' not in column) or column in ('mean_real', 'mean_imag')
            if reference.ndim == 1 and not coordinate and not amplitude:
                mask, origin = nonforward_mask(before, table, len(reference), spec)
                if mask is not None:
                    nf_peak = float(np.max(np.abs(reference[mask]), initial=0.))
                    nf_error = float(np.max(difference[mask], initial=0.))
                    selected = mask & (np.abs(reference) > 1e-8 * nf_peak)
                    relative = (float(np.max(difference[selected] / np.abs(reference[selected])))
                                if np.any(selected) else None)
                    nf_scaled = nf_error / max(1., nf_peak)
                    record['nonforward_intensity'] = dict(mask_source=origin, points=int(mask.sum()),
                        reference_peak=nf_peak, max_abs_error=nf_error, scaled_error=nf_scaled,
                        significant_threshold=1e-8 * nf_peak, significant_points=int(selected.sum()),
                        max_relative_error_above_threshold=relative,
                        passed=nf_scaled <= tolerance and (relative is None or relative <= relative_tolerance))
                    record['passed'] = record['passed'] and record['nonforward_intensity']['passed']
    return dict(missing_after=missing_after, missing_before=missing_before, columns=columns,
                passed=not missing_after and not missing_before and bool(columns)
                       and all(record['passed'] for record in columns.values()))


def compact_run(run):
    if run is None:
        return None
    timing = run['timing']
    result = {key: value for key, value in run.items() if key not in ('timing', 'configuration')}
    if timing:
        for key in ('version', 'median_seconds', 'min_seconds', 'max_seconds', 'timings_seconds',
                    'retained_cells', 'conceptual_host_atoms', 'edge_nm', 'table_rows', 'actual_methods',
                    'peak_sampled_rss_bytes', 'initial_rss_bytes', 'incremental_sampled_rss_bytes',
                    'windows_process_peak_wset_bytes', 'timing_scope', 'memory_scope',
                    'source_sha256', 'source_cif_sha256', 'environment'):
            if key in timing:
                result[key] = timing[key]
    return result


def compare_campaigns(before, after, tolerance, relative_tolerance):
    records = []
    for name in sorted(set(before) | set(after)):
        old, new = before.get(name), after.get(name)
        spec = (new or old)['case']
        record = dict(case=name, kind=spec['kind'], specification=spec,
                      before=compact_run(old), after=compact_run(new),
                      raw_output_comparison=None, accuracy_passed=None, speedup=None)
        records.append(record)
        if old is None or new is None:
            record['speedup_unavailable_reason'] = 'case absent from one campaign'
            continue
        same_case = ({k: v for k, v in old['case'].items() if k != 'repeats'}
                     == {k: v for k, v in new['case'].items() if k != 'repeats'})
        same_config = old['configuration'] == new['configuration'] and old['configuration'] is not None
        record.update(case_specification_matches=same_case, configuration_matches=same_config)
        if old['has_raw_outputs'] and new['has_raw_outputs']:
            comparison = array_errors(Path(old['path']) / 'outputs.npz',
                                      Path(new['path']) / 'outputs.npz', spec, tolerance, relative_tolerance)
            record['raw_output_comparison'] = comparison
            record['accuracy_passed'] = comparison['passed'] and same_case and same_config
        completed = old['timing'] is not None and new['timing'] is not None
        if completed:
            source_matches = old['timing']['source_cif_sha256'] == new['timing']['source_cif_sha256']
            scope_matches = old['timing']['timing_scope'] == new['timing']['timing_scope']
            threads_match = (old['timing']['environment']['torch_threads']
                             == new['timing']['environment']['torch_threads'])
            record.update(source_cif_matches=source_matches, timing_scope_matches=scope_matches,
                          thread_count_matches=threads_match)
            comparable = source_matches and scope_matches and threads_match
        else:
            comparable = False
        if completed and comparable and record['accuracy_passed']:
            record['speedup'] = old['timing']['median_seconds'] / new['timing']['median_seconds']
        elif not completed:
            record['speedup_unavailable_reason'] = 'one or both workers lack a completed timing record'
        elif not comparable:
            record['speedup_unavailable_reason'] = 'source, measurement scope or thread count differs'
        else:
            record['speedup_unavailable_reason'] = 'paired outputs or accuracy/configuration verification unavailable or failed'
        columns = (record['raw_output_comparison'] or {}).get('columns', {})
        record['worst_scaled_error'] = max((c['scaled_error'] for c in columns.values()
                                             if 'scaled_error' in c), default=None)
        nonforward = [c['nonforward_intensity'] for c in columns.values() if 'nonforward_intensity' in c]
        record['worst_nonforward_scaled_error'] = max((c['scaled_error'] for c in nonforward), default=None)
        record['worst_nonforward_relative_error_above_threshold'] = max(
            (c['max_relative_error_above_threshold'] for c in nonforward
             if c['max_relative_error_above_threshold'] is not None), default=None)
    return records


def write_csv(path, records):
    fields = ['case', 'kind', 'cells_per_edge', 'edge_nm', 'conceptual_host_atoms', 'qmax', 'points',
              'before_status', 'after_status', 'before_seconds', 'after_seconds', 'speedup',
              'accuracy_passed', 'worst_scaled_error', 'worst_nonforward_scaled_error',
              'worst_nonforward_relative_error_above_threshold', 'before_partial_samples_seconds',
              'after_partial_samples_seconds', 'before_peak_sampled_rss_MiB', 'after_peak_sampled_rss_MiB',
              'before_incremental_sampled_rss_MiB', 'after_incremental_sampled_rss_MiB',
              'before_directory', 'after_directory', 'speedup_unavailable_reason']
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for record in records:
            spec = record['specification']
            row = {key: record.get(key) for key in fields}
            row.update({key: spec.get(key) for key in ('cells_per_edge', 'qmax', 'points')})
            for side in ('before', 'after'):
                run = record.get(side) or {}
                row[side + '_status'] = run.get('timing_status', 'absent')
                row[side + '_seconds'] = run.get('median_seconds')
                row[side + '_partial_samples_seconds'] = json.dumps(run.get('partial_samples_seconds', []))
                row[side + '_directory'] = run.get('path')
                for memory in ('peak_sampled_rss', 'incremental_sampled_rss'):
                    value = run.get(memory + '_bytes')
                    row[side + '_' + memory + '_MiB'] = value / 1024**2 if value is not None else None
                if run.get('edge_nm') is not None:
                    row['edge_nm'] = run['edge_nm']
                    row['conceptual_host_atoms'] = run['conceptual_host_atoms']
            writer.writerow(row)


def plot_comparison(out, records):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.labelsize': 10,
                         'savefig.dpi': 180, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))
    panels = [('periodic', 'Periodic single crystal', 'qmax'),
              ('bulk_powder', 'Bulk powder diffraction', 'qmax'),
              ('finite', 'Finite single crystal · 2,049 q vectors', 'cells_per_edge'),
              ('powder', 'Finite powder · 121 q values', 'cells_per_edge')]
    for axis, (kind, title, dimension) in zip(axes.flat, panels):
        cases = sorted((r for r in records if r['kind'] == kind),
                       key=lambda r: r['specification'][dimension])
        x_values = [r['specification'][dimension] * (2.5832 if dimension == 'cells_per_edge' else 1)
                    for r in cases]
        incomplete = []
        for side, label, color, marker in [('before', 'Before', '#ad5729', 'o'),
                                            ('after', 'Optimized', '#19669a', 's')]:
            x, medians, low, high = [], [], [], []
            for coordinate, record in zip(x_values, cases):
                run = record.get(side)
                if run is None or run.get('median_seconds') is None:
                    if run is not None:
                        tag = f"n={record['specification'][dimension]}" if dimension == 'cells_per_edge' else f"qmax={record['specification'][dimension]}"
                        incomplete.append(f'{label} {tag}')
                    continue
                value = run['median_seconds']
                x.append(coordinate)
                medians.append(value)
                low.append(value - run['min_seconds'])
                high.append(run['max_seconds'] - value)
            if x:
                axis.errorbar(x, medians, yerr=[low, high], marker=marker, linewidth=1.6,
                              capsize=3, color=color, label=label, markersize=5)
        for coordinate, record in zip(x_values, cases):
            if record.get('speedup') is not None:
                axis.annotate(f"{record['speedup']:.2g}×", (coordinate, record['after']['median_seconds']),
                              xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9,
                              color='#19669a')
        axis.set_title(title, loc='left', pad=12)
        axis.set_yscale('log')
        axis.set_ylabel('Median calculation time (s)')
        axis.set_xticks(x_values)
        if dimension == 'cells_per_edge':
            axis.set_xticklabels([f"{x:.1f}\n(n={r['specification'][dimension]})"
                                  for x, r in zip(x_values, cases)])
            axis.set_xlabel('Cube edge (nm)')
        else:
            axis.set_xlabel(r'$q_{\mathrm{max}}$ (Å$^{-1}$)')
        axis.grid(axis='y', which='both', alpha=.18)
        axis.spines[['top', 'right']].set_visible(False)
        axis.margins(x=.15, y=.27)
        if incomplete:
            axis.text(.02, .97, 'Incomplete: ' + ', '.join(incomplete), transform=axis.transAxes,
                      va='top', fontsize=8, color='#6e4e36',
                      bbox={'facecolor': 'white', 'alpha': .9, 'edgecolor': 'none', 'pad': 2})
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(.98, .99), frameon=False, ncol=2)
    fig.suptitle('MOF-5 performance · identical observables in float64', x=.06, ha='left', y=.98, fontsize=15)
    fig.text(.06, .018, 'Points: medians of complete repeats; whiskers: measured minimum–maximum. '
             'Speedups require paired outputs passing accuracy checks.\n'
             'Incomplete workers have no invented median. Finite powder q spacing is a timing workload, not a resolution guarantee.',
             fontsize=8.5, color='#444444')
    fig.subplots_adjust(left=.075, right=.97, bottom=.15, top=.88, hspace=.52, wspace=.26)
    fig.savefig(out / 'performance_comparison.png')
    fig.savefig(out / 'performance_comparison.pdf')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, required=True)
    parser.add_argument('--after', type=Path, required=True)
    parser.add_argument('--before-extra', type=Path, action='append', default=[],
                        help='Additional fresh campaigns; matching cases explicitly supersede earlier ones.')
    parser.add_argument('--after-extra', type=Path, action='append', default=[])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tolerance', type=float, default=2e-10)
    parser.add_argument('--relative-tolerance', type=float, default=2e-7)
    parser.add_argument('--no-plot', action='store_true')
    args = parser.parse_args()
    if (not np.isfinite([args.tolerance, args.relative_tolerance]).all()
            or min(args.tolerance, args.relative_tolerance) < 0):
        parser.error('A finite nonnegative tolerance is required.')
    before_paths, after_paths = [args.before] + args.before_extra, [args.after] + args.after_extra
    records = compare_campaigns(load_campaigns(before_paths), load_campaigns(after_paths),
                                args.tolerance, args.relative_tolerance)
    compared = [r for r in records if r['raw_output_comparison'] is not None]
    report = dict(before_campaigns=[str(p.resolve()) for p in before_paths],
                  after_campaigns=[str(p.resolve()) for p in after_paths], tolerance=args.tolerance,
                  significant_nonforward_relative_tolerance=args.relative_tolerance,
                  scaled_error_definition='max(abs(after-before))/max(1,max(abs(before)))',
                  nonforward_error_definition='Same formula after excluding q=0; relative errors use |before| > 1e-8 of nonforward peak.',
                  relative_errors_are_diagnostic=False,
                  timing_policy='Use completed timing.json only. Partial samples remain explicitly incomplete; no extrapolated medians or speedups.',
                  memory_policy='Sampled process RSS over repeated runs; includes imports and possibly previous result arrays; not a hard bound.',
                  paired_raw_output_cases=len(compared),
                  all_paired_accuracy_passed=all(r['accuracy_passed'] for r in compared) if compared else None,
                  eligible_speedup_cases=sum(r['speedup'] is not None for r in records), cases=records)
    args.out.mkdir(parents=True, exist_ok=True)
    dump(args.out / 'performance_comparison.json', report)
    write_csv(args.out / 'performance_comparison.csv', records)
    if not args.no_plot:
        plot_comparison(args.out, records)
    print(json.dumps(dict(cases=len(records), paired=len(compared),
                         all_paired_accuracy_passed=report['all_paired_accuracy_passed'],
                         eligible_speedups=report['eligible_speedup_cases'], output=str(args.out.resolve()))))
    if compared and not report['all_paired_accuracy_passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
