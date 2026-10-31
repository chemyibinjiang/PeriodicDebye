"""Diagnostic gap scan: finite glass substrate, air gap, and an ideal PEC mirror.

Reuses the full grating S matrix at each gap, including its evanescent orders.
This monochromatic calculation does not average unresolved substrate fringes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
from threadpoolctl import threadpool_limits
from magnetooptics.grating_design import relative_rotation, transmission_metrics
from magnetooptics.grating_mirror import grating_mirror_return
from scan_grating import Study, geometry, save_json, save_csv, MATERIAL


def _field_metrics(matrix, axis):
    field = matrix[:, axis]
    rotation, ellipticity = relative_rotation(field, axis)
    power = float(np.vdot(field, field).real)
    qualified = bool(rotation is not None and ellipticity is not None
                     and power >= .25 and abs(ellipticity) <= 5.)
    return dict(rotation_deg=rotation, ellipticity_deg=ellipticity,
                zero_order_power=power, cross_power=float(abs(field[1-axis])**2),
                fom_deg=0. if rotation is None else float(abs(rotation)*np.sqrt(power)),
                qualified=qualified)


def _return_metrics(result, axis):
    full = _field_metrics(result['returned_jones'], axis)
    first = _field_metrics(result['first_return_jones'], axis)
    return dict(gap_nm=result['gap_nm'],
        **{'full_'+key: value for key, value in full.items()},
        **{'first_'+key: value for key, value in first.items()},
        total_returned_power=float(result['returned_power'][axis]),
        material_absorbed_power=float(result['material_absorbed_power'][axis]),
        mirror_absorbed_power=float(result['mirror_absorbed_power'][axis]),
        feedback_relative_residual=result['feedback_relative_residual'])


def _axis_difference(a, b):
    return None if a is None or b is None else float((a-b+90)%180-90)


def _comparison(returned, single):
    twice = None if single['rotation_deg'] is None else float((2*single['rotation_deg']+90)%180-90)
    return dict(twice_finite_substrate_single_pass_rotation_deg=twice,
        first_return_minus_twice_single_pass_deg=_axis_difference(returned['first_rotation_deg'], twice),
        full_return_minus_twice_single_pass_deg=_axis_difference(returned['full_rotation_deg'], twice),
        note='Ellipse axes compared modulo 180 degrees. An anisotropic grating and coherent cavity need not double the bare transmission angle.')


def _plot(output, rows, best, single, wavelength, substrate_nm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    gaps = [row['gap_nm'] for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for prefix, label, color in [('full_', 'Full coherent return', '#205c94'),
                                 ('first_', 'First return only', '#bc6f24')]:
        axes[0].plot(gaps, [row[prefix+'rotation_deg'] for row in rows], label=label, color=color)
        axes[1].plot(gaps, [row[prefix+'ellipticity_deg'] for row in rows], label=label, color=color)
        axes[2].plot(gaps, [row[prefix+'zero_order_power'] for row in rows], label=label+' zero order', color=color)
    twice = _comparison(best, single)['twice_finite_substrate_single_pass_rotation_deg']
    if twice is not None:
        axes[0].axhline(twice, color='#555555', ls='--', label='Twice bare single-pass angle (comparison only)')
    axes[2].plot(gaps, [row['total_returned_power'] for row in rows], '--', color='#28754b', label='Full return, all orders')
    axes[2].plot(gaps, [row['material_absorbed_power'] for row in rows], color='#a83939', label='Material absorption')
    for ax in axes:
        ax.axvline(best['gap_nm'], color='#333333', lw=.8, ls=':')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set(ylabel='Rotation relative to input (deg)')
    axes[1].set(ylabel='Ellipticity (deg)')
    axes[2].set(ylabel='Fraction of incident power', xlabel='Air gap behind finite substrate (nm)')
    fig.suptitle(f'Diagnostic monochromatic mirror return at {wavelength:g} nm\n'
                 f'{substrate_nm/1000:g} um lossless N-BK7 proxy + air gap + ideal PEC; fixed laboratory axes')
    fig.savefig(output/'overview.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args(argv)
    candidate = json.loads(args.candidate.read_text(encoding='utf-8'))
    axis = candidate['incident_axis']
    if isinstance(axis, bool) or axis not in (0, 1):
        raise ValueError('candidate incident_axis must be 0 (x) or 1 (y)')
    wavelength = float(candidate.get('wavelength_nm', 1560.))
    if not 1550 <= wavelength <= 1570:
        raise ValueError('candidate wavelength must lie inside the material band 1550-1570 nm')
    if (ROOT/candidate['material_file']).resolve() != MATERIAL.resolve():
        raise ValueError('this diagnostic requires the same Royer composite material as scan_grating.py')
    geom = {**geometry(candidate), 'overetch_nm': float(candidate.get('overetch_nm', 30.))}
    args.out.mkdir(parents=True, exist_ok=False)
    study = Study()
    substrate_nm, scan_order, verification_order = 500000., 21, 31
    started = perf_counter()
    with threadpool_limits(1):
        bare = study.solve(geom, wavelength, order=scan_order, finite_substrate_nm=substrate_nm)
        finite_single = transmission_metrics(bare, axis)
        # One full Maxwell solve supplies all 157 gaps; only the feedback solve
        # changes. Its full order matrices are not reduced to 2x2 Jones blocks.
        gaps = np.linspace(0., wavelength/2, 157)
        scan_started = perf_counter()
        rows = [_return_metrics(grating_mirror_return(bare, float(gap)), axis) for gap in gaps]
        gap_scan_seconds = perf_counter()-scan_started
        save_csv(args.out/'gap_scan.csv', rows)
        qualified = [row for row in rows if row['full_qualified']]
        if not qualified:
            raise RuntimeError('no mirror gap passed the diagnostic power/ellipticity screen; gap_scan.csv retained')
        selected = max(qualified, key=lambda row: row['full_fom_deg'])
        gap = selected['gap_nm']
        refined_bare = study.solve(geom, wavelength, order=verification_order, finite_substrate_nm=substrate_nm)
        refined = grating_mirror_return(refined_bare, gap)
        refined_metrics = _return_metrics(refined, axis)
        finite_single_refined = transmission_metrics(refined_bare, axis)
        changes = {key: abs(refined_metrics[key]-selected[key]) for key in
                   ['full_rotation_deg', 'full_ellipticity_deg', 'full_zero_order_power', 'total_returned_power',
                    'first_rotation_deg', 'first_ellipticity_deg', 'first_zero_order_power']}
        convergence_passed = bool(refined_metrics['full_qualified'] and all(
            change <= (.005 if key.endswith('_deg') else .001) for key, change in changes.items()))
        reversed_bare = study.solve(geom, wavelength, order=verification_order, magnetic_sign=-1.,
                                    finite_substrate_nm=substrate_nm)
        reversed_result = grating_mirror_return(reversed_bare, gap)
        reversed_metrics = _return_metrics(reversed_result, axis)
        unmagnetized_bare = study.solve(geom, wavelength, order=verification_order, magnetic_sign=0.,
                                      finite_substrate_nm=substrate_nm)
        unmagnetized = grating_mirror_return(unmagnetized_bare, gap)
        half_space = study.solve(geom, wavelength, order=verification_order)
        transpose_error = float(np.linalg.norm(refined['returned_jones']-reversed_result['returned_jones'].T))
        zero_cross = float(abs(unmagnetized['returned_jones'][1-axis, axis]))
        magnetic_passed = bool(transpose_error <= 1e-8 and zero_cross <= 1e-8)
        save_csv(args.out/'convergence.csv', [dict(order=scan_order, **selected),
                                            dict(order=verification_order, **refined_metrics)])
        source_paths = [Path(__file__), ROOT/'scripts/scan_grating.py',
                        ROOT/'src/magnetooptics/grating.py', ROOT/'src/magnetooptics/grating_design.py',
                        ROOT/'src/magnetooptics/grating_mirror.py']
        summary = dict(study='Diagnostic full-order grating / finite substrate / air gap / PEC return',
            geometry=geom, incident_axis=axis, wavelength_nm=wavelength,
            substrate_thickness_nm=substrate_nm, substrate_index=bare['layers'][-1]['epsilon_a'].real**.5,
            mirror='ideal PEC, tangential-electric rm=-1 for every order',
            gap_scan=dict(range_nm=[0., wavelength/2], samples=len(gaps), harmonic_order=scan_order,
                full_maxwell_solves=1, feedback_solves=len(gaps), seconds=gap_scan_seconds,
                selection='maximize |full-return ellipse-axis rotation|*sqrt(R0) subject to R0>=0.25 and |ellipticity|<=5deg',
                gap_discretization_nm=float(gaps[1]-gaps[0]), global_optimum_proven=False),
            selected_order21=selected, selected_verified=refined_metrics,
            harmonic_convergence=dict(orders=[scan_order, verification_order], absolute_changes=changes,
                passed=convergence_passed, note='empirical refinement at the selected sampled gap, not a rigorous error bound'),
            bare_transmission=dict(finite_substrate_order21=finite_single,
                finite_substrate_verified=finite_single_refined,
                semi_infinite_substrate_verified=transmission_metrics(half_space, axis)),
            double_pass_comparison=_comparison(refined_metrics, finite_single_refined),
            magnetic_checks=dict(plus=refined_metrics, minus=reversed_metrics,
                zero=_return_metrics(unmagnetized, axis),
                plus_minus_rotation_sum_deg=refined_metrics['full_rotation_deg']+reversed_metrics['full_rotation_deg'],
                transpose_error=transpose_error, zero_cross_amplitude=zero_cross, passed=magnetic_passed),
            returned_jones=refined['returned_jones'], first_return_jones=refined['first_return_jones'],
            returned_power_by_order=refined['reflection_by_order'], harmonic_orders=refined['harmonic_orders'],
            assumptions=['500 um substrate is an illustrative choice, not user-specified or measured',
                'SCHOTT N-BK7 dispersion proxy; glass absorption neglected',
                'normal monochromatic plane wave, infinitely periodic x and invariant y',
                'fully filled rectangular gaps and uniform residual composite cap',
                'saturated fixed axial magnetization and fixed literature composite tensor',
                'ideal PEC is not a purchased mirror coating model',
                'first return excludes prompt reflection and all later round trips',
                'all retained diffraction and evanescent orders participate in mirror feedback'],
            limitations=['no wavelength or angular averaging; coherent thick-substrate fringes need resolved sampling',
                'actual substrate thickness/loss, mirror optical data, coating profile and source coherence remain unspecified',
                'no claim of 45-degree transmission, 90-degree return, or fabrication readiness'],
            performance=dict(solver_calls=study.calls, solver_seconds=study.seconds,
                gap_scan_seconds=gap_scan_seconds, total_seconds_before_plot=perf_counter()-started,
                blas_threads=1, python=platform.python_version(), numpy=np.__version__),
            candidate_sha256=hashlib.sha256(args.candidate.read_bytes()).hexdigest(),
            material_sha256=hashlib.sha256(MATERIAL.read_bytes()).hexdigest(),
            source_sha256={str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_paths})
        save_json(args.out/'summary.json', summary)
        _plot(args.out, rows, refined_metrics, finite_single_refined, wavelength, substrate_nm)
        print(json.dumps(dict(selected=refined_metrics, convergence_passed=convergence_passed,
                              comparison=summary['double_pass_comparison'], performance=summary['performance']), indent=2), flush=True)
        if not convergence_passed:
            raise RuntimeError('mirror selected point did not meet harmonic refinement acceptance; diagnostic artifacts retained')
        if not magnetic_passed:
            raise RuntimeError('mirror magnetic-reversal check failed; diagnostic artifacts retained')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
