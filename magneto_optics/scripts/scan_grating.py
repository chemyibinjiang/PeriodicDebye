"""Reproducible fixed-material grating search, refinement and acceptance study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path
import platform
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

import numpy as np
import scipy
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
from magnetooptics.grating import grating_scattering
from magnetooptics.grating_design import (nbk7_index, prototype_layers,
    transmission_metrics, relative_rotation, stokes_band_average)
from magnetooptics.runner import material_tensor


MATERIAL = ROOT / 'material_data/royer_2020/cofe2o4_silica_1550_1570nm.json'
GEOMETRY_KEYS = ['period_nm', 'fill_fraction', 'height_nm', 'cap_nm']


def serial(value):
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    if isinstance(value, np.ndarray):
        return serial(value.tolist())
    if isinstance(value, (complex, np.complexfloating)):
        return [float(value.real), float(value.imag)]
    if isinstance(value, np.generic):
        return value.item()
    return value


def save_json(path, value):
    path.write_text(json.dumps(serial(value), indent=2, allow_nan=False)+'\n', encoding='utf-8')


def save_csv(path, rows):
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def geometry(row):
    return {key: float(row[key]) for key in GEOMETRY_KEYS}


class Study:
    def __init__(self):
        self.material = json.loads(MATERIAL.read_text(encoding='utf-8'))
        self.calls = 0
        self.seconds = 0.

    def solve(self, geom, wavelength=1560., order=5, magnetic_sign=1., finite_substrate_nm=None):
        tensor = material_tensor(self.material, wavelength)
        diagonal, coupling = tensor[0, 0], tensor[0, 1]/1j
        substrate = nbk7_index(wavelength)
        layers = prototype_layers(geom, diagonal, coupling, substrate)
        exit_index = substrate
        if finite_substrate_nm is not None:
            layers.append(dict(thickness_nm=finite_substrate_nm, fill_fraction=1.,
                               epsilon_a=substrate**2, epsilon_b=substrate**2))
            exit_index = 1.
        started = perf_counter()
        result = grating_scattering(wavelength, geom['period_nm'], layers,
            n_exit=exit_index, order=order, magnetization=magnetic_sign)
        self.seconds += perf_counter()-started
        self.calls += 1
        return result

    def rows(self, geom, wavelength=1560., order=5, magnetic_sign=1.):
        result = self.solve(geom, wavelength, order, magnetic_sign)
        return [dict(**geom, wavelength_nm=wavelength, order=order,
                     incident_axis=i, **transmission_metrics(result, i)) for i in (0, 1)]

    def uniform_reference(self, thickness, wavelength=1560.):
        tensor = material_tensor(self.material, wavelength)
        layer = dict(thickness_nm=thickness, fill_fraction=1.,
                     epsilon_a=tensor[0, 0], epsilon_b=tensor[0, 0],
                     g_a=tensor[0, 1]/1j, g_b=tensor[0, 1]/1j)
        # order=0 is exact for a uniform layer at normal incidence.
        result = grating_scattering(wavelength, 900., [layer],
                                    n_exit=nbk7_index(wavelength), order=0)
        return transmission_metrics(result, 0)


def choose_seeds(rows, count):
    ranked = sorted((r for r in rows if r['qualified']), key=lambda r: r['fom_deg'], reverse=True)
    selected = []
    for row in ranked:
        if all(row['incident_axis'] != old['incident_axis'] or
               abs(row['period_nm']-old['period_nm']) > 12 or
               abs(row['fill_fraction']-old['fill_fraction']) > .09 or
               abs(row['height_nm']-old['height_nm']) > 80 or
               abs(row['cap_nm']-old['cap_nm']) > 40 for old in selected):
            selected.append(row)
        if len(selected) >= count:
            break
    return selected


def optimize(study, seed, order, maxfev):
    history = []
    axis = seed['incident_axis']
    # Dimensionless variables have comparable scales for Powell line searches.
    scales = np.array([1000., 1., 600., 100.])
    limits = np.array([[850., 1100.], [.4, .8], [400., 800.], [0., 100.]]) / scales[:, None]
    def objective(x):
        geom = dict(zip(GEOMETRY_KEYS, x*scales))
        gap = geom['period_nm']*(1-geom['fill_fraction'])
        if gap < 200:
            return 10 + (200-gap)**2
        rows = study.rows(geom, order=order)
        history.extend(rows)
        row = rows[axis]
        # Smooth penalties guide toward the declared usable-power/ellipse screen.
        penalty = 200*max(.25-row['zero_order_transmission'], 0)**2
        penalty += max(abs(row['ellipticity_deg'])-5., 0)**2
        return -row['fom_deg'] + penalty
    fit = minimize(objective, np.array([seed[k] for k in GEOMETRY_KEYS])/scales,
                   method='Powell', bounds=limits,
                   options=dict(maxfev=maxfev, xtol=2e-5, ftol=2e-5))
    return history, dict(success=bool(fit.success), message=str(fit.message), evaluations=int(fit.nfev))


def track_resonance(study, seed, maxiter):
    """Joint small-step refinement follows a resonance as dimensions change."""
    history, cache = [], {}
    axis = seed['incident_axis']
    center = np.array([seed[k] for k in GEOMETRY_KEYS])
    scales = np.array([10., .01, 10., 10.])
    bounds = (np.array([[850., 1100.], [.4, .8], [400., 800.], [0., 100.]])
              - center[:, None])/scales[:, None]
    def evaluate(x):
        key = tuple(x)
        if key not in cache:
            rows = study.rows(dict(zip(GEOMETRY_KEYS, center+x*scales)), order=11)
            history.extend(rows)
            cache[key] = rows[axis]
        return cache[key]
    def constraints(x):
        row = evaluate(x)
        value = center+x*scales
        return [row['zero_order_transmission']-.25,
                5.-abs(row['ellipticity_deg']), value[0]*(1-value[1])-200.]
    fit = minimize(lambda x: -evaluate(x)['fom_deg'], np.zeros(4), method='SLSQP',
        bounds=bounds, constraints=[dict(type='ineq', fun=constraints)],
        options=dict(maxiter=maxiter, ftol=1e-8, eps=1e-6))
    # Keep feasible visited points even if the optimizer's iteration budget ends.
    history = [r for r in history if r['period_nm']*(1-r['fill_fraction']) >= 200.]
    return history, dict(method='SLSQP resonance tracking', success=bool(fit.success),
                         message=str(fit.message), evaluations=int(fit.nfev))


def plot_report(output, coarse, spectra, chosen, baseline, tolerances):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    ax = axes[0, 0]
    subset = [r for r in coarse if r['qualified']]
    scatter = ax.scatter([r['zero_order_transmission'] for r in subset],
                         [abs(r['rotation_deg']) for r in subset],
                         c=[r['height_nm'] for r in subset], s=8, alpha=.5, cmap='viridis')
    ax.scatter([chosen['zero_order_transmission']], [abs(chosen['rotation_deg'])],
               marker='*', c='#d84d37', s=180, label='refined / verified')
    ax.axhline(abs(baseline['rotation_deg']), c='#666666', ls='--', label='same-height uniform MO film')
    ax.set(xlabel='Zero-order transmitted power', ylabel='Absolute rotation (deg)', title='Usable candidates: T0 >= 0.25, |ellipticity| <= 5 deg')
    ax.legend(fontsize=8)
    fig.colorbar(scatter, ax=ax, label='SiN height (nm)')
    waves = np.array([r['wavelength_nm'] for r in spectra])
    ax = axes[0, 1]
    ax.plot(waves, [r['rotation_deg'] for r in spectra], label='rotation')
    ax.plot(waves, [r['ellipticity_deg'] for r in spectra], label='ellipticity')
    ax.set(xlabel='Wavelength (nm)', ylabel='Degrees', title='Selected geometry; material held fixed')
    ax.legend()
    ax = axes[1, 0]
    for key, label in [('zero_order_transmission','T0'), ('total_reflection','R all orders'),
                       ('absorption_flux_difference','A = 1 - T - R')]:
        ax.plot(waves, [r[key] for r in spectra], label=label)
    ax.set(xlabel='Wavelength (nm)', ylabel='Fraction of incident power', title='Energy accounting includes substrate diffraction')
    ax.legend(fontsize=9)
    ax = axes[1, 1]
    for parameter, color in [('period_nm','#247ba0'), ('line_width_nm','#d85c41'),
                              ('height_nm','#6d9251'), ('cap_nm','#8359a3')]:
        rows = [r for r in tolerances if r['parameter'] == parameter]
        ax.plot([r['offset_nm'] for r in rows], [r['rotation_deg'] for r in rows], 'o-',
                color=color, label=parameter.replace('_nm','').replace('_',' '), ms=3)
    ax.set(xlabel='Dimensional perturbation (nm; one parameter at a time)', ylabel='Rotation (deg)',
           title='Sensitivity at 1560 nm')
    ax.legend(fontsize=8)
    fig.suptitle('Si3N4 + cobalt-ferrite/silica | 1D grating geometry search\n'
                 'Literature composite, ideal filled gaps, normal plane wave; not fabricated performance')
    fig.savefig(output/'overview.png', dpi=170)
    plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--quick', action='store_true', help='small smoke study, not the full search')
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=False)
    study = Study()
    started = perf_counter()
    with threadpool_limits(1):
        periods = np.arange(850., 1100.1, 4. if not args.quick else 25.)
        periods = np.unique(np.append(periods, [966., 1100.]))
        fills = [.4, .5, .6, .7, .8] if not args.quick else [.5, .65]
        heights = [400., 500., 593., 700., 800.] if not args.quick else [500., 700.]
        caps = [0., 50., 100.] if not args.quick else [0., 100.]
        coarse = []
        for period, fill, height, cap in itertools.product(periods, fills, heights, caps):
            if period*(1-fill) < 200-1e-10:
                continue
            coarse.extend(study.rows(dict(period_nm=period, fill_fraction=fill, height_nm=height, cap_nm=cap)))
            if study.calls % 1000 == 0:
                print(f'coarse: {study.calls} geometries, {study.seconds:.1f}s solver time', flush=True)
        save_csv(args.out/'coarse.csv', coarse)
        seeds = choose_seeds(coarse, 8 if not args.quick else 2)
        refined, fit_status = [], []
        for i, seed in enumerate(seeds):
            history, status = optimize(study, seed, order=9, maxfev=450 if not args.quick else 60)
            refined.extend(history)
            fit_status.append(status)
            print(f'refine {i+1}/{len(seeds)}: {status}', flush=True)
        tracking_seeds = choose_seeds(refined+coarse, 6 if not args.quick else 1)
        for i, seed in enumerate(tracking_seeds):
            history, status = track_resonance(study, seed, 100 if not args.quick else 10)
            refined.extend(history)
            fit_status.append(status)
            print(f'track {i+1}/{len(tracking_seeds)}: {status}', flush=True)
        save_csv(args.out/'refined.csv', refined)
        finalists = choose_seeds(coarse+refined, 12 if not args.quick else 3)
        verified = []
        for candidate in finalists:
            verified.extend(study.rows(geometry(candidate), order=21))
        chosen = max((r for r in verified if r['qualified']), key=lambda r: r['fom_deg'])
        geom, axis = geometry(chosen), chosen['incident_axis']
        convergence = []
        for order in [3, 5, 7, 9, 13, 17, 21, 25, 31]:
            convergence.append(study.rows(geom, order=order)[axis])
        save_csv(args.out/'convergence.csv', convergence)
        chosen = convergence[-1]
        if not chosen['qualified']:
            raise RuntimeError('selected candidate failed final qualification; inspect saved results')
        last, previous = convergence[-1], convergence[-2]
        convergence_error = {key: abs(last[key]-previous[key]) for key in
                             ['rotation_deg', 'ellipticity_deg', 'zero_order_transmission']}
        if (convergence_error['rotation_deg'] > .005 or
            convergence_error['ellipticity_deg'] > .005 or
            convergence_error['zero_order_transmission'] > .001):
            raise RuntimeError('selected candidate did not meet harmonic refinement acceptance')
        final_order = 31
        plus = study.solve(geom, order=final_order)
        minus = study.solve(geom, order=final_order, magnetic_sign=-1.)
        unmagnetized = study.solve(geom, order=final_order, magnetic_sign=0.)
        opposite = transmission_metrics(minus, axis)
        theta_odd = .5*(chosen['rotation_deg']-opposite['rotation_deg'])
        zero_cross = float(abs(unmagnetized['transmission'][1-axis, axis]))
        magnetic_passed = abs(chosen['rotation_deg']+opposite['rotation_deg']) < 1e-8 and zero_cross < 1e-10
        if not magnetic_passed:
            raise RuntimeError('magnetic reversal or zero-field cross-polarization check failed')
        magnetic_checks = dict(plus=chosen, minus=opposite,
            zero=transmission_metrics(unmagnetized, axis), odd_rotation_deg=theta_odd,
            angle_branch='principal ellipse-axis changes in [-90,90); large rotations need continuous M unwrapping',
            sign_sum_deg=chosen['rotation_deg']+opposite['rotation_deg'],
            zero_cross_amplitude=zero_cross, passed=bool(magnetic_passed))
        height_baseline = study.uniform_reference(geom['height_nm']+30+geom['cap_nm'])
        volume_baseline = study.uniform_reference((1-geom['fill_fraction'])*(geom['height_nm']+30)+geom['cap_nm'])
        reference_geom = dict(period_nm=966., fill_fraction=608/966, height_nm=593., cap_nm=0.)
        published_reference = study.rows(reference_geom, order=final_order)
        waves = np.linspace(1550., 1570., 401 if not args.quick else 81)
        spectra, fields = [], []
        for wave in waves:
            result = study.solve(geom, float(wave), final_order)
            spectra.append(dict(wavelength_nm=float(wave), **transmission_metrics(result, axis)))
            fields.append(np.sqrt(result['n_exit'])*result['transmission'][:, axis])
        save_csv(args.out/'spectra.csv', spectra)
        bandwidth = []
        for fwhm in [.5, 2., 7.]:
            weights = np.exp(-4*np.log(2)*((waves-1560)/fwhm)**2)
            coarse_weights = weights[::2].copy()
            weights[[0, -1]] *= .5
            coarse_weights[[0, -1]] *= .5
            fine_average = stokes_band_average(fields, weights, axis)
            coarse_average = stokes_band_average(fields[::2], coarse_weights, axis)
            bandwidth.append(dict(fwhm_nm=fwhm, center_nm=1560., integration_band_nm=[1550.,1570.],
                note='Gaussian truncated and normalized within material band; trapezoidal Stokes integration',
                wavelength_step_nm=float(waves[1]-waves[0]),
                grid_refinement_absolute_changes={k:abs(fine_average[k]-coarse_average[k]) for k in
                    ['power', 'rotation_deg', 'ellipticity_deg', 'degree_of_polarization']}, **fine_average))
        if not args.quick and any(r['grid_refinement_absolute_changes']['rotation_deg'] > .001 or
                                 r['grid_refinement_absolute_changes']['power'] > .0001 for r in bandwidth):
            raise RuntimeError('spectral averaging grid failed refinement acceptance')
        tolerances = []
        for parameter in ['period_nm', 'line_width_nm', 'height_nm', 'cap_nm']:
            for offset in [-10., -5., -2., 0., 2., 5., 10.]:
                changed = dict(geom)
                if parameter == 'line_width_nm':
                    changed['fill_fraction'] += offset/geom['period_nm']
                else:
                    changed[parameter] += offset
                if changed['cap_nm'] < 0:
                    continue
                rec = study.rows(changed, order=final_order)[axis]
                tolerances.append(dict(parameter=parameter, offset_nm=offset,
                    perturbation_convention='fixed fill (width and gap scale)' if parameter == 'period_nm'
                                            else 'fixed period; other dimensions unchanged', **rec))
        save_csv(args.out/'tolerances.csv', tolerances)
        save_csv(args.out/'verified_candidates.csv', verified)
        rounded_designs = []
        for step in [1., 5., 10.]:
            rounded = {key: float(np.round(geom[key]/step)*step) for key in
                       ['period_nm', 'height_nm', 'cap_nm']}
            width = np.round(geom['period_nm']*geom['fill_fraction']/step)*step
            rounded['fill_fraction'] = float(width/rounded['period_nm'])
            rounded_designs.append(dict(grid_nm=step, line_width_nm=float(width),
                gap_nm=float(rounded['period_nm']-width), **study.rows(rounded, order=final_order)[axis]))
        spectral_convergence = []
        extrema = {0, len(spectra)-1, len(spectra)//2,
                   int(np.argmax([abs(r['rotation_deg']) for r in spectra])),
                   int(np.argmax([abs(r['ellipticity_deg']) for r in spectra]))}
        for index in sorted(extrema):
            fine = study.rows(geom, float(waves[index]), order=41)[axis]
            changes = {key: abs(fine[key]-spectra[index][key]) for key in
                       ['rotation_deg', 'ellipticity_deg', 'zero_order_transmission']}
            spectral_convergence.append(dict(wavelength_nm=float(waves[index]), orders=[31,41],
                                              absolute_changes=changes))
            if changes['rotation_deg'] > .005 or changes['ellipticity_deg'] > .005 or changes['zero_order_transmission'] > .001:
                raise RuntimeError('spectral extrema failed harmonic refinement acceptance')
        summary = dict(study='Exploratory 1D grating search with fixed literature material',
            quick=args.quick, center_wavelength_nm=1560., selected=chosen,
            geometry=dict(**geom, line_width_nm=geom['period_nm']*geom['fill_fraction'],
                          gap_nm=geom['period_nm']*(1-geom['fill_fraction']), overetch_nm=30.),
            scope='normal plane wave, infinitely periodic in x, invariant y; not a 2D post lattice',
            assumptions=['fully filled rectangular gaps', 'uniform residual composite cap',
                         'lossless Si3N4 index 1.95; measured loss detection limit unspecified',
                         'SCHOTT N-BK7 dispersion as proxy for unspecified BK7, glass loss neglected',
                         'semi-infinite glass for single-pass search',
                         'saturated fixed axial magnetization; material is local constant approximation'],
            search=dict(period_bounds_nm=[850.,1100.], coarse_period_step_nm=4. if not args.quick else 25.,
                        fill_fractions=fills, heights_nm=heights, caps_nm=caps, minimum_gap_nm=200.,
                        coarse_geometry_count=len(coarse)//2, coarse_order=5, refinement_orders=[9,11],
                        objective='maximize |ellipse-axis rotation|*sqrt(T0)',
                        screen=dict(min_zero_order_power=.25,max_abs_ellipticity_deg=5.),
                        refinement=fit_status, global_optimum_proven=False),
            harmonic_convergence=dict(orders=[r['order'] for r in convergence],
                                      last_pair_absolute_changes=convergence_error,
                                      spectral_extrema_checks=spectral_convergence,
                                      note='empirical refinement check, not a rigorous error bound'),
            magnetic_checks=magnetic_checks,
            baselines=dict(same_height_uniform_MO=height_baseline,equal_MO_volume_uniform=volume_baseline,
                           published_geometry_ideal_profile=published_reference),
            enhancement_vs_same_height=abs(chosen['rotation_deg']/height_baseline['rotation_deg']),
            enhancement_vs_equal_MO_volume=abs(chosen['rotation_deg']/volume_baseline['rotation_deg']),
            fom_enhancement_vs_same_height=chosen['fom_deg']/height_baseline['fom_deg'],
            geometry_rounded_to_nm_grid=rounded_designs,
            incoherent_spectral_averages=bandwidth,
            transmission_jones=plus['transmission'], reflection_jones=plus['reflection'],
            performance=dict(calls=study.calls, solver_seconds=study.seconds,
                             total_seconds_before_plot=perf_counter()-started,blas_threads=1,
                             python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__),
            input_sha256=hashlib.sha256(MATERIAL.read_bytes()).hexdigest(),
            source_sha256={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in [Path(__file__), ROOT/'src/magnetooptics/grating.py',
                                        ROOT/'src/magnetooptics/grating_design.py']},
            remaining=['angular divergence not modeled', 'actual infiltration and cap profile unmeasured',
                       'not a full reproduction of experiment; principal TE resonance is outside material band',
                       'real mirror, substrate thickness/loss and laser require sample-specific inputs'])
        save_json(args.out/'summary.json', summary)
        save_json(args.out/'candidate.json', dict(**geom, overetch_nm=30.,
            incident_axis=axis, wavelength_nm=1560., material_file=str(MATERIAL.relative_to(ROOT)),
            status='simulation candidate, not a fabrication release'))
        plot_report(args.out, coarse, spectra, chosen, height_baseline, tolerances)
        print(json.dumps(dict(selected=chosen,enhancement=summary['enhancement_vs_same_height'],
                              performance=summary['performance']), indent=2), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
