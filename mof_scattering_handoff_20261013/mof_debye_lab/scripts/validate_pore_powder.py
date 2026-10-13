#!/usr/bin/env python3
"""Independent finite-crystal pore-distribution oracles and a 103 nm example.

Synthetic simple-cubic C hosts with body-centred pores are deliberately used:
integer shell multiplicities give a full dense reference without reproducing
the production histogram algorithm. Guest sizes are a number-weighted Au
monomer/dimer catalogue, with Haar rotations and uniform-ball centre positions.
This fixture does not claim a measured MOF pore geometry or loading law.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.special import spherical_jn

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT/'src'))
sys.path.insert(0, str(PROJECT/'scripts'))
from validate_large_powder import compare_powder_curves, integer_cube_shells
from mofdebye.core import AtomCloud, form_factors
from mofdebye.distributions import ParticleLibrary, PorePopulation


def ball_characteristic(q, radius):
    """Independent special-function expression 3*j1(qR)/(qR)."""
    x = np.asarray(q, dtype=float)*radius
    out = np.ones_like(x)
    nonzero = x != 0
    out[nonzero] = 3*spherical_jn(1, x[nonzero])/x[nonzero]
    return out


def body_centered_cross_shells(cells, spacing_A):
    """Host--pore radii grouped by exact integer squared doubled coordinates."""
    axis = np.arange(1-cells, cells, dtype=np.int64)
    h, k, ell = np.meshgrid(axis, axis, axis, indexing='ij')
    squared = ((2*h+1)**2+(2*k+1)**2+(2*ell+1)**2).ravel()
    weights = ((cells-abs(h))*(cells-abs(k))*(cells-abs(ell))).ravel()
    counts = np.zeros(int(squared.max())+1, dtype=np.int64)
    np.add.at(counts, squared, weights)
    retained = np.flatnonzero(counts)
    assert int(counts.sum()) == cells**6
    return np.sqrt(retained.astype(float))*spacing_A/2, counts[retained]


def shell_sum(q, radii, weights):
    result = np.empty(len(q))
    for start in range(0, len(q), 128):
        stop = min(start+128, len(q))
        result[start:stop] = (np.sinc(np.asarray(q[start:stop, None])*radii[None, :]/np.pi)
                              * weights[None, :]).sum(axis=1)
    return result


def monomer_dimer_statistics(q, dimer_separation_A, dimer_weight):
    """Closed mean amplitude and mean internal intensity, conditional on occupancy."""
    q = np.asarray(q)
    f = form_factors(('Au',), q)['Au']
    m = f*((1-dimer_weight)+2*dimer_weight*np.sinc(q*dimer_separation_A/(2*np.pi)))
    internal = f*f*((1-dimer_weight)+2*dimer_weight*(1+np.sinc(q*dimer_separation_A/np.pi)))
    return m, internal


def body_centered_pore_reference(q, cells, spacing_A=25.832, occupancy=.37,
                                dimer_weight=.65, dimer_separation_A=2., radius_A=1.1):
    """Exact finite powder decomposition with independent pores in every cell."""
    q = np.asarray(q, dtype=float)
    radii, counts = integer_cube_shells(cells, spacing_A)
    s00 = shell_sum(q, radii, counts)
    radii, counts = body_centered_cross_shells(cells, spacing_A)
    s0c = shell_sum(q, radii, counts)
    f_host = form_factors(('C',), q)['C']
    mean, internal = monomer_dimer_statistics(q, dimer_separation_A, dimer_weight)
    occupied_mean = occupancy*mean*ball_characteristic(q, radius_A)
    k = cells**3
    parts = dict(host=f_host*f_host*s00,
                 guest_internal=k*occupancy*internal,
                 host_guest=2*f_host*occupied_mean*s0c,
                 guest_guest=occupied_mean**2*(s00-k))
    parts['total'] = sum(parts.values())
    return parts


def synthetic_population(spacing_A=25.832, occupancy=.37, dimer_weight=.65,
                         dimer_separation_A=2., radius_A=1.1, wall_radius_A=4.):
    monomer = AtomCloud(('Au',), [[0., 0., 0.]], 'synthetic Au monomer')
    half = dimer_separation_A/2
    dimer = AtomCloud(('Au', 'Au'), [[-half, 0., 0.], [half, 0., 0.]], 'synthetic Au dimer')
    library = ParticleLibrary((monomer, dimer), [1-dimer_weight, dimer_weight])
    return PorePopulation([[spacing_A/2]*3], library, occupancy=occupancy,
                          position_mode='uniform_ball', uniform_ball_radius_A=radius_A,
                          orientation='uniform', pore_radius_A=wall_radius_A)


def confined_dimer_lens_mean(q, wall_radius_A=2., half_separation_A=.5, order=96):
    """Independent quadrature for a Haar dimer confined in a spherical wall.

    A uniform proposal ball containing the complete allowed region conditions
    the co-moving displacement to a uniform lens: intersection of two spheres
    of radius R centred at +/-a. Integrate x>=0 and the cylindrical radius.
    This accounts for coupled displacement/orientation; separate characteristic
    factors would be incorrect for this active-wall model.
    """
    nodes, weights = np.polynomial.legendre.leggauss(order)
    r, a = wall_radius_A, half_separation_A
    x = (nodes+1)*(r-a)/2
    wx = weights*(r-a)/2
    maximum_rho = np.sqrt(r*r-(x+a)**2)
    rho = maximum_rho[:, None]*(nodes[None, :]+1)/2
    wrho = maximum_rho[:, None]*weights[None, :]/2
    volume_weights = 4*np.pi*wx[:, None]*wrho*rho
    volume = float(volume_weights.sum())
    exact_volume = 2*np.pi/3*(2*r+a)*(r-a)**2
    np.testing.assert_allclose(volume, exact_volume, rtol=2e-13)
    plus = np.sqrt((x[:, None]+a)**2+rho*rho)
    minus = np.sqrt((x[:, None]-a)**2+rho*rho)
    q = np.asarray(q)
    mean = np.array([np.sum(volume_weights*(np.sinc(value*plus/np.pi)
                                          + np.sinc(value*minus/np.pi)))/volume for value in q])
    return mean*form_factors(('Au',), q)['Au']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cells', type=int, default=40)
    parser.add_argument('--points', type=int, default=6001)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output directory.')
    if min(args.cells, args.threads) < 1 or args.points < 2:
        parser.error('Positive cells/threads and at least two q points are required.')
    from threadpoolctl import threadpool_limits
    from mofdebye.pore_powder import PorePowderCache
    from mofdebye.radial_guests import radial_guest_statistics
    from mofdebye.runner import environment_record
    threadpool_limits(limits=1)
    args.out.mkdir(parents=True)
    q = np.linspace(0, 6, args.points)
    population = synthetic_population()
    basis = AtomCloud(('C',), [[0., 0., 0.]])
    start = time.perf_counter()
    reference = body_centered_pore_reference(q, args.cells)
    reference_seconds = time.perf_counter()-start
    start = time.perf_counter()
    statistics = radial_guest_statistics(population, q, method='analytic')
    statistics_seconds = time.perf_counter()-start
    start = time.perf_counter()
    cache = PorePowderCache(basis, np.eye(3)*25.832, np.ones((args.cells,)*3, dtype=int),
                           population, q, bin_width=.04, order=8, threads=args.threads,
                           merge_decimals=12)
    preparation_seconds = time.perf_counter()-start
    start = time.perf_counter()
    actual = cache.iq(statistics)
    evaluation_seconds = time.perf_counter()-start
    comparison = compare_powder_curves(q, actual['total'], reference['total'])
    scale = comparison['nonforward_peak_scale']
    component_errors = {name: float(np.max(abs(actual[name][q >= .1]-values[q >= .1]))/scale)
                        for name, values in reference.items()}
    arrays = dict(q=q)
    arrays.update({f'actual_{key}': actual[key] for key in reference})
    arrays.update({f'reference_{key}': value for key, value in reference.items()})
    arrays['approximation_bound_excluding_roundoff'] = actual['approximation_bound_excluding_roundoff']
    arrays['total_mc_sem'] = actual['total_mc_sem']
    np.savez_compressed(args.out/'curves.npz', **arrays)
    sources = [Path(__file__), PROJECT/'scripts'/'validate_large_powder.py']
    sources += [PROJECT/'src'/'mofdebye'/name for name in
                ('pore_powder.py', 'radial_guests.py', 'large_powder.py', 'radial.py')]
    report = dict(environment=environment_record(), command=sys.argv,
                  source_sha256={str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
                                 for path in sources},
                  cells_per_edge=args.cells, n_cells=args.cells**3, edge_nm=args.cells*2.5832,
                  q_points=args.points, occupancy=.37, dimer_weight=.65, centre_ball_radius_A=1.1,
                  spherical_wall_radius_A=4., reference_seconds=reference_seconds,
                  statistics_seconds=statistics_seconds, preparation_seconds=preparation_seconds,
                  evaluation_seconds=evaluation_seconds, comparison=comparison,
                  component_errors_over_nonforward_total_peak=component_errors,
                  statistics_metadata=statistics.metadata, cache_metadata=cache.metadata,
                  model='Synthetic one-C simple-cubic host with body-centred independent Au monomer/dimer pores; '
                        'not measured MOF pore geometry. Isotropic uniform-ball centre support and Haar rotations. '
                        'Hard walls are certified nonbinding. Exact integer-shell geometry with closed distribution moments.')
    (args.out/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({key: value for key, value in comparison.items() if key != 'peaks'}, indent=2))
    assert comparison['intensity_checks_passed']
    assert comparison['max_peak_q_shift_A_inv'] <= 6/(args.points-1)*1.000001
    assert not np.any(actual['total_mc_sem'])


if __name__ == '__main__':
    main()
