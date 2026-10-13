#!/usr/bin/env python3
"""Independent reference and accuracy metrics for resolved finite PXRD curves.

The monatomic cube oracle uses exact integer shell multiplicities, without the
production lattice cache, basis symmetry reduction, radial histogram or FFT.
Curve metrics exclude the forward beam by default. A refinement comparison is
numerical convergence evidence and must not be labelled an exact-reference test.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import find_peaks


def integer_cube_shells(cells: int, spacing_A: float):
    """Exact ordered-pair radii/counts of an N-by-N-by-N monatomic cubic crystal.

    A displacement (h,k,l) has multiplicity (N-|h|)(N-|k|)(N-|l|).
    Equal integer h*h+k*k+l*l values define shells before any floating-point
    square root. Origin self terms and both pair orders are included.
    """
    if isinstance(cells, (bool, np.bool_)) or not isinstance(cells, (int, np.integer)) or cells < 1:
        raise ValueError('cells must be a positive integer.')
    if not np.isfinite(spacing_A) or spacing_A <= 0:
        raise ValueError('spacing_A must be positive and finite.')
    axis = np.arange(1 - cells, cells, dtype=np.int64)
    h, k, ell = np.meshgrid(axis, axis, axis, indexing='ij')
    squared = (h*h + k*k + ell*ell).ravel()
    counts = ((cells - abs(h)) * (cells - abs(k)) * (cells - abs(ell))).ravel()
    weights = np.zeros(int(squared.max()) + 1, dtype=np.int64)
    np.add.at(weights, squared, counts)
    nonzero = np.flatnonzero(weights)
    if int(weights.sum()) != cells**6 or int(weights[0]) != cells**3:
        raise ArithmeticError('Integer cube pair-count identities failed.')
    return np.sqrt(nonzero.astype(float)) * spacing_A, weights[nonzero]


def integer_cube_iq(q, cells: int, spacing_A: float, scattering_factor=1.0,
                    q_chunk: int = 128):
    """Exact finite powder reference; factors are supplied, not fitted or inferred."""
    q = np.asarray(q, dtype=float)
    if q.ndim != 1 or not q.size or not np.isfinite(q).all() or np.any(q < 0):
        raise ValueError('q must be a finite, nonnegative vector.')
    if isinstance(q_chunk, bool) or not isinstance(q_chunk, (int, np.integer)) or q_chunk < 1:
        raise ValueError('q_chunk must be a positive integer.')
    factor = np.broadcast_to(np.asarray(scattering_factor, dtype=float), q.shape)
    if not np.isfinite(factor).all():
        raise ValueError('scattering_factor must be finite and broadcast to q.')
    radii, counts = integer_cube_shells(cells, spacing_A)
    values = np.empty(q.size)
    for start in range(0, q.size, q_chunk):
        stop = min(start + q_chunk, q.size)
        kernel = np.sinc(q[start:stop, None] * radii[None, :] / np.pi)
        values[start:stop] = (kernel * counts[None, :]).sum(axis=1)
    return values * factor**2


def compare_powder_curves(q, actual, reference, *, q_min_A_inv=0.1,
                         pointwise_floor_fraction=0.01, target_fraction=0.001,
                         reference_kind='exact'):
    """Report non-forward errors without hiding small/negative intensities.

    The target is a numerical intensity tolerance. It is not an experimental
    accuracy claim or a proof that an untested input has the same error.
    Pointwise errors are reported where the reference exceeds the stated floor;
    near zeros an absolute error relative to the non-forward peak is meaningful.
    Peak positions are discrete-grid positions, with grid spacing reported.
    """
    q, actual, reference = [np.asarray(x, dtype=float) for x in (q, actual, reference)]
    if (q.ndim != 1 or q.size < 2 or actual.shape != q.shape or reference.shape != q.shape
            or not all(np.isfinite(x).all() for x in (q, actual, reference))
            or np.any(q < 0) or np.any(np.diff(q) <= 0)):
        raise ValueError('Curves must be finite equal-length vectors on an increasing nonnegative q grid.')
    if not np.isfinite(q_min_A_inv) or q_min_A_inv < 0:
        raise ValueError('q_min_A_inv must be finite and nonnegative.')
    if not 0 < pointwise_floor_fraction <= 1 or not 0 < target_fraction < 1:
        raise ValueError('Use positive fractions: floor <= 1 and target < 1.')
    region = q >= q_min_A_inv
    if np.count_nonzero(region) < 2:
        raise ValueError('At least two q points must lie in the non-forward comparison region.')
    qr, ar, rr = q[region], actual[region], reference[region]
    scale = float(np.max(abs(rr)))
    if scale == 0:
        raise ValueError('The non-forward reference has zero scale.')
    difference = ar - rr
    active = abs(rr) >= pointwise_floor_fraction * scale
    scaled = float(np.max(abs(difference)) / scale)
    pointwise = float(np.max(abs(difference[active]) / abs(rr[active])))
    area = float(trapezoid(abs(rr), qr))
    integral_error = float(abs(trapezoid(difference, qr)) / area)
    l1_error = float(trapezoid(abs(difference), qr) / area)
    maxima = find_peaks(rr, prominence=pointwise_floor_fraction * scale)[0]
    minima = find_peaks(-rr)[0]
    peaks = []
    for index in maxima:
        left_options, right_options = minima[minima < index], minima[minima > index]
        left = int(left_options[-1]) if left_options.size else 0
        right = int(right_options[0]) if right_options.size else qr.size - 1
        observed = left + int(np.argmax(ar[left:right+1]))
        window = slice(left, right+1)
        peak_area = float(trapezoid(rr[window], qr[window]))
        observed_area = float(trapezoid(ar[window], qr[window]))
        peaks.append(dict(reference_q_A_inv=float(qr[index]), actual_q_A_inv=float(qr[observed]),
                          q_shift_A_inv=float(qr[observed] - qr[index]),
                          reference_height=float(rr[index]), actual_height=float(ar[observed]),
                          window_A_inv=[float(qr[left]), float(qr[right])],
                          relative_area_error=abs(observed_area-peak_area)/abs(peak_area)
                          if peak_area != 0 else None))
    peak_area_error = max((p['relative_area_error'] for p in peaks
                           if p['relative_area_error'] is not None), default=0.)
    peak_q_shift = max((abs(p['q_shift_A_inv']) for p in peaks), default=0.)
    result = dict(reference_kind=reference_kind, target_fraction=target_fraction,
                  q_min_A_inv=q_min_A_inv, q_max_A_inv=float(qr[-1]),
                  grid_step_min_A_inv=float(np.diff(qr).min()),
                  grid_step_max_A_inv=float(np.diff(qr).max()),
                  comparison_points=int(qr.size), nonforward_peak_scale=scale,
                  max_abs_error_over_nonforward_peak=scaled,
                  pointwise_floor_fraction=pointwise_floor_fraction,
                  pointwise_points_above_floor=int(np.count_nonzero(active)),
                  max_relative_error_above_floor=pointwise,
                  relative_integral_error=integral_error,
                  relative_integrated_absolute_error=l1_error,
                  minimum_actual=float(ar.min()), minimum_reference=float(rr.min()),
                  peak_count=len(peaks), peaks=peaks,
                  max_peak_relative_area_error=peak_area_error,
                  max_peak_q_shift_A_inv=peak_q_shift,
                  intensity_checks_passed=bool(max(scaled, pointwise, integral_error,
                                                   l1_error, peak_area_error) <= target_fraction),
                  interpretation='Tested-curve numerical agreement; no bound for untested inputs. '
                                 'Refinement is not independent exact validation. No clipping is applied.')
    zero = np.flatnonzero(q == 0)
    if zero.size:
        i = int(zero[0])
        result['forward_actual'] = float(actual[i])
        result['forward_reference'] = float(reference[i])
        result['forward_relative_error'] = float(abs(actual[i]-reference[i])/abs(reference[i])) if reference[i] else None
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--compare', type=Path, help='NPZ containing q, actual, and reference arrays.')
    parser.add_argument('--reference-kind', default='exact', choices=['exact', 'refinement'])
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--q-min', type=float, default=0.1)
    parser.add_argument('--target', type=float, default=0.001)
    parser.add_argument('--cells', type=int, default=40)
    parser.add_argument('--spacing', type=float, default=25.832)
    parser.add_argument('--points', type=int, default=6001)
    args = parser.parse_args()
    if args.out.exists():
        parser.error('Use a fresh output directory.')
    if not args.compare and args.points < 2:
        parser.error('points must be at least two.')
    args.out.mkdir(parents=True)
    if args.compare:
        with np.load(args.compare) as arrays:
            result = compare_powder_curves(arrays['q'], arrays['actual'], arrays['reference'],
                                            q_min_A_inv=args.q_min, target_fraction=args.target,
                                            reference_kind=args.reference_kind)
    else:
        q = np.linspace(0, 6, args.points)
        reference = integer_cube_iq(q, args.cells, args.spacing)
        np.savez_compressed(args.out/'integer_cube_reference.npz', q=q, reference=reference)
        result = dict(reference_kind='exact integer-shell monatomic finite cube',
                      cells_per_edge=args.cells, atoms=args.cells**3, spacing_A=args.spacing,
                      edge_nm=args.cells*args.spacing/10, q_points=args.points,
                      factor=1.0, expected_forward=args.cells**6, actual_forward=float(reference[0]),
                      first_order_peak_width_scale_A_inv=2*np.pi/(args.cells*args.spacing),
                      scope='One unit scatterer per cell; tests the full geometric extent, not MOF chemistry.')
    (args.out/'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    main()
