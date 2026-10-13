"""Controlled radial interpolation for a uniform finite-powder q grid.

Positive distance weights are accumulated as local moments of sin(q*r)/r.
The Fourier transform evaluates their uniform radius centers; a Taylor
remainder bounds the radial interpolation error in exact arithmetic. This
bound deliberately does not claim to bound floating-point accumulation/FFT
roundoff. Neither radii nor intensities are clipped after evaluation.
"""
from __future__ import annotations

from math import factorial
import numpy as np
from scipy.fft import next_fast_len, rfft

from .core import validate_q, _positive_integer


class RadialMomentHistogram:
    """Reusable nonnegative-weight radial sum on q=0,dq,...,qmax.

    ``moments[k,j] = sum(w/r * ((r-j*dr)/dr)**k)`` for strictly
    positive radii assigned to nearest center j. Exact zero radii are held
    separately. A producer can call :meth:`load_moments` with compiled sums.

    The q step must resolve the largest physical radius: rmax < pi/dq.
    The Fourier grid is capped to avoid accidental allocation of enormous
    arrays for a nearly zero q step; this is a supported-domain check, not
    a silent approximation or fallback.
    """

    def __init__(self, q, max_radius, bin_width=0.04, order=8):
        self.q = np.array(validate_q(q), copy=True)
        if len(self.q) < 2 or self.q[0] != 0.0:
            raise ValueError('radial FFT requires at least two q values starting at zero.')
        dq = float(self.q[-1] / (len(self.q) - 1))
        target = np.arange(len(self.q), dtype=float) * dq
        if dq <= 0 or not np.allclose(self.q, target, rtol=2e-14, atol=0.0):
            raise ValueError('radial FFT requires a strictly increasing uniform q grid.')
        if isinstance(bin_width, (bool, np.bool_)) or not np.isfinite(bin_width) or bin_width <= 0:
            raise ValueError('bin_width must be finite and positive.')
        if isinstance(max_radius, (bool, np.bool_)) or not np.isfinite(max_radius) or max_radius < 0:
            raise ValueError('max_radius must be finite and nonnegative.')
        order = _positive_integer(order, 'order')
        if not 4 <= order <= 12:
            raise ValueError('order must be an integer from 4 through 12.')
        requested = 2 * np.pi / dq / float(bin_width)
        # The cap is tested before integer conversion, including tiny dq.
        if not np.isfinite(requested) or requested > 2**22:
            raise ValueError('radial FFT grid exceeds 4194304 points; increase q step or bin_width.')
        self.n_fft = int(next_fast_len(max(int(np.ceil(requested)), 2 * len(self.q))))
        if self.n_fft > 2**22:
            raise ValueError('radial FFT grid exceeds 4194304 points.')
        self.dr = 2 * np.pi / (self.n_fft * dq)
        self.max_radius = float(max_radius)
        if self.max_radius >= np.pi / dq:
            raise ValueError('q step is too coarse: max_radius must be less than pi/dq.')
        self.order = order
        self.dq = dq
        self.n_bins = int(np.floor(self.max_radius / self.dr + 0.5)) + 1
        self.moments = np.zeros((order + 1, self.n_bins))
        self.total_weight = 0.0
        self.zero_weight = 0.0
        self.inverse_radius_weight = 0.0
        self.remainder_mass = 0.0
        self.q.setflags(write=False)

    def add(self, distances, weights=None):
        """Accumulate one bounded block; weights must be finite and nonnegative."""
        radii = np.asarray(distances, dtype=float)
        if radii.ndim != 1 or not np.isfinite(radii).all() or np.any(radii < 0):
            raise ValueError('distances must be a finite nonnegative one-dimensional array.')
        weight = np.ones_like(radii) if weights is None else np.asarray(weights, dtype=float)
        if weight.shape != radii.shape or not np.isfinite(weight).all() or np.any(weight < 0):
            raise ValueError('weights must match distances and be finite and nonnegative.')
        if len(radii) and np.max(radii) > self.max_radius:
            raise ValueError('distance exceeds the declared max_radius.')
        positive = (radii > 0) & (weight > 0)
        radius = radii[positive]
        wp = weight[positive]
        index = np.floor(radius / self.dr + 0.5).astype(np.int64)
        delta = radius / self.dr - index
        inverse = wp / radius
        if not np.isfinite(inverse).all():
            raise ValueError('positive inverse-radius weights must be representable in float64.')
        self.total_weight += float(weight.sum())
        self.zero_weight += float(weight[radii == 0].sum())
        self.inverse_radius_weight += float(inverse.sum())
        power = np.ones_like(radius)
        for k in range(self.order + 1):
            self.moments[k] += np.bincount(index, weights=inverse * power, minlength=self.n_bins)
            power *= delta
        self.remainder_mass += float(np.sum(inverse * np.abs(power)))
        return self

    def load_moments(self, moments, total_weight, zero_weight, inverse_radius_weight,
                     remainder_mass=None):
        """Replace the histogram with a validated independent producer's sums.

        ``remainder_mass`` is sum(w/r*abs(delta/dr)**(order+1)); if omitted,
        the conservative half-bin bound is used. The returned bound excludes
        roundoff in the producer and in this transform.
        """
        array = np.asarray(moments, dtype=float)
        scalars = np.asarray([total_weight, zero_weight, inverse_radius_weight], dtype=float)
        if array.shape != self.moments.shape or not np.isfinite(array).all():
            raise ValueError('moments must be finite and match the configured histogram shape.')
        if not np.isfinite(scalars).all() or np.any(scalars < 0) or zero_weight > total_weight:
            raise ValueError('histogram weights must be finite, nonnegative and consistent.')
        if np.any(array[0] < 0):
            raise ValueError('zeroth moments must be nonnegative.')
        if not np.isclose(array[0].sum(), inverse_radius_weight, rtol=1e-8, atol=1e-12):
            raise ValueError('zeroth moments must sum to inverse_radius_weight.')
        remainder = (float(inverse_radius_weight) * 0.5**(self.order + 1)
                     if remainder_mass is None else float(remainder_mass))
        if not np.isfinite(remainder) or remainder < 0:
            raise ValueError('remainder_mass must be finite and nonnegative.')
        self.moments = np.array(array, copy=True)
        self.total_weight, self.zero_weight, self.inverse_radius_weight = map(float, scalars)
        self.remainder_mass = remainder
        return self

    def evaluate(self):
        """Return sum(w*sinc(q*r)) and its absolute Taylor truncation bound."""
        # Bin zero has no center phase. Evaluate its odd polynomial terms
        # directly: its possibly huge unused w/r zeroth moment must never
        # enter an FFT and contaminate an imaginary part through roundoff.
        work = self.moments.copy()
        work[:, 0] = 0.0
        transforms = rfft(work, n=self.n_fft, axis=1)[:, :len(self.q)]
        positive = self.q[1:]
        coefficient = np.ones(len(positive), dtype=complex)
        sine_sum = np.zeros(len(positive))
        for k in range(self.order + 1):
            sine_sum += np.imag(coefficient * transforms[k, 1:].conjugate())
            sine_sum += coefficient.imag * self.moments[k, 0]
            coefficient *= 1j * positive * self.dr / (k + 1)
        values = np.empty(len(self.q))
        values[0] = self.total_weight
        values[1:] = self.zero_weight + sine_sum / positive
        bound = np.zeros(len(self.q))
        bound[1:] = (self.remainder_mass * self.dr**(self.order + 1)
                     * positive**self.order / factorial(self.order + 1))
        return values, bound

    @property
    def metadata(self):
        return {
            'radial_transform': 'nearest_center_moments_real_fft',
            'moment_order': self.order,
            'radial_bin_width_A': self.dr,
            'radial_bins': self.n_bins,
            'fft_points': self.n_fft,
            'q_step_invA': self.dq,
            'max_radius_A': self.max_radius,
            'radial_moment_bytes': self.moments.nbytes,
            'truncation_bound': 'pointwise_absolute_Taylor_remainder_excludes_floating_point',
            'zero_q': 'exact_total_weight',
        }
