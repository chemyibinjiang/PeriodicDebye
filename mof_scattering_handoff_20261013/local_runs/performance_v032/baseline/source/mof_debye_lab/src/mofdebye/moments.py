"""Guest amplitude moments in the supplied Cartesian coordinate frame.

Physical disorder variance and the numerical variance of an estimated mean are
different quantities. Both are retained so that reusing sampled cell moments in
a finite lattice does not silently amplify sample-mean-squared bias.
"""
from __future__ import annotations

from copy import copy
from dataclasses import dataclass
from typing import Any

import numpy as np

from .core import AtomCloud, direct_iq
from .directional import amplitude, validate_q_vectors
from .distributions import PorePopulation
from .ensemble import radial_atom_sum


@dataclass(frozen=True)
class GuestMoments:
    """Total guest mean and variance for one finite object or one cell.

    ``mean`` is in electrons; the variances are in electron squared.
    ``mean_variance`` is E[|estimated mean - exact mean|**2], estimated as
    sum(s_k**2 / L) for independent sampled pores. Its square root is a complex
    mean-amplitude RMS standard error, not an intensity SEM or a material error.
    """

    mean: np.ndarray
    variance: np.ndarray
    mean_variance: np.ndarray
    metadata: dict[str, Any]

    def __post_init__(self):
        mean = np.array(self.mean, dtype=np.complex128, copy=True)
        variance = np.array(self.variance, dtype=float, copy=True)
        mean_variance = np.array(self.mean_variance, dtype=float, copy=True)
        if mean.ndim != 1 or not mean.size or not np.isfinite(mean).all():
            raise ValueError('Moment mean must be a nonempty finite complex vector.')
        for name, values in [('variance', variance), ('mean_variance', mean_variance)]:
            if values.shape != mean.shape or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError(f'Moment {name} must be finite, nonnegative and match mean.')
        for values in (mean, variance, mean_variance):
            values.setflags(write=False)
        object.__setattr__(self, 'mean', mean)
        object.__setattr__(self, 'variance', variance)
        object.__setattr__(self, 'mean_variance', mean_variance)
        object.__setattr__(self, 'metadata', dict(self.metadata))


def _metadata(method, **extra):
    return dict(method=method, amplitude_units='electron',
                variance_units='electron^2', q_units='angstrom^-1',
                normalization='unnormalized total guest amplitude in supplied frame',
                mean_variance_description='numerical variance of the complex mean amplitude estimate',
                physical_variance_description='ensemble disorder variance, including whole-particle occupancy',
                device='cpu', precision='float64', **extra)


def guest_moments(population: PorePopulation | None, q_vectors, method='analytic',
                  n_samples=512, seed=20261013, independent_pores=True):
    """Return moments of the TOTAL guest amplitude over independent pores.

    Analytic evaluation enforces uniform SO(3) particle orientations, unbounded
    isotropic Gaussian jitter, and the current number-weighted template model.
    Sampled evaluation uses the actual existing pore sampler, including its
    fixed-selected-size conditional rejection. It streams one pore at a time;
    no host atoms or full host/guest pair lists are evaluated in the draws.

    Correlated pore states must instead be enumerated with
    :func:`exact_guest_moments`. Inter-cell correlations are a separate model.
    ``None`` represents a cell/object with no guests.
    """
    q_vectors = validate_q_vectors(q_vectors)
    if method not in ('analytic', 'sampled'):
        raise ValueError("Moment method must be 'analytic' or 'sampled'.")
    if not independent_pores:
        raise ValueError('Correlated pores require explicitly enumerated joint states; use exact_guest_moments.')
    if population is None:
        zero = np.zeros(len(q_vectors))
        return GuestMoments(zero.astype(complex), zero, zero,
                            _metadata('empty', independent_pores=True, n_samples=0, seed=None))
    if type(population) is not PorePopulation:
        raise TypeError('The independent-pore engine requires PorePopulation; custom joint samplers need explicit states.')
    if method == 'analytic':
        return _analytic_moments(population, q_vectors)
    if isinstance(n_samples, bool) or not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError('At least two integer samples per pore are required for sampled moments.')
    return _sampled_moments(population, q_vectors, int(n_samples), seed)


def _analytic_moments(population, q_vectors):
    sigma = population.analytic_sigma()
    # Angular grids repeat |q|; calculate radial template statistics only once
    # per distinct magnitude, then restore the requested vector order.
    magnitudes, inverse = np.unique(np.linalg.norm(q_vectors, axis=1), return_inverse=True)
    mean_radial = np.zeros(len(magnitudes))
    mean_internal = np.zeros(len(magnitudes))
    for weight, template in zip(population.library.weights, population.library.templates):
        if weight:
            mean_radial += weight * radial_atom_sum(template, magnitudes)
            mean_internal += weight * direct_iq(template, magnitudes)
    radial = (mean_radial * np.exp(-0.5 * (magnitudes * sigma)**2))[inverse]
    internal = mean_internal[inverse]
    mean = np.zeros(len(q_vectors), dtype=complex)
    variance = np.zeros(len(q_vectors))
    for centre, shift, probability in zip(population.centres, population.mean_displacement,
                                           population.occupancy):
        pore_mean = probability * radial * np.exp(1j * (q_vectors @ (centre + shift)))
        pore_variance = probability * internal - (probability * radial)**2
        scale = np.maximum(probability * internal, 1.0)
        if np.any(pore_variance < -1e-10 * scale):
            raise ArithmeticError('Analytic guest variance is negative beyond floating-point tolerance.')
        mean += pore_mean
        variance += np.maximum(pore_variance, 0.0)
    return GuestMoments(mean, variance, np.zeros(len(q_vectors)),
                        _metadata('analytic', independent_pores=True, seed=None, n_samples=0,
                                  orientation='Haar-uniform SO(3)',
                                  displacement='unbounded isotropic Gaussian',
                                  number_weighted_sizes=True))


def _sampled_moments(population, q_vectors, n_samples, seed):
    rng = np.random.default_rng(seed)
    total_mean = np.zeros(len(q_vectors), dtype=complex)
    total_variance = np.zeros(len(q_vectors))
    occupancy_counts = []
    template_counts = np.zeros(len(population.library.templates), dtype=int)
    placement_attempts = 0
    occupied_draws = 0
    for pore in range(len(population.centres)):
        # The original sampler owns all distribution and rejection rules. A
        # shallow single-pore view shares immutable templates and the host tree,
        # while reindexing only per-pore data. It never mutates the population.
        local = copy(population)
        local.centres = population.centres[pore:pore + 1]
        local.occupancy = population.occupancy[pore:pore + 1]
        local.mean_displacement = population.mean_displacement[pore:pore + 1]
        if population.pore_radius is not None:
            local.pore_radius = population.pore_radius[pore:pore + 1]
        mean = np.zeros(len(q_vectors), dtype=complex)
        m2 = np.zeros(len(q_vectors))
        occupied = 0
        for sample_index in range(n_samples):
            draws = local.sample(rng)
            if draws:
                draw = draws[0]
                values = amplitude(draw.cloud, q_vectors)
                template_counts[draw.template_index] += 1
                placement_attempts += draw.attempts
                occupied += 1
            else:
                values = np.zeros(len(q_vectors), dtype=complex)
            delta = values - mean
            mean += delta / (sample_index + 1)
            m2 += np.real(np.conjugate(delta) * (values - mean))
        occupancy_counts.append(occupied)
        occupied_draws += occupied
        total_mean += mean
        total_variance += np.maximum(m2 / (n_samples - 1), 0.0)
    mean_variance = total_variance / n_samples
    return GuestMoments(total_mean, total_variance, mean_variance,
                        _metadata('sampled', independent_pores=True, seed=seed,
                                  n_samples=n_samples, samples_per_pore=n_samples,
                                  occupancy_counts=occupancy_counts,
                                  template_counts=template_counts.tolist(),
                                  mean_placement_attempts=(placement_attempts / occupied_draws
                                                           if occupied_draws else 0.0),
                                  physical_variance_estimator='unbiased complex sample variance',
                                  mean_variance_estimator='sum of pore sample variances / samples_per_pore',
                                  number_weighted_sizes=True,
                                  negative_bias_corrected_intensities='retained; increase sampling to resolve'))


def exact_guest_moments(states, probabilities, q_vectors):
    """Moments from enumerated FULL guest configurations and their probabilities.

    Each state is an AtomCloud, with empty clouds representing vacant states.
    This supports within-object/cell correlations explicitly present in those
    states. It does not introduce correlations between independently used cells.
    Probabilities must sum to one; they are not intensity or mass fractions.
    """
    q_vectors = validate_q_vectors(q_vectors)
    states = tuple(states)
    probabilities = np.asarray(probabilities, dtype=float)
    if (not states or probabilities.shape != (len(states),)
            or not np.isfinite(probabilities).all() or np.any(probabilities < 0)
            or not np.isclose(probabilities.sum(), 1.0, rtol=1e-12, atol=1e-12)):
        raise ValueError('Provide nonempty states and finite nonnegative probabilities summing to one.')
    if any(not isinstance(state, AtomCloud) for state in states):
        raise TypeError('Every enumerated full guest state must be an AtomCloud.')
    probabilities = probabilities / probabilities.sum()
    mean = np.zeros(len(q_vectors), dtype=complex)
    variance = np.zeros(len(q_vectors))
    accumulated = 0.0
    for state, probability in zip(states, probabilities):
        if not probability:
            continue
        values = amplitude(state, q_vectors)
        new_accumulated = accumulated + probability
        delta = values - mean
        mean += probability / new_accumulated * delta
        variance += probability * accumulated / new_accumulated * np.abs(delta)**2
        accumulated = new_accumulated
    return GuestMoments(mean, variance, np.zeros(len(q_vectors)),
                        _metadata('exact_enumeration', n_states=len(states), seed=None,
                                  within_cell_correlation='as represented by complete states',
                                  independent_pores=False))


def finite_ensemble_intensity(host: AtomCloud, q_vectors, moments: GuestMoments,
                              host_amplitude=None):
    """Finite directional mean intensity with unbiased sampled-mean correction.

    Returns ``coherent = |H + estimated mean|**2 - mean_variance``, physical
    ``diffuse = variance``, and their total. The correction is zero for exact
    or analytic moments. Sampled coherent estimates can be negative near an
    extinction; clipping would bias them. The reported amplitude standard error
    is numerical integration uncertainty, not an intensity SEM.
    """
    q_vectors = validate_q_vectors(q_vectors)
    if moments.mean.shape != (len(q_vectors),):
        raise ValueError('Moment vectors must match the requested q-vector count and order.')
    h = amplitude(host, q_vectors) if host_amplitude is None else np.asarray(host_amplitude, dtype=complex)
    if h.shape != moments.mean.shape or not np.isfinite(h).all():
        raise ValueError('Cached host amplitude must be finite and match the requested q vectors.')
    coherent_raw = np.abs(h + moments.mean)**2
    coherent = coherent_raw - moments.mean_variance
    result = dict(total=coherent + moments.variance, coherent=coherent,
                  coherent_raw=coherent_raw, diffuse=moments.variance,
                  host=np.abs(h)**2,
                  host_guest=2 * np.real(np.conjugate(h) * moments.mean),
                  guest_coherent=np.abs(moments.mean)**2 - moments.mean_variance,
                  mean_variance=moments.mean_variance,
                  mean_amplitude_sem=np.sqrt(moments.mean_variance),
                  mean_amplitude=h + moments.mean)
    result['metadata'] = dict(scattering_case='single_crystal_finite', boundary='finite',
                              intensity_units='electron^2',
                              normalization='unnormalized finite-object ensemble intensity',
                              mean_squared_bias_correction=True,
                              estimator=moments.metadata,
                              uncertainty='mean_amplitude_sem is numerical RMS amplitude error; no intensity SEM claimed')
    return result
