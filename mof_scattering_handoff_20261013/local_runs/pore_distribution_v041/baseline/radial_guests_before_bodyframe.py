"""Conditional radial statistics for independent, isotropically occupied pores.

Whole-particle occupancy belongs to the crystal-level calculation: all means
and internal intensities returned here are conditional on an occupied pore.
Spherical-wall rejection preserves each template's number weight. Two
independent replicas in each batch permit unbiased products of estimated means.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .core import direct_iq, form_factors, validate_q, weighted_sinc_sum, _positive_integer
from .distributions import ParticleLibrary, PorePopulation
from .radial import RadialMomentHistogram


@dataclass(frozen=True)
class RadialGuestStatistics:
    q: np.ndarray
    means: np.ndarray
    mean_bounds: np.ndarray
    internal: np.ndarray
    internal_bounds: np.ndarray
    centres: np.ndarray
    occupancy: np.ndarray
    metadata: dict


def _ball_characteristic(x):
    """Uniform solid-ball characteristic function, stable at and near zero."""
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)
    small = np.abs(x) < 0.05
    z = x[small]**2
    result[small] = 1 + z * (-1/10 + z * (1/280 + z * (-1/15120 + z/1330560)))
    large = x[~small]
    result[~small] = 3 * (np.sin(large) - large * np.cos(large)) / large**3
    return result


def _isotropic_variance(population):
    variance = float(np.trace(population.covariance) / 3)
    if population.position_mode == 'gaussian' and not np.allclose(
            population.covariance, np.eye(3)*variance, rtol=1e-12, atol=1e-12):
        raise ValueError('Radial guest statistics require an isotropic Gaussian covariance.')
    return max(variance, 0.0)


def _check_radial_model(population):
    if type(population) is not PorePopulation:
        raise ValueError('population must be an unmodified PorePopulation; custom joint samplers are unsupported.')
    if population.orientation != 'uniform':
        raise ValueError('Radial guest statistics require Haar-uniform orientation.')
    if population._host_tree is not None or population.min_host_distance is not None:
        raise ValueError('Host-clearance rejection is not isotropic and is unsupported by radial guests.')
    if population.external_boundary is not None or population.guest_retention != 'site':
        raise ValueError('External-boundary conditioning is unsupported by radial guests.')
    return _isotropic_variance(population)


def _readonly(array):
    result = np.array(array, copy=True)
    result.setflags(write=False)
    return result


def radial_guest_statistics(population, q, method='analytic', n_samples=512,
                            n_batches=8, seed=20261013, bin_width=0.04, order=8):
    """Return conditional mean amplitudes and exact internal template scattering.

    ``means`` and ``mean_bounds`` have shape (batches, 2, pores, q_points).
    ``internal`` has shape (pores, q_points); it is the exact number-weighted
    template Debye sum and already includes all intraparticle terms.

    For ``sampled``, n_samples counts accepted placements PER TEMPLATE, PER
    PORE, PER REPLICA across all batches; it must be divisible by n_batches.
    At least two batches and two samples per batch are required. Templates and
    occupancy are not sampled, so unequal acceptance rates cannot reweight
    particle sizes or Bernoulli occupations. Monte Carlo uncertainty belongs to
    the eventual batch intensities, not these radial interpolation bounds.
    """
    q = np.array(validate_q(q), copy=True)
    variance = _check_radial_model(population)
    if method not in ('analytic', 'sampled'):
        raise ValueError('method must be analytic or sampled.')
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError('seed must be a nonnegative integer.')
    n_samples = _positive_integer(n_samples, 'n_samples')
    n_batches = _positive_integer(n_batches, 'n_batches')
    order = _positive_integer(order, 'order')
    if not 4 <= order <= 12:
        raise ValueError('order must be an integer from 4 through 12.')
    if isinstance(bin_width, (bool, np.bool_)) or not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError('bin_width must be finite and positive.')
    bin_width = float(bin_width)
    if method == 'sampled':
        if n_batches < 2 or n_samples < 2*n_batches or n_samples % n_batches:
            raise ValueError('Sampled radial guests require at least two batches, n_samples divisible '
                             'by n_batches, and at least two samples per batch.')
        if np.any(population.mean_displacement != 0):
            raise ValueError('Sampled radial guests require zero mean displacement for isotropic wall conditioning.')
        # Reject unsupported grids before computing any template pair sums.
        RadialMomentHistogram(q, 0.0, bin_width=bin_width, order=order)
    active = [(index, template, float(weight))
              for index, (template, weight) in enumerate(zip(population.library.templates,
                                                              population.library.weights)) if weight > 0]
    k_pores = len(population.centres)
    factors = form_factors(tuple(element for _, template, _ in active
                                 for element in template.elements), q)
    internal_one = sum((weight * direct_iq(template, q) for _, template, weight in active),
                       start=np.zeros(len(q)))
    internal = np.broadcast_to(internal_one, (k_pores, len(q))).copy()
    centres = population.centres + population.mean_displacement
    max_template_radius = max(float(np.linalg.norm(template.xyz, axis=1).max())
                              for _, template, _ in active)
    attempts = 0
    accepted = 0
    maximum_radius = 0.0
    wall_status = 'absent' if population.pore_radius is None else 'conditioned_by_rejection'

    if method == 'analytic':
        if population.pore_radius is not None:
            if population.position_mode == 'gaussian' and variance > 0:
                raise ValueError('A Gaussian proposal with a spherical wall requires sampled radial guests.')
            support = population.uniform_ball_radius if population.position_mode == 'uniform_ball' else 0.0
            largest_extent = np.linalg.norm(population.mean_displacement, axis=1) + support + max_template_radius
            if np.any(largest_extent > population.pore_radius - population.wall_clearance):
                raise ValueError('The spherical wall may bind; use sampled radial guests with zero mean shift.')
            wall_status = 'certified_nonbinding_by_triangle_support_bound'
        template_mean = np.zeros(len(q))
        for _, template, weight in active:
            radii = np.linalg.norm(template.xyz, axis=1)
            for element in sorted(set(template.elements)):
                selected = np.asarray(template.elements) == element
                template_mean += weight * factors[element] * weighted_sinc_sum(radii[selected], q)
        if population.position_mode == 'gaussian':
            characteristic = np.exp(-0.5 * variance * q*q)
        else:
            characteristic = _ball_characteristic(q * population.uniform_ball_radius)
        conditional_mean = template_mean * characteristic
        means = np.broadcast_to(conditional_mean, (1, 2, k_pores, len(q))).copy()
        bounds = np.zeros_like(means)
        effective_batches = 1
        per_batch = 0
    else:
        effective_batches = n_batches
        per_batch = n_samples // n_batches
        means = np.zeros((n_batches, 2, k_pores, len(q)))
        bounds = np.zeros_like(means)
        streams = iter(np.random.SeedSequence(int(seed)).spawn(n_batches * 2 * k_pores * len(active)))
        for batch in range(n_batches):
            for replica in range(2):
                for pore in range(k_pores):
                    for _, template, weight in active:
                        rng = np.random.default_rng(next(streams))
                        local = PorePopulation(
                            np.zeros((1, 3)), ParticleLibrary((template,), np.ones(1)), occupancy=1.0,
                            displacement_covariance_A2=population.covariance,
                            position_mode=population.position_mode,
                            uniform_ball_radius_A=population.uniform_ball_radius,
                            orientation='uniform',
                            pore_radius_A=None if population.pore_radius is None else population.pore_radius[pore],
                            wall_clearance_A=population.wall_clearance,
                            max_attempts=population.max_attempts)
                        # Only one template/replica/batch of radial samples is retained.
                        radii = np.empty((per_batch, len(template)))
                        for sample in range(per_batch):
                            draw = local.sample(rng)[0]
                            radii[sample] = np.linalg.norm(draw.cloud.xyz, axis=1)
                            attempts += draw.attempts
                        accepted += per_batch
                        largest = float(radii.max())
                        maximum_radius = max(maximum_radius, largest)
                        labels = np.asarray(template.elements)
                        for element in sorted(set(template.elements)):
                            histogram = RadialMomentHistogram(q, largest, bin_width=bin_width, order=order)
                            histogram.add(radii[:, labels == element].ravel())
                            radial_sum, radial_bound = histogram.evaluate()
                            means[batch, replica, pore] += weight * factors[element] * radial_sum / per_batch
                            bounds[batch, replica, pore] += weight * abs(factors[element]) * radial_bound / per_batch
    zeros = np.flatnonzero(q == 0)
    if zeros.size:
        # Exact conditional forward amplitude: no occupancy or size-draw noise.
        forward = sum(weight * sum(factors[element][zeros[0]] for element in template.elements)
                      for _, template, weight in active)
        means[..., zeros] = forward
        bounds[..., zeros] = 0.0
    metadata = dict(
        method=method, statistic='conditional_on_occupied_pore',
        orientation='Haar_uniform', position_mode=population.position_mode,
        spherical_wall=wall_status, batches=effective_batches, replicas_per_batch=2,
        n_samples_per_template_pore_replica=n_samples if method == 'sampled' else 0,
        accepted_samples_per_template_pore_replica_batch=per_batch,
        accepted_placements=accepted, proposal_attempts=attempts,
        n_positive_weight_templates=len(active), seed=int(seed),
        conditional_mean_atom_count=float(population.library.mean_atom_count),
        template_weights='exact_number_weights_not_sampled', occupancy='excluded_from_conditional_statistics',
        replica_streams='independent_SeedSequence_children_per_batch_replica_pore_template',
        internal_scattering='exact_number_weighted_template_Debye_rigid_invariant',
        radial_maximum_observed_A=maximum_radius if method == 'sampled' else None,
        radial_bin_width_requested_A=float(bin_width), radial_moment_order=order,
        numerical_bound='mean_bounds cover radial Taylor truncation only, excluding floating-point error',
        uncertainty='Monte Carlo integration uncertainty must be computed from independent batch intensity estimates; '
                    'it is separate from physical disorder variance and interpolation error',
        sample_memory_scope='one accepted radial array of samples_per_batch times template_atoms; no full-crystal guest draws')
    return RadialGuestStatistics(
        q=_readonly(q), means=_readonly(means), mean_bounds=_readonly(bounds),
        internal=_readonly(internal), internal_bounds=_readonly(np.zeros_like(internal)),
        centres=_readonly(centres), occupancy=_readonly(population.occupancy), metadata=metadata)
