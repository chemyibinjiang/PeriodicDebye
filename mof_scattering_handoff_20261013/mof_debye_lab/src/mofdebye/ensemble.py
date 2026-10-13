"""Probability-averaged Debye pair contributions, analytical or explicitly sampled."""
from __future__ import annotations
from itertools import combinations
import numpy as np
from .core import form_factors, direct_iq, cross_iq, weighted_sinc_sum, validate_q, concatenate, _positive_integer
from .distributions import PorePopulation

COMPONENTS = ('host', 'guest_internal', 'host_guest', 'guest_guest', 'total')

def radial_atom_sum(cloud, q, centre=(0., 0., 0.)):
    """sum_i f_i(q) sinc(q*distance(atom_i,centre)); not an intensity."""
    f = form_factors(cloud.elements, q)
    out = np.zeros(len(q))
    labels = np.asarray(cloud.elements)
    for el in f:
        radii = np.linalg.norm(cloud.xyz[labels == el] - np.asarray(centre), axis=1)
        out += f[el] * weighted_sinc_sum(radii, q)
    return out

def analytic_ensemble(host, population: PorePopulation, q, host_iq=None):
    """Exact FINITE-ensemble average for independent pores, uniform rotations,
    independent size catalogue and unbounded isotropic Gaussian centre jitter.
    Works with any rigid atomic template shape. No infinite-lattice limit is taken.
    """
    q = validate_q(q)
    sigma = population.analytic_sigma()
    p = population.occupancy
    centres = population.centres + population.mean_displacement
    library = population.library
    hh = direct_iq(host, q) if host_iq is None else np.asarray(host_iq)
    template_intensities = np.stack([direct_iq(t, q) for t in library.templates])
    radial_sums = np.stack([radial_atom_sum(t, q) for t in library.templates])
    mean_particle_iq = library.weights @ template_intensities
    mean_radial = library.weights @ radial_sums
    # Same-particle pairs, including self, get p, not p^2.
    intra = p.sum() * mean_particle_iq
    centre_jitter_factor = np.exp(-.5 * (q*sigma)**2)
    hg = np.zeros(len(q))
    for centre, probability in zip(centres, p):
        hg += 2 * probability * centre_jitter_factor * mean_radial * radial_atom_sum(host, q, centre)
    gg = np.zeros(len(q))
    for i, j in combinations(range(len(centres)), 2):
        distance = np.linalg.norm(centres[i] - centres[j])
        gg += 2 * p[i] * p[j] * centre_jitter_factor**2 * mean_radial**2 * np.sinc(q*distance/np.pi)
    return dict(host=hh, guest_internal=intra, host_guest=hg, guest_guest=gg,
                total=hh + intra + hg + gg)

def sampled_ensemble(host, population: PorePopulation, q, n_samples=512, seed=20261013,
                     verify_full_first=4, retain_samples=False):
    """Explicit random finite configurations, exact Debye cross-pair calculations.
    Cache the fixed host and rigid-particle internal terms. This is algebraically
    identical to full-system Debye summation, checked for the first configurations.
    SEM describes Monte Carlo integration error, not uncertainty in the material.
    Welford accumulators retain O(components * q) statistics. Full per-configuration
    curves are allocated only when retain_samples=True; random draws are identical.
    """
    n_samples = _positive_integer(n_samples, 'n_samples')
    if n_samples < 2:
        raise ValueError('At least two samples are required for a standard error.')
    q = validate_q(q); rng = np.random.default_rng(seed)
    hh = direct_iq(host, q)
    cache = [direct_iq(t, q) for t in population.library.templates]
    values = np.zeros((n_samples, len(COMPONENTS), len(q))) if retain_samples else None
    mean = np.zeros((len(COMPONENTS), len(q)))
    m2 = np.zeros_like(mean)
    occupancies = np.zeros(len(population.centres), dtype=int)
    template_counts = np.zeros(len(population.library.templates), dtype=int)
    attempt_sum = 0
    attempt_count = 0
    explicit_max_error = 0.0
    for sample in range(n_samples):
        draws = population.sample(rng)
        intra = np.zeros(len(q)); hg = np.zeros(len(q)); gg = np.zeros(len(q))
        for draw in draws:
            intra += cache[draw.template_index]
            hg += cross_iq(host, draw.cloud, q)
            occupancies[draw.pore_index] += 1
            template_counts[draw.template_index] += 1
            attempt_sum += draw.attempts
            attempt_count += 1
        for a, b in combinations(draws, 2):
            gg += cross_iq(a.cloud, b.cloud, q)
        total = hh + intra + hg + gg
        current = np.stack([hh, intra, hg, gg, total])
        delta = current - mean
        mean += delta / (sample + 1)
        m2 += delta * (current - mean)
        if retain_samples:
            values[sample] = current
        if sample < verify_full_first:
            full = direct_iq(concatenate([host] + [d.cloud for d in draws]), q)
            explicit_max_error = max(explicit_max_error, float(np.max(np.abs(total-full))/max(np.max(np.abs(full)), 1)))
    sem = np.sqrt(np.maximum(m2, 0) / (n_samples - 1) / n_samples)
    result = {
        'mean': {name: mean[k] for k, name in enumerate(COMPONENTS)},
        'sem': {name: sem[k] for k, name in enumerate(COMPONENTS)},
        'n_samples': n_samples, 'seed': seed,
        'occupancy_counts': occupancies, 'template_counts': template_counts,
        'mean_placement_attempts': float(attempt_sum / attempt_count) if attempt_count else 0,
        'max_full_sum_relative_error': explicit_max_error,
    }
    if retain_samples:
        result['samples'] = values
    return result

def accelerated_sampled_ensemble(host, population: PorePopulation, q, n_samples=512,
                                  seed=20261013, precision='float64', device='cpu',
                                  force_grid=False, num_threads=1):
    """End-to-end sampling + adapted upstream forward kernel on every configuration.
    Returns total intensity and Monte Carlo SEM. Unlike sampled_ensemble, this
    intentionally does not compute separate reference partials. It still visits
    all atom pairs for each snapshot; it is not an optimal cached ensemble engine.
    """
    from .core import upstream_kernel_iq
    if n_samples < 2:
        raise ValueError('At least two Monte Carlo samples required.')
    q = validate_q(q); rng = np.random.default_rng(seed)
    mean = np.zeros(len(q)); m2 = np.zeros(len(q)); counts = {}
    for sample in range(n_samples):
        draws = population.sample(rng)
        full = concatenate([host] + [d.cloud for d in draws])
        values, info = upstream_kernel_iq(full, q, method='grid', precision=precision,
                              device=device, force_grid=force_grid, num_threads=num_threads, return_info=True)
        method = info['actual_method']; counts[method] = counts.get(method, 0) + 1
        delta = values - mean
        mean += delta/(sample + 1)
        m2 += delta*(values - mean)
    return dict(mean={'total':mean}, sem={'total':np.sqrt(np.maximum(m2,0)/(n_samples-1)/n_samples)},
                n_samples=n_samples, seed=seed, actual_kernel_counts=counts,
                backend='adapted DebyeCalculator forward kernel, not installed public API')
