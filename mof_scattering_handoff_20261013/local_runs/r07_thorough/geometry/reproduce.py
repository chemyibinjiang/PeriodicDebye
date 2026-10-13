"""Reproduce geometric oracle errors and exact sampling-law diagnostics.

Run from the handoff root with .venv/Scripts/python.exe followed by this path.
The project and the stress test module are resolved relative to this file.
"""
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.stats import kstest, truncnorm


output = Path(__file__).resolve().parent
project = output.parents[2] / 'mof_debye_lab'
sys.path.insert(0, str(project / 'src'))
spec = spec_from_file_location('geometry_stress', project / 'tests' / 'test_geometry_stress.py')
checks = module_from_spec(spec)
spec.loader.exec_module(checks)

from mofdebye.core import AtomCloud
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.geometry import PeriodicHostTree, contains_cell_mask


report = {
    'scope': 'Known-solution geometry and conditioned sampling, no performance benchmarks.',
    'nearest_image_equation': 'min_(atom j, integer image n) ||(q_frac - f_j - n) @ cell||',
    'membership_equation': 'point is in the union iff any selected integer cell has all three fractional coordinates in its closed [i,i+1] intervals, allowing the declared plane tolerance.',
    'sampling_equation': 'Choose occupied Bernoulli(p), choose size with number weights w, then condition N(mu,sigma^2) on [-L+radius, L-radius] at that fixed size.',
    'nearest_image': [], 'cell_mask': [], 'conditional_sampling': [],
}
for seed in range(12):
    rng = np.random.default_rng(87100 + seed)
    cell = (np.array([[2.1, 0, 0], [8.7, 1.4, 0], [.8, -.3, 2.8]])
            if seed % 3 == 0 else np.array([[4.2, 0, 0], [1.7, 3.1, 0], [.8, -.9, 5.3]]))
    cell = cell @ Rotation.random(random_state=rng).as_matrix().T
    if seed % 2:
        cell[1] *= -1
    fractional_basis = rng.uniform(.01, .99, (5, 3))
    fractional_queries = rng.uniform(.01, .99, (15, 3))
    expected, expected_indices = checks._exhaustive_images(fractional_basis, cell, fractional_queries)
    atom_shifts = rng.integers(-100_000, 100_000, (5, 3))
    query_shifts = rng.integers(-100_000, 100_000, (15, 3))
    basis = AtomCloud(('C', 'H', 'O', 'Zn', 'Au'), (fractional_basis + atom_shifts) @ cell)
    observed, indices = PeriodicHostTree(basis, cell, query_chunk=7).query((fractional_queries + query_shifts) @ cell)
    report['nearest_image'].append({
        'seed': 87100 + seed, 'basis_atoms': 5, 'query_count': 15,
        'cell_vectors_A': cell.tolist(), 'determinant_A3': float(np.linalg.det(cell)),
        'exhaustive_image_extent': 14, 'distance_tolerance_A': 3e-9,
        'max_absolute_distance_error_A': float(np.max(np.abs(observed - expected))),
        'basis_index_mismatches': int(np.sum(indices != expected_indices)),
        'expected_distances_A': expected.tolist(), 'observed_distances_A': observed.tolist(),
    })
for seed in range(8):
    # The test itself asserts every point's result against an independent union.
    checks.test_cell_mask_matches_union_of_every_selected_parallelepiped(seed)
    report['cell_mask'].append({'seed': 6600 + seed, 'oblique_points': 1030,
                               'exact_zero_tolerance_points': 62, 'mismatches': 0})
for seed in range(4):
    checks.test_spherical_atom_cut_commutes_with_rotation_translation_and_order(seed)
report['sphere_cut'] = {'seeds': [9130, 9131, 9132, 9133], 'candidate_atoms_per_case': 150,
                        'rotation_translation_species_order_and_metadata_invariance': 'passed',
                        'input_mutation': False}
for seed in [2814, 6901]:
    monomer = AtomCloud(('Au',), [[0, 0, 0]])
    dimer = AtomCloud(('Au', 'Au'), [[-.75, 0, 0], [.75, 0, 0]])
    library = ParticleLibrary((monomer, dimer), [.35, .65])
    sigma, mean, occupancy, samples = .65, .3, .63, 6000
    population = PorePopulation([[0, 0, 0]], library, occupancy=occupancy,
        displacement_covariance_A2=np.diag([sigma**2, 0., 0.]),
        mean_displacement_A=[mean, 0, 0], orientation='fixed',
        external_boundary={'kind': 'box', 'min_A': [-1, -1, -1], 'max_A': [1, 1, 1]},
        guest_retention='whole_particle')
    rng = np.random.default_rng(seed)
    positions, attempts = [[], []], [[], []]
    for _ in range(samples):
        for draw in population.sample(rng):
            positions[draw.template_index].append(draw.displacement_A[0])
            attempts[draw.template_index].append(draw.attempts)
    counts = np.asarray([len(group) for group in positions])
    entry = {'seed': seed, 'trials': samples, 'occupied_pores': int(counts.sum()),
             'expected_occupancy': occupancy, 'observed_occupancy': float(counts.sum()/samples),
             'occupancy_error_SEM': float(abs(counts.sum()/samples-occupancy)/np.sqrt(occupancy*(1-occupancy)/samples)),
             'expected_size_weights': library.weights.tolist(),
             'observed_size_weights': (counts/counts.sum()).tolist(), 'templates': []}
    for template, half_width in enumerate([1., .25]):
        law = truncnorm((-half_width-mean)/sigma, (half_width-mean)/sigma, loc=mean, scale=sigma)
        expected_mean, expected_var = law.stats(moments='mv')
        sample = np.asarray(positions[template])
        entry['templates'].append({'template_index': template, 'accepted_interval_A': [-half_width, half_width],
            'sample_count': len(sample), 'expected_mean_A': float(expected_mean),
            'observed_mean_A': float(sample.mean()), 'expected_variance_A2': float(expected_var),
            'observed_variance_A2': float(sample.var(ddof=1)),
            'mean_error_SEM': float(abs(sample.mean()-expected_mean)/np.sqrt(expected_var/len(sample))),
            'KS_statistic': float(kstest(sample, law.cdf).statistic),
            'mean_attempts': float(np.mean(attempts[template]))})
    report['conditional_sampling'].append(entry)
checks.test_sampling_is_seed_reproducible_with_periodic_clearance_and_external_boundary()
report['fixed_seed'] = {'seed': 9124, 'two_independent_generators': 'bitwise equal',
                      'draws_per_population': 30, 'populations': ['periodic_host_clearance', 'whole_particle_box_containment']}
report['summary'] = {
    'nearest_queries': sum(row['query_count'] for row in report['nearest_image']),
    'max_nearest_distance_error_A': max(row['max_absolute_distance_error_A'] for row in report['nearest_image']),
    'nearest_basis_index_mismatches': sum(row['basis_index_mismatches'] for row in report['nearest_image']),
    'cell_mask_points': sum(row['oblique_points']+row['exact_zero_tolerance_points'] for row in report['cell_mask']),
    'max_conditional_mean_error_SEM': max(t['mean_error_SEM'] for row in report['conditional_sampling'] for t in row['templates']),
    'max_KS_statistic': max(t['KS_statistic'] for row in report['conditional_sampling'] for t in row['templates']),
}
(output / 'known_solution_metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
print(json.dumps(report['summary'], indent=2))
