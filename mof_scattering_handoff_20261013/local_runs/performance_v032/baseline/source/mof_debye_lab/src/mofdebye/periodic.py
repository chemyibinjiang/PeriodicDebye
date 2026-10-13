"""Per-cell Bragg measures and continuous diffuse intensity for independent cells.

Direct and reciprocal lattice vectors are ROWS. Reflection weights include
(2*pi)^3 / cell_volume and multiply delta^3(q-G); they are not peak heights.
"""
from __future__ import annotations
import itertools
import numpy as np
from .core import validate_q
from .directional import amplitude, validate_q_vectors, lattice_shape_amplitude, powder_average
from .moments import GuestMoments, guest_moments


def validate_cell(cell):
    cell = np.asarray(cell, dtype=float)
    if cell.shape != (3, 3) or not np.isfinite(cell).all() or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError('cell must contain three finite, independent row vectors in angstrom.')
    return cell


def reciprocal_lattice(cell):
    return 2 * np.pi * np.linalg.inv(validate_cell(cell)).T


def enumerate_reflections(cell, q_max, q_min=0., include_origin=False, max_candidates=2_000_000):
    cell = validate_cell(cell)
    if not np.isfinite([q_min, q_max]).all() or not 0 <= q_min <= q_max:
        raise ValueError('Require finite 0 <= q_min <= q_max.')
    # h_i = q dot a_i / (2*pi): valid even for highly oblique cells.
    bounds = np.ceil(q_max * np.linalg.norm(cell, axis=1) / (2*np.pi)).astype(int)
    count = int(np.prod(2*bounds.astype(object)+1))
    if count > max_candidates:
        raise ValueError(f'Reflection search would visit {count} candidates; reduce q_max or raise max_candidates explicitly.')
    reciprocal = reciprocal_lattice(cell)
    retained = []
    for h in itertools.product(*(range(-int(b), int(b)+1) for b in bounds)):
        if not include_origin and h == (0, 0, 0):
            continue
        norm = np.linalg.norm(np.asarray(h) @ reciprocal)
        if q_min-1e-12 <= norm <= q_max+1e-12:
            retained.append(h)
    hkl = np.asarray(retained, dtype=int).reshape(-1, 3)
    vectors = hkl @ reciprocal
    q = np.linalg.norm(vectors, axis=1)
    order = np.argsort(q, kind='stable')
    return hkl[order], vectors[order], q[order]


def validate_periodic_population(population, cell):
    """Validate disjoint pore walls against ALL potentially overlapping lattice images."""
    if population is None:
        return
    if getattr(population, 'external_boundary', None) is not None:
        raise ValueError('Periodic populations cannot have a finite external boundary.')
    if population._host_tree is not None:
        if not getattr(population._host_tree, 'is_periodic', False):
            raise ValueError('Periodic host-clearance requires periodic neighbor handling.')
        if cell is not None and not np.allclose(population._host_tree.cell, validate_cell(cell), rtol=1e-12, atol=1e-12):
            raise ValueError('Periodic host-clearance lattice differs from the scattering cell.')
    if population.pore_radius is None:
        return
    if cell is None:
        raise ValueError('Periodic hard-wall validation requires cell vectors.')
    cell=validate_cell(cell)
    inverse=np.linalg.inv(cell)
    fractional=(population.centres@inverse)%1.
    for i in range(len(fractional)):
        for j in range(i,len(fractional)):
            delta=fractional[j]-fractional[i]
            radius=population.pore_radius[i]+population.pore_radius[j]
            bounds=radius*np.linalg.norm(inverse,axis=0)
            lower=np.ceil(-delta-bounds-1e-12).astype(int)
            upper=np.floor(-delta+bounds+1e-12).astype(int)
            if np.prod((upper-lower+1).astype(object))>500_000:
                raise ValueError('Periodic wall validation search too large; reduce pore radii or cell skew.')
            for shift in itertools.product(*(range(int(a),int(b)+1) for a,b in zip(lower,upper))):
                if i==j and shift==(0,0,0):
                    continue
                if np.linalg.norm((delta+shift)@cell)<radius-1e-10:
                    raise ValueError('Overlapping periodic hard-wall pores require an inter-particle exclusion model.')


def cell_guest_moments(population, q_vectors, method='analytic', n_samples=512, seed=20261013, cell=None):
    q_vectors = validate_q_vectors(q_vectors)
    if population is None:
        zeros = np.zeros(len(q_vectors))
        return GuestMoments(zeros.astype(complex), zeros.copy(), zeros.copy(), {'method': 'no_guests'})
    validate_periodic_population(population,cell)
    return guest_moments(population, q_vectors, method=method, n_samples=n_samples, seed=seed)


def finite_cell_ensemble(host_cell, cell, mask, q_vectors, moments, independent_cells=True):
    """Expected finite intensity for iid disordered cells, including MC bias correction."""
    if not independent_cells:
        raise ValueError('Correlated cells need a lattice covariance model, which is unsupported.')
    shape = lattice_shape_amplitude(validate_cell(cell), mask, q_vectors)
    host = amplitude(host_cell, q_vectors)
    if moments.mean.shape != host.shape:
        raise ValueError('Moment and q-vector lengths differ.')
    coherent_cell = abs(host + moments.mean)**2 - moments.mean_variance
    coherent = abs(shape)**2 * coherent_cell
    diffuse = int(np.asarray(mask).sum()) * moments.variance
    return {'total': coherent + diffuse, 'coherent': coherent, 'diffuse': diffuse,
            'mean_variance_correction': abs(shape)**2 * moments.mean_variance}


def bragg_reflections(host_cell, cell, q_max, q_min=0., population=None,
                      moment_method='analytic', n_samples=512, seed=20261013,
                      independent_cells=True):
    if not independent_cells:
        raise ValueError('Correlated cells are unsupported; an independent-cell ensemble must be declared.')
    cell = validate_cell(cell)
    hkl, vectors, q = enumerate_reflections(cell, q_max, q_min)
    # Always calculate forward scattering separately, even for an empty nonzero table.
    all_vectors = np.vstack((np.zeros((1, 3)), vectors))
    moments = cell_guest_moments(population, all_vectors, moment_method, n_samples, seed, cell)
    host = amplitude(host_cell, all_vectors)
    f_squared = abs(host + moments.mean)**2 - moments.mean_variance
    prefactor = (2*np.pi)**3 / abs(np.linalg.det(cell))
    return {'hkl': hkl, 'q_vectors': vectors, 'q': q,
            'f_squared': f_squared[1:], 'bragg_weight': prefactor*f_squared[1:],
            'diffuse_at_reflections': moments.variance[1:],
            'mean_variance': moments.mean_variance[1:],
            'forward_weight': float(prefactor*f_squared[0]),
            'measure_prefactor': float(prefactor), 'moment_metadata': moments.metadata}


def powder_shells(reflections, rtol=1e-10, atol=1e-12):
    """Radial density weights multiplying delta(q-|G|); all reciprocal vectors counted once."""
    q = np.asarray(reflections['q'])
    if np.any(q <= 0):
        raise ValueError('Forward reflection must be separated before radial projection.')
    groups = []
    for index in np.argsort(q):
        if not groups or not np.isclose(q[index], q[groups[-1][0]], rtol=rtol, atol=atol):
            groups.append([index])
        else:
            groups[-1].append(index)
    return {
        'q': np.array([q[g[0]] for g in groups]),
        'multiplicity': np.array([len(g) for g in groups], dtype=int),
        'sum_f_squared': np.array([np.asarray(reflections['f_squared'])[g].sum() for g in groups]),
        'radial_weight': np.array([np.sum(np.asarray(reflections['bragg_weight'])[g]/(4*np.pi*q[g]**2)) for g in groups]),
    }


def powder_diffuse(population, q, moment_method='analytic', n_samples=512,
                   seed=20261013, n_mu=24, n_phi=48, cell=None):
    q = validate_q(q)
    if population is None:
        return np.zeros(len(q))
    def evaluate(vectors):
        return cell_guest_moments(population, vectors, moment_method, n_samples, seed, cell).variance
    return powder_average(evaluate, q, n_mu=n_mu, n_phi=n_phi)
