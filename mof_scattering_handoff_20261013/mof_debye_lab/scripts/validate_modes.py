#!/usr/bin/env python3
"""Reproducible CPU validation, timings, memory probes and scientific figures.

Run from the project root. Fixtures are synthetic numerical models, not MOFs.
All timing measurements use one CPU thread, one warmup and three repetitions.
No instrument profile is added to the separate bulk Bragg/diffuse measures.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import itertools
import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
import torch
from threadpoolctl import threadpool_info, threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from mofdebye.core import AtomCloud, concatenate, direct_iq, form_factors, upstream_kernel_iq
from mofdebye.directional import (amplitude, intensity, cell_shape_amplitude,
                                  lattice_shape_amplitude, powder_average)
from mofdebye.distributions import ParticleLibrary, PorePopulation
from mofdebye.lattice import LatticePairCache
from mofdebye.moments import (exact_guest_moments, finite_ensemble_intensity,
                              guest_moments)
from mofdebye.periodic import (bragg_reflections, finite_cell_ensemble,
                               powder_diffuse, powder_shells, reciprocal_lattice)

SEED = 20261013


def cpu_model():
    if os.name == 'nt':
        import winreg
        try:
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as key:
                return winreg.QueryValueEx(key, 'ProcessorNameString')[0].strip()
        except OSError:
            pass
    return platform.processor()


def process_memory():
    """Windows native process totals; the peak is a process-lifetime high water."""
    if os.name != 'nt':
        return {'available': False, 'reason': 'Windows GetProcessMemoryInfo unavailable'}
    class Counters(ctypes.Structure):
        _fields_ = [('cb', ctypes.c_ulong), ('PageFaultCount', ctypes.c_ulong)] + [
            (name, ctypes.c_size_t) for name in (
                'PeakWorkingSetSize', 'WorkingSetSize', 'QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage', 'QuotaPeakNonPagedPoolUsage',
                'QuotaNonPagedPoolUsage', 'PagefileUsage', 'PeakPagefileUsage')]
    counters = Counters()
    counters.cb = ctypes.sizeof(counters)
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.GetCurrentProcess.restype = ctypes.c_void_p
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
    ok = psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb)
    if not ok:
        return {'available': False, 'windows_error': ctypes.get_last_error()}
    return dict(available=True, working_set_bytes=counters.WorkingSetSize,
                process_lifetime_peak_working_set_bytes=counters.PeakWorkingSetSize,
                pagefile_bytes=counters.PagefileUsage,
                scope='whole process including Python, loaded libraries, native arrays and prior phases; not per-method peak')


def timed(call):
    """Memory probes run separately so tracemalloc does not distort timings."""
    before = process_memory()
    call()
    seconds = []
    for _ in range(3):
        started = time.perf_counter()
        value = call()
        seconds.append(time.perf_counter() - started)
    tracemalloc.start()
    probe = call()
    current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del probe
    return value, dict(warmups=1, repeats_seconds=seconds, median_seconds=float(np.median(seconds)),
                       tracemalloc_probe_current_bytes=current, tracemalloc_probe_peak_bytes=peak,
                       tracemalloc_scope='separate evaluation; Python and registered NumPy allocations, not all native/Torch memory',
                       process_before=before, process_after=process_memory())


def residuals(actual, reference, q=None):
    actual, reference = np.asarray(actual), np.asarray(reference)
    delta = np.abs(actual - reference)
    scale = max(float(np.max(np.abs(reference))), np.finfo(float).tiny)
    weak = np.abs(reference) <= np.quantile(np.abs(reference), .2)
    result = dict(relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(reference), np.finfo(float).tiny)),
                  scaled_max=float(np.max(delta) / scale),
                  max_absolute=float(np.max(delta)),
                  weakest_20_percent_max_absolute=float(np.max(delta[weak])))
    if q is not None:
        result['scaled_max_by_q_band'] = {}
        for label, selected in [('all', q >= 0), ('q_ge_1', q >= 1), ('q_ge_5', q >= 5)]:
            if selected.any():
                result['scaled_max_by_q_band'][label] = float(
                    np.max(delta[selected]) / max(np.max(np.abs(reference[selected])), np.finfo(float).tiny))
    return result


def fixture():
    cell = np.array([[4.8, 0., 0.], [1.1, 4.2, 0.], [.4, .7, 5.1]])
    fractional = np.array([[.08, .14, .23], [.31, .26, .41], [.64, .73, .57], [.81, .37, .79]])
    basis = AtomCloud(('C', 'O', 'Zn', 'C'), fractional @ cell, 'synthetic triclinic numerical basis')
    return cell, basis


def run_benchmarks(out, arrays, cell, basis):
    q = np.linspace(0, 8, 161)
    arrays['benchmark_q'] = q
    rng = np.random.default_rng(SEED)
    directions = rng.normal(size=(len(q), 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    qv = q[:, None] * directions
    arrays['benchmark_q_vectors'] = qv
    rows = []
    for n in (2, 3, 4):
        mask = np.ones((n, n, n), dtype=bool)
        cloud = concatenate(basis.moved(index @ cell) for index in np.argwhere(mask))
        cached, construction = timed(lambda: LatticePairCache(basis, cell, mask))
        streamed, streaming_construction = timed(lambda: LatticePairCache(basis, cell, mask, cache_max_bytes=0, translation_chunk=32))
        reference, reference_time = timed(lambda: direct_iq(cloud, q))
        methods = {}
        outputs = {'reference': reference}
        for mode in ('direct', 'grid'):
            result, timing = timed(lambda mode=mode: upstream_kernel_iq(
                cloud, q, method=mode, force_grid=(mode == 'grid'), return_info=True,
                precision='float64', num_threads=1, device='cpu'))
            actual, metadata = result
            error = residuals(actual, reference, q)
            assert max(error['scaled_max_by_q_band'].values()) < 1e-9, (n, mode, error)
            methods['adapted_' + mode] = dict(timing=timing, errors=error, metadata=metadata)
            outputs['adapted_' + mode] = actual
        for name, cache in [('lattice_cached', cached), ('lattice_streamed', streamed)]:
            actual, timing = timed(lambda cache=cache: cache.iq(q))
            error = residuals(actual, reference, q)
            assert max(error['scaled_max_by_q_band'].values()) < 1e-11, (n, name, error)
            methods[name] = dict(timing=timing, errors=error, metadata=cache.metadata)
            outputs[name] = actual
        directional, explicit_time = timed(lambda: amplitude(cloud, qv))
        factorized, factorized_time = timed(lambda: cell_shape_amplitude(basis, cell, mask, qv))
        factor_error = residuals(factorized, directional, q)
        assert factor_error['scaled_max'] < 1e-12, factor_error
        row = dict(shape=list(mask.shape), n_cells=int(mask.sum()), n_atoms=len(cloud),
                   input_sha256=hashlib.sha256(cloud.xyz.tobytes() + ','.join(cloud.elements).encode()).hexdigest(),
                   cached_preprocessing=construction, streaming_preprocessing=streaming_construction,
                   reference=reference_time, methods=methods,
                   directional_explicit=explicit_time, directional_factorized=factorized_time,
                   directional_amplitude_errors=factor_error)
        rows.append(row)
        for name, values in outputs.items():
            arrays[f'n{n}_{name}'] = values
        arrays[f'n{n}_directional_explicit'] = directional
        arrays[f'n{n}_directional_factorized'] = factorized
        np.savetxt(out / f'finite_powder_n{n}.csv', np.column_stack([q, *outputs.values()]), delimiter=',',
                   header='q_inverse_A,' + ','.join(outputs), comments='')
        print(f'Benchmarked {n}^3 cells / {len(cloud)} atoms', flush=True)
    return rows


def cross_mode_checks(out, arrays, cell, basis):
    mask = np.ones((2, 2, 2), bool)
    cloud = concatenate(basis.moved(index @ cell) for index in np.argwhere(mask))
    q = np.linspace(0, 6, 31)
    expected = direct_iq(cloud, q)
    arrays['angular_q'] = q
    arrays['angular_reference'] = expected
    rows = []
    previous = None
    for n_mu in (24, 48, 96):
        started = time.perf_counter()
        actual = powder_average(lambda v: abs(cell_shape_amplitude(basis, cell, mask, v))**2,
                                q, n_mu=n_mu, n_phi=2*n_mu, q_chunk=2)
        error = residuals(actual, expected, q)
        row = dict(n_mu=n_mu, n_phi=2*n_mu, seconds=time.perf_counter()-started,
                   errors_vs_debye=error,
                   errors_vs_previous=None if previous is None else residuals(actual, previous, q))
        rows.append(row)
        arrays[f'angular_{n_mu}'] = actual
        previous = actual
        print(f'Angular refinement {n_mu} x {2*n_mu}: scaled max {error["scaled_max"]:.3g}', flush=True)
    assert max(rows[-1]['errors_vs_debye']['scaled_max_by_q_band'].values()) < 1e-9, rows[-1]
    np.savetxt(out / 'angular_refinement.csv', np.column_stack([q, expected, *[arrays[f'angular_{n}'] for n in (24, 48, 96)]]),
               delimiter=',', header='q_inverse_A,debye,angular24,angular48,angular96', comments='')
    return rows


def ensemble_checks(arrays, cell, basis):
    guest = AtomCloud(('Au',), [[0, 0, 0]])
    centre = np.array([.53, .52, .51]) @ cell
    probability = .37
    population = PorePopulation([centre], ParticleLibrary((guest,), [1]), occupancy=probability)
    qv = arrays['benchmark_q_vectors'][::8]
    moments = guest_moments(population, qv)
    mask = np.ones((3, 1, 1), bool)
    predicted = finite_cell_ensemble(basis, cell, mask, qv, moments)
    exact = np.zeros(len(qv))
    for flags in itertools.product((0, 1), repeat=3):
        clouds = [basis.moved(i * cell[0]) for i in range(3)]
        clouds.extend(guest.moved(centre + i * cell[0]) for i, flag in enumerate(flags) if flag)
        weight = np.prod([probability if flag else 1-probability for flag in flags])
        exact += weight * intensity(concatenate(clouds), qv)
    error = residuals(predicted['total'], exact)
    assert error['scaled_max'] < 1e-12, error
    native_f = form_factors(('Au',), np.linalg.norm(qv, axis=1))['Au']
    variance_error = residuals(moments.variance, probability*(1-probability)*native_f**2)
    assert variance_error['scaled_max'] < 1e-12
    repeated_cell = abs(lattice_shape_amplitude(cell, mask, qv))**2 * (abs(amplitude(basis, qv)+moments.mean)**2+moments.variance)
    assert residuals(repeated_cell, exact)['relative_l2'] > .01
    empty = AtomCloud((), np.empty((0, 3)))
    whole_particle = AtomCloud(('Au', 'Au'), [[-.7, 0, 0], [.7, 0, 0]]).moved(centre)
    exact_moments = exact_guest_moments([empty, whole_particle], [.63, .37], qv)
    finite = finite_ensemble_intensity(basis, qv, exact_moments)
    enumerated = .63*intensity(basis, qv)+.37*intensity(concatenate([basis, whole_particle]), qv)
    whole_particle_error = residuals(finite['total'], enumerated)
    assert whole_particle_error['scaled_max'] < 1e-12
    arrays.update(ensemble_q_vectors=qv, ensemble_iid_exact=exact,
                  ensemble_iid_moments=predicted['total'], ensemble_repeated_random_cell=repeated_cell,
                  whole_particle_exact=enumerated, whole_particle_moments=finite['total'])
    return population, dict(occupancy_probability=probability, n_cells=3, enumerated_states=8,
                            iid_errors=error, bernoulli_variance_errors=variance_error,
                            repeated_random_cell_error=residuals(repeated_cell, exact),
                            whole_particle_occupancy_errors=whole_particle_error)


def bulk_checks(out, arrays, cell, basis, population):
    reciprocal = reciprocal_lattice(cell)
    np.testing.assert_allclose(cell @ reciprocal.T, 2*np.pi*np.eye(3), atol=1e-14)
    reflections = bragg_reflections(basis, cell, 6., population=population)
    shells = powder_shells(reflections)
    q = np.linspace(0, 6, 121)
    diffuse = powder_diffuse(population, q, n_mu=12, n_phi=24)
    p = float(population.occupancy[0])
    expected = p*(1-p)*form_factors(('Au',), q)['Au']**2
    diffuse_error = residuals(diffuse, expected, q)
    assert diffuse_error['scaled_max'] < 1e-12
    np.testing.assert_allclose(reflections['bragg_weight'], reflections['f_squared'] * (2*np.pi)**3 / abs(np.linalg.det(cell)), rtol=1e-14)
    np.testing.assert_allclose(shells['radial_weight'],
                               shells['sum_f_squared'] * (2*np.pi)**3 / abs(np.linalg.det(cell)) / (4*np.pi*shells['q']**2), rtol=1e-12)
    assert shells['multiplicity'].sum() == len(reflections['q'])
    for name in ('hkl', 'q_vectors', 'q', 'f_squared', 'bragg_weight'):
        arrays['bulk_reflection_' + name] = reflections[name]
    for name, value in shells.items():
        arrays['bulk_shell_' + name] = value
    arrays['bulk_diffuse_q'], arrays['bulk_diffuse'] = q, diffuse
    np.savetxt(out / 'bulk_reflections.csv', np.column_stack([reflections['hkl'], reflections['q'], reflections['bragg_weight']]),
               delimiter=',', header='h,k,l,q_inverse_A,bragg_weight_electron2_inverse_A3', comments='')
    np.savetxt(out / 'bulk_powder_shells.csv', np.column_stack([shells['q'], shells['multiplicity'], shells['radial_weight']]),
               delimiter=',', header='q_inverse_A,multiplicity,radial_weight_electron2_inverse_A', comments='')
    np.savetxt(out / 'bulk_diffuse.csv', np.column_stack([q, diffuse]), delimiter=',',
               header='q_inverse_A,diffuse_electron2_per_cell', comments='')
    # Fejer convergence validates distributional geometry, not a line-shape fit.
    t = (np.arange(2048)+.5)/2048 - .5
    vectors = t[:, None] * reciprocal[0]
    prefactor = (2*np.pi)**3 / abs(np.linalg.det(cell))
    rows = []
    arrays['fejer_fractional_h_offset'] = t
    for n in (2, 4, 8, 16, 32):
        kernel = abs(lattice_shape_amplitude(cell, np.ones((n, 1, 1), bool), vectors))**2/n
        formula = (np.sin(np.pi*n*t)/np.sin(np.pi*t))**2/n
        np.testing.assert_allclose(kernel, formula, atol=2e-12, rtol=1e-9)
        mass = prefactor * kernel.mean()**3
        assert abs(mass/prefactor-1) < 1e-12
        observable = float(np.mean(kernel*np.cos(2*np.pi*t))**3)
        rows.append(dict(linear_cells=n, cells_in_3d=n**3, geometric_integrated_mass=mass,
                         bulk_measure_prefactor=prefactor, mass_relative_error=abs(mass/prefactor-1),
                         smooth_cosine_product_integral=observable,
                         exact_smooth_integral=(1-1/n)**3, bulk_smooth_limit=1.,
                         smooth_limit_error=abs(1-observable)))
        arrays[f'fejer_kernel_n{n}'] = kernel
    assert np.all(np.diff([r['smooth_limit_error'] for r in rows]) < 0)
    return dict(n_nonzero_reflections=len(reflections['q']), n_shells=len(shells['q']),
                reciprocal_identity_error=float(np.max(abs(cell@reciprocal.T-2*np.pi*np.eye(3)))),
                diffuse_errors=diffuse_error, forward_weight_separate=reflections['forward_weight'],
                normalization='per cell: Bragg W_G=(2pi)^3/V |mean F(G)|^2 multiplying delta^3(q-G); radial W_G/(4pi G^2) multiplying delta(q-G)',
                fejer_convergence=rows,
                fejer_scope='production shape amplitude compared to exact kernel; integrated geometric mass and smooth observable over reciprocal primitive cell')


def figures(out, arrays, cell, basis):
    reciprocal = reciprocal_lattice(cell)
    axis = np.linspace(-1.6, 1.6, 161)
    h, k = np.meshgrid(axis, axis)
    vectors = np.column_stack([h.ravel(), k.ravel(), np.zeros(h.size)]) @ reciprocal
    image = abs(cell_shape_amplitude(basis, cell, np.ones((4, 4, 4), bool), vectors))**2
    image = image.reshape(h.shape)
    offset = np.linspace(-.8, .8, 1201)
    line_vectors = (1+offset[:, None])*reciprocal[0]
    lines = {}
    for n in (2, 4, 8):
        lines[n] = abs(lattice_shape_amplitude(cell, np.ones((n, n, n), bool), line_vectors))**2/n**6
    arrays.update(slice_h=h, slice_k=k, slice_intensity=image, line_h_offset=offset)
    for n, values in lines.items():
        arrays[f'line_shape_n{n}'] = values
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axs = plt.subplots(2, 2, figsize=(12.8, 9.2), layout='constrained')
    mesh = axs[0, 0].pcolormesh(h, k, np.maximum(image, image.max()*1e-8), shading='auto',
                               cmap='magma', norm=LogNorm(vmin=image.max()*1e-5, vmax=image.max()))
    fig.colorbar(mesh, ax=axs[0, 0], label=r'$I(\mathbf{q})$ (electron$^2$)')
    axs[0, 0].set(title='Finite single crystal: 4 x 4 x 4 cells', xlabel='h (reciprocal basis)', ylabel='k (l = 0)')
    for n, values in lines.items():
        axs[0, 1].plot(offset, values, label=f'{n} x {n} x {n}')
    axs[0, 1].set(title='Finite-size fringes through (100)', xlabel='h - 1', ylabel=r'$|S|^2/N_{cell}^2$', ylim=(0, 1.04))
    axs[0, 1].legend(frameon=False)
    axs[1, 0].semilogy(arrays['angular_q'], arrays['angular_reference'], color='#2166ac', label='Finite Debye')
    axs[1, 0].semilogy(arrays['angular_q'], arrays['angular_96'], 'o', ms=3.5, mfc='none', color='#b2182b', label='Directional angular average')
    axs[1, 0].set(title='Finite PXRD cross-check: 2 x 2 x 2 cells', xlabel=r'q ($\AA^{-1}$)', ylabel=r'$I(q)$ (electron$^2$)')
    axs[1, 0].legend(frameon=False)
    last = axs[1, 1]
    last.vlines(arrays['bulk_shell_q'], 0, arrays['bulk_shell_radial_weight'], color='#2166ac', linewidth=.8, label='Bragg radial weights')
    last.set(title='Bulk PXRD: separate Bragg and diffuse measures', xlabel=r'q ($\AA^{-1}$)',
             ylabel=r'Bragg weight (electron$^2$ $\AA^{-1}$/cell)', xlim=(0, 6))
    right = last.twinx()
    right.plot(arrays['bulk_diffuse_q'], arrays['bulk_diffuse'], color='#b2182b', lw=1.8, label='Diffuse intensity')
    right.set_ylabel(r'Diffuse (electron$^2$/cell)', color='#b2182b')
    handles, labels = last.get_legend_handles_labels()
    handles_right, labels_right = right.get_legend_handles_labels()
    last.legend(handles + handles_right, labels + labels_right, loc='upper right', frameon=False)
    fig.suptitle('Three scattering modes | synthetic triclinic fixture | neutral IT92 factors', fontsize=14)
    fig.savefig(out / 'three_modes.png', dpi=180)
    fig.savefig(out / 'three_modes.pdf')
    plt.close(fig)
    np.savetxt(out / 'finite_shape_line.csv', np.column_stack([offset, *lines.values()]), delimiter=',',
               header='h_offset,shape2,shape4,shape8', comments='')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('../local_runs/validation_v02'))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cell, basis = fixture()
    arrays = {'cell_A': cell, 'basis_xyz_A': basis.xyz}
    with threadpool_limits(limits=1):
        report = dict(schema_version='0.2', status='running', seed=SEED,
                      environment=dict(python=sys.version, platform=platform.platform(), processor=platform.processor(), cpu_model=cpu_model(),
                                       torch=torch.__version__, numpy=np.__version__, matplotlib=matplotlib.__version__,
                                       gemmi=importlib.metadata.version('gemmi'), scipy=importlib.metadata.version('scipy'),
                                       cuda_available=torch.cuda.is_available(), torch_threads=torch.get_num_threads(),
                                       threadpools=threadpool_info()),
                      hardware_scope='CPU only; GPU not exercised', precision='float64',
                      fixture=dict(description='synthetic triclinic four-atom basis, full binary cell masks; not a chemical MOF',
                                   cell_rows_A=cell.tolist(), basis_elements=list(basis.elements), basis_xyz_A=basis.xyz.tolist()),
                      timing_protocol='one untimed warmup + 3 timed repetitions, one CPU thread; separate memory probe',
                      initial_process_memory=process_memory())
        report['benchmarks'] = run_benchmarks(args.out, arrays, cell, basis)
        report['angular_convergence'] = cross_mode_checks(args.out, arrays, cell, basis)
        population, report['ensemble_checks'] = ensemble_checks(arrays, cell, basis)
        report['bulk_checks'] = bulk_checks(args.out, arrays, cell, basis, population)
        figures(args.out, arrays, cell, basis)
        np.savez_compressed(args.out / 'raw_arrays.npz', **arrays)
        report['status'] = 'passed'
        report['final_process_memory'] = process_memory()
        report['artifacts'] = sorted(path.name for path in args.out.iterdir() if path.is_file())
        (args.out / 'validation.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(status='passed', output=str(args.out.resolve()),
                          angular_final_error=report['angular_convergence'][-1]['errors_vs_debye'],
                          iid_error=report['ensemble_checks']['iid_errors']), indent=2), flush=True)


if __name__ == '__main__':
    main()
