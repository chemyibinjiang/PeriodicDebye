#!/usr/bin/env python3
"""Validate and benchmark the actually installed DebyeCalculator 1.0.14 API.

Native Waasmaier--Kirfel factors and explicit factor-of-two normalization are
checked independently. There is no grid method in this installed release.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import numpy as np
from scipy.spatial.distance import pdist, cdist
import torch
from mofdebye.core import AtomCloud, direct_iq, form_factors, weighted_sinc_sum
from mofdebye.fixtures import finite_host
from mofdebye.installed import (_new_calculator, installed_iq, installed_metadata,
                               native_form_factors)


def native_reference(cloud, q):
    factors = native_form_factors(cloud.elements, q)
    labels = np.asarray(cloud.elements)
    result = np.zeros(len(q))
    for index, a in enumerate(factors):
        xyz = cloud.xyz[labels == a]
        result += len(xyz) * factors[a] ** 2
        result += 2 * factors[a] ** 2 * weighted_sinc_sum(pdist(xyz), q)
        for b in list(factors)[index + 1:]:
            result += 2 * factors[a] * factors[b] * weighted_sinc_sum(
                cdist(xyz, cloud.xyz[labels == b]).ravel(), q)
    return result


def errors(got, expected, q):
    delta = np.asarray(got) - expected
    bands = {}
    for low in (0., 1., 5.):
        mask = q >= low
        bands[f'q_ge_{low:g}'] = float(np.max(abs(delta[mask])) / np.max(abs(expected[mask])))
    weak = abs(expected) <= np.quantile(abs(expected), .2)
    return dict(relative_l2=float(np.linalg.norm(delta) / np.linalg.norm(expected)),
                scaled_max_by_band=bands,
                weak_feature_max_absolute_electron2=float(np.max(abs(delta[weak]))))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--out', type=Path, default=Path('results/installed_package.json'))
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA is unavailable; no GPU results were produced.')
    torch.set_num_threads(1)
    q = np.linspace(0., 8., 161)
    synchronize = torch.cuda.synchronize if args.device == 'cuda' else lambda: None
    calc = _new_calculator(q, args.device)
    scale_checks = []
    for elements, xyz in [(('Au',), [[1, 2, 3]]),
                          (('Au', 'O'), [[0, 0, 0], [3, 0, 0]]),
                          (('C', 'C'), [[0, 0, 0], [0, 0, 0]])]:
        cloud = AtomCloud(elements, xyz)
        expected = native_reference(cloud, q)
        _, native = calc.iq((list(elements), np.array(cloud.xyz)))
        adapted = installed_iq(cloud, q, device=args.device)
        np.testing.assert_allclose(native, expected / 2, rtol=2e-12, atol=1e-10)
        np.testing.assert_allclose(adapted, expected, rtol=2e-12, atol=1e-10)
        scale_checks.append(dict(elements=elements, native_q0=float(native[0]),
                                 conventional_q0=float(expected[0]),
                                 corrected_error=errors(adapted, expected, q)))
    rows = []
    arrays = {'q': q}
    for size in (2, 3):
        cloud, *_ = finite_host((size, size, size), n_basis=24)
        setup_start = time.perf_counter()
        calc = _new_calculator(q, args.device)
        structure = (list(cloud.elements), np.array(cloud.xyz))
        synchronize()
        setup_seconds = time.perf_counter() - setup_start
        calc.iq(structure)
        synchronize()
        repeats = []
        for _ in range(args.repeats):
            synchronize()
            start = time.perf_counter()
            _, native = calc.iq(structure)
            synchronize()
            repeats.append(time.perf_counter() - start)
        adapted, info = installed_iq(cloud, q, args.device, return_info=True)
        expected = native_reference(cloud, q)
        np.testing.assert_allclose(adapted, 2 * native, rtol=2e-12, atol=1e-8)
        numerical = errors(adapted, expected, q)
        assert max(numerical['scaled_max_by_band'].values()) < 1e-10, numerical
        gemmi_expected = direct_iq(cloud, q)
        rows.append(dict(n_atoms=len(cloud),
                         input_sha256=hashlib.sha256(cloud.xyz.tobytes() + ','.join(cloud.elements).encode()).hexdigest(),
                         setup_seconds=setup_seconds, warmups=1, public_iq_seconds=repeats,
                         median_public_iq_seconds=float(np.median(repeats)),
                         native_factor_arithmetic_errors=numerical,
                         gemmi_factor_model_difference=errors(adapted, gemmi_expected, q),
                         adapter_metadata=info))
        arrays[f'n{len(cloud)}_installed'] = adapted
        arrays[f'n{len(cloud)}_native_reference'] = expected
        arrays[f'n{len(cloud)}_gemmi_reference'] = gemmi_expected
    native_f = native_form_factors(('C', 'O', 'Zn', 'Au'), q)
    common_f = form_factors(tuple(native_f), q)
    factor_differences = {el: float(np.max(abs(native_f[el] - common_f[el]) / abs(common_f[el])))
                          for el in native_f}
    result = dict(status='executed_installed_public_api', **installed_metadata(),
                  environment=dict(python=sys.version, platform=platform.platform(),
                                   processor=platform.processor(), cpu_threads=torch.get_num_threads(),
                                   torch=torch.__version__, numpy=np.__version__,
                                   cuda_available=torch.cuda.is_available()),
                  device=args.device, gpu_status='executed' if args.device == 'cuda' else 'not exercised',
                  q_inverse_angstrom=q.tolist(), scale_checks=scale_checks, rows=rows,
                  max_relative_native_vs_gemmi_factor_differences=factor_differences,
                  timing_scope='iq includes public structure initialization and device transfers; calculator setup separate',
                  limitations=['No grid public API in 1.0.14', 'Full pair-distance allocation',
                               'Peak native memory not measured', 'Native factors differ from Gemmi IT92'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2), encoding='utf-8')
    arrays_path = args.out.with_suffix('.npz')
    np.savez(arrays_path, **arrays)
    print(json.dumps(dict(status=result['status'], out=str(args.out), arrays=str(arrays_path),
                          n_atoms=[row['n_atoms'] for row in rows],
                          median_seconds=[row['median_public_iq_seconds'] for row in rows]), indent=2))


if __name__ == '__main__':
    main()
