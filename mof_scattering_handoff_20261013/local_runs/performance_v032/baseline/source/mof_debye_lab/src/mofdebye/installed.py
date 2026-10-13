"""Verified DebyeCalculator 1.0.14 public-API adapter for finite powder.

Native I(q) is half the conventional full Debye intensity. This adapter doubles
it and retains native Waasmaier--Kirfel 1995 factors, which differ from the
Gemmi IT92 factors used by other backends. This release has batched direct
summation and a full pair-distance matrix; it has no distance-grid kernel.
"""
from __future__ import annotations

from functools import lru_cache
import hashlib
import importlib.metadata
import importlib.resources
import inspect
from pathlib import Path
import threading
import time
import numpy as np
from .core import validate_q

VERIFIED_VERSION = '1.0.14'
_constructor_lock = threading.RLock()


@lru_cache(maxsize=1)
def _package():
    try:
        version = importlib.metadata.version('debyecalculator')
    except importlib.metadata.PackageNotFoundError as exc:
        raise ImportError("Install the verified optional adapter with pip install '.[external]'.") from exc
    if version != VERIFIED_VERSION:
        raise RuntimeError(f'DebyeCalculator {version} is unverified; this adapter requires {VERIFIED_VERSION}.')
    from debyecalculator import DebyeCalculator
    return DebyeCalculator


@lru_cache(maxsize=1)
def _coefficients():
    _package()
    import yaml
    source = importlib.resources.files('debyecalculator.utility').joinpath('elements_info.yaml')
    return yaml.safe_load(source.read_text(encoding='utf-8'))


def native_form_factors(elements, q):
    """Evaluate table coefficients with NumPy, never by calibrating I(q)."""
    q = validate_q(q)
    table = _coefficients()
    ss = (q / (4 * np.pi)) ** 2
    result = {}
    for element in sorted(set(elements)):
        if element not in table:
            raise ValueError(f'No installed-package neutral form factors for {element}.')
        p = np.asarray(table[element][:11], dtype=np.float64)
        if not np.isfinite(p).all():
            raise ValueError(f'Incomplete installed-package form factors for {element}.')
        result[element] = np.sum(p[:5, None] * np.exp(-p[6:11, None] * ss), axis=0) + p[5]
    return result


@lru_cache(maxsize=1)
def installed_metadata():
    cls = _package()
    source = Path(inspect.getfile(cls))
    table = importlib.resources.files('debyecalculator.utility').joinpath('elements_info.yaml')
    return {
        'package': 'debyecalculator', 'version': VERIFIED_VERSION,
        'public_api': 'DebyeCalculator.iq(include_self_scattering=True)',
        'source_path': str(source),
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'factor_table_sha256': hashlib.sha256(table.read_bytes()).hexdigest(),
        'form_factors': 'neutral Waasmaier-Kirfel 1995; installed elements_info.yaml',
        'normalisation': 'full Debye intensity in electron^2; 2 * native iq',
        'native_scale': 0.5, 'actual_method': 'batched_direct',
        'grid_available': False, 'bounded_pair_memory': False, 'precision': 'float64',
    }


def _new_calculator(q, device='cpu', batch_size=10000):
    """Construct publicly and confirm the generated grid fits the request.

    1.0.14 constructs tensors in torch's default dtype before converting them.
    Constructor calls are serialized while briefly selecting float64, then the
    original default is restored even on failure. Applications concurrently
    constructing unrelated torch tensors should serialize those operations too.
    """
    import torch
    q = validate_q(q)
    if device not in ('cpu', 'cuda'):
        raise ValueError('Installed adapter device must be cpu or cuda.')
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable; no silent CPU fallback.')
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError('batch_size must be a positive integer.')
    step = float(q[1] - q[0]) if len(q) > 1 else .01
    if step <= 0 or not np.allclose(q, q[0] + np.arange(len(q)) * step,
                                    rtol=2e-12, atol=2e-12):
        raise ValueError('Installed package requires a strictly increasing uniform scalar q grid.')
    cls = _package()
    with _constructor_lock:
        original_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(torch.float64)
            calc = cls(qmin=float(q[0]), qmax=float(q[-1] + step / 2), qstep=step,
                       biso=0., rthres=0., radiation_type='xray', device=device,
                       dtype=torch.float64, batch_size=batch_size)
        finally:
            torch.set_default_dtype(original_dtype)
    generated = calc.q.detach().cpu().numpy().ravel()
    if generated.shape != q.shape or not np.allclose(generated, q, rtol=2e-12, atol=2e-12):
        raise ValueError('Installed package cannot represent the requested q grid; no interpolation is performed.')
    if str(calc.device) != device:
        raise RuntimeError('Installed package selected an unexpected device.')
    return calc


def installed_iq(cloud, q, device='cpu', return_info=False, batch_size=10000):
    """Conventional finite Debye I(q), with native Waasmaier--Kirfel factors.

    q must be uniform and increasing (a singleton is allowed). No factor-table
    substitution or q interpolation occurs. See _new_calculator for threading.
    """
    import torch
    q = validate_q(q)
    start = time.perf_counter()
    calc = _new_calculator(q, device=device, batch_size=batch_size)
    native_form_factors(cloud.elements, q)
    if device == 'cuda':
        torch.cuda.synchronize()
    ready = time.perf_counter()
    if len(cloud):
        with torch.no_grad():
            q_native, raw = calc.iq((list(cloud.elements), np.array(cloud.xyz)),
                                   include_self_scattering=True)
        if device == 'cuda':
            torch.cuda.synchronize()
        result = 2 * np.asarray(raw, dtype=np.float64)
        q_native = np.asarray(q_native)
    else:
        result = np.zeros(len(q))
        q_native = q
    finished = time.perf_counter()
    info = dict(installed_metadata(), device=device, n_atoms=len(cloud),
                batch_size=batch_size, torch_threads=torch.get_num_threads(),
                setup_seconds=ready - start, public_iq_seconds=finished - ready,
                max_q_difference=float(np.max(np.abs(q_native - q))))
    if not len(cloud):
        info['actual_method'] = 'empty'
    return (result, info) if return_info else result
