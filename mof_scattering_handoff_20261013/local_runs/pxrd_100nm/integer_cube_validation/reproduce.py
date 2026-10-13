"""Full 103.328 nm geometric reference; run with the handoff Python environment."""
import importlib.util
import json
from pathlib import Path
import sys
import time

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
PROJECT = HERE.parents[2]/'mof_debye_lab'
sys.path.insert(0, str(PROJECT/'src'))
from mofdebye.core import AtomCloud, form_factors
from mofdebye.large_powder import LargePowderCache

spec = importlib.util.spec_from_file_location('independent_cube', PROJECT/'scripts'/'validate_large_powder.py')
reference_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reference_module)

threadpool_limits(limits=1)
q = np.linspace(0, 6, 6001)
basis = AtomCloud(('C',), [[.37, 1.23, .98]])
factor = form_factors(basis.elements, q)['C']
start = time.perf_counter()
reference = reference_module.integer_cube_iq(q, 40, 25.832, factor)
reference_seconds = time.perf_counter()-start
start = time.perf_counter()
cache = LargePowderCache(basis, np.eye(3)*25.832, np.ones((40, 40, 40), dtype=int), q,
                        bin_width=.04, order=8, threads=1)
prepare_seconds = time.perf_counter()-start
start = time.perf_counter()
actual, bound = cache.iq()
evaluation_seconds = time.perf_counter()-start
metrics = reference_module.compare_powder_curves(q, actual, reference)
report = dict(cells_per_edge=40, atoms=40**3, edge_nm=103.328, q_points=6001,
              spacing_A=25.832, factor='C neutral Gemmi IT92; shared physical input',
              reference='Independent integer h^2+k^2+l^2 shells and closed pair-count multiplicities',
              reference_seconds=reference_seconds, prepare_seconds=prepare_seconds,
              evaluation_seconds=evaluation_seconds, cache_metadata=cache.metadata,
              metrics=metrics,
              scope='Geometric extent check, one C atom per cell; not MOF chemistry. Timings are single diagnostic calls, not comparative benchmark medians.')
np.savez_compressed(HERE/'curves.npz', q=q, actual=actual, reference=reference,
                    approximation_bound_excluding_roundoff=bound)
(HERE/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps({key: value for key, value in metrics.items() if key != 'peaks'}, indent=2))
assert metrics['intensity_checks_passed']
assert metrics['max_peak_q_shift_A_inv'] <= .00100000001
assert metrics['forward_relative_error'] < 2e-12
