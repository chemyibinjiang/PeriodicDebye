"""Summarize recorded performance and independent-seed sampling refinement."""
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent


def main():
    summaries = {name:json.loads((ROOT/name/'summary.json').read_text(encoding='utf-8'))
                 for name in ('converged_n40', 'bodyframe_n40', 'final_n40')}
    before = summaries['converged_n40']['distributions']['active_wall']
    after = summaries['bodyframe_n40']['distributions']['active_wall']
    final = summaries['final_n40']['distributions']['active_wall']
    low = np.load(ROOT/'bodyframe_n40'/'active_wall'/'curves.npz')
    high = np.load(ROOT/'final_n40'/'active_wall'/'curves.npz')
    np.testing.assert_array_equal(low['q'], high['q'])
    assert after['local_statistics']['seed'] != final['local_statistics']['seed']
    region = high['q']>=.1
    peak = float(np.max(abs(high['total'][region])))
    significant = region & (abs(high['total'])>=.01*peak)
    difference = abs(low['total']-high['total'])
    combined = np.hypot(low['total_mc_sem'], high['total_mc_sem'])
    floor = 1e-12*peak
    z = difference/np.maximum(combined, floor)
    maximum_three_sem = float(np.max(3*high['total_mc_sem'][significant]/abs(high['total'][significant])))
    report = dict(
        local_statistics_speedup_same_accepted_budget=before['statistics_seconds']/after['statistics_seconds'],
        first_pattern_speedup_same_accepted_budget=before['first_pattern_seconds_excluding_warmup']/after['first_pattern_seconds_excluding_warmup'],
        final_first_pattern_seconds=final['first_pattern_seconds_excluding_warmup'],
        final_max_three_sem_fraction_at_significant_points=maximum_three_sem,
        final_max_radial_bound_fraction_at_significant_points=final['max_radial_bound_fraction_at_significant_points'],
        final_peak_working_set_MiB=summaries['final_n40']['memory_after_calculations']['peak_wset']/2**20,
        refinement=dict(independent_seeds=True, q_min=.1, significant_peak_fraction=.01,
            significant_q_points=int(significant.sum()),
            maximum_difference_fraction_at_significant_points=float(np.max(difference[significant]/abs(high['total'][significant]))),
            maximum_difference_over_nonforward_peak=float(np.max(difference[region])/peak),
            maximum_combined_sem_with_roundoff_floor=float(np.max(z[region])),
            sem_comparison_roundoff_floor_electron2=floor),
        caveat='Three SEM is an estimated integration uncertainty, not a strict or simultaneous error certificate. '
               'Independent-seed refinement is not an exact reference. Radial bounds exclude floating-point error. '
               'All accuracy metrics are at the sampled q points and exclude q-grid undersampling.')
    assert maximum_three_sem < .001, report
    assert np.max(z[region]) < 5, report
    (ROOT/'sampling_and_speed_comparison.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
