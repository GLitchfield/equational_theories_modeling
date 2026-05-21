"""Phase 4b — two-part de-risk before deciding on the Phase 4 build:

  (i)  CROSS-REFERENCE: are the 334 SYMMETRIC+WRONG_DIR equations
       STRONG (high TRUE-row density) or WEAK (low)?
       This tells us whether ordered completion would add real TRUE
       coverage (strong) or just extend FALSE further (weak).

  (ii) SPOT-CHECK: print 20 sample failing CPs from the SYMMETRIC and
       WRONG_DIR buckets, manually verify they're genuine ordered-
       completion territory and not LPO edge cases.

Decision rule:
  STRONG → build Phase 4. Qualitative finding (completion can prove,
            not just refute) justifies the half-day.
  WEAK   → skip Phase 4. The +4–6pp FALSE is a measured number that
            doesn't change the story; the 12% headline is the result.

Reads:
  et_studies/completion_exploration/data/phase4a_failed_cps.parquet
  et_studies/completion_exploration/data/equation_completion.parquet
  data/raw/etp_export/export_raw_implications_5_9_2026.csv

Writes:
  outputs/analyses/completion_study/phase4b_strong_vs_weak.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--phase4a',
                    default='et_studies/completion_exploration/data/phase4a_failed_cps.parquet')
    ap.add_argument('--completions',
                    default='et_studies/completion_exploration/data/equation_completion.parquet')
    ap.add_argument('--ground-truth',
                    default='data/raw/etp_export/export_raw_implications_5_9_2026.csv')
    ap.add_argument('--output',
                    default='outputs/analyses/completion_study/phase4b_strong_vs_weak.json')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    if os.path.exists(_abs(args.output)) and not args.force:
        print(f'ERROR: {_abs(args.output)} exists; pass --force.', file=sys.stderr)
        return 2

    print(f'loading ground truth {args.ground_truth}')
    gt = np.loadtxt(_abs(args.ground_truth), delimiter=',', dtype=np.int8)
    N = gt.shape[0]
    truth_is_true = gt > 0
    # Per-equation TRUE row density: how many E2 have eq → E2 TRUE.
    # Includes the diagonal (eq → eq is TRUE by reflexivity); subtract 1.
    row_true = truth_is_true.sum(axis=1) - 1   # one entry per eq_id-1
    print(f'  median TRUE row density: {np.median(row_true):.1f}, mean: {row_true.mean():.1f}')

    p4a = pq.read_table(_abs(args.phase4a)).to_pandas()
    comp = pq.read_table(_abs(args.completions)).to_pandas()

    # Define groups by eq_id.
    sets = {
        'all_4694': set(range(1, N + 1)),
        'completed': set(int(r) for r in comp[comp['status'] == 'COMPLETED']['eq_id']),
        'fully_completed': set(int(r) for r in comp[
            (comp['status'] == 'COMPLETED') & (comp['is_fully_completed'])
        ]['eq_id']),
        'failed_orient_derived': set(int(r) for r in comp[
            comp['status'] == 'FAILED_ORIENT_DERIVED']['eq_id']),
        'phase0_excluded': set(int(r) for r in comp[
            comp['status'] == 'PHASE0_EXCLUDED']['eq_id']),
        'budget_cps': set(int(r) for r in comp[comp['status'] == 'BUDGET_CPS']['eq_id']),
        'symmetric_fail': set(int(r) for r in p4a[
            p4a['classification'] == 'SYMMETRIC']['eq_id']),
        'wrong_dir_fail': set(int(r) for r in p4a[
            p4a['classification'] == 'WRONG_DIR']['eq_id']),
        'incomparable_fail': set(int(r) for r in p4a[
            p4a['classification'] == 'INCOMPARABLE']['eq_id']),
        'phase4_recoverable': set(int(r) for r in p4a[
            p4a['classification'].isin(['SYMMETRIC', 'WRONG_DIR'])]['eq_id']),
    }

    print()
    print('=== TRUE-row-density distribution by group ===')
    print(f'  {"group":>22}  {"n":>5}  {"min":>5}  {"p25":>5}  '
          f'{"med":>5}  {"mean":>6}  {"p75":>5}  {"p95":>5}  {"max":>5}')

    group_stats = {}
    for name, ids in sets.items():
        arr = np.array([row_true[i - 1] for i in ids])
        if len(arr) == 0:
            continue
        d = {
            'n': len(arr),
            'min': int(arr.min()), 'max': int(arr.max()),
            'mean': float(arr.mean()), 'median': float(np.median(arr)),
            'p25': float(np.percentile(arr, 25)),
            'p75': float(np.percentile(arr, 75)),
            'p95': float(np.percentile(arr, 95)),
        }
        group_stats[name] = d
        print(f'  {name:>22}  {d["n"]:>5,}  {d["min"]:>5}  '
              f'{d["p25"]:>5.0f}  {d["median"]:>5.0f}  {d["mean"]:>6.1f}  '
              f'{d["p75"]:>5.0f}  {d["p95"]:>5.0f}  {d["max"]:>5}')

    # The headline question: is Phase 4 recoverable set STRONG vs WEAK?
    # Compare to currently-completing baseline and to overall corpus.
    print()
    print('=== Decision-driving comparison ===')
    base_completed = group_stats['completed']['median']
    base_all = group_stats['all_4694']['median']
    p4_med = group_stats['phase4_recoverable']['median']
    print(f'  Median TRUE row density:')
    print(f'    overall corpus            : {base_all:.0f}')
    print(f'    currently completing      : {base_completed:.0f}')
    print(f'    Phase-4-recoverable (334) : {p4_med:.0f}')

    delta_vs_completing = group_stats['phase4_recoverable']['mean'] - group_stats['completed']['mean']
    delta_vs_all = group_stats['phase4_recoverable']['mean'] - group_stats['all_4694']['mean']
    print(f'  Mean TRUE row density delta:')
    print(f'    vs currently completing   : {delta_vs_completing:+.1f}')
    print(f'    vs overall corpus         : {delta_vs_all:+.1f}')

    # Decision banner
    print()
    print('=== Decision ===')
    if delta_vs_completing > 50:
        verdict = 'STRONG — build Phase 4 (likely adds real TRUE coverage)'
    elif delta_vs_completing > 0:
        verdict = 'MIXED — slightly stronger; marginal Phase 4 build justification'
    else:
        verdict = 'WEAK — skip Phase 4 build, proceed to writeup'
    print(f'  {verdict}')

    # ---- spot-check: 10 SYMMETRIC + 10 WRONG_DIR sample failures ----
    print()
    print('=== Spot-check: 10 SYMMETRIC failing CPs ===')
    sym = p4a[p4a['classification'] == 'SYMMETRIC'].head(10)
    for _, r in sym.iterrows():
        print(f"  eq#{int(r['eq_id']):>4}  failed: '{r['failed_s_repr']}' = '{r['failed_t_repr']}'")
    print()
    print('=== Spot-check: 10 WRONG_DIR failing CPs ===')
    wd = p4a[p4a['classification'] == 'WRONG_DIR'].head(10)
    for _, r in wd.iterrows():
        print(f"  eq#{int(r['eq_id']):>4}  failed: '{r['failed_s_repr']}' = '{r['failed_t_repr']}'  "
              f"(lhs_vars={r['lhs_n_vars']}, rhs_vars={r['rhs_n_vars']})")

    # Write summary
    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o

    summary = {
        'group_true_row_density_stats': group_stats,
        'comparison': {
            'phase4_recoverable_mean_minus_completed': float(delta_vs_completing),
            'phase4_recoverable_mean_minus_all_corpus': float(delta_vs_all),
        },
        'decision': verdict,
        'spot_check_symmetric': [
            {'eq_id': int(r['eq_id']),
             'failed_s_repr': r['failed_s_repr'],
             'failed_t_repr': r['failed_t_repr']}
            for _, r in sym.iterrows()
        ],
        'spot_check_wrong_dir': [
            {'eq_id': int(r['eq_id']),
             'failed_s_repr': r['failed_s_repr'],
             'failed_t_repr': r['failed_t_repr'],
             'lhs_n_vars': int(r['lhs_n_vars']),
             'rhs_n_vars': int(r['rhs_n_vars'])}
            for _, r in wd.iterrows()
        ],
    }
    os.makedirs(os.path.dirname(_abs(args.output)), exist_ok=True)
    with open(_abs(args.output), 'w') as f:
        json.dump(_clean(summary), f, indent=2)
    print()
    print(f'wrote {_abs(args.output)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
