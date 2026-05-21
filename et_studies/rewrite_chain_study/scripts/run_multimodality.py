"""Phase 7 (Task 5) of `rewrite_chain_study_plan.md` — multimodality.

For a stratified sample of closed implications, enumerate ALL distinct
forward chains lhs2 -> ... -> rhs2 of length exactly the shortest-path
total_depth found by Task 4's bidirectional search. Reports the
distribution of "number of valid proofs per implication" — the
multimodality picture that informs the imitation-vs-verifier choice.

Per Methodology Resolution 9, sample size starts at 100; we use 500
(the upgrade tier) since Task 4 showed search is cheap. To bound
enumeration cost, restrict to pairs with total_depth ≤ 6
(96% of closed pairs); deeper pairs are marked as skipped.

Reads:
  et_studies/rewrite_chain_study/data/rewrite_engine_traces.parquet
  data/raw/etp/equations.txt

Writes:
  outputs/analyses/rewrite_chain_study/task5_multimodality.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import parse_equation  # noqa: E402
from et_studies.rewrite_chain_study.code.engine import default_size_cap, enumerate_chains  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def load_equations(path: str) -> list:
    with open(_abs(path)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    return [parse_equation(line) for line in lines]


def percentiles(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {'n': 0}
    return {
        'n': int(len(arr)),
        'min': float(arr.min()),
        'max': float(arr.max()),
        'mean': float(arr.mean()),
        'median': float(np.median(arr)),
        'p25': float(np.percentile(arr, 25)),
        'p75': float(np.percentile(arr, 75)),
        'p90': float(np.percentile(arr, 90)),
        'p95': float(np.percentile(arr, 95)),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--traces',
                    default='et_studies/rewrite_chain_study/data/rewrite_engine_traces.parquet')
    ap.add_argument('--output',
                    default='outputs/analyses/rewrite_chain_study/task5_multimodality.json')
    ap.add_argument('--sample-size', type=int, default=500)
    ap.add_argument('--max-target-depth', type=int, default=6,
                    help='skip closed pairs whose shortest proof exceeds this')
    ap.add_argument('--states-budget', type=int, default=200_000)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    print(f'loading equations from {args.equations}')
    eqs = load_equations(args.equations)
    print(f'  {len(eqs):,} equations parsed')

    print(f'loading traces from {args.traces}')
    tr = pq.read_table(_abs(args.traces)).to_pandas()
    closed = tr[tr['closed']].copy()
    enumerable = closed[closed['total_depth'] <= args.max_target_depth]
    print(f'  {len(tr):,} pairs total, {len(closed):,} closed, '
          f'{len(enumerable):,} enumerable (total_depth ≤ {args.max_target_depth})')

    # --- stratified sample ---
    rng = np.random.default_rng(args.seed)
    if len(enumerable) <= args.sample_size:
        sample = enumerable
    else:
        sample_parts = []
        for d, grp in enumerable.groupby('total_depth'):
            frac = len(grp) / len(enumerable)
            n_take = max(1, int(round(args.sample_size * frac)))
            n_take = min(n_take, len(grp))
            idx = rng.choice(len(grp), size=n_take, replace=False)
            sample_parts.append(grp.iloc[idx])
        import pandas as pd
        sample = pd.concat(sample_parts)
        if len(sample) > args.sample_size:
            sample = sample.sample(n=args.sample_size, random_state=args.seed)
    print(f'  sampled {len(sample):,} pairs')

    # --- enumerate ---
    results = []
    t_start = time.perf_counter()
    for i, row in enumerate(sample.itertuples(index=False)):
        e1_id, e2_id = int(row.e1), int(row.e2)
        D = int(row.total_depth)
        eq1, eq2 = eqs[e1_id - 1], eqs[e2_id - 1]
        cap = default_size_cap(eq2.lhs, eq2.rhs)
        t0 = time.perf_counter()
        r = enumerate_chains(eq2.lhs, eq2.rhs, eq1.lhs, eq1.rhs,
                             target_depth=D, size_cap=cap,
                             states_budget=args.states_budget)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        results.append({
            'e1': e1_id, 'e2': e2_id,
            'total_depth': D,
            'n_chains': r.n_distinct_chains,
            'states_expanded': r.states_expanded,
            'hit_budget': r.hit_budget,
            'wall_ms': wall_ms,
        })
        if (i + 1) % 50 == 0:
            elapsed = time.perf_counter() - t_start
            print(f'  {i + 1:>4}/{len(sample)}  t={elapsed:>6.1f}s', flush=True)

    total_wall = time.perf_counter() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    # --- aggregate ---
    n_chains = np.array([r['n_chains'] for r in results])
    completed = np.array([not r['hit_budget'] for r in results])
    completed_chains = n_chains[completed]

    # Distribution bins: 1, 2-3, 4-9, 10-99, 100+ (lower bound for hit-budget)
    bins = {'exactly_1': 0, '2_to_3': 0, '4_to_9': 0,
            '10_to_99': 0, '100_or_more': 0, 'hit_budget': 0}
    for r in results:
        if r['hit_budget']:
            bins['hit_budget'] += 1
            continue
        nc = r['n_chains']
        if nc == 1: bins['exactly_1'] += 1
        elif 2 <= nc <= 3: bins['2_to_3'] += 1
        elif 4 <= nc <= 9: bins['4_to_9'] += 1
        elif 10 <= nc <= 99: bins['10_to_99'] += 1
        else: bins['100_or_more'] += 1

    # Per-depth: for each total_depth D, what's the n_chains distribution?
    per_depth = {}
    for D in sorted(set(r['total_depth'] for r in results)):
        sub = [r for r in results if r['total_depth'] == D]
        sub_completed = [r for r in sub if not r['hit_budget']]
        per_depth[int(D)] = {
            'n_sampled': len(sub),
            'n_completed': len(sub_completed),
            'n_hit_budget': len(sub) - len(sub_completed),
            'n_chains_distribution': percentiles(
                np.array([r['n_chains'] for r in sub_completed])
            ),
        }

    n_skipped_depth = int((closed['total_depth'] > args.max_target_depth).sum())
    summary = {
        'config': {
            'sample_size_requested': args.sample_size,
            'sample_size_actual': int(len(sample)),
            'max_target_depth': args.max_target_depth,
            'states_budget': args.states_budget,
            'seed': args.seed,
            'n_enumerable_closed': int(len(enumerable)),
            'n_skipped_deeper_than_max': n_skipped_depth,
        },
        'multimodality_buckets': bins,
        'n_chains_distribution_completed': percentiles(completed_chains),
        'per_depth_breakdown': per_depth,
        'wall_clock_seconds': total_wall,
    }

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(_clean(summary), f, indent=2, sort_keys=False)
    print(f'wrote {out_path}')

    # --- headline ---
    print()
    print('=== Multimodality (n_chains per implication) ===')
    n_total = len(results)
    for k in ['exactly_1', '2_to_3', '4_to_9', '10_to_99', '100_or_more', 'hit_budget']:
        v = bins[k]
        print(f'  {k:>14}: {v:>4} ({100*v/n_total:5.2f}%)')
    if len(completed_chains) > 0:
        print()
        print('=== n_chains distribution (completed only) ===')
        print(f'  median {np.median(completed_chains):.0f}  '
              f'p75 {np.percentile(completed_chains, 75):.0f}  '
              f'p95 {np.percentile(completed_chains, 95):.0f}  '
              f'max {completed_chains.max():.0f}')
    print()
    print('=== per total_depth ===')
    for D, d in sorted(per_depth.items()):
        if d['n_completed'] > 0:
            m = d['n_chains_distribution']['median']
            mx = d['n_chains_distribution']['max']
            print(f'  depth={D:>2}  n={d["n_sampled"]:>3}  '
                  f'budget_hit={d["n_hit_budget"]:>2}  '
                  f'median_chains={m:>5.0f}  max_chains={mx:>5.0f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
