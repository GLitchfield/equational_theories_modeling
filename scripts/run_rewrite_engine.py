"""Phase 5 (Task 4) of `rewrite_chain_study_plan.md` — the decisive
measurement. Run the bidirectional rewrite engine on all 1,636
c_rewrite_chain implications, report coverage-vs-depth + cost
distributions + the hard tail.

Reads:
  data/raw/etp/equations.txt
  data/processed/proof_corpus/rewrite_chain_dataset.parquet

Outputs:
  outputs/analyses/rewrite_chain_study/coverage.json   (committed)
  data/processed/proof_corpus/rewrite_engine_traces.parquet
      (gitignored; per-pair result for Tasks 3 and 5)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import parse_equation  # noqa: E402
from et_studies.rewrite_chain_study.code.engine import default_size_cap, search  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def load_equations(path: str) -> list:
    with open(_abs(path)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    return [parse_equation(line) for line in lines]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--equations', default='data/raw/etp/equations.txt')
    parser.add_argument('--dataset',
                        default='data/processed/proof_corpus/rewrite_chain_dataset.parquet')
    parser.add_argument('--output-coverage',
                        default='outputs/analyses/rewrite_chain_study/coverage.json')
    parser.add_argument('--output-traces',
                        default='data/processed/proof_corpus/rewrite_engine_traces.parquet')
    parser.add_argument('--d-max', type=int, default=6)
    parser.add_argument('--size-cap', type=int, default=None,
                        help='term-size cap; default = max(50, 5*max(|lhs2|, |rhs2|)) per implication')
    parser.add_argument('--states-budget', type=int, default=200_000,
                        help='per-pair states-expanded budget (abort early past this)')
    parser.add_argument('--progress-every', type=int, default=100)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    cov_path = _abs(args.output_coverage)
    traces_path = _abs(args.output_traces)
    for p in [cov_path, traces_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    # --- 1. load ---
    print(f'loading equations from {args.equations}')
    eqs = load_equations(args.equations)
    print(f'  {len(eqs):,} equations parsed')

    print(f'loading {args.dataset}')
    df = pq.read_table(_abs(args.dataset), columns=['e1', 'e2']).to_pandas()
    n = len(df)
    print(f'  {n:,} rewrite-chain pairs')

    # --- 2. run engine on each pair ---
    print(f'running rewrite engine (d_max={args.d_max}, size_cap='
          f'{args.size_cap if args.size_cap else "per-pair default"}) ...')

    results = []
    t_start = time.perf_counter()
    for i, row in enumerate(df.itertuples(index=False)):
        e1_id, e2_id = int(row.e1), int(row.e2)
        eq1 = eqs[e1_id - 1]
        eq2 = eqs[e2_id - 1]
        cap = args.size_cap if args.size_cap is not None else default_size_cap(eq2.lhs, eq2.rhs)
        t0 = time.perf_counter()
        r = search(eq2.lhs, eq2.rhs, eq1.lhs, eq1.rhs,
                   d_max=args.d_max, size_cap=cap,
                   states_budget=args.states_budget)
        wall_ms = (time.perf_counter() - t0) * 1000.0
        results.append({
            'e1': e1_id, 'e2': e2_id,
            'closed': r.closed,
            'total_depth': r.total_depth if r.closed else -1,
            'forward_depth': r.forward_depth if r.closed else -1,
            'backward_depth': r.backward_depth if r.closed else -1,
            'states_expanded': r.states_expanded,
            'fwd_states_seen': r.forward_states_seen,
            'bwd_states_seen': r.backward_states_seen,
            'wall_ms': wall_ms,
            'size_cap': cap,
        })
        if (i + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate
            n_closed = sum(1 for x in results if x['closed'])
            print(f'  {i + 1:>5,}/{n}  closed {n_closed:>5,}  '
                  f't={elapsed:>6.1f}s  rate={rate:>5.0f}/s  eta={eta:>5.1f}s',
                  flush=True)

    total_wall = time.perf_counter() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    # --- 3. coverage curve ---
    res_df = pq.Table.from_pylist(results) if False else None   # placeholder

    closed_depth = np.array([r['total_depth'] if r['closed'] else 9999 for r in results])
    coverage_by_depth = {}
    for D in range(0, 2 * args.d_max + 1):
        coverage_by_depth[D] = int((closed_depth <= D).sum())
    n_closed = int(sum(1 for r in results if r['closed']))
    n_not_closed = n - n_closed

    # Cost distributions (over CLOSED implications)
    closed_states = np.array([r['states_expanded'] for r in results if r['closed']])
    closed_wall = np.array([r['wall_ms'] for r in results if r['closed']])

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
            'p99': float(np.percentile(arr, 99)),
        }

    # Per-depth breakdown (proofs closed AT exact total_depth D)
    per_depth = {}
    for D in range(0, 2 * args.d_max + 1):
        mask = closed_depth == D
        if mask.sum() == 0:
            continue
        states_at_D = np.array([r['states_expanded'] for r, c in zip(results, closed_depth) if c == D])
        wall_at_D = np.array([r['wall_ms'] for r, c in zip(results, closed_depth) if c == D])
        per_depth[D] = {
            'n_closed_at_this_depth': int(mask.sum()),
            'states_expanded': percentiles(states_at_D),
            'wall_ms': percentiles(wall_at_D),
        }

    summary = {
        'config': {
            'd_max': args.d_max,
            'size_cap': args.size_cap,
            'size_cap_default_formula': 'max(50, 5*max(|lhs2|, |rhs2|))',
        },
        'n_pairs': n,
        'n_closed': n_closed,
        'n_not_closed': n_not_closed,
        'coverage_overall_fraction': n_closed / n,
        'coverage_by_total_depth': {
            str(D): {'n_closed': v, 'fraction': v / n}
            for D, v in coverage_by_depth.items()
        },
        'states_expanded_distribution_closed': percentiles(closed_states),
        'wall_ms_distribution_closed': percentiles(closed_wall),
        'per_depth_breakdown': per_depth,
        'wall_clock_seconds': total_wall,
        'pairs_per_second': n / total_wall,
    }

    os.makedirs(os.path.dirname(cov_path), exist_ok=True)
    with open(cov_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'wrote {cov_path}')

    # --- 4. per-pair traces ---
    table = pa.table({
        'e1': pa.array([r['e1'] for r in results], type=pa.int32()),
        'e2': pa.array([r['e2'] for r in results], type=pa.int32()),
        'closed': pa.array([r['closed'] for r in results], type=pa.bool_()),
        'total_depth': pa.array([r['total_depth'] for r in results], type=pa.int16()),
        'forward_depth': pa.array([r['forward_depth'] for r in results], type=pa.int16()),
        'backward_depth': pa.array([r['backward_depth'] for r in results], type=pa.int16()),
        'states_expanded': pa.array([r['states_expanded'] for r in results], type=pa.int32()),
        'fwd_states_seen': pa.array([r['fwd_states_seen'] for r in results], type=pa.int32()),
        'bwd_states_seen': pa.array([r['bwd_states_seen'] for r in results], type=pa.int32()),
        'wall_ms': pa.array([r['wall_ms'] for r in results], type=pa.float64()),
        'size_cap': pa.array([r['size_cap'] for r in results], type=pa.int32()),
    })
    os.makedirs(os.path.dirname(traces_path), exist_ok=True)
    pq.write_table(table, traces_path, compression='zstd')
    print(f'wrote {traces_path}')

    # --- 5. headline ---
    print()
    print('=== Coverage by total proof depth ===')
    for D in sorted(coverage_by_depth):
        n_c = coverage_by_depth[D]
        print(f'  depth ≤ {D:>2}: {n_c:>5,} / {n:,} = {100*n_c/n:5.2f}%')
    print(f'  TOTAL closed at d_max={args.d_max} (≤ {2*args.d_max}): '
          f'{n_closed:,} / {n:,} = {100*n_closed/n:.2f}%')
    print(f'  Hard tail (not closed within d_max): {n_not_closed:,} ({100*n_not_closed/n:.2f}%)')
    print()
    if n_closed > 0:
        cs = summary['states_expanded_distribution_closed']
        cw = summary['wall_ms_distribution_closed']
        print('=== Cost (closed implications) ===')
        print(f'  states expanded: median {cs["median"]:.0f}, mean {cs["mean"]:.0f}, '
              f'p95 {cs["p95"]:.0f}, max {cs["max"]:.0f}')
        print(f'  wall (ms)      : median {cw["median"]:.2f}, mean {cw["mean"]:.2f}, '
              f'p95 {cw["p95"]:.2f}, max {cw["max"]:.2f}')
    print()
    print(f'  total wall: {total_wall:.1f}s ({n/total_wall:.0f} pairs/s)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
