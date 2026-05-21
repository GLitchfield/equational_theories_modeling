"""Phase 2 of `coverage_measurement_plan.md` — batch engine harness.

Wraps `src/coverage/safe_search.py::safe_search` for parallel execution
over a parquet of (eq1, eq2) pairs. Each worker process loads the
equation list once at initialization and then processes pairs via
multiprocessing.Pool.imap_unordered.

Soundness discipline: safe_search restricts E1 to its variable-inclusion-
valid directions, eliminating the rule-var leakage that the unsafe
engine (src/rewrite/engine.py) is subject to on E1s where
Vars(rhs) ⊋ Vars(lhs). The trade-off is documented: matcher-shape
substitution-instance proofs of variable-disjoint pairs are out of
safe_search's reach and must be picked up via the matcher's parquet
in Phase 5 (the union-of-methods combined coverage step).

Inputs:
  - --sample parquet: pairs to process (columns: e1, e2)
  - --d-max, --size-cap, --wall-seconds: per-pair budgets
  - --pool-size: workers (default os.cpu_count() // 2)

Outputs:
  - Per-pair parquet: e1, e2, solved, terminate_reason, total_depth,
    states_expanded, wall_ms.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import signal as _signal
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# ============================================================
# Worker
# ============================================================
# Module-level globals populated in worker_init() and used by worker_run().
# Each worker process gets its own copy.
_WORKER_EQS = None
_WORKER_PARAMS = None


class _PairTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _PairTimeout()


def worker_init(equations_path: str, d_max: int, size_cap: int | None,
                wall_seconds: float):
    """Pool initializer. Each worker parses equations.txt and stashes
    params in module globals."""
    sys.setrecursionlimit(20000)
    from lib.data.parser import parse_equation
    global _WORKER_EQS, _WORKER_PARAMS
    with open(_abs(equations_path)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    _WORKER_EQS = [parse_equation(ln) for ln in lines]
    _WORKER_PARAMS = (d_max, size_cap, wall_seconds)
    # Install the alarm handler once per worker (we'll arm/disarm per call).
    _signal.signal(_signal.SIGALRM, _alarm_handler)


def worker_run(pair_tuple: tuple[int, int]) -> dict:
    """Process one (e1, e2) pair. Returns a record dict."""
    from proof_generation_pilot.methods.safe_engine.safe_search import safe_search, TerminateReason
    e1, e2 = pair_tuple
    d_max, size_cap, wall_seconds = _WORKER_PARAMS
    eq1 = _WORKER_EQS[e1 - 1]
    eq2 = _WORKER_EQS[e2 - 1]

    t0 = time.perf_counter()
    _signal.setitimer(_signal.ITIMER_REAL, wall_seconds)
    try:
        r = safe_search(eq2.lhs, eq2.rhs, eq1.lhs, eq1.rhs,
                        d_max=d_max, size_cap=size_cap)
        terminate_reason = r.terminate_reason.value
        solved = r.solved
        total_depth = r.total_depth
        states_expanded = r.states_expanded
    except _PairTimeout:
        terminate_reason = TerminateReason.WALL_TIMEOUT.value
        solved = False
        total_depth = None
        states_expanded = -1   # unknown
    finally:
        _signal.setitimer(_signal.ITIMER_REAL, 0)
    wall_ms = (time.perf_counter() - t0) * 1000.0

    return {
        'e1': e1, 'e2': e2,
        'solved': solved,
        'terminate_reason': terminate_reason,
        'total_depth': total_depth if total_depth is not None else -1,
        'states_expanded': states_expanded,
        'wall_ms': wall_ms,
    }


# ============================================================
# Driver
# ============================================================

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--sample',
                    default='data/processed/coverage_measurement/sample.parquet')
    ap.add_argument('--output',
                    default='data/processed/coverage_measurement/engine_results.parquet')
    ap.add_argument('--d-max', type=int, default=6)
    ap.add_argument('--size-cap', type=int, default=None,
                    help='per-pair term-size cap; default = safe_search.default_size_cap')
    ap.add_argument('--wall-seconds', type=float, default=30.0)
    ap.add_argument('--pool-size', type=int, default=None,
                    help='workers; default = os.cpu_count() // 2')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only first N pairs (for dev)')
    ap.add_argument('--progress-every', type=int, default=1000)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    pool_size = args.pool_size or max(1, (os.cpu_count() or 2) // 2)
    print(f'loading sample from {args.sample}')
    sample = pq.read_table(_abs(args.sample), columns=['e1', 'e2']).to_pandas()
    pairs = list(zip(sample['e1'].astype(int).tolist(),
                     sample['e2'].astype(int).tolist()))
    if args.limit:
        pairs = pairs[:args.limit]
    print(f'  {len(pairs):,} pairs to process')
    print(f'  d_max={args.d_max}, wall_seconds={args.wall_seconds}s, pool_size={pool_size}')

    t_start = time.time()
    records: list[dict] = []
    ctx = mp.get_context('fork')   # avoid pickling start-up
    with ctx.Pool(
        processes=pool_size,
        initializer=worker_init,
        initargs=(args.equations, args.d_max, args.size_cap, args.wall_seconds),
    ) as pool:
        for i, rec in enumerate(pool.imap_unordered(worker_run, pairs, chunksize=64)):
            records.append(rec)
            if (i + 1) % args.progress_every == 0:
                elapsed = time.time() - t_start
                rate = (i + 1) / elapsed
                eta = (len(pairs) - i - 1) / rate if rate else 0
                solved = sum(1 for r in records if r['solved'])
                print(f'  {i+1:>6,}/{len(pairs):,}  t={elapsed:>6.1f}s  '
                      f'rate={rate:>5.0f}/s  eta={eta:>5.0f}s  '
                      f'solved={solved:>6,} ({100*solved/(i+1):.1f}%)',
                      flush=True)

    total_wall = time.time() - t_start

    table = pa.Table.from_pylist(records)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pq.write_table(table, out_path, compression='zstd')
    print(f'\ndone in {total_wall:.1f}s')
    print(f'wrote {out_path}')

    from collections import Counter
    ctr = Counter(r['terminate_reason'] for r in records)
    print()
    print('=== terminate_reason breakdown ===')
    for k, v in ctr.most_common():
        print(f'  {k:>25}: {v:>6,}  ({100*v/len(records):5.2f}%)')

    solved = sum(1 for r in records if r['solved'])
    print()
    print(f'  TOTAL: {solved:,}/{len(records):,} solved ({100*solved/len(records):.2f}%) in {total_wall:.0f}s')
    return 0


if __name__ == '__main__':
    sys.exit(main())
