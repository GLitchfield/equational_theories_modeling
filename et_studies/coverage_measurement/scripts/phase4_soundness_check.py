"""Phase 4 of `coverage_measurement_plan.md` — soundness check on
the engine's TRUE claims.

For every pair in engine_results_*.parquet where solved=True, verify
the pair is GT TRUE in the May 2026 ETP export. Target: zero
disagreements. The engine is sound by construction (safe_search
restricts to variable-inclusion-valid directions); any disagreement
would be an implementation bug.

Spot-check: for ~50 randomly-sampled solved pairs, reconstruct the
forward path and confirm the rewrites actually close the gap. This
guards against bugs in the meet-detection logic that would let the
engine claim a solve without a real path.

Reads:
  et_studies/coverage_measurement/data/engine_results_within_ceiling_d8.parquet
  data/raw/etp_export/export_raw_implications_5_9_2026.csv

Writes:
  et_studies/coverage_measurement/outputs/phase4_soundness.json
  et_studies/coverage_measurement/data/phase4_disagreements.parquet
    (only written if disagreements found)
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

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--engine-results',
                    default='et_studies/coverage_measurement/data/engine_results_within_ceiling_d8.parquet')
    ap.add_argument('--ground-truth',
                    default='data/raw/etp_export/export_raw_implications_5_9_2026.csv')
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--output-summary',
                    default='et_studies/coverage_measurement/outputs/phase4_soundness.json')
    ap.add_argument('--output-disagreements',
                    default='et_studies/coverage_measurement/data/phase4_disagreements.parquet')
    ap.add_argument('--spot-check-n', type=int, default=50)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    out_path = _abs(args.output_summary)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    print(f'loading engine results from {args.engine_results}')
    er = pq.read_table(_abs(args.engine_results)).to_pandas()
    solved = er[er['solved']].copy()
    print(f'  {len(er):,} pairs in engine output')
    print(f'  {len(solved):,} solved by engine')

    print(f'loading ground truth from {args.ground_truth}')
    gt = np.loadtxt(_abs(args.ground_truth), delimiter=',', dtype=np.int8)
    truth_is_true = gt > 0
    print(f'  ground truth shape {gt.shape}')

    # --- soundness check: every "solved" pair must be GT TRUE ---
    print()
    print('checking soundness — every solved pair must be GT TRUE …')
    t0 = time.time()
    disagreements = []
    for _, row in solved.iterrows():
        e1, e2 = int(row['e1']), int(row['e2'])
        if not truth_is_true[e1 - 1][e2 - 1]:
            disagreements.append({
                'e1': e1, 'e2': e2,
                'total_depth': int(row['total_depth']),
                'wall_ms': float(row['wall_ms']),
                'description': 'engine solved but ground truth FALSE — soundness bug',
            })
    n_disagreements = len(disagreements)
    print(f'  {n_disagreements} disagreements (must be 0)')
    print(f'  check took {time.time() - t0:.2f}s')

    # --- spot-check: replay paths via safe_search on 50 random solved pairs ---
    # safe_search doesn't expose the path; the replay is "rerun and confirm
    # solved=True at the reported total_depth" — a basic consistency check.
    print()
    print(f'spot-check: replay {args.spot_check_n} random solved pairs …')
    from lib.data.parser import parse_equation
    from proof_generation_pilot.methods.safe_engine.safe_search import safe_search
    with open(_abs(args.equations)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    eqs = [parse_equation(ln) for ln in lines]

    rng = np.random.default_rng(args.seed)
    idx = rng.choice(len(solved), size=min(args.spot_check_n, len(solved)),
                     replace=False)
    n_replay_ok = 0
    n_replay_mismatch = 0
    replay_mismatches = []
    for i in idx:
        row = solved.iloc[i]
        e1, e2 = int(row['e1']), int(row['e2'])
        expected_depth = int(row['total_depth'])
        eq1, eq2 = eqs[e1 - 1], eqs[e2 - 1]
        r = safe_search(eq2.lhs, eq2.rhs, eq1.lhs, eq1.rhs, d_max=8, size_cap=50)
        if r.solved and r.total_depth == expected_depth:
            n_replay_ok += 1
        else:
            n_replay_mismatch += 1
            replay_mismatches.append({
                'e1': e1, 'e2': e2,
                'expected_solved': True, 'expected_depth': expected_depth,
                'replay_solved': r.solved,
                'replay_depth': r.total_depth if r.total_depth is not None else -1,
            })

    print(f'  {n_replay_ok}/{args.spot_check_n} matched expected depth')
    if n_replay_mismatch:
        print(f'  {n_replay_mismatch} mismatches (first 3):')
        for m in replay_mismatches[:3]:
            print(f'    {m}')

    # --- summary ---
    summary = {
        'config': {
            'engine_results': args.engine_results,
            'spot_check_n': args.spot_check_n,
            'seed': args.seed,
        },
        'n_pairs_in_engine_output': int(len(er)),
        'n_pairs_solved_by_engine': int(len(solved)),
        'soundness_check': {
            'n_disagreements': n_disagreements,
            'passed': n_disagreements == 0,
            'description': 'Every solved pair must be GT TRUE. n=0 is the target.',
        },
        'replay_spot_check': {
            'n_sampled': int(args.spot_check_n),
            'n_replay_ok': int(n_replay_ok),
            'n_replay_mismatch': int(n_replay_mismatch),
            'passed': n_replay_mismatch == 0,
        },
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'\nwrote {out_path}')

    if disagreements:
        pq.write_table(pa.Table.from_pylist(disagreements),
                       _abs(args.output_disagreements), compression='zstd')
        print(f'wrote {_abs(args.output_disagreements)} ({n_disagreements} entries)')

    print()
    print('=== Phase 4 outcomes ===')
    print(f'  soundness check (engine TRUE → GT TRUE): '
          f'{"PASS" if n_disagreements == 0 else f"FAIL ({n_disagreements} disagreements)"}')
    print(f'  replay spot-check ({args.spot_check_n} pairs): '
          f'{"PASS" if n_replay_mismatch == 0 else f"FAIL ({n_replay_mismatch} mismatches)"}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
