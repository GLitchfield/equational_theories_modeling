"""Phase 4 main run: ordered (unfailing) KB completion on all 4,694
ETP equations.

Mirrors Phase 2 (scripts/phase2_run_completion.py) but uses
src/completion/ordered.py's `ordered_complete` instead of plain KB.
The classification scheme is simpler — no FAILED_ORIENT, just:

  COMPLETED                — finite confluent rule system (per-instance
                             ordered rewriting). is_fully_completed
                             flag distinguishes TRUE-sound from
                             TRUE+FALSE-sound (size-cap drops and
                             dormant CPs both downgrade).
  NO_RULES_FROM_INPUT      — input equation variable-disjoint.
                             Effectively Phase 0 hard core.
  BUDGET_*                 — cascade exhausted budgets.

Pre-filters Phase 0's variable-disjoint set the same way Phase 2 did.

Reads:
  data/raw/etp/equations.txt
  data/processed/completion_study/equation_orientability.parquet

Writes:
  outputs/analyses/completion_study/phase4_ordered_outcomes.json
  data/processed/completion_study/equation_ordered_completion.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from proof_generation_pilot.methods.completion.ordered import OrderedStatus, ordered_complete  # noqa: E402
from lib.data.parser import Term, parse_equation  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def refine_status(result, *, phase0_excluded: bool) -> str:
    if phase0_excluded:
        return 'PHASE0_EXCLUDED'
    s = result.status
    if s == OrderedStatus.COMPLETED:
        return 'COMPLETED'
    if s == OrderedStatus.NO_RULES_FROM_INPUT:
        return 'NO_RULES_FROM_INPUT'
    if s == OrderedStatus.BUDGET_CPS:
        return 'BUDGET_CPS'
    if s == OrderedStatus.BUDGET_RULES:
        return 'BUDGET_RULES'
    if s == OrderedStatus.BUDGET_WALL:
        return 'BUDGET_WALL'
    return f'UNKNOWN_{s.value}'


def serialize_rules(rules: list[tuple[Term, Term]]) -> list[tuple[str, str]]:
    return [(repr(l), repr(r)) for l, r in rules]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--phase0',
                    default='data/processed/completion_study/equation_orientability.parquet')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/completion_study/phase4_ordered_outcomes.json')
    ap.add_argument('--output-records',
                    default='data/processed/completion_study/equation_ordered_completion.parquet')
    ap.add_argument('--max-rules', type=int, default=50)
    ap.add_argument('--max-cps', type=int, default=500)
    ap.add_argument('--max-term-size', type=int, default=30)
    ap.add_argument('--wall-seconds', type=float, default=2.0)
    ap.add_argument('--progress-every', type=int, default=500)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    for p in [args.output_summary, args.output_records]:
        if os.path.exists(_abs(p)) and not args.force:
            print(f'ERROR: {_abs(p)} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'loading equations from {args.equations}')
    with open(_abs(args.equations)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    eqs = [parse_equation(line) for line in lines]
    print(f'  {len(eqs):,} parsed')

    print(f'loading Phase 0 classification from {args.phase0}')
    p0 = pq.read_table(_abs(args.phase0)).to_pandas()
    p0_class = {int(row.eq_id): row.classification for row in p0.itertuples(index=False)}
    n_excl = sum(1 for c in p0_class.values() if c == 'incomparable')
    print(f'  {n_excl:,} Phase 0 excluded (variable-disjoint)')

    print(f'running ordered completion: max_rules={args.max_rules}, '
          f'max_cps={args.max_cps}, term_size={args.max_term_size}, '
          f'wall={args.wall_seconds}s ...')

    records = []
    n = args.limit if args.limit else len(eqs)
    t_start = time.time()
    for i in range(n):
        eq_id = i + 1
        eq = eqs[i]
        cls = p0_class.get(eq_id, 'unknown')
        excluded = (cls == 'incomparable')

        if excluded:
            records.append({
                'eq_id': eq_id,
                'phase0_classification': cls,
                'status': 'PHASE0_EXCLUDED',
                'n_rules': 0,
                'n_critical_pairs': 0,
                'n_critical_pairs_dropped_by_size_cap': 0,
                'n_dormant': 0,
                'n_iterations': 0,
                'wall_ms': 0.0,
                'is_fully_completed': False,
                'rules_json': '[]',
            })
        else:
            result = ordered_complete(
                [(eq.lhs, eq.rhs)],
                max_rules=args.max_rules,
                max_critical_pairs=args.max_cps,
                max_term_size=args.max_term_size,
                wall_seconds=args.wall_seconds,
            )
            refined = refine_status(result, phase0_excluded=False)
            records.append({
                'eq_id': eq_id,
                'phase0_classification': cls,
                'status': refined,
                'n_rules': len(result.rules),
                'n_critical_pairs': result.n_critical_pairs_considered,
                'n_critical_pairs_dropped_by_size_cap': result.n_critical_pairs_dropped_by_size_cap,
                'n_dormant': result.n_equations_kept_as_dormant,
                'n_iterations': result.n_iterations,
                'wall_ms': result.wall_seconds * 1000.0,
                'is_fully_completed': result.is_fully_completed,
                'rules_json': json.dumps(serialize_rules(result.rules)),
            })

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            ctr = Counter(r['status'] for r in records)
            print(f'  {i+1:>5,}/{n:,}  t={elapsed:>6.1f}s  '
                  f'rate={rate:>5.0f}/s  eta={eta:>5.0f}s  '
                  f"COMP={ctr.get('COMPLETED', 0):>4} "
                  f"NO_RULES={ctr.get('NO_RULES_FROM_INPUT', 0):>4} "
                  f"BUDGET={sum(ctr.get(s, 0) for s in ['BUDGET_CPS','BUDGET_RULES','BUDGET_WALL']):>4}",
                  flush=True)

    total_wall = time.time() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    ctr = Counter(r['status'] for r in records)
    n_total = len(records)
    n_completed = ctr.get('COMPLETED', 0)
    n_fully = sum(1 for r in records if r.get('is_fully_completed'))
    n_no_rules = ctr.get('NO_RULES_FROM_INPUT', 0)
    n_p0 = ctr.get('PHASE0_EXCLUDED', 0)
    n_budget = sum(ctr.get(s, 0) for s in ['BUDGET_CPS', 'BUDGET_RULES', 'BUDGET_WALL'])
    n_dormant_used = sum(1 for r in records if r['n_dormant'] > 0)

    summary = {
        'config': {
            'n_equations': n_total,
            'max_rules': args.max_rules,
            'max_critical_pairs': args.max_cps,
            'max_term_size': args.max_term_size,
            'wall_seconds': args.wall_seconds,
            'pre_filtered_phase0_incomparable': True,
        },
        'status_counts': dict(ctr),
        'status_fractions': {k: v / n_total for k, v in ctr.items()},
        'headline': {
            'n_completed': n_completed,
            'n_fully_completed': n_fully,
            'n_completed_with_size_cap_drops_or_dormant': n_completed - n_fully,
            'completion_rate': n_completed / n_total,
            'fully_completion_rate': n_fully / n_total,
            'n_no_rules_from_input': n_no_rules,
            'n_phase0_excluded': n_p0,
            'n_budget': n_budget,
            'n_completed_with_dormant_cps': n_dormant_used,
        },
        'total_wall_seconds': total_wall,
    }

    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    os.makedirs(os.path.dirname(_abs(args.output_summary)), exist_ok=True)
    with open(_abs(args.output_summary), 'w') as f:
        json.dump(_clean(summary), f, indent=2)
    print(f'wrote {_abs(args.output_summary)}')

    os.makedirs(os.path.dirname(_abs(args.output_records)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(records),
                   _abs(args.output_records), compression='zstd')
    print(f'wrote {_abs(args.output_records)}')

    print()
    print('=== Phase 4 outcomes ===')
    print(f'  COMPLETED:              {n_completed:>5,}  ({100*n_completed/n_total:5.2f}%)')
    print(f'    of which fully:       {n_fully:>5,}  (sound TRUE+FALSE)')
    print(f'    of which downgraded:  {n_completed - n_fully:>5,}  (cap drops or dormant CPs; TRUE-only)')
    print(f'  NO_RULES_FROM_INPUT:    {n_no_rules:>5,}  ({100*n_no_rules/n_total:5.2f}%)')
    print(f'  PHASE0_EXCLUDED:        {n_p0:>5,}  ({100*n_p0/n_total:5.2f}%)')
    print(f'  BUDGET_CPS:             {ctr.get("BUDGET_CPS", 0):>5,}  ({100*ctr.get("BUDGET_CPS", 0)/n_total:5.2f}%)')
    print(f'  BUDGET_RULES:           {ctr.get("BUDGET_RULES", 0):>5,}  ({100*ctr.get("BUDGET_RULES", 0)/n_total:5.2f}%)')
    print(f'  BUDGET_WALL:            {ctr.get("BUDGET_WALL", 0):>5,}  ({100*ctr.get("BUDGET_WALL", 0)/n_total:5.2f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
