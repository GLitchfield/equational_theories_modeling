"""Phase 2 of `completion_exploration_plan.md` — run KB completion on
every ETP equation and classify outcomes.

Per the user's framing (and the Phase 1 validation results), the
classification scheme refines `FAILED_ORIENT` into two sub-cases:

  COMPLETED                — finite confluent terminating system.
  FAILED_ORIENT_ORIGINAL   — orient() refused on the input equation.
                             These are Phase 0's set: variable-disjoint
                             or strictly symmetric. Ordered completion
                             can't recover these (variable-disjoint
                             requires fresh vars on rewrite; strictly
                             symmetric requires per-instance order check
                             which ordered completion can do but the
                             abstract orientation still fails).
                             Actually, strictly symmetric cases CAN be
                             recovered by ordered completion via
                             per-instance application — these are
                             Phase 4's primary target.
  FAILED_ORIENT_DERIVED    — orient() refused on a critical pair
                             generated mid-run. Input oriented fine,
                             but the cascade produced an unorientable
                             derived equation. New information beyond
                             Phase 0.
  BUDGET_CPS               — non-terminating critical-pair cascade
                             within budget. May or may not terminate
                             with a larger budget. Ordered completion
                             can ALSO loop here — these are distinct
                             from FAILED_ORIENT_* and may be a residual
                             that no completion-based method recovers.
  BUDGET_RULES / BUDGET_WALL — similar.
  PHASE0_EXCLUDED          — pre-filtered: Phase 0 already classified
                             these as incomparable (zero orientable
                             directions). Equivalent to
                             FAILED_ORIENT_ORIGINAL but tagged a priori
                             to save the completion call.

Reads:
  data/raw/etp/equations.txt
  data/processed/completion_study/equation_orientability.parquet
    (Phase 0 output)

Writes:
  outputs/analyses/completion_study/phase2_completion_outcomes.json
    (summary)
  data/processed/completion_study/equation_completion.parquet
    (per-equation record + serialized rule systems for Phase 3 to read)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter

# Many term operations (parse, normalize, unify, term_size) recurse on
# term depth. With a size cap of 30, terms shouldn't exceed depth 30 —
# but during apply_subst chains in unification, intermediate term depth
# can spike. Set a generous ceiling so a single bad equation falls into
# BUDGET_WALL rather than crashing the whole run.
sys.setrecursionlimit(20000)

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.completion.kb import (  # noqa: E402
    CompletionStatus,
    complete,
)
from lib.data.parser import Term, parse_equation  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def refine_status(result, *, phase0_excluded: bool) -> str:
    """Map (CompletionStatus, failed_on_derived) to the refined label."""
    if phase0_excluded:
        return 'PHASE0_EXCLUDED'
    s = result.status
    if s == CompletionStatus.COMPLETED:
        return 'COMPLETED'
    if s == CompletionStatus.FAILED_ORIENT:
        if result.failed_on_derived:
            return 'FAILED_ORIENT_DERIVED'
        return 'FAILED_ORIENT_ORIGINAL'
    if s == CompletionStatus.BUDGET_CPS:
        return 'BUDGET_CPS'
    if s == CompletionStatus.BUDGET_RULES:
        return 'BUDGET_RULES'
    if s == CompletionStatus.BUDGET_WALL:
        return 'BUDGET_WALL'
    return f'UNKNOWN_{s.value}'


def serialize_rules(rules: list[tuple[Term, Term]]) -> list[tuple[str, str]]:
    """Term repr is canonical (parser-driven); good enough for the
    on-disk record."""
    return [(repr(l), repr(r)) for l, r in rules]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--phase0',
                    default='data/processed/completion_study/equation_orientability.parquet')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/completion_study/phase2_completion_outcomes.json')
    ap.add_argument('--output-records',
                    default='data/processed/completion_study/equation_completion.parquet')
    ap.add_argument('--max-rules', type=int, default=50)
    ap.add_argument('--max-cps', type=int, default=500)
    ap.add_argument('--max-term-size', type=int, default=30,
                    help='CPs producing a term larger than this are dropped (bounds normalize cost per iteration)')
    ap.add_argument('--wall-seconds', type=float, default=2.0)
    ap.add_argument('--progress-every', type=int, default=200)
    ap.add_argument('--force', action='store_true')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only first N equations (for dev/spot-check)')
    args = ap.parse_args(argv)

    out_json = _abs(args.output_summary)
    out_parquet = _abs(args.output_records)
    for p in [out_json, out_parquet]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    # --- 1. load ---
    print(f'loading equations from {args.equations}')
    with open(_abs(args.equations)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    eqs = [parse_equation(line) for line in lines]
    print(f'  {len(eqs):,} equations parsed')

    print(f'loading Phase 0 classification from {args.phase0}')
    p0 = pq.read_table(_abs(args.phase0)).to_pandas()
    p0_class = {int(row.eq_id): row.classification for row in p0.itertuples(index=False)}
    n_p0_excluded = sum(1 for cls in p0_class.values() if cls == 'incomparable')
    print(f'  {n_p0_excluded:,} equations classified incomparable (Phase 0 excluded)')

    # --- 2. run completion ---
    print(f'running completion: max_rules={args.max_rules}, '
          f'max_cps={args.max_cps}, wall={args.wall_seconds}s ...')

    records = []
    n = args.limit if args.limit else len(eqs)
    t_start = time.time()
    for i in range(n):
        eq_id = i + 1
        eq = eqs[i]
        phase0_cls = p0_class.get(eq_id, 'unknown')
        excluded = (phase0_cls == 'incomparable')

        if excluded:
            # Pre-filter: don't bother calling complete(); orient would fail
            # on the input. Record as PHASE0_EXCLUDED.
            records.append({
                'eq_id': eq_id,
                'phase0_classification': phase0_cls,
                'status': 'PHASE0_EXCLUDED',
                'n_rules': 0,
                'n_critical_pairs': 0,
                'n_critical_pairs_dropped_by_size_cap': 0,
                'n_iterations': 0,
                'wall_ms': 0.0,
                'failed_on_derived': None,
                'is_fully_completed': False,
                'rules_json': '[]',
            })
        else:
            result = complete(
                [(eq.lhs, eq.rhs)],
                max_rules=args.max_rules,
                max_critical_pairs=args.max_cps,
                max_term_size=args.max_term_size,
                wall_seconds=args.wall_seconds,
            )
            refined = refine_status(result, phase0_excluded=False)
            records.append({
                'eq_id': eq_id,
                'phase0_classification': phase0_cls,
                'status': refined,
                'n_rules': len(result.rules),
                'n_critical_pairs': result.n_critical_pairs_considered,
                'n_critical_pairs_dropped_by_size_cap': result.n_critical_pairs_dropped_by_size_cap,
                'n_iterations': result.n_iterations,
                'wall_ms': result.wall_seconds * 1000.0,
                'failed_on_derived': result.failed_on_derived,
                'is_fully_completed': result.is_fully_completed,
                'rules_json': json.dumps(serialize_rules(result.rules)),
            })

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (n - i - 1) / rate if rate > 0 else 0
            ctr = Counter(r['status'] for r in records)
            print(f'  {i + 1:>5,}/{n:,}  t={elapsed:>6.1f}s  '
                  f'rate={rate:>5.0f}/s  eta={eta:>5.0f}s  '
                  f"COMPLETED={ctr.get('COMPLETED', 0):>4} "
                  f"FAIL_ORIG={ctr.get('FAILED_ORIENT_ORIGINAL', 0):>4} "
                  f"FAIL_DER={ctr.get('FAILED_ORIENT_DERIVED', 0):>4} "
                  f"BUDGET={sum(ctr.get(s, 0) for s in ['BUDGET_CPS', 'BUDGET_RULES', 'BUDGET_WALL']):>4}",
                  flush=True)

    total_wall = time.time() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    # --- 3. summary ---
    ctr = Counter(r['status'] for r in records)
    n_total = len(records)
    n_completed = ctr.get('COMPLETED', 0)
    n_failed_orig = ctr.get('FAILED_ORIENT_ORIGINAL', 0)
    n_failed_der = ctr.get('FAILED_ORIENT_DERIVED', 0)
    n_phase0_excl = ctr.get('PHASE0_EXCLUDED', 0)
    n_budget_cps = ctr.get('BUDGET_CPS', 0)
    n_budget_rules = ctr.get('BUDGET_RULES', 0)
    n_budget_wall = ctr.get('BUDGET_WALL', 0)
    n_budget_any = n_budget_cps + n_budget_rules + n_budget_wall

    # Wall-time distribution among completed runs.
    completed_walls = np.array([r['wall_ms'] for r in records if r['status'] == 'COMPLETED'])
    rule_counts_completed = np.array([r['n_rules'] for r in records if r['status'] == 'COMPLETED'])
    budget_walls = np.array([r['wall_ms'] for r in records
                             if r['status'] in ('BUDGET_CPS', 'BUDGET_RULES', 'BUDGET_WALL')])

    def pct(arr, p):
        return float(np.percentile(arr, p)) if len(arr) else 0.0

    n_fully_completed = sum(1 for r in records if r.get('is_fully_completed'))
    n_completed_with_drops = n_completed - n_fully_completed

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
            'n_fully_completed': n_fully_completed,
            'n_completed_with_size_cap_drops': n_completed_with_drops,
            'completion_rate': n_completed / n_total,
            'fully_completion_rate': n_fully_completed / n_total,
            'n_failed_orient_original': n_failed_orig,
            'n_failed_orient_derived': n_failed_der,
            'n_phase0_excluded': n_phase0_excl,
            'n_budget': n_budget_any,
            'n_budget_cps': n_budget_cps,
            'n_budget_rules': n_budget_rules,
            'n_budget_wall': n_budget_wall,
        },
        'cost_distribution_completed': {
            'wall_ms_median': pct(completed_walls, 50),
            'wall_ms_p95': pct(completed_walls, 95),
            'wall_ms_max': float(completed_walls.max()) if len(completed_walls) else 0.0,
            'n_rules_median': pct(rule_counts_completed, 50),
            'n_rules_p95': pct(rule_counts_completed, 95),
            'n_rules_max': int(rule_counts_completed.max()) if len(rule_counts_completed) else 0,
        },
        'cost_distribution_budget_hit': {
            'wall_ms_median': pct(budget_walls, 50),
            'wall_ms_max': float(budget_walls.max()) if len(budget_walls) else 0.0,
        },
        'total_wall_seconds': total_wall,
    }

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'wrote {out_json}')

    # --- 4. parquet of per-equation records ---
    table = pa.Table.from_pylist(records)
    os.makedirs(os.path.dirname(out_parquet), exist_ok=True)
    pq.write_table(table, out_parquet, compression='zstd')
    print(f'wrote {out_parquet}')

    # --- 5. headline ---
    print()
    print('=== Phase 2 outcomes ===')
    print(f'  COMPLETED:              {n_completed:>5,}  ({100*n_completed/n_total:5.2f}%)')
    print(f'    of which fully:       {n_fully_completed:>5,}  '
          f'(sound for TRUE+FALSE in Phase 3)')
    print(f'    of which with drops:  {n_completed_with_drops:>5,}  '
          f'(sound for TRUE only)')
    print(f'  FAILED_ORIENT_ORIGINAL: {n_failed_orig:>5,}  ({100*n_failed_orig/n_total:5.2f}%)')
    print(f'  FAILED_ORIENT_DERIVED:  {n_failed_der:>5,}  ({100*n_failed_der/n_total:5.2f}%)')
    print(f'  PHASE0_EXCLUDED:        {n_phase0_excl:>5,}  ({100*n_phase0_excl/n_total:5.2f}%)')
    print(f'  BUDGET_CPS:             {n_budget_cps:>5,}  ({100*n_budget_cps/n_total:5.2f}%)')
    print(f'  BUDGET_RULES:           {n_budget_rules:>5,}  ({100*n_budget_rules/n_total:5.2f}%)')
    print(f'  BUDGET_WALL:            {n_budget_wall:>5,}  ({100*n_budget_wall/n_total:5.2f}%)')
    print()
    print(f'  Phase 0 ceiling (orientable directions): '
          f'{n_total - n_phase0_excl:,} ({100*(n_total-n_phase0_excl)/n_total:.2f}%)')
    print(f'  Plain-KB completion rate:                '
          f'{n_completed:,} ({100*n_completed/n_total:.2f}%)')
    gap = (n_total - n_phase0_excl) - n_completed
    print(f'  Gap (orientable but did not complete):   '
          f'{gap:,} ({100*gap/n_total:.2f}%)')
    print(f'    of which FAILED_ORIENT_DERIVED:        {n_failed_der:,}')
    print(f'    of which loopers (BUDGET_*):           {n_budget_any:,}')
    print(f'    of which strictly symmetric (FAIL_ORIG, not Phase 0 excluded): {n_failed_orig:,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
