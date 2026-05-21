"""Phase 4a — classify the failing critical pair in each
FAILED_ORIENT_DERIVED equation, to predict ordered completion's
recovery ceiling before building it.

For each of the 1,581 equations whose Phase 2 run failed on a derived
CP, re-run plain KB completion and capture the failing CP (the
equation `(s, t)` that orient() refused). Classify by the same
variable-inclusion scheme as Phase 0:

  SYMMETRIC      — Vars(s) == Vars(t).
                   Both directions variable-valid, but LPO leaves
                   them incomparable. This is ordered completion's
                   home turf: per-instance orientation succeeds where
                   abstract LPO doesn't. **Phase 4 fully recovers.**
  WRONG_DIR      — One direction has variable inclusion (e.g.,
                   Vars(t) ⊊ Vars(s) so s → t is var-valid), but LPO
                   has t > s (the opposite direction). Variable
                   inclusion holds one way; LPO orients the wrong
                   way. Ordered completion can sometimes apply the
                   equation per-instance. **Phase 4 partially
                   recovers.**
  INCOMPARABLE   — Vars(s) and Vars(t) are incomparable: each has a
                   variable the other lacks. Neither direction is
                   a valid rewrite rule. **Phase 4 cannot recover.**

The classification gives an upper bound on Phase 4's recovery of the
FAILED_ORIENT_DERIVED bucket. (Even fully-classified-SYMMETRIC CPs
can subsequently fail downstream in ordered completion's own cascade,
so this is an upper bound only.)

Reads:
  data/processed/completion_study/equation_completion.parquet
  data/raw/etp/equations.txt

Writes:
  outputs/analyses/completion_study/phase4a_failed_cp_classification.json
  data/processed/completion_study/phase4a_failed_cps.parquet
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

from proof_generation_pilot.methods.completion.kb import (  # noqa: E402
    CompletionStatus,
    complete,
)
from lib.data.parser import parse_equation  # noqa: E402
from lib.rewrite.lpo import lpo_gt  # noqa: E402
from lib.rewrite.unification import collect_vars  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def classify_failed_cp(s, t) -> str:
    """SYMMETRIC / WRONG_DIR / INCOMPARABLE / TRIVIAL_NOW."""
    if s == t:
        return 'TRIVIAL_NOW'   # shouldn't happen — orient() would have skipped
    vs = {v.name for v in collect_vars(s)}
    vt = {v.name for v in collect_vars(t)}
    inc_s_to_t = vt <= vs   # rule s → t needs Vars(t) ⊆ Vars(s)
    inc_t_to_s = vs <= vt   # rule t → s needs Vars(s) ⊆ Vars(t)

    if not (inc_s_to_t or inc_t_to_s):
        return 'INCOMPARABLE'
    if inc_s_to_t and inc_t_to_s:
        return 'SYMMETRIC'    # Vars(s) == Vars(t), LPO must be incomparable too
    # Exactly one direction has variable inclusion; orient() failed which means
    # LPO doesn't orient in that direction.
    return 'WRONG_DIR'


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--completions',
                    default='data/processed/completion_study/equation_completion.parquet')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/completion_study/phase4a_failed_cp_classification.json')
    ap.add_argument('--output-records',
                    default='data/processed/completion_study/phase4a_failed_cps.parquet')
    ap.add_argument('--max-rules', type=int, default=50)
    ap.add_argument('--max-cps', type=int, default=500)
    ap.add_argument('--max-term-size', type=int, default=30)
    ap.add_argument('--wall-seconds', type=float, default=2.0)
    ap.add_argument('--progress-every', type=int, default=200)
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

    print(f'loading Phase 2 completions from {args.completions}')
    comp = pq.read_table(_abs(args.completions)).to_pandas()
    derived = comp[comp['status'] == 'FAILED_ORIENT_DERIVED'].copy()
    print(f'  {len(derived):,} FAILED_ORIENT_DERIVED equations to re-run')

    records = []
    t_start = time.time()
    for i, (_, row) in enumerate(derived.iterrows()):
        eq_id = int(row['eq_id'])
        eq = eqs[eq_id - 1]
        result = complete(
            [(eq.lhs, eq.rhs)],
            max_rules=args.max_rules,
            max_critical_pairs=args.max_cps,
            max_term_size=args.max_term_size,
            wall_seconds=args.wall_seconds,
        )
        # Should match Phase 2's outcome — but be robust to occasional
        # nondeterminism (insertion-order in dicts can shift unification
        # paths slightly across re-runs).
        if result.status != CompletionStatus.FAILED_ORIENT or result.failed_equation is None:
            records.append({
                'eq_id': eq_id,
                'rerun_status': result.status.value,
                'rerun_failed_on_derived': bool(result.failed_on_derived) if result.failed_on_derived is not None else False,
                'failed_s_repr': '',
                'failed_t_repr': '',
                'classification': f'NOT_REPRODUCED_status_{result.status.value}',
                'lhs_n_vars': 0,
                'rhs_n_vars': 0,
                'shared_n_vars': 0,
                'wall_ms': result.wall_seconds * 1000.0,
            })
            continue
        s, t = result.failed_equation
        cls = classify_failed_cp(s, t)
        vs = {v.name for v in collect_vars(s)}
        vt = {v.name for v in collect_vars(t)}
        records.append({
            'eq_id': eq_id,
            'rerun_status': result.status.value,
            'rerun_failed_on_derived': bool(result.failed_on_derived),
            'failed_s_repr': repr(s),
            'failed_t_repr': repr(t),
            'classification': cls,
            'lhs_n_vars': len(vs),
            'rhs_n_vars': len(vt),
            'shared_n_vars': len(vs & vt),
            'wall_ms': result.wall_seconds * 1000.0,
        })

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t_start
            ctr = Counter(r['classification'] for r in records)
            print(f'  {i + 1:>5,}/{len(derived):,}  t={elapsed:>6.1f}s  '
                  f"SYM={ctr.get('SYMMETRIC', 0):>4} "
                  f"WRONG_DIR={ctr.get('WRONG_DIR', 0):>4} "
                  f"INCOMP={ctr.get('INCOMPARABLE', 0):>4}",
                  flush=True)

    total_wall = time.time() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    ctr = Counter(r['classification'] for r in records)
    n_total = len(records)
    summary = {
        'config': {
            'n_failed_orient_derived_equations': n_total,
            'max_rules': args.max_rules,
            'max_critical_pairs': args.max_cps,
            'max_term_size': args.max_term_size,
            'wall_seconds': args.wall_seconds,
        },
        'classification_counts': dict(ctr),
        'classification_fractions': {k: v / n_total for k, v in ctr.items()},
        'phase4_recovery_estimate': {
            'description': 'Upper bound on Phase 4 ordered-completion recovery of the FAILED_ORIENT_DERIVED bucket. SYMMETRIC are fully recoverable abstractly (per-instance orientation succeeds); WRONG_DIR are partially recoverable (some instances orient OK); INCOMPARABLE are not recoverable.',
            'symmetric_count': ctr.get('SYMMETRIC', 0),
            'wrong_dir_count': ctr.get('WRONG_DIR', 0),
            'incomparable_count': ctr.get('INCOMPARABLE', 0),
            'upper_bound_recoverable': ctr.get('SYMMETRIC', 0) + ctr.get('WRONG_DIR', 0),
            'upper_bound_unrecoverable': ctr.get('INCOMPARABLE', 0),
        },
        'wall_seconds': total_wall,
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
    print('=== Failed-CP classification ===')
    print(f'  Total FAILED_ORIENT_DERIVED inputs: {n_total:,}')
    for k in ['SYMMETRIC', 'WRONG_DIR', 'INCOMPARABLE']:
        v = ctr.get(k, 0)
        print(f'  {k:>14}: {v:>5,} ({100*v/n_total:5.2f}%)')
    other = sum(c for k, c in ctr.items()
                if k not in ('SYMMETRIC', 'WRONG_DIR', 'INCOMPARABLE', 'TRIVIAL_NOW'))
    if other:
        print(f'    other (re-run differed): {other}')
    print()
    print(f'  Phase 4 recovery upper bound:')
    print(f'    SYMMETRIC + WRONG_DIR = '
          f'{ctr.get("SYMMETRIC", 0) + ctr.get("WRONG_DIR", 0):>5,} '
          f'({100*(ctr.get("SYMMETRIC", 0) + ctr.get("WRONG_DIR", 0))/n_total:.2f}%)')
    print(f'  Phase 4 cannot recover:')
    print(f'    INCOMPARABLE = {ctr.get("INCOMPARABLE", 0):,} '
          f'({100*ctr.get("INCOMPARABLE", 0)/n_total:.2f}%)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
