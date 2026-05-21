"""Phase 3 of `completion_exploration_plan.md` — validate completion
as a decision procedure and measure pair-level coverage.

For each equation E1 that Phase 2 reported as COMPLETED, use its
rule system R to decide every (E1, E2) implication pair:

    nf_lhs = normalize(E2.lhs, R)
    nf_rhs = normalize(E2.rhs, R)
    prediction = TRUE  if nf_lhs == nf_rhs
                 FALSE if nf_lhs != nf_rhs

The soundness asymmetry, per Phase 2's `is_fully_completed` flag:

  - TRUE predictions are sound for ALL 1,005 completed E1s. Rules
    are LPO-oriented consequences of E1; an instance σ(rule) is a
    consequence too. So normalize is a sound rewriting in E1's
    equational theory; if two NFs are equal, both sides genuinely
    represent the same value in any model of E1.

  - FALSE predictions are sound ONLY for the 574 fully-completed E1s
    (no CPs dropped). When CPs were dropped by the size cap, the
    system is not confluent — two terms may have different NFs yet
    still be equal in E1's theory (via a CP we never processed).

Two headlines:

  TRUE pair coverage   = sum over 1,005 completed E1s of pairs
                          where prediction == TRUE == ground truth,
                          divided by total off-diagonal pairs.
  FALSE pair coverage  = sum over 574 fully-completed E1s of pairs
                          where prediction == FALSE == ground truth.

Any TRUE prediction that's FALSE in ground truth: soundness bug
(the completed system unsound for TRUE — should be impossible).
Any FALSE prediction from a fully-completed E1 that's TRUE in
ground truth: soundness bug (the fully-completed system unsound
for FALSE — should be impossible).

Reads:
  et_studies/completion_exploration/data/equation_completion.parquet
  data/raw/etp/equations.txt
  data/raw/etp_export/export_raw_implications_5_9_2026.csv (4694×4694 int8)

Writes:
  outputs/analyses/completion_study/phase3_decision_coverage.json
  et_studies/completion_exploration/data/phase3_per_e1_decisions.parquet
  et_studies/completion_exploration/data/phase3_disagreements.parquet
    (any soundness violations — should be empty)
"""

from __future__ import annotations

import argparse
import json
import os
import signal as _signal
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# Term operations recurse on term depth. Phase 2 size cap = 30 means
# stored terms have depth ≤ 30, but transient construction during
# normalize can spike. Generous ceiling so we never blow the stack.
sys.setrecursionlimit(20000)


class _PerE1Timeout(Exception):
    """Raised when a single E1's decision pass exceeds the per-E1 budget."""


def _e1_alarm_handler(signum, frame):
    raise _PerE1Timeout()

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from proof_generation_pilot.methods.completion.kb import normalize  # noqa: E402
from lib.data.parser import OPERATOR, Op, Term, Var, parse_equation  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def parse_term(term_repr: str) -> Term:
    """Parse a Term's repr (from `Op.__repr__` / `Var.__repr__`) back
    to a Term object. The parser in src/data/parser.py is restricted
    to canonical variables {u..z}; this helper handles repr strings
    that may contain renamed variables like `x_1`, `y_2` (artifacts of
    CP unification during completion). Format:

      Var: identifier (chars in [A-Za-z0-9_])
      Op:  '(' <term> ' ◇ ' <term> ')'
    """
    pos = [0]
    s = term_repr

    def _skip_ws():
        while pos[0] < len(s) and s[pos[0]].isspace():
            pos[0] += 1

    def _parse() -> Term:
        _skip_ws()
        if pos[0] >= len(s):
            raise ValueError(f'unexpected end of {s!r}')
        c = s[pos[0]]
        if c == '(':
            pos[0] += 1
            left = _parse()
            _skip_ws()
            if pos[0] >= len(s) or s[pos[0]] != OPERATOR:
                raise ValueError(f'expected {OPERATOR!r} at {pos[0]} in {s!r}')
            pos[0] += 1
            right = _parse()
            _skip_ws()
            if pos[0] >= len(s) or s[pos[0]] != ')':
                raise ValueError(f'expected ")" at {pos[0]} in {s!r}')
            pos[0] += 1
            return Op(left, right)
        # Variable: consume identifier chars
        start = pos[0]
        while pos[0] < len(s) and (s[pos[0]].isalnum() or s[pos[0]] == '_'):
            pos[0] += 1
        if start == pos[0]:
            raise ValueError(f'expected identifier at {pos[0]} in {s!r}')
        return Var(s[start:pos[0]])

    t = _parse()
    _skip_ws()
    if pos[0] != len(s):
        raise ValueError(f'trailing text at {pos[0]} in {s!r}')
    return t


def deserialize_rules(rules_json: str) -> list[tuple[Term, Term]]:
    pairs = json.loads(rules_json)
    return [(parse_term(l), parse_term(r)) for l, r in pairs]


def code_is_true(code: int) -> bool:
    """+3, +4 → TRUE; -3, -4 → FALSE."""
    return code > 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--completions',
                    default='et_studies/completion_exploration/data/equation_completion.parquet')
    ap.add_argument('--ground-truth',
                    default='data/raw/etp_export/export_raw_implications_5_9_2026.csv')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/completion_study/phase3_decision_coverage.json')
    ap.add_argument('--output-per-e1',
                    default='et_studies/completion_exploration/data/phase3_per_e1_decisions.parquet')
    ap.add_argument('--output-disagreements',
                    default='et_studies/completion_exploration/data/phase3_disagreements.parquet')
    ap.add_argument('--progress-every', type=int, default=100)
    ap.add_argument('--per-e1-timeout', type=float, default=30.0,
                    help='SIGALRM-enforced budget per E1 (decision pass over all 4693 E2s)')
    ap.add_argument('--limit', type=int, default=None,
                    help='process only first N completed E1s (for dev)')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    for p in [args.output_summary, args.output_per_e1, args.output_disagreements]:
        p = _abs(p)
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    # --- 1. load everything ---
    print(f'loading equations from {args.equations}')
    with open(_abs(args.equations)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    eqs = [parse_equation(line) for line in lines]
    N = len(eqs)
    print(f'  {N:,} equations parsed')

    print(f'loading ground-truth matrix from {args.ground_truth}')
    t0 = time.time()
    gt = np.loadtxt(_abs(args.ground_truth), delimiter=',', dtype=np.int8)
    print(f'  shape {gt.shape}, loaded in {time.time() - t0:.1f}s')
    assert gt.shape == (N, N), f'expected ({N},{N}), got {gt.shape}'
    # Build TRUE mask: ±3, ±4 with positive sign means TRUE.
    truth_is_true = gt > 0   # bool matrix [N, N]
    n_true_total = int(truth_is_true.sum() - N)  # exclude diagonal (always TRUE)
    n_false_total = int(N * N - truth_is_true.sum())
    n_off_diag = N * (N - 1)
    print(f'  off-diagonal pairs: {n_off_diag:,}')
    print(f'  TRUE pairs (off-diag): {n_true_total:,} ({100*n_true_total/n_off_diag:.2f}%)')
    print(f'  FALSE pairs (off-diag): {n_false_total:,} ({100*n_false_total/n_off_diag:.2f}%)')

    print(f'loading Phase 2 completions from {args.completions}')
    comp = pq.read_table(_abs(args.completions)).to_pandas()
    completed = comp[comp['status'] == 'COMPLETED'].copy()
    fully = completed[completed['is_fully_completed']].copy()
    print(f'  {len(completed):,} completed E1s')
    print(f'    of which fully (sound for TRUE+FALSE): {len(fully):,}')
    print(f'    of which with size-cap drops (TRUE-sound only): {len(completed) - len(fully):,}')

    if args.limit:
        completed = completed.head(args.limit)
        print(f'  limit applied: processing first {args.limit} completed E1s only')

    # --- 2. iterate over completed E1s; decide all (E1, E2) pairs ---
    per_e1_records = []
    disagreements = []
    timed_out_e1s = []   # IDs whose decision pass exceeded per-E1 budget
    t_start = time.time()
    n_done = 0

    # Install SIGALRM handler once; we set the timer per E1 below.
    _signal.signal(_signal.SIGALRM, _e1_alarm_handler)

    for _, row in completed.iterrows():
        e1_id = int(row['eq_id'])
        is_fully = bool(row['is_fully_completed'])
        rules = deserialize_rules(row['rules_json'])

        n_pairs = 0
        n_true_pred = 0
        n_false_pred = 0
        n_true_correct = 0          # pred TRUE, truth TRUE
        n_false_correct = 0         # pred FALSE, truth FALSE
        n_true_wrong = 0            # pred TRUE, truth FALSE — SOUNDNESS BUG
        n_false_wrong = 0           # pred FALSE, truth TRUE — bug if fully-completed
        timed_out = False
        last_e2_done = 0

        _signal.setitimer(_signal.ITIMER_REAL, args.per_e1_timeout)
        try:
            for e2_id in range(1, N + 1):
                if e2_id == e1_id:
                    continue
                n_pairs += 1
                last_e2_done = e2_id
                e2 = eqs[e2_id - 1]
                nf_lhs = normalize(e2.lhs, rules)
                nf_rhs = normalize(e2.rhs, rules)
                pred_true = (nf_lhs == nf_rhs)
                truth = bool(truth_is_true[e1_id - 1][e2_id - 1])

                if pred_true:
                    n_true_pred += 1
                    if truth:
                        n_true_correct += 1
                    else:
                        n_true_wrong += 1
                        disagreements.append({
                            'e1': e1_id, 'e2': e2_id,
                            'is_fully_completed': is_fully,
                            'prediction': 'TRUE', 'truth': 'FALSE',
                            'description': 'rules unsound: implies a non-implication',
                        })
                else:
                    n_false_pred += 1
                    if not truth:
                        n_false_correct += 1
                    else:
                        n_false_wrong += 1
                        # Bug ONLY if fully-completed; otherwise expected.
                        if is_fully:
                            disagreements.append({
                                'e1': e1_id, 'e2': e2_id,
                                'is_fully_completed': True,
                                'prediction': 'FALSE', 'truth': 'TRUE',
                                'description': 'fully-complete: NFs differ but is implication — soundness bug',
                            })
        except _PerE1Timeout:
            timed_out = True
            timed_out_e1s.append({'eq_id': e1_id, 'last_e2_completed': last_e2_done,
                                  'is_fully_completed': is_fully})
        finally:
            _signal.setitimer(_signal.ITIMER_REAL, 0)

        per_e1_records.append({
            'eq_id': e1_id,
            'is_fully_completed': is_fully,
            'n_pairs': n_pairs,
            'n_true_pred': n_true_pred,
            'n_false_pred': n_false_pred,
            'n_true_correct': n_true_correct,
            'n_false_correct': n_false_correct,
            'n_true_wrong': n_true_wrong,
            'n_false_wrong': n_false_wrong,
            'timed_out': timed_out,
            'last_e2_processed': last_e2_done,
        })

        n_done += 1
        if n_done % args.progress_every == 0 or timed_out:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (len(completed) - n_done) / rate if rate > 0 else 0
            agg_true = sum(r['n_true_correct'] for r in per_e1_records)
            agg_false_full = sum(r['n_false_correct'] for r in per_e1_records
                                 if r['is_fully_completed'])
            marker = '  TIMEOUT' if timed_out else ''
            print(f'  {n_done:>4,}/{len(completed):,}  e1={e1_id:>4}  '
                  f't={elapsed:>6.1f}s  rate={rate:>5.1f}/s  eta={eta:>5.0f}s  '
                  f'TRUE_correct={agg_true:>10,} '
                  f'FALSE_correct_fully={agg_false_full:>10,} '
                  f'disagreements={len(disagreements):>3} '
                  f'timeouts={len(timed_out_e1s):>3}{marker}',
                  flush=True)

    total_wall = time.time() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    # --- 3. aggregate headlines ---
    agg_true_correct_all = sum(r['n_true_correct'] for r in per_e1_records)
    agg_true_pred_all = sum(r['n_true_pred'] for r in per_e1_records)
    agg_true_wrong_all = sum(r['n_true_wrong'] for r in per_e1_records)
    # FALSE counts only over fully-completed E1s
    agg_false_correct_fully = sum(r['n_false_correct'] for r in per_e1_records
                                  if r['is_fully_completed'])
    agg_false_pred_fully = sum(r['n_false_pred'] for r in per_e1_records
                               if r['is_fully_completed'])
    agg_false_wrong_fully = sum(r['n_false_wrong'] for r in per_e1_records
                                if r['is_fully_completed'])
    # FALSE wrong over partial-completed = expected (NFs differ but is implication
    # via a dropped CP). Tracked but not a bug.
    agg_false_wrong_partial = sum(r['n_false_wrong'] for r in per_e1_records
                                  if not r['is_fully_completed'])

    n_completed = len(per_e1_records)
    n_fully = sum(1 for r in per_e1_records if r['is_fully_completed'])

    summary = {
        'config': {
            'n_equations': N,
            'n_off_diagonal_pairs': n_off_diag,
            'n_completed_e1_used': n_completed,
            'n_fully_completed_e1_used': n_fully,
            'ground_truth_total_TRUE': n_true_total,
            'ground_truth_total_FALSE': n_false_total,
        },
        'true_side_headline': {
            'description': ('TRUE pair coverage: among ALL 1,005 completed E1s, '
                            'pairs where the rule system predicts TRUE and the '
                            'ground truth is TRUE. Sound for any COMPLETED E1.'),
            'n_true_correct': agg_true_correct_all,
            'n_true_pred': agg_true_pred_all,
            'n_true_wrong (soundness bugs)': agg_true_wrong_all,
            'fraction_of_all_off_diag_pairs': agg_true_correct_all / n_off_diag,
            'fraction_of_ground_truth_TRUE_pairs': agg_true_correct_all / n_true_total,
        },
        'false_side_headline': {
            'description': ('FALSE pair coverage: among ONLY the fully-completed '
                            'E1s (no CPs dropped), pairs where NFs differ AND '
                            'ground truth FALSE. Sound only for fully-completed.'),
            'n_false_correct': agg_false_correct_fully,
            'n_false_pred': agg_false_pred_fully,
            'n_false_wrong (soundness bugs)': agg_false_wrong_fully,
            'fraction_of_all_off_diag_pairs': agg_false_correct_fully / n_off_diag,
            'fraction_of_ground_truth_FALSE_pairs': agg_false_correct_fully / n_false_total,
        },
        'partial_completion_FALSE_predictions': {
            'description': ('FALSE predictions from partial-completed E1s. NOT '
                            'sound — some are real FALSE, some are TRUE via a '
                            'dropped CP. Reported only as info.'),
            'n_false_wrong_partial': agg_false_wrong_partial,
        },
        'combined_pair_coverage': {
            'description': ('Total pairs (TRUE+FALSE) correctly decided by '
                            'completion, dividing by all off-diagonal pairs. '
                            'TRUE coverage from all completed E1s + FALSE '
                            'coverage from fully-completed E1s only.'),
            'n_correct_total': agg_true_correct_all + agg_false_correct_fully,
            'fraction_total': (agg_true_correct_all + agg_false_correct_fully) / n_off_diag,
        },
        'wall_seconds': total_wall,
        'n_disagreements_logged': len(disagreements),
        'n_e1_timed_out': len(timed_out_e1s),
        'per_e1_timeout_seconds': args.per_e1_timeout,
        'timed_out_e1_ids': [t['eq_id'] for t in timed_out_e1s],
    }

    def _clean(obj):
        if isinstance(obj, dict): return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, list): return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        return obj

    os.makedirs(os.path.dirname(_abs(args.output_summary)), exist_ok=True)
    with open(_abs(args.output_summary), 'w') as f:
        json.dump(_clean(summary), f, indent=2)
    print(f'wrote {_abs(args.output_summary)}')

    os.makedirs(os.path.dirname(_abs(args.output_per_e1)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(per_e1_records),
                   _abs(args.output_per_e1), compression='zstd')
    print(f'wrote {_abs(args.output_per_e1)}')

    if disagreements:
        pq.write_table(pa.Table.from_pylist(disagreements),
                       _abs(args.output_disagreements), compression='zstd')
        print(f'wrote {_abs(args.output_disagreements)}  '
              f'({len(disagreements)} disagreements logged)')
    else:
        print('no disagreements to write — soundness check passed')

    # --- 4. headline ---
    print()
    print('=== Phase 3 pair-level decision coverage ===')
    print(f'  Total off-diagonal pairs in 22M problem: {n_off_diag:,}')
    print(f'  Ground truth TRUE: {n_true_total:,}, FALSE: {n_false_total:,}')
    print()
    print(f'  TRUE-side  (base: {n_completed:,} completed E1s):')
    print(f'    n_correct          : {agg_true_correct_all:>12,}')
    print(f'    soundness bugs     : {agg_true_wrong_all:>12,}')
    print(f'    pair coverage      : {100*agg_true_correct_all/n_off_diag:6.3f}% of 22M')
    print(f'    of ground-truth TRUE: {100*agg_true_correct_all/n_true_total:6.3f}%')
    print()
    print(f'  FALSE-side (base: {n_fully:,} FULLY-completed E1s):')
    print(f'    n_correct          : {agg_false_correct_fully:>12,}')
    print(f'    soundness bugs     : {agg_false_wrong_fully:>12,}')
    print(f'    pair coverage      : {100*agg_false_correct_fully/n_off_diag:6.3f}% of 22M')
    print(f'    of ground-truth FALSE: {100*agg_false_correct_fully/n_false_total:6.3f}%')
    print()
    print(f'  Combined coverage    : {100*(agg_true_correct_all + agg_false_correct_fully)/n_off_diag:6.3f}% of 22M')
    return 0


if __name__ == '__main__':
    sys.exit(main())
