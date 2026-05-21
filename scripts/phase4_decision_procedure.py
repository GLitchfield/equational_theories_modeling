"""Phase 4 decision-procedure validation — measure pair-level coverage
of ordered-completion rule systems against the 22M ground truth.

Mirrors scripts/phase3_decision_procedure.py but uses
src/completion/ordered.py's per-instance LPO rewriting (via
`ordered_normalize`) instead of plain rewriting.

Pre-flags rules once per E1 with strict LPO orientation; rewrites
that pass strict abstract orient bypass the per-instance check at
apply time. Only ordered-rewriting equations (LPO-incomparable,
commutativity-shaped) pay the per-instance cost.

Reports the same TRUE/FALSE split as Phase 3, on the ordered-
completed base.

Reads:
  et_studies/completion_exploration/data/equation_ordered_completion.parquet
  data/raw/etp/equations.txt
  data/raw/etp_export/export_raw_implications_5_9_2026.csv

Writes:
  outputs/analyses/completion_study/phase4_decision_coverage.json
  et_studies/completion_exploration/data/phase4_per_e1_decisions.parquet
  et_studies/completion_exploration/data/phase4_disagreements.parquet
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

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import OPERATOR, Op, Term, Var, parse_equation  # noqa: E402
from lib.rewrite.lpo import DEFAULT_VAR_PRECEDENCE, lpo_gt, lpo_gt_total  # noqa: E402
from lib.rewrite.matching import all_rewrites  # noqa: E402


class _PerE1Timeout(Exception):
    pass


def _e1_alarm_handler(signum, frame):
    raise _PerE1Timeout()


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def parse_term(term_repr: str) -> Term:
    """Same as Phase 3 — handles renamed variables (x_1, etc.)."""
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


def deserialize_and_flag_rules(rules_json: str) -> list[tuple[Term, Term, bool]]:
    """Parse rules + pre-flag with strict-LPO orientation for fast normalize."""
    pairs = json.loads(rules_json)
    return [(parse_term(l), parse_term(r), lpo_gt(parse_term(l), parse_term(r)))
            for l, r in pairs]


def fast_ordered_normalize(
    t: Term,
    flagged_rules: list[tuple[Term, Term, bool]],
    precedence: tuple[str, ...] = DEFAULT_VAR_PRECEDENCE,
) -> Term:
    """ordered_normalize with rules pre-flagged. Strict-LPO-oriented
    rules skip the per-instance check; the rest use lpo_gt_total."""
    while True:
        applied = False
        for l, r, strict in flagged_rules:
            for nt in all_rewrites(t, l, r):
                if strict or lpo_gt_total(t, nt, precedence):
                    t = nt
                    applied = True
                    break
            if applied:
                break
        if not applied:
            return t


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--completions',
                    default='et_studies/completion_exploration/data/equation_ordered_completion.parquet')
    ap.add_argument('--ground-truth',
                    default='data/raw/etp_export/export_raw_implications_5_9_2026.csv')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/completion_study/phase4_decision_coverage.json')
    ap.add_argument('--output-per-e1',
                    default='et_studies/completion_exploration/data/phase4_per_e1_decisions.parquet')
    ap.add_argument('--output-disagreements',
                    default='et_studies/completion_exploration/data/phase4_disagreements.parquet')
    ap.add_argument('--progress-every', type=int, default=100)
    ap.add_argument('--per-e1-timeout', type=float, default=30.0)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    for p in [args.output_summary, args.output_per_e1, args.output_disagreements]:
        if os.path.exists(_abs(p)) and not args.force:
            print(f'ERROR: {_abs(p)} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'loading equations from {args.equations}')
    with open(_abs(args.equations)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    eqs = [parse_equation(line) for line in lines]
    N = len(eqs)
    print(f'  {N:,} parsed')

    print(f'loading ground truth from {args.ground_truth}')
    gt = np.loadtxt(_abs(args.ground_truth), delimiter=',', dtype=np.int8)
    truth_is_true = gt > 0
    n_off_diag = N * (N - 1)
    n_true_total = int(truth_is_true.sum() - N)
    n_false_total = n_off_diag - n_true_total
    print(f'  off-diag pairs: {n_off_diag:,}  TRUE {n_true_total:,}  FALSE {n_false_total:,}')

    print(f'loading Phase 4 completions from {args.completions}')
    comp = pq.read_table(_abs(args.completions)).to_pandas()
    completed = comp[comp['status'] == 'COMPLETED'].copy()
    fully = completed[completed['is_fully_completed']].copy()
    print(f'  {len(completed):,} completed E1s ({len(fully):,} fully)')

    if args.limit:
        completed = completed.head(args.limit)

    per_e1 = []
    disagreements = []
    timed_out = []
    t_start = time.time()
    n_done = 0

    _signal.signal(_signal.SIGALRM, _e1_alarm_handler)

    for _, row in completed.iterrows():
        eq_id = int(row['eq_id'])
        is_fully = bool(row['is_fully_completed'])
        flagged = deserialize_and_flag_rules(row['rules_json'])

        n_pairs = 0
        n_true_pred = 0
        n_false_pred = 0
        n_true_correct = 0
        n_false_correct = 0
        n_true_wrong = 0
        n_false_wrong = 0
        timeout = False
        last_e2 = 0

        _signal.setitimer(_signal.ITIMER_REAL, args.per_e1_timeout)
        try:
            for e2_id in range(1, N + 1):
                if e2_id == eq_id:
                    continue
                n_pairs += 1
                last_e2 = e2_id
                e2 = eqs[e2_id - 1]
                nf_lhs = fast_ordered_normalize(e2.lhs, flagged)
                nf_rhs = fast_ordered_normalize(e2.rhs, flagged)
                pred = (nf_lhs == nf_rhs)
                truth = bool(truth_is_true[eq_id - 1][e2_id - 1])
                if pred:
                    n_true_pred += 1
                    if truth:
                        n_true_correct += 1
                    else:
                        n_true_wrong += 1
                        disagreements.append({
                            'e1': eq_id, 'e2': e2_id,
                            'is_fully_completed': is_fully,
                            'prediction': 'TRUE', 'truth': 'FALSE',
                            'description': 'unsound TRUE: rules imply a non-implication',
                        })
                else:
                    n_false_pred += 1
                    if not truth:
                        n_false_correct += 1
                    else:
                        n_false_wrong += 1
                        # Note: FALSE predictions from ordered completion
                        # are NOT sound (per-instance LPO check is
                        # precedence-dependent on the goal's free
                        # variables). Don't log as bugs — just count.
        except _PerE1Timeout:
            timeout = True
            timed_out.append({'eq_id': eq_id, 'last_e2_completed': last_e2,
                              'is_fully_completed': is_fully})
        finally:
            _signal.setitimer(_signal.ITIMER_REAL, 0)

        per_e1.append({
            'eq_id': eq_id,
            'is_fully_completed': is_fully,
            'n_pairs': n_pairs,
            'n_true_pred': n_true_pred,
            'n_false_pred': n_false_pred,
            'n_true_correct': n_true_correct,
            'n_false_correct': n_false_correct,
            'n_true_wrong': n_true_wrong,
            'n_false_wrong': n_false_wrong,
            'timed_out': timeout,
            'last_e2_processed': last_e2,
        })
        n_done += 1
        if n_done % args.progress_every == 0 or timeout:
            elapsed = time.time() - t_start
            rate = n_done / elapsed
            eta = (len(completed) - n_done) / rate if rate else 0
            agg_t = sum(r['n_true_correct'] for r in per_e1)
            agg_f = sum(r['n_false_correct'] for r in per_e1 if r['is_fully_completed'])
            marker = '  TIMEOUT' if timeout else ''
            print(f'  {n_done:>4,}/{len(completed):,}  e1={eq_id:>4}  '
                  f't={elapsed:>6.1f}s  rate={rate:>5.1f}/s  eta={eta:>5.0f}s  '
                  f'TRUE_correct={agg_t:>10,} '
                  f'FALSE_correct_fully={agg_f:>10,} '
                  f'disagreements={len(disagreements):>3} '
                  f'timeouts={len(timed_out):>3}{marker}',
                  flush=True)

    total_wall = time.time() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    n_completed = len(per_e1)
    n_fully = sum(1 for r in per_e1 if r['is_fully_completed'])
    agg_true_correct = sum(r['n_true_correct'] for r in per_e1)
    agg_true_wrong = sum(r['n_true_wrong'] for r in per_e1)
    agg_false_correct_fully = sum(r['n_false_correct'] for r in per_e1 if r['is_fully_completed'])
    agg_false_wrong_fully = sum(r['n_false_wrong'] for r in per_e1 if r['is_fully_completed'])

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
            'description': ('TRUE pair coverage: rule system normalizes '
                            'both sides to the same NF. SOUND for ordered '
                            'completion — every rewrite step is a sound '
                            'consequence of E1, regardless of precedence.'),
            'base_size': n_completed,
            'n_true_correct': agg_true_correct,
            'n_true_wrong (soundness bugs — must be 0)': agg_true_wrong,
            'fraction_of_22M': agg_true_correct / n_off_diag,
            'fraction_of_ground_truth_TRUE': agg_true_correct / n_true_total,
        },
        'false_side_NOT_SOUND_for_ordered_kb': {
            'description': ('FALSE predictions from ordered completion '
                            'are NOT sound in the equational-implication '
                            'setting. The per-instance LPO check uses a '
                            'fixed variable precedence; the resulting '
                            'NFs depend on names of E2 free variables. '
                            'Two terms equal in E1 may normalize to '
                            'different NFs under one precedence and '
                            'matching NFs under another. Reported here '
                            'only as descriptive count, not a coverage '
                            'claim.'),
            'base_size_fully': n_fully,
            'n_false_predicted': sum(r['n_false_pred'] for r in per_e1 if r['is_fully_completed']),
            'n_false_predicted_agreeing_with_ground_truth': agg_false_correct_fully,
            'n_false_predicted_disagreeing': agg_false_wrong_fully,
            'note': ('Use plain-KB Phase 3 numbers for the sound FALSE '
                     'headline. Ordered KB is for TRUE-side coverage only.'),
        },
        'wall_seconds': total_wall,
        'n_disagreements_logged': len(disagreements),
        'n_e1_timed_out': len(timed_out),
        'per_e1_timeout_seconds': args.per_e1_timeout,
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

    pq.write_table(pa.Table.from_pylist(per_e1),
                   _abs(args.output_per_e1), compression='zstd')
    print(f'wrote {_abs(args.output_per_e1)}')

    if disagreements:
        pq.write_table(pa.Table.from_pylist(disagreements),
                       _abs(args.output_disagreements), compression='zstd')
        print(f'wrote {_abs(args.output_disagreements)} ({len(disagreements)} entries)')
    else:
        print('no disagreements — soundness check passed')

    print()
    print('=== Phase 4 pair-level decision coverage (TRUE-only is sound) ===')
    print(f'  TRUE-side  (base: {n_completed:,} ordered-completed E1s):')
    print(f'    n_correct           : {agg_true_correct:>12,}')
    print(f'    soundness bugs (n_true_wrong, MUST be 0): {agg_true_wrong:>5,}')
    print(f'    coverage of 22M     : {100*agg_true_correct/n_off_diag:6.3f}%')
    print(f'    of ground-truth TRUE: {100*agg_true_correct/n_true_total:6.3f}%')
    print()
    print(f'  FALSE-side (UNSOUND for ordered KB — descriptive only):')
    print(f'    n_predicted from "fully" subset: '
          f'{sum(r["n_false_pred"] for r in per_e1 if r["is_fully_completed"]):>10,}')
    print(f'    of which agree with ground truth: {agg_false_correct_fully:>10,}')
    print(f'    of which disagree (precedence-dependent): {agg_false_wrong_fully:>10,}')
    print(f'    {"(use plain-KB Phase 3 numbers for sound FALSE coverage)":>60}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
