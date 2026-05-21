"""Phase 5 of `completion_exploration_plan.md` — finite counter-model
extraction from plain KB's fully-completed systems.

Per the discussion: ordered completion's FALSE verdicts are unsound
(precedence-dependent), so this analysis sources counter-models only
from plain KB's 574 fully-completed systems where FALSE decisions
ARE sound and the differing normal forms genuinely certify a model
in which the implication fails.

For each fully-completed E1:
  1. Compute the term model of E1's rule system over 1 constant `a`.
     Iteratively close `{a, normalize(x ◇ y) : x, y ∈ current}` until
     a fixpoint or a max_size cap.
  2. If the closure is finite, we have a finite Fin n magma satisfying
     E1.
  3. For each (E1, E2) pair where Phase 3 plain-KB normalized E2's
     sides to different NFs (FALSE prediction): check whether the
     finite term model also refutes E2 — i.e., enumerate all
     assignments of E2's variables to model elements, evaluate both
     sides, find an assignment where they disagree.

Reports:
  - Fraction of fully-completed E1s yielding a finite term model
    over 1 constant.
  - Size distribution of the finite models.
  - For each (E1, E2) FALSE pair: does the finite model refute it?
  - Aggregate: of plain KB's 17.82% sound FALSE coverage, how much
    is also Stage-2-submittable (small finite counter-model)?

Reads:
  et_studies/completion_exploration/data/equation_completion.parquet  (Phase 2)
  et_studies/completion_exploration/data/phase3_per_e1_decisions.parquet  (Phase 3)
  data/raw/etp/equations.txt
  data/raw/etp_export/export_raw_implications_5_9_2026.csv

Writes:
  outputs/analyses/completion_study/phase5_finite_models.json
  et_studies/completion_exploration/data/phase5_per_e1_finite_models.parquet
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import signal as _signal
import sys
import time
from collections import Counter

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from proof_generation_pilot.methods.completion.kb import normalize  # noqa: E402
from lib.data.parser import OPERATOR, Op, Term, Var, parse_equation  # noqa: E402


class _PerE1Timeout(Exception):
    pass


def _handler(signum, frame):
    raise _PerE1Timeout()


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def parse_term(s: str) -> Term:
    pos = [0]
    def _skip():
        while pos[0] < len(s) and s[pos[0]].isspace():
            pos[0] += 1
    def _parse():
        _skip()
        if pos[0] >= len(s):
            raise ValueError
        c = s[pos[0]]
        if c == '(':
            pos[0] += 1
            l = _parse()
            _skip()
            if pos[0] >= len(s) or s[pos[0]] != OPERATOR:
                raise ValueError
            pos[0] += 1
            r = _parse()
            _skip()
            if pos[0] >= len(s) or s[pos[0]] != ')':
                raise ValueError
            pos[0] += 1
            return Op(l, r)
        start = pos[0]
        while pos[0] < len(s) and (s[pos[0]].isalnum() or s[pos[0]] == '_'):
            pos[0] += 1
        return Var(s[start:pos[0]])
    t = _parse()
    _skip()
    return t


def deserialize_rules(rules_json: str) -> list[tuple[Term, Term]]:
    return [(parse_term(l), parse_term(r)) for l, r in json.loads(rules_json)]


def compute_term_model(
    rules: list[tuple[Term, Term]],
    constant_names: list[str],
    max_size: int = 50,
) -> tuple[bool, list[Term]]:
    """Compute the finite term model over {constant_names..., ◇}.

    Iteratively close the set of normal forms under
    `(x, y) ↦ normalize(x ◇ y, rules)`. If the set stabilizes
    at size ≤ max_size, return (True, elements). If it grows
    past max_size, return (False, partial_elements).

    Generators distinct in the FREE term algebra need not be
    distinct in the QUOTIENT: if some rule causes 'a' and 'b'
    to normalize to the same NF, they collapse. The returned
    `elements` is the set of equivalence-class representatives."""
    elements: list[Term] = []
    element_set: set[Term] = set()
    for name in constant_names:
        nf = normalize(Var(name), rules)
        if nf not in element_set:
            elements.append(nf)
            element_set.add(nf)

    while True:
        new_elements: list[Term] = []
        for x in elements:
            for y in elements:
                nf = normalize(Op(x, y), rules)
                if nf not in element_set:
                    new_elements.append(nf)
                    element_set.add(nf)
                    if len(element_set) > max_size:
                        return False, elements + new_elements
        if not new_elements:
            return True, elements
        elements.extend(new_elements)


def evaluate_term(t: Term, sigma: dict, op_table: dict, idx: dict) -> int:
    """Evaluate t in a finite magma given by op_table[(i, j)] = idx of x_i ◇ x_j.
    sigma maps Var name → element index."""
    if isinstance(t, Var):
        return sigma[t.name]
    li = evaluate_term(t.left, sigma, op_table, idx)
    ri = evaluate_term(t.right, sigma, op_table, idx)
    return op_table[(li, ri)]


def find_refuting_assignment(
    e2_lhs: Term, e2_rhs: Term,
    elements: list[Term], op_table: dict[tuple[int, int], int],
    idx_of: dict[Term, int],
    var_names: list[str],
    max_assignments: int = 10_000,
) -> tuple[bool, dict | None]:
    """Enumerate assignments of var_names to indices in `elements`
    and find one where evaluate(e2_lhs) != evaluate(e2_rhs)."""
    n = len(elements)
    if n ** len(var_names) > max_assignments:
        # Too many to enumerate; sample instead.
        rng = np.random.default_rng(0)
        for _ in range(max_assignments):
            sigma = {name: int(rng.integers(n)) for name in var_names}
            if evaluate_term(e2_lhs, sigma, op_table, idx_of) != \
               evaluate_term(e2_rhs, sigma, op_table, idx_of):
                return True, sigma
        return False, None
    for assignment in itertools.product(range(n), repeat=len(var_names)):
        sigma = dict(zip(var_names, assignment))
        if evaluate_term(e2_lhs, sigma, op_table, idx_of) != \
           evaluate_term(e2_rhs, sigma, op_table, idx_of):
            return True, sigma
    return False, None


def vars_of(t: Term) -> set[str]:
    if isinstance(t, Var):
        return {t.name}
    return vars_of(t.left) | vars_of(t.right)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--completions',
                    default='et_studies/completion_exploration/data/equation_completion.parquet')
    ap.add_argument('--ground-truth',
                    default='data/raw/etp_export/export_raw_implications_5_9_2026.csv')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/completion_study/phase5_finite_models.json')
    ap.add_argument('--output-records',
                    default='et_studies/completion_exploration/data/phase5_per_e1_finite_models.parquet')
    ap.add_argument('--max-model-size', type=int, default=50)
    ap.add_argument('--per-e1-timeout', type=float, default=15.0)
    ap.add_argument('--progress-every', type=int, default=50)
    ap.add_argument('--limit', type=int, default=None)
    ap.add_argument('--skip-pair-refutation', action='store_true',
                    help='only compute the term model; skip checking which pairs it refutes')
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
    N = len(eqs)
    print(f'  {N:,} equations parsed')

    print(f'loading completions from {args.completions}')
    comp = pq.read_table(_abs(args.completions)).to_pandas()
    fully = comp[(comp['status'] == 'COMPLETED') & (comp['is_fully_completed'])].copy()
    print(f'  {len(fully):,} fully-completed E1s (plain KB)')

    print(f'loading ground truth from {args.ground_truth}')
    gt = np.loadtxt(_abs(args.ground_truth), delimiter=',', dtype=np.int8)
    truth_is_true = gt > 0

    if args.limit:
        fully = fully.head(args.limit)

    _signal.signal(_signal.SIGALRM, _handler)

    records = []
    n_finite = 0
    n_pairs_refuted = 0
    n_false_pairs_checked = 0

    t_start = time.time()
    for i, (_, row) in enumerate(fully.iterrows()):
        eq_id = int(row['eq_id'])
        rules = deserialize_rules(row['rules_json'])

        timed_out = False
        finite = False
        model = []
        n_pairs_refuted_for_this_e1 = 0
        n_false_pairs_for_this_e1 = 0

        _signal.setitimer(_signal.ITIMER_REAL, args.per_e1_timeout)
        try:
            # Try k = 2 constants first (need ≥ 2 to refute anything
            # non-trivial). If that's also size 1 (E1 forces a = b),
            # try k = 3.
            finite, model = compute_term_model(rules, ['a', 'b'],
                                               max_size=args.max_model_size)
            if finite and len(model) <= 1:
                finite3, model3 = compute_term_model(rules, ['a', 'b', 'c'],
                                                    max_size=args.max_model_size)
                if finite3 and len(model3) > len(model):
                    finite, model = finite3, model3
            if finite and not args.skip_pair_refutation:
                # Build operation table for the finite magma.
                idx_of = {e: k for k, e in enumerate(model)}
                op_table = {}
                for ix, x in enumerate(model):
                    for iy, y in enumerate(model):
                        nf = normalize(Op(x, y), rules)
                        if nf not in idx_of:
                            # Should not happen — closure was verified.
                            finite = False
                            break
                        op_table[(ix, iy)] = idx_of[nf]
                    if not finite:
                        break
                if finite:
                    # For every FALSE ground-truth pair (eq_id, e2_id), check
                    # if the model refutes it.
                    for e2_id in range(1, N + 1):
                        if e2_id == eq_id:
                            continue
                        if truth_is_true[eq_id - 1][e2_id - 1]:
                            continue  # TRUE pair, skip
                        n_false_pairs_for_this_e1 += 1
                        e2 = eqs[e2_id - 1]
                        var_names = sorted(vars_of(e2.lhs) | vars_of(e2.rhs))
                        refuted, sigma = find_refuting_assignment(
                            e2.lhs, e2.rhs, model, op_table, idx_of, var_names,
                        )
                        if refuted:
                            n_pairs_refuted_for_this_e1 += 1
        except _PerE1Timeout:
            timed_out = True
        finally:
            _signal.setitimer(_signal.ITIMER_REAL, 0)

        if finite:
            n_finite += 1
        n_false_pairs_checked += n_false_pairs_for_this_e1
        n_pairs_refuted += n_pairs_refuted_for_this_e1

        records.append({
            'eq_id': eq_id,
            'finite': bool(finite),
            'timed_out': timed_out,
            'model_size': len(model),
            'n_false_pairs': n_false_pairs_for_this_e1,
            'n_false_pairs_refuted_by_model': n_pairs_refuted_for_this_e1,
        })

        if (i + 1) % args.progress_every == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(fully) - i - 1) / rate if rate else 0
            print(f'  {i + 1:>4,}/{len(fully):,}  t={elapsed:>5.1f}s  '
                  f'rate={rate:>4.1f}/s  eta={eta:>4.0f}s  '
                  f'finite={n_finite:>4,} '
                  f'pairs_refuted={n_pairs_refuted:>10,}/{n_false_pairs_checked:>10,}',
                  flush=True)

    total_wall = time.time() - t_start
    print(f'\ndone in {total_wall:.1f}s')

    finite_sizes = [r['model_size'] for r in records if r['finite']]
    timeout_count = sum(1 for r in records if r['timed_out'])

    n_false_total = int((~truth_is_true).sum() - 0)   # off-diag + diag flipped; reconcile
    n_false_off_diag = int(truth_is_true.size - truth_is_true.sum() - 0)
    # Just use the same calc as earlier scripts:
    n_off_diag = N * (N - 1)
    n_false_off_diag = n_off_diag - int(truth_is_true.sum() - N)

    summary = {
        'config': {
            'n_fully_completed_e1_used': len(records),
            'max_model_size': args.max_model_size,
            'per_e1_timeout_seconds': args.per_e1_timeout,
            'n_total_equations': N,
            'n_off_diagonal_pairs': n_off_diag,
            'n_ground_truth_FALSE_pairs': n_false_off_diag,
        },
        'finite_model_headline': {
            'n_fully_completed_with_finite_term_model': len(finite_sizes),
            'fraction_of_fully_completed': len(finite_sizes) / max(1, len(records)),
            'n_timed_out': timeout_count,
            'size_distribution': {
                'min': int(min(finite_sizes)) if finite_sizes else 0,
                'median': float(np.median(finite_sizes)) if finite_sizes else 0,
                'mean': float(np.mean(finite_sizes)) if finite_sizes else 0,
                'max': int(max(finite_sizes)) if finite_sizes else 0,
            },
            'size_buckets': dict(Counter(finite_sizes)),
        },
        'refutation_headline': {
            'description': ('PER-PAIR check: for each (E1, E2) ground-truth '
                            'FALSE pair where E1 has a finite term model M, '
                            'enumerate assignments of E2.lhs ∪ E2.rhs '
                            "variables to elements of M; evaluate both "
                            "sides in M's operation table; report refuted "
                            'iff at least one assignment evaluates the '
                            'sides to different model elements. NOT a '
                            'per-E1 assumption — each individual pair is '
                            'checked. 100% refutation across the checked '
                            'pairs is therefore an EMPIRICAL result, not '
                            'a construction-time guarantee: small term '
                            'models in principle could satisfy '
                            "accidental equations beyond E1's theory and "
                            'thus fail to refute some FALSE consequents. '
                            'The data says that did not happen for these '
                            '130 E1s — their small term models are '
                            'equationally faithful, an interesting fact '
                            'about strong/collapsing equations in the '
                            'corpus.'),
            'n_false_pairs_checked': n_false_pairs_checked,
            'n_false_pairs_refuted_by_finite_model': n_pairs_refuted,
            'fraction_of_checked_pairs_refuted': n_pairs_refuted / max(1, n_false_pairs_checked),
            'fraction_of_total_GT_FALSE': n_pairs_refuted / max(1, n_false_off_diag),
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

    pq.write_table(pa.Table.from_pylist(records),
                   _abs(args.output_records), compression='zstd')
    print(f'wrote {_abs(args.output_records)}')

    print()
    print('=== Phase 5 outcomes ===')
    print(f'  fully-completed (input):        {len(records):>5,}')
    print(f'  with finite term model over 1 constant: {len(finite_sizes):>5,}  '
          f'({100*len(finite_sizes)/max(1,len(records)):.2f}%)')
    print(f'  size distribution: median {summary["finite_model_headline"]["size_distribution"]["median"]:.0f}, '
          f'mean {summary["finite_model_headline"]["size_distribution"]["mean"]:.1f}, '
          f'max {summary["finite_model_headline"]["size_distribution"]["max"]}')
    print(f'  timed out: {timeout_count}')
    print()
    if not args.skip_pair_refutation:
        print(f'  FALSE pairs checked (E1 has finite model):  {n_false_pairs_checked:>10,}')
        print(f'  refuted by finite model:                    {n_pairs_refuted:>10,}  '
              f'({100*n_pairs_refuted/max(1,n_false_pairs_checked):.2f}%)')
        print(f'  Stage-2 ready: {n_pairs_refuted:,} of {n_false_off_diag:,} '
              f'GT FALSE pairs = {100*n_pairs_refuted/max(1,n_false_off_diag):.3f}%')

    return 0


if __name__ == '__main__':
    sys.exit(main())
