"""Run the first-order matcher across the full implication pair corpus.

Combines Phases 5-8 of `matcher_plan.md` into a single corpus walk:

  Phase 5 — coverage on the directly-proved set
  Phase 6 — coverage on the full TRUE corpus
  Phase 7 — false-positive check on the full FALSE corpus
  Phase 8 — save the ground-truth proof dataset

Inputs:
  - data/raw/etp/equations.txt           4,694 equations, line N = Eq N
  - data/processed/implication_pairs.parquet
                                         22,028,942 (eq1_id, eq2_id, value, code)
  - et_studies/proof_survey/data/true_corpus.parquet
                                         the survey's 10,676 directly-proved
                                         implications (Phase 5 target)

Outputs:
  - et_studies/matcher_deliverable/data/matcher_proofs.parquet
        per matched pair: eq1_id, eq2_id, orientation, substitution[6],
        lean_proof_term, lean_proof_tactic. Gitignored.
  - et_studies/matcher_deliverable/outputs/coverage.json
        aggregate counts: directly_proved, full_TRUE, full_FALSE (must be 0).
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import parse_equation  # noqa: E402
from proof_generation_pilot.methods.matcher.matcher import match, ordered_vars  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def load_equations(path: str) -> list:
    """equations[i] = parse of line i+1 (i.e., Equation i+1, 1-indexed)."""
    with open(_abs(path)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    return [parse_equation(line) for line in lines]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--equations', default='data/raw/etp/equations.txt')
    parser.add_argument('--pair-table',
                        default='data/processed/implication_pairs.parquet')
    parser.add_argument('--directly-proved-corpus',
                        default='et_studies/proof_survey/data/true_corpus.parquet',
                        help="survey's 10,676 directly-proved-implication set")
    parser.add_argument('--output-proofs',
                        default='et_studies/matcher_deliverable/data/matcher_proofs.parquet')
    parser.add_argument('--output-coverage',
                        default='et_studies/matcher_deliverable/outputs/coverage.json')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--progress-every', type=int, default=1_000_000)
    args = parser.parse_args(argv)

    proofs_path = _abs(args.output_proofs)
    coverage_path = _abs(args.output_coverage)
    for p in [proofs_path, coverage_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    # --- 1. parse equations once ---
    print(f'loading equations from {args.equations}')
    eqs = load_equations(args.equations)
    print(f'  parsed {len(eqs):,} equations')

    # --- 2. load pair table ---
    print(f'loading pair table from {args.pair_table}')
    pair_tbl = pq.read_table(_abs(args.pair_table),
                             columns=['eq1_id', 'eq2_id', 'value'])
    eq1_arr = pair_tbl['eq1_id'].to_numpy()
    eq2_arr = pair_tbl['eq2_id'].to_numpy()
    val_arr = pair_tbl['value'].to_numpy().astype(np.int8)
    n_pairs = len(eq1_arr)
    n_true = int((val_arr == 1).sum())
    n_false = int((val_arr == 0).sum())
    print(f'  {n_pairs:,} pairs ({n_true:,} TRUE / {n_false:,} FALSE)')

    # --- 3. directly-proved set ---
    print(f'loading directly-proved set from {args.directly_proved_corpus}')
    direct_set: set[tuple[int, int]] = set()
    if os.path.exists(_abs(args.directly_proved_corpus)):
        dp_tbl = pq.read_table(_abs(args.directly_proved_corpus),
                               columns=['e1', 'e2'])
        for e1, e2 in zip(dp_tbl['e1'].to_numpy(), dp_tbl['e2'].to_numpy()):
            direct_set.add((int(e1), int(e2)))
        print(f'  {len(direct_set):,} directly-proved pairs')
    else:
        print(f'  (file not found; skipping directly-proved breakdown)')

    # --- 4. walk the pair table ---
    print(f'running matcher on all {n_pairs:,} pairs ...')
    t0 = time.perf_counter()

    # Aggregate buckets
    n_match_true = 0           # TRUE pairs matched (good)
    n_match_false = 0          # FALSE pairs matched (must be 0!)
    n_match_direct = 0         # matched ∩ directly-proved
    n_direct_seen = 0
    false_positive_pairs: list[tuple[int, int]] = []   # capped — for diagnostics

    # Match-result rows (one per successful match)
    matched_e1: list[int] = []
    matched_e2: list[int] = []
    matched_orient: list[int] = []
    matched_subst: list[tuple[int, ...]] = []
    matched_term: list[str] = []
    matched_tactic: list[str] = []

    for i in range(n_pairs):
        e1 = int(eq1_arr[i])
        e2 = int(eq2_arr[i])
        val = int(val_arr[i])
        eq1 = eqs[e1 - 1]
        eq2 = eqs[e2 - 1]
        m = match(eq1, eq2)
        is_direct = (e1, e2) in direct_set
        if is_direct:
            n_direct_seen += 1
        if m is not None:
            if val == 1:
                n_match_true += 1
                if is_direct:
                    n_match_direct += 1
                matched_e1.append(e1)
                matched_e2.append(e2)
                matched_orient.append(m.orientation)
                matched_subst.append(m.substitution_padded())
                matched_term.append(m.proof_term())
                matched_tactic.append(m.proof_tactic_body())
            else:
                n_match_false += 1
                if len(false_positive_pairs) < 20:
                    false_positive_pairs.append((e1, e2))

        if (i + 1) % args.progress_every == 0:
            elapsed = time.perf_counter() - t0
            rate = (i + 1) / max(elapsed, 1e-9)
            eta = (n_pairs - i - 1) / max(rate, 1e-9)
            print(f'  {i + 1:>10,} / {n_pairs:,}  '
                  f'matched_TRUE={n_match_true:>8,}  '
                  f'matched_FALSE={n_match_false:>4}  '
                  f't={elapsed:>6.1f}s  rate={rate:>8,.0f}/s  eta={eta:>6.1f}s')

    elapsed = time.perf_counter() - t0
    print(f'\ndone in {elapsed:.1f}s')

    # --- 5. coverage summary ---
    summary = {
        'corpus': {
            'n_pairs': n_pairs,
            'n_true': n_true,
            'n_false': n_false,
            'n_directly_proved_set': len(direct_set),
            'n_directly_proved_pairs_seen_in_corpus': n_direct_seen,
        },
        'matcher_coverage': {
            'matched_true': n_match_true,
            'matched_true_fraction_of_all_true': n_match_true / max(n_true, 1),
            'matched_directly_proved': n_match_direct,
            'matched_directly_proved_fraction': n_match_direct / max(n_direct_seen, 1),
            'matched_false_FALSE_POSITIVES': n_match_false,
        },
        'wall_clock_seconds': elapsed,
        'pairs_per_second': n_pairs / max(elapsed, 1e-9),
    }
    if false_positive_pairs:
        summary['false_positive_examples_first_20'] = false_positive_pairs

    os.makedirs(os.path.dirname(coverage_path), exist_ok=True)
    with open(coverage_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'wrote {coverage_path}')

    # --- 6. save matched-proof dataset ---
    print(f'saving matched-proof dataset to {proofs_path} ({len(matched_e1):,} rows)')
    subst_arr = np.array(matched_subst, dtype=np.int8)
    table = pa.table({
        'eq1_id': pa.array(matched_e1, type=pa.int32()),
        'eq2_id': pa.array(matched_e2, type=pa.int32()),
        'orientation': pa.array(matched_orient, type=pa.int8()),
        'substitution_0': pa.array(subst_arr[:, 0], type=pa.int8()),
        'substitution_1': pa.array(subst_arr[:, 1], type=pa.int8()),
        'substitution_2': pa.array(subst_arr[:, 2], type=pa.int8()),
        'substitution_3': pa.array(subst_arr[:, 3], type=pa.int8()),
        'substitution_4': pa.array(subst_arr[:, 4], type=pa.int8()),
        'substitution_5': pa.array(subst_arr[:, 5], type=pa.int8()),
        'lean_proof_term': pa.array(matched_term, type=pa.string()),
        'lean_proof_tactic': pa.array(matched_tactic, type=pa.string()),
    })
    os.makedirs(os.path.dirname(proofs_path), exist_ok=True)
    pq.write_table(table, proofs_path, compression='zstd')

    # --- 7. headline ---
    print()
    print('=== Matcher coverage ===')
    print(f"  TRUE coverage    : {n_match_true:>9,} / {n_true:,} "
          f"= {100*n_match_true/max(n_true,1):.3f}%")
    if direct_set:
        print(f"  Directly-proved  : {n_match_direct:>9,} / {n_direct_seen:,} "
              f"= {100*n_match_direct/max(n_direct_seen,1):.3f}% "
              f"(expect ~62.5% per the survey)")
    print(f"  FALSE positives  : {n_match_false:>9,} / {n_false:,} "
          f"(MUST be 0)")
    if n_match_false > 0:
        print(f"  WARNING: false positives detected. First 20: {false_positive_pairs}")
    print(f"  wall_clock       : {elapsed:.1f}s ({n_pairs/max(elapsed,1e-9):,.0f} pairs/s)")
    return 0 if n_match_false == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
