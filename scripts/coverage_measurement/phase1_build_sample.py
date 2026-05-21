"""Phase 1 of `coverage_measurement_plan.md` — build the GT-TRUE
residual relative to completion and a stratified sample for the
rewrite-engine measurement.

Residual definition:
  residual = {(E1, E2) ∈ GT TRUE : E1 ∉ ordered_completed ∪ plain_only_completed}

`ordered_completed` is the set of E1s whose Phase 4 ordered-completion
status is COMPLETED (2,129 equations). `plain_only_completed` is the
set where Phase 2 plain-KB COMPLETED but Phase 4 didn't (the tight
ordered-CP budget loses a handful of weak plain-KB completions).

For each E1 NOT in the union, stratified-sample N pairs from E1's
GT TRUE row uniformly (cap at the row's actual size). Annotate each
sampled pair with E1's variable-disjoint flag (Phase 0) and TRUE-row
density. Reproducible via fixed seed.

Reads:
  data/raw/etp/equations.txt
  data/raw/etp_export/export_raw_implications_5_9_2026.csv
  et_studies/completion_exploration/data/equation_completion.parquet         (Phase 2)
  et_studies/completion_exploration/data/equation_ordered_completion.parquet (Phase 4)
  et_studies/completion_exploration/data/equation_orientability.parquet      (Phase 0)

Writes:
  et_studies/coverage_measurement/data/residual.parquet
  et_studies/coverage_measurement/data/sample.parquet
  outputs/analyses/coverage_measurement/phase1_residual_summary.json
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--equations', default='data/raw/etp/equations.txt')
    ap.add_argument('--ground-truth',
                    default='data/raw/etp_export/export_raw_implications_5_9_2026.csv')
    ap.add_argument('--plain-completion',
                    default='et_studies/completion_exploration/data/equation_completion.parquet')
    ap.add_argument('--ordered-completion',
                    default='et_studies/completion_exploration/data/equation_ordered_completion.parquet')
    ap.add_argument('--orientability',
                    default='et_studies/completion_exploration/data/equation_orientability.parquet')
    ap.add_argument('--output-residual',
                    default='et_studies/coverage_measurement/data/residual.parquet')
    ap.add_argument('--output-sample',
                    default='et_studies/coverage_measurement/data/sample.parquet')
    ap.add_argument('--output-summary',
                    default='outputs/analyses/coverage_measurement/phase1_residual_summary.json')
    ap.add_argument('--sample-size', type=int, default=30,
                    help='N pairs per non-completed E1 (cap at row size)')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    for p in [args.output_residual, args.output_sample, args.output_summary]:
        if os.path.exists(_abs(p)) and not args.force:
            print(f'ERROR: {_abs(p)} exists; pass --force.', file=sys.stderr)
            return 2

    # --- 1. load ---
    print(f'loading ground truth from {args.ground_truth}')
    t0 = time.time()
    gt = np.loadtxt(_abs(args.ground_truth), delimiter=',', dtype=np.int8)
    N = gt.shape[0]
    truth_is_true = gt > 0
    print(f'  shape {gt.shape}, loaded in {time.time() - t0:.1f}s')
    print(f'  GT TRUE off-diagonal: {int(truth_is_true.sum() - N):,}')

    print(f'loading orientability from {args.orientability}')
    p0 = pq.read_table(_abs(args.orientability)).to_pandas()
    var_disjoint = set(int(r) for r in p0[p0['classification'] == 'incomparable']['eq_id'])
    print(f'  variable-disjoint equations: {len(var_disjoint):,}')

    print(f'loading plain-KB completion from {args.plain_completion}')
    p2 = pq.read_table(_abs(args.plain_completion)).to_pandas()
    plain_completed = set(int(r) for r in p2[p2['status'] == 'COMPLETED']['eq_id'])
    print(f'  plain-KB COMPLETED: {len(plain_completed):,}')

    print(f'loading ordered-KB completion from {args.ordered_completion}')
    p4 = pq.read_table(_abs(args.ordered_completion)).to_pandas()
    ordered_completed = set(int(r) for r in p4[p4['status'] == 'COMPLETED']['eq_id'])
    print(f'  ordered-KB COMPLETED: {len(ordered_completed):,}')

    completed = ordered_completed | plain_completed
    plain_only = plain_completed - ordered_completed
    print(f'  union completed: {len(completed):,}')
    print(f'  plain-only (not ordered): {len(plain_only):,}')

    # --- 2. residual annotation ---
    print()
    print('annotating all 4,694 equations …')
    rows = []
    for e1 in range(1, N + 1):
        row_true = int(truth_is_true[e1 - 1].sum() - 1)   # subtract diagonal
        is_var_disjoint = (e1 in var_disjoint)
        in_completed = (e1 in completed)
        rows.append({
            'eq_id': e1,
            'in_completed': in_completed,
            'in_plain_only': (e1 in plain_only),
            'in_ordered_completed': (e1 in ordered_completed),
            'in_residual': not in_completed,
            'is_variable_disjoint': is_var_disjoint,
            'true_row_density': row_true,
        })

    n_residual = sum(1 for r in rows if r['in_residual'])
    n_residual_var_disjoint = sum(1 for r in rows if r['in_residual'] and r['is_variable_disjoint'])
    n_residual_within_ceiling = n_residual - n_residual_var_disjoint
    residual_true_pairs = sum(r['true_row_density'] for r in rows if r['in_residual'])
    residual_var_disjoint_pairs = sum(r['true_row_density'] for r in rows
                                       if r['in_residual'] and r['is_variable_disjoint'])
    residual_within_ceiling_pairs = residual_true_pairs - residual_var_disjoint_pairs

    print(f'  residual E1s (non-completed): {n_residual:,}')
    print(f'    of which variable-disjoint: {n_residual_var_disjoint:,}')
    print(f'    within-ceiling (rewritable): {n_residual_within_ceiling:,}')
    print(f'  GT TRUE pairs in residual: {residual_true_pairs:,}')
    print(f'    in variable-disjoint rows: {residual_var_disjoint_pairs:,}')
    print(f'    in within-ceiling rows: {residual_within_ceiling_pairs:,}')

    # --- 3. sample ---
    print()
    print(f'sampling N={args.sample_size} per non-completed E1 (seed={args.seed}) …')
    rng = np.random.default_rng(args.seed)
    sample_rows = []
    for r in rows:
        if not r['in_residual']:
            continue
        e1 = r['eq_id']
        true_e2s = np.where(truth_is_true[e1 - 1])[0]
        # Exclude the diagonal (e1 - 1).
        true_e2s = true_e2s[true_e2s != e1 - 1]
        if len(true_e2s) == 0:
            continue
        n_take = min(args.sample_size, len(true_e2s))
        # Uniform sample without replacement.
        chosen = rng.choice(true_e2s, size=n_take, replace=False)
        for e2_idx in chosen:
            e2 = int(e2_idx) + 1
            sample_rows.append({
                'e1': e1, 'e2': e2,
                'e1_is_variable_disjoint': r['is_variable_disjoint'],
                'e1_true_row_density': r['true_row_density'],
                'in_within_ceiling': not r['is_variable_disjoint'],
            })

    n_sample = len(sample_rows)
    n_sample_within_ceiling = sum(1 for s in sample_rows if s['in_within_ceiling'])
    print(f'  total sampled pairs: {n_sample:,}')
    print(f'    within-ceiling segment: {n_sample_within_ceiling:,}')
    print(f'    variable-disjoint segment: {n_sample - n_sample_within_ceiling:,}')

    # --- 4. write outputs ---
    os.makedirs(os.path.dirname(_abs(args.output_residual)), exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows),
                   _abs(args.output_residual), compression='zstd')
    print(f'wrote {_abs(args.output_residual)}')

    pq.write_table(pa.Table.from_pylist(sample_rows),
                   _abs(args.output_sample), compression='zstd')
    print(f'wrote {_abs(args.output_sample)}')

    summary = {
        'config': {
            'sample_size_per_e1': args.sample_size,
            'seed': args.seed,
            'n_equations': N,
        },
        'completion_union': {
            'ordered_completed': len(ordered_completed),
            'plain_completed': len(plain_completed),
            'plain_only_not_ordered': len(plain_only),
            'union_completed': len(completed),
        },
        'residual': {
            'n_e1_residual': n_residual,
            'n_e1_residual_variable_disjoint': n_residual_var_disjoint,
            'n_e1_residual_within_ceiling': n_residual_within_ceiling,
            'n_true_pairs_in_residual': int(residual_true_pairs),
            'n_true_pairs_residual_variable_disjoint': int(residual_var_disjoint_pairs),
            'n_true_pairs_residual_within_ceiling': int(residual_within_ceiling_pairs),
        },
        'sample': {
            'n_pairs': n_sample,
            'n_within_ceiling': n_sample_within_ceiling,
            'n_variable_disjoint': n_sample - n_sample_within_ceiling,
        },
        'ceiling_estimate': {
            'description': 'rough preview of the structural ceiling',
            'gt_true_total_off_diag': int(truth_is_true.sum() - N),
            'gt_true_in_variable_disjoint_rows': int(
                sum(r['true_row_density'] for r in rows if r['is_variable_disjoint'])
            ),
            'gt_true_fraction_in_variable_disjoint_rows': float(
                sum(r['true_row_density'] for r in rows if r['is_variable_disjoint'])
                / max(1, int(truth_is_true.sum() - N))
            ),
        },
    }

    os.makedirs(os.path.dirname(_abs(args.output_summary)), exist_ok=True)
    with open(_abs(args.output_summary), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'wrote {_abs(args.output_summary)}')

    print()
    print('=== Phase 1 outcomes ===')
    print(f'  residual E1s: {n_residual:,} of {N:,}')
    print(f'    variable-disjoint: {n_residual_var_disjoint:,}')
    print(f'    within-ceiling: {n_residual_within_ceiling:,}')
    print(f'  GT TRUE pairs in residual: {residual_true_pairs:,}')
    print(f'    structural-impossibility (engine cannot reach): {residual_var_disjoint_pairs:,}')
    print(f'    within-ceiling (engine candidate): {residual_within_ceiling_pairs:,}')
    print()
    print(f'  Ceiling pre-check (GT TRUE in variable-disjoint rows including completed):')
    print(f'    {summary["ceiling_estimate"]["gt_true_in_variable_disjoint_rows"]:,} '
          f'({100*summary["ceiling_estimate"]["gt_true_fraction_in_variable_disjoint_rows"]:.2f}% of GT TRUE)')
    print()
    print(f'  Sampled pairs for Phase 3 engine run: {n_sample:,}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
