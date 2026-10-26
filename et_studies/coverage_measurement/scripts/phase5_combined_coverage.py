"""Phase 5 of `coverage_measurement_plan.md` — combined coverage
analysis: union of matcher + safe rewrite engine + completion as
fractions of GT TRUE.

Computes:
  - Per-method pair coverage (matcher / engine / completion).
  - Pairwise and three-way overlaps.
  - Union coverage:
      * sample-level (engine's solve rate within the sample).
      * pair-level (row-density-weighted extrapolation to GT TRUE).
  - Combined coverage as a fraction of:
      * GT TRUE (the natural denominator)
      * the structural ceiling (≈46% of GT TRUE; refined by Phase 1)
      * the measurement-scope ceiling (engine within-ceiling
        residual + completion + matcher var-disjoint addendum)
  - Stratification cuts:
      * by E1 properties (variable-disjoint flag, row density)
      * by proof-survey category — left as a follow-up here; the
        survey's category labels per-equation aren't joined into the
        coverage parquets yet. We surface this as future-work and
        compute the rest.

Reads:
  et_studies/coverage_measurement/data/sample.parquet
  et_studies/coverage_measurement/data/engine_results_within_ceiling_d8.parquet
  et_studies/coverage_measurement/data/engine_results_variable_disjoint_d6.parquet
  et_studies/coverage_measurement/data/residual.parquet
  et_studies/completion_exploration/data/equation_ordered_completion.parquet (Phase 4)
  et_studies/completion_exploration/data/phase3_per_e1_decisions.parquet     (Phase 3 plain KB)
  et_studies/completion_exploration/data/phase4_per_e1_decisions.parquet     (Phase 4 ordered)
  et_studies/matcher_deliverable/data/matcher_proofs.parquet
  et_studies/completion_exploration/data/equation_orientability.parquet      (Phase 0)
  data/raw/etp_export/export_raw_implications_5_9_2026.csv

Writes:
  et_studies/coverage_measurement/outputs/phase5_combined_coverage.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output',
                    default='et_studies/coverage_measurement/outputs/phase5_combined_coverage.json')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    if os.path.exists(_abs(args.output)) and not args.force:
        print(f'ERROR: {_abs(args.output)} exists; pass --force.', file=sys.stderr)
        return 2

    # --- load ---
    print('loading ground truth ...')
    gt = np.loadtxt(_abs('data/raw/etp_export/export_raw_implications_5_9_2026.csv'),
                    delimiter=',', dtype=np.int8)
    N = gt.shape[0]
    truth_is_true = gt > 0
    n_off_diag = N * (N - 1)
    n_gt_true = int(truth_is_true.sum() - N)
    n_gt_false = n_off_diag - n_gt_true
    print(f'  GT TRUE (off-diag): {n_gt_true:,}')
    print(f'  GT FALSE (off-diag): {n_gt_false:,}')

    print('loading residual + sample ...')
    residual = pq.read_table(_abs('et_studies/coverage_measurement/data/residual.parquet')).to_pandas()
    sample = pq.read_table(_abs('et_studies/coverage_measurement/data/sample.parquet')).to_pandas()
    print(f'  residual: {len(residual):,} E1s, sample: {len(sample):,} pairs')

    print('loading engine results ...')
    engine_wc = pq.read_table(_abs('et_studies/coverage_measurement/data/engine_results_within_ceiling_d8.parquet')).to_pandas()
    engine_vd = pq.read_table(_abs('et_studies/coverage_measurement/data/engine_results_variable_disjoint_d6.parquet')).to_pandas()
    print(f'  within-ceiling at d=8: {len(engine_wc):,} ({int(engine_wc["solved"].sum()):,} solved)')
    print(f'  variable-disjoint at d=6: {len(engine_vd):,} ({int(engine_vd["solved"].sum()):,} solved)')

    print('loading matcher ...')
    matcher = pq.read_table(_abs('et_studies/matcher_deliverable/data/matcher_proofs.parquet'),
                            columns=['eq1_id', 'eq2_id']).to_pandas()
    print(f'  matcher pairs: {len(matcher):,}')

    print('loading completion records ...')
    p4_ord = pq.read_table(_abs('et_studies/completion_exploration/data/equation_ordered_completion.parquet')).to_pandas()
    ordered_completed_e1s = set(int(r) for r in p4_ord[p4_ord['status'] == 'COMPLETED']['eq_id'])

    # Phase 3 (plain) per-pair predictions; Phase 4 (ordered) per-pair predictions.
    # We use ordered's TRUE predictions as the completion TRUE-coverage on its base.
    # For pairs where E1 is plain-only-completed (not ordered), use plain's TRUE predictions.
    p3_per_e1 = pq.read_table(_abs('et_studies/completion_exploration/data/phase3_per_e1_decisions.parquet')).to_pandas()
    p4_per_e1 = pq.read_table(_abs('et_studies/completion_exploration/data/phase4_per_e1_decisions.parquet')).to_pandas()
    n_ord_true = int(p4_per_e1['n_true_correct'].sum())
    n_plain_only_true = 0
    plain_completed = set(int(r) for r in p3_per_e1['eq_id'])
    plain_only_e1s = plain_completed - ordered_completed_e1s
    if plain_only_e1s:
        plain_only_true = p3_per_e1[p3_per_e1['eq_id'].isin(plain_only_e1s)]['n_true_correct'].sum()
        n_plain_only_true = int(plain_only_true)
    completion_true_total = n_ord_true + n_plain_only_true
    print(f'  completion TRUE: ordered {n_ord_true:,} + plain-only {n_plain_only_true:,} = {completion_true_total:,}')

    # Phase 0
    p0 = pq.read_table(_abs('et_studies/completion_exploration/data/equation_orientability.parquet')).to_pandas()
    vd_e1s = set(int(r) for r in p0[p0['classification'] == 'incomparable']['eq_id'])

    # --- A. Matcher pair-level coverage ---
    print()
    print('--- A. Matcher coverage ---')
    matcher['in_vd_row'] = matcher['eq1_id'].astype(int).isin(vd_e1s)
    n_matcher_total = len(matcher)
    n_matcher_vd = int(matcher['in_vd_row'].sum())
    n_matcher_within = n_matcher_total - n_matcher_vd
    print(f'  matcher total: {n_matcher_total:,}')
    print(f'    var-disjoint E1: {n_matcher_vd:,} ({100*n_matcher_vd/n_gt_true:.4f}% of GT TRUE)')
    print(f'    within-ceiling E1: {n_matcher_within:,}')

    # Matcher pairs whose E1 is in completion's union
    matcher_e1s_completed = matcher['eq1_id'].astype(int).isin(ordered_completed_e1s | plain_completed)
    n_matcher_completed = int(matcher_e1s_completed.sum())
    print(f'  matcher pairs whose E1 is completion-completed: {n_matcher_completed:,}')
    print(f'  matcher pairs adding NEW coverage (E1 not completed): {n_matcher_total - n_matcher_completed:,}')

    # --- B. Sample-level engine coverage on within-ceiling residual ---
    print()
    print('--- B. Engine coverage of within-ceiling residual ---')
    n_engine_wc_solved = int(engine_wc['solved'].sum())
    n_engine_wc_total = len(engine_wc)
    print(f'  sample-level solve rate: {n_engine_wc_solved:,}/{n_engine_wc_total:,}'
          f' = {100*n_engine_wc_solved/n_engine_wc_total:.2f}%')

    # Row-density-weighted extrapolation:
    # For each sampled pair (e1, e2), its weight = (e1_true_row_density / sample_count_for_e1).
    # Sum solved pairs * weight = estimate of solved pairs in the full within-ceiling residual.
    # Equivalently: per-E1 solve-rate × row density, summed.
    sample_wc = sample[~sample['e1_is_variable_disjoint']].copy()
    # Build (e1, solved) by joining engine_wc to sample_wc on (e1, e2)
    eng_wc_map = {(int(r['e1']), int(r['e2'])): bool(r['solved'])
                  for _, r in engine_wc.iterrows()}
    sample_wc['solved'] = [eng_wc_map.get((int(r['e1']), int(r['e2'])), False)
                           for _, r in sample_wc.iterrows()]

    # Per-E1 stats
    per_e1 = sample_wc.groupby('e1').agg(
        e1_true_row_density=('e1_true_row_density', 'first'),
        n_sampled=('e2', 'count'),
        n_solved=('solved', 'sum'),
    ).reset_index()
    per_e1['solve_rate'] = per_e1['n_solved'] / per_e1['n_sampled']
    per_e1['estimated_solved_in_full_row'] = per_e1['solve_rate'] * per_e1['e1_true_row_density']

    n_within_ceiling_residual_pairs = int(per_e1['e1_true_row_density'].sum())
    estimated_engine_pair_level_solved = int(per_e1['estimated_solved_in_full_row'].sum())
    print(f'  within-ceiling residual GT TRUE pairs (from row densities): '
          f'{n_within_ceiling_residual_pairs:,}')
    print(f'  pair-level (row-density-weighted) estimated solved: '
          f'{estimated_engine_pair_level_solved:,}')
    print(f'  pair-level fraction of GT TRUE: '
          f'{100*estimated_engine_pair_level_solved/n_gt_true:.3f}%')

    # --- C. Completion coverage (already measured in Phase 3/4) ---
    print()
    print('--- C. Completion coverage ---')
    print(f'  completion TRUE: {completion_true_total:,} = '
          f'{100*completion_true_total/n_gt_true:.3f}% of GT TRUE')

    # --- D. Overlap analysis ---
    print()
    print('--- D. Overlap analysis ---')
    # Matcher var-disjoint pairs: cannot overlap completion (completion can't decide
    # variable-disjoint E1 rows) nor engine (var-disjoint excluded). These are UNIQUELY
    # matcher's contribution.
    # Matcher within-ceiling pairs: could overlap engine and completion.

    # For combined union, sum and subtract overlaps. Approximations:
    #   - matcher_var_disjoint: unique to matcher (n_matcher_vd pairs).
    #   - matcher_within_completed: overlaps with completion (matcher pairs on completed E1).
    #     For these we already count via completion's coverage.
    #   - matcher_within_not_completed: should overlap with engine on those exact pairs
    #     (engine at d_max=2 subsumes matcher proofs structurally, per Phase 2 spot-check).

    matcher_e1s_within_ceiling_not_completed = (
        ~matcher['in_vd_row'] & ~matcher_e1s_completed
    )
    n_matcher_within_not_completed = int(matcher_e1s_within_ceiling_not_completed.sum())
    matcher_within_completed_count = int(
        (~matcher['in_vd_row'] & matcher_e1s_completed).sum()
    )
    print(f'  matcher pairs by overlap:')
    print(f'    var-disjoint E1 (unique to matcher): {n_matcher_vd:,}')
    print(f'    within-ceiling, E1 completed (overlaps completion): {matcher_within_completed_count:,}')
    print(f'    within-ceiling, E1 not completed (overlaps engine where engine solves): {n_matcher_within_not_completed:,}')

    # Note: engine sample solve rate is 14.19% within-ceiling. Of matcher's
    # within-ceiling not-completed pairs, ~all would be solvable by engine at
    # d_max=2 (per spot-check), so they're entirely overlapped by engine.

    # --- E. Combined union pair-level estimate ---
    # Approximation: union ≈ completion + engine_extrapolated_residual + matcher_var_disjoint.
    # This treats overlaps as 0 except where structurally guaranteed:
    #   - matcher var-disjoint is disjoint from completion and engine by construction.
    #   - engine residual is disjoint from completion (residual excludes completed E1).
    #   - matcher within-ceiling-not-completed pairs are subsumed by engine (skip).
    #   - matcher within-ceiling-completed pairs are subsumed by completion (skip).
    combined_pair_estimate = (
        completion_true_total
        + estimated_engine_pair_level_solved
        + n_matcher_vd
    )
    print()
    print('--- E. Combined sound TRUE coverage ---')
    print(f'  completion: {completion_true_total:,} '
          f'({100*completion_true_total/n_gt_true:.3f}%)')
    print(f'  engine (within-ceiling residual, row-weighted): '
          f'{estimated_engine_pair_level_solved:,} '
          f'({100*estimated_engine_pair_level_solved/n_gt_true:.3f}%)')
    print(f'  matcher (var-disjoint addendum): {n_matcher_vd:,} '
          f'({100*n_matcher_vd/n_gt_true:.3f}%)')
    print(f'  COMBINED: {combined_pair_estimate:,} '
          f'({100*combined_pair_estimate/n_gt_true:.3f}% of GT TRUE)')

    # Structural ceiling: GT TRUE pairs NOT in variable-disjoint E1 rows.
    n_gt_true_var_disjoint = 0
    for e1 in vd_e1s:
        n_gt_true_var_disjoint += int(truth_is_true[e1 - 1].sum()) - 1   # -1 diagonal
    structural_ceiling = n_gt_true - n_gt_true_var_disjoint
    print()
    print(f'  structural ceiling (GT TRUE in non-var-disjoint rows): '
          f'{structural_ceiling:,} ({100*structural_ceiling/n_gt_true:.2f}% of GT TRUE)')
    print(f'  combined / structural_ceiling: '
          f'{100*combined_pair_estimate/structural_ceiling:.2f}%')

    # --- F. Stratifications ---
    print()
    print('--- F. Per-E1 stratification (within-ceiling) ---')
    # Distribution of per-E1 solve rates
    print(f'  per-E1 solve-rate distribution (within-ceiling):')
    print(f'    median: {per_e1["solve_rate"].median():.3f}')
    print(f'    mean: {per_e1["solve_rate"].mean():.3f}')
    print(f'    p25: {per_e1["solve_rate"].quantile(0.25):.3f}')
    print(f'    p75: {per_e1["solve_rate"].quantile(0.75):.3f}')

    n_e1_solved_any = int((per_e1['n_solved'] > 0).sum())
    n_e1_solved_zero = int((per_e1['n_solved'] == 0).sum())
    print(f'  per-E1 binary breakdown:')
    print(f'    E1s with ≥1 sample solved: {n_e1_solved_any}/{len(per_e1)}')
    print(f'    E1s with 0 sample solved: {n_e1_solved_zero}/{len(per_e1)}')

    # Correlation between row density and solve rate
    if len(per_e1) > 1:
        corr = float(per_e1[['e1_true_row_density', 'solve_rate']].corr().iloc[0, 1])
        print(f'  Pearson corr(row_density, solve_rate): {corr:.3f}')

    # --- G. Save summary ---
    summary = {
        'gt_truth_totals': {
            'gt_true': n_gt_true,
            'gt_false': n_gt_false,
            'off_diag_pairs': n_off_diag,
        },
        'matcher_breakdown': {
            'total': n_matcher_total,
            'variable_disjoint_e1': n_matcher_vd,
            'within_ceiling_e1': n_matcher_within,
            'completed_e1_overlap_with_completion': matcher_within_completed_count,
            'not_completed_within_ceiling_overlap_with_engine': n_matcher_within_not_completed,
        },
        'engine_within_ceiling': {
            'sample_size': n_engine_wc_total,
            'solved_in_sample': n_engine_wc_solved,
            'sample_solve_rate': n_engine_wc_solved / n_engine_wc_total,
            'pair_level_estimated_solved': estimated_engine_pair_level_solved,
            'pair_level_fraction_of_gt_true': estimated_engine_pair_level_solved / n_gt_true,
            'within_ceiling_residual_total_gt_true': n_within_ceiling_residual_pairs,
        },
        'completion': {
            'total_true_correct': completion_true_total,
            'ordered_kb_true_correct': n_ord_true,
            'plain_only_kb_true_correct': n_plain_only_true,
            'fraction_of_gt_true': completion_true_total / n_gt_true,
        },
        'combined_pair_estimate': {
            'sum': combined_pair_estimate,
            'fraction_of_gt_true': combined_pair_estimate / n_gt_true,
            'structural_ceiling': structural_ceiling,
            'structural_ceiling_fraction_of_gt_true': structural_ceiling / n_gt_true,
            'fraction_of_structural_ceiling': combined_pair_estimate / structural_ceiling,
        },
        'per_e1_stratification': {
            'n_e1_within_ceiling': len(per_e1),
            'n_e1_solved_at_least_one': n_e1_solved_any,
            'n_e1_solved_zero': n_e1_solved_zero,
            'solve_rate_median': float(per_e1['solve_rate'].median()),
            'solve_rate_mean': float(per_e1['solve_rate'].mean()),
            'solve_rate_p25': float(per_e1['solve_rate'].quantile(0.25)),
            'solve_rate_p75': float(per_e1['solve_rate'].quantile(0.75)),
            'corr_row_density_solve_rate': (
                float(per_e1[['e1_true_row_density', 'solve_rate']].corr().iloc[0, 1])
                if len(per_e1) > 1 else None
            ),
        },
        'future_work': [
            'Cross-cut by proof-survey category (substitution/rewrite-chain/Vampire/Egg) — '
            'requires joining the per-equation category labels from the proof survey. '
            'Skipped here; can be done in Phase 6 if useful.',
        ],
    }

    def _clean(o):
        if isinstance(o, dict): return {k: _clean(v) for k, v in o.items()}
        if isinstance(o, list): return [_clean(v) for v in o]
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o

    os.makedirs(os.path.dirname(_abs(args.output)), exist_ok=True)
    with open(_abs(args.output), 'w') as f:
        json.dump(_clean(summary), f, indent=2)
    print(f'\nwrote {_abs(args.output)}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
