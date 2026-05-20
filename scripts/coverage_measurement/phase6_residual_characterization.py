"""Phase 6 of `coverage_measurement_plan.md` — residual characterization.

Cuts the residual along three dimensions:

  1. terminate_reason breakdown by depth (already in Phase 3 output;
     re-summarized here as the depth-coverage curve).
  2. Per-E1 stratification: row density vs solve rate, distribution
     of solves per E1.
  3. Proof-survey category cross-cut: for the subset of sample pairs
     that appear in `true_corpus.parquet` (directly-proved pairs with
     a category label), the engine's solve rate per category. The
     plan's expectation: Vampire-superposition pairs are exactly where
     first-order-only rewriting bites hardest.

Reads:
  data/processed/coverage_measurement/sample.parquet
  data/processed/coverage_measurement/engine_results_within_ceiling_d{4,6,8}.parquet
  data/processed/coverage_measurement/engine_results_variable_disjoint_d6.parquet
  data/processed/proof_corpus/true_corpus.parquet

Writes:
  outputs/analyses/coverage_measurement/phase6_residual_characterization.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

sys.setrecursionlimit(20000)

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--output',
                    default='outputs/analyses/coverage_measurement/phase6_residual_characterization.json')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    if os.path.exists(_abs(args.output)) and not args.force:
        print(f'ERROR: {_abs(args.output)} exists; pass --force.', file=sys.stderr)
        return 2

    print('loading sample + engine results ...')
    sample = pq.read_table(_abs('data/processed/coverage_measurement/sample.parquet')).to_pandas()
    er_wc = {}
    for d in [4, 6, 8]:
        er_wc[d] = pq.read_table(_abs(
            f'data/processed/coverage_measurement/engine_results_within_ceiling_d{d}.parquet'
        )).to_pandas()

    print('loading proof-survey categories from true_corpus ...')
    tc = pq.read_table(_abs('data/processed/proof_corpus/true_corpus.parquet'),
                       columns=['e1', 'e2', 'category']).to_pandas()
    tc['e1'] = tc['e1'].astype(int)
    tc['e2'] = tc['e2'].astype(int)
    print(f'  true_corpus: {len(tc):,} categorized pairs')

    sample_wc = sample[~sample['e1_is_variable_disjoint']].copy()
    sample_wc['e1'] = sample_wc['e1'].astype(int)
    sample_wc['e2'] = sample_wc['e2'].astype(int)

    summary = {}

    # ============================================================
    # (1) Depth-coverage curve
    # ============================================================
    print()
    print('--- (1) Depth-coverage curve (within-ceiling sample) ---')
    curve = {}
    for d in [4, 6, 8]:
        df = er_wc[d]
        n_total = len(df)
        n_solved = int(df['solved'].sum())
        reasons = Counter(df['terminate_reason'])
        curve[d] = {
            'n_solved': n_solved,
            'fraction_solved': n_solved / n_total,
            'terminate_reasons': {k: int(v) for k, v in reasons.items()},
        }
        print(f'  d_max={d}: solved {n_solved:,} '
              f'({100*n_solved/n_total:.2f}%) | '
              f'frontier_died {reasons["frontier_died"]:,} '
              f'depth_exhausted {reasons.get("depth_exhausted", 0):,}')
    summary['depth_coverage_curve'] = curve

    # ============================================================
    # (2) Per-E1 stratification (using d=8)
    # ============================================================
    print()
    print('--- (2) Per-E1 stratification (within-ceiling at d=8) ---')
    df8 = er_wc[8]
    df8_keyed = df8.set_index(['e1', 'e2'])
    sample_wc['solved'] = [bool(df8_keyed.loc[(int(r['e1']), int(r['e2']))]['solved'])
                            if (int(r['e1']), int(r['e2'])) in df8_keyed.index else False
                           for _, r in sample_wc.iterrows()]
    per_e1 = sample_wc.groupby('e1').agg(
        row_density=('e1_true_row_density', 'first'),
        n_sampled=('e2', 'count'),
        n_solved=('solved', 'sum'),
    ).reset_index()
    per_e1['solve_rate'] = per_e1['n_solved'] / per_e1['n_sampled']

    # Row-density bucketing
    buckets = [
        ('1-10',     1,    10),
        ('11-50',    11,   50),
        ('51-200',   51,   200),
        ('201-1000', 201,  1000),
        ('1001-4693', 1001, 4694),
    ]
    print(f'  per-E1 solve rate by row-density bucket:')
    bucket_stats = {}
    for name, lo, hi in buckets:
        b = per_e1[(per_e1['row_density'] >= lo) & (per_e1['row_density'] <= hi)]
        if len(b) == 0:
            continue
        total_sampled = int(b['n_sampled'].sum())
        total_solved = int(b['n_solved'].sum())
        wsolved = float((b['solve_rate'] * b['row_density']).sum())
        wdensity = float(b['row_density'].sum())
        bucket_stats[name] = {
            'n_e1': len(b),
            'mean_solve_rate': float(b['solve_rate'].mean()),
            'median_solve_rate': float(b['solve_rate'].median()),
            'n_sampled': total_sampled,
            'n_solved': total_solved,
            'sample_solve_rate': total_solved / total_sampled if total_sampled else 0,
            'pair_level_extrapolated_solved': wsolved,
            'pair_level_in_bucket': wdensity,
            'pair_level_solve_rate': wsolved / wdensity if wdensity else 0,
        }
        print(f'    {name:>10}: n_e1={len(b):>4}, mean_rate={b["solve_rate"].mean():.3f}, '
              f'pair-level estimate={wsolved:.0f} / {wdensity:.0f} = '
              f'{100*wsolved/max(1,wdensity):.2f}%')
    summary['per_e1_by_row_density'] = bucket_stats

    # Correlations + binary
    corr = float(per_e1[['row_density', 'solve_rate']].corr().iloc[0, 1])
    n_zero = int((per_e1['n_solved'] == 0).sum())
    print(f'  corr(row_density, solve_rate): {corr:.3f}')
    print(f'  E1s with 0 solves: {n_zero}/{len(per_e1)}')

    summary['per_e1_overall'] = {
        'n_e1': len(per_e1),
        'corr_row_density_solve_rate': corr,
        'mean_solve_rate': float(per_e1['solve_rate'].mean()),
        'median_solve_rate': float(per_e1['solve_rate'].median()),
        'n_e1_zero_solves': n_zero,
        'n_e1_with_solves': int(len(per_e1) - n_zero),
    }

    # ============================================================
    # (3) Proof-survey category cross-cut
    # ============================================================
    print()
    print('--- (3) Proof-survey category cross-cut ---')
    # Join sample_wc to true_corpus on (e1, e2)
    sample_wc_cat = sample_wc.merge(tc, on=['e1', 'e2'], how='left')
    sample_wc_cat['category'] = sample_wc_cat['category'].fillna('uncategorized_transitive')

    cat_dist = sample_wc_cat['category'].value_counts()
    print(f'  category distribution in within-ceiling sample:')
    for cat, n in cat_dist.items():
        print(f'    {cat:>30}: {n:>5,}')

    cat_stats = {}
    print(f'  engine solve rate by category (within-ceiling):')
    for cat in cat_dist.index:
        b = sample_wc_cat[sample_wc_cat['category'] == cat]
        n_total = len(b)
        n_solved = int(b['solved'].sum())
        cat_stats[cat] = {
            'n_total': n_total,
            'n_solved': n_solved,
            'solve_rate': n_solved / n_total if n_total else 0,
        }
        print(f'    {cat:>30}: {n_solved:>5,}/{n_total:>5,} '
              f'({100*n_solved/max(1, n_total):.2f}%)')
    summary['category_cross_cut'] = cat_stats

    # ============================================================
    # (4) Variable-disjoint stratum (already 100% structurally impossible)
    # ============================================================
    print()
    print('--- (4) Variable-disjoint stratum ---')
    er_vd = pq.read_table(_abs(
        'data/processed/coverage_measurement/engine_results_variable_disjoint_d6.parquet'
    )).to_pandas()
    summary['variable_disjoint_stratum'] = {
        'n_pairs': len(er_vd),
        'n_solved': int(er_vd['solved'].sum()),
        'n_no_valid_rule_from_e1': int((er_vd['terminate_reason'] == 'no_valid_rule_from_e1').sum()),
        'fraction_short_circuited': float((er_vd['terminate_reason'] == 'no_valid_rule_from_e1').mean()),
    }
    print(f'  100% no_valid_rule_from_e1, 0% solved — structurally impossible by rewriting.')

    # ============================================================
    # Save
    # ============================================================
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
