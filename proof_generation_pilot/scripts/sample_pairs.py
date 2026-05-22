"""Build the Phase 1 sample: variable-disjoint TRUE pairs the matcher can solve.

Source pool: matcher_proofs.parquet (49,586 pairs the matcher proved) ∩
E1 classification 'incomparable' (variable-disjoint per Phase 0 definition).
The intersection is ~30,010 pairs.

Outputs two seeded uniform samples:
  - phase1_sample_100.parquet  (smoke / pipeline debug)
  - phase1_sample_1k.parquet   (Phase 1 production target)

Per-row columns:
  pair_id              local 0-based index into the sample
  eq1_id, eq2_id       equation IDs from ETP (preserved from matcher_proofs)
  orientation          σ orientation flag (from matcher_proofs)
  substitution_0..5    σ as a permutation tuple (from matcher_proofs)
  lean_proof_term      matcher's term-mode proof body (from matcher_proofs)
  lean_proof_tactic    matcher's tactic-mode proof body (from matcher_proofs)
  e1_sub_class         'disjoint' or 'overlapping' (per Phase 0 sub-classification)

Reads:
  et_studies/matcher_deliverable/data/matcher_proofs.parquet
  et_studies/completion_exploration/data/equation_orientability.parquet

Writes:
  proof_generation_pilot/data/processed/phase1_sample_100.parquet
  proof_generation_pilot/data/processed/phase1_sample_1k.parquet
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--matcher-proofs',
                    default='et_studies/matcher_deliverable/data/matcher_proofs.parquet')
    ap.add_argument('--orientability',
                    default='et_studies/completion_exploration/data/equation_orientability.parquet')
    ap.add_argument('--output-dir',
                    default='proof_generation_pilot/data/processed')
    ap.add_argument('--smoke-size', type=int, default=100)
    ap.add_argument('--prod-size', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=20260521)
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args(argv)

    out_100 = _abs(os.path.join(args.output_dir, 'phase1_sample_100.parquet'))
    out_1k = _abs(os.path.join(args.output_dir, 'phase1_sample_1k.parquet'))
    for p in [out_100, out_1k]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    print(f'loading matcher_proofs from {args.matcher_proofs} ...')
    m = pq.read_table(_abs(args.matcher_proofs)).to_pandas()
    print(f'  {len(m):,} matcher-solved pairs')

    print(f'loading equation_orientability from {args.orientability} ...')
    o = pq.read_table(_abs(args.orientability)).to_pandas()
    print(f'  {len(o):,} equations classified')

    var_disjoint = o[o['classification'] == 'incomparable'][['eq_id', 'sub_classification']]
    print(f'  variable-disjoint (incomparable) E1s: {len(var_disjoint):,}')

    pool = m.merge(
        var_disjoint.rename(columns={'eq_id': 'eq1_id', 'sub_classification': 'e1_sub_class'}),
        on='eq1_id', how='inner',
    )
    print(f'\nmatcher ∩ var-disjoint pool: {len(pool):,} pairs')
    print(f'  sub-class breakdown:')
    print(pool['e1_sub_class'].value_counts().to_string())

    if len(pool) < args.prod_size:
        print(f'\nERROR: pool ({len(pool):,}) smaller than prod-size ({args.prod_size}); '
              f'reduce --prod-size or check filters.', file=sys.stderr)
        return 3

    rng = np.random.default_rng(args.seed)
    # Single shared shuffle, then 100-row is a prefix of 1k-row — so the
    # smoke sample is a strict subset of production, useful for resumability.
    idx = rng.permutation(len(pool))
    shuffled = pool.iloc[idx].reset_index(drop=True)
    smoke = shuffled.iloc[:args.smoke_size].copy()
    prod = shuffled.iloc[:args.prod_size].copy()
    smoke['pair_id'] = range(len(smoke))
    prod['pair_id'] = range(len(prod))

    os.makedirs(os.path.dirname(out_100), exist_ok=True)
    # column order: pair_id first, then identifying fields, then matcher metadata
    cols = ['pair_id', 'eq1_id', 'eq2_id', 'orientation',
            'substitution_0', 'substitution_1', 'substitution_2',
            'substitution_3', 'substitution_4', 'substitution_5',
            'lean_proof_term', 'lean_proof_tactic', 'e1_sub_class']
    pq.write_table(pa.Table.from_pandas(smoke[cols], preserve_index=False),
                   out_100, compression='zstd')
    pq.write_table(pa.Table.from_pandas(prod[cols], preserve_index=False),
                   out_1k, compression='zstd')

    print(f'\nwrote {out_100}  ({len(smoke):,} rows)')
    print(f'wrote {out_1k}  ({len(prod):,} rows)')
    print(f'\n(seed={args.seed}; smoke is prefix of prod for resumability)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
