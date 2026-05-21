"""Phase 1 of `rewrite_chain_study_plan.md` — extract the rewrite-chain
proof bodies into a single dataset.

Filters `et_studies/proof_survey/data/normalized_proofs.parquet` (the
survey's normalized bodies) by joining against
`et_studies/proof_survey/data/true_corpus.parquet`'s
`c_rewrite_chain` category mask on `(e1, e2)` keys. Verifies the
1,636 count against `outputs/analyses/proof_survey/true_characterization.json`.

Output: `et_studies/rewrite_chain_study/data/rewrite_chain_dataset.parquet`
with one row per c_rewrite_chain proof:

  e1, e2       int32          equation IDs (1-indexed)
  file         string         source .lean file (for traceability)
  body_raw     string         n1_whitespace (variables intact, equation
                              refs intact, only whitespace collapsed)
  body_n4      string         n4_plus_hyps (vars/eqrefs/hyps normalized
                              to v1.., E1.., h1.. — useful for vocab
                              analysis)

The raw body is the source-of-truth for tactic parsing (Phases 3-4)
and rule-set analysis. The n4 body is convenient for vocab work.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--bodies',
                        default='et_studies/proof_survey/data/normalized_proofs.parquet')
    parser.add_argument('--categories',
                        default='et_studies/proof_survey/data/true_corpus.parquet')
    parser.add_argument('--survey-summary',
                        default='outputs/analyses/proof_survey/true_characterization.json')
    parser.add_argument('--output',
                        default='et_studies/rewrite_chain_study/data/rewrite_chain_dataset.parquet')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    # --- 1. Verify expected count ---
    expected = json.load(open(_abs(args.survey_summary)))['category_distribution']['c_rewrite_chain']
    print(f'expected count from survey: {expected}')

    # --- 2. Load category mask ---
    cat = pq.read_table(_abs(args.categories), columns=['e1', 'e2', 'category']).to_pandas()
    rc_mask = cat['category'] == 'c_rewrite_chain'
    rc_keys = set(zip(cat[rc_mask]['e1'].tolist(), cat[rc_mask]['e2'].tolist()))
    print(f'  c_rewrite_chain pairs in categories table: {len(rc_keys)}')
    assert len(rc_keys) == expected, f'expected {expected}, got {len(rc_keys)}'

    # --- 3. Load bodies and filter ---
    print(f'loading bodies from {args.bodies}')
    bodies = pq.read_table(
        _abs(args.bodies), columns=['e1', 'e2', 'file', 'n1_whitespace', 'n4_plus_hyps'],
    ).to_pandas()
    print(f'  raw rows: {len(bodies)}')
    # The bodies parquet has 10 duplicate (e1, e2) pairs from Subgraph.lean
    # (likely the ETP defines the same implication twice for exposition).
    # Keep first occurrence per pair so the count matches the survey's
    # category distribution (which dedups via .value_counts() on category).
    bodies = bodies.drop_duplicates(subset=['e1', 'e2'], keep='first')
    print(f'  after dedup on (e1, e2): {len(bodies)}')
    bodies = bodies[
        [(e1, e2) in rc_keys for e1, e2 in zip(bodies['e1'], bodies['e2'])]
    ].reset_index(drop=True)
    print(f'  filtered to c_rewrite_chain: {len(bodies)}')

    # Sanity: count should match
    assert len(bodies) == expected, f'after filtering: {len(bodies)} != {expected}'

    # --- 4. Save ---
    out = bodies.rename(columns={'n1_whitespace': 'body_raw', 'n4_plus_hyps': 'body_n4'})
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out.to_parquet(out_path, compression='zstd', index=False)
    print(f'wrote {out_path} ({len(out):,} rows)')

    # --- 5. Quick spot-check ---
    print()
    print('Sample bodies (first 3):')
    for _, row in out.head(3).iterrows():
        print(f'  eq{row.e1} -> eq{row.e2}  ({row.file})')
        print(f'    raw: {row.body_raw[:140]}')
        print(f'    n4:  {row.body_n4[:140]}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
