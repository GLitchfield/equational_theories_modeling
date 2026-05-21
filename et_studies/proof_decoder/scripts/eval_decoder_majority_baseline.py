"""Phase 1.4 majority baseline for the proof decoder.

Per the plan: "for each E1 arity, predict the most common substitution
shape from the training set." Computes σ-exact-match for that fixed
prediction on each test split, per E1 arity, plus overall. The
decoder must beat this clearly for the research claim ("the embedding
encodes the witness") to hold against dataset-regularity alternatives.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--dataset',
                        default='et_studies/proof_decoder/data/decoder_phase1_dataset.parquet')
    parser.add_argument('--matcher-proofs',
                        default='et_studies/matcher_deliverable/data/matcher_proofs.parquet')
    parser.add_argument('--output',
                        default='embeddings/outputs/runs/proof_decoder_majority/eval_summary.json')
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    print(f'loading {args.dataset}')
    df = pq.read_table(_abs(args.dataset)).to_pandas()
    matcher = pq.read_table(
        _abs(args.matcher_proofs),
        columns=['eq1_id', 'eq2_id'] + [f'substitution_{i}' for i in range(6)],
    ).to_pandas()
    assert (df['eq1_id'].values[:5] == matcher['eq1_id'].values[:5]).all()
    subst = np.stack([matcher[f'substitution_{i}'].to_numpy() for i in range(6)],
                     axis=1).astype(np.int8)
    orient = df['orientation'].to_numpy().astype(np.int8)
    e1_arity = df['e1_arity'].to_numpy().astype(np.int8)
    split = df['split'].to_numpy()

    # --- 1. Learn the majority (orientation, σ) per arity from train ---
    train_mask = split == 'train'
    per_arity_counters: dict[int, Counter] = {}
    for i in np.where(train_mask)[0]:
        k = int(e1_arity[i])
        key = (int(orient[i]), tuple(int(x) for x in subst[i, :k]))
        per_arity_counters.setdefault(k, Counter())[key] += 1

    majority: dict[int, tuple[int, tuple[int, ...]]] = {}
    for k, ctr in per_arity_counters.items():
        majority[k] = ctr.most_common(1)[0][0]
    print('Majority (orientation, σ) per E1 arity (from train):')
    for k in sorted(majority):
        ori, sigma = majority[k]
        n = per_arity_counters[k][majority[k]]
        total = sum(per_arity_counters[k].values())
        print(f'  arity {k}: orient={ori} σ={sigma}  '
              f'covers {n:,} / {total:,} train pairs '
              f'({100*n/total:.1f}%)')

    # --- 2. Score against each split ---
    def score_split(split_name: str) -> dict:
        idx = np.where(split == split_name)[0]
        n = len(idx)
        per_arity: dict[int, dict[str, int]] = {}
        n_correct = 0
        for i in idx:
            k = int(e1_arity[i])
            true_ori = int(orient[i])
            true_sigma = tuple(int(x) for x in subst[i, :k])
            bucket = per_arity.setdefault(k, {'n': 0, 'correct': 0})
            bucket['n'] += 1
            if k in majority:
                pred_ori, pred_sigma = majority[k]
                if pred_ori == true_ori and pred_sigma == true_sigma:
                    n_correct += 1
                    bucket['correct'] += 1
        return {
            'n': int(n),
            'n_correct': int(n_correct),
            'sigma_rate': n_correct / max(n, 1),
            'per_arity': {
                k: {**v, 'sigma_rate': v['correct'] / max(v['n'], 1)}
                for k, v in sorted(per_arity.items())
            },
        }

    splits_result = {
        'train': score_split('train'),
        'test_one_novel': score_split('test_one_novel'),
        'test_both_novel': score_split('test_both_novel'),
    }

    summary = {
        'majority_per_arity': {
            int(k): {
                'orientation': int(v[0]),
                'substitution': list(v[1]),
                'train_coverage': per_arity_counters[k][v] / sum(per_arity_counters[k].values()),
                'train_count': per_arity_counters[k][v],
                'train_total_this_arity': sum(per_arity_counters[k].values()),
            }
            for k, v in sorted(majority.items())
        },
        'splits': splits_result,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_path}')

    print()
    print('=== Majority baseline σ-exact-match ===')
    for split_name, m in splits_result.items():
        print(f'  {split_name:<18}: {m["n_correct"]:>6,} / {m["n"]:>6,} = {m["sigma_rate"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
