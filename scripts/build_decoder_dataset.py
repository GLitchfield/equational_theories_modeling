"""Phase 1.0 of `proof_decoder_plan.md` — decoder dataset preparation.

Reads the matcher's 49,586-row ground-truth proof dataset, normalizes
each proof body to v1..v6 space, tokenizes into the 16-token vocab,
and generates an equation-level train/test split (15% eq holdout,
seed 42) per Methodology Resolution 4.

Outputs:

  data/splits/decoder_eq_split_seed42.parquet
      committed; per-equation `(eq_id, is_held_out)`.

  data/processed/proof_corpus/decoder_phase1_dataset.parquet
      gitignored, regenerable. One row per matched pair:
        eq1_id, eq2_id          int32
        orientation             int8 (0=direct, 1=flipped)
        e1_arity, e2_arity      int8 (variable counts)
        normalized_proof        string  (e.g., 'exact fun v1 v2 => h v1 v2 v1')
        tokens                  list[int8] (BOS-prefixed, EOS-suffixed)
        split                   string  {'train', 'test_one_novel', 'test_both_novel'}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.parser import parse_equation  # noqa: E402
from src.decoder.tokenizer import build_normalized_proof, tokenize  # noqa: E402
from proof_generation_pilot.methods.matcher.matcher import ordered_vars  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def load_equations(path: str) -> list:
    with open(_abs(path)) as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    return [parse_equation(line) for line in lines]


def build(
    equations_path: str,
    matcher_proofs_path: str,
    split_output: str,
    dataset_output: str,
    holdout_fraction: float,
    seed: int,
) -> dict:
    # --- 1. equations + arities ---
    print(f'loading equations from {equations_path}')
    eqs = load_equations(equations_path)
    n_eq = len(eqs)
    arities = np.array([len(ordered_vars(e)) for e in eqs], dtype=np.int8)
    print(f'  {n_eq:,} equations, arities min={arities.min()} max={arities.max()} '
          f'(distribution {dict(zip(*np.unique(arities, return_counts=True)))})')

    # --- 2. equation-level holdout split ---
    rng = np.random.default_rng(seed)
    n_holdout = int(round(n_eq * holdout_fraction))
    perm = rng.permutation(n_eq)
    held_out_eq_ids = np.sort(perm[:n_holdout] + 1).astype(np.int32)   # 1-indexed
    is_held_out = np.zeros(n_eq, dtype=bool)
    is_held_out[perm[:n_holdout]] = True
    print(f'  held out: {n_holdout:,} / {n_eq:,} = {100*n_holdout/n_eq:.1f}% (seed={seed})')

    # --- 3. load matcher proofs ---
    print(f'loading matcher proofs from {matcher_proofs_path}')
    tbl = pq.read_table(_abs(matcher_proofs_path))
    eq1_arr = tbl['eq1_id'].to_numpy()
    eq2_arr = tbl['eq2_id'].to_numpy()
    orient_arr = tbl['orientation'].to_numpy().astype(np.int8)
    subst = np.column_stack([
        tbl[f'substitution_{i}'].to_numpy().astype(np.int8) for i in range(6)
    ])
    n_pairs = len(eq1_arr)
    print(f'  {n_pairs:,} matched pairs')

    # --- 4. build per-pair records ---
    e1_arity_arr = np.empty(n_pairs, dtype=np.int8)
    e2_arity_arr = np.empty(n_pairs, dtype=np.int8)
    normalized: list[str] = []
    tokens_list: list[list[int]] = []
    split_arr = np.empty(n_pairs, dtype=object)

    n_train = n_one_novel = n_both_novel = 0
    for i in range(n_pairs):
        e1_id = int(eq1_arr[i])
        e2_id = int(eq2_arr[i])
        e1_a = int(arities[e1_id - 1])
        e2_a = int(arities[e2_id - 1])
        e1_arity_arr[i] = e1_a
        e2_arity_arr[i] = e2_a
        sigma = [int(x) for x in subst[i, :e1_a]]
        proof = build_normalized_proof(
            e2_arity=e2_a, substitution=sigma, e1_arity=e1_a, orientation=int(orient_arr[i]),
        )
        normalized.append(proof)
        tokens_list.append(tokenize(proof))
        e1_held = is_held_out[e1_id - 1]
        e2_held = is_held_out[e2_id - 1]
        if not e1_held and not e2_held:
            split_arr[i] = 'train'
            n_train += 1
        elif e1_held and e2_held:
            split_arr[i] = 'test_both_novel'
            n_both_novel += 1
        else:
            split_arr[i] = 'test_one_novel'
            n_one_novel += 1
    print(f'  splits: train={n_train:,}  one_novel={n_one_novel:,}  both_novel={n_both_novel:,}')

    # --- 5. write outputs ---
    os.makedirs(os.path.dirname(_abs(split_output)), exist_ok=True)
    split_table = pa.table({
        'eq_id': pa.array(np.arange(1, n_eq + 1, dtype=np.int32), type=pa.int32()),
        'is_held_out': pa.array(is_held_out, type=pa.bool_()),
    })
    pq.write_table(split_table, _abs(split_output), compression='zstd')
    print(f'wrote {split_output}')

    os.makedirs(os.path.dirname(_abs(dataset_output)), exist_ok=True)
    dataset = pa.table({
        'eq1_id': pa.array(eq1_arr, type=pa.int32()),
        'eq2_id': pa.array(eq2_arr, type=pa.int32()),
        'orientation': pa.array(orient_arr, type=pa.int8()),
        'e1_arity': pa.array(e1_arity_arr, type=pa.int8()),
        'e2_arity': pa.array(e2_arity_arr, type=pa.int8()),
        'normalized_proof': pa.array(normalized, type=pa.string()),
        'tokens': pa.array(tokens_list, type=pa.list_(pa.int8())),
        'split': pa.array(split_arr.tolist(), type=pa.string()),
    })
    pq.write_table(dataset, _abs(dataset_output), compression='zstd')
    print(f'wrote {dataset_output} ({n_pairs:,} rows)')

    # --- 6. summary stats ---
    seq_lens = np.array([len(t) for t in tokens_list])
    summary = {
        'n_equations': int(n_eq),
        'n_held_out': int(n_holdout),
        'holdout_fraction_actual': float(n_holdout / n_eq),
        'seed': int(seed),
        'n_pairs_total': int(n_pairs),
        'n_pairs_train': int(n_train),
        'n_pairs_test_one_novel': int(n_one_novel),
        'n_pairs_test_both_novel': int(n_both_novel),
        'token_seq_len': {
            'min': int(seq_lens.min()),
            'max': int(seq_lens.max()),
            'mean': float(seq_lens.mean()),
            'median': float(np.median(seq_lens)),
            'p95': float(np.percentile(seq_lens, 95)),
            'p99': float(np.percentile(seq_lens, 99)),
        },
        'orientation_distribution': {
            'direct': int((orient_arr == 0).sum()),
            'flipped': int((orient_arr == 1).sum()),
        },
        'e1_arity_distribution': {
            int(k): int(v) for k, v in zip(*np.unique(e1_arity_arr, return_counts=True))
        },
        'e2_arity_distribution': {
            int(k): int(v) for k, v in zip(*np.unique(e2_arity_arr, return_counts=True))
        },
    }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--equations', default='data/raw/etp/equations.txt')
    parser.add_argument('--matcher-proofs',
                        default='data/processed/proof_corpus/matcher_proofs.parquet')
    parser.add_argument('--split-output',
                        default='data/splits/decoder_eq_split_seed42.parquet')
    parser.add_argument('--dataset-output',
                        default='data/processed/proof_corpus/decoder_phase1_dataset.parquet')
    parser.add_argument('--summary-output',
                        default='outputs/analyses/proof_decoder/dataset_summary.json')
    parser.add_argument('--holdout-fraction', type=float, default=0.15)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    for p in [args.split_output, args.dataset_output, args.summary_output]:
        if os.path.exists(_abs(p)) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    summary = build(
        equations_path=args.equations,
        matcher_proofs_path=args.matcher_proofs,
        split_output=args.split_output,
        dataset_output=args.dataset_output,
        holdout_fraction=args.holdout_fraction,
        seed=args.seed,
    )

    os.makedirs(os.path.dirname(_abs(args.summary_output)), exist_ok=True)
    with open(_abs(args.summary_output), 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'wrote {args.summary_output}')

    print()
    print('=== Dataset summary ===')
    print(f"  Equations: {summary['n_equations']:,} ({summary['n_held_out']:,} held out)")
    print(f"  Pairs    : {summary['n_pairs_total']:,}")
    print(f"    train          : {summary['n_pairs_train']:,}")
    print(f"    test_one_novel : {summary['n_pairs_test_one_novel']:,}")
    print(f"    test_both_novel: {summary['n_pairs_test_both_novel']:,}")
    sl = summary['token_seq_len']
    print(f"  Token seq len: median {sl['median']:.0f}, "
          f"mean {sl['mean']:.1f}, p95 {sl['p95']:.0f}, max {sl['max']}")
    od = summary['orientation_distribution']
    print(f"  Orientation: direct {od['direct']:,}, flipped {od['flipped']:,}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
