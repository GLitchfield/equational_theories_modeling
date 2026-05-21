"""Build holdout split files from the implication pair table.

Currently builds Level 1 only (random pair holdout). Level 2 (equation
holdout) and Level 3 (equivalence-class holdout) will be added when those
levels are needed.

Reads:  data/processed/implication_pairs.parquet
Writes: data/splits/level1_holdout_seed{seed}.parquet

The output parquet has a single column ``pair_idx: int32`` containing the
row indices into ``implication_pairs.parquet`` that are held out from
training. Schema metadata captures the seed, ratio, source row counts,
and split type so a stale split (built against a different pair table
revision) can be detected at load time.

Unknowns (``value == 2``, the 1,062 Vampire timeouts) are excluded from
both training and holdout — they have no ground truth and can't be
scored. The training set is therefore::

    {row_idx for row_idx in range(num_rows)
     if pair_table.value[row_idx] != 2 and row_idx not in holdout}

Usage:
  python embeddings/scripts/build_splits.py
  python embeddings/scripts/build_splits.py --seed 7 --holdout-ratio 0.1
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.splits import random_pair_holdout  # noqa: E402


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def build_level1(
    pair_table_path: str,
    output_path: str,
    holdout_ratio: float,
    seed: int,
) -> dict:
    if not os.path.exists(pair_table_path):
        raise FileNotFoundError(f'pair table not found: {pair_table_path}')

    print(f'reading {pair_table_path}')
    t0 = time.perf_counter()
    table = pq.read_table(pair_table_path, columns=['value'])
    val = table['value'].to_numpy()
    print(f'  loaded {table.num_rows:,} rows in {time.perf_counter() - t0:.2f}s')

    labeled_mask = val != 2
    labeled_indices = np.where(labeled_mask)[0].astype(np.int32, copy=False)
    n_unknown = int((~labeled_mask).sum())
    print(f'  labeled: {len(labeled_indices):,}  unknown: {n_unknown:,}')

    print(f'sampling holdout (ratio={holdout_ratio}, seed={seed})')
    holdout = random_pair_holdout(labeled_indices, holdout_ratio=holdout_ratio, seed=seed)
    print(f'  holdout size: {len(holdout):,}')
    print(f'  train size:   {len(labeled_indices) - len(holdout):,}')

    metadata = {
        'split_type': 'level1_random_pair',
        'seed': str(seed),
        'holdout_ratio': str(holdout_ratio),
        'source_file': os.path.relpath(pair_table_path, _REPO_ROOT),
        'source_total_rows': str(table.num_rows),
        'source_labeled_rows': str(len(labeled_indices)),
        'num_holdout': str(len(holdout)),
        'num_train': str(len(labeled_indices) - len(holdout)),
    }
    schema = pa.schema(
        [('pair_idx', pa.int32())],
        metadata={k.encode(): v.encode() for k, v in metadata.items()},
    )
    out_table = pa.Table.from_arrays([pa.array(holdout)], schema=schema)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pq.write_table(out_table, output_path, compression='zstd')
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'wrote {size_mb:.2f} MB to {output_path}')
    return metadata


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Build holdout split files for the implication pair table.',
    )
    parser.add_argument(
        '--pair-table',
        default='data/processed/implication_pairs.parquet',
        help='input pair table (default: %(default)s)',
    )
    parser.add_argument(
        '--output-dir',
        default='data/splits',
        help='output directory for split files (default: %(default)s)',
    )
    parser.add_argument('--seed', type=int, default=42, help='RNG seed (default: %(default)s)')
    parser.add_argument(
        '--holdout-ratio',
        type=float,
        default=0.1,
        help='fraction of labeled pairs to hold out (default: %(default)s)',
    )
    args = parser.parse_args(argv)

    output_path = _abs(os.path.join(args.output_dir, f'level1_holdout_seed{args.seed}.parquet'))
    build_level1(
        pair_table_path=_abs(args.pair_table),
        output_path=output_path,
        holdout_ratio=args.holdout_ratio,
        seed=args.seed,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
