"""Build a parquet pair table from the ETP raw-implications CSV.

Reads:  data/raw/etp_export/export_raw_implications_5_9_2026.csv
Writes: data/processed/implication_pairs.parquet

The CSV is a 4694×4694 dense int8 matrix where ``M[i, j]`` encodes the
relationship between equation ``i+1`` and equation ``j+1`` (1-indexed
into ``equations.txt``). Codes::

   3   TRUE   — implication holds (regular proof)
   4   TRUE   — implication holds (special / structural)  ~10K entries
  -3   FALSE  — implication fails (regular counterexample)
  -4   FALSE  — implication fails (special / structural)  ~590K entries

The Austin-implications file is a curated subset of the FALSE pairs;
"Austin" pairs split across both ``-3`` and ``-4`` codes.

The CSV is the **authoritative ground truth** — it's derived from
Lean-verified results in the Equational Theories Project rather than
from Vampire's proof search. Predecessor pair tables built from
``2025-08-11-vampire.json.gz`` left 1,062 pairs as ``value=2``
("Vampire timeout"); those are now resolved to FALSE under this CSV.

Output schema (off-diagonal only — diagonal pairs eq=eq are excluded):

  eq1_id : int32  1-indexed equation ID
  eq2_id : int32  1-indexed equation ID
  value  : int8   0 = FALSE, 1 = TRUE  (binary label for training)
  code   : int8   -4 / -3 / 3 / 4      (full CSV code, for diagnostics)

Rows are sorted by (eq1_id, eq2_id) for deterministic, scan-friendly
output.

Usage:
  python scripts/build_pair_table.py
  python scripts/build_pair_table.py --input <path> --output <path>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def build_pair_table(input_path: str, output_path: str, compression: str = 'zstd') -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'CSV not found: {input_path}')

    print(f'reading {input_path}')
    t0 = time.perf_counter()
    M = np.loadtxt(input_path, delimiter=',', dtype=np.int8)
    print(f'  loaded {M.shape} matrix in {time.perf_counter() - t0:.1f}s')

    if M.shape[0] != M.shape[1]:
        raise ValueError(f'expected square matrix; got {M.shape}')
    N = M.shape[0]

    # Validate codes — anything outside {-4, -3, 3, 4} is unexpected.
    unique = set(np.unique(M).tolist())
    expected = {-4, -3, 3, 4}
    extra = unique - expected
    if extra:
        raise ValueError(f'unexpected codes in CSV: {sorted(extra)}')

    # Validate diagonal — every Eq → Eq is reflexively TRUE.
    diag = M.diagonal()
    if not np.all(diag > 0):
        n_bad = int(np.sum(diag <= 0))
        raise ValueError(f'expected positive code on every diagonal entry; got {n_bad} non-positive')

    t1 = time.perf_counter()
    # Build off-diagonal pair list. Vectorize: enumerate i,j ≠ each other.
    rows, cols = np.meshgrid(np.arange(N), np.arange(N), indexing='ij')
    mask = rows != cols
    eq1 = (rows[mask] + 1).astype(np.int32, copy=False)   # 1-indexed
    eq2 = (cols[mask] + 1).astype(np.int32, copy=False)
    code = M[mask].astype(np.int8, copy=False)
    value = (code > 0).astype(np.int8, copy=False)        # 1 if TRUE, else 0
    print(f'  flattened {len(eq1):,} off-diagonal entries in {time.perf_counter() - t1:.1f}s')

    # Already in (eq1, eq2) lex order from the meshgrid construction;
    # no explicit sort needed. Sanity-check a few rows.
    assert eq1[0] == 1 and eq2[0] == 2 and eq1[-1] == N and eq2[-1] == N - 1

    n_false = int((value == 0).sum())
    n_true = int((value == 1).sum())
    n_neg3 = int((code == -3).sum())
    n_neg4 = int((code == -4).sum())
    n_pos3 = int((code == 3).sum())
    n_pos4 = int((code == 4).sum())
    print(f'  values:  TRUE={n_true:,}  FALSE={n_false:,}')
    print(f'  codes :  +3={n_pos3:,}  +4={n_pos4:,}  -3={n_neg3:,}  -4={n_neg4:,}')
    print(f'  eq id range: 1..{N}')

    table = pa.table({
        'eq1_id': pa.array(eq1, type=pa.int32()),
        'eq2_id': pa.array(eq2, type=pa.int32()),
        'value': pa.array(value, type=pa.int8()),
        'code': pa.array(code, type=pa.int8()),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t2 = time.perf_counter()
    pq.write_table(table, output_path, compression=compression)
    print(f'  parquet write: {time.perf_counter() - t2:.1f}s')

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'wrote {size_mb:.1f} MB to {output_path}')

    return {
        'num_rows': len(eq1),
        'num_false': n_false,
        'num_true': n_true,
        'num_unknown': 0,
        'num_neg3': n_neg3,
        'num_neg4': n_neg4,
        'num_pos3': n_pos3,
        'num_pos4': n_pos4,
        'eq_id_min': 1,
        'eq_id_max': N,
        'output_size_mb': size_mb,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert the ETP raw-implications CSV to a parquet pair table.',
    )
    parser.add_argument(
        '--input',
        default='data/raw/etp_export/export_raw_implications_5_9_2026.csv',
        help='input CSV (4694×4694 int8 matrix; default: %(default)s)',
    )
    parser.add_argument(
        '--output',
        default='data/processed/implication_pairs.parquet',
        help='output parquet path (default: %(default)s)',
    )
    parser.add_argument(
        '--compression',
        default='zstd',
        choices=['zstd', 'snappy', 'gzip', 'none'],
        help='parquet compression codec (default: %(default)s)',
    )
    args = parser.parse_args(argv)

    compression = None if args.compression == 'none' else args.compression
    build_pair_table(_abs(args.input), _abs(args.output), compression=compression)
    return 0


if __name__ == '__main__':
    sys.exit(main())
