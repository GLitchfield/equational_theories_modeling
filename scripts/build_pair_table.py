"""Build a parquet pair table from the Vampire-labelled JSON.

Reads:  data/raw/etp/2025-08-11-vampire.json.gz
Writes: data/processed/implication_pairs.parquet

Schema:
  eq1_id : int32   1-indexed equation ID (line number in equations.txt)
  eq2_id : int32   1-indexed equation ID
  value  : int8    0 = FALSE, 1 = TRUE, 2 = unknown (Vampire timed out)

Rows are sorted by (eq1_id, eq2_id) for deterministic, scan-friendly output.

The JSON file is ~88 MB compressed and ~700 MB decompressed; ``json.load``
takes ~30 s on a modern CPU and uses several GB of RAM transiently. The
output parquet (zstd-compressed) is ~70 MB.

Usage:
  python scripts/build_pair_table.py
  python scripts/build_pair_table.py --input <path> --output <path>
"""

from __future__ import annotations

import argparse
import gzip
import json
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
        raise FileNotFoundError(f'vampire JSON not found: {input_path}')

    print(f'reading {input_path}')
    t0 = time.perf_counter()
    with gzip.open(input_path, 'rt') as f:
        raw = json.load(f)
    print(f'  json load: {time.perf_counter() - t0:.1f}s')

    if 'values' not in raw:
        raise ValueError(f"unexpected JSON shape; expected top-level 'values' key, got {list(raw)[:5]}")
    entries = raw['values']
    n = len(entries)
    print(f'  entries: {n:,}')

    t1 = time.perf_counter()
    eq1 = np.empty(n, dtype=np.int32)
    eq2 = np.empty(n, dtype=np.int32)
    val = np.empty(n, dtype=np.int8)
    for idx, (key, payload) in enumerate(entries.items()):
        i_str, j_str = key.split('_', 1)
        eq1[idx] = int(i_str)
        eq2[idx] = int(j_str)
        val[idx] = payload['value']
    print(f'  ndarray fill: {time.perf_counter() - t1:.1f}s')

    # Free the giant python dict before sorting + writing.
    del entries
    del raw

    t2 = time.perf_counter()
    # Composite key: (eq1, eq2) lex order. argsort with stable sort, multi-key trick:
    # sort by eq2 first, then by eq1 stably to get (eq1, eq2) lex order.
    order = np.argsort(eq2, kind='stable')
    eq1, eq2, val = eq1[order], eq2[order], val[order]
    order = np.argsort(eq1, kind='stable')
    eq1, eq2, val = eq1[order], eq2[order], val[order]
    print(f'  sort: {time.perf_counter() - t2:.1f}s')

    n_false = int((val == 0).sum())
    n_true = int((val == 1).sum())
    n_unknown = int((val == 2).sum())
    eq_min = int(eq1.min()) if n else 0
    eq_max = int(eq1.max()) if n else 0
    print(f'  distribution: FALSE={n_false:,}  TRUE={n_true:,}  unknown={n_unknown:,}')
    print(f'  eq id range: {eq_min}..{eq_max}')

    table = pa.table({
        'eq1_id': pa.array(eq1, type=pa.int32()),
        'eq2_id': pa.array(eq2, type=pa.int32()),
        'value': pa.array(val, type=pa.int8()),
    })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    t3 = time.perf_counter()
    pq.write_table(table, output_path, compression=compression)
    print(f'  parquet write: {time.perf_counter() - t3:.1f}s')

    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'wrote {size_mb:.1f} MB to {output_path}')

    return {
        'num_rows': n,
        'num_false': n_false,
        'num_true': n_true,
        'num_unknown': n_unknown,
        'eq_id_min': eq_min,
        'eq_id_max': eq_max,
        'output_size_mb': size_mb,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Convert the Vampire-labelled JSON to a parquet pair table.',
    )
    parser.add_argument(
        '--input',
        default='data/raw/etp/2025-08-11-vampire.json.gz',
        help='input gzipped JSON (default: %(default)s)',
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
