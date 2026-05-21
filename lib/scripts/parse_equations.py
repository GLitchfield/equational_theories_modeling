"""Parse all 4,694 ETP equations into PyG Data objects and cache to disk.

Reads:  data/raw/etp/equations.txt (line N is equation ID N, 1-indexed)
Writes: data/processed/equation_graphs.pt

The cache is a dict with keys:
  'graphs' : dict[int, torch_geometric.data.Data] keyed by equation ID
  'meta'   : dict capturing the encoding constants the cache was built with;
             downstream loaders should assert these match the current parser
             so stale caches are caught loudly rather than silently used.

Usage:
  python lib/scripts/parse_equations.py
  python lib/scripts/parse_equations.py --input <path> --output <path>
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# Make src/ importable when running as `python lib/scripts/parse_equations.py`.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from lib.data.parser import (  # noqa: E402
    EDGE_FEATURE_DIM,
    NODE_FEATURE_DIM,
    NUM_VAR_SLOTS,
    OPERATOR,
    equation_to_pyg,
)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def build_cache(input_path: str, output_path: str) -> dict:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f'equations file not found: {input_path}')

    with open(input_path) as f:
        lines = [line.strip() for line in f if line.strip()]
    print(f'read {len(lines)} equations from {input_path}')

    graphs: dict[int, object] = {}
    family_c_count = 0
    max_nodes = 0
    max_edges = 0
    t0 = time.perf_counter()
    for eq_id, line in enumerate(lines, start=1):
        try:
            data = equation_to_pyg(line)
        except Exception as e:
            raise RuntimeError(f'parsing line {eq_id} ({line!r}): {e}') from e
        graphs[eq_id] = data
        max_nodes = max(max_nodes, int(data.x.shape[0]))
        max_edges = max(max_edges, int(data.edge_index.shape[1]))
        # Family-C ↔ at least one edge has edge_attr summing to 0 (the root-link).
        if data.edge_attr.shape[0] > 0:
            if (data.edge_attr.sum(dim=1) == 0).any().item():
                family_c_count += 1
    elapsed = time.perf_counter() - t0
    print(f'parsed {len(graphs)} equations in {elapsed:.2f}s')
    print(f'  family-C: {family_c_count}')
    print(f'  max nodes: {max_nodes}, max edges: {max_edges}')

    cache = {
        'graphs': graphs,
        'meta': {
            'num_equations': len(graphs),
            'node_feature_dim': NODE_FEATURE_DIM,
            'edge_feature_dim': EDGE_FEATURE_DIM,
            'num_var_slots': NUM_VAR_SLOTS,
            'operator': OPERATOR,
            'family_c_count': family_c_count,
            'max_nodes': max_nodes,
            'max_edges': max_edges,
            'source_file': os.path.relpath(input_path, _REPO_ROOT),
        },
    }

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(cache, output_path)
    size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f'wrote {size_mb:.2f} MB to {output_path}')
    return cache


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Parse all ETP equations into a PyG Data cache.',
    )
    parser.add_argument(
        '--input',
        default='data/raw/etp/equations.txt',
        help='input equations file (default: %(default)s)',
    )
    parser.add_argument(
        '--output',
        default='data/processed/equation_graphs.pt',
        help='output cache path (default: %(default)s)',
    )
    args = parser.parse_args(argv)

    build_cache(_abs(args.input), _abs(args.output))
    return 0


if __name__ == '__main__':
    sys.exit(main())
