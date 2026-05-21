"""Extract and save the per-equation embedding matrix from a trained
JEPA checkpoint.

Backs Step 7 (UMAP) and Step 8 (linear probing) of
testing_plan_phase2.md. Runs the context encoder over every equation in
the parse-tree cache and writes a ``[N, embed_dim]`` matrix plus an
``equation_id`` ordering file.

Usage:
  python scripts/save_embeddings.py embeddings/outputs/checkpoints/<name>/epoch_NNN.pt
  python scripts/save_embeddings.py <ckpt> --output-dir embeddings/outputs/failure_analysis

Outputs (in --output-dir):
  equation_embeddings.npy          [N, embed_dim] float32
  equation_embeddings_ids.npy      [N] int32 — eq_id at row i

Both arrays share row order, so ``embeddings[i]`` is the embedding of
``ids[i]``. The ordering is sorted-ascending by equation_id (1..4694)
so a downstream script can index directly without consulting the IDs
file in the common case.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np
import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.dataset import load_graph_cache  # noqa: E402
from embeddings.src.model.encoder import EquationEncoder  # noqa: E402


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to a JEPA checkpoint .pt')
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for the embedding files. Defaults to '
             '``embeddings/outputs/runs/<wandb.name from checkpoint>/``.',
    )
    parser.add_argument(
        '--batch-size', type=int, default=1024,
        help='Per-pass equation batch size (default: %(default)s)',
    )
    parser.add_argument(
        '--no-normalize', action='store_true',
        help='Save the *pre*-L2-normalization encoder output instead of '
             'the unit-sphere version. Diagnostic mode — the contrastive '
             'loss only ever sees the normalized output, so analyses '
             'comparing distances should keep this off.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite existing output files',
    )
    args = parser.parse_args(argv)

    if args.output_dir is None:
        ckpt_peek = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        name_hint = (
            (ckpt_peek.get('config', {}).get('wandb', {}) or {}).get('name')
            or os.path.splitext(os.path.basename(args.checkpoint))[0]
        )
        del ckpt_peek
        args.output_dir = os.path.join('embeddings/outputs/runs', name_hint)

    out_dir = _abs(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_emb = os.path.join(out_dir, 'equation_embeddings.npy')
    out_ids = os.path.join(out_dir, 'equation_embeddings_ids.npy')
    if (os.path.exists(out_emb) or os.path.exists(out_ids)) and not args.force:
        print(f'ERROR: output files exist in {out_dir}. Pass --force.', file=sys.stderr)
        return 2

    print(f'loading {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    cfg = ckpt['config']
    name = (cfg.get('wandb', {}) or {}).get('name') \
        or os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"  run name: {name}")
    print(f"  epoch   : {ckpt['epoch']}")

    enc = EquationEncoder(
        node_features=cfg['encoder']['node_features'],
        edge_features=cfg['encoder']['edge_features'],
        hidden_dim=cfg['encoder']['hidden_dim'],
        embed_dim=cfg['encoder']['embed_dim'],
        num_layers=cfg['encoder']['num_layers'],
        num_heads=cfg['encoder']['num_heads'],
        dropout=cfg['encoder']['dropout'],
        skip_pooled=cfg['encoder'].get('skip_pooled', False),
        proj_dim=cfg['encoder'].get('proj_dim', None),
    )
    # Pull just the context encoder's weights out of the JEPA state dict.
    state = {
        k[len('context_encoder.'):]: v
        for k, v in ckpt['jepa_state_dict'].items()
        if k.startswith('context_encoder.')
    }
    enc.load_state_dict(state)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    enc = enc.to(device).eval()

    cache_path = _abs(cfg['data']['graph_cache_path'])
    print(f'reading graph cache {cache_path}')
    graphs = load_graph_cache(cache_path)
    eq_ids = sorted(graphs.keys())
    print(f'  {len(eq_ids):,} equations')

    print('encoding ...')
    t0 = time.perf_counter()
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(eq_ids), args.batch_size):
            batch = Batch.from_data_list(
                [graphs[i] for i in eq_ids[s:s + args.batch_size]]
            ).to(device)
            chunks.append(
                enc(batch, normalize=not args.no_normalize).cpu().numpy()
            )
    Z = np.concatenate(chunks, axis=0).astype(np.float32, copy=False)
    print(f'  {Z.shape} in {time.perf_counter() - t0:.1f}s')

    np.save(out_emb, Z)
    np.save(out_ids, np.array(eq_ids, dtype=np.int32))
    size_mb = (os.path.getsize(out_emb) + os.path.getsize(out_ids)) / 1024 / 1024
    print(f'wrote {size_mb:.1f} MB to {out_dir}/')
    return 0


if __name__ == '__main__':
    sys.exit(main())
