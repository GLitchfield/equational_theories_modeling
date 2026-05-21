"""Diagnostic: singular-value spectrum of a trained encoder's embedding matrix.

Loads a saved JEPA checkpoint, encodes every equation in the parse-tree
cache once with the (eval-mode) context encoder, stacks the resulting
unit-norm embeddings into a [N, embed_dim] matrix, and prints a summary
of the SVD spectrum plus saves a log-y plot of the sorted singular
values.

Use to decide whether dimensional collapse is happening in the embedding
space: if the top few singular values are large and the rest near zero,
the model is using only a low-rank subspace and regularizers like VICReg
should help. If the spectrum decays smoothly, the embedding space is
already well-utilized.

Usage:
  python scripts/embedding_spectrum.py embeddings/outputs/checkpoints/embed_dim_512/epoch_008.pt
  python scripts/embedding_spectrum.py <ckpt> --output-dir embeddings/outputs/figures

Outputs:
  - stdout: numerical summary (top-K singular values, cumulative variance,
    effective rank, participation ratio)
  - PNG: log-y plot of all sorted singular values, in <output-dir>
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.dataset import load_graph_cache  # noqa: E402
from embeddings.src.model.encoder import EquationEncoder  # noqa: E402


def _build_encoder(config: dict) -> EquationEncoder:
    enc_cfg = config['encoder']
    return EquationEncoder(
        node_features=enc_cfg['node_features'],
        edge_features=enc_cfg['edge_features'],
        hidden_dim=enc_cfg['hidden_dim'],
        embed_dim=enc_cfg['embed_dim'],
        num_layers=enc_cfg['num_layers'],
        num_heads=enc_cfg['num_heads'],
        dropout=enc_cfg['dropout'],
    )


def _extract_encoder_state(jepa_state: dict) -> dict:
    """Pull just the context encoder's parameters out of a JEPA state dict."""
    prefix = 'context_encoder.'
    return {
        k[len(prefix):]: v for k, v in jepa_state.items()
        if k.startswith(prefix)
    }


def compute_embeddings(
    encoder: EquationEncoder,
    graphs: dict[int, 'Data'],
    device: str,
    batch_size: int = 1024,
) -> tuple[np.ndarray, list[int]]:
    """Encode every graph in ``graphs`` once, return ``(N, embed_dim)`` array
    and the equation-id list aligned with rows."""
    encoder = encoder.to(device).eval()
    eq_ids = sorted(graphs.keys())
    embeddings = []
    with torch.no_grad():
        for start in range(0, len(eq_ids), batch_size):
            chunk_ids = eq_ids[start:start + batch_size]
            batch = Batch.from_data_list([graphs[i] for i in chunk_ids]).to(device)
            z = encoder(batch).cpu().numpy()
            embeddings.append(z)
    return np.concatenate(embeddings, axis=0), eq_ids


def summarize_spectrum(s: np.ndarray) -> dict:
    """Numerical summary of a singular-value array, sorted descending."""
    s = np.asarray(s)
    N = len(s)
    s2 = s ** 2
    total_var = float(s2.sum())

    cumvar = np.cumsum(s2) / total_var

    def k_for_var(v: float) -> int:
        return int(np.searchsorted(cumvar, v) + 1)

    # Effective rank (Roy & Vetterli 2007): exp(-sum(p_i log p_i))
    # where p_i = s_i^2 / sum(s_j^2). Resilient to scale.
    p = s2 / total_var
    eps = 1e-12
    eff_rank = float(np.exp(-(p * np.log(p + eps)).sum()))

    # Participation ratio: (sum s^2)^2 / (N * sum s^4) — different
    # functional form, also a "how many dims are effectively used" stat.
    participation = float(s2.sum() ** 2 / (s2 ** 2).sum() / N)

    return {
        'N': N,
        's_max': float(s[0]),
        's_min': float(s[-1]),
        'condition_number': float(s[0] / max(s[-1], 1e-30)),
        'total_variance': total_var,
        'top_1_variance_frac': float(s2[0] / total_var),
        'top_10_variance_frac': float(s2[:10].sum() / total_var),
        'top_50_variance_frac': float(s2[:50].sum() / total_var),
        'top_100_variance_frac': float(s2[:100].sum() / total_var),
        'k_for_var_0p9': k_for_var(0.9),
        'k_for_var_0p95': k_for_var(0.95),
        'k_for_var_0p99': k_for_var(0.99),
        'effective_rank': eff_rank,
        'participation_ratio_dims': participation * N,
    }


def plot_spectrum(s: np.ndarray, out_path: str, title: str) -> None:
    """Save a log-y line plot of the sorted singular values."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(np.arange(1, len(s) + 1), s, '-', color='C0')
    ax.set_yscale('log')
    ax.set_xlabel('singular-value index (sorted descending)')
    ax.set_ylabel('singular value (log scale)')
    ax.set_title(title)
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('checkpoint', help='Path to a JEPA checkpoint .pt')
    p.add_argument(
        '--output-dir', default='embeddings/outputs/figures',
        help='Where to save the spectrum plot (default: %(default)s)',
    )
    p.add_argument(
        '--batch-size', type=int, default=1024,
        help='Encode this many graphs per forward pass',
    )
    args = p.parse_args(argv)

    print(f'loading {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    config = ckpt['config']
    name = config.get('wandb', {}).get('name') \
        or os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"  run name : {name}")
    print(f"  epoch    : {ckpt['epoch']}")
    print(f"  embed_dim: {config['encoder']['embed_dim']}")
    print(f"  hidden   : {config['encoder']['hidden_dim']}")
    print(f"  layers   : {config['encoder']['num_layers']}")

    encoder = _build_encoder(config)
    encoder.load_state_dict(_extract_encoder_state(ckpt['jepa_state_dict']))

    cache_path = config['data']['graph_cache_path']
    if not os.path.isabs(cache_path):
        cache_path = os.path.join(_REPO_ROOT, cache_path)
    graphs = load_graph_cache(cache_path)
    print(f'  graphs   : {len(graphs):,} equations')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    embeddings, eq_ids = compute_embeddings(encoder, graphs, device, args.batch_size)
    embed_dim = embeddings.shape[1]
    print(f'  encoded  : {embeddings.shape[0]:,} × {embed_dim}')
    print(f'  norms (mean ± std): '
          f'{np.linalg.norm(embeddings, axis=1).mean():.4f} '
          f'± {np.linalg.norm(embeddings, axis=1).std():.4f}')

    s = np.linalg.svd(embeddings, compute_uv=False)
    summary = summarize_spectrum(s)
    print()
    print(f'=== Embedding-matrix singular-value spectrum ({embeddings.shape[0]} × {embed_dim}) ===')
    print(f"  s_max                 = {summary['s_max']:.4f}")
    print(f"  s_min                 = {summary['s_min']:.4e}")
    print(f"  condition number      = {summary['condition_number']:.2e}")
    print(f"  top-1   var fraction  = {summary['top_1_variance_frac']:.4f}")
    print(f"  top-10  var fraction  = {summary['top_10_variance_frac']:.4f}")
    print(f"  top-50  var fraction  = {summary['top_50_variance_frac']:.4f}")
    print(f"  top-100 var fraction  = {summary['top_100_variance_frac']:.4f}")
    print(f"  k for 90%  variance   = {summary['k_for_var_0p9']}")
    print(f"  k for 95%  variance   = {summary['k_for_var_0p95']}")
    print(f"  k for 99%  variance   = {summary['k_for_var_0p99']}")
    print(f"  effective rank        = {summary['effective_rank']:.2f}")
    print(f"  participation dims    = {summary['participation_ratio_dims']:.2f}")

    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'embedding_spectrum_{name}.png')
    plot_spectrum(
        s, out_path,
        title=f'{name}  ·  ep{ckpt["epoch"]}  ·  {embeddings.shape[0]} eqs × {embed_dim} dims',
    )
    print()
    print(f'saved plot: {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
