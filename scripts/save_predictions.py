"""Run a trained JEPA over every off-diagonal pair and dump per-pair
predictions to parquet.

Backs Step 2 of testing_plan_phase2.md: "evaluate the best checkpoint
on ALL ~22M pairs. Save per-pair predictions." The downstream failure
analyses all read this single parquet.

Output schema (one row per pair, ordered by pair-table row index):

  eq1_id       : int32
  eq2_id       : int32
  ground_truth : int8       0 = FALSE, 1 = TRUE  (binary, from value)
  code         : int8       full ETP code: -4 / -3 / 3 / 4
  predicted    : int8       0 = FALSE, 1 = TRUE  (distance < threshold)
  distance     : float32    pairwise_distance(z_predicted, z_target)

Threshold: by default re-calibrated on a 1M-pair random subsample of
the corpus (deterministic via seed) to maximize balanced accuracy —
matches the per-epoch monitor logic from no-holdout training.
``--threshold X`` overrides.

Usage:
  python scripts/save_predictions.py outputs/checkpoints/<name>/epoch_NNN.pt
  python scripts/save_predictions.py <ckpt> --output-dir outputs/failure_analysis
  python scripts/save_predictions.py <ckpt> --threshold 0.7636
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.dataset import load_graph_cache  # noqa: E402
from src.model.encoder import EquationEncoder  # noqa: E402
from src.model.jepa import JEPA  # noqa: E402
from src.model.predictor import ImplicationPredictor  # noqa: E402
from src.training.train import calibrate_threshold  # noqa: E402


def _build_jepa(config: dict) -> JEPA:
    enc = EquationEncoder(
        node_features=config['encoder']['node_features'],
        edge_features=config['encoder']['edge_features'],
        hidden_dim=config['encoder']['hidden_dim'],
        embed_dim=config['encoder']['embed_dim'],
        num_layers=config['encoder']['num_layers'],
        num_heads=config['encoder']['num_heads'],
        dropout=config['encoder']['dropout'],
        skip_pooled=config['encoder'].get('skip_pooled', False),
        proj_dim=config['encoder'].get('proj_dim', None),
    )
    pred = ImplicationPredictor(
        embed_dim=config['predictor']['embed_dim'],
        hidden_dim=config['predictor']['hidden_dim'],
    )
    return JEPA(enc, pred)


def _abs(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_REPO_ROOT, path)


def _encode_all_equations(
    encoder: EquationEncoder,
    graphs: dict,
    device: str,
    batch_size: int,
) -> tuple[np.ndarray, dict[int, int]]:
    """Encode every equation once. Returns:
      - ``Z``: ``[N, embed_dim]`` numpy array of unit-normalized embeddings
      - ``id_to_idx``: equation_id → row index into Z
    """
    encoder = encoder.to(device).eval()
    eq_ids = sorted(graphs.keys())
    chunks: list[np.ndarray] = []
    with torch.no_grad():
        for s in range(0, len(eq_ids), batch_size):
            batch = Batch.from_data_list(
                [graphs[i] for i in eq_ids[s:s + batch_size]]
            ).to(device)
            chunks.append(encoder(batch).cpu().numpy())
    Z = np.concatenate(chunks, axis=0)
    id_to_idx = {eq_id: idx for idx, eq_id in enumerate(eq_ids)}
    return Z, id_to_idx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', help='Path to a JEPA checkpoint .pt')
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for the predictions parquet. Defaults to '
             '``outputs/runs/<wandb.name from checkpoint>/``; pass to '
             'override (e.g. when comparing two runs side by side).',
    )
    parser.add_argument(
        '--filename', default='predictions.parquet',
        help='Output filename within --output-dir',
    )
    parser.add_argument(
        '--batch-size', type=int, default=4096,
        help='Pair-batch size for the distance pass (default: %(default)s)',
    )
    parser.add_argument(
        '--encode-batch-size', type=int, default=1024,
        help='Equation-batch size for the per-equation encode pass (default: %(default)s)',
    )
    parser.add_argument(
        '--threshold', type=float, default=None,
        help='Decision threshold (distance < t → predict TRUE). If unset, '
             'calibrate on a 1M-pair random subsample to maximize balanced '
             'accuracy.',
    )
    parser.add_argument(
        '--threshold-sample-size', type=int, default=1_000_000,
        help='Sample size for threshold calibration (default: %(default)s)',
    )
    parser.add_argument(
        '--threshold-seed', type=int, default=42,
        help='RNG seed for the threshold-calibration subsample',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite an existing output file',
    )
    args = parser.parse_args(argv)

    # Peek at the checkpoint config to derive the default output dir.
    if args.output_dir is None:
        ckpt_peek = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
        name_hint = (
            (ckpt_peek.get('config', {}).get('wandb', {}) or {}).get('name')
            or os.path.splitext(os.path.basename(args.checkpoint))[0]
        )
        del ckpt_peek
        args.output_dir = os.path.join('outputs/runs', name_hint)

    out_dir = _abs(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, args.filename)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists. Pass --force to overwrite.', file=sys.stderr)
        return 2

    print(f'loading {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    cfg = ckpt['config']
    name = (cfg.get('wandb', {}) or {}).get('name') \
        or os.path.splitext(os.path.basename(args.checkpoint))[0]
    print(f"  run name: {name}")
    print(f"  epoch   : {ckpt['epoch']}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    jepa = _build_jepa(cfg).to(device)
    jepa.load_state_dict(ckpt['jepa_state_dict'])
    jepa.eval()

    # --- 1. load corpus + encode every equation once -----------------------
    pair_path = _abs(cfg['data']['pair_table_path'])
    cache_path = _abs(cfg['data']['graph_cache_path'])
    print(f'reading pair table {pair_path}')
    tbl = pq.read_table(pair_path)
    eq1_arr = tbl['eq1_id'].to_numpy()
    eq2_arr = tbl['eq2_id'].to_numpy()
    val_arr = tbl['value'].to_numpy()
    code_arr = tbl['code'].to_numpy() if 'code' in tbl.column_names else None
    n_pairs = len(eq1_arr)
    print(f'  {n_pairs:,} pairs')

    print(f'reading graph cache {cache_path}')
    graphs = load_graph_cache(cache_path)
    print(f'  {len(graphs):,} equations')

    print('encoding every equation through context + target encoders ...')
    t0 = time.perf_counter()
    Z_ctx, id_to_idx = _encode_all_equations(
        jepa.context_encoder, graphs, device, args.encode_batch_size,
    )
    Z_tgt, _ = _encode_all_equations(
        jepa.target_encoder, graphs, device, args.encode_batch_size,
    )
    print(f'  encoded {Z_ctx.shape} (context) + {Z_tgt.shape} (target) '
          f'in {time.perf_counter() - t0:.1f}s')

    # --- 2. project context through predictor (in chunks) -----------------
    # The predictor operates on the L2-normalized encoder output. We
    # already have Z_ctx unit-normalized from the encoder; running it
    # through the predictor gives z_predicted for each equation, which
    # we then pair with z_target by row.
    print('projecting context embeddings through the predictor ...')
    Z_ctx_t = torch.from_numpy(Z_ctx).to(device)
    pred_chunks: list[np.ndarray] = []
    chunk = 16_384
    with torch.no_grad():
        for s in range(0, len(Z_ctx_t), chunk):
            pred_chunks.append(jepa.predictor(Z_ctx_t[s:s + chunk]).cpu().numpy())
    Z_pred = np.concatenate(pred_chunks, axis=0)
    print(f'  z_predicted: {Z_pred.shape}')

    # --- 3. compute pairwise distance for every row in the pair table ----
    # Lookup arrays: pair-row → embedding-row index. eq IDs are 1-indexed
    # in the table; id_to_idx maps eq_id → row in Z.
    print('computing per-pair distances ...')
    t1 = time.perf_counter()
    # Vectorize: build full-corpus index arrays.
    idx1 = np.fromiter((id_to_idx[int(i)] for i in eq1_arr), count=n_pairs, dtype=np.int64)
    idx2 = np.fromiter((id_to_idx[int(i)] for i in eq2_arr), count=n_pairs, dtype=np.int64)
    Z_pred_t = torch.from_numpy(Z_pred).to(device)
    Z_tgt_t = torch.from_numpy(Z_tgt).to(device)
    distances = np.empty(n_pairs, dtype=np.float32)
    for s in range(0, n_pairs, args.batch_size):
        e = min(s + args.batch_size, n_pairs)
        a = Z_pred_t[idx1[s:e]]
        b = Z_tgt_t[idx2[s:e]]
        distances[s:e] = F.pairwise_distance(a, b).cpu().numpy()
    print(f'  distance pass: {time.perf_counter() - t1:.1f}s')

    # --- 4. threshold ------------------------------------------------------
    if args.threshold is not None:
        threshold = float(args.threshold)
        print(f'using supplied threshold = {threshold:.4f}')
    else:
        print(f'calibrating threshold on a {args.threshold_sample_size:,}-pair sample ...')
        rng = np.random.default_rng(args.threshold_seed)
        size = min(args.threshold_sample_size, n_pairs)
        sample = rng.choice(n_pairs, size=size, replace=False)
        d_sample = torch.from_numpy(distances[sample])
        l_sample = torch.from_numpy(val_arr[sample].astype(np.float32))
        threshold = float(calibrate_threshold(d_sample, l_sample))
        print(f'  threshold = {threshold:.4f}')

    predicted = (distances < threshold).astype(np.int8)

    # --- 5. headline metrics ----------------------------------------------
    labels = val_arr.astype(np.int8)
    n_true = int((labels == 1).sum())
    n_false = int((labels == 0).sum())
    tp = int(((predicted == 1) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    true_rec = tp / max(n_true, 1)
    false_rec = tn / max(n_false, 1)
    bal_acc = 0.5 * (true_rec + false_rec)
    n_wrong = int((predicted != labels).sum())
    print()
    print(f'=== {name} (epoch {ckpt["epoch"]}) on all {n_pairs:,} pairs ===')
    print(f'  threshold      = {threshold:.4f}')
    print(f'  bal_acc        = {bal_acc:.4f}')
    print(f'  true_recall    = {true_rec:.4f}  ({tp:,}/{n_true:,})')
    print(f'  false_recall   = {false_rec:.4f}  ({tn:,}/{n_false:,})')
    print(f'  total wrong    = {n_wrong:,}  ({100*n_wrong/n_pairs:.2f}%)')
    n_wrong_true = int(((predicted == 0) & (labels == 1)).sum())
    n_wrong_false = int(((predicted == 1) & (labels == 0)).sum())
    print(f'    wrong TRUE  (predicted FALSE) : {n_wrong_true:,}')
    print(f'    wrong FALSE (predicted TRUE)  : {n_wrong_false:,}')

    # --- 6. write parquet --------------------------------------------------
    print(f'writing predictions to {out_path}')
    t2 = time.perf_counter()
    columns = {
        'eq1_id': pa.array(eq1_arr, type=pa.int32()),
        'eq2_id': pa.array(eq2_arr, type=pa.int32()),
        'ground_truth': pa.array(labels, type=pa.int8()),
        'predicted': pa.array(predicted, type=pa.int8()),
        'distance': pa.array(distances, type=pa.float32()),
    }
    if code_arr is not None:
        columns['code'] = pa.array(code_arr, type=pa.int8())
    pa_tbl = pa.table(columns)
    pq.write_table(pa_tbl, out_path, compression='zstd')
    size_mb = os.path.getsize(out_path) / 1024 / 1024
    print(f'  wrote {size_mb:.1f} MB in {time.perf_counter() - t2:.1f}s')

    # --- 7. summary JSON sidecar ------------------------------------------
    summary_path = os.path.join(out_dir, 'failure_summary.json')
    summary = {
        'name': name,
        'checkpoint': os.path.relpath(args.checkpoint, _REPO_ROOT),
        'epoch': int(ckpt['epoch']),
        'threshold': threshold,
        'n_pairs': n_pairs,
        'bal_acc': bal_acc,
        'true_recall': true_rec,
        'false_recall': false_rec,
        'n_true': n_true,
        'n_false': n_false,
        'n_wrong': n_wrong,
        'n_wrong_true': n_wrong_true,
        'n_wrong_false': n_wrong_false,
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f'wrote summary to {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
