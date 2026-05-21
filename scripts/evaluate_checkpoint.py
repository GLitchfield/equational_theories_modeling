"""Evaluate a saved JEPA checkpoint on a holdout split.

Loads ``outputs/checkpoints/<name>.pt``, rebuilds the JEPA from the
config snapshot stored in the checkpoint, and runs ``evaluate`` over the
full validation DataLoader (no batch cap by default). Prints the same
metrics as in-training eval.

Usage:
  python scripts/evaluate_checkpoint.py outputs/checkpoints/epoch_006.pt
  python scripts/evaluate_checkpoint.py <ckpt> --max-val-batches 200
  python scripts/evaluate_checkpoint.py <ckpt> --holdout <path>   # alt split
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.dataset import make_loaders  # noqa: E402
from embeddings.src.model.encoder import EquationEncoder  # noqa: E402
from embeddings.src.model.jepa import JEPA  # noqa: E402
from embeddings.src.model.predictor import ImplicationPredictor  # noqa: E402
from embeddings.src.training.train import evaluate  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Full eval of a JEPA checkpoint')
    parser.add_argument('checkpoint', help='Path to checkpoint .pt file')
    parser.add_argument('--max-val-batches', type=int, default=None,
                        help='Cap val batches (default: no cap)')
    parser.add_argument('--holdout', default=None,
                        help='Override the holdout file from the saved config')
    parser.add_argument('--batch-size', type=int, default=None,
                        help='Override batch size (default: from saved config)')
    args = parser.parse_args(argv)

    print(f'loading {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    config = ckpt['config']
    print(f"  saved at epoch {ckpt['epoch']}")
    print(f"  saved metrics: bal_acc={ckpt['metrics'].get('val/balanced_accuracy', 'NA')}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'  device: {device}')

    holdout_path = args.holdout or config['data']['holdout_path']
    batch_size = args.batch_size or int(config['training']['batch_size'])

    print('building loaders ...')
    _, val_loader, _, val_ds = make_loaders(
        pair_table_path=config['data']['pair_table_path'],
        graph_cache_path=config['data']['graph_cache_path'],
        holdout_path=holdout_path,
        batch_size=batch_size,
        num_workers=int(config['data'].get('num_workers', 0)),
        pin_memory=(device == 'cuda'),
    )
    print(f'  val: {len(val_ds):,} samples, {len(val_loader):,} batches')

    enc_cfg = config['encoder']
    pred_cfg = config['predictor']
    encoder = EquationEncoder(
        node_features=enc_cfg['node_features'],
        edge_features=enc_cfg['edge_features'],
        hidden_dim=enc_cfg['hidden_dim'],
        embed_dim=enc_cfg['embed_dim'],
        num_layers=enc_cfg['num_layers'],
        num_heads=enc_cfg['num_heads'],
        dropout=enc_cfg['dropout'],
    )
    predictor = ImplicationPredictor(
        embed_dim=pred_cfg.get('embed_dim', enc_cfg['embed_dim']),
        hidden_dim=pred_cfg['hidden_dim'],
    )
    jepa = JEPA(encoder, predictor).to(device)
    jepa.load_state_dict(ckpt['jepa_state_dict'])
    jepa.eval()
    print(f'  encoder:   {encoder.num_parameters():,} params')
    print(f'  predictor: {predictor.num_parameters():,} params')

    margin = float(config['loss']['margin'])
    true_weight = float(config['loss']['true_weight'])

    print('running eval ...')
    metrics = evaluate(
        jepa, val_loader, device,
        margin=margin, true_weight=true_weight,
        max_batches=args.max_val_batches,
    )
    print()
    print('results:')
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f'  {k:25s} {v:.6f}')
        else:
            print(f'  {k:25s} {v}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
