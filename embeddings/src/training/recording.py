"""Run a full no-cap evaluation on a saved checkpoint and write a JSON
record to ``embeddings/outputs/results/<name>.json``.

This is the function that backs both ``scripts/record_run.py`` (manual
recording from any checkpoint) and ``scripts/run_experiment.py`` (the
one-command launcher that calls it on the best epoch automatically).

The record format is the contract for ``record_run.py --summary`` and
for cross-experiment comparison:

  - run name, free-form notes, git commit at record time
  - checkpoint path (relative to repo root)
  - wandb run URL (if known)
  - epoch index of the recorded checkpoint
  - holdout split file path
  - eval metrics (loss, balanced_accuracy, true/false recall, threshold,
    n_evaluated)
  - encoder / predictor parameter counts
  - full config snapshot (the config the checkpoint was trained with)
"""

from __future__ import annotations

import json
import os
import subprocess
from typing import Any

import torch

from lib.data.dataset import make_loaders
from embeddings.src.model.encoder import EquationEncoder
from embeddings.src.model.jepa import JEPA
from embeddings.src.model.predictor import ImplicationPredictor
from embeddings.src.training.train import evaluate

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _relpath(path: str) -> str:
    return os.path.relpath(path, _REPO_ROOT) if os.path.isabs(path) else path


def record_checkpoint(
    checkpoint_path: str,
    name: str,
    *,
    notes: str = '',
    wandb_url: str | None = None,
    holdout: str | None = None,
    max_val_batches: int | None = None,
    output_dir: str = 'embeddings/outputs/results',
    force: bool = False,
) -> dict[str, Any]:
    """Evaluate the checkpoint at ``checkpoint_path`` and write the JSON
    record to ``<output_dir>/<name>.json``.

    Returns the record dict that was written.

    Raises ``FileExistsError`` if the target file already exists and
    ``force`` is not set.
    """
    out_dir = output_dir if os.path.isabs(output_dir) else os.path.join(_REPO_ROOT, output_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{name}.json')
    if os.path.exists(out_path) and not force:
        raise FileExistsError(
            f'{out_path} exists. Pass force=True to overwrite.'
        )

    print(f'loading {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
    config = ckpt['config']
    print(f"  saved at epoch {ckpt['epoch']}")
    print(f"  saved metrics: bal_acc={ckpt['metrics'].get('val/balanced_accuracy', 'NA')}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    holdout_path = holdout or config['data']['holdout_path']
    batch_size = int(config['training']['batch_size'])

    _, val_loader, _, val_ds = make_loaders(
        pair_table_path=config['data']['pair_table_path'],
        graph_cache_path=config['data']['graph_cache_path'],
        holdout_path=holdout_path,
        batch_size=batch_size,
        num_workers=int(config['data'].get('num_workers', 0)),
        pin_memory=(device == 'cuda'),
    )

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

    margin = float(config['loss']['margin'])
    true_weight = float(config['loss']['true_weight'])
    print(f'eval on {len(val_ds):,} samples, {len(val_loader):,} batches '
          f'(max_batches={max_val_batches})')
    metrics = evaluate(
        jepa, val_loader, device,
        margin=margin, true_weight=true_weight,
        max_batches=max_val_batches,
    )

    record = {
        'name': name,
        'notes': notes,
        'git_commit': _git_commit(),
        'checkpoint': _relpath(checkpoint_path),
        'wandb_url': wandb_url,
        'epoch': int(ckpt['epoch']),
        'holdout_path': _relpath(holdout_path),
        'eval': {k: (float(v) if isinstance(v, float) else v) for k, v in metrics.items()},
        'encoder_params': encoder.num_parameters(),
        'predictor_params': predictor.num_parameters(),
        'config': config,
    }

    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print()
    print(f'wrote {out_path}')
    print(
        f'  bal_acc={metrics["balanced_accuracy"]:.4f}  '
        f'true_recall={metrics["true_recall"]:.4f}  '
        f'false_recall={metrics["false_recall"]:.4f}  '
        f'threshold={metrics["threshold"]:.4f}'
    )
    return record
