"""Record a training run's results to outputs/results/<name>.json.

Loads a saved JEPA checkpoint, runs full no-cap eval on the val loader,
and writes a JSON record capturing:

  - run_name (also used as filename)
  - notes (free-form, human-supplied via --notes)
  - git_commit (HEAD SHA at record time)
  - checkpoint path
  - wandb_run_url (if --wandb-url is supplied)
  - config snapshot (the full YAML config saved with the checkpoint)
  - eval (loss, balanced_accuracy, per-class recalls, threshold, n_evaluated)

Records are committed to the repo so future variants can be compared
against committed baselines without re-running them.

Usage:
  # First time — establish a baseline
  python scripts/record_run.py outputs/checkpoints/epoch_006.pt \\
      --name baseline_phase1_l2norm \\
      --wandb-url https://wandb.ai/.../runs/fxwe85fy \\
      --notes "10 epochs, L2-normalized, default config"

  # Variant — same form, different name
  python scripts/record_run.py outputs/checkpoints/epoch_009.pt \\
      --name variant_true_weight_1p7 \\
      --notes "true_weight raised to 1.7 to upweight TRUE prior"
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.dataset import make_loaders  # noqa: E402
from src.model.encoder import EquationEncoder  # noqa: E402
from src.model.jepa import JEPA  # noqa: E402
from src.model.predictor import ImplicationPredictor  # noqa: E402
from src.training.train import evaluate  # noqa: E402


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'],
            cwd=_REPO_ROOT, stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return 'unknown'


def _summarize(records_dir: str) -> None:
    """Print a comparison table over all *.json records in records_dir."""
    if not os.path.isabs(records_dir):
        records_dir = os.path.join(_REPO_ROOT, records_dir)
    files = sorted(
        f for f in os.listdir(records_dir) if f.endswith('.json')
    )
    if not files:
        print(f'no records in {records_dir}')
        return

    rows = []
    for fname in files:
        with open(os.path.join(records_dir, fname)) as f:
            r = json.load(f)
        e = r.get('eval', {})
        rows.append({
            'name': r.get('name', fname[:-5]),
            'epoch': r.get('epoch', 'NA'),
            'bal_acc': e.get('balanced_accuracy', float('nan')),
            'true_rec': e.get('true_recall', float('nan')),
            'false_rec': e.get('false_recall', float('nan')),
            'thresh': e.get('threshold', float('nan')),
            'loss': e.get('loss', float('nan')),
            'commit': (r.get('git_commit') or '')[:8],
        })

    name_w = max((len(r['name']) for r in rows), default=4)
    print(f'{"name":<{name_w}}  {"ep":>3}  {"bal_acc":>7}  {"true":>6}  '
          f'{"false":>6}  {"thresh":>6}  {"loss":>6}  {"commit":>8}')
    for r in rows:
        print(f'{r["name"]:<{name_w}}  {str(r["epoch"]):>3}  '
              f'{r["bal_acc"]:>7.4f}  {r["true_rec"]:>6.4f}  '
              f'{r["false_rec"]:>6.4f}  {r["thresh"]:>6.4f}  '
              f'{r["loss"]:>6.4f}  {r["commit"]:>8}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='Record a training-run result')

    # --summary mode: list/compare existing records, no checkpoint required.
    if argv is None:
        argv = sys.argv[1:]
    if argv and argv[0] in ('--summary', '--list'):
        sub = argparse.ArgumentParser()
        sub.add_argument('--summary', action='store_true')
        sub.add_argument('--list', action='store_true')
        sub.add_argument('--output-dir', default='outputs/results')
        args = sub.parse_args(argv)
        _summarize(args.output_dir)
        return 0

    parser.add_argument('checkpoint', help='Path to checkpoint .pt')
    parser.add_argument('--name', required=True, help='Record name (becomes filename)')
    parser.add_argument('--notes', default='', help='Free-form notes')
    parser.add_argument('--wandb-url', default=None, help='Wandb run URL')
    parser.add_argument('--holdout', default=None, help='Override holdout file from saved config')
    parser.add_argument('--max-val-batches', type=int, default=None,
                        help='Cap val batches (default: no cap — full eval)')
    parser.add_argument('--output-dir', default='outputs/results',
                        help='Directory for the JSON record (default: %(default)s)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing record file')
    args = parser.parse_args(argv)

    out_dir = args.output_dir
    if not os.path.isabs(out_dir):
        out_dir = os.path.join(_REPO_ROOT, out_dir)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'{args.name}.json')
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists. Use --force to overwrite.', file=sys.stderr)
        return 2

    print(f'loading {args.checkpoint}')
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location='cpu')
    config = ckpt['config']
    print(f"  saved at epoch {ckpt['epoch']}")
    print(f"  saved metrics: bal_acc={ckpt['metrics'].get('val/balanced_accuracy', 'NA')}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    holdout_path = args.holdout or config['data']['holdout_path']
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
          f'(max_batches={args.max_val_batches})')
    metrics = evaluate(
        jepa, val_loader, device,
        margin=margin, true_weight=true_weight,
        max_batches=args.max_val_batches,
    )

    record = {
        'name': args.name,
        'notes': args.notes,
        'git_commit': _git_commit(),
        'checkpoint': os.path.relpath(args.checkpoint, _REPO_ROOT),
        'wandb_url': args.wandb_url,
        'epoch': int(ckpt['epoch']),
        'holdout_path': os.path.relpath(holdout_path, _REPO_ROOT)
            if os.path.isabs(holdout_path) else holdout_path,
        'eval': {k: (float(v) if isinstance(v, float) else v) for k, v in metrics.items()},
        'encoder_params': encoder.num_parameters(),
        'predictor_params': predictor.num_parameters(),
        'config': config,
    }

    with open(out_path, 'w') as f:
        json.dump(record, f, indent=2, sort_keys=True)
    print()
    print(f'wrote {out_path}')
    print()
    print(f'  bal_acc={metrics["balanced_accuracy"]:.4f}  '
          f'true_recall={metrics["true_recall"]:.4f}  '
          f'false_recall={metrics["false_recall"]:.4f}  '
          f'threshold={metrics["threshold"]:.4f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
