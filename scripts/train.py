"""CLI driver for JEPA training.

Loads a YAML config and dispatches to ``src.training.train.train``.

Usage:
  python scripts/train.py                          # full training, default config
  python scripts/train.py --sanity-check           # short run, fewer batches, no checkpoint
  python scripts/train.py --config <path>          # alternate config
  python scripts/train.py --epochs 3               # override epoch count
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.training.train import train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description='JEPA training driver')
    parser.add_argument(
        '--config',
        default='configs/default.yaml',
        help='YAML config path (default: %(default)s)',
    )
    parser.add_argument(
        '--sanity-check',
        action='store_true',
        help='Short run: 1 epoch, 50 train batches, 20 val batches, no checkpoint',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override training.epochs from the config',
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=None,
        help='Override training.batch_size from the config',
    )
    parser.add_argument(
        '--max-val-batches',
        type=int,
        default=None,
        help='Cap validation batches per eval pass (no cap by default)',
    )
    parser.add_argument(
        '--name',
        default=None,
        help='Wandb run name override',
    )
    parser.add_argument(
        '--no-holdout',
        action='store_true',
        help='Train on every labeled pair (no L1 holdout). Per-epoch eval '
             'uses an in-sample 1M-pair monitor subset for threshold '
             'calibration only — generalization estimates from this run '
             'are biased upward; intended for failure-analysis training '
             'where every pair will be re-evaluated post hoc.',
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    with open(config_path) as f:
        config = yaml.safe_load(f)

    if args.epochs is not None:
        config['training']['epochs'] = args.epochs
    if args.batch_size is not None:
        config['training']['batch_size'] = args.batch_size
    if args.max_val_batches is not None:
        config['training']['max_val_batches'] = args.max_val_batches
    if args.name is not None:
        config.setdefault('wandb', {})['name'] = args.name
    if args.no_holdout:
        config['data']['holdout_path'] = None

    train(config, sanity_check=args.sanity_check)
    return 0


if __name__ == '__main__':
    sys.exit(main())
