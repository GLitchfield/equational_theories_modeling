"""One-command experiment launcher.

Trains a JEPA on the supplied experiment YAML, picks the highest
balanced-accuracy checkpoint, runs full no-cap evaluation on the val
loader, and writes ``embeddings/outputs/results/<name>.json`` — the same record
format emitted by ``embeddings/scripts/record_run.py``.

Convention: the experiment name is derived from the YAML filename
(``embeddings/configs/experiments/baseline.yaml`` → ``embeddings/outputs/results/baseline.json``,
checkpoints in ``embeddings/outputs/checkpoints/baseline/``). The launcher forces
``config['wandb']['name']`` to match so wandb / checkpoints / result
filename all share one identifier.

Usage:
  python embeddings/scripts/run_experiment.py embeddings/configs/experiments/baseline.yaml
  python embeddings/scripts/run_experiment.py embeddings/configs/experiments/X.yaml --force
"""

from __future__ import annotations

import argparse
import os
import sys

import yaml

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings.src.training.recording import record_checkpoint  # noqa: E402
from embeddings.src.training.train import train  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description='Train a JEPA from an experiment YAML and record the best epoch',
    )
    parser.add_argument(
        'config',
        help='Path to a embeddings/configs/experiments/*.yaml file',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite an existing embeddings/outputs/results/<name>.json',
    )
    parser.add_argument(
        '--notes',
        default='',
        help='Free-form notes attached to the result record',
    )
    parser.add_argument(
        '--output-dir',
        default='embeddings/outputs/results',
        help='Directory for the JSON record (default: %(default)s)',
    )
    args = parser.parse_args(argv)

    config_path = args.config
    if not os.path.isabs(config_path):
        config_path = os.path.join(_REPO_ROOT, config_path)

    name = os.path.splitext(os.path.basename(config_path))[0]
    out_dir = args.output_dir if os.path.isabs(args.output_dir) else os.path.join(_REPO_ROOT, args.output_dir)
    out_path = os.path.join(out_dir, f'{name}.json')
    if os.path.exists(out_path) and not args.force:
        print(
            f'ERROR: {out_path} exists. Pass --force to overwrite, or rename '
            f'the YAML to produce a different result file.',
            file=sys.stderr,
        )
        return 2

    with open(config_path) as f:
        config = yaml.safe_load(f)

    config.setdefault('wandb', {})['name'] = name

    print(f'=== experiment: {name} ===')
    print(f'config: {config_path}')
    print(f'result: {out_path}')
    print()

    result = train(config, sanity_check=False)

    best_ckpt = result.get('best_checkpoint')
    if best_ckpt is None:
        print('ERROR: training produced no checkpoints (sanity-check or 0 epochs?)',
              file=sys.stderr)
        return 3

    best_epoch = result['best_epoch_metrics'].get('epoch', '?')
    best_bal = result['best_epoch_metrics'].get('val/balanced_accuracy', float('nan'))
    print(f'\nbest checkpoint: {best_ckpt}  (epoch {best_epoch}, val/bal_acc={best_bal:.4f})')
    print(f'recording with full no-cap eval ...\n')

    record_checkpoint(
        checkpoint_path=best_ckpt,
        name=name,
        notes=args.notes,
        wandb_url=result.get('wandb_url'),
        output_dir=args.output_dir,
        force=args.force,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
