"""CLI for recording / comparing training-run results.

Two modes:

  # Record one checkpoint as a named result.
  python embeddings/scripts/record_run.py embeddings/outputs/checkpoints/baseline/epoch_006.pt \\
      --name baseline_phase1_l2norm \\
      --wandb-url https://wandb.ai/.../runs/fxwe85fy \\
      --notes "10 epochs, L2-normalized, default config"

  # Compare all existing records.
  python embeddings/scripts/record_run.py --summary

The recording logic lives in ``src.training.recording.record_checkpoint``;
this script is a thin CLI wrapper plus the table-printing summary mode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings.src.training.recording import record_checkpoint  # noqa: E402


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
    if argv is None:
        argv = sys.argv[1:]

    # Summary mode — list/compare existing records, no checkpoint required.
    if argv and argv[0] in ('--summary', '--list'):
        sub = argparse.ArgumentParser()
        sub.add_argument('--summary', action='store_true')
        sub.add_argument('--list', action='store_true')
        sub.add_argument('--output-dir', default='embeddings/outputs/results')
        args = sub.parse_args(argv)
        _summarize(args.output_dir)
        return 0

    parser = argparse.ArgumentParser(description='Record a training-run result')
    parser.add_argument('checkpoint', help='Path to checkpoint .pt')
    parser.add_argument('--name', required=True, help='Record name (becomes filename)')
    parser.add_argument('--notes', default='', help='Free-form notes')
    parser.add_argument('--wandb-url', default=None, help='Wandb run URL')
    parser.add_argument('--holdout', default=None, help='Override holdout file from saved config')
    parser.add_argument('--max-val-batches', type=int, default=None,
                        help='Cap val batches (default: no cap — full eval)')
    parser.add_argument('--output-dir', default='embeddings/outputs/results',
                        help='Directory for the JSON record (default: %(default)s)')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite an existing record file')
    args = parser.parse_args(argv)

    try:
        record_checkpoint(
            checkpoint_path=args.checkpoint,
            name=args.name,
            notes=args.notes,
            wandb_url=args.wandb_url,
            holdout=args.holdout,
            max_val_batches=args.max_val_batches,
            output_dir=args.output_dir,
            force=args.force,
        )
    except FileExistsError as e:
        print(f'ERROR: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
