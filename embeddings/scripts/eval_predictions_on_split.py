"""Re-evaluate per-pair predictions on a train/test split.

Consumes a per-run ``predictions.parquet`` (from ``embeddings/scripts/save_predictions.py``,
which stored per-pair distances and ground-truth labels) plus a split
parquet (``pair_idx`` column listing holdout row indices into the pair
table). Calibrates the decision threshold on the train portion, then
reports bal_acc / TRUE recall / FALSE recall on the holdout.

Backs Phase 1.3 of `mlp_classifier_plan.md` — the matched-holdout
baseline for the JEPA distance metric. The 0.9840 number cited
elsewhere is the full 22M re-eval (data the JEPA trained on); this
script gives the apples-to-apples baseline against MLP-on-10%-holdout.

Output: ``matched_holdout_eval.json`` next to the predictions file
(or at ``--output-path``). Schema::

    {
      "predictions_path": "...",
      "split_path": "data/splits/level1_holdout_seed42.parquet",
      "n_pairs_total": 22028942,
      "n_pairs_train": 19826048,
      "n_pairs_holdout": 2202894,
      "threshold": 0.6543,
      "train": {"bal_acc": 0.984, "true_recall": ..., "false_recall": ...},
      "holdout": {"bal_acc": 0.984, "true_recall": ..., "false_recall": ...,
                  "n_true": ..., "n_false": ...}
    }

Usage:
  python embeddings/scripts/eval_predictions_on_split.py \
      embeddings/outputs/runs/skip_h128_seed42_diag/predictions.parquet
  python embeddings/scripts/eval_predictions_on_split.py <pred> --split data/splits/<file>.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pyarrow.parquet as pq
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings.src.training.train import calibrate_threshold  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def _metrics(distances: np.ndarray, labels: np.ndarray, threshold: float) -> dict:
    pred = (distances < threshold).astype(np.int8)
    n_true = int((labels == 1).sum())
    n_false = int((labels == 0).sum())
    tp = int(((pred == 1) & (labels == 1)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    true_rec = tp / max(n_true, 1)
    false_rec = tn / max(n_false, 1)
    return {
        'bal_acc': 0.5 * (true_rec + false_rec),
        'true_recall': true_rec,
        'false_recall': false_rec,
        'n_pairs': len(distances),
        'n_true': n_true,
        'n_false': n_false,
        'n_correct': tp + tn,
        'n_wrong': len(distances) - (tp + tn),
        'n_wrong_true': int(((pred == 0) & (labels == 1)).sum()),   # missed TRUE
        'n_wrong_false': int(((pred == 1) & (labels == 0)).sum()),  # spurious TRUE
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        'predictions',
        help='Path to a per-pair predictions parquet (eq1_id, eq2_id, '
             'ground_truth, distance columns required).',
    )
    parser.add_argument(
        '--split',
        default='data/splits/level1_holdout_seed42.parquet',
        help='Split parquet with pair_idx column (default: %(default)s).',
    )
    parser.add_argument(
        '--output-path', default=None,
        help='Where to write the JSON result. Defaults to '
             '<predictions_dir>/matched_holdout_eval.json.',
    )
    parser.add_argument(
        '--force', action='store_true',
        help='Overwrite an existing output file.',
    )
    args = parser.parse_args(argv)

    pred_path = _abs(args.predictions)
    split_path = _abs(args.split)

    if args.output_path is None:
        args.output_path = os.path.join(
            os.path.dirname(pred_path), 'matched_holdout_eval.json'
        )
    out_path = _abs(args.output_path)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force to overwrite.', file=sys.stderr)
        return 2

    print(f'reading predictions: {pred_path}')
    pred_table = pq.read_table(pred_path, columns=['ground_truth', 'distance'])
    labels = pred_table['ground_truth'].to_numpy().astype(np.int8)
    distances = pred_table['distance'].to_numpy().astype(np.float32)
    n_total = len(labels)
    print(f'  {n_total:,} pairs')

    print(f'reading split:       {split_path}')
    holdout_idx = pq.read_table(split_path, columns=['pair_idx'])['pair_idx'].to_numpy().astype(np.int64)
    print(f'  {len(holdout_idx):,} holdout indices')

    if holdout_idx.max() >= n_total:
        raise ValueError(
            f'split file references row indices up to {holdout_idx.max()} '
            f'but predictions table only has {n_total} rows — split was '
            f'likely built against a different pair table revision'
        )

    # Build boolean masks for train (90%) vs holdout (10%) portions.
    is_holdout = np.zeros(n_total, dtype=bool)
    is_holdout[holdout_idx] = True
    train_mask = ~is_holdout

    n_train = int(train_mask.sum())
    n_hold = int(is_holdout.sum())
    print(f'  train: {n_train:,}   holdout: {n_hold:,}')

    print('calibrating threshold on the train portion ...')
    d_train = distances[train_mask]
    l_train = labels[train_mask]
    threshold = float(calibrate_threshold(
        torch.from_numpy(d_train.astype(np.float32)),
        torch.from_numpy(l_train.astype(np.float32)),
    ))
    print(f'  threshold = {threshold:.4f}')

    print('evaluating on train + holdout ...')
    train_metrics = _metrics(d_train, l_train, threshold)
    hold_metrics = _metrics(distances[is_holdout], labels[is_holdout], threshold)

    print()
    print(f'  train   bal_acc = {train_metrics["bal_acc"]:.4f}'
          f'  TRUE_R={train_metrics["true_recall"]:.4f}'
          f'  FALSE_R={train_metrics["false_recall"]:.4f}')
    print(f'  holdout bal_acc = {hold_metrics["bal_acc"]:.4f}'
          f'  TRUE_R={hold_metrics["true_recall"]:.4f}'
          f'  FALSE_R={hold_metrics["false_recall"]:.4f}')

    out = {
        'predictions_path': os.path.relpath(pred_path, _REPO_ROOT),
        'split_path': os.path.relpath(split_path, _REPO_ROOT),
        'n_pairs_total': n_total,
        'n_pairs_train': n_train,
        'n_pairs_holdout': n_hold,
        'threshold': threshold,
        'train': train_metrics,
        'holdout': hold_metrics,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, sort_keys=True)
    print(f'wrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
