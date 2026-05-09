"""Evaluate a JEPA checkpoint on the Austin-implications holdout subset.

Austin implications (``data/raw/etp/Austin_implications.txt``) are
implication pairs that are FALSE in general (refuted by infinite magma)
but happen to be TRUE for every finite magma. They are adversarially
hard FALSE pairs: structurally near-identical to TRUE implications, so
a model that latches onto small-counterexample patterns will mis-label
them as TRUE.

This script measures FALSE-recall on the subset of Austin pairs that
fall in the holdout (so they were never seen during training) and
compares against the overall FALSE-recall on the same checkpoint. A
big gap (Austin recall ≪ overall recall) means the model relies on
finite-counterexample-like patterns — informative about *what* it's
actually learning, not just *how well*.

Usage:
  # One checkpoint:
  python scripts/austin_eval.py outputs/checkpoints/<name>/epoch_NNN.pt

  # All checkpoints with a recorded result JSON, in one table:
  python scripts/austin_eval.py --all-records
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch_geometric.data import Batch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from src.data.dataset import load_graph_cache  # noqa: E402
from src.model.encoder import EquationEncoder  # noqa: E402
from src.model.jepa import JEPA  # noqa: E402
from src.model.predictor import ImplicationPredictor  # noqa: E402


def _load_austin_pairs(path: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            m = re.match(r'Equation(\d+)\s*→\s*Equation(\d+)', line.strip())
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
    return out


def _select_holdout_austin(
    pair_table_path: str,
    holdout_path: str,
    austin: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    """Return ``[(row, eq1_id, eq2_id, value), ...]`` for Austin pairs in the
    holdout."""
    holdout_idx = pq.read_table(holdout_path)['pair_idx'].to_numpy().astype(np.int64)
    tbl = pq.read_table(pair_table_path)
    eq1_arr = tbl['eq1_id'].to_numpy()
    eq2_arr = tbl['eq2_id'].to_numpy()
    val_arr = tbl['value'].to_numpy()
    out = []
    for row in holdout_idx:
        pair = (int(eq1_arr[row]), int(eq2_arr[row]))
        if pair in austin:
            out.append((int(row), pair[0], pair[1], int(val_arr[row])))
    return out


def _evaluate_checkpoint(ckpt_path: str, austin_pairs: set, device: str = 'cuda') -> dict:
    ckpt = torch.load(ckpt_path, weights_only=False, map_location='cpu')
    cfg = ckpt['config']
    name = (cfg.get('wandb', {}) or {}).get('name') or os.path.basename(ckpt_path)

    enc = EquationEncoder(
        node_features=cfg['encoder']['node_features'],
        edge_features=cfg['encoder']['edge_features'],
        hidden_dim=cfg['encoder']['hidden_dim'],
        embed_dim=cfg['encoder']['embed_dim'],
        num_layers=cfg['encoder']['num_layers'],
        num_heads=cfg['encoder']['num_heads'],
        dropout=cfg['encoder']['dropout'],
    )
    pred = ImplicationPredictor(
        embed_dim=cfg['predictor']['embed_dim'],
        hidden_dim=cfg['predictor']['hidden_dim'],
    )
    jepa = JEPA(enc, pred).to(device)
    jepa.load_state_dict(ckpt['jepa_state_dict'])
    jepa.eval()

    pair_path = os.path.join(_REPO_ROOT, cfg['data']['pair_table_path'])
    holdout_path = os.path.join(_REPO_ROOT, cfg['data']['holdout_path'])
    cache_path = os.path.join(_REPO_ROOT, cfg['data']['graph_cache_path'])

    holdout_austin = _select_holdout_austin(pair_path, holdout_path, austin_pairs)
    if not holdout_austin:
        return {'name': name, 'epoch': int(ckpt['epoch']), 'n_austin_holdout': 0}

    graphs = load_graph_cache(cache_path)
    g1 = Batch.from_data_list([graphs[p[1]] for p in holdout_austin]).to(device)
    g2 = Batch.from_data_list([graphs[p[2]] for p in holdout_austin]).to(device)
    with torch.no_grad():
        z_pred, z_tgt = jepa(g1, g2)
    distances = F.pairwise_distance(z_pred, z_tgt).cpu().numpy()
    labels = np.array([p[3] for p in holdout_austin])

    return {
        'name': name,
        'epoch': int(ckpt['epoch']),
        'checkpoint': os.path.relpath(ckpt_path, _REPO_ROOT),
        'n_austin_holdout': len(holdout_austin),
        'n_austin_FALSE': int((labels == 0).sum()),
        'n_austin_TRUE': int((labels == 1).sum()),
        'distances': distances,
        'pairs': [(p[1], p[2]) for p in holdout_austin],
        'labels': labels,
    }


def _print_one(result: dict, threshold: float, overall_false_recall: float | None = None) -> None:
    name, epoch = result['name'], result['epoch']
    n = result['n_austin_holdout']
    if n == 0:
        print(f'{name} (ep {epoch}): no Austin pairs in holdout')
        return
    distances = result['distances']
    labels = result['labels']
    false_mask = labels == 0
    n_false = int(false_mask.sum())
    if n_false == 0:
        print(f'{name} (ep {epoch}): no FALSE Austin pairs in holdout')
        return
    preds_true = (distances < threshold).astype(int)
    n_correct_false = int(((preds_true == 0) & false_mask).sum())
    austin_false_recall = n_correct_false / n_false
    print(
        f'{name} (ep {epoch}, threshold={threshold:.4f}): '
        f'Austin FALSE recall = {austin_false_recall:.4f} '
        f'({n_correct_false}/{n_false})'
    )
    if overall_false_recall is not None:
        gap = overall_false_recall - austin_false_recall
        print(f'  vs overall FALSE recall {overall_false_recall:.4f}  →  gap {gap:+.4f}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', help='Single checkpoint .pt to evaluate')
    parser.add_argument('--all-records', action='store_true',
                        help='Evaluate every result in outputs/results/ on its referenced checkpoint')
    parser.add_argument('--detail', action='store_true',
                        help='Print per-pair distance/prediction table for the single-checkpoint mode')
    args = parser.parse_args(argv)

    austin = _load_austin_pairs(os.path.join(_REPO_ROOT, 'data/raw/etp/Austin_implications.txt'))
    print(f'Austin implications: {len(austin):,} pairs total\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.all_records:
        results_dir = os.path.join(_REPO_ROOT, 'outputs/results')
        rows = []
        for fname in sorted(os.listdir(results_dir)):
            if not fname.endswith('.json'):
                continue
            with open(os.path.join(results_dir, fname)) as f:
                rec = json.load(f)
            ckpt_rel = rec.get('checkpoint')
            if not ckpt_rel:
                continue
            ckpt_path = os.path.join(_REPO_ROOT, ckpt_rel)
            if not os.path.exists(ckpt_path):
                print(f'SKIP {rec.get("name")}: checkpoint missing at {ckpt_rel}')
                continue
            try:
                result = _evaluate_checkpoint(ckpt_path, austin, device=device)
            except Exception as e:
                print(f'SKIP {rec.get("name")}: {type(e).__name__}: {e}')
                continue
            threshold = float(rec['eval']['threshold'])
            overall_false = float(rec['eval']['false_recall'])
            overall_bal = float(rec['eval']['balanced_accuracy'])
            n = result['n_austin_holdout']
            n_false = result['n_austin_FALSE']
            distances = result['distances']
            labels = result['labels']
            false_mask = labels == 0
            preds_true = (distances < threshold).astype(int)
            austin_false_recall = float(((preds_true == 0) & false_mask).sum()) / max(n_false, 1)
            rows.append({
                'name': rec['name'],
                'overall_bal_acc': overall_bal,
                'overall_false_recall': overall_false,
                'austin_false_recall': austin_false_recall,
                'gap': overall_false - austin_false_recall,
                'n_false_austin': n_false,
            })

        # Sort by overall_bal_acc ascending so the table reads chronologically-ish.
        rows.sort(key=lambda r: r['overall_bal_acc'])
        name_w = max(len(r['name']) for r in rows)
        print(f'{"name":<{name_w}}  {"bal_acc":>7}  {"overall_F":>9}  {"austin_F":>8}  {"gap":>7}  {"n":>3}')
        for r in rows:
            print(
                f'{r["name"]:<{name_w}}  '
                f'{r["overall_bal_acc"]:>7.4f}  '
                f'{r["overall_false_recall"]:>9.4f}  '
                f'{r["austin_false_recall"]:>8.4f}  '
                f'{r["gap"]:>+7.4f}  '
                f'{r["n_false_austin"]:>3}'
            )
        return 0

    if not args.checkpoint:
        parser.print_help()
        return 2

    result = _evaluate_checkpoint(args.checkpoint, austin, device=device)
    # Try to find a sibling result JSON for the threshold + overall recall.
    threshold = None
    overall = None
    name = result['name']
    record_path = os.path.join(_REPO_ROOT, 'outputs/results', f'{name}.json')
    if os.path.exists(record_path):
        with open(record_path) as f:
            rec = json.load(f)
        threshold = float(rec['eval']['threshold'])
        overall = float(rec['eval']['false_recall'])

    if threshold is None:
        # Fall back to a self-calibrated threshold on these pairs alone —
        # but warn that the comparison to "overall" recall is then apples-
        # to-oranges.
        print('NOTE: no recorded result for this checkpoint; using median '
              'distance as a placeholder threshold.')
        threshold = float(np.median(result['distances']))

    _print_one(result, threshold, overall)
    if args.detail:
        order = np.argsort(result['distances'])
        print()
        print('per-pair detail (sorted by distance):')
        for j in order:
            d = result['distances'][j]
            label = 'FALSE' if result['labels'][j] == 0 else 'TRUE'
            pred = 'TRUE' if d < threshold else 'FALSE'
            mark = '✓' if pred == label else '✗'
            eq1, eq2 = result['pairs'][j]
            print(f'  Eq{eq1:>4} → Eq{eq2:<4}  d={d:.4f}  truth={label:<5}  pred={pred:<5}  {mark}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
