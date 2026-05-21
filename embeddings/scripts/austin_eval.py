"""Evaluate a JEPA checkpoint on the Austin-implications subset.

Austin implications (``data/raw/etp/Austin_implications.txt``) are
implication pairs that hold for every FINITE magma but fail on some
INFINITE magma. The general-case answer (over all magmas) is FALSE,
and that is the ETP's authoritative ground truth — derived from
Lean-verified infinite-counterexample constructions, not from Vampire.

All 820 Austin pairs are therefore ground-truth FALSE for our task.
They are adversarially hard: structurally near-identical to TRUE
implications, since a model trained on finite-counterexample-style
signals (which is essentially what Vampire's fmb output gives us)
has no native way to distinguish them.

How they sit in our pair table (built from Vampire's 2025-08-11
output): Vampire's fmb couldn't find finite counterexamples (none
exist) and its casc proof search couldn't close the proof (the
implication IS false in general); it timed out and marked these
pairs ``value=2`` (unknown). So 610 of the 820 are excluded from
training and the L1 holdout entirely. The remaining 210 happen to
have ``value=0`` already (presumably resolved by Vampire on a
different code path), and 21 of those fall in the L1 holdout via
the 10% random split.

Three buckets, all ground-truth FALSE:

  ``train``   — Austin pair with corpus value=0, NOT in L1 holdout
                (~189 pairs). Model has seen these during training.
                Performance measures memorization.
  ``holdout`` — Austin pair with corpus value=0, IN the L1 holdout
                (~21 pairs). Held-out generalization on adversarial
                structure.
  ``unseen``  — Austin pair with corpus value=2 (unknown to Vampire).
                ~610 pairs. Never seen during training and never in
                the holdout split. True out-of-distribution
                generalization on adversarial structure.

A model that has *learned the structure* of equational implication
should perform similarly across the three buckets. A model that has
*memorized finite-counterexample patterns* will get the train bucket
right and fail on holdout+unseen. The gap between the three is the
diagnostic.

Usage:
  # One checkpoint, three-bucket Austin recall:
  python embeddings/scripts/austin_eval.py embeddings/outputs/checkpoints/<name>/epoch_NNN.pt

  # Same plus per-pair detail (sorted by distance) for the holdout bucket:
  python embeddings/scripts/austin_eval.py <ckpt> --detail

  # All checkpoints with a recorded result JSON, in one table:
  python embeddings/scripts/austin_eval.py --all-records
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from lib.data.dataset import load_graph_cache  # noqa: E402
from embeddings.src.model.encoder import EquationEncoder  # noqa: E402
from embeddings.src.model.jepa import JEPA  # noqa: E402
from embeddings.src.model.predictor import ImplicationPredictor  # noqa: E402


def _load_austin_pairs(path: str) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    with open(path) as f:
        for line in f:
            m = re.match(r'Equation(\d+)\s*→\s*Equation(\d+)', line.strip())
            if m:
                out.add((int(m.group(1)), int(m.group(2))))
    return out


def _bucket_austin_pairs(
    pair_table_path: str,
    holdout_path: str | None,
    austin: set[tuple[int, int]],
) -> dict[str, list[tuple[int, int, int]]]:
    """Bucket every Austin pair by training visibility.

    Returns ``{'train': [...], 'holdout': [...], 'unseen': [...]}`` where
    each entry is ``(eq1_id, eq2_id, corpus_value)``. All Austin pairs
    are ground-truth FALSE per the ETP's Lean-verified infinite-
    counterexample constructions.

    With ``holdout_path=None`` (no-holdout training mode) the model has
    seen every labeled pair during training, so every Austin pair lands
    in the ``train`` bucket — ``holdout`` and ``unseen`` are empty.
    """
    if holdout_path is None:
        holdout_idx: set[int] = set()
    else:
        holdout_idx = set(pq.read_table(holdout_path)['pair_idx'].to_numpy().astype(np.int64).tolist())
    tbl = pq.read_table(pair_table_path)
    eq1_arr = tbl['eq1_id'].to_numpy()
    eq2_arr = tbl['eq2_id'].to_numpy()
    val_arr = tbl['value'].to_numpy()

    # Build pair → row index lookup so we can ask each Austin pair "which
    # bucket are you in?".
    pair_to_row: dict[tuple[int, int], int] = {}
    for i in range(len(eq1_arr)):
        pair_to_row[(int(eq1_arr[i]), int(eq2_arr[i]))] = i

    buckets: dict[str, list[tuple[int, int, int]]] = {
        'train': [], 'holdout': [], 'unseen': [],
    }
    for pair in austin:
        row = pair_to_row.get(pair)
        if row is None:
            continue  # Pair isn't in the corpus at all (shouldn't happen
                      # for the current 4694-eq corpus, which has every
                      # off-diagonal pair, but defensive).
        v = int(val_arr[row])
        if v == 2:
            buckets['unseen'].append((pair[0], pair[1], v))
        elif row in holdout_idx:
            buckets['holdout'].append((pair[0], pair[1], v))
        else:
            buckets['train'].append((pair[0], pair[1], v))
    return buckets


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
    holdout_raw = cfg['data'].get('holdout_path')
    holdout_path = (
        os.path.join(_REPO_ROOT, holdout_raw) if holdout_raw else None
    )
    cache_path = os.path.join(_REPO_ROOT, cfg['data']['graph_cache_path'])

    buckets = _bucket_austin_pairs(pair_path, holdout_path, austin_pairs)
    graphs = load_graph_cache(cache_path)

    out = {
        'name': name,
        'epoch': int(ckpt['epoch']),
        'checkpoint': os.path.relpath(ckpt_path, _REPO_ROOT),
        'buckets': {},
    }
    for bucket_name, pairs in buckets.items():
        if not pairs:
            out['buckets'][bucket_name] = {'n': 0, 'distances': np.array([]), 'pairs': []}
            continue
        g1 = Batch.from_data_list([graphs[p[0]] for p in pairs]).to(device)
        g2 = Batch.from_data_list([graphs[p[1]] for p in pairs]).to(device)
        with torch.no_grad():
            z_pred, z_tgt = jepa(g1, g2)
        distances = F.pairwise_distance(z_pred, z_tgt).cpu().numpy()
        out['buckets'][bucket_name] = {
            'n': len(pairs),
            'distances': distances,
            'pairs': [(p[0], p[1]) for p in pairs],
        }
    return out


def _bucket_recall(distances: np.ndarray, threshold: float) -> tuple[float, int, int]:
    """All Austin pairs are ground-truth FALSE, so recall = fraction with
    ``distance >= threshold``. Returns ``(recall, n_correct, n_total)``.
    """
    if distances.size == 0:
        return float('nan'), 0, 0
    n_correct = int((distances >= threshold).sum())
    return n_correct / distances.size, n_correct, int(distances.size)


def _print_one(result: dict, threshold: float, overall_false_recall: float | None = None) -> None:
    name, epoch = result['name'], result['epoch']
    print(f'{name} (ep {epoch}, threshold={threshold:.4f})')
    if overall_false_recall is not None:
        print(f'  overall FALSE recall on L1 holdout : {overall_false_recall:.4f}')
    print()
    print(f'  {"bucket":<8}  {"n":>4}  {"recall":>7}  {"correct":>9}')
    for bucket in ('train', 'holdout', 'unseen'):
        info = result['buckets'][bucket]
        n = info['n']
        if n == 0:
            print(f'  {bucket:<8}  {n:>4}     —          —')
            continue
        rec, ok, total = _bucket_recall(info['distances'], threshold)
        print(f'  {bucket:<8}  {n:>4}   {rec:>7.4f}    {ok:>3}/{total:<3}')


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', nargs='?', help='Single checkpoint .pt to evaluate')
    parser.add_argument('--all-records', action='store_true',
                        help='Evaluate every result in embeddings/outputs/results/ on its referenced checkpoint')
    parser.add_argument('--detail', action='store_true',
                        help='Print per-pair distance/prediction table for the single-checkpoint mode')
    args = parser.parse_args(argv)

    austin = _load_austin_pairs(os.path.join(_REPO_ROOT, 'data/raw/etp/Austin_implications.txt'))
    print(f'Austin implications: {len(austin):,} pairs total\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.all_records:
        results_dir = os.path.join(_REPO_ROOT, 'embeddings/outputs/results')
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
            train_rec, _, _ = _bucket_recall(result['buckets']['train']['distances'], threshold)
            ho_rec, _, _ = _bucket_recall(result['buckets']['holdout']['distances'], threshold)
            unseen_rec, _, _ = _bucket_recall(result['buckets']['unseen']['distances'], threshold)
            rows.append({
                'name': rec['name'],
                'overall_bal_acc': overall_bal,
                'overall_false_recall': overall_false,
                'train_rec': train_rec,
                'ho_rec': ho_rec,
                'unseen_rec': unseen_rec,
            })

        # Sort by overall_bal_acc ascending so the table reads
        # chronologically-ish.
        rows.sort(key=lambda r: r['overall_bal_acc'])
        name_w = max(len(r['name']) for r in rows)
        header = (
            f'{"name":<{name_w}}  {"bal_acc":>7}  {"overall_F":>9}  '
            f'{"a_train":>7}  {"a_holdout":>9}  {"a_unseen":>8}'
        )
        print(header)
        print('-' * len(header))
        for r in rows:
            print(
                f'{r["name"]:<{name_w}}  '
                f'{r["overall_bal_acc"]:>7.4f}  '
                f'{r["overall_false_recall"]:>9.4f}  '
                f'{r["train_rec"]:>7.4f}  '
                f'{r["ho_rec"]:>9.4f}  '
                f'{r["unseen_rec"]:>8.4f}'
            )
        print()
        print('a_train   = Austin FALSE recall on training-set Austin pairs (memorization)')
        print('a_holdout = Austin FALSE recall on L1-holdout Austin pairs (generalization)')
        print('a_unseen  = Austin FALSE recall on never-seen Austin pairs (OOD generalization;')
        print('            value=2 in the corpus → excluded from train + holdout)')
        return 0

    if not args.checkpoint:
        parser.print_help()
        return 2

    result = _evaluate_checkpoint(args.checkpoint, austin, device=device)
    # Threshold + overall_false_recall come from a recorded result. For
    # standard runs that's embeddings/outputs/results/<name>.json (written by
    # run_experiment / record_run). For failure-analysis no-holdout
    # runs that's embeddings/outputs/failure_analysis/failure_summary.json (written
    # by save_predictions). Fall back to all-bucket median if neither
    # exists.
    threshold = None
    overall = None
    name = result['name']
    for candidate in (
        os.path.join(_REPO_ROOT, 'embeddings/outputs/results', f'{name}.json'),
        os.path.join(_REPO_ROOT, 'embeddings/outputs/failure_analysis/failure_summary.json'),
    ):
        if not os.path.exists(candidate):
            continue
        with open(candidate) as f:
            rec = json.load(f)
        if 'eval' in rec:
            threshold = float(rec['eval']['threshold'])
            overall = float(rec['eval']['false_recall'])
        else:
            threshold = float(rec.get('threshold')) if rec.get('threshold') is not None else None
            overall = float(rec.get('false_recall')) if rec.get('false_recall') is not None else None
        if threshold is not None:
            break

    if threshold is None:
        print('NOTE: no recorded threshold for this checkpoint; using median '
              'distance across Austin pairs as a placeholder.')
        all_dists = np.concatenate(
            [result['buckets'][b]['distances'] for b in ('train', 'holdout', 'unseen')
             if result['buckets'][b]['n'] > 0]
        )
        threshold = float(np.median(all_dists)) if len(all_dists) else 1.0

    _print_one(result, threshold, overall)
    if args.detail:
        for bucket in ('holdout', 'unseen'):
            info = result['buckets'][bucket]
            if info['n'] == 0:
                continue
            print()
            print(f'per-pair detail — {bucket} (sorted by distance):')
            order = np.argsort(info['distances'])
            for j in order:
                d = info['distances'][j]
                eq1, eq2 = info['pairs'][j]
                pred = 'TRUE' if d < threshold else 'FALSE'
                mark = '✓' if pred == 'FALSE' else '✗'
                print(f'  Eq{eq1:>4} → Eq{eq2:<4}  d={d:.4f}  pred={pred:<5}  {mark}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
