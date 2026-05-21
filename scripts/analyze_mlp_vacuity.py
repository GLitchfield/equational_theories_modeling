"""Vacuity diagnostic for a trained MLP pair-classifier.

Backs Phase 4 of `mlp_classifier_plan.md`. Per Methodology Resolution 4,
the MLP "captured vacuity" iff both:
  (a) overall holdout bal_acc does not regress (≥ matched JEPA baseline)
  (b) MLP accuracy on always-wrong-trained ∩ holdout is high

This script computes (b) and the per-spine breakdown.

Inputs:
  - ``--checkpoint``     trained MLP checkpoint (from train_pair_classifier.py)
  - ``--embeddings-path``
  - ``--ids-path``
  - ``--always-wrong``   path to outputs/analyses/failure_overlap/always_wrong_trained.parquet
  - ``--split-path``     same 90/10 split used for training
  - ``--pair-table``
  - ``--invariants``     data/processed/equation_invariants.parquet (for spine breakdown)
  - ``--predictions``    optional path to a per-pair JEPA predictions.parquet (for the
                         matched-JEPA-on-same-subset baseline; if omitted, defaults to 0)

Outputs (in <output_dir>):
  - ``vacuity_eval.json``       headline metrics
  - prints a table to stdout
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings.src.model.pair_classifier import build_model  # noqa: E402

SPINE_LABELS = ['family_c', 'mixed', 'no_lhs_var_in_rhs', 'left', 'right', 'trivial']


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--embeddings-path',
                        default='outputs/runs/skip_h128_seed42_diag/equation_embeddings.npy')
    parser.add_argument('--ids-path',
                        default='outputs/runs/skip_h128_seed42_diag/equation_embeddings_ids.npy')
    parser.add_argument('--always-wrong',
                        default='outputs/analyses/failure_overlap/always_wrong_trained.parquet')
    parser.add_argument('--split-path',
                        default='data/splits/level1_holdout_seed42.parquet')
    parser.add_argument('--pair-table',
                        default='data/processed/implication_pairs.parquet')
    parser.add_argument('--invariants',
                        default='data/processed/equation_invariants.parquet')
    parser.add_argument('--predictions',
                        default='outputs/runs/skip_h128_seed42_diag/predictions.parquet',
                        help='JEPA predictions.parquet for matched-baseline comparison')
    parser.add_argument('--output-dir', default=None,
                        help='defaults to dirname(--checkpoint)')
    parser.add_argument('--eval-chunk', type=int, default=131072)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    if args.output_dir is None:
        args.output_dir = os.path.dirname(_abs(args.checkpoint))
    out_dir = _abs(args.output_dir)
    out_path = os.path.join(out_dir, 'vacuity_eval.json')
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- 1. load model ----------------------------------------------------
    print(f'loading checkpoint: {args.checkpoint}')
    ckpt = torch.load(_abs(args.checkpoint), weights_only=False, map_location='cpu')
    model_name = ckpt['model']
    embed_dim = ckpt['embed_dim']
    threshold = float(ckpt['threshold'])
    print(f'  model={model_name}  embed_dim={embed_dim}  threshold={threshold:.4f}')
    extra = {'dropout': ckpt.get('dropout', 0.1)} if model_name != 'linear' else {}
    model = build_model(model_name, embed_dim=embed_dim, **extra)
    model.load_state_dict(ckpt['state_dict'])
    model = model.to(device).eval()

    # --- 2. load embeddings + always-wrong subset + holdout filter -------
    Z = torch.from_numpy(np.load(_abs(args.embeddings_path)).astype(np.float32)).to(device)
    ids = np.load(_abs(args.ids_path))
    id_to_row = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    id_to_row[ids] = np.arange(len(ids), dtype=np.int64)
    print(f'  Z: {tuple(Z.shape)}')

    print(f'reading split:        {args.split_path}')
    holdout_idx = pq.read_table(_abs(args.split_path),
                                columns=['pair_idx'])['pair_idx'].to_numpy().astype(np.int64)
    print(f'  {len(holdout_idx):,} holdout indices')

    print(f'reading pair table:   {args.pair_table}')
    pair_tbl = pq.read_table(_abs(args.pair_table), columns=['eq1_id', 'eq2_id', 'value'])
    pair_eq1 = pair_tbl['eq1_id'].to_numpy()
    pair_eq2 = pair_tbl['eq2_id'].to_numpy()
    pair_val = pair_tbl['value'].to_numpy().astype(np.int8)
    n_pairs = len(pair_eq1)

    is_holdout = np.zeros(n_pairs, dtype=bool)
    is_holdout[holdout_idx] = True

    print(f'reading always-wrong: {args.always_wrong}')
    aw = pq.read_table(_abs(args.always_wrong)).to_pandas()
    aw_keys = set(zip(aw['eq1_id'].values.tolist(), aw['eq2_id'].values.tolist()))
    print(f'  {len(aw):,} always-wrong-trained pairs')

    # Find rows in the pair table that are (a) in the holdout and (b) in always-wrong-trained.
    print('intersecting always-wrong-trained with holdout ...')
    # Build a set of (eq1, eq2) for holdout subset for fast lookup, then mark always-wrong rows.
    # Faster: vectorize via pandas merge.
    aw_df = aw[['eq1_id', 'eq2_id']].assign(is_aw=np.int8(1))
    pair_df = pd.DataFrame({
        'eq1_id': pair_eq1, 'eq2_id': pair_eq2,
        'value': pair_val, 'is_holdout': is_holdout,
    })
    pair_df = pair_df.merge(aw_df, on=['eq1_id', 'eq2_id'], how='left')
    pair_df['is_aw'] = pair_df['is_aw'].fillna(0).astype(np.int8)

    aw_hold_mask = (pair_df['is_holdout'].values) & (pair_df['is_aw'].values == 1)
    n_aw_hold = int(aw_hold_mask.sum())
    print(f'  always-wrong-trained ∩ holdout: {n_aw_hold:,} pairs')

    # --- 3. evaluate MLP on subsets --------------------------------------

    def _score_subset(eq1_ids: np.ndarray, eq2_ids: np.ndarray) -> np.ndarray:
        e1 = torch.from_numpy(id_to_row[eq1_ids.astype(np.int64)]).to(device)
        e2 = torch.from_numpy(id_to_row[eq2_ids.astype(np.int64)]).to(device)
        out = torch.empty(len(e1), dtype=torch.float32, device=device)
        with torch.no_grad():
            for s in range(0, len(e1), args.eval_chunk):
                e = min(s + args.eval_chunk, len(e1))
                z1 = Z[e1[s:e]]
                z2 = Z[e2[s:e]]
                logit = model(z1, z2).squeeze(-1)
                out[s:e] = torch.sigmoid(logit)
        return out.cpu().numpy()

    aw_eq1 = pair_eq1[aw_hold_mask]
    aw_eq2 = pair_eq2[aw_hold_mask]
    aw_val = pair_val[aw_hold_mask]
    aw_scores = _score_subset(aw_eq1, aw_eq2)
    aw_pred = (aw_scores >= threshold).astype(np.int8)
    n_aw_true = int((aw_val == 1).sum())
    n_aw_false = int((aw_val == 0).sum())
    n_aw_correct = int((aw_pred == aw_val).sum())
    aw_recall_true = int(((aw_pred == 1) & (aw_val == 1)).sum()) / max(n_aw_true, 1)
    aw_recall_false = int(((aw_pred == 0) & (aw_val == 0)).sum()) / max(n_aw_false, 1)

    print(f'\n=== MLP on always-wrong-trained ∩ holdout ({n_aw_hold:,} pairs) ===')
    print(f'  TRUE pairs : {n_aw_true:,}   FALSE pairs: {n_aw_false:,}')
    print(f'  Accuracy   : {n_aw_correct / n_aw_hold:.4f}  ({n_aw_correct}/{n_aw_hold})')
    print(f'  TRUE recall : {aw_recall_true:.4f}')
    print(f'  FALSE recall: {aw_recall_false:.4f}')
    print(f'  Mean predicted prob: {aw_scores.mean():.4f}')

    # Per-spine breakdown (E2-spine, since vacuity is about E2 being weak/trivial)
    inv_df = None
    spine_rows = []
    if os.path.exists(_abs(args.invariants)):
        inv = pq.read_table(_abs(args.invariants)).to_pandas()
        # equation_invariants is indexed by eq_id (1..4694)
        inv_spine = inv['spine_type'].values
        aw_eq2_spine = inv_spine[aw_eq2 - 1]   # eq_id is 1-indexed
        for spine_idx in sorted(set(aw_eq2_spine.tolist())):
            mask_s = aw_eq2_spine == spine_idx
            n_s = int(mask_s.sum())
            n_correct_s = int((aw_pred[mask_s] == aw_val[mask_s]).sum())
            label = SPINE_LABELS[spine_idx] if 0 <= spine_idx < len(SPINE_LABELS) else f'unknown({spine_idx})'
            spine_rows.append({
                'spine': label,
                'n_pairs': n_s,
                'n_correct': n_correct_s,
                'accuracy': n_correct_s / max(n_s, 1),
                'mean_score': float(aw_scores[mask_s].mean()),
            })
        spine_rows.sort(key=lambda r: -r['n_pairs'])
        print(f'\n=== By E2 spine type ===')
        print(f'  {"spine":<22} {"n_pairs":>8} {"n_correct":>10} {"accuracy":>10} {"mean_score":>11}')
        for r in spine_rows:
            print(f'  {r["spine"]:<22} {r["n_pairs"]:>8,} {r["n_correct"]:>10,} {r["accuracy"]:>10.4f} {r["mean_score"]:>11.4f}')

    # --- 4. compare to JEPA baseline on this subset ----------------------
    jepa_on_aw = None
    if os.path.exists(_abs(args.predictions)):
        print(f'\nreading JEPA predictions: {args.predictions}')
        jepa_tbl = pq.read_table(
            _abs(args.predictions), columns=['eq1_id', 'eq2_id', 'predicted'],
        ).to_pandas()
        # Inner-join with our (eq1, eq2) keys for the aw-holdout subset.
        subset_df = pd.DataFrame({'eq1_id': aw_eq1, 'eq2_id': aw_eq2})
        m = subset_df.merge(jepa_tbl, on=['eq1_id', 'eq2_id'], how='left')
        jepa_pred = m['predicted'].values.astype(np.int8)
        n_jepa_correct = int((jepa_pred == aw_val).sum())
        jepa_on_aw = n_jepa_correct / max(n_aw_hold, 1)
        print(f'  JEPA accuracy on the same subset: {jepa_on_aw:.4f}  ({n_jepa_correct}/{n_aw_hold})')
        # This SHOULD be 0 by definition since this is the always-wrong-trained subset
        # and skip_h128_seed42 is one of the trained runs.
        if n_jepa_correct > 0:
            print(f'  WARNING: JEPA should be 0% on always-wrong-trained pairs, '
                  f'but got {n_jepa_correct} correct')

    # --- 5. write JSON summary -------------------------------------------
    summary = {
        'checkpoint': os.path.relpath(_abs(args.checkpoint), _REPO_ROOT),
        'always_wrong_path': os.path.relpath(_abs(args.always_wrong), _REPO_ROOT),
        'split_path': os.path.relpath(_abs(args.split_path), _REPO_ROOT),
        'predictions_path': os.path.relpath(_abs(args.predictions), _REPO_ROOT) if os.path.exists(_abs(args.predictions)) else None,
        'model': model_name,
        'threshold': threshold,
        'n_always_wrong_total': int(len(aw)),
        'n_always_wrong_holdout': n_aw_hold,
        'mlp_on_aw_holdout': {
            'accuracy': n_aw_correct / max(n_aw_hold, 1),
            'n_correct': n_aw_correct,
            'n_pairs': n_aw_hold,
            'true_recall': aw_recall_true,
            'false_recall': aw_recall_false,
            'n_true': n_aw_true,
            'n_false': n_aw_false,
            'mean_predicted_prob': float(aw_scores.mean()),
        },
        'jepa_on_aw_holdout_accuracy': jepa_on_aw,
        'per_spine': spine_rows,
    }

    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(f'\nwrote {out_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
