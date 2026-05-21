"""Train a pair-classifier (linear / MLP) on frozen JEPA embeddings.

Backs Phase 2 (linear baseline) and Phase 3 (MLP variants) of
`mlp_classifier_plan.md`. The three model variants share this single
training loop:

    --model linear           Linear(2D, 1)         (Phase 2 linear probe)
    --model mlp_concat       MLP on concat         (Phase 3 variant A)
    --model mlp_features     MLP on concat+diff+had (Phase 3 variant B)

All run on the same 90/10 pair split (`level1_holdout_seed42.parquet`)
and the same per-equation embeddings (`equation_embeddings.npy`).
Embeddings and pair indices live on the GPU; an epoch is just a shuffle
of row indices plus a slice per batch — no DataLoader overhead.

Outputs (in ``--output-dir``):
  - ``best_checkpoint.pt``       best-bal_acc checkpoint (state dict + cfg)
  - ``per_epoch_eval.csv``       per-epoch train/holdout metrics
  - ``eval_summary.json``        best-epoch headline + matched-holdout metrics

Logs each epoch to stdout. wandb logging is optional via
``--wandb`` (group ``mlp_classifier``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from embeddings.src.model.pair_classifier import MODELS, build_model  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def _metrics(labels: torch.Tensor, preds: torch.Tensor) -> dict:
    n_true = int((labels == 1).sum().item())
    n_false = int((labels == 0).sum().item())
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    true_rec = tp / max(n_true, 1)
    false_rec = tn / max(n_false, 1)
    return {
        'bal_acc': 0.5 * (true_rec + false_rec),
        'true_recall': true_rec,
        'false_recall': false_rec,
        'n_true': n_true,
        'n_false': n_false,
        'n_correct': tp + tn,
        'n_wrong': int(len(labels)) - (tp + tn),
    }


def _calibrate_threshold(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Find sigmoid-prob threshold that maximizes bal_acc on (scores, labels).

    scores: probabilities in [0,1]; labels: 0/1 floats.
    """
    scores = scores.detach().cpu()
    labels = labels.detach().cpu().float()
    n = len(scores)
    if n == 0:
        return 0.5
    n_pos = float((labels == 1).sum())
    n_neg = float((labels == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = scores.argsort()  # ascending
    sorted_s = scores[order]
    sorted_l = labels[order]
    # If threshold = sorted_s[i], predict TRUE iff prob > threshold, so
    # n_pred_true = N - (i+1) and cum_tp = total_pos - prefix_pos[i].
    prefix_pos = sorted_l.cumsum(0)
    total_pos = prefix_pos[-1]
    n_pred_true = torch.arange(n - 1, -1, -1, dtype=torch.float)  # N-1, N-2, ..., 0
    cum_tp = total_pos - prefix_pos  # TPs strictly above sorted_s[i]
    fp = n_pred_true - cum_tp
    sens = cum_tp / n_pos
    spec = (n_neg - fp) / n_neg
    bal = 0.5 * (sens + spec)
    best_i = int(bal.argmax().item())
    if best_i + 1 < n:
        threshold = float((sorted_s[best_i] + sorted_s[best_i + 1]) / 2.0)
    else:
        threshold = float(sorted_s[best_i] + 1e-6)
    return threshold


def _score_in_chunks(
    model: torch.nn.Module,
    Z: torch.Tensor,
    eq1: torch.Tensor,
    eq2: torch.Tensor,
    chunk: int,
) -> torch.Tensor:
    """Run ``model(Z[eq1], Z[eq2])`` over all pairs in chunks of ``chunk``.

    Returns scores in [0,1] (sigmoid of the logit), shape (len(eq1),).
    """
    out = torch.empty(len(eq1), dtype=torch.float32, device=eq1.device)
    model.eval()
    with torch.no_grad():
        for s in range(0, len(eq1), chunk):
            e = min(s + chunk, len(eq1))
            z1 = Z[eq1[s:e]]
            z2 = Z[eq2[s:e]]
            logit = model(z1, z2).squeeze(-1)
            out[s:e] = torch.sigmoid(logit)
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--model', required=True, choices=sorted(MODELS),
                        help='which pair-classifier variant to train')
    parser.add_argument('--embeddings-path',
                        default='embeddings/outputs/runs/skip_h128_seed42_diag/equation_embeddings.npy')
    parser.add_argument('--ids-path',
                        default='embeddings/outputs/runs/skip_h128_seed42_diag/equation_embeddings_ids.npy')
    parser.add_argument('--pair-table',
                        default='data/processed/implication_pairs.parquet')
    parser.add_argument('--split-path',
                        default='data/splits/level1_holdout_seed42.parquet')
    parser.add_argument('--output-dir', required=True,
                        help='directory for checkpoint + per-epoch eval')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=0.0,
                        help='AdamW weight decay (default: %(default)s)')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='MLP dropout rate; ignored by --model linear')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-chunk', type=int, default=131072,
                        help='chunk size for holdout/train-score inference passes')
    parser.add_argument('--wandb', action='store_true', help='enable wandb logging')
    parser.add_argument('--wandb-name', default=None)
    parser.add_argument('--force', action='store_true', help='overwrite output files')
    args = parser.parse_args(argv)

    out_dir = _abs(args.output_dir)
    os.makedirs(out_dir, exist_ok=True)
    ckpt_path = os.path.join(out_dir, 'best_checkpoint.pt')
    csv_path = os.path.join(out_dir, 'per_epoch_eval.csv')
    summary_path = os.path.join(out_dir, 'eval_summary.json')
    for p in [ckpt_path, csv_path, summary_path]:
        if os.path.exists(p) and not args.force:
            print(f'ERROR: {p} exists; pass --force.', file=sys.stderr)
            return 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. load embeddings + split + pair table -------------------------
    print(f'loading embeddings: {args.embeddings_path}')
    Z_np = np.load(_abs(args.embeddings_path))
    ids = np.load(_abs(args.ids_path))
    id_to_row = np.full(int(ids.max()) + 1, -1, dtype=np.int64)
    id_to_row[ids] = np.arange(len(ids), dtype=np.int64)
    Z = torch.from_numpy(Z_np.astype(np.float32)).to(device)
    embed_dim = Z.shape[1]
    print(f'  Z: {tuple(Z.shape)}  device={Z.device}')

    print(f'reading pair table: {args.pair_table}')
    tbl = pq.read_table(_abs(args.pair_table), columns=['eq1_id', 'eq2_id', 'value'])
    eq1_arr = tbl['eq1_id'].to_numpy().astype(np.int32)
    eq2_arr = tbl['eq2_id'].to_numpy().astype(np.int32)
    val_arr = tbl['value'].to_numpy().astype(np.int8)
    n_pairs = len(eq1_arr)
    print(f'  {n_pairs:,} pairs')

    print(f'reading split:      {args.split_path}')
    holdout_idx = pq.read_table(_abs(args.split_path),
                                columns=['pair_idx'])['pair_idx'].to_numpy().astype(np.int64)
    is_holdout = np.zeros(n_pairs, dtype=bool)
    is_holdout[holdout_idx] = True
    train_idx = np.where(~is_holdout)[0]
    print(f'  train: {len(train_idx):,}   holdout: {is_holdout.sum():,}')

    # Map eq_id -> row in Z
    train_e1 = torch.from_numpy(id_to_row[eq1_arr[train_idx]]).to(device)
    train_e2 = torch.from_numpy(id_to_row[eq2_arr[train_idx]]).to(device)
    train_y = torch.from_numpy(val_arr[train_idx].astype(np.float32)).to(device)
    hold_e1 = torch.from_numpy(id_to_row[eq1_arr[holdout_idx]]).to(device)
    hold_e2 = torch.from_numpy(id_to_row[eq2_arr[holdout_idx]]).to(device)
    hold_y = torch.from_numpy(val_arr[holdout_idx].astype(np.float32)).to(device)

    n_true_train = int(train_y.sum().item())
    n_false_train = len(train_y) - n_true_train
    pos_weight = torch.tensor([n_false_train / n_true_train], device=device)
    print(f'  pos_weight = n_FALSE/n_TRUE = {n_false_train:,}/{n_true_train:,} = {pos_weight.item():.4f}')

    # --- 2. build model + optimizer + scheduler --------------------------
    if args.model == 'linear':
        model = build_model(args.model, embed_dim=embed_dim).to(device)
    else:
        model = build_model(args.model, embed_dim=embed_dim, dropout=args.dropout).to(device)
    print(f'  model: {args.model}  params: {sum(p.numel() for p in model.parameters()):,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    steps_per_epoch = math.ceil(len(train_y) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- 3. (optional) wandb ---------------------------------------------
    wandb_run = None
    if args.wandb:
        import wandb  # noqa: WPS433
        wandb_run = wandb.init(
            project='jepa-equational-implication',
            group='mlp_classifier',
            name=args.wandb_name or f'{args.model}_seed{args.seed}',
            config={
                'model': args.model,
                'embed_dim': embed_dim,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'weight_decay': args.weight_decay,
                'dropout': args.dropout,
                'seed': args.seed,
                'pos_weight': pos_weight.item(),
                'n_pairs_train': len(train_y),
                'n_pairs_holdout': len(hold_y),
                'split_path': args.split_path,
                'embeddings_path': args.embeddings_path,
            },
        )

    # --- 4. training loop -------------------------------------------------
    csv_rows = ['epoch,train_loss,train_bal_acc,train_true_R,train_false_R,'
                'holdout_bal_acc,holdout_true_R,holdout_false_R,threshold,wall_s']
    best = {'bal_acc': -1.0, 'epoch': -1}
    epoch_start_time = time.perf_counter()

    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    for epoch in range(args.epochs):
        model.train()
        # Shuffle on GPU
        perm = torch.randperm(len(train_y), device=device, generator=g)
        running_loss = 0.0
        for s in range(0, len(train_y), args.batch_size):
            idx = perm[s:s + args.batch_size]
            z1 = Z[train_e1[idx]]
            z2 = Z[train_e2[idx]]
            y = train_y[idx]
            logits = model(z1, z2).squeeze(-1)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * len(idx)
            if wandb_run is not None and (s // args.batch_size) % 50 == 0:
                wandb_run.log({
                    'train/loss': loss.item(),
                    'train/lr': scheduler.get_last_lr()[0],
                    'epoch': epoch + s / len(train_y),
                })
        avg_loss = running_loss / len(train_y)

        # ----- eval (train scores for threshold, holdout for headline) ----
        t_eval = time.perf_counter()
        train_scores = _score_in_chunks(model, Z, train_e1, train_e2, args.eval_chunk)
        hold_scores = _score_in_chunks(model, Z, hold_e1, hold_e2, args.eval_chunk)
        threshold = _calibrate_threshold(train_scores, train_y)
        train_pred = (train_scores >= threshold).to(torch.int8)
        hold_pred = (hold_scores >= threshold).to(torch.int8)
        train_metrics = _metrics(train_y.to(torch.int8), train_pred)
        hold_metrics = _metrics(hold_y.to(torch.int8), hold_pred)
        wall = time.perf_counter() - epoch_start_time

        print(f'ep{epoch:>2}  loss={avg_loss:.4f}  '
              f'train_bal={train_metrics["bal_acc"]:.4f}  '
              f'hold_bal={hold_metrics["bal_acc"]:.4f}  '
              f'TRUE_R={hold_metrics["true_recall"]:.4f}  '
              f'FALSE_R={hold_metrics["false_recall"]:.4f}  '
              f't={threshold:.4f}  wall={wall:.1f}s')
        csv_rows.append(
            f'{epoch},{avg_loss:.6f},{train_metrics["bal_acc"]:.6f},'
            f'{train_metrics["true_recall"]:.6f},{train_metrics["false_recall"]:.6f},'
            f'{hold_metrics["bal_acc"]:.6f},{hold_metrics["true_recall"]:.6f},'
            f'{hold_metrics["false_recall"]:.6f},{threshold:.6f},{wall:.1f}'
        )
        if wandb_run is not None:
            wandb_run.log({
                'epoch_end/epoch': epoch,
                'epoch_end/train_loss': avg_loss,
                'epoch_end/train_bal_acc': train_metrics['bal_acc'],
                'epoch_end/holdout_bal_acc': hold_metrics['bal_acc'],
                'epoch_end/holdout_true_recall': hold_metrics['true_recall'],
                'epoch_end/holdout_false_recall': hold_metrics['false_recall'],
                'epoch_end/threshold': threshold,
            })

        if hold_metrics['bal_acc'] > best['bal_acc']:
            best = {
                'epoch': epoch,
                'bal_acc': hold_metrics['bal_acc'],
                'true_recall': hold_metrics['true_recall'],
                'false_recall': hold_metrics['false_recall'],
                'threshold': threshold,
                'train_bal_acc': train_metrics['bal_acc'],
                'avg_loss': avg_loss,
            }
            torch.save({
                'state_dict': model.state_dict(),
                'model': args.model,
                'embed_dim': embed_dim,
                'dropout': args.dropout,
                'epoch': epoch,
                'threshold': threshold,
                'config': vars(args),
            }, ckpt_path)

    # --- 5. write per-epoch CSV + summary --------------------------------
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_rows) + '\n')

    summary = {
        'model': args.model,
        'embed_dim': embed_dim,
        'best_epoch': best['epoch'],
        'best_holdout_bal_acc': best['bal_acc'],
        'best_holdout_true_recall': best['true_recall'],
        'best_holdout_false_recall': best['false_recall'],
        'best_threshold': best['threshold'],
        'best_train_bal_acc': best['train_bal_acc'],
        'n_pairs_train': len(train_y),
        'n_pairs_holdout': len(hold_y),
        'pos_weight': pos_weight.item(),
        'config': vars(args),
        'checkpoint_path': os.path.relpath(ckpt_path, _REPO_ROOT),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    if wandb_run is not None:
        wandb_run.summary['best_holdout_bal_acc'] = best['bal_acc']
        wandb_run.summary['best_epoch'] = best['epoch']
        wandb_run.finish()

    print()
    print(f'best: ep{best["epoch"]}  bal_acc={best["bal_acc"]:.4f}  '
          f'TRUE_R={best["true_recall"]:.4f}  FALSE_R={best["false_recall"]:.4f}')
    print(f'wrote {ckpt_path}')
    print(f'      {csv_path}')
    print(f'      {summary_path}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
