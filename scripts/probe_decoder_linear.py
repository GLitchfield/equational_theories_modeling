"""Linear-probe σ-prediction on concat(z_e1, z_e2).

Companion to the decoder phase-1 experiment. Train a single linear
classifier per substitution position (0..5) predicting σ from the
concat embedding alone. If a linear probe matches the 4.7M-param
transformer decoder, the witness is essentially linearly accessible
in the embedding geometry and the decoder is barely earning its
keep. If the linear probe gets much less, the decoder's depth is
doing real nonlinear work.

Same train/test splits as the decoder (equation-level holdout,
seed 42). σ-exact-match requires every position correct.

Output:
  outputs/runs/proof_decoder_linear_probe/eval_summary.json
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
import torch.nn as nn

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


# Substitution range: each position takes values in {-1 (sentinel), 0, 1, 2, 3, 4, 5}.
# We shift by +1 so the label space is {0..6} (7 classes).
N_POSITIONS = 6
N_CLASSES = 7   # sentinel + {0,1,2,3,4,5}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--embeddings-path',
                        default='outputs/embeddings/v1.0/embeddings.npy')
    parser.add_argument('--dataset',
                        default='data/processed/proof_corpus/decoder_phase1_dataset.parquet')
    parser.add_argument('--matcher-proofs',
                        default='data/processed/proof_corpus/matcher_proofs.parquet')
    parser.add_argument('--output',
                        default='outputs/runs/proof_decoder_linear_probe/eval_summary.json')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=1e-2)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    out_path = _abs(args.output)
    if os.path.exists(out_path) and not args.force:
        print(f'ERROR: {out_path} exists; pass --force.', file=sys.stderr)
        return 2

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    # --- 1. load data ---
    print('loading embeddings + dataset')
    emb = np.load(_abs(args.embeddings_path)).astype(np.float32)
    df = pq.read_table(_abs(args.dataset)).to_pandas()
    matcher = pq.read_table(
        _abs(args.matcher_proofs),
        columns=['eq1_id', 'eq2_id'] + [f'substitution_{i}' for i in range(6)],
    ).to_pandas()
    assert (df['eq1_id'].values[:5] == matcher['eq1_id'].values[:5]).all()
    subst = np.stack(
        [matcher[f'substitution_{i}'].to_numpy() for i in range(6)],
        axis=1,
    ).astype(np.int8)   # (N, 6)  values in {-1, 0..5}

    eq1_idx = df['eq1_id'].to_numpy() - 1
    eq2_idx = df['eq2_id'].to_numpy() - 1
    cond_np = np.concatenate([emb[eq1_idx], emb[eq2_idx]], axis=1)
    labels_np = (subst + 1).astype(np.int64)   # shift to {0..6}
    orient_np = df['orientation'].to_numpy().astype(np.int64)
    e1_arity_np = df['e1_arity'].to_numpy().astype(np.int64)

    split = df['split'].to_numpy()
    masks = {name: split == name for name in
             ('train', 'test_one_novel', 'test_both_novel')}
    for name, m in masks.items():
        print(f'  split {name}: {int(m.sum()):,}')

    cond = torch.from_numpy(cond_np).to(device)
    labels = torch.from_numpy(labels_np).to(device)              # (N, 6)
    orient = torch.from_numpy(orient_np).to(device)              # (N,)

    # --- 2. linear probe: position-wise classifier + orientation classifier ---
    # One Linear per σ position (6 of them); single Linear for orientation.
    # Each Linear has 1536 -> N_CLASSES (or 2 for orient).
    embed_dim = cond.shape[1]
    sigma_heads = nn.ModuleList([nn.Linear(embed_dim, N_CLASSES) for _ in range(N_POSITIONS)])
    orient_head = nn.Linear(embed_dim, 2)
    for h in sigma_heads: h.to(device)
    orient_head.to(device)

    params = list(orient_head.parameters())
    for h in sigma_heads:
        params += list(h.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    train_idx = np.where(masks['train'])[0]
    steps_per_epoch = math.ceil(len(train_idx) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps_per_epoch * args.epochs,
    )
    loss_fn = nn.CrossEntropyLoss()

    # --- 3. train ---
    print(f'training linear probe: {steps_per_epoch * args.epochs} steps')
    n_params = sum(p.numel() for p in params)
    print(f'  parameters: {n_params:,}')
    t_start = time.perf_counter()
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    train_idx_t = torch.from_numpy(train_idx).to(device)

    for epoch in range(args.epochs):
        for h in sigma_heads: h.train()
        orient_head.train()
        perm = train_idx_t[torch.randperm(len(train_idx_t), device=device, generator=g)]
        running_loss = 0.0
        n_b = 0
        for s in range(0, len(perm), args.batch_size):
            batch_idx = perm[s:s + args.batch_size]
            cond_b = cond[batch_idx]
            # σ loss: sum across positions.
            total_loss = torch.zeros((), device=device)
            for pos in range(N_POSITIONS):
                logits = sigma_heads[pos](cond_b)
                target = labels[batch_idx, pos]
                total_loss = total_loss + loss_fn(logits, target)
            # Orientation loss.
            o_logits = orient_head(cond_b)
            total_loss = total_loss + loss_fn(o_logits, orient[batch_idx])
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += float(total_loss)
            n_b += 1
        avg = running_loss / max(n_b, 1)
        print(f'  ep{epoch:>2}  total_loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}')

    wall = time.perf_counter() - t_start
    print(f'training done in {wall:.1f}s')

    # --- 4. evaluate on each split ---
    for h in sigma_heads: h.eval()
    orient_head.eval()

    def eval_split(name: str) -> dict:
        idx = np.where(masks[name])[0]
        if len(idx) == 0:
            return {'n': 0}
        idx_t = torch.from_numpy(idx).to(device)
        with torch.no_grad():
            cond_s = cond[idx_t]
            preds = torch.stack([
                sigma_heads[pos](cond_s).argmax(dim=-1)
                for pos in range(N_POSITIONS)
            ], dim=1)        # (n, 6)
            o_pred = orient_head(cond_s).argmax(dim=-1)
        target = labels[idx_t]
        # σ-exact: all 6 positions correct
        sigma_match = (preds == target).all(dim=1)
        # Orientation match
        orient_match = (o_pred == orient[idx_t])
        # Joint (σ + orientation correct, matching the decoder's strict σ-exact-match)
        joint = sigma_match & orient_match

        n = len(idx)
        result = {
            'n': int(n),
            'sigma_only_rate': float(sigma_match.float().mean()),
            'orient_only_rate': float(orient_match.float().mean()),
            'joint_sigma_rate': float(joint.float().mean()),
        }
        # Per-arity breakdown, including per-position accuracy AND the
        # independence-prediction `(joint)^(1/k)` so we can see how much
        # positions are correlated (measured > inverted = positively
        # correlated; positions share linear structure).
        per_arity: dict[int, dict] = {}
        e1_arr_split = torch.from_numpy(e1_arity_np[idx]).to(device)
        for k in range(2, 7):
            mask_k = (e1_arr_split == k)
            n_k = int(mask_k.sum())
            if n_k == 0:
                continue
            preds_k = preds[mask_k]
            target_k = target[mask_k]
            sigma_match_k = sigma_match[mask_k]
            joint_k = joint[mask_k]
            n_correct_joint = int(joint_k.sum())

            # Per-position accuracy ONLY over the actually-present positions
            # (positions 0..k-1; positions k..5 are always sentinel and
            # uninformative). Plus the sentinel positions' "correctness"
            # for completeness.
            per_pos = {}
            for pos in range(N_POSITIONS):
                acc = float((preds_k[:, pos] == target_k[:, pos]).float().mean())
                per_pos[pos] = acc

            # Per-position mean over the k real positions.
            real_pos_acc = sum(per_pos[p] for p in range(k)) / k
            # Independence-inversion estimate: (joint_sigma_rate)^(1/k).
            joint_rate = n_correct_joint / max(n_k, 1)
            inverted_per_pos = joint_rate ** (1.0 / k) if joint_rate > 0 else 0.0

            per_arity[k] = {
                'n': n_k,
                'correct_joint': n_correct_joint,
                'joint_sigma_rate': joint_rate,
                'per_position_accuracy': per_pos,
                'mean_real_position_accuracy': real_pos_acc,
                'inverted_per_position_from_joint': inverted_per_pos,
                'correlation_signal': real_pos_acc - inverted_per_pos,
            }
        result['per_arity'] = per_arity
        return result

    splits_result = {name: eval_split(name) for name in masks}

    summary = {
        'config': vars(args),
        'wall_clock_seconds': wall,
        'n_parameters': int(n_params),
        'splits': splits_result,
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)
    print(f'\nwrote {out_path}')

    print()
    print('=== Linear probe σ-exact-match (with orientation) ===')
    for name, m in splits_result.items():
        if m['n'] == 0:
            continue
        print(f'  {name:<18}: joint={m["joint_sigma_rate"]:.4f}  '
              f'σ_only={m["sigma_only_rate"]:.4f}  '
              f'orient={m["orient_only_rate"]:.4f}  '
              f'(n={m["n"]:,})')

    print()
    print('  Per-arity (joint σ-rate, by E1 arity):')
    for name in ('train', 'test_one_novel', 'test_both_novel'):
        m = splits_result[name]
        if not m.get('per_arity'):
            continue
        cells = []
        for k in sorted(m['per_arity']):
            v = m['per_arity'][k]
            cells.append(f'a{k}={v["joint_sigma_rate"]:.3f}({v["n"]})')
        print(f'    {name:<18}: {"  ".join(cells)}')

    print()
    print('  Per-arity measured per-position vs joint-inversion '
          '(positive ⇒ positions are positively correlated):')
    for name in ('train', 'test_one_novel', 'test_both_novel'):
        m = splits_result[name]
        if not m.get('per_arity'):
            continue
        print(f'    {name}')
        print(f'      arity   joint   measured/pos   inverted/pos   delta')
        for k in sorted(m['per_arity']):
            v = m['per_arity'][k]
            print(f'      a{k}      {v["joint_sigma_rate"]:.3f}   '
                  f'{v["mean_real_position_accuracy"]:.3f}          '
                  f'{v["inverted_per_position_from_joint"]:.3f}          '
                  f'{v["correlation_signal"]:+.3f}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
