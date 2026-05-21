"""Phase 1.3 of `proof_decoder_plan.md` — train the proof decoder.

Trains the PrefixConditionedDecoder on the 36,339-pair training split
of `data/processed/proof_corpus/decoder_phase1_dataset.parquet`, with
either the v1.0 JEPA embeddings (real experiment) or seed-42 random
vectors (control). Tracks the nested metric stack (parse rate ->
arity-correct -> σ-exact-match) per epoch on three test subsets
(train, one-novel, both-novel), saves the best-by-both-novel-σ
checkpoint.

Run twice — once per embedding source — to produce both the real
experiment and the control:

    python scripts/train_decoder.py \
        --embedding-source jepa \
        --output-dir outputs/runs/proof_decoder_jepa

    python scripts/train_decoder.py \
        --embedding-source random \
        --output-dir outputs/runs/proof_decoder_random_ctl
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

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from et_studies.proof_decoder.code.decoder import PrefixConditionedDecoder, loss_for_batch  # noqa: E402
from et_studies.proof_decoder.code.eval_helpers import TripleMetrics, parse_proof_tokens  # noqa: E402
from et_studies.proof_decoder.code.tokenizer import BOS_ID, EOS_ID, PAD_ID  # noqa: E402


def _abs(p: str) -> str:
    return p if os.path.isabs(p) else os.path.join(_REPO_ROOT, p)


def load_embeddings(source: str, embeddings_path: str, n_eq: int, embed_dim: int = 768) -> np.ndarray:
    """Return ``[n_eq, embed_dim]`` float32 embeddings, either the v1.0 JEPA
    embeddings or seed-42 frozen random vectors (L2-normalized to match
    the JEPA's unit-sphere geometry)."""
    if source == 'jepa':
        emb = np.load(_abs(embeddings_path)).astype(np.float32)
        if emb.shape != (n_eq, embed_dim):
            raise ValueError(f'embeddings shape {emb.shape} != ({n_eq}, {embed_dim})')
        return emb
    if source == 'random':
        rng = np.random.default_rng(42)
        emb = rng.standard_normal((n_eq, embed_dim)).astype(np.float32)
        emb /= np.linalg.norm(emb, axis=1, keepdims=True)
        return emb
    raise ValueError(f'unknown embedding source: {source!r}')


def build_tensors(
    dataset_path: str,
    matcher_proofs_path: str,
    emb: np.ndarray,
    device: torch.device,
    max_token_len: int,
) -> dict:
    """Load + tensorize. Returns dict with per-split tensors and per-split
    gt metadata for the metric stack."""
    print(f'loading dataset from {dataset_path}')
    df = pq.read_table(_abs(dataset_path)).to_pandas()
    print(f'  {len(df):,} rows')

    print(f'loading matcher proofs for substitution columns')
    matcher = pq.read_table(
        _abs(matcher_proofs_path),
        columns=['eq1_id', 'eq2_id'] + [f'substitution_{i}' for i in range(6)],
    ).to_pandas()
    # The dataset was built from matcher_proofs in row order, so they align.
    # Sanity check on a few rows.
    assert (df['eq1_id'].values[:5] == matcher['eq1_id'].values[:5]).all()
    assert (df['eq2_id'].values[:5] == matcher['eq2_id'].values[:5]).all()
    subst = np.stack([matcher[f'substitution_{i}'].to_numpy() for i in range(6)], axis=1).astype(np.int8)

    # Build cond from embeddings — concat(z_e1, z_e2).
    eq1_idx = df['eq1_id'].to_numpy() - 1
    eq2_idx = df['eq2_id'].to_numpy() - 1
    cond_np = np.concatenate([emb[eq1_idx], emb[eq2_idx]], axis=1)   # (N, 1536)

    # Pad tokens to max_token_len.
    n = len(df)
    tokens_np = np.full((n, max_token_len), PAD_ID, dtype=np.int64)
    for i, t in enumerate(df['tokens'].values):
        seq = list(t)
        if len(seq) > max_token_len:
            raise ValueError(f'sequence length {len(seq)} exceeds max_token_len {max_token_len}')
        tokens_np[i, :len(seq)] = seq

    cond_t = torch.from_numpy(cond_np).to(device)
    tokens_t = torch.from_numpy(tokens_np).to(device)
    orient = df['orientation'].to_numpy().astype(np.int8)
    e1_arity = df['e1_arity'].to_numpy().astype(np.int8)
    e2_arity = df['e2_arity'].to_numpy().astype(np.int8)

    # Per-split slicing.
    split_arr = df['split'].to_numpy()
    splits: dict[str, dict] = {}
    for name in ('train', 'test_one_novel', 'test_both_novel'):
        mask = split_arr == name
        idx = np.where(mask)[0]
        splits[name] = {
            'idx': idx,
            'cond': cond_t[idx],
            'tokens': tokens_t[idx],
            'orient': orient[idx],
            'e1_arity': e1_arity[idx],
            'e2_arity': e2_arity[idx],
            'subst': subst[idx],
        }
        print(f'  split {name}: {len(idx):,} pairs')

    return splits


def evaluate(
    model: PrefixConditionedDecoder,
    split: dict,
    batch_size: int,
    max_new_tokens: int,
    return_per_arity: bool = False,
) -> dict:
    """Generate over a split and compute the nested metric stack."""
    model.eval()
    cond = split['cond']
    n = cond.shape[0]
    if n == 0:
        return {'parse_rate': 0.0, 'arity_rate': 0.0, 'sigma_rate': 0.0, 'n': 0}

    n_parsed = 0
    n_arity_ok = 0
    n_sigma_ok = 0
    per_arity: dict[int, dict[str, int]] = {}

    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            cond_b = cond[s:e]
            gen = model.generate(cond_b, max_new_tokens=max_new_tokens, greedy=True)
            gen_cpu = gen.cpu().tolist()
            for j in range(e - s):
                i = s + j
                gt_e1 = int(split['e1_arity'][i])
                gt_e2 = int(split['e2_arity'][i])
                gt_sigma = tuple(int(x) for x in split['subst'][i, :gt_e1].tolist())
                parsed = parse_proof_tokens(gen_cpu[j])
                m = TripleMetrics.from_parse(
                    parsed,
                    gt_orientation=int(split['orient'][i]),
                    gt_e1_arity=gt_e1,
                    gt_e2_arity=gt_e2,
                    gt_substitution=gt_sigma,
                )
                if m.parsed:
                    n_parsed += 1
                if m.arity_correct:
                    n_arity_ok += 1
                if m.sigma_exact:
                    n_sigma_ok += 1
                if return_per_arity:
                    bucket = per_arity.setdefault(gt_e1, {'n': 0, 'parsed': 0, 'arity': 0, 'sigma': 0})
                    bucket['n'] += 1
                    if m.parsed: bucket['parsed'] += 1
                    if m.arity_correct: bucket['arity'] += 1
                    if m.sigma_exact: bucket['sigma'] += 1

    out = {
        'n': n,
        'n_parsed': n_parsed,
        'n_arity_correct': n_arity_ok,
        'n_sigma_exact': n_sigma_ok,
        'parse_rate': n_parsed / n,
        'arity_rate': n_arity_ok / n,
        'sigma_rate': n_sigma_ok / n,
    }
    if return_per_arity:
        out['per_arity'] = {
            k: {**v, 'sigma_rate': v['sigma'] / max(v['n'], 1)}
            for k, v in sorted(per_arity.items())
        }
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--embedding-source', choices=['jepa', 'random'], required=True)
    parser.add_argument('--embeddings-path',
                        default='outputs/embeddings/v1.0/embeddings.npy')
    parser.add_argument('--dataset',
                        default='data/processed/proof_corpus/decoder_phase1_dataset.parquet')
    parser.add_argument('--matcher-proofs',
                        default='data/processed/proof_corpus/matcher_proofs.parquet')
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight-decay', type=float, default=0.0)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--max-token-len', type=int, default=20)
    parser.add_argument('--max-new-tokens', type=int, default=20,
                        help='max generation length at eval (caps autoregressive rollout)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--eval-every', type=int, default=1)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb-name', default=None)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args(argv)

    out_dir = _abs(args.output_dir)
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

    # --- 1. embeddings ---
    print(f'loading {args.embedding_source} embeddings')
    n_eq = 4694
    emb = load_embeddings(args.embedding_source, args.embeddings_path, n_eq, embed_dim=768)
    print(f'  shape {emb.shape}  embedding source: {args.embedding_source}')

    # --- 2. tensors ---
    splits = build_tensors(args.dataset, args.matcher_proofs, emb,
                           torch.device(device), args.max_token_len)

    # --- 3. model + optimizer + scheduler ---
    model = PrefixConditionedDecoder(
        d_model=256, nhead=4, num_layers=4, num_prefix=4,
        embed_dim=2 * 768, dropout=args.dropout,
        max_token_len=args.max_token_len,
    ).to(device)
    print(f'  decoder params: {model.num_parameters():,}')

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    steps_per_epoch = math.ceil(len(splits['train']['tokens']) / args.batch_size)
    total_steps = steps_per_epoch * args.epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    # --- 4. (optional) wandb ---
    wandb_run = None
    if args.wandb:
        import wandb  # noqa: WPS433
        wandb_run = wandb.init(
            project='jepa-equational-implication',
            group='proof_decoder',
            name=args.wandb_name or f'decoder_{args.embedding_source}_seed{args.seed}',
            config=vars(args),
        )

    # --- 5. training loop ---
    csv_rows = [
        'epoch,train_loss,lr,'
        'train_sigma,train_arity,train_parse,'
        'one_novel_sigma,one_novel_arity,one_novel_parse,'
        'both_novel_sigma,both_novel_arity,both_novel_parse,wall_s'
    ]
    best = {'epoch': -1, 'both_novel_sigma': -1.0}
    g = torch.Generator(device=device)
    g.manual_seed(args.seed)
    t_start = time.perf_counter()

    train_cond = splits['train']['cond']
    train_tokens = splits['train']['tokens']

    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(train_tokens), device=device, generator=g)
        running_loss = 0.0
        n_b = 0
        for s in range(0, len(train_tokens), args.batch_size):
            idx = perm[s:s + args.batch_size]
            cond_b = train_cond[idx]
            tok_b = train_tokens[idx]
            # Inputs = tokens[:, :-1]; targets = tokens[:, 1:]
            logits = model(cond_b, tok_b[:, :-1])
            loss = loss_for_batch(logits, tok_b[:, 1:])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            running_loss += loss.item()
            n_b += 1
        avg_loss = running_loss / max(n_b, 1)
        cur_lr = scheduler.get_last_lr()[0]

        # eval
        if (epoch + 1) % args.eval_every == 0 or epoch == args.epochs - 1:
            train_m = evaluate(model, splits['train'], args.batch_size, args.max_new_tokens)
            one_m = evaluate(model, splits['test_one_novel'], args.batch_size, args.max_new_tokens)
            both_m = evaluate(model, splits['test_both_novel'], args.batch_size, args.max_new_tokens)
        else:
            train_m = one_m = both_m = {'sigma_rate': float('nan'), 'arity_rate': float('nan'), 'parse_rate': float('nan')}
        wall = time.perf_counter() - t_start

        print(
            f'ep{epoch:>2}  loss={avg_loss:.4f}  lr={cur_lr:.2e}  '
            f'train_σ={train_m["sigma_rate"]:.4f}  '
            f'one_σ={one_m["sigma_rate"]:.4f}  '
            f'both_σ={both_m["sigma_rate"]:.4f}  '
            f'wall={wall:.1f}s'
        )
        csv_rows.append(
            f'{epoch},{avg_loss:.6f},{cur_lr:.6e},'
            f'{train_m["sigma_rate"]:.6f},{train_m["arity_rate"]:.6f},{train_m["parse_rate"]:.6f},'
            f'{one_m["sigma_rate"]:.6f},{one_m["arity_rate"]:.6f},{one_m["parse_rate"]:.6f},'
            f'{both_m["sigma_rate"]:.6f},{both_m["arity_rate"]:.6f},{both_m["parse_rate"]:.6f},'
            f'{wall:.1f}'
        )
        if wandb_run is not None:
            wandb_run.log({
                'epoch': epoch, 'train_loss': avg_loss, 'lr': cur_lr,
                **{f'train/{k}': v for k, v in train_m.items() if isinstance(v, float)},
                **{f'one_novel/{k}': v for k, v in one_m.items() if isinstance(v, float)},
                **{f'both_novel/{k}': v for k, v in both_m.items() if isinstance(v, float)},
            })

        if both_m['sigma_rate'] > best['both_novel_sigma']:
            best = {
                'epoch': epoch,
                'both_novel_sigma': both_m['sigma_rate'],
                'one_novel_sigma': one_m['sigma_rate'],
                'train_sigma': train_m['sigma_rate'],
                'avg_loss': avg_loss,
            }
            os.makedirs(out_dir, exist_ok=True)
            torch.save({
                'state_dict': model.state_dict(),
                'config': vars(args),
                'epoch': epoch,
                'metrics': {'train': train_m, 'one_novel': one_m, 'both_novel': both_m},
            }, ckpt_path)

    # --- 6. final per-arity report on best checkpoint ---
    print('reloading best checkpoint for per-arity breakdown ...')
    ckpt = torch.load(ckpt_path, weights_only=False, map_location=device)
    model.load_state_dict(ckpt['state_dict'])

    final_metrics = {
        split_name: evaluate(
            model, splits[split_name], args.batch_size, args.max_new_tokens,
            return_per_arity=True,
        )
        for split_name in ('train', 'test_one_novel', 'test_both_novel')
    }

    os.makedirs(out_dir, exist_ok=True)
    with open(csv_path, 'w') as f:
        f.write('\n'.join(csv_rows) + '\n')

    summary = {
        'config': vars(args),
        'best_epoch': best['epoch'],
        'best_both_novel_sigma': best['both_novel_sigma'],
        'best_one_novel_sigma': best['one_novel_sigma'],
        'best_train_sigma': best['train_sigma'],
        'best_avg_loss': best['avg_loss'],
        'final_metrics': final_metrics,
        'checkpoint_path': os.path.relpath(ckpt_path, _REPO_ROOT),
    }
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=False)

    if wandb_run is not None:
        wandb_run.summary['best_both_novel_sigma'] = best['both_novel_sigma']
        wandb_run.summary['best_epoch'] = best['epoch']
        wandb_run.finish()

    print()
    print('=== Final ===')
    print(f"  best epoch       : {best['epoch']}")
    print(f"  train_σ          : {best['train_sigma']:.4f}")
    print(f"  one_novel_σ      : {best['one_novel_sigma']:.4f}")
    print(f"  both_novel_σ     : {best['both_novel_sigma']:.4f}")
    print(f"  wrote {ckpt_path}")
    print(f"        {csv_path}")
    print(f"        {summary_path}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
