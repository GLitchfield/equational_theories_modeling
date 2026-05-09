"""JEPA training loop.

Wires up data loaders, model, optimizer, schedulers, loss, and the EMA
target update. Logs every step to stdout and (optionally, gated by
``config['wandb']['enabled']``) to wandb. Saves a checkpoint at each
epoch boundary in non-sanity mode.

Phase 1 sanity goal: ≥80% balanced accuracy on the L1 holdout.

Public entry points:
  - ``train(config, sanity_check=False)``: run a full training; return a
    dict with the final-epoch metrics, the best (max-bal-acc) checkpoint
    path, the checkpoint dir, and the wandb run URL/ID if logging is on.
  - ``evaluate(jepa, loader, ...)``: compute loss + balanced-accuracy on
    a loader, optionally with a pre-supplied threshold (calibrates one
    on the loader if not given).
  - ``calibrate_threshold(distances, labels)``: find the threshold that
    maximizes balanced accuracy via an O(N log N) sweep over sorted
    distances.
"""

from __future__ import annotations

import os
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from src.data.dataset import make_loaders
from src.model.encoder import EquationEncoder
from src.model.jepa import JEPA
from src.model.predictor import ImplicationPredictor
from src.training.losses import contrastive_loss, effective_rank, vicreg_terms
from src.training.scheduler import cosine_tau, cosine_warmup_lr_multiplier


# --- threshold calibration / evaluation -------------------------------------

def calibrate_threshold(distances: torch.Tensor, labels: torch.Tensor) -> float:
    """Find the threshold that maximizes balanced accuracy.

    A pair is predicted TRUE when ``distance < threshold``. Sweep over all
    interior thresholds (boundaries between consecutive sorted distances)
    in O(N log N) total — argsort once, then a single linear pass with
    cumulative sums.
    """
    if distances.ndim != 1 or labels.ndim != 1:
        raise ValueError(
            f'distances and labels must be 1-D; got shapes '
            f'{tuple(distances.shape)}, {tuple(labels.shape)}'
        )
    if distances.shape != labels.shape:
        raise ValueError(
            f'shape mismatch: {tuple(distances.shape)} vs {tuple(labels.shape)}'
        )
    n = len(distances)
    if n == 0:
        return 0.0

    distances = distances.detach().cpu().float()
    labels = labels.detach().cpu().float()
    n_pos = int((labels == 1).sum().item())
    n_neg = int((labels == 0).sum().item())

    # Degenerate cases: only one class present. Return the median distance
    # (any threshold gives bal_acc = 0.5; the calibration is meaningless).
    if n_pos == 0 or n_neg == 0:
        return float(distances.median().item())

    sorted_idx = distances.argsort()
    sorted_d = distances[sorted_idx]
    sorted_l = labels[sorted_idx]

    # cum_tp[i] = TPs when threshold puts the first i+1 sorted pairs as TRUE
    cum_tp = (sorted_l == 1).float().cumsum(0)
    n_pred_true = torch.arange(1, n + 1, dtype=torch.float)
    fp = n_pred_true - cum_tp

    sens = cum_tp / n_pos
    spec = (n_neg - fp) / n_neg
    bal = 0.5 * (sens + spec)

    best_idx = int(bal.argmax().item())
    if best_idx + 1 < n:
        threshold = (sorted_d[best_idx] + sorted_d[best_idx + 1]) / 2.0
    else:
        threshold = sorted_d[best_idx] + 1e-6
    return float(threshold.item())


def evaluate(
    jepa: JEPA,
    loader,
    device: str,
    margin: float = 1.0,
    true_weight: float = 1.0,
    threshold: float | None = None,
    max_batches: int | None = None,
) -> dict:
    """Evaluate a JEPA on a DataLoader.

    Returns a dict of metrics: loss, balanced_accuracy, true_recall,
    false_recall, threshold (calibrated if not provided), n_evaluated.
    """
    was_training = jepa.training
    jepa.eval()
    all_d: list[torch.Tensor] = []
    all_l: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_idx, (eq1, eq2, labels) in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            eq1 = eq1.to(device)
            eq2 = eq2.to(device)
            labels = labels.to(device)
            z_pred, z_tgt = jepa(eq1, eq2)
            loss = contrastive_loss(
                z_pred, z_tgt, labels, margin=margin, true_weight=true_weight,
            )
            d = F.pairwise_distance(z_pred, z_tgt)
            all_d.append(d.cpu())
            all_l.append(labels.cpu())
            total_loss += loss.item()
            n_batches += 1

    if was_training:
        jepa.train()

    if n_batches == 0:
        return {
            'loss': 0.0, 'balanced_accuracy': 0.0,
            'true_recall': 0.0, 'false_recall': 0.0,
            'threshold': 0.0, 'n_evaluated': 0,
        }

    distances = torch.cat(all_d)
    labels = torch.cat(all_l)
    avg_loss = total_loss / n_batches

    if threshold is None:
        threshold = calibrate_threshold(distances, labels)

    preds = (distances < threshold).float()
    n_pos = max(int((labels == 1).sum().item()), 1)
    n_neg = max(int((labels == 0).sum().item()), 1)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    sens = tp / n_pos
    spec = tn / n_neg

    return {
        'loss': float(avg_loss),
        'balanced_accuracy': 0.5 * (sens + spec),
        'true_recall': sens,
        'false_recall': spec,
        'threshold': float(threshold),
        'n_evaluated': int(len(distances)),
    }


# --- logging --------------------------------------------------------------

def init_wandb(config: dict):
    """Initialize a wandb run if config['wandb']['enabled']; else None."""
    wcfg = config.get('wandb', {})
    if not wcfg.get('enabled', False):
        return None
    import wandb  # local import — only required when wandb is enabled
    run = wandb.init(
        project=wcfg['project'],
        entity=wcfg.get('entity'),
        name=wcfg.get('name'),
        config=config,
    )
    return run


def log_metrics(metrics: dict, run, step: int | None = None, prefix: str = '') -> None:
    """Print metrics to stdout and (if a run is provided) log to wandb."""
    parts = []
    for k, v in metrics.items():
        if isinstance(v, float):
            parts.append(f'{k}={v:.4f}')
        else:
            parts.append(f'{k}={v}')
    line = '  '.join(parts)
    print(f'{prefix} {line}'.strip())
    if run is not None:
        run.log(metrics, step=step)


# --- training loop --------------------------------------------------------

def _resolve_device(device_cfg: str) -> str:
    if device_cfg in ('auto', None, ''):
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return device_cfg


def _corpus_effective_ranks(
    context_encoder,
    graphs: dict,
    device: str,
    batch_size: int = 1024,
) -> tuple[float, float]:
    """Encode every unique equation through the context encoder and
    return ``(post_norm_eff_rank, pre_norm_eff_rank)`` over the corpus.

    Mirrors ``scripts/embedding_spectrum.py`` so the per-epoch metric is
    directly comparable to the offline diagnostic. Eval mode + no_grad
    so dropout is disabled and the autograd graph isn't held.
    """
    from torch_geometric.data import Batch

    was_training = context_encoder.training
    context_encoder.eval()
    eq_ids = sorted(graphs.keys())
    raw_chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(eq_ids), batch_size):
            chunk = [graphs[i] for i in eq_ids[start:start + batch_size]]
            batch = Batch.from_data_list(chunk).to(device)
            raw = context_encoder(batch, normalize=False).cpu()
            raw_chunks.append(raw)
    if was_training:
        context_encoder.train()
    z_raw = torch.cat(raw_chunks, dim=0)
    z_post = F.normalize(z_raw, dim=-1)
    return effective_rank(z_post), effective_rank(z_raw)


def _resolve_checkpoint_dir(config: dict) -> str:
    """Where to write epoch_NNN.pt for this run.

    Convention: ``<base>/<wandb.name>/`` so concurrent or sequential runs
    don't clobber each other. Falls back to a flat ``<base>/`` when no
    name is set, preserving legacy behavior for ad-hoc training calls.
    """
    base = config['training'].get('checkpoint_dir', 'outputs/checkpoints')
    name = config.get('wandb', {}).get('name')
    if name:
        return os.path.join(base, name)
    return base


def train(config: dict, sanity_check: bool = False) -> dict:
    """Run JEPA training. Returns the final epoch's metrics."""
    seed = int(config.get('seed', 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = _resolve_device(config.get('device', 'auto'))
    print(f'device: {device}')

    # --- data ---
    print('loading data ...')
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        pair_table_path=config['data']['pair_table_path'],
        graph_cache_path=config['data']['graph_cache_path'],
        holdout_path=config['data']['holdout_path'],
        batch_size=int(config['training']['batch_size']),
        num_workers=int(config['data'].get('num_workers', 0)),
        pin_memory=(device == 'cuda'),
    )
    print(f'  train: {len(train_ds):,} samples, {len(train_loader):,} batches')
    print(f'  val:   {len(val_ds):,} samples, {len(val_loader):,} batches')

    # --- model ---
    print('building model ...')
    enc_cfg = config['encoder']
    pred_cfg = config['predictor']
    encoder = EquationEncoder(
        node_features=enc_cfg['node_features'],
        edge_features=enc_cfg['edge_features'],
        hidden_dim=enc_cfg['hidden_dim'],
        embed_dim=enc_cfg['embed_dim'],
        num_layers=enc_cfg['num_layers'],
        num_heads=enc_cfg['num_heads'],
        dropout=enc_cfg['dropout'],
    )
    predictor = ImplicationPredictor(
        embed_dim=pred_cfg.get('embed_dim', enc_cfg['embed_dim']),
        hidden_dim=pred_cfg['hidden_dim'],
    )
    jepa = JEPA(encoder, predictor).to(device)
    print(f'  encoder:   {encoder.num_parameters():,} params')
    print(f'  predictor: {predictor.num_parameters():,} params')

    # --- optimizer (target encoder is frozen, exclude it) ---
    trainable_params = (
        list(jepa.context_encoder.parameters())
        + list(jepa.predictor.parameters())
    )
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=float(config['training']['learning_rate']),
        weight_decay=float(config['training']['weight_decay']),
    )

    # --- training schedule ---
    epochs = int(config['training']['epochs'])
    if sanity_check:
        epochs = 1
        max_train_batches = 50
        max_val_batches = 20
    else:
        max_train_batches = None
        max_val_batches = config['training'].get('max_val_batches')

    batches_per_epoch = len(train_loader) if max_train_batches is None else min(
        len(train_loader), max_train_batches
    )
    total_steps = batches_per_epoch * epochs

    base_lr = float(config['training']['learning_rate'])
    warmup_steps = int(config['training']['warmup_steps'])
    grad_clip = float(config['training']['grad_clip_norm'])
    margin = float(config['loss']['margin'])
    true_weight = float(config['loss']['true_weight'])
    vicreg_var_weight = float(config['loss'].get('vicreg_var_weight', 0.0))
    vicreg_cov_weight = float(config['loss'].get('vicreg_cov_weight', 0.0))
    vicreg_gamma = float(config['loss'].get('vicreg_gamma', 1.0))
    vicreg_active = (vicreg_var_weight > 0.0) or (vicreg_cov_weight > 0.0)
    tau_start = float(config['ema']['tau_start'])
    tau_final = float(config['ema']['tau_final'])
    log_every = int(config.get('wandb', {}).get('log_every', 50))

    # Cap warmup_steps to total_steps so the schedule stays well-defined.
    warmup_steps = min(warmup_steps, total_steps)

    # --- wandb ---
    run = init_wandb(config)

    print(
        f'training for {epochs} epoch(s), {batches_per_epoch:,} batches/epoch, '
        f'{total_steps:,} total steps  '
        f'(sanity_check={sanity_check}, wandb={"on" if run else "off"})'
    )

    step = 0
    t_start = time.perf_counter()
    last_metrics: dict = {}
    best_bal_acc = -float('inf')
    best_ckpt_path: str | None = None
    best_epoch_metrics: dict = {}
    ckpt_dir = _resolve_checkpoint_dir(config)

    for epoch in range(epochs):
        jepa.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        for batch_idx, (eq1, eq2, labels) in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            eq1 = eq1.to(device)
            eq2 = eq2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            if vicreg_active:
                z_pred, z_tgt, z_context_raw, z_tgt_raw = jepa(
                    eq1, eq2, return_encoder_raw=True,
                )
            else:
                z_pred, z_tgt = jepa(eq1, eq2)
            cont_loss = contrastive_loss(
                z_pred, z_tgt, labels, margin=margin, true_weight=true_weight,
            )
            loss = cont_loss
            if vicreg_active:
                # VICReg on the *encoder* pre-norm tensors: gives a direct
                # gradient into the context encoder, bypassing the
                # encoder→predictor L2-normalization bottleneck (under
                # which the unit-sphere total-variance-fixed-at-1
                # constraint makes gamma=1.0 unsatisfiable). Average over
                # the two encoder views; only z_context_raw contributes
                # gradient (z_tgt_raw is detached — target encoder picks
                # up diversity via EMA from the gradient-updated context
                # encoder).
                var_c, cov_c = vicreg_terms(z_context_raw, gamma=vicreg_gamma)
                var_t, cov_t = vicreg_terms(z_tgt_raw, gamma=vicreg_gamma)
                var_loss = 0.5 * (var_c + var_t)
                cov_loss = 0.5 * (cov_c + cov_t)
                loss = (
                    cont_loss
                    + vicreg_var_weight * var_loss
                    + vicreg_cov_weight * cov_loss
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)

            lr_mult = cosine_warmup_lr_multiplier(step, total_steps, warmup_steps)
            for g in optimizer.param_groups:
                g['lr'] = base_lr * lr_mult
            optimizer.step()

            tau = cosine_tau(step, total_steps, tau_start, tau_final)
            jepa.update_target(tau)

            epoch_loss_sum += loss.item()
            epoch_batches += 1

            if step % log_every == 0:
                step_metrics = {
                    'train/loss': loss.item(),
                    'train/contrastive_loss': cont_loss.item(),
                    'train/lr': base_lr * lr_mult,
                    'train/tau': tau,
                    'train/elapsed_sec': time.perf_counter() - t_start,
                }
                if vicreg_active:
                    step_metrics['train/vicreg_var'] = var_loss.item()
                    step_metrics['train/vicreg_cov'] = cov_loss.item()
                log_metrics(
                    step_metrics, run, step=step,
                    prefix=f'[ep{epoch} step{step}]',
                )
            step += 1

        avg_train_loss = epoch_loss_sum / max(epoch_batches, 1)
        print(f'\n[ep{epoch}] avg train loss {avg_train_loss:.4f}; evaluating ...')
        val = evaluate(
            jepa, val_loader, device,
            margin=margin, true_weight=true_weight,
            max_batches=max_val_batches,
        )

        # Corpus-wide effective rank: encode every unique equation in the
        # graph cache through the context encoder (eval mode, batched),
        # compute Roy-Vetterli effective rank on the resulting [N, embed_dim]
        # matrix for both pre- and post-L2-normalization. This is the
        # diagnostic from scripts/embedding_spectrum.py, run per epoch so
        # the collapse-resolve trajectory is visible in real time. Cheap
        # (~5 forward passes on 1024 graphs each on a 4080).
        eff_rank_corpus, eff_rank_corpus_raw = _corpus_effective_ranks(
            jepa.context_encoder, val_ds.graphs, device,
        )

        epoch_metrics = {
            'epoch': epoch,
            'train/avg_loss': avg_train_loss,
            'val/loss': val['loss'],
            'val/balanced_accuracy': val['balanced_accuracy'],
            'val/true_recall': val['true_recall'],
            'val/false_recall': val['false_recall'],
            'val/threshold': val['threshold'],
            'val/n_evaluated': val['n_evaluated'],
            'val/corpus_eff_rank': eff_rank_corpus,
            'val/corpus_eff_rank_raw': eff_rank_corpus_raw,
            'elapsed_sec': time.perf_counter() - t_start,
        }
        log_metrics(epoch_metrics, run, step=step, prefix=f'[ep{epoch} eval]')
        last_metrics = epoch_metrics

        if not sanity_check:
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f'epoch_{epoch:03d}.pt')
            torch.save({
                'jepa_state_dict': jepa.state_dict(),
                'epoch': epoch,
                'metrics': epoch_metrics,
                'config': config,
            }, ckpt_path)
            print(f'  saved checkpoint: {ckpt_path}')

            if val['balanced_accuracy'] > best_bal_acc:
                best_bal_acc = val['balanced_accuracy']
                best_ckpt_path = ckpt_path
                best_epoch_metrics = epoch_metrics

    wandb_url = getattr(run, 'url', None) if run is not None else None
    wandb_run_id = getattr(run, 'id', None) if run is not None else None
    if run is not None:
        run.finish()

    elapsed = time.perf_counter() - t_start
    print(f'\ntraining complete in {elapsed:.1f}s ({elapsed / 60:.1f} min)')
    return {
        'final_metrics': last_metrics,
        'best_checkpoint': best_ckpt_path,
        'best_epoch_metrics': best_epoch_metrics,
        'checkpoint_dir': ckpt_dir,
        'wandb_url': wandb_url,
        'wandb_run_id': wandb_run_id,
    }
