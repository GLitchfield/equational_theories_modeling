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

from lib.data.dataset import make_loaders
from embeddings.src.model.aux_heads import (
    AuxiliaryHeads,
    HEAD_SPECS as AUX_HEAD_SPECS,
    resolve_head_names,
    auxiliary_accuracy,
    auxiliary_loss,
)
from embeddings.src.model.encoder import EquationEncoder
from embeddings.src.model.jepa import JEPA
from embeddings.src.model.predictor import ImplicationPredictor
from embeddings.src.training.losses import contrastive_loss, effective_rank, vicreg_terms
from embeddings.src.training.sigreg import sigreg
from embeddings.src.training.scheduler import cosine_tau, cosine_warmup_lr_multiplier


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
    aux_heads_module: AuxiliaryHeads | None = None,
) -> dict:
    """Evaluate a JEPA on a DataLoader.

    Returns a dict of metrics: loss, balanced_accuracy, true_recall,
    false_recall, threshold (calibrated if not provided), n_evaluated.

    If ``aux_heads_module`` is provided, also computes per-head argmax
    accuracy on eq1 (sufficient for per-epoch monitoring; eq2 would be
    redundant since both are processed by the same encoder). The
    loader must be aux-active for this to work — i.e. yielding 5-tuples.
    """
    was_training = jepa.training
    jepa.eval()
    if aux_heads_module is not None:
        aux_was_training = aux_heads_module.training
        aux_heads_module.eval()
    all_d: list[torch.Tensor] = []
    all_l: list[torch.Tensor] = []
    total_loss = 0.0
    n_batches = 0
    # For per-head accuracy: stash predictions and labels per ACTIVE head
    # (the loader provides labels for all 18 heads regardless; we only
    # log accuracy for the heads being trained).
    active_head_names = aux_heads_module.head_names if aux_heads_module is not None else []
    aux_preds: dict[str, list[torch.Tensor]] = {n: [] for n in active_head_names}
    aux_labels: dict[str, list[torch.Tensor]] = {n: [] for n in active_head_names}

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            if aux_heads_module is not None and len(batch) == 5:
                eq1, eq2, labels, eq1_aux, _eq2_aux = batch
            else:
                eq1, eq2, labels = batch[:3]
            eq1 = eq1.to(device)
            eq2 = eq2.to(device)
            labels = labels.to(device)
            if aux_heads_module is not None:
                z_pred, z_tgt, pooled_eq1, _ = jepa(eq1, eq2, return_pooled=True)
                logits_e1 = aux_heads_module(pooled_eq1)
                for name in active_head_names:
                    aux_preds[name].append(logits_e1[name].argmax(dim=-1).cpu())
                    aux_labels[name].append(eq1_aux[name].cpu())
            else:
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
    if aux_heads_module is not None and aux_was_training:
        aux_heads_module.train()

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

    # Per-class distance distribution diagnostics. Tracking these lets us
    # see whether the correct-TRUE / correct-FALSE distance populations
    # are pulling apart over training (the explicit goal of widening the
    # contrastive margin) or staying overlapped near the threshold.
    true_mask = labels == 1
    false_mask = labels == 0
    true_d = distances[true_mask]
    false_d = distances[false_mask]
    pcts = torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95])
    if len(true_d) > 0:
        td_q = torch.quantile(true_d, pcts).tolist()
    else:
        td_q = [0.0] * 5
    if len(false_d) > 0:
        fd_q = torch.quantile(false_d, pcts).tolist()
    else:
        fd_q = [0.0] * 5
    # Overlap proxy: fraction of TRUE pairs whose distance is above the
    # FALSE-class median, AND fraction of FALSE pairs below the TRUE-class
    # median. Both are zero in a perfectly separated geometry; both
    # approach 0.5 if the classes are indistinguishable.
    fd_median = fd_q[2] if len(false_d) > 0 else 0.0
    td_median = td_q[2] if len(true_d) > 0 else 0.0
    true_above_false_med = float((true_d > fd_median).float().mean().item()) if len(true_d) > 0 else 0.0
    false_below_true_med = float((false_d < td_median).float().mean().item()) if len(false_d) > 0 else 0.0

    metrics = {
        'loss': float(avg_loss),
        'balanced_accuracy': 0.5 * (sens + spec),
        'true_recall': sens,
        'false_recall': spec,
        'threshold': float(threshold),
        'n_evaluated': int(len(distances)),
        'dist_true_p05': td_q[0],
        'dist_true_p25': td_q[1],
        'dist_true_p50': td_q[2],
        'dist_true_p75': td_q[3],
        'dist_true_p95': td_q[4],
        'dist_false_p05': fd_q[0],
        'dist_false_p25': fd_q[1],
        'dist_false_p50': fd_q[2],
        'dist_false_p75': fd_q[3],
        'dist_false_p95': fd_q[4],
        'dist_gap_median': fd_median - td_median,
        'dist_overlap_true_above_false_med': true_above_false_med,
        'dist_overlap_false_below_true_med': false_below_true_med,
    }
    # Per-head argmax accuracy on the eq1 side of every val batch.
    # Logged as 'aux_acc_<head_name>' so it sits alongside the val/dist_*
    # diagnostics in the per-epoch eval line. Only the active heads
    # (the ones the model was trained on) are reported.
    if aux_heads_module is not None and aux_preds:
        for name in aux_preds:
            preds_cat = torch.cat(aux_preds[name])
            labels_cat = torch.cat(aux_labels[name])
            metrics[f'aux_acc_{name}'] = float((preds_cat == labels_cat).float().mean())
    return metrics


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


def _evaluate_channel_mask(
    jepa: JEPA,
    loader,
    device: str,
    mask: str,
    threshold: float,
    margin: float,
    true_weight: float,
    max_batches: int | None = 50,
) -> dict:
    """Skip-pooled eval with one channel zeroed out and re-normalized.

    For diagnostic only — the renorm puts the kept half on a different
    unit sphere than the full embedding, so the bal_acc here is NOT
    directly comparable to the main eval. Track per-epoch trends, not
    absolute values.

    Args:
        mask: ``"proj_only"`` (zero the skip half) or ``"skip_only"``
            (zero the proj half).
        threshold: distance threshold from the main eval (so we use a
            consistent decision boundary, not a re-calibrated one).
        max_batches: cap (default 50) — diagnostic only, no need to run
            the full val set.
    """
    encoder = jepa.context_encoder
    target_encoder = jepa.target_encoder
    if not encoder.skip_pooled:
        raise ValueError('channel_mask eval only valid for skip_pooled encoders')
    proj_d = encoder.proj_dim
    pool_d = encoder.pool_dim

    def _mask_and_renorm(z_raw: torch.Tensor) -> torch.Tensor:
        z = z_raw.clone()
        if mask == 'proj_only':
            z[:, proj_d:] = 0
        elif mask == 'skip_only':
            z[:, :proj_d] = 0
        else:
            raise ValueError(f'unknown mask {mask!r}')
        return F.normalize(z, dim=-1)

    was_training = jepa.training
    jepa.eval()
    all_d: list[torch.Tensor] = []
    all_l: list[torch.Tensor] = []
    n_batches = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if max_batches is not None and batch_idx >= max_batches:
                break
            eq1, eq2, labels = batch[:3]
            eq1 = eq1.to(device)
            eq2 = eq2.to(device)
            labels = labels.to(device)
            z_ctx_raw = encoder(eq1, normalize=False)
            z_tgt_raw = target_encoder(eq2, normalize=False)
            z_ctx = _mask_and_renorm(z_ctx_raw)
            z_tgt = _mask_and_renorm(z_tgt_raw)
            z_pred = jepa.predictor(z_ctx)
            d = F.pairwise_distance(z_pred, z_tgt)
            all_d.append(d.cpu())
            all_l.append(labels.cpu())
            n_batches += 1
    if was_training:
        jepa.train()

    if n_batches == 0:
        return {'balanced_accuracy': 0.0, 'true_recall': 0.0, 'false_recall': 0.0,
                'n_evaluated': 0}

    distances = torch.cat(all_d)
    labels = torch.cat(all_l)
    preds = (distances < threshold).float()
    n_pos = max(int((labels == 1).sum().item()), 1)
    n_neg = max(int((labels == 0).sum().item()), 1)
    tp = int(((preds == 1) & (labels == 1)).sum().item())
    tn = int(((preds == 0) & (labels == 0)).sum().item())
    sens = tp / n_pos
    spec = tn / n_neg
    return {
        'balanced_accuracy': 0.5 * (sens + spec),
        'true_recall': sens,
        'false_recall': spec,
        'n_evaluated': int(len(distances)),
    }


def _corpus_effective_ranks(
    context_encoder,
    graphs: dict,
    device: str,
    batch_size: int = 1024,
) -> dict[str, float]:
    """Encode every unique equation through the context encoder and
    return effective-rank metrics over the corpus.

    Always returns ``post_norm`` (the L2-normalized full embedding's
    rank, matching what the contrastive loss sees) and ``pre_norm``
    (the un-normalized full embedding's rank). When the encoder is in
    skip-pooled mode, also returns per-channel ranks ``proj`` and
    ``skip`` (slices of the pre-norm embedding) so we can detect
    one-channel collapse.

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
    out = {
        'post_norm': float(effective_rank(z_post)),
        'pre_norm': float(effective_rank(z_raw)),
    }
    if getattr(context_encoder, 'skip_pooled', False):
        proj_d = context_encoder.proj_dim
        out['proj'] = float(effective_rank(z_raw[:, :proj_d]))
        out['skip'] = float(effective_rank(z_raw[:, proj_d:]))
    return out


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
    holdout_path = config['data'].get('holdout_path')
    if holdout_path == '' or (isinstance(holdout_path, str) and holdout_path.lower() == 'none'):
        holdout_path = None
    aux_labels_path = config['data'].get('aux_labels_path') or None
    aux_weight = float(config['loss'].get('aux_weight', 0.0))
    aux_active = aux_weight > 0.0
    if aux_active and not aux_labels_path:
        raise ValueError(
            'loss.aux_weight > 0 but data.aux_labels_path is unset; '
            'aux heads need a label parquet (build with '
            'scripts/build_equation_invariants.py).'
        )
    # Optional subset of HEAD_SPECS to actually train. Accepts a string
    # (named subset like "structural" / "cohom" / "all") or an explicit
    # list of head names. Defaults to all heads (backward compatible).
    aux_heads_spec = config['loss'].get('aux_heads', None)
    aux_head_names = resolve_head_names(aux_heads_spec) if aux_active else []
    train_loader, val_loader, train_ds, val_ds = make_loaders(
        pair_table_path=config['data']['pair_table_path'],
        graph_cache_path=config['data']['graph_cache_path'],
        holdout_path=holdout_path,
        batch_size=int(config['training']['batch_size']),
        num_workers=int(config['data'].get('num_workers', 0)),
        pin_memory=(device == 'cuda'),
        aux_labels_path=aux_labels_path if aux_active else None,
    )
    print(f'  train: {len(train_ds):,} samples, {len(train_loader):,} batches')
    if holdout_path is None:
        print(f'  val (no-holdout monitor subset): {len(val_ds):,} samples, '
              f'{len(val_loader):,} batches')
    else:
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
        skip_pooled=enc_cfg.get('skip_pooled', False),
        proj_dim=enc_cfg.get('proj_dim', None),
    )
    predictor = ImplicationPredictor(
        embed_dim=pred_cfg.get('embed_dim', enc_cfg['embed_dim']),
        hidden_dim=pred_cfg['hidden_dim'],
    )
    jepa = JEPA(encoder, predictor).to(device)
    print(f'  encoder:   {encoder.num_parameters():,} params')
    print(f'  predictor: {predictor.num_parameters():,} params')

    # Auxiliary heads probe the encoder's pooled (mean+max) representation,
    # which is 2 * hidden_dim wide. Built only when aux_active so unset
    # configs cost nothing.
    aux_heads_module: AuxiliaryHeads | None = None
    if aux_active:
        pool_dim = 2 * int(enc_cfg['hidden_dim'])
        aux_heads_module = AuxiliaryHeads(
            pool_dim=pool_dim, head_names=aux_head_names,
        ).to(device)
        print(f'  aux_heads: {sum(p.numel() for p in aux_heads_module.parameters()):,} params '
              f'({len(aux_heads_module.head_names)} heads of {len(AUX_HEAD_SPECS)} available, '
              f'pool_dim={pool_dim})')

    # --- optimizer (target encoder is frozen, exclude it) ---
    trainable_params = (
        list(jepa.context_encoder.parameters())
        + list(jepa.predictor.parameters())
    )
    if aux_heads_module is not None:
        trainable_params = trainable_params + list(aux_heads_module.parameters())
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
    sigreg_lambda = float(config['loss'].get('sigreg_lambda', 0.0))
    sigreg_num_slices = int(config['loss'].get('sigreg_num_slices', 1024))
    sigreg_active = sigreg_lambda > 0.0
    if vicreg_active and sigreg_active:
        raise ValueError(
            'VICReg and SIGReg are both enabled in the loss config; '
            'pick one regularizer at a time.'
        )
    needs_encoder_raw = vicreg_active or sigreg_active
    needs_pooled = aux_active
    # When the contrastive loss should operate on unnormalized
    # encoder/predictor outputs (LeJEPA-style: SIGReg shapes the raw
    # distribution, contrastive operates in unconstrained R^embed_dim).
    # Default True preserves existing VICReg/contrastive behavior.
    normalize_outputs = bool(config.get('model', {}).get('l2_normalize_outputs', True))
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

    # Diagnostics: counters maintained across the epoch and reset each
    # epoch boundary, used to summarize per-epoch instability.
    epoch_clipped_steps = 0
    epoch_grad_norm_sum = 0.0
    encoder_params = list(jepa.context_encoder.parameters())
    predictor_params = list(jepa.predictor.parameters())

    for epoch in range(epochs):
        jepa.train()
        epoch_loss_sum = 0.0
        epoch_batches = 0
        epoch_clipped_steps = 0
        epoch_grad_norm_sum = 0.0
        for batch_idx, batch in enumerate(train_loader):
            if max_train_batches is not None and batch_idx >= max_train_batches:
                break
            # Aux-active loaders yield 5-tuples; otherwise 3-tuples.
            if aux_active:
                eq1, eq2, labels, eq1_aux, eq2_aux = batch
                eq1_aux = {k: v.to(device) for k, v in eq1_aux.items()}
                eq2_aux = {k: v.to(device) for k, v in eq2_aux.items()}
            else:
                eq1, eq2, labels = batch
            eq1 = eq1.to(device)
            eq2 = eq2.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            pooled_eq1 = pooled_eq2 = None
            if needs_encoder_raw and needs_pooled:
                z_pred, z_tgt, z_context_raw, z_tgt_raw, pooled_eq1, pooled_eq2 = jepa(
                    eq1, eq2,
                    return_encoder_raw=True,
                    return_pooled=True,
                    normalize_outputs=normalize_outputs,
                )
            elif needs_encoder_raw:
                z_pred, z_tgt, z_context_raw, z_tgt_raw = jepa(
                    eq1, eq2,
                    return_encoder_raw=True,
                    normalize_outputs=normalize_outputs,
                )
            elif needs_pooled:
                z_pred, z_tgt, pooled_eq1, pooled_eq2 = jepa(
                    eq1, eq2,
                    return_pooled=True,
                    normalize_outputs=normalize_outputs,
                )
            else:
                z_pred, z_tgt = jepa(eq1, eq2, normalize_outputs=normalize_outputs)
            cont_loss = contrastive_loss(
                z_pred, z_tgt, labels, margin=margin, true_weight=true_weight,
            )
            # Per-class contrastive components for diagnostics. Recomputes
            # d / pos / neg cheaply; logs the mean over each class so we
            # can see which side is destabilizing the loss curve.
            with torch.no_grad():
                _d_diag = F.pairwise_distance(z_pred, z_tgt)
                _pos_per = _d_diag.pow(2)                          # TRUE-pull per pair
                _neg_per = F.relu(margin - _d_diag).pow(2)         # FALSE-push per pair
                _n_true = labels.sum().clamp_min(1)
                _n_false = (1.0 - labels).sum().clamp_min(1)
                cont_loss_true = float((labels * _pos_per).sum() / _n_true)
                cont_loss_false = float(((1.0 - labels) * _neg_per).sum() / _n_false)
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
                #
                # In skip-pooled mode, slice to the projection half only.
                # The skip half is a deterministic LayerNorm(pooled) with
                # no trainable interaction with VICReg's covariance
                # objective (skip values are determined by the GNN
                # backbone, which the aux heads anchor); applying VICReg
                # there would push for decorrelation against the natural
                # cohom covariance structure of the pooled rep.
                if jepa.context_encoder.skip_pooled:
                    proj_d = jepa.context_encoder.proj_dim
                    z_ctx_for_vicreg = z_context_raw[:, :proj_d]
                    z_tgt_for_vicreg = z_tgt_raw[:, :proj_d]
                else:
                    z_ctx_for_vicreg = z_context_raw
                    z_tgt_for_vicreg = z_tgt_raw
                var_c, cov_c = vicreg_terms(z_ctx_for_vicreg, gamma=vicreg_gamma)
                var_t, cov_t = vicreg_terms(z_tgt_for_vicreg, gamma=vicreg_gamma)
                var_loss = 0.5 * (var_c + var_t)
                cov_loss = 0.5 * (cov_c + cov_t)
                loss = (
                    cont_loss
                    + vicreg_var_weight * var_loss
                    + vicreg_cov_weight * cov_loss
                )
            elif sigreg_active:
                # SIGReg on the context encoder pre-norm only (target
                # picks up diversity via EMA from the gradient-updated
                # context encoder, same rationale as VICReg). Per-step
                # direction sampling is seeded on `step` for
                # reproducibility — see src/training/sigreg.py.
                sigreg_per_slice = sigreg(
                    z_context_raw, global_step=step,
                    num_slices=sigreg_num_slices,
                )
                sigreg_loss = sigreg_per_slice.mean()
                loss = cont_loss + sigreg_lambda * sigreg_loss
            # Auxiliary anchor heads (additive on top of any regularizer
            # path). Logits come from the CONTEXT encoder's pooled
            # representation for both eq1 and eq2 — gradient flows back
            # through the encoder for both sides, anchoring structural
            # features (spine, lhs, num_vars, magma bits) so the
            # contrastive force can't suppress them during the late-
            # training simplicity-bias drift.
            if aux_active:
                aux_logits_e1 = aux_heads_module(pooled_eq1)
                aux_logits_e2 = aux_heads_module(pooled_eq2)
                aux_loss_e1, aux_per_head_e1 = auxiliary_loss(aux_logits_e1, eq1_aux)
                aux_loss_e2, aux_per_head_e2 = auxiliary_loss(aux_logits_e2, eq2_aux)
                aux_loss_total = 0.5 * (aux_loss_e1 + aux_loss_e2)
                loss = loss + aux_weight * aux_loss_total
            loss.backward()

            # Per-module grad norms BEFORE clipping (clip_grad_norm_ scales
            # all grads in place). Useful for spotting which side
            # (encoder vs predictor) is driving spikes.
            with torch.no_grad():
                enc_grad_norm = float(torch.norm(torch.stack([
                    p.grad.detach().data.norm(2)
                    for p in encoder_params if p.grad is not None
                ])))
                pred_grad_norm = float(torch.norm(torch.stack([
                    p.grad.detach().data.norm(2)
                    for p in predictor_params if p.grad is not None
                ])))
            # clip_grad_norm_ returns the *pre-clip* total norm; capture it
            # before applying the clip so we can tell when clipping fired.
            grad_norm_pre = float(torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip))
            grad_norm_post = min(grad_norm_pre, grad_clip)
            clipped = grad_norm_pre > grad_clip
            epoch_clipped_steps += int(clipped)
            epoch_grad_norm_sum += grad_norm_pre

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
                    'train/contrastive_loss_true': cont_loss_true,
                    'train/contrastive_loss_false': cont_loss_false,
                    'train/grad_norm_pre_clip': grad_norm_pre,
                    'train/grad_norm_post_clip': grad_norm_post,
                    'train/grad_norm_encoder': enc_grad_norm,
                    'train/grad_norm_predictor': pred_grad_norm,
                    'train/grad_clipped': float(clipped),
                    'train/lr': base_lr * lr_mult,
                    'train/tau': tau,
                    'train/elapsed_sec': time.perf_counter() - t_start,
                }
                if vicreg_active:
                    step_metrics['train/vicreg_var'] = var_loss.item()
                    step_metrics['train/vicreg_cov'] = cov_loss.item()
                if sigreg_active:
                    step_metrics['train/sigreg_loss'] = sigreg_loss.item()
                if aux_active:
                    step_metrics['train/aux_loss'] = aux_loss_total.item()
                    # Average per-head loss across eq1+eq2 for trace logs
                    # (matches the auxiliary_loss aggregation). Only
                    # active heads are present in the per-head dicts.
                    for name in aux_per_head_e1:
                        step_metrics[f'train/aux_{name}_loss'] = (
                            0.5 * (aux_per_head_e1[name] + aux_per_head_e2[name])
                        )
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
            aux_heads_module=aux_heads_module,
        )

        # Corpus-wide effective rank: encode every unique equation in the
        # graph cache through the context encoder (eval mode, batched),
        # compute Roy-Vetterli effective rank on the resulting [N, embed_dim]
        # matrix for both pre- and post-L2-normalization. This is the
        # diagnostic from scripts/embedding_spectrum.py, run per epoch so
        # the collapse-resolve trajectory is visible in real time. In
        # skip-pooled mode, also report per-channel ranks. Cheap
        # (~5 forward passes on 1024 graphs each on a 4080).
        rank_metrics = _corpus_effective_ranks(
            jepa.context_encoder, val_ds.graphs, device,
        )

        epoch_metrics = {
            'epoch': epoch,
            'train/avg_loss': avg_train_loss,
            'train/avg_grad_norm_pre_clip': epoch_grad_norm_sum / max(epoch_batches, 1),
            'train/clipped_step_fraction': epoch_clipped_steps / max(epoch_batches, 1),
            'val/loss': val['loss'],
            'val/balanced_accuracy': val['balanced_accuracy'],
            'val/true_recall': val['true_recall'],
            'val/false_recall': val['false_recall'],
            'val/threshold': val['threshold'],
            'val/n_evaluated': val['n_evaluated'],
            'val/corpus_eff_rank': rank_metrics['post_norm'],
            'val/corpus_eff_rank_raw': rank_metrics['pre_norm'],
            'val/dist_true_p05': val['dist_true_p05'],
            'val/dist_true_p50': val['dist_true_p50'],
            'val/dist_true_p95': val['dist_true_p95'],
            'val/dist_false_p05': val['dist_false_p05'],
            'val/dist_false_p50': val['dist_false_p50'],
            'val/dist_false_p95': val['dist_false_p95'],
            'val/dist_gap_median': val['dist_gap_median'],
            'val/dist_overlap_true_above_false_med': val['dist_overlap_true_above_false_med'],
            'val/dist_overlap_false_below_true_med': val['dist_overlap_false_below_true_med'],
            'elapsed_sec': time.perf_counter() - t_start,
        }
        # Skip-pooled mode: per-channel effective rank so we can detect
        # one-channel collapse (skip-channel locked at LayerNorm output =
        # high rank by construction; proj-channel could collapse if the
        # contrastive loss finds the skip alone sufficient).
        if 'proj' in rank_metrics:
            epoch_metrics['val/corpus_eff_rank_proj'] = rank_metrics['proj']
            epoch_metrics['val/corpus_eff_rank_skip'] = rank_metrics['skip']

            # Channel-ablation eval (zero one half, re-normalize, measure
            # bal_acc on a 50-batch subset using the main eval's threshold).
            # Diagnostic only — re-normalization changes the unit sphere,
            # so bal_acc here is NOT directly comparable to val['balanced_accuracy'].
            # Track per-epoch trends to see whether the model is increasingly
            # using each channel.
            for mask, label in [('proj_only', 'proj'), ('skip_only', 'skip')]:
                m = _evaluate_channel_mask(
                    jepa, val_loader, device, mask=mask,
                    threshold=val['threshold'], margin=margin,
                    true_weight=true_weight, max_batches=50,
                )
                epoch_metrics[f'val/bal_acc_{label}_only'] = m['balanced_accuracy']
                epoch_metrics[f'val/true_recall_{label}_only'] = m['true_recall']
                epoch_metrics[f'val/false_recall_{label}_only'] = m['false_recall']
        if aux_active:
            for name in aux_heads_module.head_names:
                key = f'aux_acc_{name}'
                if key in val:
                    epoch_metrics[f'val/{key}'] = val[key]
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
