"""Loss functions for JEPA training.

The plan §"Loss Function" specifies a contrastive loss that pulls TRUE
pairs together (minimize distance between z_predicted and z_target) and
pushes FALSE pairs apart up to a margin. ``contrastive_loss`` below is
the standard formulation.

``vicreg_terms`` adds the variance + covariance regularization from
VICReg (Bardes, Ponce & LeCun, 2022) — the invariance term is omitted
because the contrastive loss already provides invariance via the
TRUE-pair pull. Used when the embedding-spectrum diagnostic shows
dimensional collapse (effective rank ≪ embed_dim).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def contrastive_loss(
    z_predicted: torch.Tensor,
    z_target: torch.Tensor,
    labels: torch.Tensor,
    margin: float = 1.0,
    true_weight: float = 1.0,
) -> torch.Tensor:
    """JEPA contrastive loss.

    Pulls TRUE pairs (label=1) together and pushes FALSE pairs (label=0)
    apart up to ``margin``::

        d   = pairwise_distance(z_predicted, z_target)
        pos = labels       * d^2                       # contributes only for TRUE
        neg = (1 - labels) * relu(margin - d)^2        # contributes only for FALSE
        loss = mean(true_weight * pos + neg)

    Args:
        z_predicted: ``[B, embed_dim]``, predictor output.
        z_target: ``[B, embed_dim]``, target encoder output (typically
            with no gradient track).
        labels: ``[B]`` float tensor with 0.0 (FALSE) or 1.0 (TRUE).
        margin: positive scalar; FALSE pairs are pushed apart only up to
            this distance. Pairs already this far apart contribute 0.
        true_weight: multiplier on the TRUE term. Raise above 1 to
            upweight TRUE against the corpus's 63% FALSE prior.

    Returns:
        Scalar loss.
    """
    if z_predicted.shape != z_target.shape:
        raise ValueError(
            f'z_predicted and z_target shape mismatch: '
            f'{tuple(z_predicted.shape)} vs {tuple(z_target.shape)}'
        )
    if labels.shape[0] != z_predicted.shape[0]:
        raise ValueError(
            f'labels batch dim {labels.shape[0]} does not match '
            f'embedding batch dim {z_predicted.shape[0]}'
        )

    d = F.pairwise_distance(z_predicted, z_target)
    pos = labels * d.pow(2)
    neg = (1.0 - labels) * F.relu(margin - d).pow(2)
    return (true_weight * pos + neg).mean()


def vicreg_terms(
    z: torch.Tensor,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Variance and covariance regularization terms from VICReg.

    Operates on a *pre-normalization* embedding ``z`` of shape ``[B, D]``.
    Applying these terms to L2-normalized vectors fights a hard
    constraint (total variance fixed at 1 by the unit-sphere
    projection); the unconstrained pre-normalization space is where
    VICReg can actually push dimensions to be active and decorrelated.

    Variance term::

        std_d = sqrt(var(z[:, d]) + eps)        per dim d
        var_loss = mean over d of relu(gamma - std_d)

    Hinges per-dim std at ``gamma``: dimensions whose std falls below
    that threshold are pushed back up; dimensions already healthy
    contribute zero. ``gamma=1.0`` is the canonical choice.

    Covariance term::

        z_centered = z - mean(z, dim=0)
        cov = (z_centered.T @ z_centered) / (B - 1)
        cov_loss = sum of squared off-diagonals / D

    Penalizes correlation between any two distinct dimensions; pushes
    the embedding toward an isotropic distribution within the active
    subspace.

    Args:
        z: ``[B, D]`` pre-normalization embedding (predictor or target
            encoder raw output).
        gamma: variance hinge threshold (per-dim std target).
        eps: numerical-stability term inside the std computation.

    Returns:
        ``(variance_loss, covariance_loss)`` — both scalar tensors.
        Caller weights and sums them into the total training loss.
    """
    if z.dim() != 2:
        raise ValueError(f'z must be 2-D [B, D]; got shape {tuple(z.shape)}')
    B, D = z.shape
    if B < 2:
        raise ValueError(f'VICReg needs B >= 2 for unbiased covariance; got B={B}')

    # Variance term: hinge per-dim std at gamma.
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    var_loss = F.relu(gamma - std).mean()

    # Covariance term: sum of squared off-diagonals, scaled by D.
    z_centered = z - z.mean(dim=0, keepdim=True)
    cov = (z_centered.T @ z_centered) / max(B - 1, 1)
    off_diag_sq = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag_sq / D

    return var_loss, cov_loss


def effective_rank(z: torch.Tensor) -> float:
    """Roy–Vetterli effective rank of a ``[B, D]`` embedding matrix.

    Defined as ``exp(-Σ p_i log p_i)`` where ``p_i = s_i² / Σ s_j²`` and
    ``s_i`` are the singular values of ``z``. Resilient to scale; gives
    a single-scalar measure of "how many directions are effectively
    used" that correlates with the more visual k-for-99%-variance
    quantity reported by ``embeddings/scripts/embedding_spectrum.py``.

    Returns a Python float so it logs cleanly without keeping the
    autograd graph alive.
    """
    if z.dim() != 2:
        raise ValueError(f'z must be 2-D [B, D]; got shape {tuple(z.shape)}')
    with torch.no_grad():
        s = torch.linalg.svdvals(z.detach().float())
        s2 = s.pow(2)
        total = s2.sum()
        if total.item() <= 0:
            return 0.0
        p = s2 / total
        eps = 1e-12
        entropy = -(p * (p + eps).log()).sum()
        return float(entropy.exp().item())
