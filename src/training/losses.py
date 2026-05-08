"""Loss functions for JEPA training.

The plan §"Loss Function" specifies a contrastive loss that pulls TRUE
pairs together (minimize distance between z_predicted and z_target) and
pushes FALSE pairs apart up to a margin. ``contrastive_loss`` below is
the standard formulation. ``InfoNCE`` and ``VICReg`` are not implemented
here yet — only add them if Phase 1 sanity training shows embedding
collapse.
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
