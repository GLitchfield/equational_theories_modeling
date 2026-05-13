"""Auxiliary prediction heads for the JEPA encoder.

Per ``aux_heads_implementation_plan.md``: a small set of single-layer
linear probes attached to the encoder's pooled representation that
predict per-equation structural invariants. These act as an anchor
against the simplicity-bias drift observed at h=256 — the encoder
can't suppress features that the auxiliary loss requires it to keep
linearly separable.

Heads (all single ``nn.Linear``, no hidden, no activation):
  - ``spine_type``                          : 6-class
  - ``lhs_structure``                       : 2-class
  - ``num_variables``                       : 6-class
  - ``magma_bit_{0,1,2,3,4,6,7}``           : 7 × 2-class
  - ``cohom_count_{2,3,5,7}``               : 5/5/6/6-class (binned counts)
  - ``cohom_mean_h2_{2,3,5,7}``             : 5/6/6/6-class (binned mean dim H², null bin for count==0)

The 5 sparse magma bits (5, 8, 9, 10, 11) are dropped at the
data-pipeline stage; see ``scripts/build_equation_invariants.py``.

The cohomology summary heads were added in a follow-up to test whether
per-prime cohomological structure (count of satisfying (a,b) pairs over
F_p, mean dim H² across satisfying pairs) provides additional anchoring
beyond syntactic structural invariants. Bin edges and class names live in
``scripts/build_equation_invariants.py::COHOM_COUNT_BINS / COHOM_MEAN_BINS``.

The heads are training-time-only anchors. They do NOT participate in
inference (the contrastive distance still drives prediction).
"""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# Class-count specs. Order is also the canonical iteration order for
# averaging the auxiliary loss. Names match the columns of
# data/processed/equation_invariants.parquet.
HEAD_SPECS: dict[str, int] = {
    "spine_type": 6,
    "lhs_structure": 2,
    "num_variables": 6,
    "magma_bit_0": 2,
    "magma_bit_1": 2,
    "magma_bit_2": 2,
    "magma_bit_3": 2,
    "magma_bit_4": 2,
    "magma_bit_6": 2,
    "magma_bit_7": 2,
    "cohom_count_2": 5,
    "cohom_count_3": 5,
    "cohom_count_5": 6,
    "cohom_count_7": 6,
    "cohom_mean_h2_2": 5,
    "cohom_mean_h2_3": 6,
    "cohom_mean_h2_5": 6,
    "cohom_mean_h2_7": 6,
}


class AuxiliaryHeads(nn.Module):
    """Bank of single-layer linear probes from a pooled encoder representation.

    Each head outputs raw logits; cross-entropy loss is applied
    elsewhere (see ``auxiliary_loss``).

    Args:
        pool_dim: dimensionality of the encoder's pooled representation
            (mean+max concat = 2 × hidden_dim).
    """

    def __init__(self, pool_dim: int) -> None:
        super().__init__()
        self.pool_dim = pool_dim
        self.heads = nn.ModuleDict({
            name: nn.Linear(pool_dim, num_classes)
            for name, num_classes in HEAD_SPECS.items()
        })

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute logits for each head.

        Args:
            pooled: ``[batch, pool_dim]`` — encoder output BEFORE the
                output projection and L2 normalization.
        Returns:
            Dict of ``{head_name: logits[batch, num_classes]}``.
        """
        return {name: head(pooled) for name, head in self.heads.items()}


def auxiliary_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average cross-entropy across all heads.

    Args:
        logits: dict of ``{head_name: logits[batch, num_classes]}`` from
            ``AuxiliaryHeads.forward``.
        labels: dict of ``{head_name: int64_labels[batch]}`` for each head.
    Returns:
        ``(scalar_loss, per_head_loss_dict)`` — the scalar mean loss
        across heads (for backprop) and a dict of per-head loss values
        (floats, for logging).
    """
    if set(logits.keys()) != set(HEAD_SPECS.keys()):
        raise ValueError(
            f"logits keys {sorted(logits.keys())} do not match HEAD_SPECS "
            f"{sorted(HEAD_SPECS.keys())}"
        )
    if set(labels.keys()) != set(HEAD_SPECS.keys()):
        missing = set(HEAD_SPECS.keys()) - set(labels.keys())
        raise ValueError(f"missing aux labels: {sorted(missing)}")

    per_head: dict[str, torch.Tensor] = {}
    for name in HEAD_SPECS:
        per_head[name] = F.cross_entropy(logits[name], labels[name])
    total = torch.stack(list(per_head.values())).mean()
    return total, {name: float(v.detach()) for name, v in per_head.items()}


def auxiliary_accuracy(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-head argmax accuracy. Used for monitoring, not training."""
    out: dict[str, float] = {}
    for name in HEAD_SPECS:
        preds = logits[name].argmax(dim=-1)
        out[name] = float((preds == labels[name]).float().mean())
    return out
