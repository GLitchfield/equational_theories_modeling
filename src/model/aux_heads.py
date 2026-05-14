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

# Pre-defined head subsets for common configurations. Train configs can
# either reference these by name (e.g. ``loss.aux_heads: structural``)
# or pass an explicit list (``loss.aux_heads: [spine_type, ...]``).
HEAD_SUBSETS: dict[str, list[str]] = {
    "structural": [
        "spine_type", "lhs_structure", "num_variables",
        "magma_bit_0", "magma_bit_1", "magma_bit_2",
        "magma_bit_3", "magma_bit_4", "magma_bit_6", "magma_bit_7",
    ],
    "cohom": [
        "cohom_count_2", "cohom_count_3", "cohom_count_5", "cohom_count_7",
        "cohom_mean_h2_2", "cohom_mean_h2_3", "cohom_mean_h2_5", "cohom_mean_h2_7",
    ],
    "all": list(HEAD_SPECS.keys()),
}


def resolve_head_names(spec: str | list[str] | None) -> list[str]:
    """Convert a config-side spec to a concrete list of head names.

    Args:
        spec: ``None`` -> all heads; a string -> one of the named subsets
            in ``HEAD_SUBSETS``; a list -> taken verbatim (validated
            against ``HEAD_SPECS``).
    """
    if spec is None:
        return list(HEAD_SPECS.keys())
    if isinstance(spec, str):
        if spec not in HEAD_SUBSETS:
            raise ValueError(
                f"unknown head subset {spec!r}; valid: {sorted(HEAD_SUBSETS)}"
            )
        return list(HEAD_SUBSETS[spec])
    if isinstance(spec, list):
        bad = [n for n in spec if n not in HEAD_SPECS]
        if bad:
            raise ValueError(f"unknown head names {bad}; valid: {sorted(HEAD_SPECS)}")
        if len(spec) == 0:
            raise ValueError("aux_heads list is empty; either omit the key or set aux_weight=0")
        return list(spec)
    raise TypeError(f"aux_heads spec must be None, str, or list[str]; got {type(spec).__name__}")


class AuxiliaryHeads(nn.Module):
    """Bank of single-layer linear probes from a pooled encoder representation.

    Each head outputs raw logits; cross-entropy loss is applied
    elsewhere (see ``auxiliary_loss``).

    Args:
        pool_dim: dimensionality of the encoder's pooled representation
            (mean+max concat = 2 × hidden_dim).
        head_names: optional subset of ``HEAD_SPECS`` keys to instantiate.
            If ``None``, all heads in ``HEAD_SPECS`` are created (back-
            compatible default). Order is preserved for canonical
            iteration in ``forward`` and ``auxiliary_loss``.
    """

    def __init__(self, pool_dim: int, head_names: list[str] | None = None) -> None:
        super().__init__()
        self.pool_dim = pool_dim
        if head_names is None:
            head_names = list(HEAD_SPECS.keys())
        bad = [n for n in head_names if n not in HEAD_SPECS]
        if bad:
            raise ValueError(f"unknown head names: {bad}")
        self.head_names = list(head_names)
        self.heads = nn.ModuleDict({
            name: nn.Linear(pool_dim, HEAD_SPECS[name])
            for name in self.head_names
        })

    def forward(self, pooled: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute logits for each active head.

        Args:
            pooled: ``[batch, pool_dim]`` — encoder output BEFORE the
                output projection and L2 normalization.
        Returns:
            Dict of ``{head_name: logits[batch, num_classes]}`` for
            heads in ``self.head_names`` (in iteration order).
        """
        return {name: self.heads[name](pooled) for name in self.head_names}


def auxiliary_loss(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
) -> tuple[torch.Tensor, dict[str, float]]:
    """Average cross-entropy across all active heads.

    Args:
        logits: dict of ``{head_name: logits[batch, num_classes]}`` from
            ``AuxiliaryHeads.forward``. The set of heads to apply loss
            over is taken from ``logits.keys()``.
        labels: dict of ``{head_name: int64_labels[batch]}`` — must
            include every key that appears in ``logits``.
    Returns:
        ``(scalar_loss, per_head_loss_dict)`` — the scalar mean loss
        across heads (for backprop) and a dict of per-head loss values
        (floats, for logging).
    """
    bad = [n for n in logits if n not in HEAD_SPECS]
    if bad:
        raise ValueError(f"logits contain unknown heads: {bad}")
    missing = [n for n in logits if n not in labels]
    if missing:
        raise ValueError(f"missing aux labels: {sorted(missing)}")

    per_head: dict[str, torch.Tensor] = {}
    for name in logits:
        per_head[name] = F.cross_entropy(logits[name], labels[name])
    total = torch.stack(list(per_head.values())).mean()
    return total, {name: float(v.detach()) for name, v in per_head.items()}


def auxiliary_accuracy(
    logits: Dict[str, torch.Tensor],
    labels: Dict[str, torch.Tensor],
) -> dict[str, float]:
    """Per-head argmax accuracy over the active heads. For monitoring."""
    out: dict[str, float] = {}
    for name in logits:
        preds = logits[name].argmax(dim=-1)
        out[name] = float((preds == labels[name]).float().mean())
    return out
