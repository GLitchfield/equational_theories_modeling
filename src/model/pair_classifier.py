"""Pair classifiers operating on frozen JEPA embeddings.

Three variants, all taking ``(z_e1, z_e2)`` with ``z_*: [B, D]`` and
returning a single logit per pair:

  - ``PairLinear``               Linear(2D, 1).                Phase 2.
  - ``PairMLPConcat``            MLP on concat(z_e1, z_e2).    Phase 3 variant A.
  - ``PairMLPFeatures``          MLP on concat + |diff| + z1*z2.  Phase 3 variant B.

The embeddings are frozen — gradient flows only through the classifier
head. See `mlp_classifier_plan.md` §"Methodology resolutions" for the
training conventions (context encoder for both sides, ``pos_weight``
to handle the 37/63 class imbalance, etc.).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class PairLinear(nn.Module):
    """Single-layer linear probe on ``concat(z_e1, z_e2)`` — the torch
    equivalent of sklearn's LogisticRegression on the same features."""

    def __init__(self, embed_dim: int = 768) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.linear = nn.Linear(embed_dim * 2, 1)

    def forward(self, z_e1: torch.Tensor, z_e2: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([z_e1, z_e2], dim=-1)
        return self.linear(pair)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PairMLPConcat(nn.Module):
    """2-hidden-layer MLP on ``concat(z_e1, z_e2)`` — the plan's
    ``PairClassifier``."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 2, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, z_e1: torch.Tensor, z_e2: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([z_e1, z_e2], dim=-1)
        return self.net(pair)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class PairMLPFeatures(nn.Module):
    """2-hidden-layer MLP on ``[z_e1, z_e2, |z_e1 - z_e2|, z_e1 * z_e2]``
    — the plan's ``PairClassifierWithFeatures``. Element-wise relational
    features (abs-diff and Hadamard product) on top of the raw concat."""

    def __init__(
        self,
        embed_dim: int = 768,
        hidden_dims: tuple[int, int] = (512, 256),
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        h1, h2 = hidden_dims
        self.net = nn.Sequential(
            nn.Linear(embed_dim * 4, h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h2, 1),
        )

    def forward(self, z_e1: torch.Tensor, z_e2: torch.Tensor) -> torch.Tensor:
        pair = torch.cat([
            z_e1,
            z_e2,
            torch.abs(z_e1 - z_e2),
            z_e1 * z_e2,
        ], dim=-1)
        return self.net(pair)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


MODELS: dict[str, type[nn.Module]] = {
    'linear': PairLinear,
    'mlp_concat': PairMLPConcat,
    'mlp_features': PairMLPFeatures,
}


def build_model(name: str, embed_dim: int = 768, **kwargs) -> nn.Module:
    """Factory for the three variants, dispatched by string name."""
    if name not in MODELS:
        raise ValueError(
            f'unknown pair-classifier name {name!r}; expected one of {sorted(MODELS)}'
        )
    return MODELS[name](embed_dim=embed_dim, **kwargs)
