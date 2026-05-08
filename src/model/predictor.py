"""MLP predictor for JEPA contrastive embedding prediction.

Per the engineering plan, the predictor is a small MLP mapping z_context →
z_predicted. It is deliberately bottlenecked: if it's too expressive, it
bypasses the encoder's need to learn good representations.

Design (the plan only specifies "2 layers, hidden_dim=128, smaller than
encoder"; rest clarified by the user 2026-05-07, see CHANGELOG):
  - 2 weight matrices: ``Linear(embed_dim → hidden) → ReLU → Linear(hidden → embed_dim)``
    (BYOL/SimSiam convention; not 3 weight matrices).
  - No dropout. The predictor must be a clean, deterministic mapping;
    dropout would inject noise into z_predicted and degrade the
    contrastive loss. BYOL and SimSiam predictors have no dropout.
  - No layer normalization or output L2-normalization. If embedding
    collapse occurs at training time, try VICReg regularization first.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ImplicationPredictor(nn.Module):
    """Two-layer MLP predictor.

    Args:
        embed_dim: input and output dimension (default 64, matching
            ``EquationEncoder.embed_dim``).
        hidden_dim: bottleneck width between the two linear layers
            (default 128, per the plan).
    """

    def __init__(self, embed_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embed_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
