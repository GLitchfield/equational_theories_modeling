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
  - No layer normalization between layers.
  - Output is L2-normalized along the embedding axis (added 2026-05-07
    after the first sanity training showed unbounded distance growth;
    see CHANGELOG). With both encoder and predictor outputs on the unit
    sphere, distances are bounded to [0, 2] and the contrastive loss's
    margin sits inside the achievable range.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ImplicationPredictor(nn.Module):
    """Two-layer MLP predictor with L2-normalized output.

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

    def forward(self, z: torch.Tensor, normalize: bool = True) -> torch.Tensor:
        """Map ``z`` (typically a unit-norm embedding) through the MLP.

        When ``normalize=True`` (default), the output is L2-normalized so
        the contrastive loss's margin in [0, 2] is reachable. When
        ``normalize=False``, the raw MLP output is returned — used by
        VICReg, which needs to operate on the pre-normalization space
        (where variance and covariance are not constrained to live on
        a sphere of fixed radius).
        """
        out = self.net(z)
        return F.normalize(out, dim=-1) if normalize else out

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
