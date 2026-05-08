"""JEPA module: context encoder + EMA-updated target encoder + predictor.

Per the engineering plan:
  - Context encoder: trained by gradient descent.
  - Target encoder: identical architecture, EMA-updated from context, no SGD.
  - Predictor: trained by gradient descent.
  - Forward: ``(eq1, eq2) -> (z_predicted, z_target)``. The contrastive
    loss takes it from there.

The target encoder is constructed as a ``deepcopy`` of the context encoder
at init, with ``requires_grad=False`` on every target parameter so
gradients can't leak in even if the user forgets to wrap target calls in
``torch.no_grad()``. The ``update_target(tau)`` method does the EMA step:

    target_param ← tau * target_param + (1 - tau) * context_param

Standard JEPA / BYOL convention. The tau schedule itself (cosine anneal
from 0.996 to 0.999) lives in the training scheduler module, not here.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn

from src.model.encoder import EquationEncoder
from src.model.predictor import ImplicationPredictor


class JEPA(nn.Module):
    """Wraps a context encoder, EMA-updated target encoder, and predictor.

    Args:
        context_encoder: trainable encoder; its weights drive both training
            and (via EMA) the target encoder.
        predictor: trainable MLP from z_context → z_predicted.

    The target encoder is created automatically as a deep copy of
    ``context_encoder`` with all parameters frozen.
    """

    def __init__(
        self,
        context_encoder: EquationEncoder,
        predictor: ImplicationPredictor,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.predictor = predictor

        # Target encoder: identical architecture, weights cloned, no grad.
        self.target_encoder = copy.deepcopy(context_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)

    def forward(
        self,
        eq1_batch,
        eq2_batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the JEPA forward pass.

        Returns:
            ``(z_predicted, z_target)`` — both shape ``[num_graphs, embed_dim]``.
            ``z_target`` is computed under ``torch.no_grad`` so it carries no
            gradient track; combined with the target encoder's frozen
            parameters, this makes the target side a true stop-gradient.
        """
        z_context = self.context_encoder(eq1_batch)
        with torch.no_grad():
            z_target = self.target_encoder(eq2_batch)
        z_predicted = self.predictor(z_context)
        return z_predicted, z_target

    @torch.no_grad()
    def update_target(self, tau: float) -> None:
        """EMA update of the target encoder.

        ``target_param ← tau * target_param + (1 - tau) * context_param``

        ``tau`` ∈ [0, 1]:
          - ``tau = 1.0``: target unchanged (no tracking).
          - ``tau = 0.0``: target fully overwritten by context (full copy).
          - ``tau ≈ 0.996 → 0.999`` over training (plan's cosine schedule).
        """
        if not 0.0 <= tau <= 1.0:
            raise ValueError(f'tau must be in [0, 1]; got {tau}')
        for p_t, p_c in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            p_t.data.mul_(tau).add_(p_c.data, alpha=1.0 - tau)
        # Copy buffers too. Current encoder has none, but this is defensive
        # in case future encoders include BatchNorm running stats etc.
        for b_t, b_c in zip(
            self.target_encoder.buffers(),
            self.context_encoder.buffers(),
        ):
            b_t.data.copy_(b_c.data)
