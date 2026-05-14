"""Cohomology MLP — the algebra channel of the dual-encoder architecture.

A tiny two-layer MLP that consumes the 174-dim cohomology profile
(``[mask(87) | values_normalized(87)]``, built by
``scripts/build_cohom_profiles.py``) and produces a 128-dim algebraic
embedding.

The output gets concatenated with the GNN's 384-dim syntactic projection
to form the 512-dim pre-norm embedding (see
``src/model/encoder.py::EquationEncoder``). The combined 512-dim vector
is then L2-normalized for the contrastive loss.

Design choices per ``dual_encoder_plan.md`` (locked-in resolutions):
  - Input is 174-dim (87 mask + 87 per-prime-normalized values), not
    87-dim with None→0 (which conflated "doesn't satisfy" with
    "satisfies, dim H² = 0" for ~33% of satisfying p=2 cells).
  - Values normalized to ``[0, 1]`` already in the pre-bake step.
  - Two layers, ReLU activation, ~38K params (Linear(174, 128) +
    Linear(128, 128) + biases).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CohomologyEncoder(nn.Module):
    """MLP from the 174-dim cohomology profile to an algebraic embedding.

    Args:
        input_dim:  width of the cohom profile (174 for the canonical
                    mask+values layout).
        hidden_dim: width of the intermediate ReLU layer.
        output_dim: width of the algebraic embedding emitted into the
                    dual-encoder's concat. Must match the
                    ``encoder.algebra_dim`` config value.
    """

    def __init__(
        self,
        input_dim: int = 174,
        hidden_dim: int = 128,
        output_dim: int = 128,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cohom_profile: torch.Tensor) -> torch.Tensor:
        """Encode ``[batch, input_dim]`` profiles into ``[batch, output_dim]``.

        The profile is expected to be the mask-and-values concat from
        ``scripts/build_cohom_profiles.py``; no further normalization
        happens here.
        """
        if cohom_profile.dim() != 2 or cohom_profile.size(1) != self.input_dim:
            raise ValueError(
                f"cohom_profile shape {tuple(cohom_profile.shape)} != "
                f"(batch, {self.input_dim})"
            )
        return self.net(cohom_profile)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
