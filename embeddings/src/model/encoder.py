"""GATv2-based encoder for equation parse trees.

Takes a PyG ``Batch`` of equation graphs and produces one embedding per
graph. The embedding is what JEPA's predictor / target encoder operate on
and what downstream probing / clustering analysis consumes.

Architecture (per the engineering plan §"Encoder Architecture"):
  - input projection: 9-dim node features → hidden_dim
  - num_layers × GATv2Conv(hidden_dim → per_head_dim, heads=num_heads,
    concat=True, edge_dim=2) with residual connections and post-norm
  - global pooling: concat(mean, max) over each graph's nodes
  - final projection: 2*hidden_dim → proj_dim

Edges in the input graphs are parent→child only (engineering plan), so
information propagates top-down. Self-loops are added by GATv2Conv (the
default), which keeps single-node graphs (e.g. ``x = x``) well-defined.

Skip-connection mode (per ``skip_connection_plan.md``):
  When ``skip_pooled=True``, the pre-projection pooled rep
  (mean+max pool concat, ``2*hidden_dim``) is concatenated alongside
  the projection output to form a ``proj_dim + 2*hidden_dim`` =
  ``embed_dim`` raw embedding before L2-normalization. Both halves
  are passed through their own LayerNorm before concat so neither
  half dominates the unit vector at init (LayerNorm'd hidden states
  have ~unit per-dim scale; a randomly-initialized Linear's output
  per-dim values are much smaller, so without LayerNorm the skip
  channel would dominate at init and the projection's contribution
  would have to be earned over many epochs).

  Probing motivation: the GNN's pooled rep encodes per-equation
  cohomology features at higher accuracy than the projected output
  (most pronounced for ``count_p2``: 0.98 → 0.88 through the
  projection; 1-4pp losses on the rest of the cohom summaries). The
  skip preserves these features without the projection bottleneck.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv, global_max_pool, global_mean_pool


class EquationEncoder(nn.Module):
    """GATv2-based encoder; ``forward(batch) -> [num_graphs, embed_dim]``."""

    def __init__(
        self,
        node_features: int = 9,
        edge_features: int = 4,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        num_layers: int = 4,
        num_heads: int = 4,
        dropout: float = 0.1,
        skip_pooled: bool = False,
        proj_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})'
            )
        per_head_dim = hidden_dim // num_heads
        pool_dim = 2 * hidden_dim

        # Resolve projection dim. In skip-pooled mode the final embedding
        # is ``concat([proj_norm(proj(pooled)), skip_norm(pooled)])`` of
        # width ``proj_dim + pool_dim``, which must equal embed_dim.
        if skip_pooled:
            if proj_dim is None:
                proj_dim = embed_dim - pool_dim
            if proj_dim <= 0 or proj_dim + pool_dim != embed_dim:
                raise ValueError(
                    f'skip_pooled=True requires proj_dim + 2*hidden_dim == '
                    f'embed_dim; got proj_dim={proj_dim}, '
                    f'2*hidden_dim={pool_dim}, embed_dim={embed_dim} '
                    f'(sum={proj_dim + pool_dim})'
                )
        else:
            if proj_dim is not None and proj_dim != embed_dim:
                raise ValueError(
                    f'skip_pooled=False requires proj_dim==embed_dim or '
                    f'proj_dim=None; got proj_dim={proj_dim}, embed_dim={embed_dim}'
                )
            proj_dim = embed_dim

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.proj_dim = proj_dim
        self.pool_dim = pool_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.skip_pooled = skip_pooled

        self.input_proj = nn.Linear(node_features, hidden_dim)

        self.layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                GATv2Conv(
                    in_channels=hidden_dim,
                    out_channels=per_head_dim,
                    heads=num_heads,
                    concat=True,
                    edge_dim=edge_features,
                    dropout=dropout,
                )
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(pool_dim, proj_dim)

        if skip_pooled:
            # Per-channel LayerNorm so neither half dominates the unit
            # vector at init. ~1.5K params total at the canonical
            # 512+256 split. See module docstring for rationale.
            self.proj_norm = nn.LayerNorm(proj_dim)
            self.skip_norm = nn.LayerNorm(pool_dim)
        else:
            self.proj_norm = None
            self.skip_norm = None

    def forward(
        self,
        batch: Any,
        normalize: bool = True,
        return_pooled: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a PyG ``Batch`` of equation graphs.

        Returns a tensor of shape ``(batch.num_graphs, embed_dim)``.

        When ``normalize=True`` (default), the output is L2-normalized
        along the embedding axis — every row lies on the unit sphere
        (see CHANGELOG 2026-05-07: without this, distances grew
        unboundedly and the contrastive margin became unreachable).

        When ``normalize=False``, the raw post-projection output is
        returned. Used by VICReg to operate on the unconstrained
        embedding space, since enforcing variance/decorrelation on
        already-unit-norm vectors fights a hard constraint (total
        variance is fixed at 1 by the L2 projection).

        When ``return_pooled=True``, also returns the pre-projection
        pooled representation ``(mean_pool || max_pool)`` of shape
        ``(batch.num_graphs, 2 * hidden_dim)``. Used by the auxiliary
        prediction heads (``src/model/aux_heads.py``), which probe the
        pooled representation directly — the richest internal feature
        space, before the output projection compresses it down.
        """
        x, edge_index, edge_attr, b = (
            batch.x,
            batch.edge_index,
            batch.edge_attr,
            batch.batch,
        )

        h = self.input_proj(x)
        for layer, norm in zip(self.layers, self.norms):
            h_new = layer(h, edge_index, edge_attr)
            h_new = self.dropout(F.relu(h_new))
            h = norm(h + h_new)

        mean_pool = global_mean_pool(h, b)
        max_pool = global_max_pool(h, b)
        pooled = torch.cat([mean_pool, max_pool], dim=-1)
        if self.skip_pooled:
            z_proj = self.proj_norm(self.proj(pooled))
            z_skip = self.skip_norm(pooled)
            out = torch.cat([z_proj, z_skip], dim=-1)
        else:
            out = self.proj(pooled)
        out = F.normalize(out, dim=-1) if normalize else out
        if return_pooled:
            return out, pooled
        return out

    def num_parameters(self) -> int:
        """Count of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
