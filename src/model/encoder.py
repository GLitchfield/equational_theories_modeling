"""GATv2-based encoder for equation parse trees.

Takes a PyG ``Batch`` of equation graphs and produces one embedding per
graph. The embedding is what JEPA's predictor / target encoder operate on
and what downstream probing / clustering analysis consumes.

Architecture (per the engineering plan §"Encoder Architecture"):
  - input projection: 9-dim node features → hidden_dim
  - num_layers × GATv2Conv(hidden_dim → per_head_dim, heads=num_heads,
    concat=True, edge_dim=2) with residual connections and post-norm
  - global pooling: concat(mean, max) over each graph's nodes
  - final projection: 2*hidden_dim → embed_dim

Edges in the input graphs are parent→child only (engineering plan), so
information propagates top-down. Self-loops are added by GATv2Conv (the
default), which keeps single-node graphs (e.g. ``x = x``) well-defined.
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
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f'hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})'
            )
        per_head_dim = hidden_dim // num_heads

        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_heads = num_heads

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
        self.proj = nn.Linear(2 * hidden_dim, embed_dim)

    def forward(self, batch: Any) -> torch.Tensor:
        """Encode a PyG ``Batch`` of equation graphs.

        Returns a tensor of shape ``(batch.num_graphs, embed_dim)``.
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
        return self.proj(pooled)

    def num_parameters(self) -> int:
        """Count of learnable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
