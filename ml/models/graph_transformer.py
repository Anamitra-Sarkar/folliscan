"""Multi-task Graph Transformer (MT-GT) backbone layers."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_batch

from ml.data.featurize import BOND_F


class GPSLayer(nn.Module):
    """GPS-style layer: local edge-aware self-attention (TransformerConv)
    combined with global node-level multi-head attention."""

    def __init__(self, dim: int, heads: int = 4, dropout: float = 0.15):
        super().__init__()
        assert dim % heads == 0
        self.dim, self.heads = dim, heads
        self.local = TransformerConv(
            dim, dim // heads, heads=heads,
            edge_dim=BOND_F, beta=True, root_weight=True,
        )
        self.global_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout),
                                nn.Linear(dim * 2, dim), nn.Dropout(dropout))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, edge_index, edge_attr, batch):
        h_local = self.local(x, edge_index, edge_attr)

        dense, mask = to_dense_batch(x, batch)          # [B,N,F], [B,N]
        key_pad = ~mask
        g, _ = self.global_attn(dense, dense, dense, key_padding_mask=key_pad,
                                need_weights=False)
        h_global = g[mask]                               # back to packed layout

        h = self.norm1(x + self.dropout(h_local) + self.dropout(h_global))
        h = self.norm2(h + self.ff(h))
        return h


class GraphTransformerBackbone(nn.Module):
    """Stack of GPS layers producing node embeddings."""

    def __init__(self, in_dim: int, dim: int = 256, n_layers: int = 4,
                 heads: int = 8, dropout: float = 0.15):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(in_dim, dim), nn.GELU(),
                                        nn.Dropout(dropout))
        self.layers = nn.ModuleList([GPSLayer(dim, heads, dropout) for _ in range(n_layers)])
        self.dim = dim

    def forward(self, x, edge_index, edge_attr, batch):
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr, batch)
        return h

    @staticmethod
    def masked_pool(h, batch, num_graphs):
        mean_p = torch.zeros(num_graphs, h.size(-1), device=h.device)
        max_p = torch.full((num_graphs, h.size(-1)), -1e9, device=h.device)
        mean_p.index_add_(0, batch, h)
        count = torch.bincount(batch, minlength=num_graphs).clamp(min=1).unsqueeze(-1)
        max_p.scatter_reduce_(0, batch.unsqueeze(-1).expand_as(h), h, reduce="amax", include_self=True)
        return torch.cat([mean_p / count, F.relu(max_p)], dim=-1)   # [B, 2*dim]
