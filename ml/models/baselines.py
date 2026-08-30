"""Baseline models for ablation/benchmark studies.

Both share the FolliscanNet forward interface (data, multihot) -> dict with
'logits' so Trainer/eval harnesses treat them identically. They deliberately
ignore motif features and pathway knowledge (that is the point of a baseline).
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_mean_pool

from ml.data.featurize import ATOM_F
from ml.data.task_registry import GROUP_INDICES


class _GCNBase(nn.Module):
    def __init__(self, dim: int = 128, n_layers: int = 3, dropout: float = 0.2):
        super().__init__()
        self.input_proj = nn.Linear(ATOM_F, dim)
        self.convs = nn.ModuleList([GCNConv(dim, dim) for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)

    def encode(self, data):
        h = self.input_proj(data.x)
        for conv in self.convs:
            h = conv(h, data.edge_index).relu()
            h = self.dropout(h)
        return global_mean_pool(h, data.batch)


class MultiTaskGCN(_GCNBase):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.head = nn.Sequential(nn.Linear(kw.get("dim", 128), 256), nn.GELU(),
                                  nn.Dropout(0.2), nn.Linear(256, 21))
        # Trainer._state_blob() does **self.model.config -- real crash on a
        # live ablation run (AttributeError: 'MultiTaskGCN' object has no
        # attribute 'config') since these baselines were never given one.
        self.config = {"dim": kw.get("dim", 128), "n_layers": kw.get("n_layers", 3),
                       "dropout": kw.get("dropout", 0.2), "model": "MultiTaskGCN"}

    def forward(self, data, multihot=None):
        return {"logits": self.head(self.encode(data)), "motif_contrib": None,
                "pathway_relevance": None}


class SingleTaskGCN(nn.Module):
    """One independent GCN per task group; logits assembled into the 21-vector."""

    def __init__(self, **kw):
        super().__init__()
        dim = kw.get("dim", 128)
        self.nets = nn.ModuleDict({g: MultiTaskGCN(dim=dim) for g in GROUP_INDICES})
        self.slices = GROUP_INDICES
        self.config = {"dim": dim, "model": "SingleTaskGCN", "n_task_groups": len(GROUP_INDICES)}

    def forward(self, data, multihot=None):
        logits = torch.zeros(data.num_graphs, 21, device=data.x.device)
        for g, net in self.nets.items():
            out = net(data, None)["logits"]
            ix = torch.tensor(self.slices[g], device=logits.device)
            logits[:, ix] = out[:, ix]
        return {"logits": logits, "motif_contrib": None, "pathway_relevance": None}
