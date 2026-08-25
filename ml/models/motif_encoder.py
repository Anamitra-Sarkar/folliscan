"""Substructure Motif Encoder (SME).

Embeds the multi-hot motif-match vector so the model reasons explicitly over
known pharmacophores/toxicophores. Also exposes per-motif contribution
scores used by the explanation layer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.data.motifs import N_MOTIFS


class MotifEncoder(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.15):
        super().__init__()
        self.embed = nn.Embedding(N_MOTIFS, dim)
        self.gate = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, multihot: torch.Tensor):
        """multihot: [B, M] float -> (motif_vec [B, dim], contributions [B, M])"""
        idx = multihot.nonzero(as_tuple=False)                     # [K, 2] (batch, motif)
        B = multihot.size(0)
        if idx.numel() == 0:
            zero = multihot.new_zeros(B, self.embed.embedding_dim)
            return zero, multihot.new_zeros(B, N_MOTIFS)

        e = self.embed(idx[:, 1])                                  # [K, dim]
        gated = e * self.gate(e)
        vec = multihot.new_zeros(B, self.embed.embedding_dim)
        vec.index_add_(0, idx[:, 0], gated)
        count = multihot.sum(dim=1, keepdim=True).clamp(min=1).sqrt()
        vec = self.norm(self.dropout(vec / count))

        # contribution of each active motif ~ norm of its gated embedding
        contrib = torch.zeros_like(multihot)
        norms = gated.norm(dim=-1).detach()
        contrib[idx[:, 0], idx[:, 1]] = norms
        return vec, contrib
