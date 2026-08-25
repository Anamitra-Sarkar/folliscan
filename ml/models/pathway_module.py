"""Pathway-Informed Knowledge Injection Module.

A fixed heterogeneous knowledge table of hair-biology & toxicology pathways
(Wnt/Shh/BMP/Notch cascades, KGF signalling, steroidogenic/AR axis, and
Adverse Outcome Pathway anchors such as Keap1-Nrf2, p53, mitochondrial
stress). The molecule representation attends over pathway embeddings,
yielding (a) a pathway-conditioned context vector merged into prediction
heads and (b) an interpretable relevance distribution over pathways that is
surfaced verbatim by the API explanation payload.
"""

from __future__ import annotations

import torch
import torch.nn as nn

PATHWAYS = [
    {"name": "Wnt/beta-catenin", "group": "hair", "role": "pro-anagen signaling"},
    {"name": "Sonic hedgehog (Shh)", "group": "hair", "role": "follicle morphogenesis"},
    {"name": "BMP", "group": "hair", "role": "catagen/quiescence regulation"},
    {"name": "Notch", "group": "hair", "role": "stem-cell lineage decisions"},
    {"name": "KGF/FGF7-FGFR2b", "group": "hair", "role": "keratinocyte proliferation"},
    {"name": "Androgen receptor axis", "group": "hair", "role": "androgenic alopecia driver"},
    {"name": "5a-reductase steroidogenesis", "group": "hair", "role": "DHT synthesis"},
    {"name": "SULT1A1 bioactivation", "group": "hair", "role": "minoxidil sulfate activation"},
    {"name": "Ahr xenobiotic response", "group": "tox", "role": "dioxin-like activation"},
    {"name": "Estrogen receptor signaling", "group": "tox", "role": "endocrine disruption"},
    {"name": "p53 DNA-damage response", "group": "tox", "role": "genotoxicity key event"},
    {"name": "Keap1-Nrf2 oxidative stress", "group": "tox", "role": "ARE induction"},
    {"name": "Mitochondrial dysfunction", "group": "tox", "role": "MMP loss / cytotoxicity"},
    {"name": "Heat-shock protein stress", "group": "tox", "role": "protein unfolding stress"},
    {"name": "Keratinocyte inflammatory sensitization", "group": "safety", "role": "skin sensitization AOP"},
    {"name": "Skin barrier irritancy", "group": "safety", "role": "surfactant/corrosive AOP"},
]

PATHWAY_GROUPS = {i: p["group"] for i, p in enumerate(PATHWAYS)}
N_PATHWAYS = len(PATHWAYS)


class PathwayModule(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, dropout: float = 0.15):
        super().__init__()
        # Learned embeddings over the fixed pathway knowledge graph nodes.
        self.pathway_embed = nn.Embedding(N_PATHWAYS, dim)
        nn.init.xavier_uniform_(self.pathway_embed.weight)
        self.attn = nn.MultiheadAttention(dim, n_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.out_proj = nn.Sequential(nn.Linear(dim, dim), nn.GELU())

    def forward(self, mol_vec: torch.Tensor):
        """mol_vec: [B, dim] -> (context [B, dim], relevance [B, P])"""
        keys = self.pathway_embed.weight.unsqueeze(0).expand(mol_vec.size(0), -1, -1)
        queries = mol_vec.unsqueeze(1)                       # [B,1,dim]
        ctx, attn_w = self.attn(queries, keys, keys, need_weights=True, average_attn_weights=True)
        relevance = attn_w.squeeze(1)                        # [B,P]
        return self.out_proj(self.norm(ctx.squeeze(1))), relevance
