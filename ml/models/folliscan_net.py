"""FolliscanNet: full multi-task architecture.

    molecular graph ──► GraphTransformerBackbone ──► masked mean+max pooling
    motif multi-hot  ──► MotifEncoder                ──┐
    pooled molecule  ──► PathwayModule (attention)   ──┤── concat ──► GroupHeads
                                                        ──► logits [B, 21]
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.data.featurize import ATOM_F
from ml.data.motifs import N_MOTIFS
from .graph_transformer import GraphTransformerBackbone
from .motif_encoder import MotifEncoder
from .pathway_module import PathwayModule, N_PATHWAYS
from .heads import GroupHeads

DEFAULT_CONFIG = {
    "in_dim": ATOM_F,
    "dim": 256,
    "n_layers": 4,
    "heads": 8,
    "dropout": 0.15,
    "head_dropout": 0.2,
    "trunk_hidden": 512,
    "n_tasks": 21,
    "n_motifs": N_MOTIFS,
    "n_pathways": N_PATHWAYS,
}


class FolliscanNet(nn.Module):
    def __init__(self, config: dict | None = None):
        super().__init__()
        cfg = dict(DEFAULT_CONFIG)
        cfg.update(config or {})
        self.config = cfg

        self.backbone = GraphTransformerBackbone(
            in_dim=cfg["in_dim"], dim=cfg["dim"], n_layers=cfg["n_layers"],
            heads=cfg["heads"], dropout=cfg["dropout"],
        )
        self.motif_encoder = MotifEncoder(cfg["dim"], dropout=cfg["dropout"])
        self.pathway_module = PathwayModule(cfg["dim"], dropout=cfg["dropout"])
        # Ablation switches (pipeline 03); defaults keep the full architecture.
        self.use_sme = bool(cfg.get("use_sme", True))
        self.use_pathway = bool(cfg.get("use_pathway", True))
        z_dim = 2 * cfg["dim"]
        if self.use_sme:
            z_dim += cfg["dim"]
        if self.use_pathway:
            z_dim += cfg["dim"]
        self.heads = GroupHeads(z_dim, hidden=cfg["trunk_hidden"],
                                n_tasks=cfg["n_tasks"], dropout=cfg["head_dropout"])

    def forward(self, data, motif_multihot: torch.Tensor) -> dict:
        """data: PyG batch with x/edge_index/edge_attr/batch.
        motif_multihot: [B, M] float tensor aligned with the batch order."""
        h = self.backbone(data.x, data.edge_index, data.edge_attr, data.batch)
        pooled = GraphTransformerBackbone.masked_pool(h, data.batch, data.num_graphs)
        parts = [pooled]
        if self.use_sme:
            motif_vec, motif_contrib = self.motif_encoder(motif_multihot)
            parts.append(motif_vec)
        else:
            motif_vec = None
            motif_contrib = None
        if self.use_pathway:
            pw_ctx, pw_relevance = self.pathway_module(pooled[:, : self.config["dim"]])
            parts.append(pw_ctx)
        else:
            pw_relevance = None
        z = torch.cat(parts, dim=-1)
        return {
            "logits": self.heads(z),
            "pooled": pooled,
            "motif_contrib": motif_contrib,      # [B, M] interpretability (None if SME off)
            "pathway_relevance": pw_relevance,   # [B, P] interpretability (None if pathway off)
        }
