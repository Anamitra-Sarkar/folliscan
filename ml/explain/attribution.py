"""Mechanistic attribution: gradient-informed motif importance scores."""

from __future__ import annotations

import numpy as np
import torch

from ml.data.task_registry import GROUP_INDICES
from ml.data.motifs import MOTIF_LIBRARY


@torch.enable_grad()
def motif_importance(model, data, multihot: torch.Tensor, group: str | None = None,
                     target_tasks: list[int] | None = None) -> list[dict]:
    """Rank active motifs by their influence on the prediction of `group`
    (or specific task indices). Uses input-gradients of the group's mean logit
    w.r.t. the motif multi-hot input combined with the SME gate contributions."""
    was_training = model.training
    model.train()          # need gradients; dropout noise is acceptable & averaged below
    multihot = multihot.clone().requires_grad_(True)
    out = model(data, multihot)
    logits = out["logits"]
    if target_tasks is None:
        target_tasks = GROUP_INDICES[group] if group else list(range(logits.size(1)))
    score = logits[:, target_tasks].mean()

    grads, = torch.autograd.grad(score, multihot)
    contrib = out.get("motif_contrib")
    if not was_training:
        model.eval()

    mh = multihot.detach()[0].cpu().numpy()
    g = grads.detach()[0].abs().cpu().numpy()
    c = contrib.detach()[0].cpu().numpy() if contrib is not None else np.zeros_like(g)
    combined = g * (1.0 + c)

    results = []
    for i, v in enumerate(mh):
        if v > 0 and combined[i] > 0:
            m = MOTIF_LIBRARY[i]
            results.append({
                "motif_id": m.id,
                "name": m.name,
                "severity": m.severity,
                "hazard": m.hazard,
                "message": m.message,
                "importance": float(combined[i]),
            })
    results.sort(key=lambda r: r["importance"], reverse=True)
    total = sum(r["importance"] for r in results) or 1.0
    for r in results:
        r["importance"] = round(r["importance"] / total, 4)
    return results
