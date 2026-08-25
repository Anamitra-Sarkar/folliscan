"""Composite training objective: masked multi-task BCE with uncertainty
weighting (Kendall et al.) plus the PINN-style regulatory constraint penalty."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Safety-relevant task indices that each hard-rule alert constrains.
ALERT_TO_TASKS = {
    "mutagenicity_alert": ["SR-p53", "SR-ATAD5"],
    "sensitization_alert": ["skin_sensitizer"],
    "irritancy_alert": ["irritancy_alert"],
    "cosing_alert": ["cosing_prohibited"],
}

GROUP_OF = {
    "hair": list(range(0, 6)),
    "tox": list(range(6, 18)),
    "safety": list(range(18, 21)),
}


def masked_bce(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """BCE over labelled entries only. labels/mask: [B, T]; mask in {0,1}."""
    if mask.sum() == 0:
        return logits.sum() * 0.0
    losses = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none"
    )
    return (losses * mask).sum() / mask.sum()


class UncertaintyWeights(nn.Module):
    """Learned homoscedastic uncertainty weights per task group."""

    def __init__(self, groups=("hair", "tox", "safety")):
        super().__init__()
        self.log_vars = nn.ParameterDict({g: nn.Parameter(torch.zeros(1)) for g in groups})

    def forward(self, group_losses: dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        for g, loss in group_losses.items():
            lv = self.log_vars[g]
            total = total + 0.5 * (torch.exp(-lv) * loss + lv.squeeze())
        return total


def pinn_penalty(probs: torch.Tensor, hazard_flags_batch: list[dict],
                 margin: float = 0.5) -> torch.Tensor:
    """Soft regulatory constraint: when a hard structural-alert rule fires for a
    molecule, penalize the model for predicting LOW hazard probability on the
    corresponding tasks. probs are sigmoid probabilities [B, T]."""
    from ml.data.task_registry import TASK_INDEX

    device = probs.device
    penalty = torch.zeros((), device=device)
    n_hit = 0
    for i, flags in enumerate(hazard_flags_batch):
        for flag_name, task_ids in ALERT_TO_TASKS.items():
            if not flags.get(flag_name):
                continue
            for tid in task_ids:
                p_hazard = probs[i, TASK_INDEX[tid]]
                penalty = penalty + F.relu(margin - p_hazard)
                n_hit += 1
    return penalty / max(n_hit, 1)


class FolliscanLoss(nn.Module):
    def __init__(self, use_pinn: bool = True, pinn_lambda: float = 0.3):
        super().__init__()
        self.uw = UncertaintyWeights()
        self.use_pinn = use_pinn
        self.pinn_lambda = pinn_lambda

    def forward(self, logits, labels, mask, hazard_flags_batch=None):
        group_losses = {}
        for g, idx in GROUP_OF.items():
            gi = torch.tensor(idx, device=logits.device)
            group_losses[g] = masked_bce(logits[:, gi], labels[:, gi], mask[:, gi])
        total = self.uw(group_losses)
        out = {"total": total, **{f"loss_{g}": v.detach() for g, v in group_losses.items()}}
        if self.use_pinn and hazard_flags_batch:
            probs = torch.sigmoid(logits.detach())  # constraint on prediction, not grads through UW
            lp = pinn_penalty(probs, hazard_flags_batch)
            out["loss_pinn"] = lp.detach()
            total = total + self.pinn_lambda * lp
            out["total"] = total
        return out
