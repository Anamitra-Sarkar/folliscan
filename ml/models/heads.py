"""Task-group prediction heads."""

from __future__ import annotations

import torch
import torch.nn as nn

from ml.data.task_registry import GROUP_INDICES


class GroupHeads(nn.Module):
    """Shared trunk + one MLP head per task group (hair / tox / safety)."""

    def __init__(self, in_dim: int, hidden: int = 512, n_tasks: int = 21, dropout: float = 0.2):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2), nn.GELU(),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({g: nn.Linear(hidden // 2, len(ix)) for g, ix in GROUP_INDICES.items()})
        self._group_slices = {g: ix for g, ix in GROUP_INDICES.items()}
        self.n_tasks = n_tasks

    def forward(self, z: torch.Tensor):
        h = self.trunk(z)
        logits = torch.zeros(h.size(0), self.n_tasks, device=h.device, dtype=h.dtype)
        for g, head in self.heads.items():
            logits[:, self._group_slices[g]] = head(h)
        return logits
