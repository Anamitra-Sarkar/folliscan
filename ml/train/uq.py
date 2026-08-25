"""Uncertainty quantification: Monte-Carlo Dropout + split Conformal Prediction."""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn.functional as F


def enable_mc_dropout(model: torch.nn.Module):
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_predict(model, data, motif_multihot, n_samples: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """Returns (mean_probs [T], std [T]) over stochastic passes."""
    enable_mc_dropout(model)
    stack = []
    for _ in range(n_samples):
        out = model(data, motif_multihot)
        stack.append(torch.sigmoid(out["logits"]).squeeze(0))
    probs = torch.stack(stack)
    return probs.mean(0).cpu().numpy(), probs.std(0).cpu().numpy()


class ConformalCalibrator:
    """Split conformal prediction on absolute-error nonconformity scores.

    Per-task interval [p-q_t, p+q_t] guaranteed to cover the true label at
    (1-alpha) under exchangeability; q_t is the ceil((n+1)(1-alpha))/n quantile
    of calibration scores.
    """

    def __init__(self, alpha: float = 0.05, n_tasks: int = 21):
        self.alpha = alpha
        self.n_tasks = n_tasks
        self.q: np.ndarray | None = None

    @staticmethod
    def _quantile(scores: np.ndarray, alpha: float) -> float:
        n = len(scores)
        if n == 0:
            return 1.0
        level = int(np.ceil((n + 1) * (1 - alpha))) - 1
        level = min(max(level, 0), n - 1)
        return float(np.sort(scores)[level])

    def fit(self, probs: np.ndarray, labels: np.ndarray, mask: np.ndarray):
        """probs/labels/mask: [N, T]; mask 1 where label present."""
        q = np.full(self.n_tasks, 1.0)
        for t in range(self.n_tasks):
            sel = mask[:, t] == 1
            if sel.sum() < 5:
                continue
            scores = np.abs(labels[sel, t] - probs[sel, t])
            q[t] = self._quantile(scores, self.alpha)
        self.q = q
        return self

    def predict_set(self, mean_probs: np.ndarray) -> np.ndarray:
        assert self.q is not None, "calibrate first"
        lo = np.clip(mean_probs - self.q, 0.0, 1.0)
        hi = np.clip(mean_probs + self.q, 0.0, 1.0)
        return np.stack([lo, hi], axis=-1)

    def coverage(self, probs: np.ndarray, labels: np.ndarray, mask: np.ndarray) -> dict[str, float]:
        """Empirical coverage on labeled entries."""
        assert self.q is not None
        out = {}
        sets = self.predict_set(probs)
        for t in range(self.n_tasks):
            sel = mask[:, t] == 1
            if sel.sum() < 5:
                continue
            covered = (labels[sel, t] >= sets[sel, t, 0]) & (labels[sel, t] <= sets[sel, t, 1])
            out[f"task_{t}"] = float(covered.mean())
        vals = np.array(list(out.values()))
        out["mean"] = float(vals.mean()) if len(vals) else float("nan")
        return out

    def to_dict(self) -> dict:
        return {"alpha": self.alpha, "q": None if self.q is None else self.q.tolist()}

    @classmethod
    def from_dict(cls, d: dict) -> "ConformalCalibrator":
        c = cls(alpha=d["alpha"])
        c.q = np.array(d["q"], dtype=float) if d.get("q") else None
        return c

    def save_json(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f)

    @classmethod
    def load_json(cls, path: str) -> "ConformalCalibrator":
        with open(path) as f:
            return cls.from_dict(json.load(f))
