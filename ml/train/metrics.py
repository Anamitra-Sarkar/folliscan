"""Evaluation metrics: AUROC/AUPRC/Brier/ECE, bootstrap CIs, permutation test."""

from __future__ import annotations

import numpy as np

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def _auroc(y, p):
    if not _HAS_SKLEARN or len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def _auprc(y, p):
    if not _HAS_SKLEARN or len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y, p))


def brier_score(y, p):
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y, p, n_bins: int = 10):
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.digitize(p, bins) - 1
    ece = 0.0
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() == 0:
            continue
        conf = p[sel].mean()
        acc = y[sel].mean()
        ece += (sel.sum() / len(p)) * abs(acc - conf)
    return float(ece)


def evaluate(y_true: np.ndarray, probs: np.ndarray, mask: np.ndarray,
             task_ids: list[str]) -> dict:
    """y_true/probs/mask: [N, T]. Returns per-task + macro metrics."""
    per_task = {}
    for t, tid in enumerate(task_ids):
        sel = mask[:, t] == 1
        n = int(sel.sum())
        if n < 2:
            per_task[tid] = {"n": n, "auroc": None}
            continue
        y, p = y_true[sel, t], probs[sel, t]
        per_task[tid] = {
            "n": n,
            "positive_rate": round(float(y.mean()), 4),
            "auroc": round(_auroc(y, p), 4),
            "auprc": round(_auprc(y, p), 4),
            "brier": round(brier_score(y, p), 4),
            "ece": round(expected_calibration_error(y, p), 4),
        }
    aurocs = [v["auroc"] for v in per_task.values() if v.get("auroc") is not None]
    auprcs = [v["auprc"] for v in per_task.values() if "auprc" in v and v["auprc"] is not None]
    summary = {
        "macro_auroc": round(float(np.nanmean(aurocs)), 4) if aurocs else None,
        "macro_auprc": round(float(np.nanmean(auprcs)), 4) if auprcs else None,
        "n_tasks_evaluated": len(aurocs),
    }
    return {"per_task": per_task, "summary": summary}


def bootstrap_ci(y_true, probs, mask, metric_fn, n_boot: int = 1000,
                 seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = y_true.shape[0]
    vals = []
    for _ in range(n_boot):
        ix = rng.integers(0, n, size=n)
        v = metric_fn(y_true[ix], probs[ix], mask[ix])
        if not np.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return (round(float(lo), 4), round(float(hi), 4))


def permutation_test(y_true, probs, mask, metric_fn, n_perm: int = 500,
                     seed: int = 42) -> float:
    """One-sided p-value: probability that a label-permuted run scores >= observed."""
    observed = metric_fn(y_true, probs, mask)
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        perm = rng.permutation(y_true.shape[0])
        v = metric_fn(y_true[perm], probs, mask)
        if not np.isnan(v) and v >= observed:
            count += 1
    return round((count + 1) / (n_perm + 1), 4)


def macro_auroc_fn(task_ids):
    def fn(y, p, m):
        r = evaluate(y, p, m, task_ids)
        s = r["summary"]["macro_auroc"]
        return float("nan") if s is None else s
    return fn
