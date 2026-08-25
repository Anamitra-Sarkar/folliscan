"""Merge per-task label frames into the master multi-label dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ml.data.task_registry import TASK_REGISTRY


def build_master_dataset(task_dfs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Returns frame: smiles | labels (list[float] 21) | mask (list[int] 21).

    Consensus rule when a molecule carries multiple measurements for one task:
    majority vote; ties resolved to POSITIVE (conservative for hazard endpoints,
    and matches 'any qualifying activity' semantics for hair-health).
    """
    n_tasks = len(TASK_REGISTRY)
    idx_of = {t["id"]: t["index"] for t in TASK_REGISTRY}

    # collect votes per molecule per task
    votes: dict[str, list[list[int]]] = {}

    def add(smiles: str, task_id: str, label: int):
        slot = votes.setdefault(smiles, [[0, 0]] * n_tasks)  # type: ignore[assignment]
        v = slot[idx_of[task_id]]
        v[label] += 1

    for task_id, df in task_dfs.items():
        assert task_id in idx_of, f"unknown task {task_id}"
        for r in df.itertuples():
            add(r.smiles, task_id, int(r.label))

    rows = []
    for smiles, slots in votes.items():
        labels, mask = [], []
        for pos_v, neg_v in slots:
            if pos_v + neg_v == 0:
                labels.append(-1.0)
                mask.append(0)
            else:
                labels.append(float(pos_v >= neg_v))
                mask.append(1)
        rows.append({"smiles": smiles, "labels": labels, "mask": mask})
    return pd.DataFrame(rows).sort_values("smiles").reset_index(drop=True)


def task_label_stats(df: pd.DataFrame) -> pd.DataFrame:
    stats = []
    for t in TASK_REGISTRY:
        i = t["index"]
        m = np.array([r.mask[i] for r in df.itertuples()])
        y = np.array([r.labels[i] for r in df.itertuples()])
        labelled = int(m.sum())
        pos = int(sum(1 for mi, yi in zip(m, y) if mi and yi == 1))
        stats.append({"task": t["id"], "group": t["group"],
                      "labelled": labelled, "positive": pos,
                      "positive_rate": round(pos / labelled, 3) if labelled else None})
    return pd.DataFrame(stats)
