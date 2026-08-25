"""Toxicity task builder: Tox21 high-throughput screening assays.

Source: MoleculeNet/Tox21 curated release (12 nuclear-receptor and stress-
response assays, qualitative 0/1 labels) distributed by DeepChem's public
dataset bucket. Labels are used as published; missing entries become mask=0.
"""

from __future__ import annotations

import io
import logging
import os

import pandas as pd
import requests

from ml.data.featurize import strip_salts
from ml.data.task_registry import TASK_REGISTRY

log = logging.getLogger(__name__)

TOX21_CSV_URLS = [
    # primary: DeepChem repo mirror; fallback: S3 bucket (intermittently 403s)
    "https://raw.githubusercontent.com/deepchem/deepchem/master/datasets/tox21.csv.gz",
    "https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/tox21.csv",
]
CACHE_PATH = os.environ.get("FOLLISCAN_CACHE_DIR", "/tmp/folliscan_cache") + "/tox21.csv"

TOX21_TASKS = [t["id"] for t in TASK_REGISTRY if t["group"] == "tox"]


def download_tox21(force: bool = False) -> pd.DataFrame:
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    if not force and os.path.exists(CACHE_PATH):
        return pd.read_csv(CACHE_PATH)
    import gzip

    last_err: Exception | None = None
    for url in TOX21_CSV_URLS:
        try:
            log.info("downloading Tox21 from %s", url)
            r = requests.get(url, timeout=180,
                             headers={"User-Agent": "folliscan-data-builder/1.0"})
            r.raise_for_status()
            text = gzip.decompress(r.content).decode() if url.endswith(".gz") else r.text
            df = pd.read_csv(io.StringIO(text))
            df.to_csv(CACHE_PATH, index=False)
            return df
        except Exception as e:  # noqa: BLE001 - try next mirror
            last_err = e
            log.warning("source failed (%s): %s", url, e)
    raise RuntimeError(f"all Tox21 sources failed: {last_err}")


def build_tox_tasks() -> dict[str, pd.DataFrame]:
    """Returns {task_id: DataFrame(smiles,label)} for all 12 Tox21 tasks."""
    raw = download_tox21()
    assert "smiles" in raw.columns, f"unexpected Tox21 schema: {raw.columns.tolist()}"
    out: dict[str, pd.DataFrame] = {}
    canon_cache: dict[str, str | None] = {}

    for task in TOX21_TASKS:
        if task not in raw.columns:
            log.warning("Tox21 column missing: %s (available: %s)", task, raw.columns.tolist())
            out[task] = pd.DataFrame(columns=["smiles", "label"])
            continue
        sub = raw[raw[task].notna()][["smiles", task]].copy()
        sub["label"] = sub[task].astype(int)
        smiles_col = []
        for s in sub["smiles"]:
            if s not in canon_cache:
                canon_cache[s] = strip_salts(str(s))
            smiles_col.append(canon_cache[s])
        sub["smiles"] = smiles_col
        sub = sub.dropna(subset=["smiles"]).drop_duplicates(subset="smiles")
        out[task] = sub[["smiles", "label"]].reset_index(drop=True)

    return out
