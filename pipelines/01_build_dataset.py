"""Pipeline 01 — Build the Folliscan multi-task dataset and push to HuggingFace.

Runs on Kaggle (network + modest CPU). Supports `--sample N` for a fast
smoke run locally. Steps:
  1. Build hair-health tasks from ChEMBL (+ literature set)
  2. Download & label Tox21 assays
  3. Curate safety tasks (sensitization / CosIng / irritancy)
  4. Merge into master 21-task frame, scaffold split 80/10/10
  5. Write parquet + dataset card, upload to HF datasets repo
"""

import argparse
import datetime as _dt
import json
import logging
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("pipeline01")


def build_all(sample: int | None = None) -> pd.DataFrame:
    from ml.data.chembl_hair import build_hair_tasks
    from ml.data.tox21_toxcast import build_tox_tasks
    from ml.data.cosing_safety import build_safety_tasks
    from ml.data.merge import build_master_dataset

    hair = build_hair_tasks()
    tox = build_tox_tasks()
    safety = build_safety_tasks()

    all_tasks = {**hair, **tox, **safety}
    if sample:
        all_tasks = {k: v.head(max(20, sample // len(all_tasks))) for k, v in all_tasks.items()}

    for k, v in sorted(all_tasks.items()):
        log.info("task %-28s labelled=%d positives=%d", k, len(v), int(v["label"].sum()) if not v.empty else 0)

    master = build_master_dataset(all_tasks)
    log.info("master dataset molecules: %d", len(master))
    return master


def make_splits(master: pd.DataFrame):
    from ml.data.splits import scaffold_split

    return scaffold_split(master)


def serialize(df: pd.DataFrame) -> pd.DataFrame:
    out = df[["smiles"]].copy()
    out["labels"] = df["labels"].map(lambda x: json.dumps(x))
    out["mask"] = df["mask"].map(lambda x: json.dumps(x))
    if "source" not in df.columns:
        out["source"] = "folliscan-v1"
    return out.reset_index(drop=True)


def write_dataset_card(out_dir: str, stats_df: pd.DataFrame, n_total: int,
                       split_sizes: tuple[int, int, int]) -> str:
    from ml.data.task_registry import TASK_REGISTRY

    table = stats_df.to_markdown(index=False)
    card = f"""---
license: cc-by-4.0
task_categories:
- tabular-classification
tags:
- chemistry
- molecular-property-prediction
- cosmetic-safety
- hair-health
- toxicity
size_categories:
- 1K<n<100K
---

# Folliscan Multi-task Dataset (v1)

Joint hair-health activity ({sum(1 for t in TASK_REGISTRY if t['group']=='hair')} tasks),
toxicity (Tox21, {sum(1 for t in TASK_REGISTRY if t['group']=='tox')} tasks) and
cosmetic-safety ({sum(1 for t in TASK_REGISTRY if t['group']=='safety')} tasks) labels for
{n_total} unique molecules (canonical SMILES, salts stripped).

Sources: ChEMBL REST bioactivities (pChEMBL-thresholded, consensus-labelled),
Tox21 (MoleculeNet/DeepChem release), curated LLNA-informed sensitization set,
curated EU CosIng Annex II representative set, documented irritant classes.
Built {_dt.date.today().isoformat()}.

Split: Bemis-Murcko scaffold grouping (80/10/10), leakage-checked per scaffold key.
Labels: `labels` JSON list of 21 floats (-1 = unlabelled); `mask` JSON list of 0/1.

## Task statistics

{table}
"""
    path = os.path.join(out_dir, "dataset_card.md")
    with open(path, "w") as f:
        f.write(card)
    return path


def push_to_hub(out_dir: str, repo_id: str):
    token = os.environ.get("HF_TOKEN")
    if not token or not repo_id:
        log.warning("HF_TOKEN/HF_DATA_REPO unset; skipping hub upload")
        return False
    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="dataset", exist_ok=True, private=False)
    for fn in os.listdir(out_dir):
        api.upload_file(os.path.join(out_dir, fn), fn, repo_id, repo_type="dataset",
                        commit_message=f"add {fn}")
    log.info("uploaded dataset to https://huggingface.co/datasets/%s", repo_id)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=None,
                    help="cap records per task for a smoke run")
    ap.add_argument("--out-dir", default="/tmp/folliscan_dataset")
    ap.add_argument("--hf-repo", default=os.environ.get("HF_DATA_REPO"))
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    master = build_all(args.sample)
    train, val, test = make_splits(master)
    log.info("split sizes train/val/test: %d/%d/%d", len(train), len(val), len(test))

    from ml.data.merge import task_label_stats
    stats = task_label_stats(master)
    log.info("\n%s", stats.to_string(index=False))

    serialize(train).to_parquet(os.path.join(args.out_dir, "train.parquet"))
    serialize(val).to_parquet(os.path.join(args.out_dir, "val.parquet"))
    serialize(test).to_parquet(os.path.join(args.out_dir, "test.parquet"))

    from ml.data.task_registry import get_registry
    with open(os.path.join(args.out_dir, "tasks.json"), "w") as f:
        json.dump(get_registry(), f, indent=2)
    write_dataset_card(args.out_dir, stats, len(master), (len(train), len(val), len(test)))

    push_to_hub(args.out_dir, args.hf_repo)
    log.info("DONE pipeline01")


if __name__ == "__main__":
    main()
