"""Shared helpers for Folliscan pipelines."""

import json
import os


def load_split(repo_id: str, name: str):
    """Download one parquet split from the HF dataset repo into a DataFrame."""
    from huggingface_hub import hf_hub_download
    import pandas as pd

    path = hf_hub_download(repo_id, f"{name}.parquet", repo_type="dataset",
                           token=os.environ.get("HF_TOKEN"))
    df = pd.read_parquet(path)
    df["labels"] = df["labels"].map(lambda s: json.loads(s))
    df["mask"] = df["mask"].map(lambda s: json.loads(s))
    return df
