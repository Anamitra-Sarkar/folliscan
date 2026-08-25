"""Folliscan training on Modal GPU (dataset build -> train -> ablations).

Secrets are read from local files ONLY at launch time and injected as runtime
environment variables into the Modal container. They are never written into
this repo. Run:  python3 infra/run_training_modal.py [--quick] [--skip-ablations]
"""

import argparse
import base64
import json
import os
import subprocess
import sys

import modal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KEYS_DIR = "/home/anamitra/Downloads/API_Keys_and_Secrets/api keys for new set of projects"

image = (
    modal.Image.debian_slim(python_version="3.10")
    .pip_install(
        "torch>=2.1,<2.4",
        "torch_geometric>=2.5",
        "rdkit==2023.9.6",
        "numpy<2",
        "pandas>=2.0",
        "pyarrow>=15",
        "scikit-learn>=1.4",
        "huggingface_hub>=0.23",
        "requests>=2.31",
        "tabulate>=0.9",
    )
    .add_local_dir(os.path.join(ROOT, "ml"), remote_path="/root/folliscan/ml")
    .add_local_dir(os.path.join(ROOT, "pipelines"), remote_path="/root/folliscan/pipelines")
)

app = modal.App("folliscan-training", image=image)


def _read_key(fn: str) -> str:
    with open(os.path.join(KEYS_DIR, fn)) as f:
        return f.read().strip()


def _load_runtime_env() -> dict:
    """Resolve HF username from token; build runtime env (never logged)."""
    import requests

    hf_token = _read_key("bhumika-hf.txt")
    r = requests.get("https://huggingface.co/api/whoami-v2",
                     headers={"Authorization": f"Bearer {hf_token}"}, timeout=20)
    if r.status_code != 200:
        raise RuntimeError(f"HF token check failed: HTTP {r.status_code}")
    hf_user = r.json()["name"]

    admin_json = open(os.path.join(
        KEYS_DIR, "cabbage-guard-firebase-adminsdk-fbsvc-07fc830b13.json")).read()

    return {
        "HF_TOKEN": hf_token,
        "HF_USERNAME": hf_user,
        "HF_DATA_REPO": f"{hf_user}/folliscan-data",
        "HF_MODEL_REPO": f"{hf_user}/folliscan-model",
        "FIREBASE_CREDENTIALS_B64": base64.b64encode(admin_json.encode()).decode(),
        "FIREBASE_PROJECT_ID": "cabbage-guard",
    }


@app.function(gpu="T4", timeout=60 * 60 * 6)
def run_pipeline(script: str, env: dict, extra_env: dict | None = None):
    os.environ.update({**env, **(extra_env or {})})
    workdir = "/root/folliscan"
    proc = subprocess.run([sys.executable, "-u", f"pipelines/{script}"],
                          cwd=workdir)
    if proc.returncode != 0:
        raise RuntimeError(f"pipeline {script} failed with exit code {proc.returncode}")
    return f"{script} completed"


@app.local_entrypoint()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="reduced epochs / smoke run")
    ap.add_argument("--skip-dataset", action="store_true", help="reuse existing HF dataset")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-ablations", action="store_true")
    args = ap.parse_args()

    env = _load_runtime_env()
    print("launching on Modal with model repo:", env["HF_MODEL_REPO"])

    if not args.skip_dataset:
        run_pipeline.remote("01_build_dataset.py", env)

    train_extra = {}
    ablation_extra = {}
    if args.quick:
        train_extra["EPOCHS"] = "12"
        ablation_extra["EPOCHS"] = "12"
        ablation_extra["ABLATION_QUICK"] = "1"

    if not args.skip_train:
        print(run_pipeline.remote("02_train_model.py", env, train_extra))
    if not args.skip_ablations:
        print(run_pipeline.remote("03_evaluate_ablation.py", env, ablation_extra))
