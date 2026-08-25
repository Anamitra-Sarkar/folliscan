"""Pipeline 02 — Train the full Folliscan model on Kaggle and publish artifacts.

Requires env: HF_TOKEN, HF_DATA_REPO, HF_MODEL_REPO.
Steps: pull parquet splits from HF -> train (per-epoch ckpt pushes) ->
conformal calibration -> test metrics -> upload model.pt/config/calib/metrics.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline02")


def main():
    data_repo = os.environ["HF_DATA_REPO"]
    model_repo = os.environ["HF_MODEL_REPO"]

    train = load_split(data_repo, "train")
    val = load_split(data_repo, "val")
    test = load_split(data_repo, "test")
    log.info("splits loaded: %d/%d/%d", len(train), len(val), len(test))

    from ml.train.trainer import Trainer, TrainConfig

    cfg = TrainConfig(
        epochs=int(os.environ.get("EPOCHS", 60)),
        batch_size=int(os.environ.get("BATCH_SIZE", 256)),
        lr=float(os.environ.get("LR", 3e-4)),
        hf_model_repo=model_repo,
        run_tag="full",
        use_sme=True,
        use_pathway=True,
        use_pinn=True,
    )
    trainer = Trainer(train, val, test, cfg)
    result = trainer.run()
    log.info("training done: best val macro AUROC %.4f @ epoch %d",
             result["best_val_macro_auroc"], result["best_epoch"])

    out = trainer.calibrate_and_evaluate()
    metrics = {
        "run": "full",
        **out["metrics"],
        "train_history": result["history"],
        "best_val_macro_auroc": result["best_val_macro_auroc"],
    }

    # ---- publish artifacts ----
    import io
    import torch
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(model_repo, repo_type="model", exist_ok=True)

    buf = io.BytesIO()
    torch.save({
        "state_dict": trainer.model.state_dict(),
        "config": {**trainer.model.config, "use_sme": True, "use_pathway": True},
    }, buf)
    files = {
        "model.pt": buf.getvalue(),
        "calib.json": json.dumps(out["calib"].to_dict()),
        "metrics.json": json.dumps(metrics, indent=2),
        "config.json": json.dumps(trainer.model.config, indent=2),
    }
    for name, blob in files.items():
        api.upload_file(path_or_fileobj=blob, path_in_repo=name, repo_id=model_repo,
                        repo_type="model", commit_message=f"pipeline02: {name}")
    log.info("published artifacts to https://huggingface.co/%s", model_repo)

    summary = metrics["summary"]
    log.info("TEST macro AUROC=%s AUPRC=%s conformal coverage=%s",
             summary.get("macro_auroc"), summary.get("macro_auprc"),
             metrics["conformal"]["coverage_validation_mean"])
    log.info("DONE pipeline02")


if __name__ == "__main__":
    main()
