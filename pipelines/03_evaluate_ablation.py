"""Pipeline 03 — Ablation & baseline benchmark suite (runs on Kaggle GPU).

Runs: full model, -SME, -pathway, -PINN, MT-GT-only, multi-task GCN,
single-task GCN. Produces ablations.json with per-config macro AUROC/AUPRC,
bootstrap CIs and paired permutation tests of full vs each alternative.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_split  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("pipeline03")


def run():
    data_repo = os.environ["HF_DATA_REPO"]
    model_repo = os.environ["HF_MODEL_REPO"]
    quick = os.environ.get("ABLATION_QUICK", "0") == "1"

    train = load_split(data_repo, "train")
    val = load_split(data_repo, "val")
    test = load_split(data_repo, "test")

    import io

    import numpy as np
    import torch
    from huggingface_hub import HfApi, hf_hub_download

    from ml.data.task_registry import TASK_IDS
    from ml.models.baselines import MultiTaskGCN, SingleTaskGCN
    from ml.train.metrics import (bootstrap_ci, evaluate, macro_auroc_fn,
                                  permutation_test)
    from ml.train.trainer import Trainer, TrainConfig, predict_probs

    base_epochs = int(os.environ.get("EPOCHS", 60))
    epochs = max(8, base_epochs // 3) if quick else base_epochs
    fn = macro_auroc_fn(TASK_IDS)

    results: dict = {}
    test_preds: dict[str, tuple] = {}   # tag -> (probs, y_true, mask)

    def eval_and_store(trainer, tag):
        pt, yt, mt = predict_probs(trainer.model, trainer.test_loader, trainer.device)
        rep = evaluate(yt, pt, mt, TASK_IDS)["summary"]
        results[tag] = {"summary": rep}
        test_preds[tag] = (pt, yt, mt)
        log.info("%s -> %s", tag, rep)

    configs = {
        "full": dict(use_sme=True, use_pathway=True, use_pinn=True),
        "no_sme": dict(use_sme=False, use_pathway=True, use_pinn=True),
        "no_pathway": dict(use_sme=True, use_pathway=False, use_pinn=True),
        "no_pinn": dict(use_sme=True, use_pathway=True, use_pinn=False),
        "mtgt_only": dict(use_sme=False, use_pathway=False, use_pinn=False),
    }
    for tag, kw in configs.items():
        log.info("=== training %s ===", tag)
        tr = Trainer(train, val, test,
                     TrainConfig(epochs=epochs, batch_size=256, lr=3e-4,
                                 hf_model_repo=model_repo, seed=42, run_tag=tag, **kw))
        res = tr.run()
        # eval_and_store() is what actually creates results[tag] (= {"summary": rep});
        # the old order set results[tag]["best_val"] first, on a key that didn't
        # exist yet (real crash: KeyError: 'full', the first tag processed).
        eval_and_store(tr, tag)
        results[tag]["best_val"] = res["best_val_macro_auroc"]

    # GCN baselines reuse the Trainer harness with swapped model classes
    for name, cls in [("mt_gcn_baseline", MultiTaskGCN), ("st_gcn_baseline", SingleTaskGCN)]:
        log.info("=== training %s ===", name)
        tr = Trainer(train, val, test,
                     TrainConfig(epochs=epochs, batch_size=256, lr=3e-4,
                                 hf_model_repo=model_repo, seed=42, run_tag=name))
        tr.model = cls().to(tr.device)
        tr.opt = torch.optim.AdamW(tr.model.parameters(), lr=tr.cfg.lr,
                                   weight_decay=tr.cfg.weight_decay)
        res = tr.run()
        eval_and_store(tr, name)
        results[name]["best_val"] = res["best_val_macro_auroc"]

    # ---- statistics ----
    pf, yf, mf = test_preds["full"]
    results["full"]["auroc_ci95"] = bootstrap_ci(yf, pf, mf, fn, n_boot=500)
    for alt in ("mt_gcn_baseline", "st_gcn_baseline"):
        pa, ya, ma = test_preds[alt]

        def paired_metric(y, p_a, p_f, m_a, m_f):
            r = evaluate(y, p_f, m_f, TASK_IDS)["summary"]["macro_auroc"]
            return float("nan") if r is None else r

        # paired permutation on the difference in macro AUROC
        rng = np.random.default_rng(42)
        observed = (results["full"]["summary"]["macro_auroc"]
                    - results[alt]["summary"]["macro_auroc"])
        n = len(yf)
        count = 0
        perms = 200
        for _ in range(perms):
            flip = rng.random(n) < 0.5
            d = []
            for t in range(21):
                sel = mf[:, t] == 1
                if sel.sum() < 2:
                    continue
                a_sel, f_sel = pa[:, t][sel], pf[:, t][sel]
                a_sw = np.where(flip[sel], f_sel, a_sel)
                f_sw = np.where(flip[sel], a_sel, f_sel)
                from ml.train.metrics import _auroc
                ra, rf_ = _auroc(ya[:, t][sel], a_sw), _auroc(yf[:, t][sel], f_sw)
                if not np.isnan(ra) and not np.isnan(rf_):
                    d.append(rf_ - ra)
            diff_perm = np.nanmean(d) if d else 0.0
            if diff_perm >= observed:
                count += 1
        results["full"][f"paired_permutation_p_vs_{alt}"] = round((count + 1) / (perms + 1), 4)

    # merge into existing metrics.json
    existing: dict = {}
    try:
        p = hf_hub_download(model_repo, "metrics.json", repo_type="model",
                            token=os.environ.get("HF_TOKEN"))
        existing = json.load(open(p))
    except Exception:
        pass
    existing["ablations"] = results

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.upload_file(path_or_fileobj=json.dumps(existing, indent=2).encode(),
                    path_in_repo="metrics.json", repo_id=model_repo, repo_type="model",
                    commit_message="pipeline03: ablation study")

    log.info("\n%s", json.dumps({k: v.get("summary") for k, v in results.items()}, indent=2))
    log.info("DONE pipeline03")


if __name__ == "__main__":
    run()
