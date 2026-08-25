"""Training pipeline with per-epoch checkpoint pushes to the HuggingFace Hub."""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field, asdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch_geometric.loader import DataLoader as PyGLoader

from ml.data.featurize import smiles_to_graph
from ml.data.motifs import match_motifs, motif_multihot, N_MOTIFS
from ml.models.folliscan_net import FolliscanNet
from .losses import FolliscanLoss
from .uq import ConformalCalibrator, mc_predict
from .metrics import evaluate, bootstrap_ci, macro_auroc_fn
from ml.data.task_registry import TASK_IDS

log = logging.getLogger(__name__)


class MoleculeTaskDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        assert "smiles" in df.columns and "labels" in df.columns and "mask" in df.columns
        self.rows = [
            {
                "graph": smiles_to_graph(r.smiles),
                "labels": torch.tensor(r.labels, dtype=torch.float32),
                "mask": torch.tensor(r.mask, dtype=torch.float32),
                "motifs": torch.tensor(motif_multihot(r.smiles), dtype=torch.float32),
            }
            for r in df.itertuples()
        ]
        self.rows = [r for r in self.rows if r["graph"] is not None]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


def collate(batch):
    from torch_geometric.data import Batch

    graphs = Batch.from_data_list([b["graph"] for b in batch])
    motifs = torch.stack([b["motifs"] for b in batch])
    labels = torch.stack([b["labels"] for b in batch])
    mask = torch.stack([b["mask"] for b in batch])
    return graphs, motifs, labels, mask


@torch.no_grad()
def predict_probs(model, loader, device) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic pass -> (probs, labels, mask) each [N, 21]."""
    model.eval()
    P, Y, M = [], [], []
    for g, mo, y, m in loader:
        g = g.to(device)
        out = model(g, mo.to(device))
        P.append(torch.sigmoid(out["logits"]).cpu())
        Y.append(y)
        M.append(m)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(M).numpy()


def _val_macro_auroc(model, loader, device) -> float:
    p, y, m = predict_probs(model, loader, device)
    s = evaluate(y, p, m, TASK_IDS)["summary"]["macro_auroc"]
    return float("nan") if s is None else s


@dataclass
class TrainConfig:
    epochs: int = 60
    batch_size: int = 256
    lr: float = 3e-4
    weight_decay: float = 1e-5
    warmup_epochs: float = 3.0
    patience: int = 8
    n_mc_samples_calib: int = 10
    alpha_conformal: float = 0.05
    use_pinn: bool = True
    pinn_lambda: float = 0.3
    use_sme: bool = True
    use_pathway: bool = True
    num_workers: int = 2
    seed: int = 42
    hf_model_repo: str | None = None       # e.g. "<user>/folliscan-model"
    run_tag: str = "full"
    extra: dict = field(default_factory=dict)


class Trainer:
    def __init__(self, train_df, val_df, test_df, cfg: TrainConfig,
                 model_config_overrides: dict | None = None, device: str | None = None):
        torch.manual_seed(cfg.seed)
        np.random.seed(cfg.seed)
        self.cfg = cfg
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.train_loader = PyGLoader(MoleculeTaskDataset(train_df), batch_size=cfg.batch_size,
                                      shuffle=True, num_workers=cfg.num_workers)
        self.val_loader = PyGLoader(MoleculeTaskDataset(val_df), batch_size=cfg.batch_size * 2,
                                    shuffle=False, num_workers=cfg.num_workers)
        self.test_loader = PyGLoader(MoleculeTaskDataset(test_df), batch_size=cfg.batch_size * 2,
                                     shuffle=False, num_workers=cfg.num_workers)

        model_cfg = {"use_sme": cfg.use_sme, "use_pathway": cfg.use_pathway}
        model_cfg.update(model_config_overrides or {})
        self.model = FolliscanNet(model_cfg).to(self.device)
        self.criterion = FolliscanLoss(use_pinn=cfg.use_pinn, pinn_lambda=cfg.pinn_lambda).to(self.device)
        self.opt = torch.optim.AdamW(
            list(self.model.parameters()) + list(self.criterion.parameters()),
            lr=cfg.lr, weight_decay=cfg.weight_decay)

        def lr_lambda(epoch):
            if epoch < cfg.warmup_epochs:
                return (epoch + 1) / max(cfg.warmup_epochs, 1e-6)
            t = (epoch - cfg.warmup_epochs) / max(cfg.epochs - cfg.warmup_epochs, 1e-6)
            return 0.5 * (1 + math.cos(math.pi * min(t, 1.0)))

        self.sched = torch.optim.lr_scheduler.LambdaLR(self.opt, lr_lambda)

    # ---------- HF Hub checkpointing ----------
    def _hf_push(self, name: str, blob: bytes):
        repo = self.cfg.hf_model_repo
        if not repo:
            return False
        try:
            from huggingface_hub import HfApi
            token = os.environ.get("HF_TOKEN")
            api = HfApi(token=token)
            api.create_repo(repo, repo_type="model", exist_ok=True, private=False)
            api.upload_file(path_or_fileobj=blob, path_in_repo=name, repo_id=repo,
                            repo_type="model", commit_message=f"ckpt {name} ({self.cfg.run_tag})")
            return True
        except Exception as e:
            log.warning("HF push failed for %s: %s", name, e)
            return False

    def _state_blob(self, extra: dict | None = None) -> bytes:
        payload = {
            "state_dict": {k: v.cpu() for k, v in self.model.state_dict().items()},
            "config": {**self.model.config, "use_sme": self.cfg.use_sme,
                       "use_pathway": self.cfg.use_pathway, "run_tag": self.cfg.run_tag},
        }
        if extra:
            payload.update(extra)
        import io
        buf = io.BytesIO()
        torch.save(payload, buf)
        return buf.getvalue()

    # ---------- loop ----------
    def run(self) -> dict:
        best_val, best_epoch, bad = float("-inf"), -1, 0
        history = []
        for epoch in range(self.cfg.epochs):
            t0 = time.time()
            self.model.train()
            running = {}
            for g, mo, y, m in self.train_loader:
                g, mo, y, m = g.to(self.device), mo.to(self.device), y.to(self.device), m.to(self.device)
                flags = [hazard_flags_from_multihot(mm) for mm in mo.detach().cpu().numpy()] \
                    if self.cfg.use_pinn else None
                out = self.model(g, mo)
                losses = self.criterion(out["logits"], y, m, flags)
                self.opt.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                self.opt.step()
                for k, v in losses.items():
                    running[k] = running.get(k, 0.0) + float(v)
            self.sched.step()

            val_auroc = _val_macro_auroc(self.model, self.val_loader, self.device)
            elapsed = round(time.time() - t0, 1)
            entry = {"epoch": epoch, "val_macro_auroc": val_auroc,
                     "train_total": running.get("total", 0.0) / max(len(self.train_loader), 1),
                     "sec": elapsed}
            history.append(entry)
            log.info("epoch %d val_macro_auroc=%.4f (%ss)", epoch, val_auroc, elapsed)

            ok = self._hf_push(f"checkpoints/{self.cfg.run_tag}/last.pt",
                               self._state_blob({"history": history}))
            if not math.isnan(val_auroc) and val_auroc > best_val:
                best_val, best_epoch, bad = val_auroc, epoch, 0
                self.best_state = {k: v.detach().cpu().clone()
                                   for k, v in self.model.state_dict().items()}
                self._hf_push(f"checkpoints/{self.cfg.run_tag}/best.pt",
                              self._state_blob({"best_val_macro_auroc": best_val, "epoch": epoch}))
            else:
                bad += 1
                if bad >= self.cfg.patience:
                    log.info("early stopping at epoch %d", epoch)
                    break

        # ---- restore best local state ----
        if getattr(self, "best_state", None) is not None:
            self.model.load_state_dict(self.best_state)
        result = {"config": asdict(self.cfg), "history": history,
                  "best_val_macro_auroc": best_val, "best_epoch": best_epoch}
        return result

    def load_best_state(self, blob_bytes: bytes):
        import io
        buf = io.BytesIO(blob_bytes)
        sd = torch.load(buf, map_location=self.device)["state_dict"]
        self.model.load_state_dict(sd)

    # ---- post-training: conformal calibration + test evaluation ----
    def calibrate_and_evaluate(self) -> dict:
        calib = ConformalCalibrator(alpha=self.cfg.alpha_conformal)
        pv, yv, mv = mc_predict_dataset(self.model, self.val_loader, self.device,
                                        n_samples=self.cfg.n_mc_samples_calib)
        calib.fit(pv, yv, mv)
        cov_val = calib.coverage(pv, yv, mv)

        pt, yt, mt = predict_probs(self.model, self.test_loader, self.device)
        report = evaluate(yt, pt, mt, TASK_IDS)
        fn = macro_auroc_fn(TASK_IDS)
        report["summary"]["auroc_bootstrap_ci95"] = bootstrap_ci(yt, pt, mt, fn, n_boot=500)
        report["conformal"] = {
            "alpha": self.cfg.alpha_conformal,
            "coverage_validation_mean": cov_val.get("mean"),
        }
        return {"metrics": report, "calib": calib}


def hazard_flags_from_multihot(multihot_row: np.ndarray) -> dict[str, bool]:
    """Reconstruct hard-rule alert flags directly from a motif multi-hot vector."""
    from ml.data.motifs import MOTIF_LIBRARY, MOTIF_INDEX

    ids = {MOTIF_LIBRARY[i].id for i, v in enumerate(multihot_row) if v > 0}
    mutagenic = {"aromatic_nitro", "nitroso", "n_nitrosamine", "sulfonate_ester",
                 "benzyl_halide", "hydrazine", "triazene", "diazonium", "epoxide"}
    sens = {"michael_acceptor_enone", "acrylamide", "alpha_methylene_gamma_butyrolactone",
            "cinnamyl", "isothiazolinone", "para_phenylenediamine", "aldehyde"}
    irrit = {"quaternary_ammonium", "sulfate_ester", "long_chain_surfactant", "peroxide"}
    cosing = {"hydroquinone", "quinone"}
    return {
        "mutagenicity_alert": bool(ids & mutagenic),
        "sensitization_alert": bool(ids & sens),
        "irritancy_alert": bool(ids & irrit),
        "cosing_alert": bool(ids & cosing),
    }


@torch.no_grad()
def mc_predict_dataset(model, loader, device, n_samples: int = 20) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """MC-dropout probabilities over a dataset loader -> (probs, labels, mask)."""
    from .uq import enable_mc_dropout

    enable_mc_dropout(model)
    P, Y, M = [], [], []
    for g, mo, y, m in loader:
        g = g.to(device)
        acc = []
        for _ in range(n_samples):
            out = model(g, mo.to(device))
            acc.append(torch.sigmoid(out["logits"]).cpu())
        P.append(torch.stack(acc).mean(0))
        Y.append(y)
        M.append(m)
    return torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(M).numpy()
