"""Model loading & single-molecule inference pipeline."""

from __future__ import annotations

import json
import logging
import os

import numpy as np
import torch

from ml.data.featurize import smiles_to_graph, canonicalize
from ml.data.motifs import match_motifs, MOTIFS_BY_ID, hazard_flags, MOTIF_INDEX
from ml.data.render import render_svg
from ml.data.task_registry import TASK_REGISTRY
from ml.data.merge import build_master_dataset  # noqa: F401  (keeps parity checks importable)
from ml.models.folliscan_net import FolliscanNet
from ml.models.pathway_module import PATHWAYS
from ml.explain.attribution import motif_importance
from ml.train.uq import ConformalCalibrator, mc_predict

log = logging.getLogger(__name__)

TASK_INDEX = {t["id"]: t["index"] for t in TASK_REGISTRY}


class InferenceEngine:
    def __init__(self):
        self.model: FolliscanNet | None = None
        self.calib: ConformalCalibrator | None = None
        self.device = "cpu"

    # ---------- loading ----------
    def load(self, repo_id: str | None = None) -> bool:
        repo = repo_id or os.environ.get("HF_MODEL_REPO")
        if not repo:
            log.error("HF_MODEL_REPO not set; cannot load model")
            return False
        try:
            from huggingface_hub import hf_hub_download

            model_path = hf_hub_download(repo, "model.pt", repo_type="model")
            calib_path = hf_hub_download(repo, "calib.json", repo_type="model")
        except Exception as e:
            log.error("failed to fetch artifacts from %s: %s", repo, e)
            return False

        payload = torch.load(model_path, map_location="cpu")
        cfg = payload["config"]
        self.model = FolliscanNet(cfg)
        state = payload["state_dict"]
        # tolerate ablated runs saved with/without optional modules
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected:
            log.warning("unexpected keys ignored: %s", list(unexpected)[:5])
        self.model.to(self.device).eval()

        self.calib = ConformalCalibrator.load_json(calib_path)
        log.info("model loaded from %s (run_tag=%s)", repo, cfg.get("run_tag"))
        return True

    @property
    def ready(self) -> bool:
        return self.model is not None and self.calib is not None

    # ---------- prediction ----------
    def predict_payload(self, smiles: str) -> dict:
        assert self.ready
        canon = canonicalize(smiles)
        graph = smiles_to_graph(smiles)
        if canon is None or graph is None:
            return {"valid": False, "input_smiles": smiles,
                    "error": "invalid SMILES: molecule could not be parsed"}

        from torch_geometric.data import Batch

        batch = Batch.from_data_list([graph])
        mo = torch.tensor(
            [[1.0 if i in {MOTIF_INDEX[mid] for mid, _ in match_motifs(graph)} else 0.0
              for i in range(len(MOTIF_INDEX))]],
            dtype=torch.float32)

        mean, std = mc_predict(self.model, batch, mo, n_samples=20)
        sets = self.calib.predict_set(mean)

        predictions = []
        for t in TASK_REGISTRY:
            i = t["index"]
            predictions.append({
                "task_id": t["id"], "group": t["group"],
                "probability": round(float(mean[i]), 4),
                "std": round(float(std[i]), 4),
                "conformal_set": [round(float(sets[0][i][0]), 4),
                                  round(float(sets[0][i][1]), 4)],
                "desc": t["desc"],
            })

        raw_matches = match_motifs(canon)
        imp = motif_importance(self.model, batch, mo, group=None)
        imp_by_id = {r["motif_id"]: r["importance"] for r in imp}
        motifs_out = []
        alerts = []
        for mid, atoms in raw_matches:
            m = MOTIFS_BY_ID[mid]
            motifs_out.append({
                "id": mid, "name": m.name, "smarts": m.smarts,
                "atom_indices": [int(a) for a in atoms],
                "importance": float(imp_by_id.get(mid, 0.0)),
                "severity": m.severity,
            })
            if m.hazard and m.message:
                alerts.append({"motif_id": mid, "message": m.message})

        pathways = []
        if self.model.use_pathway:
            with torch.no_grad():
                out = self.model(batch, mo)
                rel = out["pathway_relevance"][0].detach().cpu().numpy()
            pathways = [{"name": p["name"], "group": p["group"], "role": p["role"],
                         "relevance": round(float(rel[i]), 4)}
                        for i, p in enumerate(PATHWAYS)]
            pathways.sort(key=lambda x: x["relevance"], reverse=True)

        flags = hazard_flags(canon)
        active_alerts = [k for k, v in flags.items() if v]

        mean_std = float(np.mean(std))
        uncertainty_note = ("high" if mean_std > 0.18 else
                            "moderate" if mean_std > 0.09 else "low")

        svg = render_svg(canon, highlight_atoms=[
            a for mid, atoms in raw_matches
            if MOTIFS_BY_ID[mid].severity != "info"
            for a in atoms][:24])

        return {
            "valid": True,
            "input_smiles": smiles,
            "canonical_smiles": canon,
            "molecule_svg": svg,
            "predictions": predictions,
            "motifs": motifs_out,
            "pathways": pathways,
            "alerts": alerts,
            "regulatory_flags": active_alerts,
            "uncertainty_note": uncertainty_note,
            "mean_epistemic_std": round(mean_std, 4),
        }


ENGINE = InferenceEngine()
