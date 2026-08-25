# Folliscan

**Hair-Tox-Safe Framework** — a multi-task graph neural network that screens cosmetic
ingredient candidates for **hair-health activity**, **toxicity**, and **cosmetic safety**
in one pass, with statistically calibrated uncertainty and mechanistic explanations.

## What it does

Given a molecule (SMILES), Folliscan returns:

- **21 endpoint predictions** — 6 hair-health tasks (SULT1A1 bioactivation, AR antagonism,
  5α-reductase inhibition, Wnt/β-catenin activation, FGF7/KGF activity, literature-curated
  hair-growth signal), 12 Tox21 nuclear-receptor/stress assays, 3 safety tasks
  (skin sensitization, CosIng Annex II match, irritancy alert)
- **Uncertainty quantification** — Monte-Carlo Dropout variance + split **conformal
  prediction intervals** (95% nominal coverage)
- **Mechanistic interpretability** — matched structural motifs/toxicophores ranked by
  gradient-informed importance, attention over a pathway knowledge graph
  (Wnt/Shh/BMP/Notch cascades + AOP anchors), and an LLM expert readout (Groq)

## Architecture

```
frontend/            Next.js 14 app (Vercel)  — Firebase Auth, analysis dashboard, history
services/user-api/   FastAPI user/profile/history service (Render) — Firestore Admin
backend/             FastAPI inference service (HuggingFace Space) — loads model from HF Hub
ml/                  Shared ML library (featurization, MT-GT model, training, UQ, explain)
pipelines/           Kaggle/Modal pipelines: 01 dataset build · 02 train · 03 ablations
infra/               Modal GPU training launcher (secrets injected at runtime only)
.github/workflows/   CI + path-filtered deploys (backend→HF Space, user-api→Render, UI→Vercel)
```

Model: GPS-style graph transformer (4 layers × 256 dim) + Substructure Motif Encoder
(62 curated SMARTS pharmacophores/toxicophores) + Pathway-Informed attention module +
PINN-style regulatory constraint loss. Trained with scaffold-grouped splits
(Bemis–Murcko, leakage-checked) on ChEMBL + Tox21 + curated CosIng/sensitization data.

## Deployment topology

| Component | Platform | Trigger |
|---|---|---|
| Frontend | Vercel | push to `main` touching `frontend/**` |
| Inference API | HuggingFace Space (Docker) | push to `main` touching `backend/**` or `ml/**` |
| User API | Render (Docker from GHCR) | push to `main` touching `services/user-api/**` |
| Model artifacts | HuggingFace Hub | pushed by training runs each epoch |
| Datasets | HuggingFace Hub | pushed by pipeline 01 |

All secrets live exclusively in platform secret stores (GitHub Actions secrets,
HF Space secrets, Render/Vercel env vars, runtime-injected env in Modal/Kaggle).
Nothing is ever committed.

## Local development

```bash
pip install -r backend/requirements.txt
python -m pytest tests/          # CPU-runnable unit suite

cd frontend && npm install && npm run dev
# required env: NEXT_PUBLIC_FIREBASE_*, NEXT_PUBLIC_API_URL, NEXT_PUBLIC_USER_API_URL
```

## Training

```bash
python3 infra/run_training_modal.py            # full run on Modal T4
python3 infra/run_training_modal.py --quick    # smoke run
```

Artifacts land at `hf.co/<user>/folliscan-model` and `hf.co/<user>/folliscan-data`.

## Scientific notes

- Labels: pChEMBL-thresholded ChEMBL bioactivities (consensus vote per molecule),
  Tox21 assay calls, LLNA-informed sensitization curation, EU CosIng Annex II
  representative set resolved via PubChem.
- Split protocol forbids shared Bemis-Murcko scaffolds across train/val/test;
  acyclic molecules become singleton groups.
- Evaluation: AUROC/AUPRC/Brier/ECE per task + macro, bootstrap 95% CIs, paired
  permutation tests vs baselines (multi-task GCN, single-task GCN) and ablations
  (−SME, −pathway, −PINN, MT-GT-only).

*Research tool for ingredient screening support; not a substitute for regulatory testing.*
