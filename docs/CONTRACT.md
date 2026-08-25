# Folliscan — Engineering Contract (single source of truth)

Every agent/module MUST conform to this contract. If you must deviate, note it in your final report.

## 1. Task Registry (fixed order = model output index)

Groups: `hair` (6), `tox` (12), `safety` (3). Total 21 binary tasks.

```json
[
  {"id": "SULT1A1_active",          "group": "hair",   "desc": "SULT1A1 bioactivation competence (minoxidil-type)"},
  {"id": "AR_antagonist",           "group": "hair",   "desc": "Androgen receptor antagonism"},
  {"id": "SRD5A1_inhibitor",        "group": "hair",   "desc": "5-alpha reductase type 1 inhibition"},
  {"id": "Wnt_bcatenin_activator",  "group": "hair",   "desc": "Wnt/beta-catenin pathway activation"},
  {"id": "FGF7_KGF_active",         "group": "hair",   "desc": "KGF/FGF7 fibroblast pathway activity"},
  {"id": "hair_growth_lit_positive","group": "hair",   "desc": "Literature-curated hair-growth positive"},
  {"id": "NR-AR",                   "group": "tox",    "desc": "Tox21 androgen receptor agonist/antagonist"},
  {"id": "NR-AR-LBD",               "group": "tox"},
  {"id": "NR-AhR",                  "group": "tox"},
  {"id": "NR-Aromatase",            "group": "tox"},
  {"id": "NR-ER",                   "group": "tox"},
  {"id": "NR-ER-LBD",               "group": "tox"},
  {"id": "NR-PPAR-gamma",           "group": "tox"},
  {"id": "SR-ARE",                  "group": "tox"},
  {"id": "SR-ATAD5",                "group": "tox"},
  {"id": "SR-HSE",                  "group": "tox"},
  {"id": "SR-MMP",                  "group": "tox"},
  {"id": "SR-p53",                  "group": "tox"},
  {"id": "skin_sensitizer",         "group": "safety", "desc": "Skin sensitization positive"},
  {"id": "cosing_prohibited",       "group": "safety", "desc": "Matches CosIng Annex II prohibited/restricted"},
  {"id": "irritancy_alert",         "group": "safety"}
]
```

This exact list lives in code at `ml/data/task_registry.py` as `TASK_REGISTRY: list[dict]` with fields `id, group, desc, index`. All components import it — never hardcode task lists elsewhere.

## 2. Dataset artifact (HuggingFace)

Repo: `{HF_USERNAME}/folliscan-data`. Files:
- `data/train.parquet`, `val.parquet`, `test.parquet`
- Columns: `smiles` (canonical, salt-stripped), `labels` (list[float] len 21, NaN where missing → store -1.0 sentinel AND a parallel `mask` list[int] of 0/1), `scaffold` (Bemis-Murcko), `source` (str)
- `tasks.json` (the registry), `dataset_card.md`, provenance CSVs per source.

Split: scaffold split grouped by Bemis-Murcko scaffold, 80/10/10, seed 42. No scaffold shared across splits.

## 3. Model artifact (HuggingFace)

Repo: `{HF_USERNAME}/folliscan-model`. Files:
- `model.pt` — `{"state_dict": ..., "config": {...}}`
- `config.json` — hyperparams (layers=4, dim=256, heads=8, dropout=0.15, etc.)
- `calib.json` — per-task conformal calibration parameters
- `metrics.json` — test metrics + ablation table

## 4. Python module APIs (`ml/` package; importable as `ml.*` from repo root)

```
ml.data.task_registry.TASK_REGISTRY                       # §1
ml.data.featurize.smiles_to_graph(smiles) -> torch_geometric.data.Data
  # node feats dim = 9 cat one-hot concat -> use fixed ATOM_FEATS dims;
  # Data attrs: x [n_atoms, F], edge_index [2,E], edge_attr [E,Fb], smiles str
ml.data.featurize.ATOM_F / BOND_F                          # feature spec dicts
ml.data.motifs.MOTIF_LIBRARY -> list[Motif(id,name,smarts,severity,hazard)]
ml.data.motifs.match_motifs(mol) -> list[(motif_id, atom_idx_tuple)]
ml.data.splits.scaffold_split(df) -> (train_df, val_df, test_df)
ml.models.folliscan_net.FolliscanNet(config) -> nn.Module
  # forward(data, motif_feats, ...) -> logits [B, 21]
ml.train.losses.FolliscanLoss(masked_bce + kendall_uncertainty_weights + pinn_penalty)
ml.train.uq.mc_predict(model, batch, n_samples=20) -> (mean_probs, std)
ml.train.uq.ConformalCalibrator.fit(scores, labels, mask, alpha=0.05); .predict_set(probs)
ml.train.metrics.evaluate(y_true, y_pred, mask) -> per-task AUROC/AUPRC/Brier/ECE + macro
ml.train.trainer.Trainer(model, loaders, cfg).run()  # pushes ckpt each epoch to HF Hub
ml.explain.attribution.motif_importance(model, data, motif_matches) -> {motif_id: float}
ml.explain.narrative.generate_narrative(payload) -> str   # Groq chat completion
```

Model forward contract: `FolliscanNet.forward(graph_batch, motif_vector)` where `motif_vector [B, n_motifs]` is the multi-hot match vector; internal SME embeds it. Pathway module is internal (fixed protein/pathway embedding table shipped in code).

## 5. Heavy backend REST API (HF Space `folliscan-api`)

Auth on all routes except `/health`: header `Authorization: Bearer <firebase-id-token>`; verify with firebase-admin (project `cabbage-guard`). 401 JSON on failure. Rate limit: 30 predictions/min/UID.

- `GET /health` → `{"status":"ok","model_loaded":bool}`
- `GET /tasks` → registry
- `POST /predict` body `{"smiles":"CCO"}` →
```json
{
  "valid": true, "input_smiles": "...", "canonical_smiles": "...",
  "molecule_svg": "<svg.../>",
  "predictions": [{"task_id","group","probability","std","conformal_set":[..]}],
  "motifs": [{"id","name","smarts","atom_indices":[..],"importance":0.83}],
  "pathways": [{"name":"Wnt/beta-catenin","relevance":0.72}],
  "alerts": [{"motif_id":"michael_acceptor","message":"..."}],
  "uncertainty_note": "high|moderate|low"
}
```
- `POST /explain` body `{"smiles":"...", "payload": <previous predict response>}` → `{"narrative":"..."}` (Groq model: resolve current production model dynamically via client.models.list(); fallback chain llama-3.3-70b-versatile → llama-3.1-8b-instant).

## 6. Light backend REST API (Render `folliscan-user-api`)

Same auth scheme. Firestore collection layout: `users/{uid}` doc (email, displayName, createdAt, prefs), subcollection `history/{docId}` (smiles, canonical_smiles, result payload, createdAt).
- `GET /me` → profile (auto-provision on first call — NEVER error if user "exists in another app")
- `PUT /me` → update displayName/prefs
- `GET /history?limit=20&cursor=` → `{items:[{id,...}], next_cursor}`
- `POST /history` → creates entry (server stamps uid/createdAt)
- `DELETE /history/{id}`
- CORS: allow Vercel domain(s) only.

## 7. Frontend ↔ backends wiring

Env vars (Vercel): `NEXT_PUBLIC_FIREBASE_API_KEY/AUTH_DOMAIN/PROJECT_ID/STORAGE_BUCKET/MESSAGING_SENDER_ID/APP_ID`, `NEXT_PUBLIC_API_URL` (HF space), `NEXT_PUBLIC_USER_API_URL` (Render). Auth mandatory: Next.js middleware checks session cookie; unauthenticated → redirect `/login`.

## 8. Environment variables / secrets (names everywhere)

`FIREBASE_CREDENTIALS_B64` (base64 admin SDK json), `FIREBASE_PROJECT_ID=cabbage-guard`, `GROQ_API_KEY`, `HF_TOKEN`, `HF_USERNAME`, `RENDER_DEPLOY_HOOK`, `VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`.

Secret source files live ONLY at:
`/home/anamitra/Downloads/API_Keys_and_Secrets/api keys for new set of projects/{bhumika-hf.txt, cabbage-guard-firebase-adminsdk-fbsvc-07fc830b13.json, git token, groq_api.txt, kaggle.json, render-api.txt, vercel.txt}`
NEVER copy their contents into any repo file. `.gitignore` blocks `.env*`, `*.key`, service-account json.

## 9. Rules for all agents

- Work ONLY inside your assigned paths under `/home/anamitra/folliscan`.
- NO git commands (add/commit/push) — the orchestrator commits once.
- No secrets in code/files/tests. Use env vars.
- Production quality: full implementations, no TODO stubs, no dummy returns.
- Add pytest tests under `tests/` matching your scope (CPU-runnable, tiny fixtures, mocked network).
