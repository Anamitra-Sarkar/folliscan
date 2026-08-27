# Folliscan — Deployment & Operations Runbook

## Live URLs

| Component | URL |
|---|---|
| Frontend | https://folliscan.vercel.app |
| Inference API | https://bhumika-tewari-282006-folliscan-api.hf.space |
| User API | https://folliscan-user-api.onrender.com (after activation, see below) |
| Model artifacts | https://huggingface.co/bhumika-tewari-282006/folliscan-model |
| Dataset | https://huggingface.co/bhumika-tewari-282006/folliscan-data |

## Verified runtime status — 2026-08-27

The Hugging Face inference endpoint returned HTTP 200 for `GET /health` with the complete JSON payload `{"status":"ok","model_loaded":false}` both with and without a Hub bearer token. This confirms the public liveness contract is functioning; it does **not** indicate model availability. The authenticated `/predict` route must continue to return an unavailable response while `model_loaded` is false, and no prediction, model-performance, or clinical-use claim may be made from this runtime state.

The earlier empty-body observation was a silent-request artifact rather than a missing endpoint. The next permitted operational action is to load only a reviewed, provenance-pinned model artifact through the documented Space configuration, then retest authenticated prediction and abstention behavior. Do not substitute a fixture model or synthetic result to make the service appear ready.

## CI/CD (all path-filtered, trigger on push to `main`)

| Workflow | Fires on | Action |
|---|---|---|
| `ci.yml` | any push/PR | pytest suite + frontend build |
| `deploy-backend.yml` | `backend/**`, `ml/**` | syncs backend+ml into HF Space → Space rebuilds |
| `deploy-userapi.yml` | `services/user-api/**` | builds Docker image → GHCR → triggers Render deploy via API |
| `deploy-frontend.yml` | `frontend/**` | vercel build + prod deploy |

GitHub repo variables/secrets: `HF_SPACE_ID` (variable), `HF_TOKEN`,
`VERCEL_TOKEN`, `VERCEL_ORG_ID`, `VERCEL_PROJECT_ID`, `RENDER_API_KEY`,
`RENDER_SERVICE_ID`, `GROQ_API_KEY`.

Secrets in HF Space settings: FIREBASE_CREDENTIALS_B64, FIREBASE_PROJECT_ID,
GROQ_API_KEY, HF_TOKEN, HF_MODEL_REPO.

## Training

```bash
modal run infra/run_training_modal.py --epochs 45          # full
modal run infra/run_training_modal.py --quick              # smoke
```

Each epoch pushes a checkpoint to the HF model repo (`checkpoints/<tag>/last.pt`,
`best.pt`) so an interrupted run never loses progress. Final artifacts:
`model.pt`, `config.json`, `calib.json` (conformal), `metrics.json`
(includes ablation table). The Space loads `model.pt` + `calib.json` at boot;
a redeploy is triggered automatically only if `backend/**` or `ml/**` changed —
after a training-only run, restart the Space once (or dispatch
`deploy-backend.yml` manually) to pick up new weights.

## Known manual steps / constraints

1. **GHCR package visibility** — GitHub provides no API to flip container
   package visibility; GITHUB_TOKEN pushes always land private, and Render
   cannot pull private images without dashboard-configured credentials.
   Fix: github.com/users/bhumika-tewari-282006/packages/container/folliscan-user-api
   → Package settings → Change visibility → Public.
   Alternative (no click): host the light API as a second small HF Space.
2. **Render billing** — creating services through the Render API from a free
   workspace returned HTTP 402 ("payment information required"). Add a card at
   https://dashboard.render.com/billing if prompted when activating the service.

## Verification commands

```bash
curl -s https://bhumika-tewari-282006-folliscan-api.hf.space/health
curl -s -o /dev/null -w '%{http_code}\n' -X POST \
  https://bhumika-tewari-282006-folliscan-api.hf.space/predict \
  -H 'Content-Type: application/json' -d '{"smiles":"CCO"}'   # expect 401
python3 -m pytest tests/ -q                                   # 29 tests
curl -s -o /dev/null -w '%{http_code}\n' https://folliscan.vercel.app
```

E2E test account: `folliscan.e2e.test@gmail.com` (Firebase Auth project
`cabbage-guard`) — used to verify the authenticated predict round-trip.
