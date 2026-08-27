# Deployment Verification — 27 August 2026

FolliScan deployment `dpl_AVTzeYatC2hVRDeVraK1JfPZGkNX` failed because the Git-linked Vercel project was configured at the repository root while its Next application lives in `frontend/`. Vercel’s build log reported that it could not find a `pages` or `app` directory.

The existing Vercel project was corrected to use `frontend` as its root directory. A verified-author trigger commit, `9f48a6f9d319b1b425f6fdc6241d693badb6eb00`, then produced deployment `dpl_FrUuAsJCgTc7LUoG4SRVsNmXCgXs`, which Vercel marked `READY`. Vercel Authentication protects that deployment; an authorized temporary verification request returned HTTP 200 and the page title `Folliscan — Cosmetic ingredient intelligence`. Deployment protection was not changed.

The local release gate passed after installing the documented PyTorch Geometric test dependency: the Next frontend production build completed, 32 Python tests passed, Python compilation passed, and diff validation passed. The deployed Hugging Face `/health` endpoint also returned `{"status":"ok","model_loaded":false}`. The deployment recovery did not load, create, evaluate, or promote a model, and no prediction request was sent.
