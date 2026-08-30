# Deployment Verification — 27 August 2026

FolliScan deployment `dpl_AVTzeYatC2hVRDeVraK1JfPZGkNX` failed because the Git-linked Vercel project was configured at the repository root while its Next application lives in `frontend/`. Vercel’s build log reported that it could not find a `pages` or `app` directory.

The existing Vercel project was corrected to use `frontend` as its root directory. A verified-author trigger commit, `9f48a6f9d319b1b425f6fdc6241d693badb6eb00`, then produced deployment `dpl_FrUuAsJCgTc7LUoG4SRVsNmXCgXs`, which Vercel marked `READY`. Vercel Authentication protects that deployment; an authorized temporary verification request returned HTTP 200 and the page title `Folliscan — Cosmetic ingredient intelligence`. Deployment protection was not changed.

The local release gate passed after installing the documented PyTorch Geometric test dependency: the Next frontend production build completed, 32 Python tests passed, Python compilation passed, and diff validation passed. The deployed Hugging Face `/health` endpoint also returned `{"status":"ok","model_loaded":false}`. The deployment recovery did not load, create, evaluate, or promote a model, and no prediction request was sent.

## Real training completion and production promotion — 30 August 2026

`infra/run_training_modal.py` was run for real on Modal T4 GPU against the real HF dataset (`bhumika-tewari-282006/folliscan-data`, 43460/5166/4865 train/val/test rows). Two real bugs were found and fixed before this could complete: `Trainer`'s three `torch_geometric.loader.DataLoader` instances silently discarded the project's own `collate_fn` (PyG's `DataLoader.__init__` does `kwargs.pop('collate_fn', None)` and always substitutes its own `Collater`, confirmed by reading the installed 2.8.0 source directly), causing `for g, mo, y, m in loader` to unpack a batched dict's KEYS instead of values; and `pipelines/02_train_model.py` passed un-encoded JSON strings to `HfApi.upload_file`'s `path_or_fileobj`, which treats a plain `str` as a local file path, not literal content.

With both fixed, real training completed:

```
TEST macro AUROC=0.8549 AUPRC=0.5508 conformal coverage=0.9558
```

Real artifacts (`model.pt`, `calib.json`, `metrics.json`, `config.json`) published to `https://huggingface.co/bhumika-tewari-282006/folliscan-model`.

**Production promotion:** the live HF Space (`bhumika-tewari-282006/folliscan-api`) already had `HF_MODEL_REPO` configured as a secret pointing at this repo — it simply had no real weights to load until now. A stray duplicate variable of the same name (added by mistake during verification, immediately deleted) briefly caused a `CONFIG_ERROR` ("Collision on variables and secrets names"); once removed the Space restarted cleanly.

```
curl https://bhumika-tewari-282006-folliscan-api.hf.space/health
{"status":"ok","model_loaded":true}
```

Real model, real metrics, live in production. Frontend at `https://folliscan.vercel.app` should now be able to serve real predictions rather than a "no model" state — not independently re-verified end-to-end in this entry, but the backend's own health contract confirms the model is loaded.
