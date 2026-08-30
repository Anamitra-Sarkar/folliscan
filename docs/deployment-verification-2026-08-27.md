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

## Real ablation suite — 30 August 2026

`pipelines/03_evaluate_ablation.py` completed a full real run on Modal T4 (all configs retrained end-to-end, ~epochs=60 each, against the same real HF dataset split as above). Two more real bugs were found and fixed along the way: `HfApi.upload_file` again needed `.encode("utf-8")` on JSON-string values (same class of bug as the training script), and `results[tag]["best_val"]` was being set before `eval_and_store()` had created `results[tag]` (real `KeyError: 'full'` on the first tag processed) — fixed by reordering.

Real test macro AUROC/AUPRC for every configuration:

| config | macro AUROC | macro AUPRC |
|---|---|---|
| full | 0.8354 | 0.5202 |
| no_sme (motif encoder removed) | **0.678** | **0.2613** |
| no_pathway | 0.8475 | 0.5390 |
| no_pinn | 0.8388 | 0.5259 |
| mtgt_only (motif+pathway+PINN all off) | 0.8280 | 0.5114 |
| mt_gcn_baseline | 0.7742 | 0.3625 |
| st_gcn_baseline | 0.7975 | 0.4025 |

**Honest read:** the SME (structure/motif encoder) is the component that actually
carries the model's real performance — removing it collapses macro AUROC from
0.835 to 0.678 and AUPRC from 0.520 to 0.261, by far the largest effect of any
ablation. The pathway-knowledge and PINN (physics-informed) components show **no
measurable benefit** in this run — `no_pathway` and `no_pinn` both score
marginally *higher* than the full model, and `mtgt_only` (all three switched off)
still beats both GCN baselines. This should not be read as proof those components
are useless (a single run at ~60 epochs isn't a rigorous ablation study — no
repeated seeds, no significance testing was performed here), but it is a real,
honest signal that the pathway/PINN machinery is not currently pulling its weight
relative to the motif encoder, and any future architecture work on this project
should prioritize the SME path over the pathway/PINN paths. Both custom-architecture
variants (full and mtgt_only) beat both plain-GCN baselines, confirming the overall
approach (task-registry-aware heads + motif features) has real value over a generic
GCN, independent of the pathway/PINN question.
