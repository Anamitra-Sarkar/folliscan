# Project TODO

- [x] Record the verified inference health contract: `/health` returns transport liveness with `model_loaded: false`, and no model is treated as ready or promoted.
- [x] Diagnose and repair the current Vercel deployment error for commit `f48f53d` without changing the intentional no-model readiness state or introducing synthetic prediction data.
- [x] Reverify the repaired public frontend and the existing no-model API boundary after any deployment fix.
