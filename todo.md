# Project TODO

- [x] Record the verified inference health contract: `/health` returns transport liveness with `model_loaded: false`, and no model is treated as ready or promoted.
- [x] Diagnose and repair the current Vercel deployment error for commit `f48f53d` without changing the intentional no-model readiness state or introducing synthetic prediction data.
- [x] Reverify the repaired public frontend and the existing no-model API boundary after any deployment fix.
- [x] Diagnose and repair the blank public production interface observed on 2026-08-27, then reverify the frontend without changing the no-model inference state.
- [x] Add a committed non-interactive frontend lint configuration so quality verification cannot stall on an ESLint setup prompt.
- [x] Review and safely remediate FolliScan’s production dependency advisories through controlled compatible upgrades; do not use forced automated upgrades.
- [x] Migrate the unchanged protected-route guard from Next’s deprecated middleware convention to the supported proxy convention after the controlled Next 16 upgrade.
- [x] Deploy and reverify the canonical FolliScan public landing page, protected-route redirect, and no-model boundary after the frontend correction.
- [ ] Correct the frontend deployment workflow’s duplicated `frontend/frontend` working-directory error and verify the path-filtered GitHub deployment workflow succeeds.
