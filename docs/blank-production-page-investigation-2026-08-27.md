# Blank Production Page Investigation — 2026-08-27

## Observed production behavior

The canonical public page at `https://folliscan.vercel.app/` returned HTTP content and the expected document title, but the browser-visible interface remained a near-empty screen containing only the `Folliscan` loading fallback. A follow-up browser view did not resolve the fallback and showed no interactive elements or browser-console output.

The rendered HTML contained the Next.js static loading shell, `ClientPageRoot`, and the deployed `app/page` client bundle, but no rendered application content. This points to a client-startup or loading-state issue rather than an inference response. No model request was made, and the existing provider contract remains `model_loaded: false`.

## Scope

This investigation does not alter the research-only/no-model inference boundary. The next step is to trace the client page’s bootstrap, configuration, and loading-state handling locally, add a regression for the resolved behavior, then redeploy and recheck the canonical page.

## Local correction and validation

The root route was a permanent static loading fallback rather than a real landing page. It now renders an explicit research-only overview with a link to the existing protected sign-in flow, an evidence-review explanation, and a visible no-live-result/model-release-pending state. It does not import the authenticated API client or make an inference request.

The standard frontend build now runs a source-level public-boundary regression and explicitly sets the production environment. A committed lint configuration and compatible development dependencies make the existing lint command non-interactive. The local gate passed: public-boundary regression, lint with no warnings or errors, production build of all five routes, 32 Python tests, Python compilation, and repository whitespace checks. The observed FastAPI startup-hook and Starlette test-client messages are upstream deprecation warnings; they are not test failures or claims about model readiness.

The dependency remediation was explicit and reviewable rather than an automated audit fix: Next.js `16.3.3`, Firebase `12.18.0`, PostCSS `8.5.6`, and the matching ESLint 9/Next configuration were installed. The production-only dependency audit then reported zero known vulnerabilities. The equivalent protected-route guard was moved from the deprecated `middleware` convention to Next’s supported `proxy` convention without changing its cookie-gating logic. External Vercel verification remains pending until this tested revision is deployed.

## Production verification

Verified-author commit `19f365d` produced Vercel deployment `dpl_4NHDiPizsd4cHBWx5S3AJQVeaLFG`, which reached `READY` for the production target and aliases `https://folliscan.vercel.app`. A cache-busting browser check confirmed that the canonical page now renders the research-only landing rather than an indefinite loading screen. It displays the no-live-result and model-release-pending state and routes to the existing sign-in flow. An anonymous check of `/dashboard` redirected to `/login` with the requested protected path preserved. No prediction was requested; the provider continues to report no loaded model.

The repository’s separate frontend deployment workflow initially failed because its `frontend` working directory was combined with the Vercel project’s existing `frontend` root, causing an attempted `frontend/frontend/package.json` lookup. The workflow now invokes the Vercel CLI from the repository root so that root is applied once. GitHub Actions run `33123313569` completed successfully for frontend-scoped commit `116ce7d`; it validates the corrected path-filtered deployment route without altering model, data, or release configuration.
