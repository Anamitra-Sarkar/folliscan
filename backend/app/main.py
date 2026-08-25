"""Folliscan heavy inference API — deployed as a HuggingFace Docker Space."""

from __future__ import annotations

import logging
import os
import time
from collections import defaultdict, deque

import httpx
from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from ml.data.task_registry import get_registry
from .auth import init_firebase, require_user, AuthError
from .inference import ENGINE

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("folliscan-api")

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*").split(",")
RATE_LIMIT_PER_MIN = int(os.environ.get("RATE_LIMIT_PER_MIN", "30"))

app = FastAPI(title="Folliscan API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGIN,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

_rate: dict[str, deque] = defaultdict(lambda: deque(maxlen=200))


def _check_rate(uid: str):
    now = time.time()
    q = _rate[uid]
    while q and q[0] < now - 60:
        q.popleft()
    if len(q) >= RATE_LIMIT_PER_MIN:
        raise HTTPException(429, "rate limit exceeded (30 predictions/minute)")
    q.append(now)


def _user(authorization: str | None) -> dict:
    try:
        return require_user(authorization)
    except AuthError as e:
        raise HTTPException(401, str(e))


@app.on_event("startup")
async def startup():
    init_firebase()
    ok = ENGINE.load()
    log.info("model ready=%s", ok)


class PredictIn(BaseModel):
    smiles: str


class ExplainIn(BaseModel):
    smiles: str
    payload: dict | None = None


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": ENGINE.ready}


@app.get("/tasks")
async def tasks():
    return {"tasks": get_registry()}


@app.post("/predict")
async def predict(body: PredictIn, authorization: str | None = Header(None)):
    user = _user(authorization)
    _check_rate(user["uid"])
    if not ENGINE.ready:
        raise HTTPException(503, "model not loaded yet; retry shortly")
    result = ENGINE.predict_payload(body.smiles)
    return result


def _resolve_groq_model(api_key: str) -> str:
    preferred = ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"]
    try:
        r = httpx.get("https://api.groq.com/openai/v1/models",
                      headers={"Authorization": f"Bearer {api_key}"}, timeout=15)
        available = [m["id"] for m in r.json().get("data", [])] if r.status_code == 200 else []
        for p in preferred:
            if p in available:
                return p
        if available:
            return available[0]
    except Exception as e:
        log.warning("groq model listing failed: %s", e)
    return "llama-3.1-8b-instant"


@app.post("/explain")
async def explain(body: ExplainIn, authorization: str | None = Header(None)):
    user = _user(authorization)
    _check_rate(user["uid"])
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(503, "GROQ_API_KEY not configured")

    payload = body.payload
    if not payload or not payload.get("valid"):
        if not ENGINE.ready:
            raise HTTPException(503, "model not loaded")
        payload = ENGINE.predict_payload(body.smiles)

    preds = sorted(payload["predictions"], key=lambda p: p["probability"], reverse=True)
    top_hair = next((p for p in preds if p["group"] == "hair"), None)
    top_safety = min((p for p in preds if p["group"] == "safety"),
                     key=lambda p: p["probability"], default=None)
    top_tox = max((p for p in preds if p["group"] == "tox"),
                  key=lambda p: p["probability"], default=None)

    prompt = f"""You are a cosmetic chemistry expert assistant. Interpret this AI safety/efficacy screening for a cosmetic ingredient candidate. Be precise, reference the mechanistic evidence given, note uncertainty. Do not give medical advice.

Molecule SMILES: {payload['canonical_smiles']}
Overall uncertainty: {payload['uncertainty_note']} (mean epistemic std {payload['mean_epistemic_std']})

Hair-health highlights: {[(p['task_id'], round(p['probability'],2)) for p in preds if p['group']=='hair'][:4]}
Toxicity highlights: {[(p['task_id'], round(p['probability'],2)) for p in ([top_tox] if top_tox else [])]}
Safety highlights: {[(p['task_id'], round(p['probability'],2)) for p in ([top_safety] if top_safety else [])]}

Structural alerts found: {[a['message'] for a in payload['alerts']] or 'none'}
Top driving motifs: {[(m['name'], m['severity']) for m in sorted(payload['motifs'], key=lambda x: x['importance'], reverse=True)[:6]]}
Pathway relevance (top): {[(p['name'], p['relevance']) for p in payload['pathways'][:5]]}

Write a concise structured assessment (~180 words) with sections: Summary, Efficacy signals, Safety concerns, Confidence & caveats."""

    model = _resolve_groq_model(api_key)
    try:
        r = httpx.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {api_key}"},
            json={"model": model,
                  "messages": [{"role": "user", "content": prompt}],
                  "temperature": 0.4, "max_tokens": 600},
            timeout=60,
        )
        r.raise_for_status()
        narrative = r.json()["choices"][0]["message"]["content"]
    except httpx.HTTPError as e:
        raise HTTPException(502, f"LLM provider error: {e}")

    return {"narrative": narrative, "groq_model": model}
