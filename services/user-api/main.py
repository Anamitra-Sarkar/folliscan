"""Folliscan User API — lightweight per-user profile & history service (Render).

Firestore layout:
  users/{uid}                -> profile doc (email, displayName, createdAt, prefs)
  users/{uid}/history/{doc}  -> prediction history entries
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from google.cloud import firestore
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("folliscan-user-api")

FRONTEND_ORIGIN = os.environ.get("FRONTEND_ORIGIN", "*").split(",")
MAX_HISTORY_ITEMS = 500

app = FastAPI(title="Folliscan User API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGIN,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["Authorization", "Content-Type"],
)

_db: firestore.Client | None = None


def get_db() -> firestore.Client:
    global _db
    if _db is not None:
        return _db
    b64 = os.environ.get("FIREBASE_CREDENTIALS_B64")
    if not b64:
        raise RuntimeError("FIREBASE_CREDENTIALS_B64 env var is required")
    try:
        cred_dict = json.loads(base64.b64decode(b64))
    except (binascii.Error, json.JSONDecodeError) as e:
        raise RuntimeError(f"invalid FIREBASE_CREDENTIALS_B64 payload: {e}")
    cred_path = "/tmp/firebase_creds.json"
    with open(cred_path, "w") as f:
        json.dump(cred_dict, f)
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = cred_path
    _db = firestore.Client(project=cred_dict.get("project_id"))
    return _db


# ---- auth ----
def require_user(authorization: str | None) -> dict:
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(401, "missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    from firebase_admin import auth as fb_auth, credentials as fb_cred, initialize_app

    if not fb_auth._apps:  # noqa: SLF001 - idempotent init
        b64 = os.environ.get("FIREBASE_CREDENTIALS_B64", "")
        try:
            initialize_app(fb_cred.Certificate(json.loads(base64.b64decode(b64))))
        except Exception as e:
            raise HTTPException(500, f"auth backend misconfigured: {e}")
    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception as e:
        raise HTTPException(401, f"invalid token: {e}")
    expected_project = os.environ.get("FIREBASE_PROJECT_ID", "cabbage-guard")
    if decoded.get("aud") != expected_project:
        raise HTTPException(401, "token audience mismatch")
    return {"uid": decoded["uid"], "email": decoded.get("email", "")}


# ---- models ----
class ProfileUpdate(BaseModel):
    displayName: str | None = None
    prefs: dict | None = None


class HistoryCreate(BaseModel):
    smiles: str = Field(min_length=1, max_length=2048)
    canonical_smiles: str = ""
    result: dict = Field(default_factory=dict)


# ---- routes ----
@app.on_event("startup")
async def startup():
    get_db()
    log.info("firestore client ready")


@app.get("/health")
async def health():
    return {"status": "ok"}


def _user_doc(uid: str):
    return get_db().collection("users").document(uid)


@app.get("/me")
async def me(authorization: str | None = Header(None)):
    """Auto-provisioning profile fetch — never errors for existing Firebase users,
    regardless of which app in the project they signed up through."""
    user = require_user(authorization)
    doc = _user_doc(user["uid"]).get()
    if doc.exists:
        return {"uid": user["uid"], **doc.to_dict()}
    profile = {
        "email": user["email"],
        "displayName": user["email"].split("@")[0],
        "prefs": {},
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    _user_doc(user["uid"]).set(profile, merge=True)
    return {"uid": user["uid"], **profile}


@app.put("/me")
async def update_me(body: ProfileUpdate, authorization: str | None = Header(None)):
    user = require_user(authorization)
    update: dict = {}
    if body.displayName is not None:
        update["displayName"] = body.displayName[:120]
    if body.prefs is not None:
        update["prefs"] = body.prefs
    if update:
        update["updatedAt"] = datetime.now(timezone.utc).isoformat()
        _user_doc(user["uid"]).set(update, merge=True)
    doc = _user_doc(user["uid"]).get()
    return {"uid": user["uid"], **(doc.to_dict() or {})}


@app.get("/history")
async def list_history(limit: int = 20, cursor: str | None = None,
                       authorization: str | None = Header(None)):
    user = require_user(authorization)
    limit = max(1, min(limit, 100))
    q = (_user_doc(user["uid"]).collection("history")
         .order_by("createdAt", direction=firestore.Query.DESCENDING)
         .limit(limit))
    if cursor:
        snap = _user_doc(user["uid"]).collection("history").document(cursor).get()
        if snap.exists:
            q = q.start_after({u"createdAt": snap.to_dict().get("createdAt")})
    docs = list(q.stream())
    items = [{"id": d.id, **d.to_dict()} for d in docs]
    next_cursor = docs[-1].id if len(docs) == limit else None
    return {"items": items, "next_cursor": next_cursor}


@app.post("/history")
async def create_history(body: HistoryCreate, authorization: str | None = Header(None)):
    user = require_user(authorization)
    count_q = list(_user_doc(user["uid"]).collection("history")
                   .order_by("createdAt").limit(MAX_HISTORY_ITEMS + 1).stream())
    if len(count_q) > MAX_HISTORY_ITEMS:
        oldest = count_q[0]
        oldest.reference.delete()

    entry = {
        "smiles": body.smiles[:2048],
        "canonical_smiles": body.canonical_smiles[:2048] or body.smiles[:2048],
        "result": body.result,
        "createdAt": datetime.now(timezone.utc).isoformat(),
    }
    _, ref = _user_doc(user["uid"]).collection("history").add(entry)
    return {"id": ref.id, **entry}


@app.delete("/history/{item_id}")
async def delete_history(item_id: str, authorization: str | None = Header(None)):
    user = require_user(authorization)
    ref = _user_doc(user["uid"]).collection("history").document(item_id)
    if not ref.get().exists:
        raise HTTPException(404, "not found")
    ref.delete()
    return {"deleted": item_id}
