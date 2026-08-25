"""Firebase ID-token authentication (shared project `cabbage-guard`)."""

from __future__ import annotations

import base64
import binascii
import json
import logging
import os

import firebase_admin
from firebase_admin import auth as fb_auth, credentials

log = logging.getLogger(__name__)
_app = None


def init_firebase():
    global _app
    if _app is not None:
        return _app
    b64 = os.environ.get("FIREBASE_CREDENTIALS_B64")
    if not b64:
        raise RuntimeError("FIREBASE_CREDENTIALS_B64 env var is required")
    try:
        raw = base64.b64decode(b64)
        cred_dict = json.loads(raw)
    except (binascii.Error, json.JSONDecodeError) as e:
        raise RuntimeError(f"invalid FIREBASE_CREDENTIALS_B64 payload: {e}")
    cred = credentials.Certificate(cred_dict)
    _app = firebase_admin.initialize_app(cred)
    log.info("firebase admin initialized for project %s",
             os.environ.get("FIREBASE_PROJECT_ID", "cabbage-guard"))
    return _app


class AuthError(Exception):
    def __init__(self, msg: str):
        super().__init__(msg)


def require_user(authorization: str | None) -> dict:
    """Verify Bearer Firebase ID token -> {uid, email}. Raises AuthError."""
    if not authorization or not authorization.startswith("Bearer "):
        raise AuthError("missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    try:
        decoded = fb_auth.verify_id_token(token)
    except Exception as e:
        raise AuthError(f"invalid token: {e}")
    expected_project = os.environ.get("FIREBASE_PROJECT_ID", "cabbage-guard")
    if decoded.get("aud") != expected_project:
        raise AuthError("token audience mismatch")
    return {"uid": decoded["uid"], "email": decoded.get("email", "")}
