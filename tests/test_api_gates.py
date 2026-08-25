"""Auth-gate tests: unauthenticated requests must be rejected without
requiring real Firebase credentials."""

import importlib
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "services" / "user-api"))

from backend.app.auth import require_user, AuthError  # noqa: E402


def test_missing_token_rejected():
    with pytest.raises(AuthError):
        require_user(None)
    with pytest.raises(AuthError):
        require_user("Token abc")       # not Bearer scheme


def test_user_api_health_and_auth_gate():
    user_api = importlib.import_module("main")
    client = TestClient(user_api.app)

    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok"

    # auth-gated routes reject anonymous access
    assert client.get("/me").status_code == 401
    assert client.get("/history").status_code == 401
    assert client.post(
        "/history", json={"smiles": "CCO"}
    ).status_code == 401
    assert client.delete("/history/xyz").status_code == 401
