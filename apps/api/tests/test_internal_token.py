"""INTERNAL_API_TOKEN middleware tests."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings


def test_health_open_when_token_unset(client: TestClient) -> None:
    assert client.get("/health").status_code == 200


def test_rejects_missing_token_when_configured(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-test-token")
    get_settings.cache_clear()
    from app.main import app

    res = TestClient(app).get("/health")
    assert res.status_code == 401
    assert res.json()["detail"] == "Unauthorized"


def test_accepts_matching_token(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("INTERNAL_API_TOKEN", "secret-test-token")
    get_settings.cache_clear()
    from app.main import app

    res = TestClient(app).get("/health", headers={"X-Internal-Api-Token": "secret-test-token"})
    assert res.status_code == 200
