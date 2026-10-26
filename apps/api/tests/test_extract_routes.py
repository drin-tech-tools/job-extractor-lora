from __future__ import annotations

from fastapi.testclient import TestClient

from app.config import get_settings
from tests.conftest import SAMPLE_POSTING


def test_health_reports_mock_backends(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["mock"] is True
    assert body["backends"] == {"finetuned": True, "base": True, "gemini": True}


def test_extract_single(client: TestClient) -> None:
    res = client.post("/extract", json={"text": SAMPLE_POSTING, "model": "finetuned"})
    assert res.status_code == 200
    body = res.json()
    assert body["valid_json"] is True
    assert body["extraction"]["seniority"] == "senior"
    assert body["model"] == "finetuned" and body["model_id"] == "mock-heuristic"


def test_extract_defaults_to_finetuned(client: TestClient) -> None:
    assert client.post("/extract", json={"text": SAMPLE_POSTING}).json()["model"] == "finetuned"


def test_extract_validates_input(client: TestClient) -> None:
    assert client.post("/extract", json={"text": "too short"}).status_code == 422
    assert client.post("/extract", json={"text": "x" * 6001}).status_code == 422
    assert client.post("/extract", json={"text": SAMPLE_POSTING, "model": "gpt"}).status_code == 422


def test_compare_runs_all_models_in_order_and_dedupes(client: TestClient) -> None:
    res = client.post("/extract/compare", json={"text": SAMPLE_POSTING})
    assert [r["model"] for r in res.json()["results"]] == ["finetuned", "base", "gemini"]
    res = client.post(
        "/extract/compare", json={"text": SAMPLE_POSTING, "models": ["base", "base", "gemini"]}
    )
    assert [r["model"] for r in res.json()["results"]] == ["base", "gemini"]


def test_unconfigured_model_is_503_single_but_error_row_in_compare(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setenv("EXTRACTOR_MOCK", "0")
    get_settings.cache_clear()
    from app.main import app

    client = TestClient(app)
    single = client.post("/extract", json={"text": SAMPLE_POSTING, "model": "gemini"})
    assert single.status_code == 503

    cmp = client.post("/extract/compare", json={"text": SAMPLE_POSTING}).json()["results"]
    assert len(cmp) == 3
    assert all(r["valid_json"] is False and "not configured" in r["error"] for r in cmp)
