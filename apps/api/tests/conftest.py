"""Shared pytest fixtures."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.config import get_settings


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch: pytest.MonkeyPatch):
    for var in (
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "FINETUNED_API_URL",
        "FINETUNED_MODEL",
        "BASE_API_URL",
        "BASE_MODEL",
        "HF_TOKEN",
        "INTERNAL_API_TOKEN",
    ):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("ALLOWED_ORIGINS", "http://localhost:3000")
    # Default test seam: heuristic extractor, no network.
    monkeypatch.setenv("EXTRACTOR_MOCK", "1")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def client() -> TestClient:
    from app.main import app

    return TestClient(app)


SAMPLE_POSTING = """\
Senior Backend Engineer - Acme Cloud

We are hiring a Senior Backend Engineer to join our fully remote team.

Requirements:
- 5+ years of professional experience
- Strong Python and SQL; experience with AWS and Docker
- Nice to have: Kubernetes

Compensation: $120,000 - $150,000 per year, plus equity.
"""

GOOD_JSON = (
    '{"title":"Senior Backend Engineer","seniority":"senior","required_skills":["Python","SQL"],'
    '"years_experience_min":5,"years_experience_max":null,"work_mode":"remote",'
    '"salary_min":120000,"salary_max":150000,"salary_currency":"USD","salary_period":"year"}'
)
