"""Make `app.*` (apps/api) importable, and keep tests off real credentials."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

API = Path(__file__).resolve().parents[2] / "apps" / "api"
if str(API) not in sys.path:
    sys.path.insert(0, str(API))


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch):
    from app.config import get_settings

    for var in (
        "GEMINI_API_KEY", "GEMINI_MODEL", "FINETUNED_API_URL", "FINETUNED_MODEL",
        "BASE_API_URL", "BASE_MODEL", "HF_TOKEN", "EXTRACTOR_MOCK",
    ):  # fmt: skip
        monkeypatch.delenv(var, raising=False)
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
