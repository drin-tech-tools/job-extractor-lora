"""HTTP backend + Gemini fallback logic, with all network mocked."""

from __future__ import annotations

import httpx
import pytest
from pytest_httpx import HTTPXMock

from app.config import Settings, get_settings
from app.services.extractors import (
    ChatCompletionsExtractor,
    ExtractorUnavailable,
    GeminiExtractor,
    MockExtractor,
    get_extractor,
)
from tests.conftest import GOOD_JSON, SAMPLE_POSTING

URL = "https://endpoint.example/v1"


def _hf(style: str = "tuned") -> ChatCompletionsExtractor:
    return ChatCompletionsExtractor(
        api_url=URL, model="org/model", token="hf_x", style=style, timeout_s=5
    )


async def test_chat_extractor_happy_path(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{URL}/chat/completions",
        json={"choices": [{"message": {"content": GOOD_JSON}}]},
    )
    result = await _hf().extract(SAMPLE_POSTING)
    assert result.valid_json and result.extraction.title == "Senior Backend Engineer"
    request = httpx_mock.get_request()
    assert request.headers["authorization"] == "Bearer hf_x"
    body = request.read().decode()
    assert '"temperature":0' in body.replace(" ", "")
    assert "Extract the job posting into JSON." in body  # tuned style prompt


async def test_chat_extractor_reports_invalid_json_not_exception(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(
        url=f"{URL}/chat/completions",
        json={"choices": [{"message": {"content": "Sorry, I can't help with that."}}]},
    )
    result = await _hf("base").extract(SAMPLE_POSTING)
    assert not result.valid_json
    assert result.error.startswith("invalid_json")
    assert result.raw == "Sorry, I can't help with that."


async def test_chat_extractor_http_error_and_timeout(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{URL}/chat/completions", status_code=503)
    assert (await _hf().extract(SAMPLE_POSTING)).error == "upstream HTTP 503"

    httpx_mock.add_exception(httpx.ReadTimeout("slow"), url=f"{URL}/chat/completions")
    assert "timeout" in (await _hf().extract(SAMPLE_POSTING)).error


async def test_chat_extractor_malformed_envelope(httpx_mock: HTTPXMock) -> None:
    httpx_mock.add_response(url=f"{URL}/chat/completions", json={"unexpected": True})
    result = await _hf().extract(SAMPLE_POSTING)
    assert not result.valid_json and result.error.startswith("upstream error")


def test_factory_unavailable_when_unconfigured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EXTRACTOR_MOCK", "0")
    get_settings.cache_clear()
    settings = get_settings()
    for key in ("finetuned", "base", "gemini"):
        with pytest.raises(ExtractorUnavailable):
            get_extractor(key, settings)  # type: ignore[arg-type]


def test_factory_picks_prompt_style_per_model() -> None:
    settings = Settings(
        extractor_mock=False,
        finetuned_api_url=URL, finetuned_model="ft",
        base_api_url=URL, base_model="b",
    )  # fmt: skip
    assert get_extractor("finetuned", settings).style == "tuned"  # type: ignore[attr-defined]
    assert get_extractor("base", settings).style == "base"  # type: ignore[attr-defined]


def test_factory_mock_overrides_everything() -> None:
    assert isinstance(get_extractor("gemini", Settings(extractor_mock=True)), MockExtractor)


def test_gemini_model_order_pins_first() -> None:
    models = Settings(gemini_model="gemini-flash-latest").gemini_models
    assert models[0] == "gemini-flash-latest" and models.count("gemini-flash-latest") == 1


class _FakeModels:
    def __init__(self, behaviours: dict[str, object]) -> None:
        self.behaviours = behaviours
        self.calls: list[str] = []

    async def generate_content(self, *, model: str, contents: str, config: object):  # noqa: ARG002
        self.calls.append(model)
        outcome = self.behaviours[model]
        if isinstance(outcome, Exception):
            raise outcome
        return type("Resp", (), {"text": outcome})()


class _GeminiError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


def _gemini(behaviours: dict[str, object]) -> tuple[GeminiExtractor, _FakeModels]:
    ex = GeminiExtractor(api_key="k", models=list(behaviours))
    fake = _FakeModels(behaviours)
    ex._client = type("C", (), {"aio": type("A", (), {"models": fake})()})()  # type: ignore[assignment]
    return ex, fake


async def test_gemini_falls_through_unavailable_models() -> None:
    ex, fake = _gemini({"m1": _GeminiError(404), "m2": GOOD_JSON})
    result = await ex.extract(SAMPLE_POSTING)
    assert result.valid_json and result.model_id == "m2"
    assert fake.calls == ["m1", "m2"]


async def test_gemini_rate_limit_stops_and_reports() -> None:
    ex, fake = _gemini({"m1": _GeminiError(429), "m2": GOOD_JSON})
    result = await ex.extract(SAMPLE_POSTING)
    assert not result.valid_json and "429" in result.error
    assert fake.calls == ["m1"]  # a 429 is not a reason to burn the next model's quota
