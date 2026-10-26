"""Extraction backends: fine-tuned model, base model, Gemini — plus a mock for demos/CI.

All backends return an `ExtractResult` and never raise for model-side problems (bad JSON,
timeouts, HTTP errors): those are recorded on the result so the compare view can show them.
`ExtractorUnavailable` is raised only for missing configuration.
"""

from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Protocol

import httpx

from app.config import Settings
from app.models.schemas import JobExtraction, ModelKey
from app.services.prompts import (
    EXTRACTION_RULES,
    OutputError,
    build_messages,
    parse_model_output,
    target_json,
)

logger = logging.getLogger(__name__)

RAW_PREVIEW_CHARS = 2000


class ExtractorUnavailable(RuntimeError):
    """The backend for this model key isn't configured on this deployment."""


@dataclass
class ExtractResult:
    model_id: str
    latency_ms: int
    extraction: JobExtraction | None = None
    raw: str | None = None
    error: str | None = None

    @property
    def valid_json(self) -> bool:
        return self.extraction is not None


class Extractor(Protocol):
    async def extract(self, text: str) -> ExtractResult: ...


def _ms(started: float) -> int:
    return int((time.perf_counter() - started) * 1000)


def _finish(model_id: str, started: float, raw: str) -> ExtractResult:
    """Parse raw model text into a result; parse failures are results, not exceptions."""
    preview = raw[:RAW_PREVIEW_CHARS]
    try:
        return ExtractResult(model_id, _ms(started), parse_model_output(raw), preview)
    except OutputError as exc:
        return ExtractResult(model_id, _ms(started), None, preview, str(exc))


# ---------------------------------------------------------------- OpenAI-compatible (HF)


class ChatCompletionsExtractor:
    """Any OpenAI-compatible /chat/completions endpoint (HF Inference Endpoint, vLLM, ...)."""

    def __init__(
        self,
        *,
        api_url: str,
        model: str,
        token: str,
        style: str,
        timeout_s: float,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.api_url = api_url.rstrip("/")
        self.model = model
        self.token = token
        self.style = style
        self.timeout_s = timeout_s
        self._client = client

    async def extract(self, text: str) -> ExtractResult:
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}
        payload = {
            "model": self.model,
            "messages": build_messages(text, self.style),  # type: ignore[arg-type]
            "temperature": 0,
            "max_tokens": 600,
        }
        owns_client = self._client is None
        client = self._client or httpx.AsyncClient(timeout=self.timeout_s)
        try:
            resp = await client.post(
                f"{self.api_url}/chat/completions", json=payload, headers=headers
            )
            resp.raise_for_status()
            raw = resp.json()["choices"][0]["message"]["content"] or ""
        except httpx.TimeoutException:
            return ExtractResult(self.model, _ms(started), error="timeout (endpoint may be cold)")
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            return ExtractResult(self.model, _ms(started), error=f"upstream HTTP {code}")
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            logger.warning("Chat endpoint failed for %s: %s", self.model, exc)
            return ExtractResult(self.model, _ms(started), error=f"upstream error: {exc}")
        finally:
            if owns_client:
                await client.aclose()
        return _finish(self.model, started, raw)


# ---------------------------------------------------------------- Gemini


class GeminiExtractor:
    """Gemini with a response schema — the upper reference (and the labeler's model family)."""

    def __init__(self, *, api_key: str, models: list[str]) -> None:
        from google import genai  # lazy: not needed when only the HF backends are used

        self._client = genai.Client(api_key=api_key)
        self.models = models

    async def extract(self, text: str) -> ExtractResult:
        from google.genai import types

        started = time.perf_counter()
        config = types.GenerateContentConfig(
            system_instruction=EXTRACTION_RULES,
            response_mime_type="application/json",
            response_schema=JobExtraction,
            temperature=0.0,
        )
        last_error = "no Gemini model available"
        for model in self.models:
            try:
                resp = await self._client.aio.models.generate_content(
                    model=model, contents=f"JOB POSTING:\n\n{text}", config=config
                )
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None)
                last_error = f"{model}: HTTP {code}" if code else f"{model}: {exc}"
                if code in (400, 403, 404):  # model not available to this key -> try the next
                    continue
                return ExtractResult(model, _ms(started), error=last_error)
            return _finish(model, started, resp.text or "")
        return ExtractResult(self.models[-1], _ms(started), error=last_error)


# ---------------------------------------------------------------- mock (heuristics)

_SKILL_VOCAB = [
    "Python", "SQL", "AWS", "GCP", "Azure", "Docker", "Kubernetes", "TypeScript", "JavaScript",
    "React", "Node.js", "Go", "Java", "Rust", "PyTorch", "TensorFlow", "FastAPI", "Django",
    "PostgreSQL", "Terraform", "Spark", "Airflow", "Git", "Figma",
]  # fmt: skip
_CURRENCY = {"$": "USD", "€": "EUR", "£": "GBP"}
_PERIODS = [
    (r"per hour|/\s*h(?:ou)?r|hourly", "hour"),
    (r"per month|/\s*mo(?:nth)?|monthly", "month"),
    (r"per week|/\s*w(?:ee)?k|weekly", "week"),
    (r"per day|/\s*day|daily", "day"),
    (r"per year|/\s*y(?:ea)?r|annual|yearly|a year", "year"),
]
_SALARY_RE = re.compile(
    r"([$€£])\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?\s*(?:-|–|—|to)\s*[$€£]?\s?(\d[\d,]*(?:\.\d+)?)\s*(k)?",
    re.I,
)
_YEARS_RE = re.compile(r"(\d+)\s*(?:\+|-\s*(\d+)|to\s*(\d+))?\s*(?:years|yrs)", re.I)


def _amount(num: str, k: str | None) -> float:
    value = float(num.replace(",", ""))
    return value * 1000 if k else value


def heuristic_extract(text: str) -> JobExtraction:
    """Regex stand-in for a model. Deliberately simple: it exists so the demo and CI run
    end to end with no keys — it is NOT one of the models being evaluated."""
    lowered = text.lower()
    first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    title = re.split(r"\s+[-–|@]\s+|\s+at\s+", first_line)[0][:100]

    seniority = "unspecified"
    for pattern, level in [
        (r"\bintern(ship)?\b", "intern"),
        (r"\b(director|vp|vice president|chief|head of)\b", "executive"),
        (r"\b(lead|staff|principal|manager)\b", "lead"),
        (r"\b(senior|sr\.?)\b", "senior"),
        (r"\b(junior|jr\.?|entry[- ]level)\b", "junior"),
    ]:
        if re.search(pattern, title.lower()):
            seniority = level
            break

    if "hybrid" in lowered:
        work_mode = "hybrid"
    elif re.search(r"\bremote\b|work from anywhere", lowered):
        work_mode = "remote"
    elif re.search(r"on-?site|in[- ]office", lowered):
        work_mode = "onsite"
    else:
        work_mode = "unspecified"

    years_min = years_max = None
    if m := _YEARS_RE.search(text):
        years_min = int(m.group(1))
        high = m.group(2) or m.group(3)
        years_max = int(high) if high else None

    salary: dict = {}
    if m := _SALARY_RE.search(text):
        window = lowered[m.end() : m.end() + 40]
        period = next((p for pat, p in _PERIODS if re.search(pat, window)), None)
        salary = {
            "salary_min": _amount(m.group(2), m.group(3) or m.group(5)),
            "salary_max": _amount(m.group(4), m.group(5)),
            "salary_currency": _CURRENCY[m.group(1)],
            "salary_period": period or "year",
        }

    skills = [s for s in _SKILL_VOCAB if re.search(rf"(?<!\w){re.escape(s)}(?!\w)", text, re.I)]
    return JobExtraction.model_validate(
        {
            "title": title or "Unknown",
            "seniority": seniority,
            "required_skills": skills,
            "years_experience_min": years_min,
            "years_experience_max": years_max,
            "work_mode": work_mode,
            **salary,
        }
    )


class MockExtractor:
    model_id = "mock-heuristic"

    async def extract(self, text: str) -> ExtractResult:
        started = time.perf_counter()
        extraction = heuristic_extract(text)
        return ExtractResult(self.model_id, _ms(started), extraction, target_json(extraction))


# ---------------------------------------------------------------- factory


def get_extractor(model: ModelKey, settings: Settings) -> Extractor:
    """Resolve a model key to a backend, or raise ExtractorUnavailable."""
    if settings.extractor_mock:
        return MockExtractor()
    if model == "gemini":
        if not settings.gemini_configured:
            raise ExtractorUnavailable("Gemini is not configured (GEMINI_API_KEY)")
        return GeminiExtractor(api_key=settings.gemini_api_key, models=settings.gemini_models)
    if model == "finetuned":
        if not settings.finetuned_configured:
            raise ExtractorUnavailable("Fine-tuned model is not configured (FINETUNED_API_URL)")
        return ChatCompletionsExtractor(
            api_url=settings.finetuned_api_url,
            model=settings.finetuned_model,
            token=settings.hf_token,
            style="tuned",
            timeout_s=settings.request_timeout_s,
        )
    if not settings.base_configured:
        raise ExtractorUnavailable("Base model is not configured (BASE_API_URL)")
    return ChatCompletionsExtractor(
        api_url=settings.base_api_url,
        model=settings.base_model,
        token=settings.hf_token,
        style="base",
        timeout_s=settings.request_timeout_s,
    )
