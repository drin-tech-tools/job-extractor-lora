"""Prompts and output parsing shared by training, evals and serving.

Keeping these in one module is what prevents train/inference mismatch: the notebook builds
its chat examples with `build_messages(..., style="tuned")`, the API calls the fine-tuned
model with the same call, and evals score every model through `parse_model_output`.
"""

from __future__ import annotations

import json
import re
from typing import Literal

from pydantic import ValidationError

from app.models.schemas import (
    MAX_SKILLS,
    JobExtraction,
    SalaryPeriod,
    Seniority,
    WorkMode,
)

PromptStyle = Literal["base", "tuned"]

# Used verbatim by the Gemini labeler (distillation) and the Gemini backend.
EXTRACTION_RULES = f"""\
You extract structured data from a single job posting. Use ONLY what the posting states.
When something is not stated, use null (or "unspecified" for seniority/work_mode). Never guess.

Field rules:
- title: the job title as written, without company name, location or marketing words.
- seniority: intern | junior | mid | senior | lead | executive | unspecified.
  Judge from the title first, then the requirements. lead covers lead/staff/principal and
  engineering managers; executive covers director, VP and C-level.
- required_skills: technical skills, tools, languages, frameworks or domains the posting
  REQUIRES. Exclude nice-to-haves ("bonus", "preferred", "a plus") and generic soft skills
  ("communication", "team player"). Use short names as written ("Python", "AWS", "SQL").
  At most {MAX_SKILLS}, no duplicates.
- years_experience_min: overall years of experience required ("5+ years" -> 5). If years are
  stated only per skill, use the largest. years_experience_max only for an explicit range
  ("3-5 years" -> min 3, max 5). Otherwise null.
- work_mode: remote (fully remote, work from anywhere) | hybrid | onsite | unspecified.
  "Remote within the US" is still remote.
- salary_min / salary_max: figures exactly as stated, no currency conversion; expand
  shorthand ("$120k" -> 120000). A single figure sets both. null when no pay is stated.
  Ignore equity, bonuses and benefits.
- salary_currency: ISO 4217 code (USD, EUR, GBP, PHP ...). salary_period: year | month |
  week | day | hour. Both null when no salary is stated.
"""

# The fine-tuned model has seen the rules thousands of times; it only needs the task name.
TUNED_INSTRUCTION = "Extract the job posting into JSON."


def schema_hint() -> str:
    """A JSON template for models that haven't been fine-tuned. Built from the enums so it
    cannot drift from the schema."""

    def alt(enum_cls: type) -> str:
        return " | ".join(f'"{m.value}"' for m in enum_cls)

    return "\n".join(
        [
            "{",
            '  "title": string,',
            f"  \"seniority\": {alt(Seniority)},",
            '  "required_skills": [string],',
            '  "years_experience_min": integer | null,',
            '  "years_experience_max": integer | null,',
            f"  \"work_mode\": {alt(WorkMode)},",
            '  "salary_min": number | null,',
            '  "salary_max": number | null,',
            '  "salary_currency": string | null,',
            f"  \"salary_period\": {alt(SalaryPeriod)} | null",
            "}",
        ]
    )


def build_messages(text: str, style: PromptStyle) -> list[dict[str, str]]:
    """Chat messages for one posting. `base` carries the full rules; `tuned` is minimal."""
    if style == "tuned":
        return [
            {"role": "system", "content": TUNED_INSTRUCTION},
            {"role": "user", "content": text},
        ]
    system = (
        f"{EXTRACTION_RULES}\n"
        f"Respond with ONLY one JSON object, no prose and no code fences, shaped like:\n"
        f"{schema_hint()}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"JOB POSTING:\n\n{text}"},
    ]


def target_json(extraction: JobExtraction | dict) -> str:
    """The exact assistant string the model is trained to produce: compact, schema key order."""
    if isinstance(extraction, JobExtraction):
        extraction = extraction.model_dump(mode="json")
    data = extraction
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"))


class OutputError(ValueError):
    """A model reply that can't be used. `kind` separates the two failure modes for evals."""

    def __init__(self, kind: Literal["invalid_json", "schema_mismatch"], detail: str) -> None:
        super().__init__(f"{kind}: {detail}")
        self.kind = kind
        self.detail = detail


_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)


def parse_model_output(raw: str) -> JobExtraction:
    """Turn raw model text into a validated extraction.

    Tolerates a code fence or prose around ONE JSON object — small models do this — but
    nothing else: truncated JSON, wrong enums or wrong types all raise OutputError.
    """
    text = _FENCE_RE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise OutputError("invalid_json", "no JSON object found")
    try:
        data = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise OutputError("invalid_json", str(exc)) from exc
    if not isinstance(data, dict):
        raise OutputError("invalid_json", "top-level value is not an object")
    try:
        return JobExtraction.model_validate(data)
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise OutputError("schema_mismatch", f"{loc}: {first['msg']}") from exc
