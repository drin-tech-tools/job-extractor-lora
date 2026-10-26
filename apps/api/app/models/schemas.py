"""The extraction task, defined once.

`JobExtraction` is the single source of truth: the Gemini labeler asks for exactly this
shape, the training notebook formats it into the assistant target, evals/ scores it field
by field, and the API validates every model's output against it. Change a field here and
every stage has to be re-run.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

MAX_SKILLS = 15
# Postings are cut to this length when building the dataset, so the API accepts no more.
MAX_POSTING_CHARS = 6000
MIN_POSTING_CHARS = 40


class Seniority(StrEnum):
    INTERN = "intern"
    JUNIOR = "junior"
    MID = "mid"
    SENIOR = "senior"
    LEAD = "lead"  # lead / staff / principal / engineering manager
    EXECUTIVE = "executive"  # director, VP, C-level
    UNSPECIFIED = "unspecified"


class WorkMode(StrEnum):
    REMOTE = "remote"
    HYBRID = "hybrid"
    ONSITE = "onsite"
    UNSPECIFIED = "unspecified"


class SalaryPeriod(StrEnum):
    YEAR = "year"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"


class JobExtraction(BaseModel):
    title: str = Field(description="Job title as written, without company or location")
    seniority: Seniority
    required_skills: list[str] = Field(
        default_factory=list,
        description="Explicitly REQUIRED technical skills/tools/domains, short names as written",
    )
    years_experience_min: int | None = Field(
        default=None, description="Minimum years of experience required; null if not stated"
    )
    years_experience_max: int | None = Field(
        default=None, description="Maximum years, only when a range is stated; else null"
    )
    work_mode: WorkMode
    salary_min: float | None = Field(
        default=None, description="Lower salary bound exactly as stated (no currency conversion)"
    )
    salary_max: float | None = Field(
        default=None, description="Upper salary bound; equal to salary_min for a single figure"
    )
    salary_currency: str | None = Field(
        default=None, description="ISO 4217 code, e.g. USD, EUR, PHP; null if no salary stated"
    )
    salary_period: SalaryPeriod | None = Field(
        default=None, description="Pay period of the stated salary; null if no salary stated"
    )

    @model_validator(mode="after")
    def _tidy(self) -> JobExtraction:
        """Normalise what is cheap to normalise so outputs are comparable across models."""
        self.title = " ".join(self.title.split())
        seen: set[str] = set()
        skills: list[str] = []
        for skill in self.required_skills:
            skill = " ".join(skill.split())
            if skill and skill.lower() not in seen:
                seen.add(skill.lower())
                skills.append(skill)
        self.required_skills = skills[:MAX_SKILLS]
        if self.salary_currency:
            self.salary_currency = self.salary_currency.strip().upper()
        return self


# ---- API request / response models -------------------------------------------------------

ModelKey = Literal["finetuned", "base", "gemini"]
ALL_MODELS: tuple[ModelKey, ...] = ("finetuned", "base", "gemini")


class ExtractRequest(BaseModel):
    text: str = Field(..., min_length=MIN_POSTING_CHARS, max_length=MAX_POSTING_CHARS)
    model: ModelKey = "finetuned"


class CompareRequest(BaseModel):
    text: str = Field(..., min_length=MIN_POSTING_CHARS, max_length=MAX_POSTING_CHARS)
    models: list[ModelKey] = Field(default_factory=lambda: list(ALL_MODELS), min_length=1)


class ExtractResponse(BaseModel):
    model: ModelKey
    model_id: str = Field(description="The concrete model that answered, e.g. an HF repo id")
    extraction: JobExtraction | None = None
    valid_json: bool = Field(description="Output parsed AND matched the schema")
    latency_ms: int
    error: str | None = None
    raw: str | None = Field(default=None, description="Raw model text (truncated), for debugging")


class CompareResponse(BaseModel):
    results: list[ExtractResponse]
