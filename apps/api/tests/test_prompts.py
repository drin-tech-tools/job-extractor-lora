"""Prompt building and the output-parsing contract that evals rely on."""

from __future__ import annotations

import pytest

from app.models.schemas import JobExtraction
from app.services.prompts import (
    OutputError,
    build_messages,
    parse_model_output,
    schema_hint,
    target_json,
)
from tests.conftest import GOOD_JSON


def test_schema_hint_mentions_every_field() -> None:
    hint = schema_hint()
    for name in JobExtraction.model_fields:
        assert f'"{name}"' in hint, f"{name} missing from schema_hint — it drifted from the schema"


def test_tuned_style_is_minimal_and_base_carries_rules() -> None:
    tuned = build_messages("posting text", "tuned")
    base = build_messages("posting text", "base")
    assert [m["role"] for m in tuned] == ["system", "user"]
    assert tuned[1]["content"] == "posting text"
    assert len(tuned[0]["content"]) < 100
    assert "required_skills" in base[0]["content"]
    assert "posting text" in base[1]["content"]


def test_parse_valid_and_roundtrip_through_target_json() -> None:
    extraction = parse_model_output(GOOD_JSON)
    assert extraction.seniority == "senior"
    assert parse_model_output(target_json(extraction)) == extraction


def test_target_json_is_compact_and_in_schema_order() -> None:
    text = target_json(parse_model_output(GOOD_JSON))
    assert " " not in text.split('"title":')[0]
    keys = [k for k in JobExtraction.model_fields if f'"{k}"' in text]
    assert keys == list(JobExtraction.model_fields)
    assert text.index('"title"') < text.index('"seniority"') < text.index('"salary_period"')


def test_parse_tolerates_code_fence_and_prose() -> None:
    assert parse_model_output(f"```json\n{GOOD_JSON}\n```").title == "Senior Backend Engineer"
    chatty = f"Sure! Here you go: {GOOD_JSON} Hope that helps"
    assert parse_model_output(chatty).work_mode == "remote"


@pytest.mark.parametrize(
    ("raw", "kind"),
    [
        ("I cannot do that", "invalid_json"),
        ('{"title": "x", "seniority": "senior"', "invalid_json"),  # truncated
        ("[1, 2, 3]", "invalid_json"),
        ('{"title":"x","seniority":"wizard","work_mode":"remote"}', "schema_mismatch"),
        ('{"seniority":"senior","work_mode":"remote"}', "schema_mismatch"),  # no title
    ],
)
def test_parse_rejects_with_distinct_kinds(raw: str, kind: str) -> None:
    with pytest.raises(OutputError) as exc:
        parse_model_output(raw)
    assert exc.value.kind == kind


def test_schema_normalises_skills_and_currency() -> None:
    e = JobExtraction.model_validate(
        {
            "title": "  Data   Engineer ",
            "seniority": "mid",
            "work_mode": "hybrid",
            "required_skills": ["Python", "python", " SQL ", ""],
            "salary_currency": " usd ",
        }
    )
    assert e.title == "Data Engineer"
    assert e.required_skills == ["Python", "SQL"]
    assert e.salary_currency == "USD"
