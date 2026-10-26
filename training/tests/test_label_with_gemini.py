from __future__ import annotations

import json
from pathlib import Path

import label_with_gemini as lg
import pytest
from app.models.schemas import JobExtraction

POSTING = """\
Senior Backend Engineer at Acme

Fully remote. You have 5+ years of experience with Python and SQL.
Compensation: $120,000 - $150,000 per year.
"""

GOOD = {
    "title": "Senior Backend Engineer",
    "seniority": "senior",
    "required_skills": ["Python", "SQL"],
    "years_experience_min": 5,
    "years_experience_max": None,
    "work_mode": "remote",
    "salary_min": 120000,
    "salary_max": 150000,
    "salary_currency": "USD",
    "salary_period": "year",
}


def _label(**overrides: object) -> JobExtraction:
    return JobExtraction.model_validate({**GOOD, **overrides})


class _HttpError(Exception):
    def __init__(self, code: int) -> None:
        super().__init__(f"HTTP {code}")
        self.code = code


# ---------------------------------------------------------------- grounding flags


def test_grounded_label_has_no_flags() -> None:
    assert lg.grounding_flags(_label(), POSTING) == []


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"title": "Staff Wizard"}, "title_not_in_text"),
        ({"required_skills": ["Python", "Haskell"]}, "skills_not_in_text:Haskell"),
        ({"salary_min": 90000}, "salary_min_not_in_text"),
        ({"salary_max": 999999}, "salary_max_not_in_text"),
        ({"years_experience_min": 9}, "years_not_in_text"),
        ({"salary_max": None}, "salary_half_filled"),
        ({"salary_currency": None}, "salary_missing_currency_or_period"),
        ({"salary_min": 150000, "salary_max": 120000}, "salary_min_gt_max"),
        ({"years_experience_max": 3}, "years_min_gt_max"),
    ],
)
def test_flags_catch_ungrounded_fields(overrides: dict, expected: str) -> None:
    assert expected in lg.grounding_flags(_label(**overrides), POSTING)


def test_salary_written_as_k_is_recognised() -> None:
    text = "Remote role, pay $120k-$150k a year. 5 years of Python needed. Senior Backend Engineer"
    flags = lg.grounding_flags(_label(required_skills=["Python"]), text)
    assert not [f for f in flags if f.startswith("salary")]


# ---------------------------------------------------------------- label_text


def test_label_text_first_model_succeeds() -> None:
    calls: list[str] = []

    def generate(model: str, text: str) -> str:
        calls.append(model)
        return json.dumps(GOOD)

    label, model = lg.label_text(POSTING, generate, ["m1", "m2"], sleep=lambda _: None)
    assert (label.title, model, calls) == ("Senior Backend Engineer", "m1", ["m1"])


def test_unavailable_model_falls_through_to_next() -> None:
    def generate(model: str, text: str) -> str:
        if model == "m1":
            raise _HttpError(404)
        return json.dumps(GOOD)

    _, model = lg.label_text(POSTING, generate, ["m1", "m2"], sleep=lambda _: None)
    assert model == "m2"


def test_rate_limit_backs_off_then_succeeds() -> None:
    attempts = {"n": 0}
    waits: list[float] = []

    def generate(model: str, text: str) -> str:
        attempts["n"] += 1
        if attempts["n"] < 3:
            raise _HttpError(429)
        return json.dumps(GOOD)

    _, model = lg.label_text(POSTING, generate, ["m1"], sleep=waits.append)
    assert model == "m1" and attempts["n"] == 3
    assert waits == [2, 4]  # exponential


def test_schema_violation_gets_one_retry_then_next_model() -> None:
    calls: list[str] = []

    def generate(model: str, text: str) -> str:
        calls.append(model)
        return '{"title": "x", "seniority": "wizard"}' if model == "m1" else json.dumps(GOOD)

    _, model = lg.label_text(POSTING, generate, ["m1", "m2"], sleep=lambda _: None)
    assert model == "m2" and calls == ["m1", "m1", "m2"]


def test_all_models_failing_raises() -> None:
    def generate(model: str, text: str) -> str:
        raise _HttpError(403)

    with pytest.raises(RuntimeError, match="all models failed"):
        lg.label_text(POSTING, generate, ["m1", "m2"], sleep=lambda _: None)


# ---------------------------------------------------------------- driver


def test_pending_skips_done_filters_split_and_limits() -> None:
    postings = [{"id": str(i), "split": "test" if i % 2 else "train"} for i in range(10)]
    todo = lg.pending(postings, {"0", "1"}, split=None, limit=None)
    assert [p["id"] for p in todo][:2] == ["2", "3"]
    only_test = lg.pending(postings, set(), split="test", limit=None)
    assert {p["split"] for p in only_test} == {"test"}
    assert len(lg.pending(postings, set(), split=None, limit=3)) == 3


def test_model_candidates_pin_goes_first(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    assert lg.model_candidates() == lg.GEMINI_MODEL_FALLBACK
    monkeypatch.setenv("GEMINI_MODEL", "gemini-3.8-flash")
    models = lg.model_candidates()
    assert models[0] == "gemini-3.8-flash" and models.count("gemini-3.8-flash") == 1


def test_main_is_resumable_and_records_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:  # type: ignore[no-untyped-def]
    raw, out, err = tmp_path / "raw.jsonl", tmp_path / "labels.jsonl", tmp_path / "errors.jsonl"
    rows = [
        {"id": f"id{i}", "split": "test" if i == 0 else "train", "text": POSTING} for i in range(3)
    ]
    raw.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    state = {"fail_ids": {"id1"}}

    def fake_make_generate(api_key: str):  # type: ignore[no-untyped-def]
        def generate(model: str, text: str) -> str:
            # POSTING is shared, so fail by call order: second posting always errors
            generate.n += 1  # type: ignore[attr-defined]
            if generate.n == 2 and state["fail_ids"]:  # type: ignore[attr-defined]
                raise _HttpError(403)
            return json.dumps(GOOD)

        generate.n = 0  # type: ignore[attr-defined]
        return generate

    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setattr(lg, "make_generate", fake_make_generate)
    monkeypatch.setattr(lg, "GEMINI_MODEL_FALLBACK", ["only-model"])
    args = ["--in", str(raw), "--out", str(out), "--errors", str(err), "--rpm", "600000"]

    assert lg.main(args) == 1  # one posting failed
    labeled = [json.loads(line) for line in out.read_text().splitlines()]
    assert [r["id"] for r in labeled] == ["id0", "id2"]
    assert labeled[0]["split"] == "test" and labeled[0]["flags"] == []
    assert json.loads(err.read_text().splitlines()[0])["id"] == "id1"

    state["fail_ids"] = set()
    assert lg.main(args) == 0  # re-run only does the failed one
    ids = [json.loads(line)["id"] for line in out.read_text().splitlines()]
    assert sorted(ids) == ["id0", "id1", "id2"] and len(ids) == 3
    assert "1 to do" in capsys.readouterr().out


def test_main_requires_key_and_input(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert lg.main(["--in", str(tmp_path / "missing.jsonl")]) == 2
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    assert lg.main(["--in", str(tmp_path / "missing.jsonl")]) == 2
