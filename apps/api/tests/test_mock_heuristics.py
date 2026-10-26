from __future__ import annotations

import pytest

from app.services.extractors import heuristic_extract
from tests.conftest import SAMPLE_POSTING


def test_sample_posting() -> None:
    e = heuristic_extract(SAMPLE_POSTING)
    assert e.title == "Senior Backend Engineer"
    assert e.seniority == "senior"
    assert e.work_mode == "remote"
    assert e.years_experience_min == 5
    assert (e.salary_min, e.salary_max) == (120000, 150000)
    assert (e.salary_currency, e.salary_period) == ("USD", "year")
    assert {"Python", "SQL", "AWS", "Docker"} <= set(e.required_skills)


@pytest.mark.parametrize(
    ("text", "lo", "hi", "period"),
    [
        ("Designer. Pay: $50-70k a year.", 50000, 70000, "year"),
        ("Contractor, €25 - €40 per hour, Berlin.", 25, 40, "hour"),
    ],
)
def test_salary_variants(text: str, lo: float, hi: float, period: str) -> None:
    e = heuristic_extract(text)
    assert (e.salary_min, e.salary_max, e.salary_period) == (lo, hi, period)


def test_no_salary_no_fields() -> None:
    e = heuristic_extract("Office Manager\nOn-site in Manila. 2-4 years experience.")
    assert e.salary_min is None and e.salary_currency is None
    assert (e.years_experience_min, e.years_experience_max) == (2, 4)
    assert e.work_mode == "onsite"
