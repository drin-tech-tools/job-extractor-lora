from __future__ import annotations

import pytest
from app.models.schemas import JobExtraction

import metrics as m

GOLD = JobExtraction.model_validate(
    {
        "title": "Senior Backend Engineer",
        "seniority": "senior",
        "required_skills": ["Python", "SQL", "AWS"],
        "years_experience_min": 5,
        "years_experience_max": None,
        "work_mode": "remote",
        "salary_min": 120000,
        "salary_max": 150000,
        "salary_currency": "USD",
        "salary_period": "year",
    }
)


def pred(**overrides: object) -> JobExtraction:
    return JobExtraction.model_validate({**GOLD.model_dump(mode="json"), **overrides})


def test_perfect_prediction_scores_one_everywhere() -> None:
    assert m.score_item(pred(), GOLD) == dict.fromkeys(m.METRICS, 1.0)


def test_invalid_prediction_scores_zero_everywhere_even_when_gold_is_null() -> None:
    """The reason invalid output is not scored as 'all null': it would earn free credit."""
    no_salary = pred(salary_min=None, salary_max=None, salary_currency=None, salary_period=None)
    assert m.score_item(None, no_salary) == dict.fromkeys(m.METRICS, 0.0)


@pytest.mark.parametrize(
    ("field", "overrides"),
    [
        ("title", {"title": "Staff Engineer"}),
        ("seniority", {"seniority": "lead"}),
        ("work_mode", {"work_mode": "hybrid"}),
        ("years", {"years_experience_min": 3}),
        ("years", {"years_experience_max": 8}),
        ("salary", {"salary_min": 100000}),
        ("salary", {"salary_currency": "EUR"}),
        ("salary", {"salary_period": "month"}),
        ("salary", {"salary_max": None}),
    ],
)
def test_a_single_wrong_field_zeroes_only_that_field(field: str, overrides: dict) -> None:
    scores = m.score_item(pred(**overrides), GOLD)
    assert scores[field] == 0.0
    others = [f for f in m.FIELDS if f != field]
    assert all(scores[f] == 1.0 for f in others)
    assert scores["valid_json"] == 1.0
    assert scores["overall"] == pytest.approx(5 / 6)


def test_title_ignores_case_and_spacing() -> None:
    assert m.score_item(pred(title="senior  BACKEND engineer"), GOLD)["title"] == 1.0


def test_salary_none_is_correct_only_when_gold_has_none() -> None:
    cleared = dict.fromkeys(
        ("salary_min", "salary_max", "salary_currency", "salary_period")
    )
    no_salary_gold = JobExtraction.model_validate({**GOLD.model_dump(mode="json"), **cleared})
    assert m.score_item(pred(**cleared), no_salary_gold)["salary"] == 1.0
    assert m.score_item(pred(), no_salary_gold)["salary"] == 0.0  # hallucinated salary
    assert m.score_item(pred(**cleared), GOLD)["salary"] == 0.0  # missed salary


def test_skills_f1_values() -> None:
    assert m.skills_f1(["python", " SQL "], ["Python", "SQL"]) == 1.0  # case/space-insensitive
    assert m.skills_f1(["Python"], ["Python", "SQL"]) == pytest.approx(2 / 3)
    assert m.skills_f1(["Python", "Go", "Rust", "C"], ["Python"]) == pytest.approx(0.4)
    assert m.skills_f1(["Go"], ["Python"]) == 0.0
    assert m.skills_f1([], []) == 1.0  # nothing required, nothing claimed
    assert m.skills_f1(["Python"], []) == 0.0 and m.skills_f1([], ["Python"]) == 0.0


def test_aggregate_is_a_per_metric_mean() -> None:
    items = [m.score_item(pred(), GOLD), m.score_item(None, GOLD)]
    agg = m.aggregate(items)
    assert agg["valid_json"] == 0.5 and agg["overall"] == 0.5 and agg["title"] == 0.5


def test_percentile() -> None:
    assert m.percentile([], 50) == 0.0
    assert m.percentile([10], 95) == 10
    assert m.percentile([1, 2, 3, 4], 50) == 2.5
    assert m.percentile([0, 100], 95) == pytest.approx(95)
    assert m.percentile([5, 1, 3], 0) == 1 and m.percentile([5, 1, 3], 100) == 5


def test_bootstrap_is_seeded_and_shaped() -> None:
    a = m.bootstrap_samples(10, 50, seed=1)
    assert a == m.bootstrap_samples(10, 50, seed=1) and a != m.bootstrap_samples(10, 50, seed=2)
    assert len(a) == 50 and all(len(s) == 10 and max(s) < 10 for s in a)


def test_mean_ci_brackets_the_mean_and_shrinks_with_more_data() -> None:
    small = [1.0] * 6 + [0.0] * 4
    big = small * 25
    lo_s, hi_s = m.mean_ci(small, m.bootstrap_samples(len(small), 1000, 0))
    lo_b, hi_b = m.mean_ci(big, m.bootstrap_samples(len(big), 1000, 0))
    assert lo_s <= 0.6 <= hi_s and lo_b <= 0.6 <= hi_b
    assert (hi_b - lo_b) < (hi_s - lo_s)


def test_mean_ci_degenerate_cases() -> None:
    samples = m.bootstrap_samples(5, 200, 0)
    assert m.mean_ci([1.0] * 5, samples) == (1.0, 1.0)  # no variance -> zero-width interval
    assert m.mean_ci([], samples) == (0.0, 0.0)


def test_paired_diff_detects_a_real_gap_and_a_null_one() -> None:
    n = 100
    samples = m.bootstrap_samples(n, 1000, 0)
    better = [1.0] * 90 + [0.0] * 10
    worse = [1.0] * 60 + [0.0] * 40
    diff, lo, hi = m.paired_diff_ci(better, worse, samples)
    assert diff == pytest.approx(0.3) and lo > 0  # CI excludes zero

    same_diff, lo, hi = m.paired_diff_ci(better, list(better), samples)
    assert same_diff == 0 and (lo, hi) == (0.0, 0.0)

    with pytest.raises(ValueError):
        m.paired_diff_ci([1.0], [1.0, 0.0], samples)


def test_wilson_matches_known_values() -> None:
    lo, hi = m.wilson_ci(50, 100)
    assert (lo, hi) == (pytest.approx(0.404, abs=1e-3), pytest.approx(0.596, abs=1e-3))
    lo, hi = m.wilson_ci(150, 150)
    assert lo == pytest.approx(0.975, abs=1e-3) and hi == 1.0  # not a zero-width [1, 1]
    lo, hi = m.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.2775, abs=1e-3)
    assert m.wilson_ci(0, 0) == (0.0, 0.0)


def test_interval_picks_wilson_for_pass_fail_and_bootstrap_for_graded_scores() -> None:
    samples = m.bootstrap_samples(8, 200, 0)
    perfect = m.interval([1.0] * 8, samples)
    assert perfect[0] < 0.7 and perfect[1] == 1.0  # honest about n=8; bootstrap would say [1, 1]

    graded = [1.0, 0.5, 0.0, 1.0, 0.5, 1.0, 0.0, 1.0]
    assert m.interval(graded, samples) == m.mean_ci(graded, samples)
    assert m.interval([], samples) == (0.0, 0.0)
