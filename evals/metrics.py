"""Scoring for the extraction task. Pure functions — no I/O, no network, no model calls.

Every model goes through `score_item`: its output is already parsed into a `JobExtraction`
(or None when it couldn't be), and is compared field by field with the hand-checked label.
The definitions here are the ones documented in evals/README.md.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from statistics import NormalDist

from app.models.schemas import JobExtraction

FIELDS = ("title", "seniority", "work_mode", "years", "salary", "skills_f1")
METRICS = ("valid_json", *FIELDS, "overall")


# ---------------------------------------------------------------- per-item scoring


def norm_text(value: str) -> str:
    return " ".join(value.casefold().split())


def skills_f1(pred: Sequence[str], gold: Sequence[str]) -> float:
    """Set F1 over case/whitespace-normalised skills. Both empty counts as a perfect match."""
    p, g = {norm_text(s) for s in pred if s.strip()}, {norm_text(s) for s in gold if s.strip()}
    if not p and not g:
        return 1.0
    hits = len(p & g)
    if not hits:
        return 0.0
    precision, recall = hits / len(p), hits / len(g)
    return 2 * precision * recall / (precision + recall)


def _same_number(a: float | None, b: float | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-6)


def _same_str(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return norm_text(a) == norm_text(b)


def score_item(pred: JobExtraction | None, gold: JobExtraction) -> dict[str, float]:
    """Score one posting. A missing/invalid prediction scores 0 on EVERY metric.

    That is deliberate: if invalid output were scored as an all-null answer, a model that always
    fails would still get credit wherever the gold label is null (most postings have no salary).
    """
    if pred is None:
        return dict.fromkeys(METRICS, 0.0)

    field_scores = {
        "title": float(norm_text(pred.title) == norm_text(gold.title)),
        "seniority": float(pred.seniority == gold.seniority),
        "work_mode": float(pred.work_mode == gold.work_mode),
        "years": float(
            pred.years_experience_min == gold.years_experience_min
            and pred.years_experience_max == gold.years_experience_max
        ),
        # All four salary fields must match; "no salary" is right only if the model also says so.
        "salary": float(
            _same_number(pred.salary_min, gold.salary_min)
            and _same_number(pred.salary_max, gold.salary_max)
            and _same_str(pred.salary_currency, gold.salary_currency)
            and pred.salary_period == gold.salary_period
        ),
        "skills_f1": skills_f1(pred.required_skills, gold.required_skills),
    }
    return {
        "valid_json": 1.0,
        **field_scores,
        "overall": sum(field_scores.values()) / len(field_scores),
    }


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def aggregate(items: Sequence[dict[str, float]]) -> dict[str, float]:
    return {m: mean([it[m] for it in items]) for m in METRICS}


# ---------------------------------------------------------------- latency


def percentile(values: Sequence[float], q: float) -> float:
    """Linear-interpolated percentile, q in [0, 100]. Empty input -> 0."""
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


# ---------------------------------------------------------------- bootstrap


def bootstrap_samples(n: int, resamples: int, seed: int) -> list[list[int]]:
    """Index resamples drawn once and shared by every metric and model, so comparisons between
    models are paired (the same resampled postings are scored for both)."""
    rng = random.Random(seed)
    return [[rng.randrange(n) for _ in range(n)] for _ in range(resamples)]


def _interval(stats: list[float], level: float) -> tuple[float, float]:
    ordered = sorted(stats)
    alpha = (1 - level) / 2
    lo = ordered[int(alpha * len(ordered))]
    hi = ordered[min(len(ordered) - 1, int((1 - alpha) * len(ordered)))]
    return lo, hi


def mean_ci(
    values: Sequence[float], samples: list[list[int]], level: float = 0.95
) -> tuple[float, float]:
    """Percentile-bootstrap confidence interval for the mean of `values`."""
    if not values or not samples:
        return 0.0, 0.0
    return _interval([mean([values[i] for i in idx]) for idx in samples], level)


def wilson_ci(successes: float, n: int, level: float = 0.95) -> tuple[float, float]:
    """Wilson score interval for a proportion.

    Used for pass/fail metrics instead of the bootstrap because a percentile bootstrap collapses
    to a zero-width interval at 0% or 100% (8/8 would read as "certainly 100%"). Wilson doesn't:
    150/150 still has a lower bound of about 97.5%.
    """
    if n <= 0:
        return 0.0, 0.0
    z = NormalDist().inv_cdf((1 + level) / 2)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    # At the boundaries the bounds are exactly 0 and 1; snap them rather than publish float noise.
    lo = 0.0 if successes <= 0 else max(0.0, centre - half)
    hi = 1.0 if successes >= n else min(1.0, centre + half)
    return lo, hi


def interval(
    values: Sequence[float], samples: list[list[int]], level: float = 0.95
) -> tuple[float, float]:
    """Confidence interval for a metric's mean: Wilson if every score is 0 or 1, else bootstrap."""
    if values and all(v in (0.0, 1.0) for v in values):
        return wilson_ci(sum(values), len(values), level)
    return mean_ci(values, samples, level)


def paired_diff_ci(
    a: Sequence[float], b: Sequence[float], samples: list[list[int]], level: float = 0.95
) -> tuple[float, float, float]:
    """(mean(a) - mean(b), lo, hi) with a paired bootstrap over the same postings."""
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length score lists")
    if not a or not samples:
        return 0.0, 0.0, 0.0
    diffs = [x - y for x, y in zip(a, b, strict=True)]
    lo, hi = _interval([mean([diffs[i] for i in idx]) for idx in samples], level)
    return mean(diffs), lo, hi
