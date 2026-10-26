#!/usr/bin/env python3
"""Label raw postings with Gemini (distillation: big model -> training data for a small one).

Reads data/raw/postings.jsonl, writes data/labeled/labels.jsonl. Resumable: ids already in
the output are skipped, so a rate-limit stop or Ctrl-C costs nothing. Labels for the test
split are only DRAFTS — they become ground truth after hand review, not before.

Each label carries `flags`: cheap checks that the label is grounded in the posting text
(salary digits present, skills mentioned, title present). Flagged rows are where Gemini is
most likely wrong, so review those first.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
import time
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING.parent / "apps" / "api"))

from app.config import GEMINI_MODEL_FALLBACK  # noqa: E402
from app.models.schemas import JobExtraction  # noqa: E402
from app.services.prompts import EXTRACTION_RULES  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_IN = TRAINING / "data" / "raw" / "postings.jsonl"
DEFAULT_OUT = TRAINING / "data" / "labeled" / "labels.jsonl"
DEFAULT_ERRORS = TRAINING / "data" / "labeled" / "errors.jsonl"


# ---------------------------------------------------------------- grounding checks


def _salary_tokens(value: float) -> set[str]:
    """Ways a salary figure might be written: 120000, 120,000, 120k, 120.5k."""
    n = int(value) if float(value).is_integer() else value
    tokens = {str(n), f"{n:,}"}
    if isinstance(n, int) and n >= 1000 and n % 100 == 0:
        k = n / 1000
        tokens.add(f"{k:g}k")
    return tokens


def grounding_flags(label: JobExtraction, text: str) -> list[str]:
    """Return human-readable reasons the label may not be supported by `text`."""
    flags: list[str] = []
    lowered = text.lower()
    squashed = lowered.replace(" ", "")

    if label.title and label.title.lower() not in lowered:
        flags.append("title_not_in_text")

    missing = [s for s in label.required_skills if s.lower() not in lowered]
    if missing:
        flags.append(f"skills_not_in_text:{'|'.join(missing[:5])}")

    for name, value in (("salary_min", label.salary_min), ("salary_max", label.salary_max)):
        if value is not None and not any(t in squashed for t in _salary_tokens(value)):
            flags.append(f"{name}_not_in_text")

    if (label.salary_min is None) != (label.salary_max is None):
        flags.append("salary_half_filled")
    if label.salary_min is not None and not (label.salary_currency and label.salary_period):
        flags.append("salary_missing_currency_or_period")
    if (
        label.salary_min is not None
        and label.salary_max is not None
        and label.salary_min > label.salary_max
    ):
        flags.append("salary_min_gt_max")

    yrs = [label.years_experience_min, label.years_experience_max]
    for y in yrs:
        if y is not None and not re.search(rf"\b{y}\b|\b{y}\+", lowered):
            flags.append("years_not_in_text")
            break
    if (
        label.years_experience_min is not None
        and label.years_experience_max is not None
        and label.years_experience_min > label.years_experience_max
    ):
        flags.append("years_min_gt_max")
    return flags


# ---------------------------------------------------------------- Gemini calls


def model_candidates() -> list[str]:
    pinned = os.getenv("GEMINI_MODEL")
    if not pinned:
        return list(GEMINI_MODEL_FALLBACK)
    return [pinned, *[m for m in GEMINI_MODEL_FALLBACK if m != pinned]]


def make_generate(api_key: str) -> Callable[[str, str], str]:
    """Return generate(model, posting_text) -> raw JSON string."""
    from google import genai
    from google.genai import types

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        system_instruction=EXTRACTION_RULES,
        response_mime_type="application/json",
        response_schema=JobExtraction,
        temperature=0.0,
    )

    def generate(model: str, text: str) -> str:
        resp = client.models.generate_content(
            model=model, contents=f"JOB POSTING:\n\n{text}", config=config
        )
        return resp.text or ""

    return generate


def _status(exc: Exception) -> int | None:
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)
    return code if isinstance(code, int) else None


def label_text(
    text: str,
    generate: Callable[[str, str], str],
    models: list[str],
    *,
    retries: int = 4,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[JobExtraction, str]:
    """Label one posting; retry transient errors, fall to the next model if one is unavailable.

    Returns (label, model_used). Raises RuntimeError when every model is exhausted.
    """
    last_error: Exception | None = None
    for model in models:
        for attempt in range(retries + 1):
            try:
                raw = generate(model, text)
                return JobExtraction.model_validate_json(raw), model
            except ValidationError as exc:  # model replied, but not with our schema
                last_error = exc
                if attempt >= 1:  # one retry at temperature 0 is enough to rule out a blip
                    break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
                status = _status(exc)
                if status in (400, 403, 404):  # model unavailable to this key -> next model
                    logger.warning("Model %s unavailable (%s); trying next", model, status)
                    break
                wait = min(2**attempt * 2, 60)  # 429 / 5xx / network
                logger.info("%s error on %s (%s); retry in %ds", status, model, exc, wait)
                sleep(wait)
    raise RuntimeError(f"all models failed; last error: {last_error}")


# ---------------------------------------------------------------- driver


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def pending(
    postings: list[dict[str, Any]], done: set[str], *, split: str | None, limit: int | None
) -> list[dict[str, Any]]:
    todo = [p for p in postings if p["id"] not in done and (split is None or p["split"] == split)]
    return todo[:limit] if limit else todo


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--in", dest="inp", type=Path, default=DEFAULT_IN)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--errors", type=Path, default=DEFAULT_ERRORS)
    p.add_argument("--split", choices=["train", "test"], help="label only this split")
    p.add_argument("--limit", type=int, help="label at most N postings this run")
    p.add_argument("--rpm", type=float, default=10.0, help="requests/min cap (free tier)")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("GEMINI_API_KEY is not set (see .env.example)", file=sys.stderr)
        return 2
    if not args.inp.exists():
        print(f"{args.inp} not found — run data/build_dataset.py first", file=sys.stderr)
        return 2

    postings = read_jsonl(args.inp)
    done = {r["id"] for r in read_jsonl(args.out)}
    todo = pending(postings, done, split=args.split, limit=args.limit)
    print(f"{len(postings)} postings, {len(done)} already labeled, {len(todo)} to do")

    generate = make_generate(api_key)
    models = model_candidates()
    interval = 60.0 / args.rpm
    args.out.parent.mkdir(parents=True, exist_ok=True)

    ok = failed = flagged = 0
    models_used: Counter = Counter()
    args.errors.parent.mkdir(parents=True, exist_ok=True)
    with (
        args.out.open("a", encoding="utf-8") as out,
        args.errors.open("a", encoding="utf-8") as err,
    ):
        for i, posting in enumerate(todo, 1):
            started = time.monotonic()
            try:
                label, model = label_text(posting["text"], generate, models)
            except RuntimeError as exc:
                failed += 1
                err.write(json.dumps({"id": posting["id"], "error": str(exc)}) + "\n")
                err.flush()
                logger.warning("[%d/%d] %s failed", i, len(todo), posting["id"])
                continue

            flags = grounding_flags(label, posting["text"])
            record = {
                "id": posting["id"],
                "split": posting["split"],
                "text": posting["text"],
                "label": label.model_dump(mode="json"),
                "flags": flags,
                "labeler": model,
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            out.flush()
            ok += 1
            flagged += bool(flags)
            models_used[model] += 1
            if i % 25 == 0:
                print(f"  [{i}/{len(todo)}] ok={ok} failed={failed} flagged={flagged}")
            time.sleep(max(interval - (time.monotonic() - started), 0))

    print(f"\nLabeled {ok}, failed {failed}, flagged {flagged}. Models: {dict(models_used)}")
    print(f"  labels: {args.out}")
    if failed:
        print(f"  errors: {args.errors} (re-run to retry — finished ids are skipped)")
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
