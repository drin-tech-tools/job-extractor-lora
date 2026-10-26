#!/usr/bin/env python3
"""Score base vs fine-tuned vs Gemini on the hand-checked test set.

    cd evals && uv sync --group dev && cp .env.example .env   # fill in what you want to generate
    uv run python eval.py                                     # all three models
    uv run python eval.py --models base gemini                # baseline stage only
    uv run python eval.py --predictions finetuned=colab_preds.jsonl --no-generate
    uv run python eval.py --write-web                         # publish to the Results page

Models are called through the same extractors the API serves with (apps/api), so prompts and
output parsing cannot differ between evaluation and production. Predictions are cached per model
in predictions/<model>.jsonl: re-scoring after a metrics change costs no model calls, and
predictions generated elsewhere (e.g. in the Colab notebook) can be imported with --predictions.

Failures are split in two. A model that answers with bad JSON is scored 0 (it is the model's
fault). A request that fails for infrastructure reasons — timeout, 5xx, rate limit — says nothing
about the model, so it is NOT cached and NOT counted against it; the run is reported incomplete
and refuses to publish until a re-run fills the gap (or you pass --allow-incomplete).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "apps" / "api"))
sys.path.insert(0, str(HERE))

from app.config import get_settings  # noqa: E402
from app.models.schemas import ALL_MODELS, JobExtraction  # noqa: E402
from app.services.extractors import (  # noqa: E402
    ExtractorUnavailable,
    ExtractResult,
    get_extractor,
)
from app.services.prompts import OutputError, parse_model_output  # noqa: E402
from pydantic import ValidationError  # noqa: E402

import metrics  # noqa: E402

DEFAULT_TEST_SET = ROOT / "training" / "data" / "test_set" / "test.jsonl"
DEFAULT_PREDICTIONS = HERE / "predictions"
DEFAULT_RESULTS = HERE / "results"
DEFAULT_WEB_OUT = ROOT / "apps" / "web" / "data" / "results.json"

MODEL_LABELS: dict[str, str] = {
    "base": "Base model (prompted)",
    "finetuned": "Base + LoRA (fine-tuned)",
    "gemini": "Gemini (reference)",
}
WEB_ORDER = ("base", "finetuned", "gemini")
COMPARISONS = (("finetuned", "base"), ("finetuned", "gemini"))
MODEL_SIDE_ERRORS = ("invalid_json", "schema_mismatch")


class EvalError(RuntimeError):
    """A problem with the inputs that the user has to fix; printed without a traceback."""


@dataclass(frozen=True)
class EvalItem:
    id: str
    text: str
    gold: JobExtraction


# ---------------------------------------------------------------- inputs


def load_test_set(path: Path, limit: int | None = None) -> list[EvalItem]:
    if not path.exists():
        raise EvalError(
            f"{path} not found — run `training/data/review_test_set.py finalize` to create it."
        )
    items: list[EvalItem] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in seen:
                raise EvalError(f"{path}:{n}: duplicate id {row['id']}")
            seen.add(row["id"])
            try:
                gold = JobExtraction.model_validate(row["label"])
            except ValidationError as exc:
                raise EvalError(f"{path}:{n}: label for {row['id']} is invalid: {exc}") from exc
            items.append(EvalItem(row["id"], row["text"], gold))
    if not items:
        raise EvalError(f"{path} has no postings")
    return items[:limit] if limit else items


def record_from_raw(item_id: str, raw: str, latency_ms: int, model_id: str) -> dict[str, Any]:
    """Cache record from raw model text (parsed from the FULL text, not a preview)."""
    try:
        parsed, error = parse_model_output(raw).model_dump(mode="json"), None
    except OutputError as exc:
        parsed, error = None, str(exc)
    return {
        "id": item_id,
        "model_id": model_id,
        "latency_ms": latency_ms,
        "parsed": parsed,
        "error": error,
        "raw": raw[:2000],
    }


def record_from_result(item_id: str, result: ExtractResult) -> dict[str, Any] | None:
    """Cache record from a live call, or None if the call failed for infrastructure reasons."""
    if result.error and not result.error.startswith(MODEL_SIDE_ERRORS):
        return None
    return {
        "id": item_id,
        "model_id": result.model_id,
        "latency_ms": result.latency_ms,
        "parsed": result.extraction.model_dump(mode="json") if result.extraction else None,
        "error": result.error,
        "raw": result.raw,
    }


def read_external_predictions(path: Path) -> list[dict[str, Any]]:
    """Lines of {"id", "raw", "latency_ms"?, "model_id"?} produced outside this script."""
    if not path.exists():
        raise EvalError(f"predictions file not found: {path}")
    records: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if "id" not in row or "raw" not in row:
                raise EvalError(f"{path}:{n}: each line needs 'id' and 'raw'")
            records[row["id"]] = record_from_raw(
                row["id"],
                row["raw"],
                int(row.get("latency_ms") or 0),
                row.get("model_id", "imported"),
            )
    return list(records.values())


class Predictions:
    """Append-only JSONL cache of one model's predictions, keyed by posting id."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: dict[str, dict[str, Any]] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        rec = json.loads(line)
                        self.records[rec["id"]] = rec

    def get(self, item_id: str) -> dict[str, Any] | None:
        return self.records.get(item_id)

    def add(self, record: dict[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records[record["id"]] = record

    def replace_all(self, records: list[dict[str, Any]]) -> None:
        self.path.unlink(missing_ok=True)
        self.records = {}
        for rec in records:
            self.add(rec)

    def missing(self, items: list[EvalItem]) -> list[EvalItem]:
        return [it for it in items if it.id not in self.records]


# ---------------------------------------------------------------- generation


async def generate(
    items: list[EvalItem], store: Predictions, extractor: Any, *, rpm: float, label: str
) -> int:
    """Fill the cache for `items`. Returns how many requests failed for infrastructure reasons."""
    interval = 60.0 / rpm if rpm else 0.0
    infra_failures = 0
    for n, item in enumerate(items, 1):
        started = time.monotonic()
        record = record_from_result(item.id, await extractor.extract(item.text))
        if record is None:
            infra_failures += 1
        else:
            store.add(record)
        if n % 25 == 0 or n == len(items):
            print(f"  [{label}] {n}/{len(items)} done, {infra_failures} infra failures")
        if interval and n < len(items):
            await asyncio.sleep(max(interval - (time.monotonic() - started), 0))
    return infra_failures


# ---------------------------------------------------------------- scoring


def score_model(items: list[EvalItem], store: Predictions) -> dict[str, Any]:
    per_item: list[dict[str, float]] = []
    latencies: list[float] = []
    model_ids: set[str] = set()
    missing = 0
    for item in items:
        rec = store.get(item.id)
        if rec is None:
            missing += 1
            per_item.append(metrics.score_item(None, item.gold))  # unscored == invalid, but flagged
            continue
        try:
            pred = JobExtraction.model_validate(rec["parsed"]) if rec["parsed"] else None
        except ValidationError as exc:
            raise EvalError(
                f"{store.path.name}: cached prediction for {item.id} no longer matches the schema "
                f"({exc.errors()[0]['msg']}). Re-run with --regenerate."
            ) from exc
        per_item.append(metrics.score_item(pred, item.gold))
        if rec["latency_ms"]:
            latencies.append(rec["latency_ms"])
        model_ids.add(rec["model_id"])
    return {
        "per_item": per_item,
        "missing": missing,
        "latencies": latencies,
        "model_ids": sorted(model_ids),
    }


def build_results(
    n_items: int,
    scored: dict[str, dict[str, Any]],
    *,
    resamples: int,
    seed: int,
    params: dict[str, str],
    mock: bool,
    limited: bool,
    now: datetime | None = None,
) -> dict[str, Any]:
    samples = metrics.bootstrap_samples(n_items, resamples, seed)
    models: dict[str, Any] = {}
    for key, sc in scored.items():
        per_item = sc["per_item"]
        models[key] = {
            "label": MODEL_LABELS[key],
            "params": params.get(key),
            "model_ids": sc["model_ids"],
            "n_missing": sc["missing"],
            "metrics": metrics.aggregate(per_item),
            "ci": {
                m: list(metrics.interval([it[m] for it in per_item], samples))
                for m in metrics.METRICS
            },
            "latency": {
                "p50_ms": metrics.percentile(sc["latencies"], 50),
                "p95_ms": metrics.percentile(sc["latencies"], 95),
            },
        }

    comparisons = []
    for a, b in COMPARISONS:
        if a in scored and b in scored:
            for m in metrics.METRICS:
                diff, lo, hi = metrics.paired_diff_ci(
                    [it[m] for it in scored[a]["per_item"]],
                    [it[m] for it in scored[b]["per_item"]],
                    samples,
                )
                comparisons.append({"a": a, "b": b, "metric": m, "diff": diff, "ci": [lo, hi]})

    incomplete = any(m["n_missing"] for m in models.values())
    if mock:
        status = "mock"
    elif incomplete:
        status = "incomplete"
    elif set(scored) >= set(WEB_ORDER):
        status = "final"
    else:
        status = "partial"
    return {
        "status": status,
        "generated_at": (now or datetime.now(UTC)).isoformat(timespec="seconds"),
        "test_set_size": n_items,
        "limited_run": limited,
        "bootstrap": {"resamples": resamples, "level": 0.95, "seed": seed},
        "models": models,
        "comparisons": comparisons,
    }


# ---------------------------------------------------------------- output


def _pct(x: float) -> str:
    return f"{100 * x:5.1f}%"


TABLE_COLUMNS = [
    ("valid_json", "valid"), ("title", "title"), ("seniority", "senior."), ("work_mode", "work"),
    ("years", "years"), ("salary", "salary"), ("skills_f1", "skillsF1"), ("overall", "overall"),
]  # fmt: skip


def format_table(results: dict[str, Any]) -> str:
    header = f"{'model':<11}{'missing':>8}" + "".join(f"{h:>9}" for _, h in TABLE_COLUMNS)
    lines = [header + f"{'p50 ms':>9}"]
    for key, m in results["models"].items():
        row = f"{key:<11}{m['n_missing']:>8}" + "".join(
            f"{_pct(m['metrics'][c]):>9}" for c, _ in TABLE_COLUMNS
        )
        lines.append(row + f"{m['latency']['p50_ms']:>9.0f}")
    return f"n = {results['test_set_size']} postings\n" + "\n".join(lines)


def format_markdown(results: dict[str, Any]) -> str:
    n = results["test_set_size"]
    cols = [("valid_json", "Valid JSON"), ("title", "Title"), ("seniority", "Seniority"),
            ("work_mode", "Work mode"), ("years", "Experience"), ("salary", "Salary"),
            ("skills_f1", "Skills F1"), ("overall", "Overall")]  # fmt: skip
    lines = [
        f"Test set: {n} hand-checked postings. Values are means with 95% confidence intervals.",
        "",
        "| Model | " + " | ".join(label for _, label in cols) + " | Latency p50 |",
        "|---|" + "---:|" * (len(cols) + 1),
    ]
    for m in results["models"].values():
        cells = []
        for c, _ in cols:
            lo, hi = m["ci"][c]
            cells.append(f"{100 * m['metrics'][c]:.1f}% ({100 * lo:.1f}–{100 * hi:.1f})")
        lines.append(
            f"| {m['label']} | " + " | ".join(cells) + f" | {m['latency']['p50_ms']:.0f} ms |"
        )
    overall = [c for c in results["comparisons"] if c["metric"] == "overall"]
    if overall:
        lines += ["", "Paired differences in overall score (95% CI):", ""]
        for c in overall:
            lo, hi = c["ci"]
            verdict = "difference is clear" if lo > 0 or hi < 0 else "within noise"
            lines.append(
                f"- {c['a']} vs {c['b']}: {100 * c['diff']:+.1f} pts "
                f"({100 * lo:+.1f} to {100 * hi:+.1f}) — {verdict}"
            )
    return "\n".join(lines) + "\n"


def to_web(results: dict[str, Any], existing: dict[str, Any]) -> dict[str, Any]:
    """Shape `results` for apps/web/data/results.json, keeping the page's hand-written limits."""
    previous = {m["key"]: m for m in existing.get("models", [])}
    models = []
    for key in WEB_ORDER:
        entry: dict[str, Any] = {
            "key": key,
            "label": MODEL_LABELS[key],
            "params": previous.get(key, {}).get("params"),
            "metrics": None,
        }
        scored = results["models"].get(key)
        if scored:
            entry["params"] = scored["params"] or entry["params"]
            entry["metrics"] = {**scored["metrics"], "latency_p50_ms": scored["latency"]["p50_ms"]}
            entry["ci"] = scored["ci"]
            entry["model_ids"] = scored["model_ids"]
        models.append(entry)

    final = results["status"] == "final"
    done = ", ".join(MODEL_LABELS[k] for k in WEB_ORDER if k in results["models"])
    missing = sum(m["n_missing"] for m in results["models"].values())
    note = (
        f"Scored on {results['test_set_size']} hand-checked postings; 95% confidence "
        "intervals shown under each value."
        if final
        else f"Partial results — evaluated so far: {done}."
        + (f" {missing} predictions were missing and counted as invalid." if missing else "")
    )
    return {
        "status": "final" if final else "partial",
        "generated_at": results["generated_at"],
        "test_set_size": results["test_set_size"],
        "note": note,
        "models": models,
        "comparisons": [c for c in results["comparisons"] if c["metric"] == "overall"],
        "limits": existing.get("limits", []),
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- CLI


def _key_value(text: str) -> tuple[str, str]:
    key, sep, value = text.partition("=")
    if not sep or key not in ALL_MODELS or not value:
        raise argparse.ArgumentTypeError(f"expected KEY=VALUE with KEY in {list(ALL_MODELS)}")
    return key, value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET)
    p.add_argument("--models", nargs="+", choices=ALL_MODELS, default=list(ALL_MODELS))
    p.add_argument(
        "--predictions", action="append", default=[], type=_key_value, metavar="KEY=PATH",
        help="import predictions made elsewhere (JSONL of id, raw, latency_ms?, model_id?)",
    )  # fmt: skip
    p.add_argument("--no-generate", action="store_true", help="score cached/imported only")
    p.add_argument("--regenerate", action="store_true", help="discard cached predictions first")
    p.add_argument(
        "--params", action="append", default=[], type=_key_value, metavar="KEY=TEXT",
        help="parameter label for the table, e.g. --params finetuned='1.5B + LoRA'",
    )  # fmt: skip
    p.add_argument("--gemini-rpm", type=float, default=10.0, help="Gemini requests/min (free tier)")
    p.add_argument("--resamples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--limit", type=int, help="first N postings only (smoke test; never published)")
    p.add_argument("--predictions-dir", type=Path, default=DEFAULT_PREDICTIONS)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    p.add_argument("--write-web", action="store_true", help="publish to the Results page")
    p.add_argument("--web-out", type=Path, default=DEFAULT_WEB_OUT)
    p.add_argument("--allow-incomplete", action="store_true")
    p.add_argument("--allow-mock", action="store_true", help="run against EXTRACTOR_MOCK (testing)")
    return p.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    items = load_test_set(args.test_set, args.limit)
    settings = get_settings()
    imported = dict(args.predictions)
    params = dict(args.params)
    scored: dict[str, dict[str, Any]] = {}
    infra: dict[str, int] = {}

    for key in args.models:
        store = Predictions(args.predictions_dir / f"{key}.jsonl")
        if key in imported:
            store.replace_all(read_external_predictions(Path(imported[key])))
            print(f"[{key}] imported {len(store.records)} predictions from {imported[key]}")
        elif args.regenerate:
            store.replace_all([])

        todo = store.missing(items)
        if todo and not args.no_generate:
            if settings.extractor_mock and not args.allow_mock:
                print(
                    "EXTRACTOR_MOCK is on: these would be regex outputs, not model results. "
                    "Unset it, or pass --allow-mock for a pipeline test.",
                    file=sys.stderr,
                )
                return 2
            try:
                extractor = get_extractor(key, settings)
            except ExtractorUnavailable as exc:
                print(f"[{key}] cannot generate: {exc}")
            else:
                rpm = args.gemini_rpm if key == "gemini" else 0.0
                print(f"[{key}] generating {len(todo)} predictions ({len(store.records)} cached)")
                infra[key] = asyncio.run(generate(todo, store, extractor, rpm=rpm, label=key))

        if not store.records:
            print(f"[{key}] no predictions — skipped (generate, or import with --predictions)")
            continue
        scored[key] = score_model(items, store)
        ids = scored[key]["model_ids"]
        if key != "gemini" and len(ids) > 1:
            print(f"[{key}] WARNING: predictions from {len(ids)} model ids {ids} — stale cache?")

    if not scored:
        print("Nothing to score.", file=sys.stderr)
        return 2

    results = build_results(
        len(items), scored, resamples=args.resamples, seed=args.seed, params=params,
        mock=settings.extractor_mock, limited=bool(args.limit),
    )  # fmt: skip
    write_json(args.results_dir / "latest.json", results)
    (args.results_dir / "latest.md").write_text(format_markdown(results), encoding="utf-8")
    print("\n" + format_table(results))
    print(f"\nWrote {args.results_dir / 'latest.json'} and latest.md (status: {results['status']})")

    incomplete = {k: m["n_missing"] for k, m in results["models"].items() if m["n_missing"]}
    if incomplete:
        print(
            f"\nINCOMPLETE: predictions missing {incomplete} (infra failures: {infra or 'none'}). "
            "Missing items are scored as invalid. Re-run to retry them.",
            file=sys.stderr,
        )

    if args.write_web:
        blockers = [
            why for why, bad in [
                ("this was a mock run", results["status"] == "mock"),
                ("--limit was used", bool(args.limit)),
                ("predictions are missing (use --allow-incomplete to publish anyway)",
                 bool(incomplete) and not args.allow_incomplete),
            ] if bad
        ]  # fmt: skip
        if blockers:
            print(f"Not publishing to the web: {'; '.join(blockers)}.", file=sys.stderr)
            return 1
        existing = json.loads(args.web_out.read_text()) if args.web_out.exists() else {}
        write_json(args.web_out, to_web(results, existing))
        print(f"Published to {args.web_out}")
    return 1 if incomplete and not args.allow_incomplete else 0


def main(argv: list[str] | None = None) -> int:
    try:
        return run(parse_args(argv))
    except EvalError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
