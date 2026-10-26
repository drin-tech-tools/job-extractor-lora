from __future__ import annotations

import argparse
import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from app.models.schemas import JobExtraction
from app.services.extractors import ExtractorUnavailable, ExtractResult

import eval as ev

SENIORITY = ["junior", "mid", "senior", "lead", "senior", "mid"]


def gold_label(i: int) -> dict:
    has_salary = i % 2 == 0
    return {
        "title": f"Engineer {i}",
        "seniority": SENIORITY[i % len(SENIORITY)],
        "required_skills": ["Python", f"Tool{i}"],
        "years_experience_min": i,
        "years_experience_max": None,
        "work_mode": "remote",
        "salary_min": 100000 + i if has_salary else None,
        "salary_max": 120000 + i if has_salary else None,
        "salary_currency": "USD" if has_salary else None,
        "salary_period": "year" if has_salary else None,
    }


N = 6


def write_test_set(path: Path, n: int = N) -> list[dict]:
    rows = [
        {"id": f"id{i}", "text": f"Posting {i} " + "body " * 20, "label": gold_label(i),
         "review_status": "accepted"}
        for i in range(n)
    ]  # fmt: skip
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return rows


class Fake:
    """An extractor whose answer is a function of the posting text."""

    def __init__(self, fn) -> None:  # type: ignore[no-untyped-def]
        self.fn = fn
        self.calls = 0

    async def extract(self, text: str) -> ExtractResult:
        self.calls += 1
        return self.fn(text)


def index_of(text: str) -> int:
    return int(text.split()[1])


def perfect(model_id: str = "fake-perfect"):  # type: ignore[no-untyped-def]
    def fn(text: str) -> ExtractResult:
        gold = JobExtraction.model_validate(gold_label(index_of(text)))
        return ExtractResult(model_id, 20, gold, gold.model_dump_json())

    return fn


def degraded(text: str) -> ExtractResult:
    """Right except it always says 'mid' (correct for 2 of the 6 postings)."""
    label = {**gold_label(index_of(text)), "seniority": "mid"}
    return ExtractResult("fake-degraded", 10, JobExtraction.model_validate(label), "{}")


def broken(text: str) -> ExtractResult:
    return ExtractResult("fake-broken", 5, None, "nope", "invalid_json: no JSON object found")


@pytest.fixture
def ws(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "test": tmp_path / "test_set" / "test.jsonl",
        "preds": tmp_path / "predictions",
        "results": tmp_path / "results",
        "web": tmp_path / "web" / "results.json",
    }
    write_test_set(paths["test"])
    return paths


def cli(ws: dict[str, Path], *extra: str) -> list[str]:
    return [
        "--test-set", str(ws["test"]), "--predictions-dir", str(ws["preds"]),
        "--results-dir", str(ws["results"]), "--web-out", str(ws["web"]),
        "--gemini-rpm", "0", "--resamples", "50", *extra,
    ]  # fmt: skip


def install(monkeypatch: pytest.MonkeyPatch, **fakes: Fake) -> None:
    def get(key, settings):  # type: ignore[no-untyped-def]
        if key not in fakes:
            raise ExtractorUnavailable(f"{key} not configured")
        return fakes[key]

    monkeypatch.setattr(ev, "get_extractor", get)


def latest(ws: dict[str, Path]) -> dict:
    return json.loads((ws["results"] / "latest.json").read_text())


# ---------------------------------------------------------------- inputs


def test_load_test_set_and_limit(ws: dict[str, Path]) -> None:
    items = ev.load_test_set(ws["test"])
    assert [i.id for i in items] == [f"id{i}" for i in range(N)]
    assert len(ev.load_test_set(ws["test"], limit=2)) == 2


def test_load_test_set_rejects_bad_files(tmp_path: Path, ws: dict[str, Path]) -> None:
    with pytest.raises(ev.EvalError, match="finalize"):
        ev.load_test_set(tmp_path / "missing.jsonl")

    empty = tmp_path / "empty.jsonl"
    empty.write_text("\n")
    with pytest.raises(ev.EvalError, match="no postings"):
        ev.load_test_set(empty)

    rows = [json.loads(line) for line in ws["test"].read_text().splitlines()]
    dup = tmp_path / "dup.jsonl"
    dup.write_text("\n".join(json.dumps(r) for r in [rows[0], rows[0]]))
    with pytest.raises(ev.EvalError, match="duplicate id"):
        ev.load_test_set(dup)

    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({**rows[0], "label": {**rows[0]["label"], "seniority": "wizard"}}))
    with pytest.raises(ev.EvalError, match="invalid"):
        ev.load_test_set(bad)


def test_record_from_result_separates_model_faults_from_infrastructure() -> None:
    gold = JobExtraction.model_validate(gold_label(0))
    ok = ev.record_from_result("a", ExtractResult("m", 12, gold, "{}"))
    assert ok and ok["parsed"]["title"] == "Engineer 0" and ok["error"] is None

    invalid = ev.record_from_result("a", ExtractResult("m", 7, None, "x", "invalid_json: nope"))
    assert invalid and invalid["parsed"] is None and invalid["error"].startswith("invalid_json")

    mismatch = ExtractResult("m", 7, None, "x", "schema_mismatch: seniority")
    assert ev.record_from_result("a", mismatch) is not None

    for infra in ("timeout (endpoint may be cold)", "upstream HTTP 503", "gemini-x: HTTP 429"):
        assert ev.record_from_result("a", ExtractResult("m", 1, error=infra)) is None


def test_record_from_raw_parses_the_full_text_not_the_preview() -> None:
    good = json.dumps(gold_label(1))
    rec = ev.record_from_raw("a", good + "\n" + "trailing words " * 400, 33, "colab")
    assert rec["parsed"]["title"] == "Engineer 1" and rec["latency_ms"] == 33
    assert len(rec["raw"]) == 2000  # stored preview is truncated; parsing wasn't

    bad = ev.record_from_raw("a", "I cannot help", 5, "colab")
    assert bad["parsed"] is None and bad["error"].startswith("invalid_json")


def test_read_external_predictions(tmp_path: Path) -> None:
    src = tmp_path / "preds.jsonl"
    src.write_text(
        "\n".join(
            json.dumps(r)
            for r in [
                {"id": "id0", "raw": json.dumps(gold_label(0)), "latency_ms": 40, "model_id": "m"},
                {"id": "id1", "raw": "garbage"},
                {"id": "id0", "raw": json.dumps(gold_label(0)), "latency_ms": 50, "model_id": "m"},
            ]
        )
    )
    records = {r["id"]: r for r in ev.read_external_predictions(src)}
    assert set(records) == {"id0", "id1"}
    assert records["id0"]["latency_ms"] == 50  # last write wins
    assert records["id1"]["parsed"] is None and records["id1"]["model_id"] == "imported"

    src.write_text(json.dumps({"id": "x"}))
    with pytest.raises(ev.EvalError, match="'id' and 'raw'"):
        ev.read_external_predictions(src)
    with pytest.raises(ev.EvalError, match="not found"):
        ev.read_external_predictions(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------- cache + generation


def test_predictions_cache_persists_and_replaces(tmp_path: Path, ws: dict[str, Path]) -> None:
    items = ev.load_test_set(ws["test"])
    store = ev.Predictions(tmp_path / "p" / "m.jsonl")
    assert len(store.missing(items)) == N
    store.add(
        {"id": "id0", "model_id": "m", "latency_ms": 1, "parsed": None, "error": "e", "raw": ""}
    )

    reloaded = ev.Predictions(store.path)
    assert reloaded.get("id0") and len(reloaded.missing(items)) == N - 1
    reloaded.replace_all([])
    assert not ev.Predictions(store.path).records


def test_generate_retries_only_infra_failures(ws: dict[str, Path], tmp_path: Path) -> None:
    items = ev.load_test_set(ws["test"])
    outage = {"on": True}

    def flaky(text: str) -> ExtractResult:
        if outage["on"] and index_of(text) % 2:
            return ExtractResult("m", 1, error="upstream HTTP 503")
        return perfect("m")(text)

    fake, store = Fake(flaky), ev.Predictions(tmp_path / "m.jsonl")
    failures = asyncio.run(ev.generate(items, store, fake, rpm=0, label="m"))
    assert failures == 3 and len(store.records) == 3 and fake.calls == N

    outage["on"] = False  # endpoint recovers; only the 3 uncached ones are requested
    retry = asyncio.run(ev.generate(store.missing(items), store, fake, rpm=0, label="m"))
    assert retry == 0
    assert fake.calls == N + 3 and not store.missing(items)


# ---------------------------------------------------------------- scoring


def test_score_model_counts_missing_and_latencies(ws: dict[str, Path], tmp_path: Path) -> None:
    items = ev.load_test_set(ws["test"])
    store = ev.Predictions(tmp_path / "m.jsonl")
    for item in items[:4]:
        store.add(ev.record_from_result(item.id, perfect("m")(item.text)))  # type: ignore[arg-type]
    scored = ev.score_model(items, store)
    assert scored["missing"] == 2 and len(scored["latencies"]) == 4
    assert scored["model_ids"] == ["m"]
    assert [s["overall"] for s in scored["per_item"]] == [1, 1, 1, 1, 0, 0]


def test_score_model_flags_a_cache_that_no_longer_matches_the_schema(
    ws: dict[str, Path], tmp_path: Path
) -> None:
    items = ev.load_test_set(ws["test"])
    store = ev.Predictions(tmp_path / "m.jsonl")
    store.add({"id": "id0", "model_id": "m", "latency_ms": 1, "parsed": {"title": "x"},
               "error": None, "raw": ""})  # fmt: skip
    with pytest.raises(ev.EvalError, match="--regenerate"):
        ev.score_model(items, store)


def _scored(per_overall: list[float], missing: int = 0) -> dict:
    per_item = [dict.fromkeys(ev.metrics.METRICS, v) for v in per_overall]
    return {"per_item": per_item, "missing": missing, "latencies": [10, 20, 30], "model_ids": ["m"]}


def test_build_results_statuses_and_structure() -> None:
    now = datetime(2026, 1, 2, tzinfo=UTC)
    full = {k: _scored([1, 0, 1, 1]) for k in ev.WEB_ORDER}

    def build(scored: dict, **kw: bool) -> dict:
        args = {"resamples": 50, "seed": 0, "params": {}, "mock": False, "limited": False}
        return ev.build_results(4, scored, **{**args, **kw}, now=now)

    final = build(full)
    assert final["status"] == "final" and final["generated_at"] == "2026-01-02T00:00:00+00:00"
    assert final["models"]["base"]["metrics"]["overall"] == 0.75
    assert final["models"]["base"]["latency"] == {"p50_ms": 20.0, "p95_ms": pytest.approx(29)}
    assert len(final["models"]["base"]["ci"]["overall"]) == 2
    pairs = {(c["a"], c["b"]) for c in final["comparisons"]}
    assert pairs == {("finetuned", "base"), ("finetuned", "gemini")}

    assert build({k: full[k] for k in ("base", "gemini")})["status"] == "partial"
    assert build({**full, "base": _scored([1, 0, 1, 1], missing=1)})["status"] == "incomplete"
    assert build(full, mock=True)["status"] == "mock"
    assert build({"base": full["base"]})["comparisons"] == []  # nothing to compare yet


def test_to_web_preserves_hand_written_content_and_orders_models() -> None:
    existing = {
        "limits": ["a limit"],
        "models": [{"key": "finetuned", "params": "1.5B + LoRA"}, {"key": "base", "params": None}],
    }
    scored = {k: _scored([1, 1, 0, 1]) for k in ("gemini", "finetuned", "base")}
    results = ev.build_results(
        4, scored, resamples=50, seed=0, params={"base": "1.5B"}, mock=False, limited=False
    )
    web = ev.to_web(results, existing)

    assert [m["key"] for m in web["models"]] == ["base", "finetuned", "gemini"]
    by_key = {m["key"]: m for m in web["models"]}
    assert by_key["base"]["params"] == "1.5B"  # CLI value wins
    assert by_key["finetuned"]["params"] == "1.5B + LoRA"  # else the existing value is kept
    assert by_key["base"]["metrics"]["latency_p50_ms"] == 20.0
    assert web["limits"] == ["a limit"] and web["status"] == "final"
    assert {c["metric"] for c in web["comparisons"]} == {"overall"}
    assert "4 hand-checked postings" in web["note"]


def test_to_web_partial_keeps_unevaluated_rows_empty() -> None:
    results = ev.build_results(
        4,
        {"base": _scored([1, 0, 1, 1])},
        resamples=50,
        seed=0,
        params={},
        mock=False,
        limited=False,
    )
    web = ev.to_web(results, {})
    assert web["status"] == "partial" and "Base model" in web["note"]
    assert {m["key"]: m["metrics"] is None for m in web["models"]} == {
        "base": False, "finetuned": True, "gemini": True,
    }  # fmt: skip


def test_markdown_calls_out_noise_vs_real_differences() -> None:
    scored = {
        "base": _scored([0.0] * 20 + [1.0] * 20),
        "finetuned": _scored([1.0] * 40),
        "gemini": _scored([0.0] * 20 + [1.0] * 20),
    }
    results = ev.build_results(
        40, scored, resamples=300, seed=0, params={}, mock=False, limited=False
    )
    md = ev.format_markdown(results)
    assert "| Base + LoRA (fine-tuned) |" in md and "95% confidence intervals" in md
    assert "finetuned vs base" in md and "difference is clear" in md

    same = ev.build_results(
        40, {"base": scored["base"], "finetuned": scored["base"], "gemini": scored["base"]},
        resamples=300, seed=0, params={}, mock=False, limited=False,
    )  # fmt: skip
    assert "within noise" in ev.format_markdown(same)
    assert "valid" in ev.format_table(results) and "n = 40" in ev.format_table(results)


# ---------------------------------------------------------------- CLI end to end


def test_main_scores_three_models_caches_and_publishes(
    ws: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    fakes = {
        "gemini": Fake(perfect("fake-gemini")),
        "finetuned": Fake(degraded),
        "base": Fake(broken),
    }
    install(monkeypatch, **fakes)
    ws["web"].parent.mkdir(parents=True)
    ws["web"].write_text(json.dumps({"limits": ["keep me"], "models": []}))

    assert ev.main(cli(ws, "--write-web", "--params", "finetuned=1.5B + LoRA")) == 0

    res = latest(ws)["models"]
    assert res["gemini"]["metrics"]["overall"] == 1.0
    assert res["base"]["metrics"]["valid_json"] == 0.0 and res["base"]["metrics"]["overall"] == 0.0
    assert res["finetuned"]["metrics"]["valid_json"] == 1.0
    assert res["finetuned"]["metrics"]["seniority"] == pytest.approx(2 / 6)  # 'mid' is right twice
    assert res["finetuned"]["metrics"]["title"] == 1.0
    assert res["gemini"]["model_ids"] == ["fake-gemini"]
    assert (ws["results"] / "latest.md").exists()

    web = json.loads(ws["web"].read_text())
    assert web["status"] == "final" and web["limits"] == ["keep me"]
    assert {m["key"]: m["params"] for m in web["models"]}["finetuned"] == "1.5B + LoRA"

    # Second run: everything is cached, so no model is called again.
    calls = {k: f.calls for k, f in fakes.items()}
    assert ev.main(cli(ws)) == 0
    assert {k: f.calls for k, f in fakes.items()} == calls
    # ...until --regenerate throws the cache away.
    assert ev.main(cli(ws, "--regenerate", "--models", "gemini")) == 0
    assert fakes["gemini"].calls == calls["gemini"] + N


def test_main_infra_outage_is_incomplete_and_blocks_publishing(
    ws: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    def down(text: str) -> ExtractResult:
        if index_of(text) >= 4:
            return ExtractResult("m", 1, error="timeout (endpoint may be cold)")
        return perfect("m")(text)

    install(monkeypatch, gemini=Fake(down))
    assert ev.main(cli(ws, "--models", "gemini", "--write-web")) == 1
    assert latest(ws)["status"] == "incomplete" and latest(ws)["models"]["gemini"]["n_missing"] == 2
    assert not ws["web"].exists()  # nothing was published

    # --allow-incomplete publishes, as partial, with the gap stated in the note.
    assert ev.main(cli(ws, "--models", "gemini", "--write-web", "--allow-incomplete")) == 0
    web = json.loads(ws["web"].read_text())
    assert web["status"] == "partial" and "2 predictions were missing" in web["note"]


def test_main_recovers_after_an_outage_without_redoing_finished_work(
    ws: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    state = {"down": True}

    def maybe(text: str) -> ExtractResult:
        if state["down"] and index_of(text) >= 4:
            return ExtractResult("m", 1, error="upstream HTTP 503")
        return perfect("m")(text)

    fake = Fake(maybe)
    install(monkeypatch, gemini=fake)
    assert ev.main(cli(ws, "--models", "gemini")) == 1
    state["down"] = False
    assert ev.main(cli(ws, "--models", "gemini")) == 0
    assert fake.calls == N + 2  # 6 first time, then only the 2 that had failed
    assert latest(ws)["status"] == "partial"  # gemini alone is still a partial picture


def test_main_imports_predictions_made_elsewhere(
    ws: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch)  # nothing configured: import only
    src = tmp_path / "colab.jsonl"
    rows = [{"id": f"id{i}", "raw": json.dumps(gold_label(i)), "latency_ms": 90} for i in range(3)]
    rows += [{"id": f"id{i}", "raw": "not json"} for i in range(3, N)]
    src.write_text("\n".join(json.dumps(r) for r in rows))

    code = ev.main(
        cli(ws, "--models", "finetuned", "--predictions", f"finetuned={src}", "--no-generate")
    )
    assert code == 0
    m = latest(ws)["models"]["finetuned"]
    assert m["metrics"]["valid_json"] == 0.5 and m["metrics"]["overall"] == 0.5
    assert m["latency"]["p50_ms"] == 90 and m["model_ids"] == ["imported"]


def test_main_refuses_mock_output_and_never_publishes_it(
    ws: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("EXTRACTOR_MOCK", "1")
    from app.config import get_settings

    get_settings.cache_clear()
    assert ev.main(cli(ws)) == 2  # refuses to generate regex output as "model results"
    assert not (ws["results"] / "latest.json").exists()

    assert ev.main(cli(ws, "--allow-mock", "--write-web")) == 1
    assert latest(ws)["status"] == "mock" and not ws["web"].exists()


def test_main_limit_runs_are_never_published(
    ws: dict[str, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    install(monkeypatch, gemini=Fake(perfect()))
    assert ev.main(cli(ws, "--models", "gemini", "--limit", "3", "--write-web")) == 1
    assert latest(ws)["test_set_size"] == 3 and latest(ws)["limited_run"] is True
    assert not ws["web"].exists()


def test_main_skips_unconfigured_models_and_errors_when_nothing_to_score(
    ws: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    install(monkeypatch, gemini=Fake(perfect()))  # base + finetuned unavailable
    assert ev.main(cli(ws)) == 0
    assert set(latest(ws)["models"]) == {"gemini"}
    assert "[base] cannot generate" in capsys.readouterr().out

    install(monkeypatch)
    ws["preds"].joinpath("gemini.jsonl").unlink()
    assert ev.main(cli(ws)) == 2


def test_main_reports_input_problems_without_a_traceback(
    tmp_path: Path, capsys: pytest.CaptureFixture
) -> None:
    assert ev.main(["--test-set", str(tmp_path / "missing.jsonl")]) == 2
    assert "finalize" in capsys.readouterr().err


def test_key_value_parser_validates_model_names() -> None:
    assert ev._key_value("base=1.5B") == ("base", "1.5B")
    for bad in ("base", "base=", "gpt=1B"):
        with pytest.raises(argparse.ArgumentTypeError):
            ev._key_value(bad)
