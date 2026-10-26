from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pytest
import review_test_set as rv

BASE_LABEL = {
    "title": "Senior Backend Engineer",
    "seniority": "senior",
    "required_skills": ["Python", "SQL"],
    "years_experience_min": 5,
    "years_experience_max": None,
    "work_mode": "remote",
    "salary_min": 120000.0,
    "salary_max": 150000.0,
    "salary_currency": "USD",
    "salary_period": "year",
}


def draft(i: int, *, split: str = "test", flags: list[str] | None = None, **label: object) -> dict:
    return {
        "id": f"id{i:02d}",
        "split": split,
        "text": f"Posting {i}. " + "Details of the role and what we expect. " * 8,
        "label": {**BASE_LABEL, **label},
        "flags": flags or [],
        "labeler": "gemini-test",
    }


class Script:
    """Feeds scripted keystrokes to review_loop and records everything it prints."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []
        self.output: list[str] = []

    def read(self, prompt: str) -> str:
        self.prompts.append(prompt)
        if not self.answers:
            raise EOFError
        return self.answers.pop(0)

    def run(
        self,
        drafts: list[dict],
        reviews: list[dict] | None = None,
        editor: Callable[[str], str] | None = None,
    ) -> tuple[list[dict], list[int]]:
        saves: list[int] = []
        reviews = [] if reviews is None else reviews
        kwargs = {"editor": editor} if editor else {}
        rv.review_loop(
            drafts,
            reviews,
            lambda rs: saves.append(len(rs)),
            read=self.read,
            out=self.output.append,
            **kwargs,
        )
        return reviews, saves


# ---------------------------------------------------------------- apply_edits


def test_edit_scalar_enum_and_title() -> None:
    out = rv.apply_edits(BASE_LABEL, "seniority=Lead; title=Staff Backend Engineer")
    assert out["seniority"] == "lead" and out["title"] == "Staff Backend Engineer"


def test_edit_numbers_accept_k_suffix_and_commas() -> None:
    out = rv.apply_edits(BASE_LABEL, "salary_min=90k; salary_max=110,000; years_experience_max=8")
    assert (out["salary_min"], out["salary_max"], out["years_experience_max"]) == (90000, 110000, 8)


def test_edit_skills_replace_and_clear() -> None:
    out = rv.apply_edits(BASE_LABEL, "required_skills=Go, Rust")
    assert out["required_skills"] == ["Go", "Rust"]
    assert rv.apply_edits(BASE_LABEL, "required_skills=none")["required_skills"] == []


def test_edit_currency_is_uppercased_by_schema() -> None:
    assert rv.apply_edits(BASE_LABEL, "salary_currency=eur")["salary_currency"] == "EUR"


@pytest.mark.parametrize(
    ("cmd", "gone"),
    [("salary=none", rv.GROUPS["salary"]), ("years=none", rv.GROUPS["years"])],
)
def test_group_clear(cmd: str, gone: list[str]) -> None:
    out = rv.apply_edits(BASE_LABEL, cmd)
    assert all(out[f] is None for f in gone)


@pytest.mark.parametrize(
    ("cmd", "message"),
    [
        ("seniority=wizard", "seniority"),  # bad enum, reported by the schema
        ("colour=blue", "unknown field"),
        ("title=none", "can't be null"),
        ("work_mode=null", "can't be null"),
        ("years_experience_min=2.5", "whole number"),
        ("salary_min=lots", "not a number"),
        ("salary=90k", "only accepts none"),
        ("seniority", "expected field=value"),
        (" ; ", "nothing to apply"),
    ],
)
def test_bad_edits_raise_readable_errors(cmd: str, message: str) -> None:
    with pytest.raises(rv.EditError, match=message):
        rv.apply_edits(BASE_LABEL, cmd)


def test_changed_fields_treats_skills_as_a_case_insensitive_set() -> None:
    same = {**BASE_LABEL, "required_skills": ["sql", "PYTHON"]}
    assert rv.changed_fields(BASE_LABEL, same) == []
    diff = {**BASE_LABEL, "work_mode": "hybrid", "salary_max": None}
    assert rv.changed_fields(BASE_LABEL, diff) == ["work_mode", "salary_max"]


# ---------------------------------------------------------------- drafts / ordering


def test_test_drafts_filters_split_and_puts_flagged_first(tmp_path: Path) -> None:
    labels = tmp_path / "labels.jsonl"
    rows = [
        draft(1),
        draft(2, flags=["a"]),
        draft(3, split="train", flags=["a", "b", "c"]),
        draft(4, flags=["a", "b"]),
    ]
    labels.write_text("\n".join(json.dumps(r) for r in rows))
    assert [d["id"] for d in rv.test_drafts(labels)] == ["id04", "id02", "id01"]


# ---------------------------------------------------------------- review loop


def test_accept_and_fix_and_skip() -> None:
    drafts = [draft(1), draft(2), draft(3)]
    script = Script("", "seniority=lead", "", "s")  # accept; fix + confirm; skip the third
    reviews, saves = script.run(drafts)

    by_id = {r["id"]: r for r in reviews}
    assert by_id["id01"]["status"] == "accepted" and by_id["id01"]["changed_fields"] == []
    assert by_id["id02"]["status"] == "corrected"
    assert by_id["id02"]["changed_fields"] == ["seniority"]
    assert by_id["id02"]["label"]["seniority"] == "lead"
    assert by_id["id02"]["draft"]["seniority"] == "senior"  # the original draft is kept
    assert "id03" not in by_id  # skipped stays pending
    assert saves == [1, 2]  # saved after every decision


def test_rejecting_the_confirmation_keeps_editing() -> None:
    script = Script("seniority=lead", "n", "work_mode=hybrid", "")
    reviews, _ = script.run([draft(1)])
    assert reviews[0]["label"]["seniority"] == "senior"  # first edit discarded
    assert reviews[0]["label"]["work_mode"] == "hybrid"


def test_bad_edit_shows_error_and_stays_on_the_same_posting() -> None:
    script = Script("seniority=wizard", "")
    reviews, _ = script.run([draft(1)])
    assert [r["id"] for r in reviews] == ["id01"]
    assert any("!" in line and "seniority" in line for line in script.output)


def test_quit_saves_and_a_new_session_resumes() -> None:
    drafts = [draft(1), draft(2), draft(3)]
    reviews, _ = Script("", "q").run(drafts)
    assert [r["id"] for r in reviews] == ["id01"]

    second = Script("", "")  # resume: only id02, id03 remain
    reviews, _ = second.run(drafts, reviews)
    assert [r["id"] for r in reviews] == ["id01", "id02", "id03"]
    assert sum("[2/3]" in line or "[3/3]" in line for line in second.output) == 2


def test_undo_reshows_the_previous_posting_without_duplicating() -> None:
    drafts = [draft(1), draft(2)]
    # accept 1; undo (on 2); redo 1 with an edit + confirm; accept 2
    script = Script("", "u", "work_mode=onsite", "", "")
    reviews, saves = script.run(drafts)

    assert [r["id"] for r in reviews] == ["id01", "id02"]  # no duplicates
    assert reviews[0]["label"]["work_mode"] == "onsite"  # the redo replaced the original decision
    assert saves == [1, 0, 1, 2]  # the undo was persisted too


def test_undo_with_nothing_to_undo_is_harmless() -> None:
    script = Script("u", "")
    reviews, _ = script.run([draft(1)])
    assert len(reviews) == 1 and any("nothing to undo" in line for line in script.output)


def test_eof_quits_cleanly_with_progress_saved() -> None:
    reviews, saves = Script("").run([draft(1), draft(2)])  # one answer, then EOF
    assert [r["id"] for r in reviews] == ["id01"] and saves == [1]


def test_editor_path_accepts_valid_json_and_rejects_invalid() -> None:
    edited = json.dumps({**BASE_LABEL, "title": "Principal Engineer", "seniority": "lead"})
    ok, _ = Script("e").run([draft(1)], editor=lambda _text: edited)
    assert ok[0]["label"]["title"] == "Principal Engineer"
    assert set(ok[0]["changed_fields"]) == {"title", "seniority"}

    bad = Script("e", "")  # editor returns junk -> error, then accept the draft
    reviews, _ = bad.run([draft(1)], editor=lambda _text: "{not json")
    assert reviews[0]["status"] == "accepted"
    assert any("invalid JSON" in line for line in bad.output)


def test_editor_receives_the_current_label_as_json() -> None:
    seen: list[str] = []

    def editor(text: str) -> str:
        seen.append(text)
        return text

    Script("e").run([draft(1)], editor=editor)
    assert json.loads(seen[0]) == BASE_LABEL


# ---------------------------------------------------------------- stats / finalize


def _reviewed(drafts: list[dict], fixes: dict[str, str]) -> list[dict]:
    out = []
    for d in drafts:
        final = rv.apply_edits(d["label"], fixes[d["id"]]) if d["id"] in fixes else d["label"]
        out.append(rv.make_review(d, final))
    return out


def test_correction_stats() -> None:
    drafts = [draft(i) for i in range(4)]
    reviews = _reviewed(drafts, {"id00": "seniority=lead", "id01": "seniority=mid; salary=none"})
    stats = rv.correction_stats(reviews)
    assert (stats["reviewed"], stats["accepted"], stats["corrected"]) == (4, 2, 2)
    assert stats["fields_changed"]["seniority"] == 2 and stats["fields_changed"]["salary_min"] == 1
    assert "title" not in stats["fields_changed"]
    assert "flatter Gemini" in rv.format_stats(stats, 4)


def test_build_test_set_requires_every_posting_unless_partial() -> None:
    drafts = [draft(1), draft(2), draft(3)]
    reviews = _reviewed(drafts[:2], {"id01": "work_mode=hybrid"})

    records, missing = rv.build_test_set(drafts, reviews, allow_partial=False)
    assert records == [] and missing == ["id03"]

    records, missing = rv.build_test_set(drafts, reviews, allow_partial=True)
    assert [r["id"] for r in records] == ["id01", "id02"] and missing == ["id03"]
    assert records[0]["label"]["work_mode"] == "hybrid"  # final label, not the draft
    assert records[0]["text"].startswith("Posting 1.")


def test_leakage_detects_a_train_copy_of_a_test_text() -> None:
    test = [draft(1)]
    train_copy = {**draft(2, split="train"), "text": test[0]["text"].upper() + "!!"}
    unrelated = draft(3, split="train")
    assert rv.leakage(test, [*test, train_copy, unrelated]) == ["id02"]


# ---------------------------------------------------------------- CLI


@pytest.fixture
def workspace(tmp_path: Path) -> dict[str, Path]:
    labels = tmp_path / "labels.jsonl"
    rows = [draft(i) for i in range(3)] + [draft(10, split="train")]
    labels.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return {
        "labels": labels,
        "reviews": tmp_path / "review" / "reviews.jsonl",
        "out": tmp_path / "test_set" / "test.jsonl",
    }


def _args(ws: dict[str, Path], command: str, *extra: str) -> list[str]:
    return [command, "--labels", str(ws["labels"]), "--reviews", str(ws["reviews"]),
            "--out", str(ws["out"]), *extra]  # fmt: skip


def test_main_review_then_finalize_end_to_end(
    workspace: dict[str, Path], monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    answers = iter(["", "seniority=lead", "", ""])  # accept; fix + confirm; accept
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    assert rv.main(_args(workspace, "review")) == 0

    saved = rv.read_jsonl(workspace["reviews"])
    assert len(saved) == 3
    assert not any("text" in r for r in saved)  # reviews never carry posting text

    capsys.readouterr()
    assert rv.main(_args(workspace, "finalize")) == 0
    out = capsys.readouterr().out
    assert "Reviewed 3/3: 2 accepted as-is, 1 corrected" in out
    assert "Leakage check" in out

    final = rv.read_jsonl(workspace["out"])
    assert [r["id"] for r in final] == ["id00", "id01", "id02"]  # test split only, train excluded
    assert {r["review_status"] for r in final} == {"accepted", "corrected"}


def test_finalize_refuses_unfinished_review_but_allows_partial(workspace: dict[str, Path]) -> None:
    drafts = rv.test_drafts(workspace["labels"])
    rv.write_jsonl(workspace["reviews"], _reviewed(drafts[:1], {}))

    assert rv.main(_args(workspace, "finalize")) == 1
    assert not workspace["out"].exists()

    assert rv.main(_args(workspace, "finalize", "--allow-partial")) == 0
    assert len(rv.read_jsonl(workspace["out"])) == 1


def test_finalize_blocks_on_leakage(workspace: dict[str, Path]) -> None:
    rows = rv.read_jsonl(workspace["labels"])
    rows.append({**draft(99, split="train"), "text": rows[0]["text"]})
    rv.write_jsonl(workspace["labels"], rows)
    rv.write_jsonl(workspace["reviews"], _reviewed(rv.test_drafts(workspace["labels"]), {}))

    assert rv.main(_args(workspace, "finalize")) == 1
    assert not workspace["out"].exists()


def test_main_status_and_error_paths(workspace: dict[str, Path], tmp_path: Path) -> None:
    assert rv.main(_args(workspace, "status")) == 0

    missing = _args(workspace, "status")
    missing[missing.index("--labels") + 1] = str(tmp_path / "nope.jsonl")
    assert rv.main(missing) == 2  # no labels yet

    ghost = {"id": "ghost", "status": "accepted", "changed_fields": []}
    rv.write_jsonl(workspace["reviews"], [ghost])
    assert rv.main(_args(workspace, "status")) == 2  # review that matches no draft
