from __future__ import annotations

import json
import random
from pathlib import Path

import build_dataset as bd


def _posting(i: int, **meta: str) -> dict:
    """A record with a unique >=300-char body."""
    text = f"Posting number {i}. " + ("We build things and need people who care. " * 12)
    rec = bd.make_record("test", text, **meta)
    assert rec is not None
    return rec


# ---------------------------------------------------------------- cleaning


def test_clean_text_converts_html_keeping_structure() -> None:
    raw = "<h2>Role</h2><p>Build &amp; ship APIs.</p><ul><li>Python</li><li>SQL</li></ul>"
    text = bd.clean_text(raw)
    assert "Build & ship APIs." in text
    assert "- Python\n- SQL" in text
    assert "<" not in text


def test_clean_text_leaves_plain_text_alone_but_normalises_whitespace() -> None:
    assert bd.clean_text("Senior   dev\xa0needed\n\n\n\n\nApply   now") == (
        "Senior dev needed\n\nApply now"
    )


def test_clean_text_truncates_on_a_line_boundary() -> None:
    raw = "\n".join(f"line {i} " + "x" * 80 for i in range(200))
    text = bd.clean_text(raw)
    assert len(text) <= bd.MAX_CHARS
    assert text.splitlines()[-1].endswith("x" * 80)  # no half-cut final line


def test_make_record_drops_stubs() -> None:
    assert bd.make_record("test", "Too short to be a posting") is None


def test_fingerprint_ignores_case_punctuation_and_spacing() -> None:
    a = "Senior Engineer — apply now!  Python/SQL"
    b = "senior engineer apply now python sql"
    assert bd.fingerprint(a) == bd.fingerprint(b)
    assert bd.fingerprint(a) != bd.fingerprint(a + " rust")


# ---------------------------------------------------------------- dedupe


def test_dedupe_by_text_role_and_against_existing() -> None:
    existing = [_posting(1, company="Acme", title="Data Engineer")]
    same_text = dict(existing[0])
    same_role = _posting(2, company="ACME", title="data engineer!")  # a repost, new text
    different = _posting(3, company="Acme", title="Designer")
    no_meta_a, no_meta_b = _posting(4), _posting(4)

    kept, dropped = bd.dedupe(existing, [same_text, same_role, different, no_meta_a, no_meta_b])

    assert [r["id"] for r in kept] == [different["id"], no_meta_a["id"]]
    assert dropped == {"duplicate_text": 2, "duplicate_role": 1}


# ---------------------------------------------------------------- splits


def test_assign_splits_fills_test_quota_then_train() -> None:
    new = [_posting(i) for i in range(20)]
    merged = bd.assign_splits([], new, test_size=5, seed=1)
    assert sum(r["split"] == "test" for r in merged) == 5
    assert sum(r["split"] == "train" for r in merged) == 15


def test_assign_splits_is_independent_of_input_order() -> None:
    new = [_posting(i) for i in range(30)]
    shuffled = random.Random(0).sample(new, len(new))

    def splits(records: list[dict]) -> dict[str, str]:
        merged = bd.assign_splits([], [dict(r) for r in records], test_size=7, seed=3)
        return {r["id"]: r["split"] for r in merged}

    assert splits(new) == splits(shuffled)


def test_existing_assignments_never_move() -> None:
    first = bd.assign_splits([], [_posting(i) for i in range(10)], test_size=4, seed=1)
    frozen = {r["id"]: r["split"] for r in first}

    more = [_posting(i) for i in range(10, 25)]
    second = bd.assign_splits(first, more, test_size=4, seed=1)

    for rec in second:
        if rec["id"] in frozen:
            assert rec["split"] == frozen[rec["id"]]
    assert sum(r["split"] == "test" for r in second) == 4  # quota was already met


def test_quota_topped_up_when_corpus_grows() -> None:
    first = bd.assign_splits([], [_posting(i) for i in range(3)], test_size=10, seed=1)
    assert all(r["split"] == "test" for r in first)  # not enough data yet: everything is test
    second = bd.assign_splits(first, [_posting(i) for i in range(3, 20)], test_size=10, seed=1)
    assert sum(r["split"] == "test" for r in second) == 10


# ---------------------------------------------------------------- CLI end to end


def _write_source(path: Path, rows: range) -> None:
    with path.open("w") as fh:
        for i in rows:
            body = f"Role {i}. " + ("Responsibilities and requirements go here. " * 12)
            fh.write(
                json.dumps({"description": f"<p>{body}</p>", "title": f"T{i}", "company": f"C{i}"})
                + "\n"
            )


def test_main_is_incremental_and_keeps_the_test_set_frozen(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    out = tmp_path / "raw" / "postings.jsonl"
    src = tmp_path / "src.jsonl"

    _write_source(src, range(0, 12))
    args = ["--jsonl", str(src), "--out", str(out), "--test-size", "4", "--target", "100"]
    assert bd.main(args) == 0
    first = {r["id"]: r["split"] for r in bd.load_existing(out)}
    assert len(first) == 12 and sum(v == "test" for v in first.values()) == 4

    _write_source(src, range(0, 30))  # same rows + 18 new ones
    assert bd.main(args) == 0
    second = {r["id"]: r["split"] for r in bd.load_existing(out)}

    assert len(second) == 30  # old rows were de-duplicated, not doubled
    assert all(second[i] == split for i, split in first.items())
    assert sum(v == "test" for v in second.values()) == 4
    assert "added 18 new" in capsys.readouterr().out


def test_main_respects_target(tmp_path: Path) -> None:
    out, src = tmp_path / "p.jsonl", tmp_path / "s.jsonl"
    _write_source(src, range(0, 40))
    bd.main(["--jsonl", str(src), "--out", str(out), "--test-size", "2", "--target", "15"])
    assert len(bd.load_existing(out)) == 15
