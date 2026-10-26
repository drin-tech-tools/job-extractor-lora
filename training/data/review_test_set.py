#!/usr/bin/env python3
"""Hand-check the test split's Gemini draft labels and freeze them as ground truth.

    review    step through the drafts: accept, or fix fields. Resumable; saved after each one.
    status    progress and how often the drafts needed fixing.
    finalize  write data/test_set/test.jsonl — the file evals/eval.py scores against.

At the review prompt:
    Enter / y         accept the draft as-is
    field=value       fix a field, e.g.  seniority=lead  salary_min=90k  required_skills=Python,SQL
                      several at once with ';'.  null/none clears a nullable field.
                      salary=none / years=none clear every salary_* / years_* field.
    e                 edit the whole label as JSON in $EDITOR
    s                 skip for now (comes back next session)      u   undo the last decision
    q                 quit (everything so far is saved)

Decisions are stored WITHOUT posting text in data/review/reviews.jsonl (small, and your own work,
so it is safe to commit). Only `finalize` joins them back to the text.

Caveat worth keeping in the write-up: reviewing a draft anchors you toward accepting it, so
agreement between the drafts and the final labels is an optimistic estimate of Gemini's accuracy.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from collections import Counter, deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING.parent / "apps" / "api"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.models.schemas import JobExtraction  # noqa: E402
from build_dataset import fingerprint  # noqa: E402

DEFAULT_LABELS = TRAINING / "data" / "labeled" / "labels.jsonl"
DEFAULT_REVIEWS = TRAINING / "data" / "review" / "reviews.jsonl"
DEFAULT_TEST_SET = TRAINING / "data" / "test_set" / "test.jsonl"

FIELDS = list(JobExtraction.model_fields)
NULLABLE = {
    "years_experience_min",
    "years_experience_max",
    "salary_min",
    "salary_max",
    "salary_currency",
    "salary_period",
}
INT_FIELDS = {"years_experience_min", "years_experience_max"}
FLOAT_FIELDS = {"salary_min", "salary_max"}
NULL_WORDS = {"null", "none", "-", ""}
GROUPS = {
    "salary": ["salary_min", "salary_max", "salary_currency", "salary_period"],
    "years": ["years_experience_min", "years_experience_max"],
}


class EditError(ValueError):
    """A review edit the user typed that can't be applied; the message is shown back to them."""


# ---------------------------------------------------------------- edits


def _number(raw: str) -> float:
    """'90000', '90,000', '90k', '1.5m' -> float."""
    text = raw.strip().lower().replace(",", "").replace("_", "")
    mult = 1.0
    if text.endswith("k"):
        mult, text = 1_000.0, text[:-1]
    elif text.endswith("m"):
        mult, text = 1_000_000.0, text[:-1]
    try:
        return float(text) * mult
    except ValueError as exc:
        raise EditError(f"not a number: {raw!r}") from exc


def _validate(label: dict[str, Any]) -> dict[str, Any]:
    try:
        return JobExtraction.model_validate(label).model_dump(mode="json")
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(p) for p in first["loc"])
        raise EditError(f"{loc}: {first['msg']}") from exc


def apply_edits(label: dict[str, Any], command: str) -> dict[str, Any]:
    """Apply `field=value; field=value` to a label and return the validated result."""
    updated = dict(label)
    parts = [p.strip() for p in command.split(";") if p.strip()]
    if not parts:
        raise EditError("nothing to apply")
    for part in parts:
        if "=" not in part:
            raise EditError(f"expected field=value, got {part!r}")
        name, _, value = part.partition("=")
        name, value = name.strip(), value.strip()
        is_null = value.lower() in NULL_WORDS

        if name in GROUPS:
            if not is_null:
                raise EditError(f"{name}= only accepts none (set the individual fields instead)")
            for member in GROUPS[name]:
                updated[member] = None
        elif name not in FIELDS:
            raise EditError(f"unknown field {name!r}; fields: {', '.join(FIELDS)}")
        elif name == "required_skills":
            updated[name] = [] if is_null else [s.strip() for s in value.split(",") if s.strip()]
        elif is_null:
            if name not in NULLABLE:
                raise EditError(f"{name} can't be null")
            updated[name] = None
        elif name in INT_FIELDS:
            number = _number(value)
            if number != int(number):
                raise EditError(f"{name} must be a whole number")
            updated[name] = int(number)
        elif name in FLOAT_FIELDS:
            updated[name] = _number(value)
        else:
            updated[name] = value.lower() if name != "title" else value
    return _validate(updated)


def parse_json_edit(text: str) -> dict[str, Any]:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise EditError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise EditError("expected a JSON object")
    return _validate(data)


def _canon(field: str, value: Any) -> Any:
    if field == "required_skills":
        return sorted({" ".join(s.casefold().split()) for s in value})
    return value


def changed_fields(draft: dict[str, Any], final: dict[str, Any]) -> list[str]:
    """Fields whose value differs; skills compare as a case-insensitive set."""
    return [f for f in FIELDS if _canon(f, draft.get(f)) != _canon(f, final.get(f))]


# ---------------------------------------------------------------- storage


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def test_drafts(labels_path: Path) -> list[dict[str, Any]]:
    """Gemini drafts for the test split, riskiest first (most grounding flags), then by id."""
    drafts = [r for r in read_jsonl(labels_path) if r["split"] == "test"]
    return sorted(drafts, key=lambda r: (-len(r.get("flags") or []), r["id"]))


def make_review(draft: dict[str, Any], final: dict[str, Any]) -> dict[str, Any]:
    changed = changed_fields(draft["label"], final)
    return {
        "id": draft["id"],
        "status": "corrected" if changed else "accepted",
        "label": final,
        "draft": draft["label"],
        "changed_fields": changed,
        "flags": draft.get("flags") or [],
        "reviewed_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }


# ---------------------------------------------------------------- interactive review


def render(draft: dict[str, Any], label: dict[str, Any], pos: int, total: int) -> str:
    flags = ", ".join(draft.get("flags") or []) or "none"
    rows = []
    for field in FIELDS:
        value = label[field]
        shown = ", ".join(value) if field == "required_skills" else value
        rows.append(f"  {field:<22} {'—' if shown in (None, '', []) else shown}")
    return (
        f"\n[{pos}/{total}]  id={draft['id']}  flags: {flags}\n"
        f"{'─' * 24} POSTING {'─' * 24}\n{draft['text']}\n"
        f"{'─' * 20} DRAFT LABEL ({draft.get('labeler', 'gemini')}) {'─' * 20}\n" + "\n".join(rows)
    )


def edit_in_editor(text: str) -> str:
    """Open `text` in $EDITOR (default vi) and return what was saved."""
    editor = os.environ.get("EDITOR") or os.environ.get("VISUAL") or "vi"
    with tempfile.NamedTemporaryFile("w+", suffix=".json", delete=False, encoding="utf-8") as fh:
        fh.write(text)
        path = fh.name
    try:
        subprocess.run([*editor.split(), path], check=True)
        return Path(path).read_text(encoding="utf-8")
    finally:
        os.unlink(path)


def review_loop(
    drafts: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
    save: Callable[[list[dict[str, Any]]], None],
    *,
    read: Callable[[str], str] | None = None,
    out: Callable[[str], None] = print,
    editor: Callable[[str], str] = edit_in_editor,
) -> list[dict[str, Any]]:
    """Run one review session. `reviews` is mutated and saved after every decision."""
    read = read or input  # resolved per call, not at import, so tests can swap stdin
    done = {r["id"] for r in reviews}
    queue = deque(d for d in drafts if d["id"] not in done)
    total = len(drafts)
    history: list[dict[str, Any]] = []  # drafts decided this session, newest last, for undo
    skipped = 0
    out(f"{len(done)}/{total} reviewed so far, {len(queue)} to go.")
    out("Enter=accept  field=value  e=editor  s=skip  u=undo  q=quit")

    def ask(prompt: str) -> str:
        try:
            return read(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            return "q"  # decisions are saved as we go, so quitting is always safe

    while queue:
        draft = queue.popleft()
        label = draft["label"]
        out(render(draft, label, len(reviews) + 1, total))
        while True:  # stay on this posting until it is decided, skipped or we quit
            cmd = ask("\n> ")
            low = cmd.lower()
            if low in ("", "y", "yes"):
                final = label
            elif low == "q":
                out(f"\nSaved. {len(reviews)}/{total} reviewed.")
                return reviews
            elif low == "s":
                skipped += 1
                break
            elif low == "u":
                if not history:
                    out("  nothing to undo")
                    continue
                last = history.pop()
                reviews[:] = [r for r in reviews if r["id"] != last["id"]]
                save(reviews)
                queue.appendleft(draft)  # the one we were on comes back after the undone one
                queue.appendleft(last)
                break
            elif low == "e":
                try:
                    final = parse_json_edit(editor(json.dumps(label, indent=2, ensure_ascii=False)))
                except (EditError, subprocess.CalledProcessError, OSError) as exc:
                    out(f"  ! {exc}")
                    continue
            else:
                try:
                    final = apply_edits(label, cmd)
                except EditError as exc:
                    out(f"  ! {exc}")
                    continue
                out("  changes:")
                for f in changed_fields(label, final):
                    out(f"    {f}: {label[f]!r} -> {final[f]!r}")
                if ask("  keep these? [Enter=yes / n=keep editing] ").lower() in ("n", "no"):
                    continue

            reviews.append(make_review(draft, final))
            save(reviews)
            history.append(draft)
            break

    out(f"\nDone. {len(reviews)}/{total} reviewed" + (f", {skipped} skipped." if skipped else "."))
    return reviews


# ---------------------------------------------------------------- status / finalize


def correction_stats(reviews: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(reviews)
    per_field = Counter(f for r in reviews for f in r["changed_fields"])
    return {
        "reviewed": n,
        "accepted": sum(r["status"] == "accepted" for r in reviews),
        "corrected": sum(r["status"] == "corrected" for r in reviews),
        "fields_changed": {f: per_field[f] for f in FIELDS if per_field[f]},
    }


def format_stats(stats: dict[str, Any], total: int) -> str:
    n = stats["reviewed"]
    lines = [
        f"Reviewed {n}/{total}: {stats['accepted']} accepted as-is, "
        f"{stats['corrected']} corrected."
    ]
    if n:
        lines.append("Draft fields needing a fix (count / % of reviewed):")
        for field in FIELDS:
            c = stats["fields_changed"].get(field, 0)
            lines.append(f"  {field:<22} {c:>4}  {100 * c / n:5.1f}%")
        lines.append(
            "Note: reviewing a draft anchors you toward accepting it, so these rates flatter "
            "Gemini."
        )
    return "\n".join(lines)


def build_test_set(
    drafts: list[dict[str, Any]], reviews: list[dict[str, Any]], *, allow_partial: bool
) -> tuple[list[dict[str, Any]], list[str]]:
    """Join reviews to posting text. Returns (records, ids still unreviewed)."""
    by_id = {r["id"]: r for r in reviews}
    missing = [d["id"] for d in drafts if d["id"] not in by_id]
    if missing and not allow_partial:
        return [], missing
    records = [
        {
            "id": d["id"],
            "text": d["text"],
            "label": by_id[d["id"]]["label"],
            "review_status": by_id[d["id"]]["status"],
        }
        for d in sorted(drafts, key=lambda d: d["id"])
        if d["id"] in by_id
    ]
    return records, missing


def leakage(test_records: list[dict[str, Any]], all_labels: list[dict[str, Any]]) -> list[str]:
    """Ids of train postings whose text fingerprint also appears in the test set."""
    test_prints = {fingerprint(r["text"]) for r in test_records}
    return [
        r["id"]
        for r in all_labels
        if r["split"] != "test" and fingerprint(r["text"]) in test_prints
    ]


# ---------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("command", choices=["review", "status", "finalize"], nargs="?", default="review")
    p.add_argument("--labels", type=Path, default=DEFAULT_LABELS)
    p.add_argument("--reviews", type=Path, default=DEFAULT_REVIEWS)
    p.add_argument("--out", type=Path, default=DEFAULT_TEST_SET, help="finalize: output path")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="finalize with only the postings reviewed so far",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    drafts = test_drafts(args.labels)
    if not drafts:
        print(
            f"No test-split labels in {args.labels}. Run data/build_dataset.py then "
            "data/label_with_gemini.py first.",
            file=sys.stderr,
        )
        return 2
    reviews = read_jsonl(args.reviews)
    known = {d["id"] for d in drafts}
    stray = [r["id"] for r in reviews if r["id"] not in known]
    if stray:
        print(
            f"{len(stray)} reviews in {args.reviews} match no test draft (e.g. {stray[0]}).",
            file=sys.stderr,
        )
        return 2

    if args.command == "review":
        review_loop(drafts, reviews, lambda rs: write_jsonl(args.reviews, rs))
        return 0

    print(format_stats(correction_stats(reviews), len(drafts)))
    if args.command == "status":
        return 0

    records, missing = build_test_set(drafts, reviews, allow_partial=args.allow_partial)
    if missing and not args.allow_partial:
        print(
            f"\n{len(missing)} test postings are not reviewed yet — finish `review`, or pass "
            "--allow-partial to finalize what exists.",
            file=sys.stderr,
        )
        return 1
    leaked = leakage(records, read_jsonl(args.labels))
    if leaked:
        print(
            f"\nLEAKAGE: {len(leaked)} train postings duplicate a test posting "
            f"(e.g. {leaked[0]}).",
            file=sys.stderr,
        )
        return 1
    write_jsonl(args.out, records)
    print(f"\nWrote {len(records)} hand-checked postings to {args.out}")
    print("Leakage check: no train posting shares text with the test set.")
    if missing:
        print(f"WARNING: partial test set — {len(missing)} postings left out.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
