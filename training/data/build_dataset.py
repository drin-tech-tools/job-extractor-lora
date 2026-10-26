#!/usr/bin/env python3
"""Build the raw job-posting corpus (no labels yet) and fix the test split.

Sources (combine freely):
  --hf DATASET        any Hugging Face dataset with a text column of postings
  --boards            live Himalayas + Jobicy postings, same endpoints as job-hunt-agent
  --jsonl PATH        a local JSONL file (e.g. an export from job-hunt-agent's cache)

Output: data/raw/postings.jsonl, one record per line:
  {"id", "source", "url", "meta": {"title", "company"}, "text", "split": "train" | "test"}

The test split is decided HERE, before any labeling, and never changes: re-running with
more data keeps every existing assignment. That is what keeps the final test set clean —
a record that was ever "test" is excluded from training no matter how the data grows.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import random
import re
import sys
from collections import Counter
from collections.abc import Iterable, Iterator
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import httpx

TRAINING = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRAINING.parent / "apps" / "api"))

from app.models.schemas import MAX_POSTING_CHARS  # noqa: E402

logger = logging.getLogger(__name__)

DEFAULT_OUT = TRAINING / "data" / "raw" / "postings.jsonl"

MIN_CHARS = 300  # shorter than this is a stub, not a posting
MAX_CHARS = MAX_POSTING_CHARS  # the API rejects longer input, so training never sees it either

HIMALAYAS_URL = "https://himalayas.app/jobs/api"
JOBICY_URL = "https://jobicy.com/api/v2/remote-jobs"
JOBICY_TAGS = ["python", "javascript", "data", "devops", "design", "marketing", "sales", "support"]

_BLOCK_TAGS = {"p", "div", "br", "li", "ul", "ol", "h1", "h2", "h3", "h4", "h5", "h6", "tr"}
_HTML_RE = re.compile(r"</?[a-z][^>]*>", re.I)


class _TextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "li":
            self.parts.append("\n- ")
        elif tag in _BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        # <li> already starts its own line; adding one here would double-space every bullet.
        if tag in _BLOCK_TAGS and tag != "li":
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def clean_text(raw: str) -> str:
    """HTML (if any) -> plain text with paragraph breaks kept, whitespace normalised, capped."""
    text = raw
    if _HTML_RE.search(text):
        parser = _TextExtractor()
        parser.feed(text)
        text = "".join(parser.parts)
    text = html.unescape(text).replace("\xa0", " ")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    text = re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()
    if len(text) > MAX_CHARS:
        cut = text.rfind("\n", 0, MAX_CHARS)
        text = text[: cut if cut > MAX_CHARS // 2 else MAX_CHARS].rstrip()
    return text


def fingerprint(text: str) -> str:
    """Hash of the alphanumeric content, so whitespace/case/punctuation edits still collide."""
    squashed = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    return hashlib.sha1(squashed.encode()).hexdigest()  # noqa: S324 — non-crypto id


def _role_key(meta: dict[str, str]) -> str | None:
    """Reposts of the same role (same company + title) must not straddle train and test."""
    company = re.sub(r"\W+", "", (meta.get("company") or "").lower())
    title = re.sub(r"\W+", "", (meta.get("title") or "").lower())
    return f"{company}|{title}" if company and title else None


def make_record(
    source: str, raw_text: str, *, url: str | None = None, title: str = "", company: str = ""
) -> dict[str, Any] | None:
    text = clean_text(raw_text)
    if len(text) < MIN_CHARS:
        return None
    return {
        "id": fingerprint(text)[:16],
        "source": source,
        "url": url,
        "meta": {"title": title.strip(), "company": company.strip()},
        "text": text,
    }


# ---------------------------------------------------------------- sources


def iter_hf(
    dataset: str, *, split: str, text_column: str, title_column: str | None, limit: int, seed: int
) -> Iterator[dict[str, Any]]:
    """Stream a Hugging Face dataset; shuffled so we don't just take its first (sorted) rows."""
    from datasets import load_dataset  # heavy import — only when this source is used

    ds = load_dataset(dataset, split=split, streaming=True).shuffle(seed=seed, buffer_size=10_000)
    for row in ds.take(limit):
        raw = row.get(text_column)
        if not isinstance(raw, str):
            continue
        title = str(row.get(title_column) or "") if title_column else ""
        rec = make_record(f"hf:{dataset}", raw, title=title)
        if rec:
            yield rec


def iter_jsonl(path: Path, *, text_field: str) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            raw = row.get(text_field)
            if not isinstance(raw, str):
                continue
            rec = make_record(
                row.get("source") or f"jsonl:{path.name}",
                raw,
                url=row.get("url") or row.get("apply_url"),
                title=str(row.get("title") or ""),
                company=str(row.get("company") or ""),
            )
            if rec:
                yield rec


def iter_boards(*, pages: int, timeout_s: float = 20.0) -> Iterator[dict[str, Any]]:
    """Himalayas (paged) + Jobicy (per tag). Failures are logged and skipped, like project 2."""
    with httpx.Client(timeout=timeout_s, follow_redirects=True) as client:
        for page in range(pages):
            try:
                resp = client.get(HIMALAYAS_URL, params={"limit": 20, "offset": page * 20})
                resp.raise_for_status()
                jobs = resp.json().get("jobs") or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Himalayas page %d failed: %s", page, exc)
                break
            if not jobs:
                break
            for item in jobs:
                rec = make_record(
                    "himalayas",
                    str(item.get("description") or item.get("excerpt") or ""),
                    url=item.get("applicationLink") or item.get("guid"),
                    title=str(item.get("title") or ""),
                    company=str(item.get("companyName") or ""),
                )
                if rec:
                    yield rec

        for tag in JOBICY_TAGS:
            try:
                resp = client.get(JOBICY_URL, params={"count": 50, "tag": tag})
                resp.raise_for_status()
                jobs = resp.json().get("jobs") or []
            except Exception as exc:  # noqa: BLE001
                logger.warning("Jobicy tag %r failed: %s", tag, exc)
                continue
            for item in jobs:
                rec = make_record(
                    "jobicy",
                    str(item.get("jobDescription") or item.get("description") or ""),
                    url=item.get("url"),
                    title=str(item.get("jobTitle") or ""),
                    company=str(item.get("companyName") or ""),
                )
                if rec:
                    yield rec


# ---------------------------------------------------------------- assembly


def load_existing(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def dedupe(
    existing: list[dict[str, Any]], candidates: Iterable[dict[str, Any]]
) -> tuple[list[dict[str, Any]], Counter]:
    """Keep candidates not already seen by text hash, by id, or by (company, title)."""
    seen_ids = {r["id"] for r in existing}
    seen_roles = {k for r in existing if (k := _role_key(r["meta"]))}
    kept: list[dict[str, Any]] = []
    dropped: Counter = Counter()
    for rec in candidates:
        role = _role_key(rec["meta"])
        if rec["id"] in seen_ids:
            dropped["duplicate_text"] += 1
        elif role and role in seen_roles:
            dropped["duplicate_role"] += 1
        else:
            seen_ids.add(rec["id"])
            if role:
                seen_roles.add(role)
            kept.append(rec)
    return kept, dropped


def assign_splits(
    existing: list[dict[str, Any]], new: list[dict[str, Any]], *, test_size: int, seed: int
) -> list[dict[str, Any]]:
    """Existing records keep their split forever; new ones fill the test quota first.

    New records are ordered by a seeded hash of their id (not input order), so the choice
    does not depend on which source or page they happened to arrive from.
    """
    n_test = sum(1 for r in existing if r["split"] == "test")
    ordered = sorted(new, key=lambda r: hashlib.sha1(f"{seed}:{r['id']}".encode()).hexdigest())  # noqa: S324
    for rec in ordered:
        if n_test < test_size:
            rec["split"] = "test"
            n_test += 1
        else:
            rec["split"] = "train"
    rng = random.Random(seed)
    merged = [*existing, *new]
    rng.shuffle(merged)
    return merged


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp.replace(path)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--hf", action="append", default=[], metavar="DATASET")
    p.add_argument("--hf-split", default="train")
    p.add_argument("--hf-text-column", default="description")
    p.add_argument("--hf-title-column", default=None)
    p.add_argument("--hf-limit", type=int, default=1500, help="max rows to read per HF dataset")
    p.add_argument("--boards", action="store_true", help="fetch live Himalayas + Jobicy postings")
    p.add_argument("--board-pages", type=int, default=10, help="Himalayas pages of 20")
    p.add_argument("--jsonl", action="append", default=[], type=Path, metavar="PATH")
    p.add_argument("--jsonl-text-field", default="description")
    p.add_argument("--target", type=int, default=1500, help="stop once the corpus has this many")
    p.add_argument("--test-size", type=int, default=150)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args(argv)
    if not (args.hf or args.boards or args.jsonl):
        p.error("pick at least one source: --hf, --boards or --jsonl")
    return args


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)

    existing = load_existing(args.out)
    logger.info("Existing corpus: %d records (%s)", len(existing), args.out)

    streams: list[Iterator[dict[str, Any]]] = []
    for name in args.hf:
        streams.append(
            iter_hf(
                name,
                split=args.hf_split,
                text_column=args.hf_text_column,
                title_column=args.hf_title_column,
                limit=args.hf_limit,
                seed=args.seed,
            )
        )
    if args.boards:
        streams.append(iter_boards(pages=args.board_pages))
    for path in args.jsonl:
        streams.append(iter_jsonl(path, text_field=args.jsonl_text_field))

    room = max(args.target - len(existing), 0)
    candidates: list[dict[str, Any]] = []
    for stream in streams:
        for rec in stream:
            candidates.append(rec)
            if len(candidates) >= room * 2:  # over-read; dedupe will drop some
                break

    new, dropped = dedupe(existing, candidates)
    new = new[:room]
    merged = assign_splits(existing, new, test_size=args.test_size, seed=args.seed)
    write_jsonl(args.out, merged)

    by_split = Counter(r["split"] for r in merged)
    by_source = Counter(r["source"] for r in merged)
    print(f"\nWrote {len(merged)} postings to {args.out}")
    print(f"  added {len(new)} new; dropped {dict(dropped) or 'none'}")
    print(f"  splits: {dict(by_split)}")
    print(f"  sources: {dict(by_source)}")
    if by_split["test"] < args.test_size:
        print(f"  WARNING: only {by_split['test']}/{args.test_size} test postings so far")
    return 0


if __name__ == "__main__":
    sys.exit(main())
