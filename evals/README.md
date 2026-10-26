# Evals

Scores **base vs fine-tuned vs Gemini** on the hand-checked test set and writes the table the
[Results page](../apps/web/app/results/page.tsx) renders.

```bash
cd evals
uv sync --group dev
cp .env.example .env                    # only what you want to *generate* with
uv run python eval.py                   # all three models → results/latest.{json,md}
uv run python eval.py --models base gemini            # the baseline stage, before any training
uv run python eval.py --params finetuned='1.5B + LoRA' --write-web   # publish to the Results page
uv run pytest -q
```

Inputs: `training/data/test_set/test.jsonl`, produced by
`training/data/review_test_set.py finalize` — hand-corrected labels, **never** Gemini's drafts.

## How a model is run

Models are called through the same extractors the API serves with (`apps/api/app/services`), so
prompts and output parsing cannot differ between evaluation and production. The base model gets the
full rules + schema prompt; the fine-tuned model gets the short prompt it was trained on.

Every answer is cached per model in `predictions/<model>.jsonl` (git-ignored). Re-running only asks
for postings that have no prediction yet, so a metrics change or an interrupted run costs nothing.
`--regenerate` throws a model's cache away — do that after retraining or re-pointing an endpoint.

**Predictions made elsewhere** (e.g. the Colab notebook running the model locally on a T4) can be
imported: a JSONL of `{"id", "raw", "latency_ms"?, "model_id"?}` per line.

```bash
uv run python eval.py --models finetuned --predictions finetuned=colab_preds.jsonl --no-generate
```

## What counts against a model

| Outcome | Scored as |
|---|---|
| Valid JSON matching the schema | field by field, below |
| Garbage, truncated JSON, wrong enum or type (`invalid_json`, `schema_mismatch`) | **0 on every metric** — the model's fault |
| Timeout, HTTP 5xx, rate limit | **not counted**: not cached, retried on the next run |

Invalid output scores 0 everywhere rather than as an "all-null answer": most postings have no
salary, so a model that always failed would otherwise collect credit for "correctly" finding none.

Infrastructure failures say nothing about the model, so they don't hurt it — but while any remain
the run is `incomplete`, exits non-zero, and **refuses to publish** (`--allow-incomplete` overrides,
counting the gaps as invalid and saying so in the note). `--limit` runs and mock runs
(`EXTRACTOR_MOCK`) can never be published.

## Metrics

| Metric | Definition |
|---|---|
| `valid_json` | Output parsed **and** matched the `JobExtraction` schema. |
| `title` | Case/whitespace-insensitive exact match. |
| `seniority`, `work_mode` | Exact match on the enum value. |
| `years` | Both `years_experience_min` and `years_experience_max` match. |
| `salary` | `salary_min`, `salary_max`, `salary_currency` and `salary_period` **all** match. "No salary" is right only if the model also returns null. |
| `skills_f1` | Set F1 over case-normalised `required_skills`, averaged per posting. Both empty = 1.0. |
| `overall` | Unweighted mean of the six field scores (`title` … `skills_f1`). One headline number; it hides which fields fail, so read the columns too. |
| latency | p50 / p95 wall-clock per request. For hosted endpoints this includes the network and any cold start — state the hardware when you quote it. |

**Intervals.** Pass/fail metrics get a Wilson interval (it stays honest at 0% / 100% — 150/150 is
still ≥ 97.5%, where a bootstrap would claim "certainly 100%"). F1 and `overall` get a percentile
bootstrap (2000 resamples, seeded). Comparisons between models use a **paired** bootstrap over the
same postings, and the Results page only calls a gap real when its interval excludes zero. With
~150 postings, gaps of a few points usually aren't.

## Limits of this evaluation

- The test labels start as Gemini drafts that a human corrects. Reviewers anchor toward accepting
  what they're shown, so ground truth leans toward Gemini's choices and **Gemini's score is likely
  optimistic**. `review_test_set.py status` prints how often drafts needed fixing.
- Gemini is evaluated by re-running it live (with the API's prompt), not by reusing the labeling
  pass, so its row is a real measurement — against labels it helped create.
- One annotator, no inter-annotator agreement.
