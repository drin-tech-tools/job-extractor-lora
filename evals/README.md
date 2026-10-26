# Evals

`eval.py` (not written yet) will run the held-out test set through each model and write
[`apps/web/data/results.json`](../apps/web/data/results.json), which the Results page renders.

## Contract

Every model is scored through the same path: raw text → `parse_model_output` → compare with the
hand-checked label. The test set is `split == "test"` from `training/data/raw/postings.jsonl`,
with labels corrected by hand — never the Gemini draft.

| Metric | Definition |
|---|---|
| `valid_json` | Output parsed **and** matched the `JobExtraction` schema (share of test postings). Invalid outputs score as the all-null answer on every other metric, so a model can't hide failures. |
| `title` | Case/whitespace-insensitive exact match. |
| `seniority`, `work_mode` | Exact match on the enum value. |
| `years` | Both `years_experience_min` and `years_experience_max` match. |
| `salary` | `salary_min`, `salary_max`, `salary_currency` and `salary_period` all match (a posting with no salary is correct only if the model also returns null). |
| `skills_f1` | Set F1 over lower-cased `required_skills`, averaged per posting. |
| `latency_p50_ms` | Median wall-clock per request on the stated hardware. |

Report a confidence interval alongside each rate: with ~150 test postings, differences of a few
points are noise.
