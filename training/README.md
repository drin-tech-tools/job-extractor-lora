# Training pipeline

Everything that happens before and around the deployed app: building the dataset, labeling it with
Gemini (distillation), fine-tuning with QLoRA on Colab's free T4, and publishing the adapter.

The schema, extraction rules and prompt format are imported from [`apps/api`](../apps/api) — the
labeler, the training notebook, the evals and the deployed API all share one definition, so they
cannot drift apart.

```bash
cd training
cp .env.example .env            # GEMINI_API_KEY
uv sync --group dev

# 1. Build the raw corpus and freeze the test split (any combination of sources)
uv run python data/build_dataset.py --hf <hf-dataset> --hf-text-column <column> --boards --target 1500

# 2. Label with Gemini (resumable; free-tier friendly)
set -a; source .env; set +a
uv run python data/label_with_gemini.py

uv run pytest -q
```

## Test-set discipline

- The test split is decided in step 1, **before any labeling**, from a seeded hash of each posting.
  Re-running with more data never moves a posting between splits.
- Postings are de-duplicated by text hash **and** by (company, title), so a reposted role can't
  land in train while its twin sits in test.
- Gemini's labels for the test split are drafts. They become ground truth only after hand review;
  `flags` on each label (salary digits missing from the text, skills not mentioned, ...) show where
  to look first.

## Data licensing

Raw postings come from third-party sources with their own terms (Himalayas and Jobicy require
attribution and restrict redistribution; check any Hugging Face dataset's license). `data/raw/` and
`data/labeled/` are git-ignored on purpose. Publish the adapter and the eval numbers — not the
postings — unless the source's license allows it.

## Still to do

- `notebooks/train_qlora.ipynb` (Colab, Unsloth or TRL/PEFT)
- `data/review_test_set.py` — export the test split for hand correction
- `../evals/eval.py`
