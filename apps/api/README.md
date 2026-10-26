# Job Extractor LoRA — API

Internal FastAPI service (Vercel Services). Not publicly routed — called only from the Next.js BFF via the `API_URL` binding. See the root [README](../../README.md) and [DEPLOYMENT.md](../../DEPLOYMENT.md).

```bash
cp .env.example .env
uv sync --group dev
EXTRACTOR_MOCK=1 uv run uvicorn app.main:app --reload --port 8000
uv run pytest -q
```

Endpoints: `GET /health`, `POST /extract` (one model), `POST /extract/compare` (several models, run concurrently).

Optional: set `INTERNAL_API_TOKEN` to require `X-Internal-Api-Token` on every request.
