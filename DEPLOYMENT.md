# Deployment (single Vercel project — Services)

Deploy this monorepo as **one Vercel project** using [Vercel Services](https://vercel.com/docs/services) (Beta): a public Next.js `web` service and an **internal** FastAPI `api` service ([`vercel.json`](vercel.json)). The browser only talks to `web`; `web` calls `api` over a private service binding (`API_URL`, injected at runtime — do not set it by hand).

> **Hobby requirement:** as with job-hunt-agent, the GitHub repository must be **public** for org/team Hobby deploys from a connected repo.

There is no database and no login: the demo is public, so the BFF validates input with zod and rate-limits per IP.

## 1. Serve the models

The API calls the fine-tuned and base models over any **OpenAI-compatible `/chat/completions`** endpoint — a Hugging Face Inference Endpoint, a Space running vLLM or llama.cpp, etc. Merge the LoRA adapter into the base weights first, or use a server that supports LoRA adapters. A ~1.5B model is too large for a Vercel function, which is why it is served separately.

Dedicated endpoints can scale to zero; the first request after idle can take a minute, which is why the routes allow `maxDuration: 120` and the UI warns about cold starts.

## 2. Gemini (free tier)

Create a key at <https://aistudio.google.com/apikey>. Used for the "upper reference" column and (offline) for labeling.

## 3. Environment variables

Set on the **api** service:

| Variable | Purpose |
|---|---|
| `GEMINI_API_KEY` | Gemini reference model |
| `FINETUNED_API_URL`, `FINETUNED_MODEL` | Endpoint (including `/v1`) and model name for the fine-tuned model |
| `BASE_API_URL`, `BASE_MODEL` | Same for the un-tuned base model |
| `HF_TOKEN` | Bearer token for the endpoints above |
| `INTERNAL_API_TOKEN` | Optional shared secret; if set, also set it on **web** |
| `GEMINI_MODEL` | Optional: pin a Gemini model (otherwise an ordered fallback is used) |
| `EXTRACTOR_MOCK` | `1` serves every model key from a regex stand-in. **Never set in a real deployment** — the UI shows a demo-mode banner when it is on. |

On the **web** service: `INTERNAL_API_TOKEN` (optional) and `RATE_LIMIT_PER_MIN` (default 10). `API_URL` comes from the binding.

A model whose variables are unset shows as "(offline)" in the playground and as an error row in comparisons; it doesn't break the others.

## 4. Deploy

Import the repo in Vercel; it reads `vercel.json`. Verify `GET /api/health` on the deployed URL shows the backends you configured as `true`.

## Rate-limit caveat

The limiter is in-memory per warm serverless instance. It blunts casual abuse of the free-tier quotas but is not a hard guarantee; use a shared store (Upstash/Vercel KV) if the demo is shared widely.
