# Job Extractor LoRA — Web

Next.js App Router UI **and** the BFF. The browser only talks to this service (`/api/*`); FastAPI stays internal behind the `API_URL` service binding. See the root [README](../../README.md).

```bash
cp .env.example .env.local
npm install
npm run dev          # needs the API on :8000 (EXTRACTOR_MOCK=1 works with no keys)
npm run typecheck && npm test && npm run build
```

BFF routes (`app/api/`) validate with zod, rate-limit per IP, then forward to the allowlisted FastAPI path.
