/**
 * Resolve the internal FastAPI base URL.
 *
 * - On Vercel (and `vercel dev`), the `web` → `api` service binding injects `API_URL` at
 *   function runtime. Do not set `API_URL` manually in Vercel project env.
 * - Bindings are unavailable during builds and in middleware — only Route Handlers.
 * - Outside Vercel (plain `next dev`), fall back to localhost in development only.
 */
export function getApiUrl(): string {
  const bound = process.env.API_URL?.replace(/\/$/, "");
  if (bound) return bound;

  if (process.env.NODE_ENV !== "production") {
    return "http://127.0.0.1:8000";
  }

  throw new Error(
    "API_URL is not set. In production it must come from the Vercel service binding.",
  );
}

/**
 * Join an API path onto the internal base URL, preserving any path prefix on the base.
 *
 * `new URL("/extract", "https://host/_/api")` discards `/_/api` and hits the *web* service's
 * own `/extract` page instead of the API.
 */
export function buildApiUrl(path: string, base: string = getApiUrl()): URL {
  return new URL(path.replace(/^\/+/, ""), base.endsWith("/") ? base : `${base}/`);
}

/** Optional shared secret for defense-in-depth against the internal API. */
export function internalApiHeaders(): HeadersInit {
  const token = process.env.INTERNAL_API_TOKEN;
  if (!token) return {};
  return { "X-Internal-Api-Token": token };
}
