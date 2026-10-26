/**
 * Best-effort per-IP sliding-window rate limit for the public BFF.
 *
 * State lives in module memory, so on serverless it is per warm instance: it blunts casual
 * abuse of the free-tier Gemini / HF quotas but is not a hard guarantee. A shared store
 * (Upstash, Vercel KV) would be needed for that.
 */
const WINDOW_MS = 60_000;
const hits = new Map<string, number[]>();

export function getLimit(): number {
  const n = Number(process.env.RATE_LIMIT_PER_MIN);
  return Number.isFinite(n) && n > 0 ? Math.floor(n) : 10;
}

export interface RateResult {
  allowed: boolean;
  /** Seconds until the oldest hit leaves the window (only meaningful when blocked). */
  retryAfterS: number;
}

export function checkRateLimit(
  key: string,
  limit: number = getLimit(),
  now: number = Date.now(),
): RateResult {
  const recent = (hits.get(key) ?? []).filter((t) => now - t < WINDOW_MS);
  if (recent.length >= limit) {
    hits.set(key, recent);
    return { allowed: false, retryAfterS: Math.max(1, Math.ceil((recent[0] + WINDOW_MS - now) / 1000)) };
  }
  recent.push(now);
  hits.set(key, recent);
  // Keep the map from growing without bound on a long-lived instance.
  if (hits.size > 5000) {
    for (const [k, ts] of hits) if (ts.every((t) => now - t >= WINDOW_MS)) hits.delete(k);
  }
  return { allowed: true, retryAfterS: 0 };
}

export function resetRateLimitForTests(): void {
  hits.clear();
}

/** Client IP as seen behind Vercel's proxy; falls back to a shared bucket. */
export function clientIp(headers: Headers): string {
  const fwd = headers.get("x-forwarded-for");
  if (fwd) return fwd.split(",")[0].trim();
  return headers.get("x-real-ip")?.trim() || "unknown";
}
