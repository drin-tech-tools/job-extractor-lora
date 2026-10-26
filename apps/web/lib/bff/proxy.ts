import { NextResponse } from "next/server";
import { ZodError, type ZodTypeAny } from "zod";

import { buildApiUrl, internalApiHeaders } from "./api-base";
import { checkRateLimit, clientIp } from "./rate-limit";

/** Proxy a JSON request to an allowlisted FastAPI path, adding the internal token. */
export async function proxyJson(path: string, init: RequestInit = {}): Promise<Response> {
  const headers = new Headers(init.headers);
  if (!headers.has("Content-Type") && init.body) {
    headers.set("Content-Type", "application/json");
  }
  for (const [k, v] of Object.entries(internalApiHeaders())) {
    headers.set(k, v);
  }

  let upstream: Response;
  try {
    upstream = await fetch(buildApiUrl(path), {
      ...init,
      headers,
      // Do not forward browser cookies to the internal service.
      cache: "no-store",
    });
  } catch (err) {
    const message = err instanceof Error ? err.message : "Upstream unreachable";
    return NextResponse.json({ detail: message }, { status: 502 });
  }

  const body = await upstream.arrayBuffer();
  return new NextResponse(body, {
    status: upstream.status,
    headers: { "Content-Type": upstream.headers.get("Content-Type") || "application/json" },
  });
}

export function zodErrorResponse(err: ZodError): NextResponse {
  return NextResponse.json({ detail: "Invalid request", issues: err.flatten() }, { status: 400 });
}

/**
 * Shared handler for the extraction routes: rate-limit by IP, validate the body with zod,
 * then forward the parsed (not raw) body to FastAPI.
 */
export async function handleExtraction(
  req: Request,
  path: string,
  schema: ZodTypeAny,
): Promise<Response> {
  const rate = checkRateLimit(clientIp(req.headers));
  if (!rate.allowed) {
    return NextResponse.json(
      { detail: `Rate limit reached. Try again in ${rate.retryAfterS}s.` },
      { status: 429, headers: { "Retry-After": String(rate.retryAfterS) } },
    );
  }

  let body: unknown;
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ detail: "Invalid JSON body" }, { status: 400 });
  }

  let parsed: unknown;
  try {
    parsed = schema.parse(body);
  } catch (err) {
    if (err instanceof ZodError) return zodErrorResponse(err);
    return NextResponse.json({ detail: "Invalid request" }, { status: 400 });
  }

  return proxyJson(path, { method: "POST", body: JSON.stringify(parsed), signal: req.signal });
}
