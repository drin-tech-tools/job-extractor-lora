/**
 * Browser client for the same-origin Next.js BFF under `/api/*`.
 *
 * FastAPI is internal-only (Vercel service binding). The UI never talks to it directly.
 */

export const BFF_PREFIX = "/api";

export type ModelKey = "finetuned" | "base" | "gemini";
export const MODEL_KEYS: ModelKey[] = ["finetuned", "base", "gemini"];

export interface JobExtraction {
  title: string;
  seniority: "intern" | "junior" | "mid" | "senior" | "lead" | "executive" | "unspecified";
  required_skills: string[];
  years_experience_min: number | null;
  years_experience_max: number | null;
  work_mode: "remote" | "hybrid" | "onsite" | "unspecified";
  salary_min: number | null;
  salary_max: number | null;
  salary_currency: string | null;
  salary_period: "year" | "month" | "week" | "day" | "hour" | null;
}

export interface ExtractResult {
  model: ModelKey;
  model_id: string;
  extraction: JobExtraction | null;
  valid_json: boolean;
  latency_ms: number;
  error: string | null;
  raw: string | null;
}

export interface Health {
  status: string;
  version: string;
  mock: boolean;
  backends: Record<ModelKey, boolean>;
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
  ) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${BFF_PREFIX}${path}`, { cache: "no-store", ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(detail || `Request failed (${res.status})`, res.status);
  }
  return (await res.json()) as T;
}

export function getHealth(signal?: AbortSignal): Promise<Health> {
  return request<Health>("/health", { signal });
}

export async function compareModels(
  text: string,
  models: ModelKey[],
  signal?: AbortSignal,
): Promise<ExtractResult[]> {
  const body = await request<{ results: ExtractResult[] }>("/extract/compare", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, models }),
    signal,
  });
  return body.results;
}
