import { proxyJson } from "@/lib/bff/proxy";

export const runtime = "nodejs";

/** GET /api/health → FastAPI GET /health */
export async function GET() {
  return proxyJson("/health", { method: "GET" });
}
