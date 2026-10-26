import { handleExtraction } from "@/lib/bff/proxy";
import { compareSchema } from "@/lib/bff/schemas";

export const runtime = "nodejs";
export const maxDuration = 120;

/** POST /api/extract/compare → FastAPI POST /extract/compare */
export async function POST(req: Request) {
  return handleExtraction(req, "/extract/compare", compareSchema);
}
