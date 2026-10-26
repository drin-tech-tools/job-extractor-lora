import { handleExtraction } from "@/lib/bff/proxy";
import { extractSchema } from "@/lib/bff/schemas";

export const runtime = "nodejs";
/** A cold HF endpoint can take a minute to wake. */
export const maxDuration = 120;

/** POST /api/extract → FastAPI POST /extract */
export async function POST(req: Request) {
  return handleExtraction(req, "/extract", extractSchema);
}
