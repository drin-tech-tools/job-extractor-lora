import { z } from "zod";

/** Mirrors apps/api/app/models/schemas.py — keep MIN/MAX in sync. */
export const MIN_POSTING_CHARS = 40;
export const MAX_POSTING_CHARS = 6000;

export const modelKeySchema = z.enum(["finetuned", "base", "gemini"]);

const textSchema = z.string().min(MIN_POSTING_CHARS).max(MAX_POSTING_CHARS);

export const extractSchema = z.object({
  text: textSchema,
  model: modelKeySchema.default("finetuned"),
});

export const compareSchema = z.object({
  text: textSchema,
  models: z.array(modelKeySchema).min(1).max(3).default(["finetuned", "base", "gemini"]),
});
