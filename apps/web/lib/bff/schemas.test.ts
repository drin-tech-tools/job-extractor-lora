import assert from "node:assert/strict";
import { test } from "node:test";

import { MAX_POSTING_CHARS, compareSchema, extractSchema } from "./schemas";

const ok = "x".repeat(60);

test("extract defaults to the fine-tuned model", () => {
  assert.equal(extractSchema.parse({ text: ok }).model, "finetuned");
});

test("rejects too-short, too-long and unknown-model requests", () => {
  assert.equal(extractSchema.safeParse({ text: "short" }).success, false);
  assert.equal(extractSchema.safeParse({ text: "x".repeat(MAX_POSTING_CHARS + 1) }).success, false);
  assert.equal(extractSchema.safeParse({ text: ok, model: "gpt" }).success, false);
  assert.equal(compareSchema.safeParse({ text: ok, models: [] }).success, false);
});

test("compare defaults to all three models", () => {
  assert.deepEqual(compareSchema.parse({ text: ok }).models, ["finetuned", "base", "gemini"]);
});
