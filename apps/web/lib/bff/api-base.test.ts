import assert from "node:assert/strict";
import { test } from "node:test";

import { buildApiUrl } from "./api-base";

test("keeps the base path prefix when joining an absolute path", () => {
  assert.equal(
    buildApiUrl("/extract", "https://h.vercel.app/_/api").href,
    "https://h.vercel.app/_/api/extract",
  );
  assert.equal(
    buildApiUrl("/extract", "https://h.vercel.app/_/api/").href,
    "https://h.vercel.app/_/api/extract",
  );
});

test("works with a bare host", () => {
  assert.equal(
    buildApiUrl("/extract/compare", "http://127.0.0.1:8000").href,
    "http://127.0.0.1:8000/extract/compare",
  );
});
