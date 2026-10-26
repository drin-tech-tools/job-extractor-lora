import assert from "node:assert/strict";
import { beforeEach, test } from "node:test";

import { checkRateLimit, clientIp, resetRateLimitForTests } from "./rate-limit";

beforeEach(() => resetRateLimitForTests());

test("allows up to the limit then blocks with a retry hint", () => {
  const t0 = 1_000_000;
  for (let i = 0; i < 3; i++) assert.equal(checkRateLimit("a", 3, t0 + i).allowed, true);
  const blocked = checkRateLimit("a", 3, t0 + 10);
  assert.equal(blocked.allowed, false);
  assert.ok(blocked.retryAfterS >= 1 && blocked.retryAfterS <= 60);
});

test("window slides: old hits expire", () => {
  const t0 = 1_000_000;
  for (let i = 0; i < 3; i++) checkRateLimit("a", 3, t0);
  assert.equal(checkRateLimit("a", 3, t0 + 30_000).allowed, false);
  assert.equal(checkRateLimit("a", 3, t0 + 61_000).allowed, true);
});

test("keys are independent", () => {
  checkRateLimit("a", 1, 0);
  assert.equal(checkRateLimit("a", 1, 1).allowed, false);
  assert.equal(checkRateLimit("b", 1, 1).allowed, true);
});

test("clientIp prefers the first x-forwarded-for hop", () => {
  assert.equal(clientIp(new Headers({ "x-forwarded-for": "1.2.3.4, 10.0.0.1" })), "1.2.3.4");
  assert.equal(clientIp(new Headers({ "x-real-ip": "5.6.7.8" })), "5.6.7.8");
  assert.equal(clientIp(new Headers()), "unknown");
});
