import assert from "node:assert/strict";
import { test } from "node:test";

import type { JobExtraction } from "./api";
import { differsFromReference, displayValue } from "./compare";

const base: JobExtraction = {
  title: "Senior Backend Engineer",
  seniority: "senior",
  required_skills: ["Python", "SQL"],
  years_experience_min: 5,
  years_experience_max: null,
  work_mode: "remote",
  salary_min: 120000,
  salary_max: 150000,
  salary_currency: "USD",
  salary_period: "year",
};

test("formats years and salary", () => {
  assert.equal(displayValue(base, "years"), "5+ yrs");
  assert.equal(displayValue({ ...base, years_experience_max: 8 }, "years"), "5–8 yrs");
  assert.equal(displayValue(base, "salary"), "USD 120,000–150,000 / year");
  assert.equal(displayValue({ ...base, salary_max: 120000 }, "salary"), "USD 120,000 / year");
  assert.equal(displayValue({ ...base, salary_min: null, salary_max: null }, "salary"), "—");
});

test("skills agree regardless of order and case", () => {
  const other = { ...base, required_skills: ["sql", "PYTHON"] };
  assert.equal(differsFromReference(other, base, "skills"), false);
  assert.equal(differsFromReference({ ...base, required_skills: ["Python"] }, base, "skills"), true);
});

test("detects a differing scalar field", () => {
  assert.equal(differsFromReference({ ...base, work_mode: "hybrid" }, base, "work_mode"), true);
  assert.equal(differsFromReference({ ...base, title: "senior backend engineer" }, base, "title"), false);
});
