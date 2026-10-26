import type { JobExtraction } from "./api";

/** The rows the UI shows. `years`/`salary`/`skills` each fold several schema fields. */
export const FIELD_UNITS = [
  { key: "title", label: "Title" },
  { key: "seniority", label: "Seniority" },
  { key: "work_mode", label: "Work mode" },
  { key: "years", label: "Experience" },
  { key: "salary", label: "Salary" },
  { key: "skills", label: "Required skills" },
] as const;

export type FieldUnit = (typeof FIELD_UNITS)[number]["key"];

const num = (n: number) => n.toLocaleString("en-US", { maximumFractionDigits: 2 });

export function displayValue(e: JobExtraction, unit: FieldUnit): string {
  switch (unit) {
    case "title":
      return e.title || "—";
    case "seniority":
      return e.seniority;
    case "work_mode":
      return e.work_mode;
    case "years": {
      const { years_experience_min: lo, years_experience_max: hi } = e;
      if (lo == null && hi == null) return "—";
      if (lo != null && hi != null) return `${lo}–${hi} yrs`;
      return lo != null ? `${lo}+ yrs` : `up to ${hi} yrs`;
    }
    case "salary": {
      const { salary_min: lo, salary_max: hi, salary_currency: cur, salary_period: per } = e;
      if (lo == null && hi == null) return "—";
      const range = lo != null && hi != null && lo !== hi ? `${num(lo)}–${num(hi)}` : num((lo ?? hi)!);
      return [cur, range, per ? `/ ${per}` : ""].filter(Boolean).join(" ");
    }
    case "skills":
      return e.required_skills.length ? e.required_skills.join(", ") : "—";
  }
}

/** Order- and case-insensitive form used to decide whether two models agree on a field. */
export function normalizedValue(e: JobExtraction, unit: FieldUnit): string {
  if (unit === "skills") {
    return [...e.required_skills].map((s) => s.trim().toLowerCase()).sort().join("|");
  }
  return displayValue(e, unit).trim().toLowerCase();
}

export function differsFromReference(
  e: JobExtraction,
  reference: JobExtraction,
  unit: FieldUnit,
): boolean {
  return normalizedValue(e, unit) !== normalizedValue(reference, unit);
}
