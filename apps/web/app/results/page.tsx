import results from "@/data/results.json";

interface Metrics {
  valid_json: number;
  title: number;
  seniority: number;
  work_mode: number;
  years: number;
  salary: number;
  skills_f1: number;
  latency_p50_ms: number;
}

interface ModelRow {
  key: string;
  label: string;
  params: string | null;
  metrics: Metrics | null;
}

const COLUMNS: { key: keyof Metrics; label: string; kind: "pct" | "ms" }[] = [
  { key: "valid_json", label: "Valid JSON", kind: "pct" },
  { key: "title", label: "Title", kind: "pct" },
  { key: "seniority", label: "Seniority", kind: "pct" },
  { key: "work_mode", label: "Work mode", kind: "pct" },
  { key: "years", label: "Experience", kind: "pct" },
  { key: "salary", label: "Salary", kind: "pct" },
  { key: "skills_f1", label: "Skills F1", kind: "pct" },
  { key: "latency_p50_ms", label: "Latency p50", kind: "ms" },
];

function cell(value: number | undefined, kind: "pct" | "ms"): string {
  if (value == null) return "—";
  return kind === "pct" ? `${(value * 100).toFixed(1)}%` : `${Math.round(value)} ms`;
}

export default function ResultsPage() {
  const rows = results.models as ModelRow[];
  const pending = results.status !== "final";

  return (
    <div className="animate-fade-up space-y-8">
      <div>
        <h1
          className="text-3xl font-semibold tracking-tight"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Results
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">
          Every model is scored on the same held-out test set, which the fine-tuned model never
          trained on. Fields are scored independently; skills use set F1.
          {results.test_set_size ? ` Test set: ${results.test_set_size} hand-checked postings.` : ""}
        </p>
        {pending && (
          <p
            className="mt-3 inline-block rounded-md border border-[var(--warn)]/40 bg-[var(--warn)]/10 px-2.5 py-1 text-xs text-[var(--warn)]"
            data-testid="results-pending"
          >
            {results.note}
          </p>
        )}
      </div>

      <div className="glass overflow-x-auto rounded-xl">
        <table className="w-full min-w-[720px] text-left text-sm" data-testid="results-table">
          <thead>
            <tr className="border-b border-[var(--line)] text-xs text-[var(--muted)]">
              <th className="px-4 py-3 font-medium">Model</th>
              {COLUMNS.map((c) => (
                <th key={c.key} className="px-3 py-3 text-right font-medium">
                  {c.label}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {rows.map((row) => (
              <tr
                key={row.key}
                className="border-b border-[var(--line)] last:border-0"
                data-testid={`results-row-${row.key}`}
              >
                <td className="px-4 py-3">
                  <div className="font-medium">{row.label}</div>
                  {row.params && <div className="text-xs text-[var(--muted)]">{row.params}</div>}
                </td>
                {COLUMNS.map((c) => (
                  <td key={c.key} className="px-3 py-3 text-right font-mono tabular-nums">
                    {cell(row.metrics?.[c.key], c.kind)}
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <section>
        <h2 className="text-lg font-semibold" style={{ fontFamily: "var(--font-display)" }}>
          Known limits
        </h2>
        <ul className="mt-3 list-disc space-y-2 pl-5 text-sm text-[var(--muted)]" data-testid="limits">
          {results.limits.map((l) => (
            <li key={l}>{l}</li>
          ))}
        </ul>
      </section>
    </div>
  );
}
