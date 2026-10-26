import raw from "@/data/results.json";

interface Metrics {
  valid_json: number;
  title: number;
  seniority: number;
  work_mode: number;
  years: number;
  salary: number;
  skills_f1: number;
  overall: number;
  latency_p50_ms: number;
}

interface ModelRow {
  key: string;
  label: string;
  params: string | null;
  metrics: Metrics | null;
  /** 95% interval per metric, [low, high] as fractions. Absent until evals/eval.py has run. */
  ci?: Partial<Record<keyof Metrics, [number, number]>>;
}

interface Comparison {
  a: string;
  b: string;
  metric: string;
  diff: number;
  ci: [number, number];
}

/** Shape written by evals/eval.py (`--write-web`); the committed placeholder has no scores yet. */
interface ResultsFile {
  status: string;
  test_set_size: number | null;
  note: string;
  models: ModelRow[];
  comparisons?: Comparison[];
  limits: string[];
}

const results = raw as unknown as ResultsFile;

const COLUMNS: { key: keyof Metrics; label: string; kind: "pct" | "ms" }[] = [
  { key: "valid_json", label: "Valid JSON", kind: "pct" },
  { key: "title", label: "Title", kind: "pct" },
  { key: "seniority", label: "Seniority", kind: "pct" },
  { key: "work_mode", label: "Work mode", kind: "pct" },
  { key: "years", label: "Experience", kind: "pct" },
  { key: "salary", label: "Salary", kind: "pct" },
  { key: "skills_f1", label: "Skills F1", kind: "pct" },
  { key: "overall", label: "Overall", kind: "pct" },
  { key: "latency_p50_ms", label: "Latency p50", kind: "ms" },
];

const MODEL_NAMES: Record<string, string> = {
  base: "Base model",
  finetuned: "Fine-tuned",
  gemini: "Gemini",
};

const pts = (x: number) => `${x >= 0 ? "+" : "−"}${Math.abs(x * 100).toFixed(1)}`;

/** A paired difference is only called real when its interval excludes zero. */
function verdict(c: Comparison): string {
  return c.ci[0] > 0 || c.ci[1] < 0 ? "clear difference" : "within noise";
}

function cell(value: number | undefined, kind: "pct" | "ms"): string {
  if (value == null) return "—";
  return kind === "pct" ? `${(value * 100).toFixed(1)}%` : `${Math.round(value)} ms`;
}

function interval(range: [number, number] | undefined): string | null {
  return range ? `${(range[0] * 100).toFixed(1)}–${(range[1] * 100).toFixed(1)}` : null;
}

export default function ResultsPage() {
  const rows = results.models;
  const pending = results.status !== "final";
  const comparisons = results.comparisons ?? [];

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
          trained on. Fields are scored independently; skills use set F1; an invalid
          output scores 0 on every column.
        </p>
        {pending ? (
          <p
            className="mt-3 inline-block rounded-md border border-[var(--warn)]/40 bg-[var(--warn)]/10 px-2.5 py-1 text-xs text-[var(--warn)]"
            data-testid="results-pending"
          >
            {results.note}
          </p>
        ) : (
          <p className="mt-3 text-xs text-[var(--muted)]" data-testid="results-note">
            {results.note}
          </p>
        )}
      </div>

      <div className="glass overflow-x-auto rounded-xl">
        <table className="w-full min-w-[860px] text-left text-sm" data-testid="results-table">
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
                {COLUMNS.map((c) => {
                  const range = c.kind === "pct" ? interval(row.ci?.[c.key]) : null;
                  return (
                    <td key={c.key} className="px-3 py-3 text-right font-mono tabular-nums">
                      <div className={c.key === "overall" ? "font-semibold" : undefined}>
                        {cell(row.metrics?.[c.key], c.kind)}
                      </div>
                      {range && (
                        <div className="text-[10px] text-[var(--muted)]" title="95% interval">
                          {range}
                        </div>
                      )}
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {comparisons.length > 0 && (
        <section data-testid="comparisons">
          <h2 className="text-lg font-semibold" style={{ fontFamily: "var(--font-display)" }}>
            Is the gap real?
          </h2>
          <p className="mt-1 text-sm text-[var(--muted)]">
            Paired difference in overall score on the same postings, with a 95% interval. A gap
            counts only if the interval excludes zero.
          </p>
          <ul className="mt-3 space-y-1.5 text-sm">
            {comparisons.map((c) => (
              <li key={`${c.a}-${c.b}`} data-testid={`comparison-${c.a}-${c.b}`}>
                <span className="font-medium">
                  {MODEL_NAMES[c.a] ?? c.a} vs {MODEL_NAMES[c.b] ?? c.b}:
                </span>{" "}
                <span className="font-mono">
                  {pts(c.diff)} pts ({pts(c.ci[0])} to {pts(c.ci[1])})
                </span>{" "}
                <span className="text-[var(--muted)]">— {verdict(c)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}

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
