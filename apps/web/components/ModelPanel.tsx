import type { ExtractResult, JobExtraction, ModelKey } from "@/lib/api";
import { FIELD_UNITS, differsFromReference, displayValue } from "@/lib/compare";

export const MODEL_LABELS: Record<ModelKey, string> = {
  finetuned: "Fine-tuned (LoRA)",
  base: "Base model",
  gemini: "Gemini (reference)",
};

interface Props {
  result: ExtractResult;
  /** Gemini's answer, when it produced one; other panels flag fields that disagree with it. */
  reference: JobExtraction | null;
}

export function ModelPanel({ result, reference }: Props) {
  const { extraction } = result;
  const isReference = result.model === "gemini";

  return (
    <section
      className="glass flex flex-col rounded-xl p-4"
      data-testid={`panel-${result.model}`}
      aria-label={MODEL_LABELS[result.model]}
    >
      <header className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold" style={{ fontFamily: "var(--font-display)" }}>
            {MODEL_LABELS[result.model]}
          </h3>
          <p className="truncate text-xs text-[var(--muted)]" title={result.model_id}>
            {result.model_id}
          </p>
        </div>
        <div className="flex shrink-0 flex-col items-end gap-1 text-xs">
          <span
            data-testid={`valid-${result.model}`}
            className={`rounded px-1.5 py-0.5 font-medium ${
              result.valid_json
                ? "bg-[var(--accent)]/15 text-[var(--accent)]"
                : "bg-[var(--danger)]/15 text-[var(--danger)]"
            }`}
          >
            {result.valid_json ? "valid JSON" : "invalid"}
          </span>
          {result.latency_ms > 0 && (
            <span className="font-mono text-[var(--muted)]">{result.latency_ms} ms</span>
          )}
        </div>
      </header>

      {extraction ? (
        <dl className="space-y-2.5 text-sm">
          {FIELD_UNITS.map(({ key, label }) => {
            const differs =
              reference && !isReference && differsFromReference(extraction, reference, key);
            return (
              <div key={key} data-testid={`field-${result.model}-${key}`}>
                <dt className="flex items-center gap-1.5 text-xs text-[var(--muted)]">
                  {label}
                  {differs && (
                    <span
                      className="inline-block h-1.5 w-1.5 rounded-full bg-[var(--warn)]"
                      title="Differs from Gemini"
                    />
                  )}
                </dt>
                <dd className={key === "skills" ? "mt-1 flex flex-wrap gap-1" : "mt-0.5"}>
                  {key === "skills" && extraction.required_skills.length > 0
                    ? extraction.required_skills.map((s) => (
                        <span
                          key={s}
                          className="rounded border border-[var(--line)] bg-[var(--bg-2)] px-1.5 py-0.5 text-xs"
                        >
                          {s}
                        </span>
                      ))
                    : displayValue(extraction, key)}
                </dd>
              </div>
            );
          })}
        </dl>
      ) : (
        <div className="space-y-2 text-sm" data-testid={`error-${result.model}`}>
          <p className="text-[var(--danger)]">{result.error ?? "No output"}</p>
        </div>
      )}

      {result.raw && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-[var(--muted)] hover:text-[var(--text)]">
            Raw output
          </summary>
          <pre className="mt-2 max-h-48 overflow-auto whitespace-pre-wrap break-words rounded bg-[var(--bg-0)] p-2 font-mono text-[11px] leading-relaxed">
            {result.raw}
          </pre>
        </details>
      )}
    </section>
  );
}
