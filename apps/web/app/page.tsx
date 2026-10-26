"use client";

import { ModelPanel, MODEL_LABELS } from "@/components/ModelPanel";
import {
  ApiError,
  MODEL_KEYS,
  compareModels,
  getHealth,
  type ExtractResult,
  type Health,
  type ModelKey,
} from "@/lib/api";
import { MAX_POSTING_CHARS, MIN_POSTING_CHARS } from "@/lib/bff/schemas";
import { SAMPLES } from "@/lib/samples";
import { useEffect, useRef, useState } from "react";

export default function PlaygroundPage() {
  const [text, setText] = useState(SAMPLES[0].text);
  const [selected, setSelected] = useState<ModelKey[]>([...MODEL_KEYS]);
  const [health, setHealth] = useState<Health | null>(null);
  const [results, setResults] = useState<ExtractResult[] | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    const ac = new AbortController();
    getHealth(ac.signal)
      .then(setHealth)
      .catch(() => {
        /* health is advisory — the form still works and errors surface on submit */
      });
    return () => ac.abort();
  }, []);

  const tooShort = text.trim().length < MIN_POSTING_CHARS;
  const tooLong = text.length > MAX_POSTING_CHARS;
  const canRun = !running && !tooShort && !tooLong && selected.length > 0;

  function toggle(key: ModelKey) {
    setSelected((prev) => (prev.includes(key) ? prev.filter((k) => k !== key) : [...prev, key]));
  }

  async function run() {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;
    setRunning(true);
    setError(null);
    setResults(null);
    try {
      // Preserve a stable panel order regardless of click order.
      const models = MODEL_KEYS.filter((k) => selected.includes(k));
      setResults(await compareModels(text.trim(), models, ac.signal));
    } catch (err) {
      if ((err as Error).name === "AbortError") return;
      setError(err instanceof ApiError || err instanceof Error ? err.message : "Request failed");
    } finally {
      setRunning(false);
    }
  }

  const reference = results?.find((r) => r.model === "gemini")?.extraction ?? null;

  return (
    <div className="animate-fade-up space-y-8">
      <div>
        <h1
          className="text-3xl font-semibold tracking-tight"
          style={{ fontFamily: "var(--font-display)" }}
        >
          Job posting → structured JSON
        </h1>
        <p className="mt-2 max-w-2xl text-sm text-[var(--muted)]">
          A ~1.5B-parameter model fine-tuned with QLoRA to extract title, seniority, required
          skills, experience, work mode and salary. Paste a posting and compare it with the
          un-tuned base model and Gemini, side by side.
        </p>
        {health?.mock && (
          <p
            className="mt-3 inline-block rounded-md border border-[var(--warn)]/40 bg-[var(--warn)]/10 px-2.5 py-1 text-xs text-[var(--warn)]"
            data-testid="mock-banner"
          >
            Demo mode: every column is served by a regex stand-in, not a real model.
          </p>
        )}
      </div>

      <div className="glass space-y-4 rounded-xl p-4 sm:p-5" data-testid="extract-form">
        <div className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-[var(--muted)]">Try a sample:</span>
          {SAMPLES.map((s) => (
            <button
              key={s.id}
              type="button"
              onClick={() => setText(s.text)}
              className="rounded-md border border-[var(--line)] px-2.5 py-1 text-xs text-[var(--muted)] transition hover:border-[var(--accent)] hover:text-[var(--text)]"
            >
              {s.label}
            </button>
          ))}
        </div>

        <label className="block text-sm">
          <span className="sr-only">Job posting text</span>
          <textarea
            data-testid="posting-text"
            value={text}
            onChange={(e) => setText(e.target.value)}
            rows={12}
            placeholder="Paste a job posting…"
            className="w-full rounded-md border border-[var(--line)] bg-[var(--bg-0)] px-3 py-2 font-mono text-[13px] leading-relaxed outline-none focus:border-[var(--accent)]"
          />
          <span
            className={`mt-1 block text-right text-xs ${
              tooLong ? "text-[var(--danger)]" : "text-[var(--muted)]"
            }`}
          >
            {text.length.toLocaleString()} / {MAX_POSTING_CHARS.toLocaleString()}
          </span>
        </label>

        <div className="flex flex-wrap items-center justify-between gap-3">
          <fieldset className="flex flex-wrap items-center gap-4 text-sm">
            <legend className="sr-only">Models to run</legend>
            {MODEL_KEYS.map((key) => {
              const unavailable = health ? !health.backends[key] : false;
              return (
                <label
                  key={key}
                  className={`flex items-center gap-1.5 ${unavailable ? "opacity-50" : ""}`}
                  title={unavailable ? "Not configured on this deployment" : undefined}
                >
                  <input
                    type="checkbox"
                    data-testid={`model-${key}`}
                    checked={selected.includes(key)}
                    onChange={() => toggle(key)}
                    className="accent-[var(--accent)]"
                  />
                  {MODEL_LABELS[key]}
                  {unavailable && <span className="text-xs text-[var(--muted)]">(offline)</span>}
                </label>
              );
            })}
          </fieldset>
          <button
            type="button"
            data-testid="extract-submit"
            onClick={run}
            disabled={!canRun}
            className="rounded-md bg-[var(--accent)] px-4 py-2 text-sm font-semibold text-[var(--bg-0)] transition hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
          >
            {running ? "Extracting…" : "Extract"}
          </button>
        </div>
        {tooShort && (
          <p className="text-xs text-[var(--muted)]">
            Enter at least {MIN_POSTING_CHARS} characters.
          </p>
        )}
      </div>

      {error && (
        <p
          role="alert"
          data-testid="extract-error"
          className="rounded-md border border-[var(--danger)]/40 bg-[var(--danger)]/10 px-3 py-2 text-sm text-[var(--danger)]"
        >
          {error}
        </p>
      )}

      {running && (
        <p className="animate-pulse-line text-sm text-[var(--muted)]">
          Running {selected.length} model{selected.length === 1 ? "" : "s"} — a cold endpoint can
          take up to a minute…
        </p>
      )}

      {results && (
        <div className="space-y-3" data-testid="results">
          <div
            className={`grid gap-4 ${
              results.length === 1
                ? "md:grid-cols-1"
                : results.length === 2
                  ? "md:grid-cols-2"
                  : "md:grid-cols-3"
            }`}
          >
            {results.map((r) => (
              <ModelPanel key={r.model} result={r} reference={reference} />
            ))}
          </div>
          {reference && results.length > 1 && (
            <p className="text-xs text-[var(--muted)]">
              <span className="mr-1 inline-block h-1.5 w-1.5 rounded-full bg-[var(--warn)] align-middle" />
              marks a field that differs from Gemini. Gemini is a reference, not ground truth — see
              the Results page for scores against hand-checked labels.
            </p>
          )}
        </div>
      )}
    </div>
  );
}
