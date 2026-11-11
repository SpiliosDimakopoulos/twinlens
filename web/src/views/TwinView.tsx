import { useCallback, useEffect, useRef, useState } from "react";
import { api, STATIC, type ExplainResponse, type Feature, type RunResponse, type Summary } from "../api";
import { Glyph } from "../components/Glyph";
import { Info } from "../components/Info";
import { Skeleton, Stage } from "../components/Skeleton";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { PRESETS } from "../constants";
import { EXAMPLES, REPO } from "../examples";
import { GLOSSARY } from "../glossary";
import { isTypingTarget } from "../keys";
import { tokenText } from "../lib";
import { readParam, writeParams } from "../url";
import { useLatest } from "../useLatest";
import { ReasoningGraph } from "./ReasoningGraph";
import { ErrorStrip } from "./twin/ErrorStrip";
import { Generation } from "./twin/Generation";
import { Hero } from "./twin/Hero";
import { Specimens } from "./twin/Specimens";
import { WhatIf } from "./WhatIf";

const STEPS = ["Run GPT-2", "Decode its 12 layers into features", "Measure each feature's push on the answer"];
const fkey = (f: Feature) => `${f.layer}-${f.pos}-${f.feature}`;

export function TwinView({ prompt, setPrompt }: { prompt: string; setPrompt: (p: string) => void }) {
  const [run, setRun] = useState<RunResponse | null>(null);
  const [exp, setExp] = useState<ExplainResponse | null>(null);
  const [sel, setSel] = useState<Feature | null>(null);
  const [mult, setMult] = useState(() => {
    const m = Number(readParam("m"));
    return PRESETS.some(([, v]) => v === m) ? m : 0;
  });
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [ran, setRan] = useState("");
  const [summary, setSummary] = useState<Summary | null>(null);
  const wanted = useRef<string | null>(readParam("f"));
  const [firstPrompt] = useState(prompt);
  const begin = useLatest();
  const graphBox = useRef<HTMLDivElement>(null);
  const active = EXAMPLES.findIndex(([t]) => t === prompt);

  useEffect(() => { api.summary().then(setSummary).catch(() => {}); }, []);

  const go = useCallback(async (p: string) => {
    if (!p.trim()) { setError("Write a prompt first."); return; }
    const isLatest = begin();  // only the latest request may update the view
    setBusy(true); setError(null); setStage(0);
    const timers = [setTimeout(() => setStage(1), 500), setTimeout(() => setStage(2), 1400)];
    try {
      const [r, e] = await Promise.all([api.run(p), api.explain(p)]);
      if (!isLatest()) return;
      setRun(r); setExp(e); setRan(p);
      const fromUrl = e.features.find((f) => fkey(f) === wanted.current);
      wanted.current = null;  // one-shot: a stale request returns above, before this
      setSel(fromUrl ?? e.features.find((f) => f.effect > 0) ?? e.features[0] ?? null);
    } catch (err) { if (isLatest()) setError(String(err)); }
    finally {
      timers.forEach(clearTimeout);
      if (isLatest()) setBusy(false);
    }
  }, [begin]);

  // Trace the prompt the page opened with; later prompts are traced on demand.
  useEffect(() => { go(firstPrompt); }, [go, firstPrompt]);

  // Keep the URL shareable: example (or custom prompt), selected feature, edit strength.
  useEffect(() => {
    if (!ran) return;
    const idx = EXAMPLES.findIndex(([t]) => t === ran);
    writeParams({ p: idx >= 0 ? idx + 1 : null, q: idx >= 0 ? null : ran, f: sel ? fkey(sel) : null, m: mult || null });
  }, [ran, sel, mult]);

  useEffect(() => {
    const el = graphBox.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [exp]);

  const pick = useCallback((i: number) => {
    const example = EXAMPLES[i];
    if (!example) return;
    setPrompt(example[0]); setMult(0); go(example[0]);
  }, [go, setPrompt]);

  // Keyboard: ← → cycle features by strength, [ ] cycle examples. Only while focus is on the
  // page itself or inside the graph, so the keys never hijack another control.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (isTypingTarget(e) || e.metaKey || e.ctrlKey || e.altKey || !exp || busy) return;
      const a = document.activeElement;
      if (a && a !== document.body && !graphBox.current?.contains(a)) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const fs = exp.features;
        if (!fs.length) return;
        const i = sel ? fs.findIndex((f) => fkey(f) === fkey(sel)) : -1;
        const step = e.key === "ArrowRight" ? 1 : -1;
        const next = fs[i < 0 ? (step > 0 ? 0 : fs.length - 1) : (i + step + fs.length) % fs.length];
        if (next) { e.preventDefault(); setSel(next); }
      } else if (e.key === "]" || e.key === "[") {
        const base = active >= 0 ? active : 0;
        pick((base + (e.key === "]" ? 1 : -1) + EXAMPLES.length) % EXAMPLES.length);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exp, sel, active, pick, busy]);

  return (
    <div>
      <Hero summary={summary} />

      <section className="border-t pt-5 rule">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="kicker">How to use it</div>
          {STATIC && (
            <a className="text-[12px] text-ink2 underline decoration-[var(--baseline)] hover:text-ink" href={`${REPO}#quickstart`} target="_blank" rel="noreferrer">
              Demo with precomputed runs · install locally to try your own prompts
            </a>
          )}
        </div>
        <ol className="mt-2 grid gap-x-6 gap-y-1 text-[12.5px] text-ink2 sm:grid-cols-3">
          <li><b className="text-ink">1.</b> Pick an example prompt below{STATIC ? "" : " or write your own"}. GPT-2 predicts the next word and the graph shows which features drove it.</li>
          <li><b className="text-ink">2.</b> Click a dot in the graph to switch that feature off (or up) as an experiment.</li>
          <li><b className="text-ink">3.</b> In the "What if" panel, compare each dot (a prediction) with the black line (what GPT-2 really did).</li>
        </ol>
        <div className="mt-3"><Specimens active={active} onPick={pick} disabled={busy} /></div>
        {!STATIC && (
          <div className="mt-3 flex flex-col gap-2 sm:flex-row">
            <label htmlFor="prompt" className="sr-only">Prompt for GPT-2 small</label>
            <textarea id="prompt" rows={1} value={prompt} onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); if (!busy) go(prompt); } }}
              className="tok ring-baseline min-h-[2.5rem] flex-1 resize-y rounded-[5px] bg-[var(--surface-1)] px-3 py-2 text-[13.5px] text-ink outline-none"
              maxLength={240} />
            <Button onClick={() => go(prompt)} disabled={busy || !prompt.trim()}>{busy ? "Tracing…" : <>Trace <Glyph name="arrow" size={12} /></>}</Button>
          </div>
        )}
      </section>

      <div className="mt-5" role="alert"><ErrorNote error={error} /></div>

      {busy && !exp && (
        <div className="mt-5 space-y-3">
          <Stage steps={STEPS} active={stage} />
          <div className="grid gap-5 lg:grid-cols-[1.45fr_1fr]">
            <Skeleton className="h-[480px]" />
            <div className="space-y-3"><Skeleton className="h-24" /><Skeleton className="h-72" /></div>
          </div>
        </div>
      )}

      {run && exp && (
        <div className={`mt-5 space-y-5 transition-opacity ${busy ? "opacity-50" : ""}`} aria-busy={busy}>
          <div className="grid items-start gap-5 lg:grid-cols-[1.45fr_1fr]">
            <div className="min-w-0 lg:col-start-1 lg:row-start-1">
              <Card label="Fig. 1"
                title={<>The reasoning path behind “<span className="tok">{tokenText(exp.target)}</span>”</>}
                subtitle={<>The features that push GPT-2 most toward (blue) or away from (red) this answer, by word and layer<Info wide>{GLOSSARY.graph}</Info>. Hover a dot to isolate it; click it to test what happens if you switch it off.</>}>
                <div ref={graphBox} className="-mx-1 overflow-x-auto px-1">
                  <ReasoningGraph tokens={exp.tokens} target={exp.target} features={exp.features} selected={sel} onSelect={setSel} />
                </div>
                <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-muted">
                  <span><kbd>←</kbd> <kbd>→</kbd> step through features</span>
                  <span><kbd>[</kbd> <kbd>]</kbd> change example</span>
                  <span className="sm:hidden">scroll the figure for earlier tokens</span>
                </p>
              </Card>
            </div>
            <div className="min-w-0 self-stretch lg:col-start-2 lg:row-span-2 lg:row-start-1">
              <div className="lg:sticky lg:top-4">
                <WhatIf prompt={ran} feature={sel} tokens={exp.tokens} mult={mult} setMult={setMult} />
              </div>
            </div>
            <div className="min-w-0 lg:col-start-1 lg:row-start-2">
              <Card label="Fig. 3" title="What GPT-2 predicts" subtitle={<>The most likely next words after “<span className="tok whitespace-pre-wrap">{tokenText(ran).slice(-36)}</span>”</>}>
                <TokenBars items={run.top} />
              </Card>
            </div>
          </div>
          <div className="grid items-start gap-5 md:grid-cols-[1fr_1.4fr]">
            <Card label="Fig. 4" title={<>What the twin cannot explain<Info>{GLOSSARY.error}</Info></>}
              subtitle="For each layer (0 to 11), the share of the MLP's output at the last word that the twin's transcoder cannot reproduce. Higher bars mean less of the computation is explained.">
              <ErrorStrip exp={exp} />
            </Card>
            <Generation prompt={ran} />
          </div>
        </div>
      )}
    </div>
  );
}
