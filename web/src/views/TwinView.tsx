import { useEffect, useRef, useState } from "react";
import { api, generate, type ExplainResponse, type Feature, type GenerateStep, type RunResponse } from "../api";
import { Info } from "../components/Info";
import { EXAMPLES } from "../examples";
import { GLOSSARY } from "../glossary";
import { Skeleton, Stage } from "../components/Skeleton";
import { useTooltip } from "../components/Tooltip";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { decidedAt, tokenText } from "../lib";
import { ReasoningGraph } from "./ReasoningGraph";
import { WhatIf } from "./WhatIf";


const STEPS = ["Run GPT-2", "Encode 12 transcoders", "Attribute every feature"];

function HowItWorks() {
  const items: [string, string][] = [
    ["Trace", "GPT-2 answers your prompt while the twin, an interpretable copy built from transcoder features, explains the answer as a reasoning path."],
    ["Click a feature", "The twin predicts what happens if that feature is turned off or up, at five levels of approximation."],
    ["Check the twin", "The real GPT-2 runs the same edit. A trust score, computed before GPT-2 runs, tells you whether to believe the graph."],
  ];
  return (
    <ol className="grid gap-3 sm:grid-cols-3">
      {items.map(([title, body], i) => (
        <li key={title} className="card flex gap-3 p-3.5">
          <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full text-xs font-semibold"
            style={{ background: "var(--text-primary)", color: "var(--surface-1)" }}>{i + 1}</span>
          <div>
            <div className="text-sm font-semibold text-ink">{title}</div>
            <p className="mt-0.5 text-xs leading-relaxed text-ink2">{body}</p>
          </div>
        </li>
      ))}
    </ol>
  );
}

function Generation({ prompt }: { prompt: string }) {
  const [steps, setSteps] = useState<GenerateStep[]>([]);
  const [busy, setBusy] = useState(false);
  const stop = useRef<() => void>(() => {});
  const { show, hide } = useTooltip();
  useEffect(() => { setSteps([]); }, [prompt]);
  const start = () => {
    setSteps([]); setBusy(true);
    stop.current = generate(prompt, (s) => setSteps((p) => [...p, s]), () => setBusy(false), 16);
  };
  return (
    <Card title="Live generation"
      subtitle="GPT-2 continues the prompt. Each token is shaded by the layer at which the logit lens settles on it: darker means decided later."
      right={busy ? <Button kind="ghost" onClick={() => stop.current()}>Stop</Button> : <Button kind="ghost" onClick={start}>Generate ▸</Button>}>
      <div className="flex flex-wrap items-end gap-1 font-mono text-sm leading-7">
        <span className="tok text-ink2">{prompt}</span>
        {steps.map((s, i) => {
          const l = decidedAt(s);
          const shade = `color-mix(in srgb, var(--series-1) ${15 + (l / 12) * 70}%, transparent)`;
          return (
            <span key={i} className="tok rounded px-0.5" style={{ background: shade }}
              onMouseMove={(e) => show(e, (
                <div>
                  <div><b>“{tokenText(s.token)}”</b> settled at layer {l}</div>
                  <div className="text-ink2">alternatives: {s.top.slice(1, 4).map((t) => `“${tokenText(t.token)}” ${(t.prob * 100).toFixed(0)}%`).join(", ")}</div>
                </div>
              ))}
              onMouseLeave={hide}>{s.token}</span>
          );
        })}
        {busy && <span className="animate-pulse text-muted">▍</span>}
      </div>
    </Card>
  );
}

function ErrorStrip({ exp }: { exp: ExplainResponse }) {
  const { show, hide } = useTooltip();
  return (
    <div className="flex h-40 items-end gap-0.5" role="img" aria-label="Transcoder error fraction per layer">
      {Object.entries(exp.error_fraction).map(([l, v]) => (
        <div key={l} className="flex flex-1 flex-col items-center gap-1"
          onMouseMove={(e) => show(e, <div><b>Layer {l}</b>: {(v * 100).toFixed(0)}% of the MLP output is unexplained</div>)}
          onMouseLeave={hide}>
          <div className="w-full rounded-t" style={{ height: `${Math.max(2, Math.min(1, v) * 136)}px`, background: "var(--series-2)" }} />
          <span className="text-[9px] text-muted tnum">{l}</span>
        </div>
      ))}
    </div>
  );
}

export function TwinView({ prompt, setPrompt }: { prompt: string; setPrompt: (p: string) => void }) {
  const [run, setRun] = useState<RunResponse | null>(null);
  const [exp, setExp] = useState<ExplainResponse | null>(null);
  const [sel, setSel] = useState<Feature | null>(null);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);
  const [error, setError] = useState<string | null>(null);
  const [ran, setRan] = useState("");

  const go = async (p = prompt) => {
    if (!p.trim()) return;
    setBusy(true); setError(null); setStage(0);
    const timers = [setTimeout(() => setStage(1), 500), setTimeout(() => setStage(2), 1400)];
    try {
      const [r, e] = await Promise.all([api.run(p), api.explain(p)]);
      setRun(r); setExp(e); setRan(p);
      setSel(e.features.find((f) => f.effect > 0) ?? e.features[0] ?? null);  // start with the strongest supporter
    } catch (err) { setError(String(err)); }
    finally { timers.forEach(clearTimeout); setBusy(false); }
  };

  useEffect(() => { go(); }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // On narrow screens the graph scrolls; start at the right edge, where the prediction is.
  const graphBox = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = graphBox.current;
    if (el) el.scrollLeft = el.scrollWidth;
  }, [exp]);

  return (
    <div className="space-y-4">
      {!exp && <HowItWorks />}
      <div className="card p-4">
        <label htmlFor="prompt" className="text-xs font-medium text-ink2">Prompt for GPT-2 small</label>
        <div className="mt-1.5 flex flex-col gap-2 sm:flex-row">
          <textarea id="prompt" rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); } }}
            className="tok min-h-[2.75rem] flex-1 resize-y rounded-lg border bg-transparent px-3 py-2 text-sm text-ink outline-none focus:ring-2"
            style={{ borderColor: "var(--border)" }} maxLength={240} />
          <Button onClick={() => go()} disabled={busy || !prompt.trim()}>{busy ? "Tracing…" : "Trace ↵"}</Button>
        </div>
        <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
          <span className="text-xs text-muted">Try:</span>
          {EXAMPLES.map(([ex, kind]) => (
            <button key={ex} onClick={() => { setPrompt(ex); go(ex); }} disabled={busy} title={ex}
              className="rounded-full border px-2.5 py-0.5 text-xs text-ink2 transition hover:text-ink disabled:opacity-50"
              style={{ borderColor: "var(--border)" }}>{kind}</button>
          ))}
        </div>
      </div>
      <ErrorNote error={error} />

      {busy && !exp && (
        <div className="space-y-3">
          <Stage steps={STEPS} active={stage} />
          <div className="grid gap-4 lg:grid-cols-[1.55fr_1fr]">
            <Skeleton className="h-[460px]" />
            <div className="space-y-3"><Skeleton className="h-40" /><Skeleton className="h-64" /></div>
          </div>
        </div>
      )}

      {run && exp && (
        <div className={`space-y-4 transition-opacity ${busy ? "opacity-50" : ""}`} aria-busy={busy}>
          {busy && <Stage steps={STEPS} active={stage} />}
          <div className="grid items-start gap-4 lg:grid-cols-[1.55fr_1fr]">
            <Card title="Reasoning path"
              subtitle={<>The features with the largest direct effect<Info>{GLOSSARY.directEffect}</Info> on predicting “<span className="tok">{tokenText(exp.target)}</span>”, by token and layer. Hover to isolate one, click to edit it.</>}>
              <div ref={graphBox} className="-mx-1 overflow-x-auto px-1"><ReasoningGraph tokens={exp.tokens} target={exp.target} features={exp.features} selected={sel} onSelect={setSel} /></div>
              <p className="mt-1 text-[11px] text-muted sm:hidden">← scroll for earlier tokens</p>
            </Card>
            <div className="min-w-0 space-y-4 lg:sticky lg:top-4">
              <WhatIf prompt={ran} feature={sel} tokens={exp.tokens} />
            </div>
          </div>
          <div className="grid gap-4 md:grid-cols-2">
            <Card title="Prediction" subtitle={<>GPT-2's next-token distribution for “<span className="tok">{tokenText(ran).slice(-40)}</span>”</>}>
              <TokenBars items={run.top} />
            </Card>
            <Card title={<>What the twin cannot explain<Info>{GLOSSARY.error}</Info></>}
              subtitle="Share of each layer's MLP output at the last token that the transcoder does not reconstruct">
              <ErrorStrip exp={exp} />
            </Card>
          </div>
          <Generation prompt={ran} />
        </div>
      )}
    </div>
  );
}
