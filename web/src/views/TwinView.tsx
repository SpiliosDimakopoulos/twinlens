import { useRef, useState } from "react";
import { api, generate, type ExplainResponse, type Feature, type GenerateStep, type RunResponse } from "../api";
import { useTooltip } from "../components/Tooltip";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { decidedAt, tokenText } from "../lib";
import { ReasoningGraph } from "./ReasoningGraph";
import { WhatIf } from "./WhatIf";

export const EXAMPLES = [
  "Michael Jordan plays the sport of",
  "When Mary and John went to the store, John gave a drink to",
  "The Colosseum is located in the city of",
  "def add(a, b):\n    return",
  "Monday, Tuesday, Wednesday,",
];

function Generation({ prompt }: { prompt: string }) {
  const [steps, setSteps] = useState<GenerateStep[]>([]);
  const [busy, setBusy] = useState(false);
  const stop = useRef<() => void>(() => {});
  const { show, hide } = useTooltip();
  const start = () => {
    setSteps([]); setBusy(true);
    stop.current = generate(prompt, (s) => setSteps((p) => [...p, s]), () => setBusy(false), 16);
  };
  return (
    <Card title="Live generation"
      subtitle="Each generated token, shaded by the layer at which the logit lens already settles on it: darker = decided later."
      right={busy ? <Button kind="ghost" onClick={() => stop.current()}>Stop</Button> : <Button kind="ghost" onClick={start}>Generate</Button>}>
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

export function TwinView({ prompt, setPrompt }: { prompt: string; setPrompt: (p: string) => void }) {
  const [run, setRun] = useState<RunResponse | null>(null);
  const [exp, setExp] = useState<ExplainResponse | null>(null);
  const [sel, setSel] = useState<Feature | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [ran, setRan] = useState("");

  const go = async (p = prompt) => {
    setBusy(true); setError(null); setSel(null);
    try {
      const [r, e] = await Promise.all([api.run(p), api.explain(p)]);
      setRun(r); setExp(e); setRan(p);
    } catch (err) { setError(String(err)); }
    finally { setBusy(false); }
  };

  return (
    <div className="space-y-4">
      <div className="card p-4">
        <label htmlFor="prompt" className="text-xs font-medium text-ink2">Prompt for GPT-2 small</label>
        <div className="mt-1.5 flex flex-col gap-2 sm:flex-row">
          <textarea id="prompt" rows={2} value={prompt} onChange={(e) => setPrompt(e.target.value)}
            onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(); } }}
            className="tok min-h-[2.75rem] flex-1 resize-y rounded-lg border bg-transparent px-3 py-2 text-sm text-ink outline-none focus:ring-2"
            style={{ borderColor: "var(--border)" }} />
          <Button onClick={() => go()} disabled={busy || !prompt.trim()}>{busy ? "Tracing…" : "Trace"}</Button>
        </div>
        <div className="mt-2 flex flex-wrap gap-1.5">
          {EXAMPLES.map((ex) => (
            <button key={ex} onClick={() => { setPrompt(ex); go(ex); }}
              className="tok rounded-full border px-2.5 py-0.5 text-xs text-ink2 hover:text-ink"
              style={{ borderColor: "var(--border)" }}>{tokenText(ex).slice(0, 38)}</button>
          ))}
        </div>
      </div>
      <ErrorNote error={error} />
      {run && exp && (
        <>
          <div className="grid gap-4 lg:grid-cols-[2fr_1fr]">
            <Card title="Reasoning path"
              subtitle={<>The 24 transcoder features with the largest direct effect on predicting “<span className="tok">{tokenText(exp.target)}</span>”, placed by token and layer.</>}>
              <ReasoningGraph tokens={exp.tokens} target={exp.target} features={exp.features} selected={sel} onSelect={setSel} />
            </Card>
            <div className="space-y-4">
              <Card title="Prediction" subtitle="GPT-2's next-token distribution">
                <TokenBars items={run.top} />
              </Card>
              <Card title="What the twin cannot explain"
                subtitle="Share of each MLP's output at the last token left in the transcoder's error term">
                <div className="flex h-16 items-end gap-0.5" role="img" aria-label="Transcoder error fraction per layer">
                  {Object.entries(exp.error_fraction).map(([l, v]) => (
                    <div key={l} className="flex flex-1 flex-col items-center gap-1">
                      <div className="w-full rounded-t" title={`L${l}: ${(v * 100).toFixed(0)}%`}
                        style={{ height: `${Math.min(1, v) * 48}px`, background: "var(--series-2)" }} />
                      <span className="text-[9px] text-muted tnum">{l}</span>
                    </div>
                  ))}
                </div>
              </Card>
            </div>
          </div>
          <WhatIf prompt={ran} feature={sel} tokens={exp.tokens} />
          <Generation prompt={ran} />
        </>
      )}
    </div>
  );
}
