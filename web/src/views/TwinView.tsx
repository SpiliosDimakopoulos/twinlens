import { useCallback, useEffect, useRef, useState } from "react";
import { api, generate, STATIC, type ExplainResponse, type Feature, type GenerateStep, type RunResponse, type Summary } from "../api";
import { Glyph } from "../components/Glyph";
import { Info } from "../components/Info";
import { Skeleton, Stage } from "../components/Skeleton";
import { useTooltip } from "../components/Tooltip";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { EXAMPLES, REPO } from "../examples";
import { GLOSSARY } from "../glossary";
import { decidedAt, tokenText } from "../lib";
import { readParam, typing, writeParams } from "../url";
import { ReasoningGraph } from "./ReasoningGraph";
import { PRESETS, WhatIf } from "./WhatIf";

const STEPS = ["Run GPT-2", "Encode 12 transcoders", "Attribute every feature"];
const fkey = (f: Feature) => `${f.layer}-${f.pos}-${f.feature}`;

/** The finding, stated up front, with the three numbers that carry it. */
function Hero({ summary }: { summary: Summary | null }) {
  const fix = summary?.fix?.ladder;
  const rho0 = fix?.linear?.spearman?.[0], rho1 = fix?.linear_fulljac?.spearman?.[0];
  const cat0 = fix?.linear?.catastrophic?.[0], cat1 = fix?.linear_fulljac?.catastrophic?.[0];
  const auc = summary?.trust.cv_auroc;
  return (
    <section className="grid gap-8 pb-8 pt-4 lg:grid-cols-[1.35fr_1fr] lg:gap-14">
      <div>
        <div className="kicker">Mechanistic interpretability · GPT-2 small · 8,064 interventions</div>
        <h1 className="display mt-3 text-[38px] text-ink sm:text-[50px]">
          Attribution graphs fail because of the derivatives they drop, <em className="italic text-ink2">not because they are linear.</em>
        </h1>
        <p className="mt-4 max-w-[60ch] text-[16px] leading-relaxed text-ink2">
          An attribution graph explains a model through an interpretable twin of it. TwinLens runs that
          twin beside the real GPT-2, asks both what happens when a feature is switched off, and measures
          the gap. Here the gap is often large. Put two first-order terms back (LayerNorm's scale and
          the attention pattern) and the graph stays linear but stops getting answers backwards.
        </p>
      </div>
      <div className="grid content-end gap-5 sm:grid-cols-3 lg:grid-cols-1">
        <HeroStat value={rho0 !== undefined && rho1 !== undefined ? `${rho0.toFixed(2)} → ${rho1.toFixed(2)}` : "—"}
          label="Rank agreement with GPT-2" note="Spearman ρ of predicted vs real effect, graph today → graph + Jacobians" />
        <HeroStat value={cat0 !== undefined && cat1 !== undefined ? `${pct(cat0)} → ${pct(cat1)}` : "—"}
          label="Catastrophic misses" note="Predictions off by more than one nat" />
        <HeroStat value={auc !== undefined ? auc.toFixed(2) : "—"}
          label="Knows when it's wrong" note="AUROC of the trust score on unseen prompts" />
      </div>
    </section>
  );
}

const pct = (v: number) => `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`;

function HeroStat({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="border-t pt-2.5 rule">
      <div className="kicker">{label}</div>
      <div className="serif mt-1 text-[30px] leading-none tracking-tight text-ink tnum">{value}</div>
      <div className="mt-1 text-[12px] leading-snug text-muted">{note}</div>
    </div>
  );
}

/** Example prompts presented as a specimen list: the task family over the prompt itself. */
function Specimens({ active, onPick, disabled }: { active: number; onPick: (i: number) => void; disabled: boolean }) {
  return (
    <div className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1" role="tablist" aria-label="Example prompts">
      {EXAMPLES.map(([text, kind], i) => (
        <button key={text} role="tab" aria-selected={i === active} disabled={disabled} onClick={() => onPick(i)}
          className="group min-w-[150px] max-w-[220px] shrink-0 rounded-[5px] px-3 py-2 text-left transition-colors disabled:opacity-60 lg:min-w-0 lg:max-w-none lg:flex-1"
          style={i === active ? { background: "var(--surface-1)", boxShadow: "inset 0 0 0 1px var(--baseline), inset 0 -2px 0 var(--text-primary)" } : {}}>
          <div className="kicker truncate" style={i === active ? { color: "var(--text-primary)" } : {}}>{String(i + 1).padStart(2, "0")} · {kind}</div>
          <div className="tok mt-1 truncate text-[12px] text-ink2 group-hover:text-ink">{tokenText(text)}</div>
        </button>
      ))}
    </div>
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
    <Card label="Fig. 5" title="Live generation"
      subtitle="GPT-2 continues the prompt. Each token is shaded by the layer at which the logit lens settles on it: darker means decided later."
      right={busy
        ? <Button kind="ghost" onClick={() => stop.current()}><Glyph name="stop" size={10} />Stop</Button>
        : <Button kind="ghost" onClick={start}><Glyph name="play" size={10} />Generate</Button>}>
      <div className="flex flex-wrap items-end gap-[3px] text-[13.5px] leading-7">
        <span className="tok whitespace-pre-wrap break-words text-muted">{prompt}</span>
        {steps.map((s, i) => {
          const l = decidedAt(s);
          const shade = `color-mix(in srgb, var(--series-1) ${12 + (l / 12) * 68}%, transparent)`;
          return (
            <span key={i} className="tok rounded-[3px] px-[2px]" style={{ background: shade }}
              onMouseMove={(e) => show(e, (
                <div>
                  <div><b>“{tokenText(s.token)}”</b> settled at layer {l}</div>
                  <div className="text-ink2">runners-up: {s.top.slice(1, 4).map((t) => `“${tokenText(t.token)}” ${(t.prob * 100).toFixed(0)}%`).join(", ")}</div>
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
    <div className="flex h-28 items-end gap-[3px]" role="img" aria-label="Transcoder error fraction per layer">
      {Object.entries(exp.error_fraction).map(([l, v]) => (
        <div key={l} className="flex flex-1 flex-col items-center gap-1"
          onMouseMove={(e) => show(e, <div><b>Layer {l}</b>: {(v * 100).toFixed(0)}% of the MLP output is unexplained</div>)}
          onMouseLeave={hide}>
          <div className="w-full rounded-t-[3px]" style={{ height: `${Math.max(2, Math.min(1, v) * 86)}px`, background: "var(--series-2)", opacity: 0.6 }} />
          <span className="text-[9.5px] text-muted tnum">{l}</span>
        </div>
      ))}
    </div>
  );
}

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
  const graphBox = useRef<HTMLDivElement>(null);
  const active = EXAMPLES.findIndex(([t]) => t === prompt);

  useEffect(() => { api.summary().then(setSummary).catch(() => {}); }, []);

  const go = useCallback(async (p: string) => {
    if (!p.trim()) return;
    setBusy(true); setError(null); setStage(0);
    const timers = [setTimeout(() => setStage(1), 500), setTimeout(() => setStage(2), 1400)];
    try {
      const [r, e] = await Promise.all([api.run(p), api.explain(p)]);
      setRun(r); setExp(e); setRan(p);
      const fromUrl = e.features.find((f) => fkey(f) === wanted.current);
      wanted.current = null;
      setSel(fromUrl ?? e.features.find((f) => f.effect > 0) ?? e.features[0] ?? null);
    } catch (err) { setError(String(err)); }
    finally { timers.forEach(clearTimeout); setBusy(false); }
  }, []);

  useEffect(() => { go(prompt); }, []); // eslint-disable-line react-hooks/exhaustive-deps

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

  const pick = useCallback((i: number) => { setPrompt(EXAMPLES[i][0]); setMult(0); go(EXAMPLES[i][0]); }, [go, setPrompt]);

  // Keyboard: ← → cycle features by strength, [ ] cycle examples.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (typing(e) || e.metaKey || e.ctrlKey || e.altKey || !exp) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        const fs = exp.features;
        const i = sel ? fs.findIndex((f) => fkey(f) === fkey(sel)) : -1;
        const next = fs[(i + (e.key === "ArrowRight" ? 1 : -1) + fs.length) % fs.length];
        if (next) { e.preventDefault(); setSel(next); }
      } else if (e.key === "]" || e.key === "[") {
        const base = active >= 0 ? active : 0;
        pick((base + (e.key === "]" ? 1 : -1) + EXAMPLES.length) % EXAMPLES.length);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [exp, sel, active, pick]);

  return (
    <div>
      <Hero summary={summary} />

      <section className="border-t pt-5 rule">
        <div className="flex flex-wrap items-baseline justify-between gap-2">
          <div className="kicker">Choose a specimen{STATIC ? "" : " or write your own"}</div>
          {STATIC && (
            <a className="text-[12px] text-ink2 underline decoration-[var(--baseline)] hover:text-ink" href={`${REPO}#quickstart`} target="_blank" rel="noreferrer">
              Static demo with precomputed GPT-2 runs · run locally for any prompt
            </a>
          )}
        </div>
        <div className="mt-2"><Specimens active={active} onPick={pick} disabled={busy} /></div>
        {!STATIC && (
          <div className="mt-3 flex flex-col gap-2 sm:flex-row">
            <label htmlFor="prompt" className="sr-only">Prompt for GPT-2 small</label>
            <textarea id="prompt" rows={1} value={prompt} onChange={(e) => setPrompt(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); go(prompt); } }}
              className="tok min-h-[2.5rem] flex-1 resize-y rounded-[5px] bg-[var(--surface-1)] px-3 py-2 text-[13.5px] text-ink outline-none"
              style={{ boxShadow: "inset 0 0 0 1px var(--baseline)" }} maxLength={240} />
            <Button onClick={() => go(prompt)} disabled={busy || !prompt.trim()}>{busy ? "Tracing…" : <>Trace <Glyph name="arrow" size={12} /></>}</Button>
          </div>
        )}
      </section>

      <div className="mt-5"><ErrorNote error={error} /></div>

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
                subtitle={<>The transcoder features with the largest direct effect<Info>{GLOSSARY.directEffect}</Info> on the prediction, by token and layer. Hover to isolate one, click to test it.</>}>
                <div ref={graphBox} className="-mx-1 overflow-x-auto px-1">
                  <ReasoningGraph tokens={exp.tokens} target={exp.target} features={exp.features} selected={sel} onSelect={setSel} />
                </div>
                <p className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-[11.5px] text-muted">
                  <span><kbd>←</kbd> <kbd>→</kbd> step through features</span>
                  <span><kbd>[</kbd> <kbd>]</kbd> change specimen</span>
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
              <Card label="Fig. 3" title="What GPT-2 predicts" subtitle={<>Next-token distribution after “<span className="tok whitespace-pre-wrap">{tokenText(ran).slice(-36)}</span>”</>}>
                <TokenBars items={run.top} />
              </Card>
            </div>
          </div>
          <div className="grid items-start gap-5 md:grid-cols-[1fr_1.4fr]">
            <Card label="Fig. 4" title={<>What the twin cannot explain<Info>{GLOSSARY.error}</Info></>}
              subtitle="Share of each layer's MLP output at the last token that the transcoder does not reconstruct">
              <ErrorStrip exp={exp} />
            </Card>
            <Generation prompt={ran} />
          </div>
        </div>
      )}
    </div>
  );
}
