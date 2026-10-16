import { useEffect, useRef, useState } from "react";
import { api, type Feature, type WhatIfResponse } from "../api";
import { Info } from "../components/Info";
import { GLOSSARY } from "../glossary";
import { Skeleton } from "../components/Skeleton";
import { LADDER_VARS } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { Card, ErrorNote, TokenBars } from "../components/ui";
import { SIGNAL_LABELS, tokenText, trustBand, verdict } from "../lib";

const LEVEL_LABEL: Record<string, string> = {
  direct: "Direct edges",
  linear: "Linear (graph)",
  replacement: "Replacement",
  replacement_ln: "+ live LayerNorm",
  replacement_full: "+ live attention",
  real: "Real GPT-2",
};

const PRESETS: [string, number][] = [["Ablate", 0], ["½×", 0.5], ["2×", 2], ["3×", 3]];

function TrustMeter({ data }: { data: WhatIfResponse }) {
  if (data.trust === null)
    return <p className="text-xs text-muted">Trust model not trained yet: run the fidelity sweep.</p>;
  const b = trustBand(data.trust);
  return (
    <div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-xs text-ink2">Trust in the graph's prediction<Info>{GLOSSARY.trust}</Info></span>
        <span className="text-3xl font-semibold tnum">{Math.round(data.trust * 100)}%</span>
      </div>
      <div className="mt-2 h-2 w-full rounded-full" style={{ background: "var(--gridline)" }} role="meter"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(data.trust * 100)} aria-label="Trust score">
        <div className="h-2 rounded-full transition-all duration-500" style={{ width: `${data.trust * 100}%`, background: `var(--${b.status})` }} />
      </div>
      <div className="mt-2 flex items-center gap-2 text-sm font-medium">
        <span aria-hidden className="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold"
          style={{ background: `var(--${b.status})`, color: "#111" }}>{b.icon}</span>
        {b.label}
      </div>
      {data.trust_factors.length > 0 && (
        <div className="mt-3">
          <div className="mb-1 text-xs text-ink2">Why, from the clean run alone:</div>
          <ul className="space-y-1 text-xs">
            {data.trust_factors.map((f) => (
              <li key={f.signal} className="flex items-center gap-2">
                <span aria-hidden className="w-4 text-center font-bold"
                  style={{ color: f.contribution >= 0 ? "var(--good)" : "var(--critical)" }}>
                  {f.contribution >= 0 ? "▲" : "▼"}
                </span>
                <span className="text-ink">{SIGNAL_LABELS[f.signal] ?? f.signal}</span>
                <span className="text-muted">{f.contribution >= 0 ? "raises" : "lowers"} trust</span>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
}

/** Diverging bars of Δ log p(target) per twin level, with the real model as the reference row. */
function LadderEffects({ data }: { data: WhatIfResponse }) {
  const { show, hide } = useTooltip();
  const order = [...Object.keys(LEVEL_LABEL).filter((k) => k !== "real" && data.levels[k]), "real"];
  const max = Math.max(...order.map((k) => Math.abs(data.levels[k].dlogp)), 0.05);
  const real = data.levels.real.dlogp;
  return (
    <div>
      <div className="mb-2 flex items-center justify-between text-[11px] text-muted">
        <span>← lowers the prediction</span><span>raises it →</span>
      </div>
      <div className="space-y-1.5">
        {order.map((k, i) => {
          const v = data.levels[k].dlogp;
          const w = (Math.abs(v) / max) * 50;
          const isReal = k === "real";
          return (
            <div key={k} className={`grid grid-cols-[7.5rem_1fr_3.75rem] items-center gap-2 text-xs ${isReal ? "border-t pt-1.5" : ""}`}
              style={isReal ? { borderColor: "var(--gridline)" } : undefined}
              onMouseMove={(e) => show(e, (
                <div>
                  <b>{LEVEL_LABEL[k]}</b>: Δ log p(“{tokenText(data.target)}”) = {v.toFixed(3)}
                  {!isReal && <div className="text-ink2">error vs GPT-2: {Math.abs(v - real).toFixed(3)} nats</div>}
                  <div className="text-ink2">new top token: “{tokenText(data.levels[k].top[0].token)}”</div>
                </div>
              ))}
              onMouseLeave={hide}>
              <span className={isReal ? "font-semibold text-ink" : k === "linear" ? "font-medium text-ink" : "text-ink2"}>{LEVEL_LABEL[k]}</span>
              <div className="relative h-4">
                <div className="absolute inset-y-0 left-1/2 w-px" style={{ background: "var(--baseline)" }} />
                <div className="absolute top-1 h-2 transition-all duration-500" style={{
                  left: v < 0 ? `${50 - w}%` : "50%", width: `${Math.max(w, 0.4)}%`,
                  background: isReal ? "var(--real)" : `var(${LADDER_VARS[i]})`,
                  borderRadius: v < 0 ? "4px 0 0 4px" : "0 4px 4px 0",
                }} />
                {!isReal && (
                  <div className="absolute top-0 h-4 w-0.5" style={{ left: `${50 + (real / max) * 50}%`, background: "var(--real)" }} />
                )}
              </div>
              <span className="tnum text-right">{v >= 0 ? "+" : ""}{v.toFixed(3)}</span>
            </div>
          );
        })}
      </div>
      <p className="mt-2 text-[11px] text-muted">Change in log-probability of the original prediction. The tick on each row marks what GPT-2 really did.</p>
    </div>
  );
}

export function WhatIf({ prompt, feature, tokens }: { prompt: string; feature: Feature | null; tokens: string[] }) {
  const [mult, setMult] = useState(0);
  const [res, setRes] = useState<WhatIfResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const req = useRef(0);

  useEffect(() => { setMult(0); }, [feature]);

  // Re-simulate whenever the feature or the edit changes (debounced for the slider).
  useEffect(() => {
    if (!feature) return;
    const id = ++req.current;
    setBusy(true); setError(null);
    const timer = setTimeout(async () => {
      try {
        const r = await api.whatif(prompt, feature.layer, feature.pos, feature.feature, feature.activation * mult);
        if (id === req.current) setRes(r);
      } catch (e) {
        if (id === req.current) setError(String(e));
      } finally {
        if (id === req.current) setBusy(false);
      }
    }, 300);
    return () => clearTimeout(timer);
  }, [prompt, feature, mult]);

  if (!feature)
    return (
      <Card title="What if?" subtitle="Click any node in the reasoning path.">
        <p className="text-sm text-ink2">The twin predicts what changes if you turn that feature off or up, at five levels of approximation. Then GPT-2 itself answers, so you can see how far the explanation holds.</p>
      </Card>
    );

  return (
    <Card
      title={`What if we edit L${feature.layer} · feature ${feature.feature}?`}
      subtitle={<>Fires on “<span className="tok">{tokenText(tokens[feature.pos])}</span>” · promotes <span className="tok">{feature.label.slice(0, 4).map(tokenText).join(" · ")}</span></>}
      right={busy ? <span className="text-xs text-muted animate-pulse" aria-live="polite">simulating…</span> : null}>
      <div className="flex flex-wrap items-center gap-2" role="group" aria-label="Edit strength">
        {PRESETS.map(([label, m]) => (
          <button key={label} onClick={() => setMult(m)} aria-pressed={mult === m}
            className="rounded-full px-3 py-1 text-xs font-medium transition"
            style={mult === m
              ? { background: "var(--text-primary)", color: "var(--surface-1)" }
              : { border: "1px solid var(--border)", color: "var(--text-secondary)" }}>
            {label}
          </button>
        ))}
        <label className="sr-only" htmlFor="mult">Activation multiplier</label>
        <input id="mult" type="range" min={0} max={3} step={0.25} value={mult}
          onChange={(e) => setMult(Number(e.target.value))} className="ml-1 w-28 accent-[var(--series-1)]" />
        <span className="tnum text-xs text-ink2">{(feature.activation * mult).toFixed(2)} <span className="text-muted">(was {feature.activation.toFixed(2)})</span></span>
      </div>
      <ErrorNote error={error} />
      {!res ? (
        <div className="mt-4 space-y-2"><Skeleton className="h-4 w-3/4" /><Skeleton className="h-24" /><Skeleton className="h-16" /></div>
      ) : (
        <div className={`mt-4 space-y-4 transition-opacity ${busy ? "opacity-60" : ""}`}>
          <p className="rounded-lg px-3 py-2 text-sm" style={{ background: "var(--page)" }}>
            {verdict(res.levels.linear.dlogp, res.levels.real.dlogp, res.target)}
          </p>
          <LadderEffects data={res} />
          <TrustMeter data={res} />
          <div>
            <div className="mb-1.5 text-xs text-ink2">GPT-2's next token after the edit</div>
            <TokenBars items={res.levels.real.top} color="var(--text-secondary)" />
          </div>
        </div>
      )}
    </Card>
  );
}
