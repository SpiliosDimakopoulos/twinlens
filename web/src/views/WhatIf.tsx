import { useEffect, useState } from "react";
import { api, type Feature, type WhatIfResponse } from "../api";
import { LADDER_VARS } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { tokenText, trustBand } from "../lib";

const LEVEL_LABEL: Record<string, string> = {
  direct: "Direct edges",
  linear: "Linear (graph)",
  replacement: "Replacement",
  replacement_ln: "+ live LayerNorm",
  replacement_full: "+ live attention",
  real: "Real GPT-2",
};

function TrustMeter({ trust }: { trust: number | null }) {
  if (trust === null)
    return <p className="text-xs text-muted">Trust model not trained yet: run the fidelity sweep.</p>;
  const b = trustBand(trust);
  const band = [b.status, b.icon, b.label];
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-ink2">Twin trust score (predicted before running GPT-2)</span>
        <span className="text-2xl font-semibold tnum">{Math.round(trust * 100)}%</span>
      </div>
      <div className="mt-2 h-2 w-full rounded-full" style={{ background: "var(--gridline)" }}>
        <div className="h-2 rounded-full" style={{ width: `${trust * 100}%`, background: `var(--${band[0]})` }} />
      </div>
      <div className="mt-2 flex items-center gap-2 text-sm">
        <span aria-hidden className="inline-flex h-5 w-5 items-center justify-center rounded-full text-xs font-bold"
          style={{ background: `var(--${band[0]})`, color: "#111" }}>{band[1]}</span>
        <span>{band[2]}</span>
      </div>
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
    <div className="space-y-1.5">
      {order.map((k, i) => {
        const v = data.levels[k].dlogp;
        const w = (Math.abs(v) / max) * 50;
        const isReal = k === "real";
        return (
          <div key={k} className="grid grid-cols-[8.5rem_1fr_4.5rem] items-center gap-2 text-xs"
            onMouseMove={(e) => show(e, (
              <div>
                <b>{LEVEL_LABEL[k]}</b>: Δ log p(“{tokenText(data.target)}”) = {v.toFixed(3)}
                {!isReal && <div className="text-ink2">error vs real: {Math.abs(v - real).toFixed(3)} nats</div>}
                <div className="text-ink2">new top token: “{tokenText(data.levels[k].top[0].token)}”</div>
              </div>
            ))}
            onMouseLeave={hide}>
            <span className={isReal ? "font-semibold text-ink" : "text-ink2"}>{LEVEL_LABEL[k]}</span>
            <div className="relative h-4">
              <div className="absolute inset-y-0 left-1/2 w-px" style={{ background: "var(--baseline)" }} />
              <div className="absolute top-1 h-2" style={{
                left: v < 0 ? `${50 - w}%` : "50%", width: `${Math.max(w, 0.4)}%`,
                background: isReal ? "var(--real)" : `var(${LADDER_VARS[i]})`,
                borderRadius: v < 0 ? "4px 0 0 4px" : "0 4px 4px 0",
              }} />
              {isReal ? null : (
                <div className="absolute top-0 h-4 w-0.5" title="real effect"
                  style={{ left: `${50 + (real / max) * 50}%`, background: "var(--real)" }} />
              )}
            </div>
            <span className="tnum text-right">{v >= 0 ? "+" : ""}{v.toFixed(3)}</span>
          </div>
        );
      })}
      <p className="pt-1 text-xs text-muted">Bars: predicted change in log-probability of the original prediction. The thin black tick on each twin row marks what GPT-2 really did.</p>
    </div>
  );
}

export function WhatIf({ prompt, feature, tokens }: { prompt: string; feature: Feature | null; tokens: string[] }) {
  const [mult, setMult] = useState(0);
  const [res, setRes] = useState<WhatIfResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => { setRes(null); setMult(0); }, [feature]);

  if (!feature)
    return (
      <Card title="What if?" subtitle="Pick a feature in the reasoning graph.">
        <p className="text-sm text-ink2">The twin will predict what changes if you turn that feature off or up, at five levels of approximation, and then GPT-2 itself will answer, so you can see how far the explanation holds.</p>
      </Card>
    );

  const run = async () => {
    setBusy(true); setError(null);
    try { setRes(await api.whatif(prompt, feature.layer, feature.pos, feature.feature, feature.activation * mult)); }
    catch (e) { setError(String(e)); }
    finally { setBusy(false); }
  };

  return (
    <Card title={`What if: L${feature.layer} · feature ${feature.feature}`}
      subtitle={<>on “<span className="tok">{tokenText(tokens[feature.pos])}</span>” · promotes <span className="tok">{feature.label.slice(0, 4).map(tokenText).join(" · ")}</span></>}>
      <div className="flex flex-wrap items-center gap-3">
        <label className="text-xs text-ink2" htmlFor="mult">Set activation to</label>
        <input id="mult" type="range" min={0} max={3} step={0.25} value={mult}
          onChange={(e) => setMult(Number(e.target.value))} className="w-40 accent-[var(--series-1)]" />
        <span className="tnum text-sm">{mult === 0 ? "0 (ablate)" : `${mult}× (${(feature.activation * mult).toFixed(2)})`}</span>
        <Button onClick={run} disabled={busy}>{busy ? "Simulating…" : "Ask the twin, then GPT-2"}</Button>
      </div>
      <ErrorNote error={error} />
      {res && (
        <div className="mt-4 grid gap-5 md:grid-cols-[1.4fr_1fr]">
          <LadderEffects data={res} />
          <div className="space-y-4">
            <TrustMeter trust={res.trust} />
            <div>
              <div className="mb-1.5 text-xs text-ink2">GPT-2 after the edit</div>
              <TokenBars items={res.levels.real.top} color="var(--text-secondary)" />
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
