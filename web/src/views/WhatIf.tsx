import { useEffect, useRef, useState } from "react";
import { api, STATIC, type Feature, type WhatIfResponse } from "../api";
import { Glyph } from "../components/Glyph";
import { Info } from "../components/Info";
import { Skeleton } from "../components/Skeleton";
import { LADDER_VARS } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { Card, ErrorNote, TokenBars } from "../components/ui";
import { GLOSSARY } from "../glossary";
import { SIGNAL_LABELS, tokenText, trustBand, verdict } from "../lib";

export const PRESETS: [string, number][] = [["Ablate", 0], ["½×", 0.5], ["2×", 2], ["3×", 3]];

type Row = { key: string; label: string; color: string; strong?: boolean; fix?: boolean };
const ROWS: Row[] = [
  { key: "linear", label: "Graph today", color: `var(${LADDER_VARS[1]})`, strong: true },
  { key: "direct", label: "Direct edges only", color: `var(${LADDER_VARS[0]})` },
  { key: "replacement", label: "Replacement", color: `var(${LADDER_VARS[2]})` },
  { key: "replacement_ln", label: "+ live LayerNorm", color: `var(${LADDER_VARS[3]})` },
  { key: "replacement_full", label: "+ live attention", color: `var(${LADDER_VARS[4]})` },
  { key: "linear_fulljac", label: "Graph + Jacobians", color: "var(--fix)", strong: true, fix: true },
];

/** The gap plot: every twin's prediction as a dot on one axis, GPT-2's answer as a rule. */
function GapPlot({ data }: { data: WhatIfResponse }) {
  const { show, hide } = useTooltip();
  const rows = ROWS.filter((r) => data.levels[r.key]);
  const real = data.levels.real.dlogp;
  const vals = [...rows.map((r) => data.levels[r.key].dlogp), real, 0];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const span = Math.max(hi - lo, 0.05);
  lo -= span * 0.08; hi += span * 0.08;
  const W = 520, L = 128, R = 64, top = 26, rowH = 26, fixGap = 16;
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const y = (i: number, r: Row) => top + i * rowH + (r.fix ? fixGap : 0) + rowH / 2;
  const H = top + rows.length * rowH + fixGap + 22;
  const ticks = niceTicks(lo, hi, 4);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Predicted change at each twin level vs GPT-2's actual change of ${real.toFixed(2)}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top - 4} y2={H - 20} stroke="var(--gridline)" />
          <text x={x(t)} y={H - 6} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="tnum">{t > 0 ? "+" : ""}{fmtTick(t)}</text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={top - 4} y2={H - 20} stroke="var(--baseline)" strokeDasharray="2 3" />
      <line x1={x(real)} x2={x(real)} y1={top - 10} y2={H - 20} stroke="var(--real)" strokeWidth={1.5} />
      <text x={x(real)} y={top - 14} fontSize={10.5} textAnchor="middle" fill="var(--text-primary)" fontWeight={600}>GPT-2 {fmt(real)}</text>
      {rows.map((r, i) => {
        const v = data.levels[r.key].dlogp;
        const cy = y(i, r);
        const gap = Math.abs(v - real);
        return (
          <g key={r.key}
            onMouseMove={(e) => show(e, (
              <div className="space-y-0.5">
                <div className="font-semibold">{r.label}</div>
                <div className="tnum">predicts Δ log p = {fmt(v)}</div>
                <div className="tnum text-ink2">off by {gap.toFixed(3)} nats from GPT-2</div>
                <div className="text-ink2">its new top token: <span className="tok">“{tokenText(data.levels[r.key].top[0].token)}”</span></div>
              </div>
            ))}
            onMouseLeave={hide}>
            <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
            {r.fix && i > 0 && (
              <text x={0} y={cy - rowH / 2 - 3} fontSize={9.5} fill="var(--fix)" className="figlabel" style={{ letterSpacing: "0.06em" }}>PROPOSED FIX · STILL LINEAR</text>
            )}
            <text x={0} y={cy + 3.5} fontSize={12} fill={r.strong ? "var(--text-primary)" : "var(--text-secondary)"} fontWeight={r.strong ? 600 : 400}>{r.label}</text>
            <line x1={x(v)} x2={x(real)} y1={cy} y2={cy} stroke={r.color} strokeOpacity={0.35} strokeWidth={r.strong ? 3 : 2} strokeLinecap="round" />
            <circle cx={x(v)} cy={cy} r={r.strong ? 5.5 : 4.5} fill={r.color} stroke="var(--surface-1)" strokeWidth={2} />
            <text x={W} y={cy + 3.5} fontSize={11.5} textAnchor="end" className="tnum" fill={r.strong ? "var(--text-primary)" : "var(--text-secondary)"}>{fmt(v)}</text>
          </g>
        );
      })}
    </svg>
  );
}

const fmt = (v: number) => `${v >= 0 ? "+" : "−"}${Math.abs(v) < 0.1 ? Math.abs(v).toFixed(3) : Math.abs(v).toFixed(2)}`;
const fmtTick = (v: number) => (Math.abs(v) < 1 ? v.toFixed(2).replace("-", "−") : v.toFixed(0).replace("-", "−"));
function niceTicks(lo: number, hi: number, n: number): number[] {
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.abs(t) < 1e-9 ? 0 : t);
  return out;
}

function TrustPanel({ data }: { data: WhatIfResponse }) {
  if (data.trust === null) return <p className="text-xs text-muted">Trust model not trained yet: run the fidelity sweep.</p>;
  const b = trustBand(data.trust);
  const glyph = b.status === "good" ? "check" : b.status === "warning" ? "alert" : "cross";
  return (
    <div>
      <div className="kicker flex items-center">Should you trust the graph here?<Info>{GLOSSARY.trust}</Info></div>
      <div className="mt-2 flex items-end gap-3">
        <span className="serif text-[44px] leading-none tracking-tight tnum">{Math.round(data.trust * 100)}<span className="text-[24px] text-ink2">%</span></span>
        <span className="mb-1 inline-flex items-center gap-1.5 text-[13px] font-medium">
          <span className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-full" style={{ background: `var(--${b.status})`, color: "#111" }}>
            <Glyph name={glyph} size={12} />
          </span>
          {b.label}
        </span>
      </div>
      <div className="relative mt-3 h-[3px] w-full rounded-full" style={{ background: "var(--gridline)" }} role="meter"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(data.trust * 100)} aria-label="Trust score">
        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-500" style={{ width: `${data.trust * 100}%`, background: `var(--${b.status})` }} />
        {[40, 70].map((t) => <div key={t} className="absolute -top-1 h-[11px] w-px" style={{ left: `${t}%`, background: "var(--baseline)" }} />)}
      </div>
      {data.trust_factors.length > 0 && (
        <ul className="mt-3 space-y-1 text-[12.5px]">
          {data.trust_factors.map((f) => (
            <li key={f.signal} className="flex items-center gap-2 text-ink2">
              <Glyph name={f.contribution >= 0 ? "up" : "down"} size={12}
                className={f.contribution >= 0 ? "text-[var(--good)]" : "text-[var(--critical)]"} />
              <span><span className="text-ink">{SIGNAL_LABELS[f.signal] ?? f.signal}</span> {f.contribution >= 0 ? "raises" : "lowers"} it</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-[11.5px] text-muted">Computed from the clean run alone, before GPT-2 sees the edit.</p>
    </div>
  );
}

export function WhatIf({ prompt, feature, tokens, mult, setMult, label = "Fig. 2" }: {
  prompt: string; feature: Feature | null; tokens: string[]; mult: number; setMult: (m: number) => void; label?: string;
}) {
  const [res, setRes] = useState<WhatIfResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const req = useRef(0);

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
    }, STATIC ? 0 : 250);
    return () => clearTimeout(timer);
  }, [prompt, feature, mult]);

  if (!feature)
    return (
      <Card label={label} title="What if?" subtitle="Click any node in the reasoning path.">
        <p className="text-sm text-ink2">The twin predicts what changes if you turn that feature off or up. Then GPT-2 itself answers.</p>
      </Card>
    );

  const verb = mult === 0 ? "remove" : mult < 1 ? "halve" : mult === 2 ? "double" : `scale ×${mult}`;
  return (
    <Card label={label}
      title={<>What if we {verb} this feature?</>}
      subtitle={<>L{feature.layer} · #{feature.feature}, firing on “<span className="tok">{tokenText(tokens[feature.pos])}</span>”, promotes <span className="tok">{feature.label.slice(0, 3).map(tokenText).join(" · ")}</span></>}
      right={busy ? <span className="kicker animate-pulse" aria-live="polite">simulating…</span> : null}>
      <div className="flex flex-wrap items-center gap-3">
        <div className="inline-flex rounded-[5px] p-0.5" style={{ boxShadow: "inset 0 0 0 1px var(--baseline)" }} role="group" aria-label="Edit strength">
          {PRESETS.map(([l, m]) => (
            <button key={l} onClick={() => setMult(m)} aria-pressed={mult === m}
              className="rounded-[4px] px-2.5 py-1 text-[12.5px] font-medium transition-colors"
              style={mult === m ? { background: "var(--text-primary)", color: "var(--page)" } : { color: "var(--text-secondary)" }}>
              {l}
            </button>
          ))}
        </div>
        {!STATIC && (
          <>
            <label className="sr-only" htmlFor="mult">Activation multiplier</label>
            <input id="mult" type="range" min={0} max={3} step={0.25} value={mult}
              onChange={(e) => setMult(Number(e.target.value))} className="w-24 accent-[var(--text-primary)]" />
          </>
        )}
        <span className="tnum text-[12px] text-muted">activation {feature.activation.toFixed(2)} → {(feature.activation * mult).toFixed(2)}</span>
      </div>
      <ErrorNote error={error} />
      {!res ? (
        <div className="mt-4 space-y-2"><Skeleton className="h-5 w-4/5" /><Skeleton className="h-40" /><Skeleton className="h-20" /></div>
      ) : (
        <div className={`mt-4 transition-opacity ${busy ? "opacity-55" : ""}`}>
          <blockquote className="serif border-l-2 pl-3 text-[17px] leading-snug text-ink rule">
            {verdict(res.levels.linear.dlogp, res.levels.real.dlogp)}
            {res.levels.linear_fulljac && (
              <span className="mt-1 block text-[14px] text-ink2">
                With the fix, the graph is off by {Math.abs(res.levels.linear_fulljac.dlogp - res.levels.real.dlogp).toFixed(2)} instead of {Math.abs(res.levels.linear.dlogp - res.levels.real.dlogp).toFixed(2)}.
              </span>
            )}
          </blockquote>
          <div className="mt-4"><GapPlot data={res} /></div>
          <p className="mt-1 text-[11.5px] text-muted">Change in log-probability of “<span className="tok">{tokenText(res.target)}</span>”. Each dot is a twin's prediction; the line it trails is its distance from what GPT-2 really did.</p>
          <div className="mt-5 grid gap-5 border-t pt-4 rule">
            <TrustPanel data={res} />
            <div>
              <div className="kicker mb-2">GPT-2's next token after the edit</div>
              <TokenBars items={res.levels.real.top} color="var(--text-secondary)" />
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
