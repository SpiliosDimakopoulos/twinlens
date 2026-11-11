import type { WhatIfResponse } from "../../api";
import { useTooltip } from "../../components/useTooltip";
import { signed, trueMinus } from "../../format";
import { tokenText } from "../../lib";
import { presentRows } from "./rows";

// Geometry, in viewBox units.
const W = 520;       // viewBox width
const L = 128;       // left column for the row labels
const R = 64;        // right column for the predicted values
const TOP = 26;      // headroom for the "GPT-2 actually" label above the rows
const ROW_H = 26;    // height of one row
const FIX_GAP = 16;  // extra space above the fix row for its "PROPOSED FIX" caption
const AXIS_H = 22;   // tick labels below the rows
const PAD = 0.08;    // share of the value range added on each side
const MIN_SPAN = 0.05; // smallest value range, so near-zero effects are not blown up

const fmtTick = (v: number) => trueMinus(v.toFixed(Math.abs(v) < 1 ? 2 : 0));

function niceTicks(lo: number, hi: number, n: number): number[] {
  const step0 = (hi - lo) / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((s) => s >= step0) ?? step0;
  const out = [];
  for (let t = Math.ceil(lo / step) * step; t <= hi + 1e-9; t += step) out.push(Math.abs(t) < 1e-9 ? 0 : t);
  return out;
}

/** The gap plot: every twin's prediction as a dot on one axis, GPT-2's answer as a rule. */
export function GapPlot({ data }: { data: WhatIfResponse }) {
  const { hover } = useTooltip();
  const rows = presentRows(data.levels);
  const real = data.levels.real.dlogp;
  const vals = [...rows.map((r) => r.result.dlogp), real, 0];
  let lo = Math.min(...vals), hi = Math.max(...vals);
  const span = Math.max(hi - lo, MIN_SPAN);
  lo -= span * PAD; hi += span * PAD;
  const x = (v: number) => L + ((v - lo) / (hi - lo)) * (W - L - R);
  const y = (i: number, fix?: boolean) => TOP + i * ROW_H + (fix ? FIX_GAP : 0) + ROW_H / 2;
  const H = TOP + rows.length * ROW_H + FIX_GAP + AXIS_H;
  const ticks = niceTicks(lo, hi, 4);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Predicted change at each twin level vs GPT-2's actual change of ${real.toFixed(2)}`}>
      {ticks.map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={TOP - 4} y2={H - 20} stroke="var(--gridline)" />
          <text x={x(t)} y={H - 6} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="tnum">{t > 0 ? "+" : ""}{fmtTick(t)}</text>
        </g>
      ))}
      <line x1={x(0)} x2={x(0)} y1={TOP - 4} y2={H - 20} stroke="var(--baseline)" strokeDasharray="2 3" />
      <line x1={x(real)} x2={x(real)} y1={TOP - 10} y2={H - 20} stroke="var(--real)" strokeWidth={1.5} />
      <text x={x(real)} y={TOP - 14} fontSize={10.5} textAnchor="middle" fill="var(--text-primary)" fontWeight={600}>GPT-2 actually: {signed(real)}</text>
      {rows.map((r, i) => {
        const v = r.result.dlogp;
        const cy = y(i, r.fix);
        const gap = Math.abs(v - real);
        return (
          <g key={r.key}
            {...hover(
              <div className="space-y-0.5">
                <div className="font-semibold">{r.label}</div>
                <div className="max-w-[16rem] text-ink2">{r.desc}</div>
                <div className="tnum">predicts a change of {signed(v)}</div>
                <div className="tnum text-ink2">off by {gap.toFixed(3)} from what GPT-2 did</div>
                <div className="text-ink2">its new top token: <span className="tok">“{tokenText(r.result.top[0]?.token ?? "")}”</span></div>
              </div>,
            )}>
            <rect x={0} y={cy - ROW_H / 2} width={W} height={ROW_H} fill="transparent" />
            {r.fix && i > 0 && (
              <text x={0} y={cy - ROW_H / 2 - 3} fontSize={9.5} fill="var(--fix)" className="figlabel" style={{ letterSpacing: "0.06em" }}>PROPOSED FIX · STILL LINEAR</text>
            )}
            <text x={0} y={cy + 3.5} fontSize={12} fill={r.strong ? "var(--text-primary)" : "var(--text-secondary)"} fontWeight={r.strong ? 600 : 400}>{r.label}</text>
            <line x1={x(v)} x2={x(real)} y1={cy} y2={cy} stroke={r.color} strokeOpacity={0.35} strokeWidth={r.strong ? 3 : 2} strokeLinecap="round" />
            <circle cx={x(v)} cy={cy} r={r.strong ? 5.5 : 4.5} fill={r.color} stroke="var(--surface-1)" strokeWidth={2} />
            <text x={W} y={cy + 3.5} fontSize={11.5} textAnchor="end" className="tnum" fill={r.strong ? "var(--text-primary)" : "var(--text-secondary)"}>{signed(v)}</text>
          </g>
        );
      })}
    </svg>
  );
}
