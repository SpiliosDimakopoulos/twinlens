import type { Summary } from "../../api";
import { useTooltip } from "../../components/useTooltip";
import { familyLabel } from "../../lib";

// Geometry, in viewBox units.
const W = 560;      // viewBox width
const L = 92;       // left column for the family names
const R = 16;       // right margin
const ROW_H = 26;   // height of one family row
const TOP = 8;      // top margin
const AXIS_H = 26;  // tick labels below the rows
const X_MAX = 0.65; // right end of the ρ axis, just above the best family

/** Rank correlation per task family: today's graph -> with the fix, ring = nonlinear twin. */
export function Families({ fix }: { fix: NonNullable<Summary["fix"]> }) {
  const { hover } = useTooltip();
  const fams = Object.entries(fix.by_category_spearman)
    .flatMap(([f, m]) =>
      m.linear !== undefined && m.linear_fulljac !== undefined && m.replacement_full !== undefined
        ? [{ f, a: m.linear, b: m.linear_fulljac, r: m.replacement_full }]
        : [])
    .sort((p, q) => q.a - p.a);
  const x = (v: number) => L + (Math.max(0, v) / X_MAX) * (W - L - R);
  const H = TOP + fams.length * ROW_H + AXIS_H;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Rank correlation per task family, graph today vs graph with the fix">
      {[0, 0.2, 0.4, 0.6].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={TOP} y2={H - 22} stroke="var(--gridline)" />
          <text x={x(t)} y={H - 8} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="tnum">{t.toFixed(1)}</text>
        </g>
      ))}
      {fams.map(({ f, a, b, r }, i) => {
        const cy = TOP + i * ROW_H + ROW_H / 2;
        return (
          <g key={f} {...hover(<div><b>{familyLabel(f)}</b>: graph today ρ = {a.toFixed(2)} → with the fix {b.toFixed(2)} <span className="text-ink2">(nonlinear twin {r.toFixed(2)})</span></div>)}>
            <rect x={0} y={cy - ROW_H / 2} width={W} height={ROW_H} fill="transparent" />
            <text x={L - 10} y={cy + 4} fontSize={11.5} textAnchor="end" fill="var(--text-primary)">{familyLabel(f)}</text>
            <line x1={x(a)} x2={x(b)} y1={cy} y2={cy} stroke="var(--fix)" strokeOpacity={0.3} strokeWidth={3} strokeLinecap="round" />
            <circle cx={x(r)} cy={cy} r={6.5} fill="none" stroke="var(--text-secondary)" strokeWidth={1.2} />
            <circle cx={x(a)} cy={cy} r={4} fill="var(--baseline)" />
            <circle cx={x(b)} cy={cy} r={4.5} fill="var(--fix)" stroke="var(--surface-1)" strokeWidth={1.5} />
          </g>
        );
      })}
    </svg>
  );
}
