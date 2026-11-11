import type { Summary } from "../../api";
import { useTooltip } from "../../components/useTooltip";
import { LADDER } from "./rows";

// Geometry, in viewBox units.
const W = 420;         // viewBox width
const H = 230;         // baseline of the bin labels; the viewBox adds 6 below for descenders
const L = 30;          // left column for the y-axis labels
const R = 122;         // right column for the line labels
const BOTTOM = 24;     // space below the plot for the bin labels
const PLOT_H = H - 44; // plot height: BOTTOM plus 20 of headroom above cosine 1
const LABEL_GAP = 14;  // minimum vertical distance between two line labels

/** Median cosine between predicted and real change, by distance from the edit to the output. */
export function Decay({ s }: { s: Summary }) {
  const { hover } = useTooltip();
  const first = s.levels[0];
  const bins = (first ? s.decay[first] : undefined)?.map((d) => d.layers_after) ?? [];
  const x = (i: number) => L + (i / (bins.length - 1)) * (W - L - R);
  const y = (v: number) => H - BOTTOM - v * PLOT_H;
  const series = LADDER.flatMap((r) => {
    const d = s.decay[r.key];
    return d ? [{ r, d }] : [];
  });
  // Line labels at the right end, nudged apart (bottom-up) so they never overlap.
  let prev = Infinity;
  const ends = series
    .map(({ r, d }) => ({ r, y: y(Math.max(0, d[d.length - 1]?.median_cos ?? 0)) }))
    .sort((a, b) => b.y - a.y)
    .map((e) => {
      prev = Math.min(e.y, prev - LABEL_GAP);
      return { ...e, y: prev };
    });
  return (
    <svg viewBox={`0 0 ${W} ${H + 6}`} width="100%" role="img" aria-label="Median cosine by depth for each twin level">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--gridline)" />
          <text x={L - 8} y={y(g) + 3.5} fontSize={10} textAnchor="end" fill="var(--text-muted)" className="tnum">{g}</text>
        </g>
      ))}
      {bins.map((b, i) => <text key={b} x={x(i)} y={H} fontSize={10.5} textAnchor="middle" fill="var(--text-muted)">{b} layers</text>)}
      {series.map(({ r, d }) => {
        const pts = d.map((bin, i) => ({ px: x(i), py: y(Math.max(0, bin.median_cos)), cos: bin.median_cos }));
        return (
          <g key={r.key}>
            <polyline points={pts.map((p) => `${p.px},${p.py}`).join(" ")} fill="none" stroke={r.color} strokeWidth={r.key === "linear" ? 2.5 : 1.75} />
            {pts.map(({ px, py, cos }, i) => (
              <circle key={i} cx={px} cy={py} r={3.5} fill={r.color} stroke="var(--surface-1)" strokeWidth={1.5}
                {...hover(<div><b>{r.label}</b>, {bins[i]} layers to the output: median cosine {cos.toFixed(2)}</div>)} />
            ))}
          </g>
        );
      })}
      {ends.map(({ r, y: ly }) => (
        <text key={r.key} x={x(bins.length - 1) + 10} y={ly + 3.5} fontSize={10.5} fill="var(--text-secondary)" fontWeight={r.key === "linear" ? 600 : 400}>{r.label}</text>
      ))}
    </svg>
  );
}
