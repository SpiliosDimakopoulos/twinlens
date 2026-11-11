import type { PairedMetric, Summary } from "../../api";
import { useTooltip } from "../../components/useTooltip";
import { signedFixed } from "../../format";

/** Rows: metric key, label, and whether it is a share (shown in percentage points). */
const GEN_METRICS: [PairedMetric, string, boolean][] = [
  ["spearman", "Δ rank correlation ρ", false],
  ["sign_agree", "Δ sign agreement", true],
  ["within_tol_big", "Δ within tolerance (edits that matter)", true],
  ["catastrophic", "Δ catastrophic misses (lower is better)", true],
];
const COLORS = ["var(--lv-5)", "var(--series-2)"];

/** Paired improvement of the fix over today's graph on each model, with 95% CIs. */
export function Generalization({ g }: { g: NonNullable<Summary["generalization"]> }) {
  const { hover } = useTooltip();
  const models = Object.entries(g);
  return (
    <div className="space-y-4">
      {GEN_METRICS.map(([m, label, pct]) => {
        const all = models.flatMap(([, d]) => d.paired_diff[m]);
        const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
        const X = (v: number) => `${((v - lo) / (hi - lo || 1)) * 100}%`;
        const f = (v: number) => (pct ? `${signedFixed(v * 100, 1)} pp` : signedFixed(v, 2));
        return (
          <div key={m}>
            <div className="kicker mb-1.5">{label}</div>
            {models.map(([k, d], i) => {
              const [v, l, h] = d.paired_diff[m];
              const color = COLORS[i];
              return (
                <div key={k} className="grid grid-cols-[7.5rem_1fr_4.5rem] items-center gap-3 py-0.5 text-[12.5px]"
                  {...hover(<div><b>{d.label}</b>: {f(v)} <span className="text-ink2">(95% CI {f(l)} to {f(h)}; {d.n_edits.toLocaleString()} edits)</span></div>)}>
                  <span className="text-ink2">{d.label}</span>
                  <div className="relative h-3">
                    <div className="track absolute inset-y-[5px] left-0 right-0" />
                    <div className="absolute inset-y-0 w-px" style={{ left: X(0), background: "var(--text-muted)" }} />
                    <div className="absolute top-[5px] h-[2px]" style={{ left: X(l), width: `calc(${X(h)} - ${X(l)})`, background: color }} />
                    <div className="absolute top-[1px] h-[10px] w-[10px] -translate-x-1/2 rounded-full" style={{ left: X(v), background: color, boxShadow: "0 0 0 2px var(--surface-1)" }} />
                  </div>
                  <span className="tnum text-right text-ink">{f(v)}</span>
                </div>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
