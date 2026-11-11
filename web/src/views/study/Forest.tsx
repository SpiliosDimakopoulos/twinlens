import type { Ci, LevelMap, Metric } from "../../api";
import { Info } from "../../components/Info";
import { useTooltip } from "../../components/useTooltip";
import { adaptivePercent } from "../../format";
import { GLOSSARY } from "../../glossary";
import type { LevelRow } from "./rows";

/** Columns: metric key, header, and whether the value is a share (shown as a percentage). */
const METRICS: [Metric, string, boolean][] = [
  ["spearman", "Rank correlation ρ ↑", false],
  ["median_cos", "Median cosine ↑", false],
  ["sign_agree", "Right direction ↑", true],
  ["catastrophic", "Badly wrong ↓", true],
];

const value = (v: number, pct: boolean) => (pct ? adaptivePercent(v) : v.toFixed(2));

/** Forest plot: one row per level, one mini-axis per metric, dot = estimate, whisker = 95% CI. */
export function Forest({ rows, data }: { rows: LevelRow[]; data: LevelMap<Record<Metric, Ci>> }) {
  const { hover } = useTooltip();
  const present = rows.flatMap((r) => {
    const cis = data[r.key];
    return cis ? [{ ...r, cis }] : [];
  });
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-[12.5px]">
        <thead>
          <tr>
            <th className="w-[190px] pb-2 text-left font-normal"><span className="kicker inline-flex items-center">Metrics<Info wide label="What do these metrics mean?">{GLOSSARY.metrics}</Info></span></th>
            {METRICS.map(([, label]) => <th key={label} className="kicker pb-2 text-left font-normal">{label}</th>)}
          </tr>
        </thead>
        <tbody>
          {present.map((r) => (
            <tr key={r.key} className="border-t rule">
              <td className="py-2 pr-3">
                <span className="mr-2 inline-block h-2 w-2 rounded-full align-middle" style={{ background: r.color }} />
                <span className="align-middle text-ink">{r.label}</span>
              </td>
              {METRICS.map(([m, , pct]) => {
                // Shared axis per column: 0 to just past the widest upper bound.
                const max = Math.max(...present.map((x) => x.cis[m][2]), 1e-6) * 1.05;
                const [v, lo, hi] = r.cis[m];
                const X = (t: number) => `${(t / max) * 100}%`;
                return (
                  <td key={m} className="py-2 pr-4"
                    {...hover(<div><b>{r.label}</b>: {value(v, pct)} <span className="text-ink2">(95% CI {value(lo, pct)} to {value(hi, pct)})</span></div>)}>
                    <div className="flex items-center gap-2">
                      <div className="relative h-3 flex-1">
                        <div className="track absolute inset-y-[5px] left-0 right-0" />
                        <div className="absolute top-[5px] h-[2px]" style={{ left: X(lo), width: `calc(${X(hi)} - ${X(lo)})`, background: r.color, opacity: 0.5 }} />
                        <div className="absolute top-[1px] h-[10px] w-[10px] -translate-x-1/2 rounded-full" style={{ left: X(v), background: r.color, boxShadow: "0 0 0 2px var(--surface-1)" }} />
                      </div>
                      <span className="w-12 text-right tnum text-ink2">{value(v, pct)}</span>
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
