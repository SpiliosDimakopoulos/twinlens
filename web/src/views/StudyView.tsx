import { useEffect, useState } from "react";
import { api, type Summary } from "../api";
import { LADDER_VARS } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { Card, ErrorNote, Stat } from "../components/ui";

const SIGNAL_NAMES: Record<string, string> = {
  layers_after: "Layers after edit",
  pos_from_end: "Distance to last token",
  activation: "Feature activation",
  rel_edit_size: "Relative edit size",
  abs_direct_effect: "|Direct effect| (graph edge)",
  downstream_err_frac: "Downstream transcoder error",
  attn_to_pos: "Attention to edited token",
  n_active_downstream: "Active features downstream",
  clean_top1_prob: "Model confidence",
};

/** One metric across the ladder: thin vertical bars (ordinal blue) with 95% CI whiskers. */
function LadderMetric({ s, metric, title, pct }: { s: Summary; metric: string; title: string; pct?: boolean }) {
  const { show, hide } = useTooltip();
  const vals = s.levels.map((lv) => s.ladder[lv][metric]);
  const max = Math.max(...vals.map((v) => v[2]), 1e-6);
  const H = 120, W = 300, bw = 36, gap = (W - bw * vals.length) / (vals.length + 1);
  const fmt = (v: number) => (pct ? `${(v * 100).toFixed(v < 0.1 ? 1 : 0)}%` : v.toFixed(2));
  return (
    <div>
      <div className="mb-1 text-xs font-medium text-ink">{title}</div>
      <svg viewBox={`0 0 ${W} ${H + 34}`} width="100%" role="img" aria-label={title}>
        <line x1={0} x2={W} y1={H} y2={H} stroke="var(--baseline)" />
        {vals.map((v, i) => {
          const x = gap + i * (bw + gap);
          const h = (Math.max(v[0], 0) / max) * (H - 16);
          const lo = H - (v[1] / max) * (H - 16), hi = H - (v[2] / max) * (H - 16);
          return (
            <g key={i} onMouseMove={(e) => show(e, <div><b>{s.level_labels[s.levels[i]]}</b>: {fmt(v[0])} <span className="text-ink2">(95% CI {fmt(v[1])}–{fmt(v[2])})</span></div>)} onMouseLeave={hide}>
              <rect x={x - 6} y={0} width={bw + 12} height={H} fill="transparent" />
              <path d={`M${x},${H} v${-Math.max(h - 4, 0)} q0,-4 4,-4 h${bw - 8} q4,0 4,4 v${Math.max(h - 4, 0)} z`} fill={`var(${LADDER_VARS[i]})`} />
              <line x1={x + bw / 2} x2={x + bw / 2} y1={lo} y2={hi} stroke="var(--text-secondary)" />
              <text x={x + bw / 2} y={hi - 4} fontSize={9} textAnchor="middle" fill="var(--text-secondary)" className="tnum">{fmt(v[0])}</text>
              <text x={x + bw / 2} y={H + 13} fontSize={8.5} textAnchor="middle" fill="var(--text-muted)">{i + 1}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function Decay({ s }: { s: Summary }) {
  const { show, hide } = useTooltip();
  const bins = s.decay[s.levels[0]].map((d) => d.layers_after);
  const W = 520, H = 200, L = 34, R = 130;
  const x = (i: number) => L + (i / (bins.length - 1)) * (W - L - R);
  const y = (v: number) => H - 20 - v * (H - 40);
  return (
    <svg viewBox={`0 0 ${W} ${H + 10}`} width="100%" role="img" aria-label="Median cosine by depth for each twin level">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--gridline)" />
          <text x={L - 6} y={y(g) + 3} fontSize={9} textAnchor="end" fill="var(--text-muted)" className="tnum">{g}</text>
        </g>
      ))}
      {bins.map((b, i) => <text key={b} x={x(i)} y={H + 2} fontSize={9.5} textAnchor="middle" fill="var(--text-muted)">{b}</text>)}
      {(() => {
        // End labels, nudged apart so they never overlap.
        const ends = s.levels.map((lv) => ({ lv, y: y(Math.max(0, s.decay[lv][s.decay[lv].length - 1].median_cos)) }))
          .sort((a, b) => b.y - a.y);
        for (let i = 1; i < ends.length; i++) ends[i].y = Math.min(ends[i].y, ends[i - 1].y - 12);
        return ends.map(({ lv, y: ly }) => (
          <text key={`l-${lv}`} x={x(bins.length - 1) + 8} y={ly + 3} fontSize={9.5} fill="var(--text-secondary)">{s.level_labels[lv]}</text>
        ));
      })()}
      {s.levels.map((lv, k) => {
        const pts = s.decay[lv].map((d, i) => [x(i), y(Math.max(0, d.median_cos))] as const);
        return (
          <g key={lv}>
            <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={`var(${LADDER_VARS[k]})`} strokeWidth={2} />
            {pts.map(([px, py], i) => (
              <circle key={i} cx={px} cy={py} r={4} fill={`var(${LADDER_VARS[k]})`} stroke="var(--surface-1)" strokeWidth={2}
                onMouseMove={(e) => show(e, <div><b>{s.level_labels[lv]}</b>, {bins[i]} layers after: median cos {s.decay[lv][i].median_cos.toFixed(2)} (n={s.decay[lv][i].n})</div>)}
                onMouseLeave={hide} />
            ))}
          </g>
        );
      })}
    </svg>
  );
}

export function StudyView() {
  const [s, setS] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.summary().then(setS).catch((e) => setError(String(e))); }, []);
  if (error) return <ErrorNote error={`${error}. Run: python experiments/fidelity_sweep.py && python experiments/analyze.py`} />;
  if (!s) return <p className="text-sm text-muted">Loading study…</p>;

  const cat = (lv: string) => s.ladder[lv].catastrophic[0];
  const singles = Object.entries(s.trust.single_signal_auroc).sort((a, b) => b[1] - a[1]);
  const dec = s.decomposition;
  return (
    <div className="space-y-4">
      <div className="max-w-3xl text-sm text-ink2">
        <p><b className="text-ink">Question.</b> Attribution graphs explain an LLM through an interpretable replacement model, a "twin". When the twin predicts what an intervention will do, how often is it right, why is it wrong, and can it tell in advance?</p>
        <p className="mt-1"><b className="text-ink">The hard part.</b> Most edits barely move the output (median |Δlog p| = {s.real_effect.median_abs_dlogp.toFixed(3)}). On the {(s.real_effect.frac_big * 100).toFixed(0)}% that do (|Δlog p| &gt; 0.05), the graph twin is within tolerance only {(s.trust.base_rate_big * 100).toFixed(0)}% of the time, and the trust model still separates its hits from its misses (AUROC {s.trust.cv_auroc_big.toFixed(2)}).</p>
        <p className="mt-1">{s.n_edits.toLocaleString()} transcoder-feature edits (ablate / amplify ×2) on {s.n_prompts} prompts in 8 families; each edit is simulated at five twin levels and run on the real GPT-2.</p>
      </div>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
        <Stat label="Trust model AUROC (held-out prompts)" value={s.trust.cv_auroc.toFixed(2)} note={`predicts when the graph twin is right; base rate ${(s.trust.base_rate * 100).toFixed(0)}%`} />
        <Stat label="Right when trusted vs flagged" value={`${(s.trust.ok_rate_trusted_half * 100).toFixed(1)}% vs ${(s.trust.ok_rate_flagged_half * 100).toFixed(0)}%`} note="top vs bottom half of trust scores" />
        <Stat label="Catastrophic misses, frozen → live LN" value={`${(cat("replacement") * 100).toFixed(1)}% → ${(cat("replacement_ln") * 100).toFixed(1)}%`} note="|error| > 1 nat" />
        <Stat label="Rank correlation, graph → live attention" value={`${s.ladder.linear.spearman[0].toFixed(2)} → ${s.ladder.replacement_full.spearman[0].toFixed(2)}`} note="Spearman ρ with the real effect" />
      </div>
      <Card title="The twin ladder" subtitle={<>Levels: {s.levels.map((lv, i) => `${i + 1} = ${s.level_labels[lv]}`).join(" · ")}. Whiskers: 95% cluster bootstrap over prompts.</>}>
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <LadderMetric s={s} metric="spearman" title="Rank correlation (ρ)" />
          <LadderMetric s={s} metric="median_cos" title="Median cosine of Δlogits" />
          <LadderMetric s={s} metric="sign_agree" title="Sign agreement" pct />
          <LadderMetric s={s} metric="catastrophic" title="Catastrophic misses" pct />
        </div>
      </Card>
      <div className="grid gap-4 lg:grid-cols-2">
        <Card title="Fidelity vs depth" subtitle="Median cosine between twin and real Δlogits, by layers between the edit and the output">
          <Decay s={s} />
        </Card>
        <div className="space-y-4">
          <Card title="Where the error comes from" subtitle="Share of the replacement twin's (clipped) mean error removed by each unfreezing step">
            <div className="flex h-7 w-full overflow-hidden rounded" role="img" aria-label="Error decomposition">
              {[["Frozen LayerNorm", dec.frozen_layernorm, "--series-1"], ["Frozen attention", dec.frozen_attention, "--series-2"], ["Transcoder (remaining)", dec.transcoder_residual, "--series-3"]].map(([name, v, c]) => (
                <div key={name as string} style={{ width: `${Math.max(0, v as number) * 100}%`, background: `var(${c})`, borderRight: "2px solid var(--surface-1)" }} title={`${name}: ${((v as number) * 100).toFixed(0)}%`} />
              ))}
            </div>
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 text-xs text-ink2">
              <span><i className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: "var(--series-1)" }} />Frozen LayerNorm {(dec.frozen_layernorm * 100).toFixed(0)}%</span>
              <span><i className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: "var(--series-2)" }} />Frozen attention {(dec.frozen_attention * 100).toFixed(0)}%</span>
              <span><i className="mr-1 inline-block h-2 w-2 rounded-sm" style={{ background: "var(--series-3)" }} />Transcoder error {(dec.transcoder_residual * 100).toFixed(0)}%</span>
            </div>
          </Card>
          <Card title="Which signals predict a wrong explanation" subtitle="AUROC of each signal alone vs the combined trust model (cross-validated by prompt)">
            <ul className="space-y-1 text-xs">
              {[["Trust model (all signals)", s.trust.cv_auroc, true] as const, ...singles.map(([k, v]) => [SIGNAL_NAMES[k] ?? k, v, false] as const)].map(([name, v, main]) => (
                <li key={name} className="grid grid-cols-[11rem_1fr_3rem] items-center gap-2">
                  <span className={main ? "font-semibold text-ink" : "text-ink2"}>{name}</span>
                  <span className="h-2 rounded-r" style={{ width: `${((v - 0.5) / 0.5) * 100}%`, minWidth: 2, background: main ? "var(--series-1)" : "var(--baseline)" }} />
                  <span className="tnum text-right">{v.toFixed(2)}</span>
                </li>
              ))}
            </ul>
            <p className="mt-2 text-xs text-muted">Bars start at 0.5 (chance).</p>
          </Card>
        </div>
      </div>
    </div>
  );
}
