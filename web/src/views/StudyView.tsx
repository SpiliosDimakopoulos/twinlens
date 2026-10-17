import { useEffect, useState, type ReactNode } from "react";
import { api, STATIC, type Ci, type Summary } from "../api";
import { LADDER_VARS } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { Card, ErrorNote } from "../components/ui";
import { REPO } from "../examples";

const SIGNAL_NAMES: Record<string, string> = {
  layers_after: "Layers after the edit",
  pos_from_end: "Distance to last token",
  activation: "Feature activation",
  rel_edit_size: "Edit size vs. residual",
  abs_direct_effect: "|Direct effect| (edge)",
  downstream_err_frac: "Downstream transcoder error",
  attn_to_pos: "Attention to edited token",
  n_active_downstream: "Active features downstream",
  clean_top1_prob: "Model confidence",
};

type LevelRow = { key: string; label: string; color: string; group?: string };
const LADDER: LevelRow[] = [
  { key: "direct", label: "Direct edges only", color: `var(${LADDER_VARS[0]})` },
  { key: "linear", label: "Graph today (linear)", color: `var(${LADDER_VARS[1]})` },
  { key: "replacement", label: "Replacement", color: `var(${LADDER_VARS[2]})` },
  { key: "replacement_ln", label: "+ live LayerNorm", color: `var(${LADDER_VARS[3]})` },
  { key: "replacement_full", label: "+ live attention", color: `var(${LADDER_VARS[4]})` },
];
const FIXES: LevelRow[] = [
  { key: "linear", label: "Graph today", color: "var(--baseline)" },
  { key: "linear_lnjac", label: "+ LayerNorm Jacobian", color: "color-mix(in srgb, var(--fix) 55%, transparent)" },
  { key: "linear_fulljac", label: "+ LN & attention Jacobians", color: "var(--fix)" },
  { key: "replacement_full", label: "Nonlinear twin (reference)", color: "var(--real)" },
];
const METRICS: [string, string, boolean][] = [
  ["spearman", "Rank correlation ρ", false],
  ["median_cos", "Median cosine", false],
  ["sign_agree", "Sign agreement", true],
  ["catastrophic", "Catastrophic misses", true],
];

const fmtV = (v: number, pct: boolean) => (pct ? `${(v * 100).toFixed(v < 0.01 ? 2 : v < 0.1 ? 1 : 0)}%` : v.toFixed(2));

/** Forest plot: one row per level, one mini-axis per metric, dot = estimate, whisker = 95% CI. */
function Forest({ rows, data }: { rows: LevelRow[]; data: Record<string, Record<string, Ci>> }) {
  const { show, hide } = useTooltip();
  const present = rows.filter((r) => data[r.key]);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[640px] border-collapse text-[12.5px]">
        <thead>
          <tr>
            <th className="w-[190px]" />
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
                const max = Math.max(...present.map((x) => data[x.key][m][2]), 1e-6) * 1.05;
                const [v, lo, hi] = data[r.key][m];
                const X = (t: number) => `${(t / max) * 100}%`;
                return (
                  <td key={m} className="py-2 pr-4"
                    onMouseMove={(e) => show(e, <div><b>{r.label}</b>: {fmtV(v, pct)} <span className="text-ink2">(95% CI {fmtV(lo, pct)} to {fmtV(hi, pct)})</span></div>)}
                    onMouseLeave={hide}>
                    <div className="flex items-center gap-2">
                      <div className="relative h-3 flex-1">
                        <div className="absolute inset-y-[5px] left-0 right-0" style={{ background: "var(--gridline)" }} />
                        <div className="absolute top-[5px] h-[2px]" style={{ left: X(lo), width: `calc(${X(hi)} - ${X(lo)})`, background: r.color, opacity: 0.5 }} />
                        <div className="absolute top-[1px] h-[10px] w-[10px] -translate-x-1/2 rounded-full" style={{ left: X(v), background: r.color, boxShadow: "0 0 0 2px var(--surface-1)" }} />
                      </div>
                      <span className="w-12 text-right tnum text-ink2">{fmtV(v, pct)}</span>
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

function Decay({ s }: { s: Summary }) {
  const { show, hide } = useTooltip();
  const bins = s.decay[s.levels[0]].map((d) => d.layers_after);
  const W = 420, H = 230, L = 30, R = 122;
  const x = (i: number) => L + (i / (bins.length - 1)) * (W - L - R);
  const y = (v: number) => H - 24 - v * (H - 44);
  const ends = LADDER.map((r) => ({ r, y: y(Math.max(0, s.decay[r.key][s.decay[r.key].length - 1].median_cos)) }))
    .sort((a, b) => b.y - a.y);
  for (let i = 1; i < ends.length; i++) ends[i].y = Math.min(ends[i].y, ends[i - 1].y - 14);
  return (
    <svg viewBox={`0 0 ${W} ${H + 6}`} width="100%" role="img" aria-label="Median cosine by depth for each twin level">
      {[0, 0.25, 0.5, 0.75, 1].map((g) => (
        <g key={g}>
          <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--gridline)" />
          <text x={L - 8} y={y(g) + 3.5} fontSize={10} textAnchor="end" fill="var(--text-muted)" className="tnum">{g}</text>
        </g>
      ))}
      {bins.map((b, i) => <text key={b} x={x(i)} y={H} fontSize={10.5} textAnchor="middle" fill="var(--text-muted)">{b} layers</text>)}
      {LADDER.map((r) => {
        const pts = s.decay[r.key].map((d, i) => [x(i), y(Math.max(0, d.median_cos))] as const);
        return (
          <g key={r.key}>
            <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke={r.color} strokeWidth={r.key === "linear" ? 2.5 : 1.75} />
            {pts.map(([px, py], i) => (
              <circle key={i} cx={px} cy={py} r={3.5} fill={r.color} stroke="var(--surface-1)" strokeWidth={1.5}
                onMouseMove={(e) => show(e, <div><b>{r.label}</b>, {bins[i]} layers to the output: median cosine {s.decay[r.key][i].median_cos.toFixed(2)}</div>)}
                onMouseLeave={hide} />
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

function Families({ s }: { s: Summary }) {
  const { show, hide } = useTooltip();
  const by = s.fix!.by_category_spearman;
  const fams = Object.keys(by).sort((a, b) => by[b].linear - by[a].linear);
  const W = 560, L = 92, R = 16, rowH = 26, top = 8;
  const x = (v: number) => L + (Math.max(0, v) / 0.65) * (W - L - R);
  const H = top + fams.length * rowH + 26;
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img" aria-label="Rank correlation per task family, graph today vs graph with Jacobians">
      {[0, 0.2, 0.4, 0.6].map((t) => (
        <g key={t}>
          <line x1={x(t)} x2={x(t)} y1={top} y2={H - 22} stroke="var(--gridline)" />
          <text x={x(t)} y={H - 8} fontSize={10} textAnchor="middle" fill="var(--text-muted)" className="tnum">{t.toFixed(1)}</text>
        </g>
      ))}
      {fams.map((f, i) => {
        const cy = top + i * rowH + rowH / 2;
        const a = by[f].linear, b = by[f].linear_fulljac, r = by[f].replacement_full;
        return (
          <g key={f} onMouseMove={(e) => show(e, <div><b>{f === "ioi" ? "IOI" : f}</b>: graph today ρ = {a.toFixed(2)} → with Jacobians {b.toFixed(2)} <span className="text-ink2">(nonlinear twin {r.toFixed(2)})</span></div>)} onMouseLeave={hide}>
            <rect x={0} y={cy - rowH / 2} width={W} height={rowH} fill="transparent" />
            <text x={L - 10} y={cy + 4} fontSize={11.5} textAnchor="end" fill="var(--text-primary)">{f === "ioi" ? "IOI" : f[0].toUpperCase() + f.slice(1)}</text>
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

const GEN_METRICS: [string, string, boolean][] = [
  ["spearman", "Δ rank correlation ρ", false],
  ["sign_agree", "Δ sign agreement", true],
  ["within_tol_big", "Δ within tolerance (edits that matter)", true],
  ["catastrophic", "Δ catastrophic misses (lower is better)", true],
];

/** Paired improvement of the fix over today's graph on each model, with 95% CIs. */
function Generalization({ g }: { g: NonNullable<Summary["generalization"]> }) {
  const { show, hide } = useTooltip();
  const models = Object.keys(g);
  const colors = ["var(--lv-5)", "var(--series-2)"];
  return (
    <div className="space-y-4">
      {GEN_METRICS.map(([m, label, pct]) => {
        const all = models.flatMap((k) => g[k].paired_diff[m]);
        const lo = Math.min(0, ...all), hi = Math.max(0, ...all);
        const X = (v: number) => `${((v - lo) / (hi - lo || 1)) * 100}%`;
        const f = (v: number) => (pct ? `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)} pp` : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(2)}`);
        return (
          <div key={m}>
            <div className="kicker mb-1.5">{label}</div>
            {models.map((k, i) => {
              const [v, l, h] = g[k].paired_diff[m];
              return (
                <div key={k} className="grid grid-cols-[7.5rem_1fr_4.5rem] items-center gap-3 py-0.5 text-[12.5px]"
                  onMouseMove={(e) => show(e, <div><b>{g[k].label}</b>: {f(v)} <span className="text-ink2">(95% CI {f(l)} to {f(h)}; {g[k].n_edits.toLocaleString()} edits)</span></div>)}
                  onMouseLeave={hide}>
                  <span className="text-ink2">{g[k].label}</span>
                  <div className="relative h-3">
                    <div className="absolute inset-y-[5px] left-0 right-0" style={{ background: "var(--gridline)" }} />
                    <div className="absolute inset-y-0 w-px" style={{ left: X(0), background: "var(--text-muted)" }} />
                    <div className="absolute top-[5px] h-[2px]" style={{ left: X(l), width: `calc(${X(h)} - ${X(l)})`, background: colors[i] }} />
                    <div className="absolute top-[1px] h-[10px] w-[10px] -translate-x-1/2 rounded-full" style={{ left: X(v), background: colors[i], boxShadow: "0 0 0 2px var(--surface-1)" }} />
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

function Section({ n, title, children, lede }: { n: string; title: string; lede: ReactNode; children: ReactNode }) {
  return (
    <section className="grid gap-x-12 gap-y-4 border-t pb-10 pt-6 rule lg:grid-cols-[minmax(0,330px)_1fr]">
      <div>
        <div className="kicker">§ {n}</div>
        <h2 className="display mt-2 text-[28px] text-ink">{title}</h2>
        <div className="mt-3 space-y-3 text-[14.5px] leading-relaxed text-ink2">{lede}</div>
      </div>
      <div className="min-w-0 space-y-5">{children}</div>
    </section>
  );
}

export function StudyView() {
  const [s, setS] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.summary().then(setS).catch((e) => setError(String(e))); }, []);
  if (error) return <ErrorNote error={`${error}. Run: python experiments/fidelity_sweep.py && python experiments/analyze.py`} />;
  if (!s) return <p className="text-sm text-muted">Loading the study…</p>;

  const L = s.ladder, F = s.fix, T = s.trust;
  const pay = T.payoff?.top_edges_big_effect;
  const transfer = T.transfer_lofo;
  const pct = (v: number) => `${(v * 100).toFixed(v < 0.01 ? 2 : v < 0.1 ? 1 : 0)}%`;
  const reportHref = STATIC ? "./report/" : `${REPO}/blob/main/paper/REPORT.md`;

  return (
    <article>
      <header className="grid gap-6 pb-8 pt-4 lg:grid-cols-[1.3fr_1fr] lg:gap-14">
        <div>
          <div className="kicker">The study · {s.n_edits.toLocaleString()} interventions · {s.n_prompts} prompts · 8 task families</div>
          <h1 className="display mt-3 text-[38px] text-ink sm:text-[46px]">Can you trust what an attribution graph says will happen?</h1>
        </div>
        <p className="self-end text-[15px] leading-relaxed text-ink2">
          We switched off or amplified 8,064 transcoder features in GPT-2 small. Before running each edit we asked a ladder
          of twins (from today's attribution graph up to a nonlinear replacement model) to predict what would happen, then
          ran GPT-2. Intervals are 95% bootstraps over prompts. <a className="text-ink underline decoration-[var(--baseline)]" href={reportHref} target="_blank" rel="noreferrer">Full report →</a>
        </p>
      </header>

      <Section n="1" title="The graph twin is a weak predictor"
        lede={<>
          <p>Today's attribution graph ranks the real effects with ρ = <b className="text-ink">{L.linear.spearman[0].toFixed(2)}</b>. Most edits barely matter, and on the {pct(s.real_effect.frac_big)} that do, it lands within tolerance only <b className="text-ink">{pct(T.base_rate_big)}</b> of the time.</p>
          <p>Each step up the ladder unfreezes one component of the real model; the gaps between steps say where the error comes from.</p>
        </>}>
        <Card label="Fig. 6" title="Agreement with GPT-2 at each level of the twin ladder" subtitle="Dot: estimate; whisker: 95% CI. Hover for exact values.">
          <Forest rows={LADDER} data={L} />
        </Card>
      </Section>

      <Section n="2" title="Two frozen components cause it"
        lede={<>
          <p><b className="text-ink">Frozen LayerNorm</b> causes {pct(s.decomposition.frozen_layernorm)} of the replacement twin's error and nearly all of its catastrophic misses: perturbations that the real model renormalises instead compound layer after layer.</p>
          <p><b className="text-ink">Frozen attention</b> costs little mean error but most of the ranking: unfreezing it lifts ρ from {L.replacement_ln.spearman[0].toFixed(2)} to {L.replacement_full.spearman[0].toFixed(2)}. Every twin degrades with depth.</p>
        </>}>
        <div className="grid gap-5 xl:grid-cols-[1.25fr_1fr]">
          <Card label="Fig. 7" title="Fidelity decays with depth" subtitle="Median cosine between predicted and real Δlogits, by layers between the edit and the output">
            <Decay s={s} />
          </Card>
          <Card label="Fig. 8" title="Where the replacement twin's error comes from" subtitle="Share of clipped mean error removed by each unfreezing step">
            <div className="mt-2 flex h-6 w-full overflow-hidden rounded-[3px]" role="img" aria-label="Error decomposition">
              {([["Frozen LayerNorm", s.decomposition.frozen_layernorm, "--series-1"], ["Frozen attention", s.decomposition.frozen_attention, "--series-2"], ["Transcoders", s.decomposition.transcoder_residual, "--series-3"]] as const).map(([name, v, c]) => (
                <div key={name} style={{ width: `${Math.max(0, v) * 100}%`, background: `var(${c})`, boxShadow: "inset -2px 0 0 var(--surface-1)" }} title={`${name}: ${(v * 100).toFixed(1)}%`} />
              ))}
            </div>
            <dl className="mt-4 grid grid-cols-3 gap-3 text-[12px]">
              {([["Frozen LayerNorm", s.decomposition.frozen_layernorm, "--series-1"], ["Frozen attention", s.decomposition.frozen_attention, "--series-2"], ["Transcoder error", s.decomposition.transcoder_residual, "--series-3"]] as const).map(([name, v, c]) => (
                <div key={name} className="border-t pt-2" style={{ borderColor: `var(${c})` }}>
                  <dt className="text-ink2">{name}</dt>
                  <dd className="serif mt-0.5 text-[24px] leading-none text-ink tnum">{(v * 100).toFixed(v < 0.1 ? 1 : 0)}%</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
      </Section>

      {F && (
        <Section n="3" title="Keep the derivatives, keep the linearity"
          lede={<>
            <p>Freezing a scale or a pattern silently drops its first-order derivative. We put both back (the LayerNorm-scale Jacobian and the softmax-pattern Jacobian), which leaves the graph <b className="text-ink">exactly linear</b>: effects still add up over sources.</p>
            <p>That linear graph matches the full nonlinear twin on every metric: ρ {F.ladder.linear.spearman[0].toFixed(2)} → <b className="text-ink">{F.ladder.linear_fulljac.spearman[0].toFixed(2)}</b>, catastrophic misses {pct(F.ladder.linear.catastrophic[0])} → <b className="text-ink">{pct(F.ladder.linear_fulljac.catastrophic[0])}</b>. The problem was never linearity.</p>
          </>}>
          <Card label="Fig. 9" title="The proposed fix against today's graph and the nonlinear twin" subtitle="Same 8,064 edits; dot: estimate, whisker: 95% CI">
            <Forest rows={FIXES} data={F.ladder} />
          </Card>
          <Card label="Fig. 10" title="Attention-heavy tasks gain most" subtitle={<>Rank correlation per task family: <span className="inline-block h-2 w-2 rounded-full align-middle" style={{ background: "var(--baseline)" }} /> graph today → <span className="inline-block h-2 w-2 rounded-full align-middle" style={{ background: "var(--fix)" }} /> graph + Jacobians; ring: nonlinear twin</>}>
            <Families s={s} />
          </Card>
          {s.generalization && (
            <Card label="Fig. 10b" title="It holds on a second architecture"
              subtitle={<>Improvement of the fix over today's graph, paired on the same edits, on GPT-2 small and Qwen3-0.6B (RMSNorm, GQA, RoPE, QK-norm, SwiGLU; layers 10–27). Dot: estimate; whisker: 95% CI. Every interval excludes zero.</>}>
              <Generalization g={s.generalization} />
            </Card>
          )}
        </Section>
      )}

      <Section n="4" title="The twin can tell when it is wrong"
        lede={<>
          <p>Nine signals from the clean run alone predict whether today's graph will be right about an edit: AUROC <b className="text-ink">{T.cv_auroc.toFixed(2)}</b> on unseen prompts{transfer ? <>, and <b className="text-ink">{transfer.mean.toFixed(2)}</b> on task families it never saw</> : null}.</p>
          {pay && <p>On the edges a user would inspect, the graph is right {pct(pay.ok_all)} of the time; on the ones the trust score clears, <b className="text-ink">{pct(pay.ok_trusted)}</b>; on the ones it flags, {pct(pay.ok_flagged)}.</p>}
        </>}>
        <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]">
          <Card label="Fig. 11" title="Which signals predict a wrong explanation" subtitle="AUROC of each signal alone vs. the combined trust model; bars start at chance (0.5)">
            <ul className="space-y-[7px] text-[12.5px]">
              {[["Trust model (all signals)", T.cv_auroc, true] as const, ...Object.entries(T.single_signal_auroc).sort((a, b) => b[1] - a[1]).map(([k, v]) => [SIGNAL_NAMES[k] ?? k, v, false] as const)].map(([name, v, main]) => (
                <li key={name} className="grid grid-cols-[11.5rem_1fr_2.5rem] items-center gap-3">
                  <span className={main ? "font-semibold text-ink" : "text-ink2"}>{name}</span>
                  <span className="relative h-[6px] rounded-full" style={{ background: "var(--gridline)" }}>
                    <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${((v - 0.5) / 0.5) * 100}%`, background: main ? "var(--text-primary)" : "var(--baseline)" }} />
                  </span>
                  <span className="tnum text-right text-ink2">{v.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </Card>
          {transfer && (
            <Card label="Fig. 12" title="Transfer to unseen task families" subtitle="Trained on seven families, tested on the eighth">
              <ul className="space-y-[7px] text-[12.5px]">
                {Object.entries(transfer).filter(([k]) => k !== "mean").sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <li key={k} className="grid grid-cols-[6.5rem_1fr_2.5rem] items-center gap-3">
                    <span className="text-ink2">{k === "ioi" ? "IOI" : k[0].toUpperCase() + k.slice(1)}</span>
                    <span className="relative h-[6px] rounded-full" style={{ background: "var(--gridline)" }}>
                      <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${((v - 0.5) / 0.5) * 100}%`, background: "var(--series-1)" }} />
                    </span>
                    <span className="tnum text-right text-ink2">{v.toFixed(2)}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </div>
      </Section>
    </article>
  );
}
