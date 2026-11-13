import { useEffect, useState } from "react";
import { api, STATIC, type Ci, type Level, type LevelMap, type Metric, type Summary } from "../api";
import { Info } from "../components/Info";
import { Card, ErrorNote } from "../components/ui";
import { REPO } from "../examples";
import { adaptivePercent, percent } from "../format";
import { GLOSSARY } from "../glossary";
import { familyLabel, SIGNALS } from "../lib";
import { Decay } from "./study/Decay";
import { Families } from "./study/Families";
import { Forest } from "./study/Forest";
import { Generalization } from "./study/Generalization";
import { FIXES, LADDER } from "./study/rows";
import { Section } from "./study/Section";

/** Point estimate of one metric at one level, formatted; "—" when the level is missing. */
function point(ladder: LevelMap<Record<Metric, Ci>>, level: Level, metric: Metric, fmt: (v: number) => string): string {
  const ci = ladder[level]?.[metric];
  return ci ? fmt(ci[0]) : "—";
}
const rho = (ladder: LevelMap<Record<Metric, Ci>>, level: Level) => point(ladder, level, "spearman", (v) => v.toFixed(2));
const badly = (ladder: LevelMap<Record<Metric, Ci>>, level: Level) => point(ladder, level, "catastrophic", adaptivePercent);

export function StudyView() {
  const [s, setS] = useState<Summary | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => { api.summary().then(setS).catch((e) => setError(String(e))); }, []);
  if (error) return <ErrorNote error={`The study results could not be loaded (${error.replace(/^Error:\s*/, "")}).`} />;
  if (!s) return <p className="text-sm text-muted">Loading the study…</p>;

  const L = s.ladder, F = s.fix, T = s.trust;
  const pay = T.payoff?.top_edges_big_effect;
  const transfer = T.transfer_lofo;
  const reportHref = STATIC ? "./report/" : `${REPO}/blob/main/paper/REPORT.md`;

  return (
    <article>
      <header className="grid gap-6 pb-8 pt-4 lg:grid-cols-[1.3fr_1fr] lg:gap-14">
        <div>
          <div className="kicker">The study · {s.n_edits.toLocaleString()} interventions · {s.n_prompts} prompts · 8 task families</div>
          <h1 className="display mt-3 text-[38px] text-ink sm:text-[46px]">Can you trust what an attribution graph says will happen?</h1>
        </div>
        <p className="self-end text-[15px] leading-relaxed text-ink2">
          We switched off or amplified 8,064 features inside GPT-2 small, one at a time. Before each edit we asked several
          interpretable twins of GPT-2, from today's attribution graph up to the most faithful twin, to predict what would
          happen. Then we ran the real GPT-2 and compared. Arrows mark whether higher (↑) or lower (↓) is better; whiskers show the 95% uncertainty range across prompts. <a className="text-ink underline decoration-[var(--baseline)]" href={reportHref} target="_blank" rel="noreferrer">Full report →</a>
        </p>
      </header>

      <Section n="1" title="Today's graph is a weak predictor"
        lede={<>
          <p>Today's attribution graph gets the order of the real effects only weakly right: ρ = <b className="text-ink">{rho(L, "linear")}</b> (0 = no relation, 1 = perfect). Most edits barely matter. On the {adaptivePercent(s.real_effect.frac_big)} that do, its prediction is close to the real effect only <b className="text-ink">{adaptivePercent(T.base_rate_big)}</b> of the time, which is worse than always predicting "no change".</p>
          <p>Each step up the ladder switches one more part of the real model back on<Info wide>{GLOSSARY.ladder}</Info>, so the gap between two steps shows how much error that part caused.</p>
          <p>Because most edits barely change the answer, a predictor that always says "nothing happens" already has small errors. The ranking and the direction of effects are therefore the metrics to watch.</p>
        </>}>
        <Card label="Fig. 6" title="Agreement with GPT-2 as parts of the model are switched back on" subtitle="Dot: estimate; whisker: 95% uncertainty range. Hover for exact values.">
          <Forest rows={LADDER} data={L} />
        </Card>
      </Section>

      <Section n="2" title="Two frozen components cause it"
        lede={<>
          <p><b className="text-ink">Frozen LayerNorm</b> causes {adaptivePercent(s.decomposition.frozen_layernorm)} of the twin's error and nearly all of its badly wrong predictions. LayerNorm normally scales the model's internal state back to a fixed size. With it frozen, a change is never scaled back, so it grows layer after layer. Most of these failures come from edits in the first layers.</p>
          <p><b className="text-ink">Frozen attention</b> adds little to the size of the error but spoils the ranking: switching it back on lifts ρ from {rho(L, "replacement_ln")} to {rho(L, "replacement_full")}. All versions get worse the further the edit is from the output.</p>
        </>}>
        <div className="grid gap-5 xl:grid-cols-[1.25fr_1fr]">
          <Card label="Fig. 7" title="Accuracy drops with distance" subtitle="How well the predicted change matches the real one (median cosine; 1 = same direction, 0 = unrelated), by how many layers separate the edit from the output">
            <Decay s={s} />
          </Card>
          <Card label="Fig. 8" title="Where the error comes from" subtitle="Share of the twin's average error (each error capped at 5 nats) that disappears when each part is switched back on; the rest is the transcoders' own error">
            <div className="mt-2 flex h-6 w-full overflow-hidden rounded-[3px]" role="img" aria-label="Error decomposition">
              {([["Frozen LayerNorm", s.decomposition.frozen_layernorm, "--series-1"], ["Frozen attention", s.decomposition.frozen_attention, "--series-2"], ["Transcoders", s.decomposition.transcoder_residual, "--series-3"]] as const).map(([name, v, c]) => (
                <div key={name} style={{ width: `${Math.max(0, v) * 100}%`, background: `var(${c})`, boxShadow: "inset -2px 0 0 var(--surface-1)" }} title={`${name}: ${percent(v, 1)}`} />
              ))}
            </div>
            <dl className="mt-4 grid grid-cols-3 gap-3 text-[12px]">
              {([["Frozen LayerNorm", s.decomposition.frozen_layernorm, "--series-1"], ["Frozen attention", s.decomposition.frozen_attention, "--series-2"], ["Transcoder error", s.decomposition.transcoder_residual, "--series-3"]] as const).map(([name, v, c]) => (
                <div key={name} className="border-t pt-2" style={{ borderColor: `var(${c})` }}>
                  <dt className="text-ink2">{name}</dt>
                  <dd className="serif mt-0.5 text-[24px] leading-none text-ink tnum">{percent(v, v < 0.1 ? 1 : 0)}</dd>
                </div>
              ))}
            </dl>
          </Card>
        </div>
      </Section>

      {F && (
        <Section n="3" title="Keep the derivatives, keep the linearity"
          lede={<>
            <p>Freezing LayerNorm and attention means the graph ignores how they react to an edit. We add that reaction back to first order (the "Jacobians" of LayerNorm and of the attention softmax). The graph stays <b className="text-ink">linear</b>: each feature's effect is still a single number, and effects still add up.</p>
            <p>That graph matches the nonlinear twin on every metric: ρ {rho(F.ladder, "linear")} → <b className="text-ink">{rho(F.ladder, "linear_fulljac")}</b>, badly wrong predictions {badly(F.ladder, "linear")} → <b className="text-ink">{badly(F.ladder, "linear_fulljac")}</b>. In this setting the problem is the missing reactions, not linearity.</p>
          </>}>
          <Card label="Fig. 9" title="The proposed fix against today's graph and the nonlinear twin" subtitle="Same 8,064 edits; dot: estimate, whisker: 95% uncertainty range. Hover for exact values.">
            <Forest rows={FIXES} data={F.ladder} />
          </Card>
          <Card label="Fig. 10" title="Tasks that rely on attention gain most" subtitle={<>Rank correlation (ρ, higher is better) per kind of prompt: <span className="inline-block h-2 w-2 rounded-full align-middle" style={{ background: "var(--baseline)" }} /> graph today → <span className="inline-block h-2 w-2 rounded-full align-middle" style={{ background: "var(--fix)" }} /> graph with the fix; ring: nonlinear twin</>}>
            <Families fix={F} />
          </Card>
          {s.generalization && (
            <Card label="Fig. 10b" title="It holds on a second architecture"
              subtitle={<>How much the fix improves on today's graph, on the same edits, in GPT-2 small and in Qwen3-0.6B, a newer model built differently (layers 10–27 only, 16 prompts). Right of the zero line means better. Every range stays on the better side; the gain on Qwen3 is smaller. On Gemma-2-2B (17,472 edits, report §4.9) the fix also lifts ρ from 0.21 to 0.36.</>}>
              <Generalization g={s.generalization} />
            </Card>
          )}
        </Section>
      )}

      <Section n="4" title="Bad predictions can be spotted in advance"
        lede={<>
          <p>Nine signals from the unedited run predict whether today's graph will be right about an edit. On edits that matter they separate right from wrong with AUROC<Info>{GLOSSARY.auroc}</Info> <b className="text-ink">{T.cv_auroc_big.toFixed(2)}</b> on unseen prompts ({T.cv_auroc.toFixed(2)} over all edits, most of which are trivially easy{transfer ? <>; {transfer.mean.toFixed(2)} on kinds of prompt it never saw</> : null}).</p>
          <p>A simpler warning works about as well: when the graph predicts a very large effect, it is usually wrong.</p>
          {pay && <p>On the arrows a user would inspect first, the graph is right {adaptivePercent(pay.ok_all)} of the time; on the ones the trust score clears, <b className="text-ink">{adaptivePercent(pay.ok_trusted)}</b>; on the ones it flags, only {adaptivePercent(pay.ok_flagged)}.</p>}
        </>}>
        <div className="grid gap-5 xl:grid-cols-[1.2fr_1fr]">
          <Card label="Fig. 11" title="Which signals predict a wrong explanation" subtitle="How well each signal alone spots wrong predictions (AUROC over all edits), against the combined trust score. Bars start at guessing (0.5); a full bar is perfect (1).">
            <ul className="space-y-[7px] text-[12.5px]">
              {[["Trust score (all signals)", T.cv_auroc, true] as const, ...Object.entries(T.single_signal_auroc).sort((a, b) => b[1] - a[1]).map(([k, v]) => [SIGNALS[k]?.name ?? k, v, false] as const)].map(([name, v, main]) => (
                <li key={name} className="grid grid-cols-[11.5rem_1fr_2.5rem] items-center gap-3">
                  <span className={main ? "font-semibold text-ink" : "text-ink2"}>{name}</span>
                  <span className="track relative h-[6px] rounded-full">
                    <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${((v - 0.5) / 0.5) * 100}%`, background: main ? "var(--text-primary)" : "var(--baseline)" }} />
                  </span>
                  <span className="tnum text-right text-ink2">{v.toFixed(2)}</span>
                </li>
              ))}
            </ul>
          </Card>
          {transfer && (
            <Card label="Fig. 12" title="Does it work on new kinds of prompt?" subtitle="Trained on seven kinds of prompt, tested on the eighth (AUROC over all edits; 0.5 = guessing, 1 = perfect)">
              <ul className="space-y-[7px] text-[12.5px]">
                {Object.entries(transfer).filter(([k]) => k !== "mean").sort((a, b) => b[1] - a[1]).map(([k, v]) => (
                  <li key={k} className="grid grid-cols-[6.5rem_1fr_2.5rem] items-center gap-3">
                    <span className="text-ink2">{familyLabel(k)}</span>
                    <span className="track relative h-[6px] rounded-full">
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
