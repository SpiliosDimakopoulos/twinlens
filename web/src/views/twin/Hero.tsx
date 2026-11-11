import type { Summary } from "../../api";
import { percent } from "../../format";

/** The finding, stated up front, with the three numbers that carry it. */
export function Hero({ summary }: { summary: Summary | null }) {
  const fix = summary?.fix?.ladder;
  const rho0 = fix?.linear?.spearman?.[0], rho1 = fix?.linear_fulljac?.spearman?.[0];
  const cat0 = fix?.linear?.catastrophic?.[0], cat1 = fix?.linear_fulljac?.catastrophic?.[0];
  const auc = summary?.trust.cv_auroc_big;
  return (
    <section className="grid gap-8 pb-8 pt-4 lg:grid-cols-[1.35fr_1fr] lg:gap-14">
      <div>
        <div className="kicker">Mechanistic interpretability · GPT-2 small · 8,064 interventions</div>
        <h1 className="display mt-3 text-[38px] text-ink sm:text-[50px]">
          Attribution graphs fail because of the derivatives they drop, <em className="italic text-ink2">not because they are linear.</em>
        </h1>
        <p className="mt-4 max-w-[60ch] text-[16px] leading-relaxed text-ink2">
          An attribution graph is a map of which internal features of a language model led to its answer.
          Each arrow is a claim: "switch this feature off and the answer changes this much". TwinLens checks
          those claims. It asks the graph what an edit will do, makes the same edit in the real GPT-2, and
          shows the gap. The gap is often large, because the graph ignores how two parts of the model
          (LayerNorm and attention) react to the edit. Adding that reaction back keeps the graph just as
          simple to read and makes it far more accurate.
        </p>
      </div>
      <div className="grid content-end gap-5 sm:grid-cols-3 lg:grid-cols-1">
        <HeroStat value={rho0 !== undefined && rho1 !== undefined ? `${rho0.toFixed(2)} → ${rho1.toFixed(2)}` : "—"}
          label="Agreement with GPT-2" note="Rank correlation (0 = none, 1 = perfect) between predicted and real effects, graph today → with the fix" />
        <HeroStat value={cat0 !== undefined && cat1 !== undefined ? `${heroPercent(cat0)} → ${heroPercent(cat1)}` : "—"}
          label="Badly wrong predictions" note="Share of edits where the graph is far off (more than 1 nat), today → with the fix" />
        <HeroStat value={auc !== undefined ? auc.toFixed(2) : "—"}
          label="Warns when it's wrong" note="How well the trust score spots bad predictions on unseen prompts, for edits that matter (AUROC: 0.5 = guessing, 1 = perfect)" />
      </div>
    </section>
  );
}

/** One decimal, two below 1%: keeps the small catastrophic share readable. */
const heroPercent = (v: number) => percent(v, v < 0.01 ? 2 : 1);

function HeroStat({ value, label, note }: { value: string; label: string; note: string }) {
  return (
    <div className="border-t pt-2.5 rule">
      <div className="kicker">{label}</div>
      <div className="serif mt-1 text-[30px] leading-none tracking-tight text-ink tnum">{value}</div>
      <div className="mt-1 text-[12px] leading-snug text-muted">{note}</div>
    </div>
  );
}
