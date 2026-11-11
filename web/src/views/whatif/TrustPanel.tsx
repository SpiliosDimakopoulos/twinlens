import type { WhatIfResponse } from "../../api";
import { Glyph } from "../../components/Glyph";
import { Info } from "../../components/Info";
import { GLOSSARY } from "../../glossary";
import { SIGNALS, trustBand } from "../../lib";

/** The trust score of one edit: the number, its band on a meter, and the signals behind it. */
export function TrustPanel({ data }: { data: WhatIfResponse }) {
  if (data.trust === null) return <p className="text-xs text-muted">The trust score is not available in this build.</p>;
  const b = trustBand(data.trust);
  const glyph = b.status === "good" ? "check" : b.status === "warning" ? "alert" : "cross";
  return (
    <div>
      <div className="kicker flex items-center">Should you trust the graph here?<Info wide>{GLOSSARY.trust}</Info></div>
      <div className="mt-2 flex items-end gap-3">
        <span className="serif text-[44px] leading-none tracking-tight tnum">{Math.round(data.trust * 100)}<span className="text-[24px] text-ink2">%</span></span>
        <span className="mb-1 inline-flex items-center gap-1.5 text-[13px] font-medium">
          <span className="inline-flex h-[18px] w-[18px] items-center justify-center rounded-full" style={{ background: `var(--${b.status})`, color: "#111" }}>
            <Glyph name={glyph} size={12} />
          </span>
          {b.label}
        </span>
      </div>
      <div className="track relative mt-3 h-[3px] w-full rounded-full" role="meter"
        aria-valuemin={0} aria-valuemax={100} aria-valuenow={Math.round(data.trust * 100)} aria-label="Trust score"
        aria-valuetext={`${Math.round(data.trust * 100)}%: ${b.label}`}>
        <div className="absolute inset-y-0 left-0 rounded-full transition-all duration-500" style={{ width: `${data.trust * 100}%`, background: `var(--${b.status})` }} />
        {[40, 70].map((t) => <div key={t} className="absolute -top-1 h-[11px] w-px" style={{ left: `${t}%`, background: "var(--baseline)" }} />)}
      </div>
      <div className="relative mt-1 h-3 text-[10px] text-muted" aria-hidden>
        <span className="absolute left-0">don't trust</span>
        <span className="absolute" style={{ left: "40%" }}>check</span>
        <span className="absolute" style={{ left: "70%" }}>trust</span>
      </div>
      {data.trust_factors.length > 0 && (
        <ul className="mt-3 space-y-1 text-[12.5px]" aria-label="Main reasons for this score">
          <li className="kicker">Main reasons</li>
          {data.trust_factors.map((f) => (
            <li key={f.signal} className="flex items-center gap-2 text-ink2">
              <Glyph name={f.contribution >= 0 ? "up" : "down"} size={12}
                className={f.contribution >= 0 ? "text-[var(--good)]" : "text-[var(--critical)]"} />
              <span><span className="text-ink">{SIGNALS[f.signal]?.phrase ?? f.signal}</span> {f.contribution >= 0 ? "raises" : "lowers"} it</span>
            </li>
          ))}
        </ul>
      )}
      <p className="mt-2 text-[11.5px] text-muted">Computed from the unedited run only, before GPT-2 is asked about the edit. Even a high score is sometimes wrong.</p>
    </div>
  );
}
