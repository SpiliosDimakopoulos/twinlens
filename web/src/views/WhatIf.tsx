import { useEffect, useRef, useState } from "react";
import { api, STATIC, type Feature, type WhatIfResponse } from "../api";
import { Info } from "../components/Info";
import { Skeleton } from "../components/Skeleton";
import { Card, ErrorNote, TokenBars } from "../components/ui";
import { PRESET_HINT, PRESET_VERB, PRESETS } from "../constants";
import { signed } from "../format";
import { GLOSSARY } from "../glossary";
import { tokenText, verdict } from "../lib";
import { useLatest } from "../useLatest";
import { GapPlot } from "./whatif/GapPlot";
import { presentRows } from "./whatif/rows";
import { TrustPanel } from "./whatif/TrustPanel";

const DEBOUNCE_MS = 250;  // live API only: wait for the slider to settle before asking the server

export function WhatIf({ prompt, feature, tokens, mult, setMult, label = "Fig. 2" }: {
  prompt: string; feature: Feature | null; tokens: string[]; mult: number; setMult: (m: number) => void; label?: string;
}) {
  const [res, setRes] = useState<WhatIfResponse | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const begin = useLatest();
  const shownFor = useRef("");
  const key = feature ? `${prompt}|${feature.layer}-${feature.pos}-${feature.feature}` : "";

  useEffect(() => {
    if (!feature) return;
    const isLatest = begin();
    // A different feature or prompt: drop the old result rather than show it under the new title.
    if (shownFor.current !== key) setRes(null);
    setBusy(true); setError(null);
    const timer = setTimeout(async () => {
      try {
        const r = await api.whatif(prompt, feature.layer, feature.pos, feature.feature, feature.activation * mult);
        if (isLatest()) { setRes(r); shownFor.current = key; }
      } catch (e) {
        if (isLatest()) { setError(String(e)); setRes(null); }
      } finally {
        if (isLatest()) setBusy(false);
      }
    }, STATIC ? 0 : DEBOUNCE_MS);
    return () => clearTimeout(timer);
  }, [prompt, feature, mult, key, begin]);

  if (!feature)
    return (
      <Card label={label} title="What if?" subtitle="Click any dot in the reasoning path (or press ← →).">
        <p className="text-sm text-ink2">The graph predicts what changes if you switch that feature off or turn it up. Then GPT-2 itself is run with the same edit, so you can see who was right.</p>
      </Card>
    );

  const verb = PRESET_VERB[mult] ?? `scale ×${mult}`;
  return (
    <Card label={label}
      title={<>What if we {verb} this feature?</>}
      subtitle={<>Layer {feature.layer}, feature #{feature.feature}. It is active on the word “<span className="tok">{tokenText(tokens[feature.pos] ?? "")}</span>” and mostly pushes up the words <span className="tok">{feature.label.slice(0, 3).map(tokenText).join(" · ")}</span>.</>}
      right={<span className="kicker animate-pulse" role="status">{busy ? "simulating…" : ""}</span>}>
      <div className="flex flex-wrap items-center gap-3">
        <div className="ring-baseline inline-flex rounded-[5px] p-0.5" role="group" aria-label="Edit strength">
          {PRESETS.map(([l, m]) => (
            <button key={l} onClick={() => setMult(m)} aria-pressed={mult === m} title={PRESET_HINT[m]}
              className="rounded-[4px] px-2.5 py-1 text-[12.5px] font-medium transition-colors"
              style={mult === m ? { background: "var(--text-primary)", color: "var(--page)" } : { color: "var(--text-secondary)" }}>
              {l}
            </button>
          ))}
        </div>
        {!STATIC && (
          <>
            <label className="sr-only" htmlFor="mult">Activation multiplier</label>
            <input id="mult" type="range" min={0} max={3} step={0.25} value={mult}
              onChange={(e) => setMult(Number(e.target.value))} className="w-24 accent-[var(--text-primary)]" />
          </>
        )}
        <span className="tnum text-[12px] text-muted">feature strength {feature.activation.toFixed(2)} → {(feature.activation * mult).toFixed(2)}</span>
      </div>
      <div role="alert"><ErrorNote error={error} /></div>
      {error ? null : !res ? (
        <div className="mt-4 space-y-2"><Skeleton className="h-5 w-4/5" /><Skeleton className="h-40" /><Skeleton className="h-20" /></div>
      ) : (
        <div className={`mt-4 transition-opacity ${busy ? "opacity-55" : ""}`}>
          <blockquote className="serif border-l-2 pl-3 text-[17px] leading-snug text-ink rule">
            {verdict(res.levels.linear.dlogp, res.levels.real.dlogp)}
            {res.levels.linear_fulljac && (
              <span className="mt-1 block text-[14px] text-ink2">
                With the fix, the graph's error is {Math.abs(res.levels.linear_fulljac.dlogp - res.levels.real.dlogp).toFixed(2)} instead of {Math.abs(res.levels.linear.dlogp - res.levels.real.dlogp).toFixed(2)}.
              </span>
            )}
          </blockquote>
          <div className="mt-4"><GapPlot data={res} /></div>
          <ul className="sr-only" aria-label="Predicted and real change">
            {presentRows(res.levels).map((r) => (
              <li key={r.key}>{r.label} predicts {signed(r.result.dlogp)}</li>
            ))}
            <li>GPT-2 actually changes by {signed(res.levels.real.dlogp)}</li>
          </ul>
          <p className="mt-1 text-[11.5px] text-muted">
            How much the edit changes the chance that GPT-2 says “<span className="tok">{tokenText(res.target)}</span>”<Info>{GLOSSARY.dlogp}</Info>.
            Each row is one way of predicting it<Info wide>{GLOSSARY.ladder}</Info>; the dot is the prediction, the vertical line is what GPT-2 really did, and the trail between them is the error.
          </p>
          <div className="mt-5 grid gap-5 border-t pt-4 rule">
            <TrustPanel data={res} />
            <div>
              <div className="kicker mb-2">GPT-2's most likely next words after the edit</div>
              <TokenBars items={res.levels.real.top} color="var(--text-secondary)" />
            </div>
          </div>
        </div>
      )}
    </Card>
  );
}
