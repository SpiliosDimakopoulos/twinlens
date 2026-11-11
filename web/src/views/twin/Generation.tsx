import { useEffect, useRef, useState } from "react";
import { api, type GenerateStep } from "../../api";
import { Glyph } from "../../components/Glyph";
import { Info } from "../../components/Info";
import { useTooltip } from "../../components/useTooltip";
import { Button, Card } from "../../components/ui";
import { percent } from "../../format";
import { GLOSSARY } from "../../glossary";
import { decidedAt, tokenText } from "../../lib";

const MAX_NEW_TOKENS = 16;

/** Fig. 5: GPT-2 continues the prompt, each word shaded by the layer that decided it. */
export function Generation({ prompt }: { prompt: string }) {
  const [steps, setSteps] = useState<GenerateStep[]>([]);
  const [busy, setBusy] = useState(false);
  const stop = useRef<() => void>(() => {});
  const { hide, hover } = useTooltip();
  // A new prompt (or leaving the view) ends any running stream.
  useEffect(() => {
    setSteps([]);
    return () => { stop.current(); stop.current = () => {}; setBusy(false); hide(); };
  }, [prompt, hide]);
  const start = () => {
    setSteps([]); setBusy(true);
    stop.current = api.generate(prompt, (s) => setSteps((p) => [...p, s]), () => setBusy(false), MAX_NEW_TOKENS);
  };
  return (
    <Card label="Fig. 5" title="Live generation"
      subtitle={<>GPT-2 continues the prompt. Each word is shaded by how deep in the model it was decided: light = early layers, dark = last layers<Info>{GLOSSARY.generation}</Info>. Hover a word for details.</>}
      right={busy
        ? <Button kind="ghost" onClick={() => stop.current()}><Glyph name="stop" size={10} />Stop</Button>
        : <Button kind="ghost" onClick={start}><Glyph name="play" size={10} />Generate</Button>}>
      <div className="flex flex-wrap items-end gap-[3px] text-[13.5px] leading-7">
        <span className="tok whitespace-pre-wrap break-words text-muted">{prompt}</span>
        {steps.map((s, i) => {
          const l = decidedAt(s);
          // Tint from 12% (decided at the embeddings) to 80% (decided at the last layer).
          const shade = `color-mix(in srgb, var(--series-1) ${12 + (l / 12) * 68}%, transparent)`;
          return (
            <span key={i} className="tok rounded-[3px] px-[2px]" style={{ background: shade }}
              title={`decided by layer ${l} of 11`}
              {...hover(
                <div>
                  <div><b>“{tokenText(s.token)}”</b> decided by layer {l} of 11</div>
                  <div className="text-ink2">other candidates: {s.top.slice(1, 4).map((t) => `“${tokenText(t.token)}” ${percent(t.prob, 0)}`).join(", ")}</div>
                </div>,
              )}>{s.token}</span>
          );
        })}
        {busy && <span className="animate-pulse text-muted">▍</span>}
      </div>
    </Card>
  );
}
