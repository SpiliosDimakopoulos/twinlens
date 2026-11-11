import { useCallback, useEffect, useMemo, useState } from "react";
import { api, STATIC, type InterveneResponse, type RunResponse } from "../api";
import { Heatmap } from "../components/Heatmap";
import { Info } from "../components/Info";
import { GLOSSARY } from "../glossary";
import { LensReading } from "./LensReading";
import { diverging, sequential, useTokens } from "../components/theme";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { percent } from "../format";
import { cell, tokenText } from "../lib";
import { useLatest } from "../useLatest";

function useRun(prompt: string) {
  const [run, setRun] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const begin = useLatest();
  const go = useCallback(async () => {
    const isLatest = begin();
    setBusy(true); setError(null);
    try {
      const r = await api.run(prompt);
      if (isLatest()) setRun(r);
    } catch (e) {
      if (isLatest()) setError(String(e));
    } finally {
      if (isLatest()) setBusy(false);
    }
  }, [prompt, begin]);
  return { run, error, busy, go };
}

export function InternalsView({ prompt }: { prompt: string }) {
  const t = useTokens();
  const { run, error, busy, go } = useRun(prompt);
  const [layer, setLayer] = useState(9);
  const [head, setHead] = useState(9);
  const [ablated, setAblated] = useState<{ layer: number; head: number; res: InterveneResponse } | null>(null);
  const [ablError, setAblError] = useState<string | null>(null);
  const beginAblation = useLatest();

  useEffect(() => { go(); }, [go]);  // go changes with the prompt

  const headGrid = useMemo(() => {
    if (!run) return null;
    const g: number[][] = Array.from({ length: 12 }, () => Array<number>(12).fill(0));
    const mlp: number[] = Array<number>(12).fill(0);
    for (const d of run.dla) {
      const m = /^L(\d+)H(\d+)$/.exec(d.component);
      const row = m ? g[Number(m[1])] : undefined;
      if (row) row[Number(m?.[2])] = d.value;
      const n = /^L(\d+)\.mlp$/.exec(d.component);
      if (n) mlp[Number(n[1])] = d.value;
    }
    return { g, mlp, max: Math.max(...g.flat().map(Math.abs), ...mlp.map(Math.abs)) };
  }, [run]);

  const ablate = async (l: number, h: number) => {
    const isLatest = beginAblation();
    setAblError(null);
    try {
      const res = await api.intervene(prompt, [{ kind: "head", layer: l, head: h }]);
      if (isLatest()) setAblated({ layer: l, head: h, res });
    } catch (e) {
      if (isLatest()) { setAblated(null); setAblError(String(e)); }
    }
  };
  useEffect(() => { setAblated(null); setAblError(null); }, [prompt]);

  return (
    <div className="space-y-5">
      <header className="grid gap-4 pb-3 pt-4 lg:grid-cols-[1.3fr_1fr] lg:gap-14">
        <div>
          <div className="kicker">Internals · the real GPT-2, no twin</div>
          <h1 className="display mt-3 text-[36px] text-ink sm:text-[42px]">Inside the forward pass</h1>
        </div>
        <div className="self-end text-[14.5px] leading-relaxed text-ink2">
          <p>How the answer forms layer by layer, which attention heads write it, and what they look at. This tab uses the real GPT-2 only; the prompt is the one chosen on the Twin tab.</p>
          <p className="mt-2 flex flex-wrap items-center gap-2">
            <span className="kicker">Prompt</span><span className="tok text-ink">{tokenText(prompt)}</span>
            {!STATIC && <Button kind="ghost" onClick={go} disabled={busy}>{busy ? "Running…" : "Refresh"}</Button>}
          </p>
        </div>
      </header>
      <ErrorNote error={error} />
      {run && headGrid && (
        <>
          <Card label="Fig. 13" title={<>Logit lens<Info>{GLOSSARY.logitLens}</Info></>}
            subtitle={<>What GPT-2 would predict after each layer (bottom row: before any layer; top row: the last layer). Text in a cell = its top guess for the next word; darker = more probability on the model's final answer.</>}>
            <div className="grid gap-8 lg:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
            <div className="overflow-x-auto">
              <Heatmap ariaLabel="Logit lens" rows={13} cols={run.tokens.length} cell={46} labelWidth={46}
                rowLabel={(r) => (r === 12 ? "emb" : `L${11 - r}`)}
                colLabel={(c) => tokenText(run.tokens[c] ?? "").slice(0, 10)}
                color={(r, c) => sequential(t)(cell(run.logit_lens.final_prob, 12 - r, c, 0))}
                cellText={(r, c) => tokenText(cell(run.logit_lens.top_token, 12 - r, c, "")).slice(0, 6)}
                tooltip={(r, c) => (
                  <div>
                    <div><b>{12 - r === 0 ? "embeddings" : `after layer ${11 - r}`}</b> at “{tokenText(run.tokens[c] ?? "")}”</div>
                    <div>top guess “{tokenText(cell(run.logit_lens.top_token, 12 - r, c, ""))}” ({percent(cell(run.logit_lens.top_prob, 12 - r, c, 0), 1)})</div>
                    <div className="text-ink2">final answer: rank {cell(run.logit_lens.final_rank, 12 - r, c, 0)}, p = {percent(cell(run.logit_lens.final_prob, 12 - r, c, 0), 1)}</div>
                  </div>
                )} />
            </div>
            <LensReading run={run} />
            </div>
          </Card>

          <div className="grid gap-5 lg:grid-cols-2">
            <Card label="Fig. 14" title={<>Which heads write the answer<Info>{GLOSSARY.dla}</Info></>}
              subtitle={<>Each cell is one attention head (row = layer, column = head). Blue heads push “<span className="tok">{tokenText(run.target.token)}</span>” up, red heads push it down. Click a head to switch it off and see what changes.</>}>
              <Heatmap ariaLabel="Direct logit attribution per attention head" rows={12} cols={12} cell={24} labelWidth={30}
                rowLabel={(r) => `L${r}`} colLabel={(c) => `H${c}`}
                color={(r, c) => diverging(t, headGrid.max)(cell(headGrid.g, r, c, 0))}
                outlined={(r, c) => ablated?.layer === r && ablated?.head === c}
                onClick={ablate}
                tooltip={(r, c) => <div><b>L{r}H{c}</b>: {cell(headGrid.g, r, c, 0) >= 0 ? "+" : ""}{cell(headGrid.g, r, c, 0).toFixed(3)} logits<div className="text-muted">click to switch this head off</div></div>} />
              <div role="alert"><ErrorNote error={ablError} /></div>
              {ablated && (
                <div className="mt-3 grid grid-cols-2 gap-3" aria-live="polite">
                  <div><div className="mb-1 text-xs text-ink2">Before</div><TokenBars items={ablated.res.before.slice(0, 5)} color="var(--text-secondary)" /></div>
                  <div><div className="mb-1 flex items-center text-xs text-ink2">After ablating L{ablated.layer}H{ablated.head} (switched off; overall change, KL {ablated.res.kl.toFixed(3)})<Info>{GLOSSARY.kl}</Info></div><TokenBars items={ablated.res.after.slice(0, 5)} /></div>
                </div>
              )}
            </Card>
            <Card label="Fig. 15" title="Attention pattern" subtitle="For the chosen head, how much each word (row) looks at each earlier word (column). Darker = more attention. Try layer 9, head 9: in name prompts it copies the right name (see Reproductions)."
              right={
                <div className="flex gap-2 text-xs">
                  <label>Layer <select value={layer} onChange={(e) => setLayer(+e.target.value)} className="rounded border bg-transparent px-1" style={{ borderColor: "var(--border)" }}>
                    {Array.from({ length: 12 }, (_, i) => <option key={i} value={i}>{i}</option>)}</select></label>
                  <label>Head <select value={head} onChange={(e) => setHead(+e.target.value)} className="rounded border bg-transparent px-1" style={{ borderColor: "var(--border)" }}>
                    {Array.from({ length: 12 }, (_, i) => <option key={i} value={i}>{i}</option>)}</select></label>
                </div>
              }>
              <div className="overflow-x-auto">
                <Heatmap ariaLabel={`Attention pattern of layer ${layer} head ${head}`} rows={run.tokens.length} cols={run.tokens.length}
                  cell={Math.max(16, Math.min(28, 420 / run.tokens.length))} labelWidth={74}
                  rowLabel={(r) => tokenText(run.tokens[r] ?? "").slice(0, 10)} colLabel={(c) => tokenText(run.tokens[c] ?? "").slice(0, 10)}
                  color={(r, c) => sequential(t)(cell(run.attention[layer]?.[head], r, c, 0))}
                  tooltip={(r, c) => <div>“{tokenText(run.tokens[r] ?? "")}” → “{tokenText(run.tokens[c] ?? "")}”: {percent(cell(run.attention[layer]?.[head], r, c, 0), 1)}</div>} />
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
