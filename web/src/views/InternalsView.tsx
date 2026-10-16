import { useEffect, useMemo, useState } from "react";
import { api, type InterveneResponse, type RunResponse } from "../api";
import { Heatmap } from "../components/Heatmap";
import { diverging, sequential, useTokens } from "../components/theme";
import { Button, Card, ErrorNote, TokenBars } from "../components/ui";
import { tokenText } from "../lib";

function useRun(prompt: string) {
  const [run, setRun] = useState<RunResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const go = async () => {
    setBusy(true); setError(null);
    try { setRun(await api.run(prompt)); } catch (e) { setError(String(e)); } finally { setBusy(false); }
  };
  return { run, error, busy, go };
}

export function InternalsView({ prompt }: { prompt: string }) {
  const t = useTokens();
  const { run, error, busy, go } = useRun(prompt);
  const [layer, setLayer] = useState(9);
  const [head, setHead] = useState(9);
  const [ablated, setAblated] = useState<{ layer: number; head: number; res: InterveneResponse } | null>(null);

  useEffect(() => { go(); }, [prompt]); // eslint-disable-line react-hooks/exhaustive-deps

  const headGrid = useMemo(() => {
    if (!run) return null;
    const g = Array.from({ length: 12 }, () => Array(12).fill(0));
    const mlp = Array(12).fill(0);
    for (const d of run.dla) {
      const m = /^L(\d+)H(\d+)$/.exec(d.component);
      if (m) g[+m[1]][+m[2]] = d.value;
      const n = /^L(\d+)\.mlp$/.exec(d.component);
      if (n) mlp[+n[1]] = d.value;
    }
    return { g, mlp, max: Math.max(...g.flat().map(Math.abs), ...mlp.map(Math.abs)) };
  }, [run]);

  const ablate = async (l: number, h: number) => {
    const res = await api.intervene(prompt, [{ kind: "head", layer: l, head: h }]);
    setAblated({ layer: l, head: h, res });
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center gap-3 text-sm text-ink2">
        <span>Prompt: <span className="tok text-ink">{tokenText(prompt)}</span></span>
        <Button kind="ghost" onClick={go} disabled={busy}>{busy ? "Running…" : "Refresh"}</Button>
      </div>
      <ErrorNote error={error} />
      {run && headGrid && (
        <>
          <Card title="Logit lens"
            subtitle={<>The residual stream decoded after each layer. Text = top guess; shade = probability of the model's final answer at that position.</>}>
            <div className="overflow-x-auto">
              <Heatmap ariaLabel="Logit lens" rows={13} cols={run.tokens.length} cell={46} labelWidth={46}
                rowLabel={(r) => (r === 12 ? "emb" : `L${11 - r}`)}
                colLabel={(c) => tokenText(run.tokens[c]).slice(0, 10)}
                color={(r, c) => sequential(t)(run.logit_lens.final_prob[12 - r][c])}
                cellText={(r, c) => tokenText(run.logit_lens.top_token[12 - r][c]).slice(0, 6)}
                tooltip={(r, c) => (
                  <div>
                    <div><b>{12 - r === 0 ? "embeddings" : `after layer ${11 - r}`}</b> at “{tokenText(run.tokens[c])}”</div>
                    <div>top guess “{tokenText(run.logit_lens.top_token[12 - r][c])}” ({(run.logit_lens.top_prob[12 - r][c] * 100).toFixed(1)}%)</div>
                    <div className="text-ink2">final answer: rank {run.logit_lens.final_rank[12 - r][c]}, p = {(run.logit_lens.final_prob[12 - r][c] * 100).toFixed(1)}%</div>
                  </div>
                )} />
            </div>
          </Card>

          <div className="grid gap-4 lg:grid-cols-2">
            <Card title="Which heads write the answer"
              subtitle={<>Direct logit attribution per head for “<span className="tok">{tokenText(run.target.token)}</span>”. Click a head to ablate it.</>}>
              <Heatmap ariaLabel="Direct logit attribution per attention head" rows={12} cols={12} cell={24} labelWidth={30}
                rowLabel={(r) => `L${r}`} colLabel={(c) => `H${c}`}
                color={(r, c) => diverging(t, headGrid.max)(headGrid.g[r][c])}
                outlined={(r, c) => ablated?.layer === r && ablated?.head === c}
                onClick={ablate}
                tooltip={(r, c) => <div><b>L{r}H{c}</b>: {headGrid.g[r][c] >= 0 ? "+" : ""}{headGrid.g[r][c].toFixed(3)} logits<div className="text-muted">click to zero-ablate</div></div>} />
              {ablated && (
                <div className="mt-3 grid grid-cols-2 gap-3">
                  <div><div className="mb-1 text-xs text-ink2">Before</div><TokenBars items={ablated.res.before.slice(0, 5)} color="var(--text-secondary)" /></div>
                  <div><div className="mb-1 text-xs text-ink2">After ablating L{ablated.layer}H{ablated.head} (KL {ablated.res.kl.toFixed(3)})</div><TokenBars items={ablated.res.after.slice(0, 5)} /></div>
                </div>
              )}
            </Card>
            <Card title="Attention pattern" subtitle="Who attends to whom (row = query token, column = key token)"
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
                  rowLabel={(r) => tokenText(run.tokens[r]).slice(0, 10)} colLabel={(c) => tokenText(run.tokens[c]).slice(0, 10)}
                  color={(r, c) => sequential(t)(run.attention[layer][head][r][c])}
                  tooltip={(r, c) => <div>“{tokenText(run.tokens[r])}” → “{tokenText(run.tokens[c])}”: {(run.attention[layer][head][r][c] * 100).toFixed(1)}%</div>} />
              </div>
            </Card>
          </div>
        </>
      )}
    </div>
  );
}
