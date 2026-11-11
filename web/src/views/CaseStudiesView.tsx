import { useEffect, useState } from "react";
import { api, type CaseStudy } from "../api";
import { Heatmap } from "../components/Heatmap";
import { diverging, sequential, useTokens } from "../components/theme";
import { Info } from "../components/Info";
import { Card, ErrorNote } from "../components/ui";
import { GLOSSARY } from "../glossary";
import { cell } from "../lib";

const has = (list: [number, number][] | undefined, r: number, c: number) =>
  !!list?.some(([l, h]) => l === r && h === c);

export function CaseStudiesView() {
  const t = useTokens();
  const [ind, setInd] = useState<CaseStudy | null>(null);
  const [ioi, setIoi] = useState<CaseStudy | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.caseStudy("induction").then(setInd).catch((e) => setError(String(e)));
    api.caseStudy("ioi").then(setIoi).catch((e) => setError(String(e)));
  }, []);
  return (
    <div className="space-y-5">
      <header className="grid gap-4 pb-3 pt-4 lg:grid-cols-[1.3fr_1fr] lg:gap-14">
        <div>
          <div className="kicker">Reproductions · calibrating the instrument</div>
          <h1 className="display mt-3 text-[36px] text-ink sm:text-[42px]">Published circuits, recomputed</h1>
        </div>
        <p className="self-end text-[14.5px] leading-relaxed text-ink2">
          A measuring instrument is only as good as its calibration. Before measuring anything new, TwinLens must
          reproduce known GPT-2 results from its own implementation; each is also pinned as a test. Each grid has one cell per attention head (row = layer, column = head); outlined cells are the heads the literature names.
        </p>
      </header>
      <ErrorNote error={error} />
      <div className="grid gap-5 lg:grid-cols-2">
        <Card label="Fig. 16" title="Induction heads" subtitle="Induction heads (Olsson et al., 2022) continue repeated text: on seeing a word again, they look at what followed it last time. Score = that attention, on repeated random tokens; outlined = heads commonly reported in GPT-2 small.">
          {ind ? (
            <Heatmap ariaLabel="Induction score per head" rows={12} cols={12} cell={24} labelWidth={30}
              rowLabel={(r) => `L${r}`} colLabel={(c) => `H${c}`}
              color={(r, c) => sequential(t)(cell(ind.grid, r, c, 0))}
              outlined={(r, c) => has(ind.known, r, c)}
              tooltip={(r, c) => <div><b>L{r}H{c}</b>: induction score {cell(ind.grid, r, c, 0).toFixed(2)}{has(ind.known, r, c) && <div>✓ reported induction head</div>}</div>} />
          ) : <p className="text-sm text-muted">Computing…</p>}
        </Card>
        <Card label="Fig. 17" title="Indirect object identification"
          subtitle={<>Wang et al. (2022). How much each head pushes the right name over the wrong one (IO − S)<Info>{GLOSSARY.ioi}</Info>. Name movers push the right name up (blue); negative name movers push it down (red).{ioi?.logit_diff !== undefined && <> On average GPT-2 prefers the right name by {ioi.logit_diff.toFixed(2)} logits.</>}</>}>
          {ioi ? (
            <>
              <Heatmap ariaLabel="IOI direct logit attribution per head" rows={12} cols={12} cell={24} labelWidth={30}
                rowLabel={(r) => `L${r}`} colLabel={(c) => `H${c}`}
                color={(r, c) => diverging(t, Math.max(...ioi.grid.flat().map(Math.abs)))(cell(ioi.grid, r, c, 0))}
                outlined={(r, c) => has(ioi.known, r, c) || has(ioi.negative, r, c)}
                tooltip={(r, c) => <div><b>L{r}H{c}</b>: {cell(ioi.grid, r, c, 0).toFixed(2)} logits{has(ioi.known, r, c) && <div>✓ name mover</div>}{has(ioi.negative, r, c) && <div>✓ negative name mover</div>}</div>} />
              <p className="tok mt-2 text-xs text-muted">e.g. {ioi.example}</p>
            </>
          ) : <p className="text-sm text-muted">Computing…</p>}
        </Card>
      </div>
    </div>
  );
}
