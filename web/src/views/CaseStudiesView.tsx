import { useEffect, useState } from "react";
import { api, type CaseStudy } from "../api";
import { Heatmap } from "../components/Heatmap";
import { diverging, sequential, useTokens } from "../components/theme";
import { Card, ErrorNote } from "../components/ui";

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
          reproduce known GPT-2 results from its own implementation; each is also pinned as a test. Outlined cells are the heads the papers name.
        </p>
      </header>
      <ErrorNote error={error} />
      <div className="grid gap-5 lg:grid-cols-2">
        <Card label="Fig. 16" title="Induction heads" subtitle="Olsson et al. (2022). Attention to the token after the previous occurrence, on repeated random tokens.">
          {ind ? (
            <Heatmap ariaLabel="Induction score per head" rows={12} cols={12} cell={24} labelWidth={30}
              rowLabel={(r) => `L${r}`} colLabel={(c) => `H${c}`}
              color={(r, c) => sequential(t)(ind.grid[r][c])}
              outlined={(r, c) => has(ind.known, r, c)}
              tooltip={(r, c) => <div><b>L{r}H{c}</b>: induction score {ind.grid[r][c].toFixed(2)}{has(ind.known, r, c) && <div>✓ reported induction head</div>}</div>} />
          ) : <p className="text-sm text-muted">Computing…</p>}
        </Card>
        <Card label="Fig. 17" title="Indirect object identification"
          subtitle={<>Wang et al. (2022). Direct logit attribution to IO − S. Name movers push the right name up (blue); negative name movers push it down (red).{ioi?.logit_diff !== undefined && <> Mean logit diff {ioi.logit_diff.toFixed(2)}.</>}</>}>
          {ioi ? (
            <>
              <Heatmap ariaLabel="IOI direct logit attribution per head" rows={12} cols={12} cell={24} labelWidth={30}
                rowLabel={(r) => `L${r}`} colLabel={(c) => `H${c}`}
                color={(r, c) => diverging(t, Math.max(...ioi.grid.flat().map(Math.abs)))(ioi.grid[r][c])}
                outlined={(r, c) => has(ioi.known, r, c) || has(ioi.negative, r, c)}
                tooltip={(r, c) => <div><b>L{r}H{c}</b>: {ioi.grid[r][c].toFixed(2)} logits{has(ioi.known, r, c) && <div>✓ name mover</div>}{has(ioi.negative, r, c) && <div>✓ negative name mover</div>}</div>} />
              <p className="tok mt-2 text-xs text-muted">e.g. {ioi.example}</p>
            </>
          ) : <p className="text-sm text-muted">Computing…</p>}
        </Card>
      </div>
    </div>
  );
}
