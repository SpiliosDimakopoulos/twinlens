import type { ExplainResponse } from "../../api";
import { useTooltip } from "../../components/useTooltip";
import { percent } from "../../format";

const MAX_BAR_PX = 86; // a fully unexplained layer: the strip's 112px height minus the layer label
const MIN_BAR_PX = 2;  // keep well-explained layers visible

/** Fig. 4: share of each layer's MLP output the transcoder cannot explain, as bars. */
export function ErrorStrip({ exp }: { exp: ExplainResponse }) {
  const { hover } = useTooltip();
  return (
    <div className="flex h-28 items-end gap-[3px]" role="img" aria-label="Transcoder error fraction per layer">
      {Object.entries(exp.error_fraction).map(([l, v]) => (
        <div key={l} className="flex flex-1 flex-col items-center gap-1"
          {...hover(<div><b>Layer {l}</b>: {percent(v, 0)} of this layer's MLP output is not explained by its features</div>)}>
          <div className="w-full rounded-t-[3px]" style={{ height: `${Math.max(MIN_BAR_PX, Math.min(1, v) * MAX_BAR_PX)}px`, background: "var(--series-2)", opacity: 0.6 }} />
          <span className="text-[9.5px] text-muted tnum">{l}</span>
        </div>
      ))}
    </div>
  );
}
