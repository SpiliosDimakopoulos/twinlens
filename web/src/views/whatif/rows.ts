import type { Level, LevelResult, WhatIfResponse } from "../../api";
import { LADDER_VARS } from "../../components/theme";

export type Row = { key: Level; label: string; desc: string; color: string; strong?: boolean; fix?: boolean };

/** The rows of the gap plot, top to bottom: today's graph, the ladder of twins, then the fix. */
export const ROWS: Row[] = [
  { key: "linear", label: "Graph today", desc: "The attribution graph as it is drawn today: LayerNorm and attention frozen.", color: `var(${LADDER_VARS[1]})`, strong: true },
  { key: "direct", label: "Direct edges only", desc: "Only this feature's own arrow to the answer; nothing downstream reacts.", color: `var(${LADDER_VARS[0]})` },
  { key: "replacement", label: "Replacement", desc: "The twin, with later features allowed to switch on and off.", color: `var(${LADDER_VARS[2]})` },
  { key: "replacement_ln", label: "+ live LayerNorm", desc: "As above, with LayerNorm reacting to the edit as in the real model.", color: `var(${LADDER_VARS[3]})` },
  { key: "replacement_full", label: "+ live attention", desc: "As above, with attention reacting too: the most faithful twin.", color: `var(${LADDER_VARS[4]})` },
  { key: "linear_fulljac", label: "Graph with the fix", desc: "The proposed fix: still a simple linear graph, but it includes how LayerNorm and attention react.", color: "var(--fix)", strong: true, fix: true },
];

/** The rows whose level is in this response, each with its result. */
export function presentRows(levels: WhatIfResponse["levels"]): (Row & { result: LevelResult })[] {
  return ROWS.flatMap((r) => {
    const result = levels[r.key];
    return result ? [{ ...r, result }] : [];
  });
}
