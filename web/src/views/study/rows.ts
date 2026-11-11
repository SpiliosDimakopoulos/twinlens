import type { Level } from "../../api";
import { LADDER_VARS } from "../../components/theme";

export type LevelRow = { key: Level; label: string; color: string };

/** The ladder of twins, from direct edges up to the most faithful twin. */
export const LADDER: LevelRow[] = [
  { key: "direct", label: "Direct edges only", color: `var(${LADDER_VARS[0]})` },
  { key: "linear", label: "Graph today (linear)", color: `var(${LADDER_VARS[1]})` },
  { key: "replacement", label: "Replacement", color: `var(${LADDER_VARS[2]})` },
  { key: "replacement_ln", label: "+ live LayerNorm", color: `var(${LADDER_VARS[3]})` },
  { key: "replacement_full", label: "+ live attention", color: `var(${LADDER_VARS[4]})` },
];

/** Today's graph, the two parts of the fix, and the nonlinear twin they are measured against. */
export const FIXES: LevelRow[] = [
  { key: "linear", label: "Graph today", color: "var(--baseline)" },
  { key: "linear_lnjac", label: "Fix, LayerNorm part only", color: "color-mix(in srgb, var(--fix) 55%, transparent)" },
  { key: "linear_fulljac", label: "Graph with the fix", color: "var(--fix)" },
  { key: "replacement_full", label: "Nonlinear twin (reference)", color: "var(--real)" },
];
