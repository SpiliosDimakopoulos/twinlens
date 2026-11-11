import type { GenerateStep } from "./api";
import { signed } from "./format";

/** Printable form of a token: escapes newlines/tabs so whitespace tokens stay visible. */
export function tokenText(t: string): string {
  if (t === "<|endoftext|>") return "‹start›";
  return JSON.stringify(t).slice(1, -1);
}

/** grid[r][c], or `fallback` when either index is outside the grid. */
export function cell<T>(grid: readonly (readonly T[])[] | undefined, r: number, c: number, fallback: T): T {
  return grid?.[r]?.[c] ?? fallback;
}

/** First layer from which the logit lens ranks the final token top-1 at every later layer. */
export function decidedAt(step: Pick<GenerateStep, "token" | "lens">): number {
  for (let l = 0; l < step.lens.length; l++)
    if (step.lens.slice(l).every((t) => t === step.token)) return l;
  return step.lens.length - 1;
}

/** Trust band shown next to a what-if result (status colour + glyph + label, never colour alone). */
export function trustBand(trust: number): { status: "good" | "warning" | "critical"; label: string } {
  if (trust >= 0.7) return { status: "good", label: "Trust the graph" };
  if (trust >= 0.4) return { status: "warning", label: "Check before relying on it" };
  return { status: "critical", label: "Don't trust the graph here" };
}

/** The nine reliability signals (twinlens/fidelity/signals.py): a short name for charts and a
 * phrase that reads inside a sentence ("<phrase> raises it"). */
export const SIGNALS: Record<string, { name: string; phrase: string }> = {
  layers_after: { name: "Layers after the edit", phrase: "number of layers after the edit" },
  pos_from_end: { name: "Distance to the last word", phrase: "distance from the edited word to the last word" },
  activation: { name: "Feature strength", phrase: "how strongly the feature is active" },
  rel_edit_size: { name: "Relative size of the edit", phrase: "size of the edit compared with the model's internal state" },
  abs_direct_effect: { name: "Size of the graph arrow", phrase: "size of the feature's arrow in the graph" },
  downstream_err_frac: { name: "Unexplained share later on", phrase: "share of later computation the twin cannot explain" },
  attn_to_pos: { name: "Attention to the edited word", phrase: "how much the last word attends to the edited word" },
  n_active_downstream: { name: "Active features later on", phrase: "number of active features after the edit" },
  clean_top1_prob: { name: "GPT-2's confidence", phrase: "how confident GPT-2 is in its answer" },
};

/** Display name of a task family key: "ioi" -> "IOI", "factual" -> "Factual". */
export function familyLabel(k: string): string {
  return k === "ioi" ? "IOI" : k.charAt(0).toUpperCase() + k.slice(1);
}

/** One plain-language sentence comparing the graph's prediction with what GPT-2 did. */
export function verdict(graph: number, real: number): string {
  if (Math.abs(real) < 0.01 && Math.abs(graph) < 0.01) return "Neither the graph nor GPT-2 sees a real effect.";
  const head = `The graph predicts ${signed(graph)}. GPT-2 actually moves ${signed(real)}`;
  const sameSign = Math.sign(graph) === Math.sign(real) || Math.abs(real) < 0.01;
  const ratio = Math.abs(graph) / Math.max(Math.abs(real), 1e-3);
  if (!sameSign) return `${head}: the opposite direction.`;
  if (ratio > 2) return `${head}: the graph overstates the effect ${ratio.toFixed(ratio > 10 ? 0 : 1)}×.`;
  if (ratio < 0.5) return `${head}: the graph understates the effect.`;
  return `${head}: about right.`;
}
