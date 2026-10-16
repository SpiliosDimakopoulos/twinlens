import type { GenerateStep } from "./api";

/** Printable form of a token: escapes newlines/tabs so whitespace tokens stay visible. */
export function tokenText(t: string): string {
  if (t === "<|endoftext|>") return "BOS";
  return JSON.stringify(t).slice(1, -1);
}

/** First layer from which the logit lens ranks the final token top-1 at every later layer. */
export function decidedAt(step: Pick<GenerateStep, "token" | "lens">): number {
  for (let l = 0; l < step.lens.length; l++)
    if (step.lens.slice(l).every((t) => t === step.token)) return l;
  return step.lens.length - 1;
}

/** Trust band shown next to a what-if result (status colour + icon + label, never colour alone). */
export function trustBand(trust: number): { status: "good" | "warning" | "critical"; icon: string; label: string } {
  if (trust >= 0.7) return { status: "good", icon: "✓", label: "Trust the graph" };
  if (trust >= 0.4) return { status: "warning", icon: "!", label: "Check before relying on it" };
  return { status: "critical", icon: "✕", label: "Don't trust the graph here" };
}

export const SIGNAL_LABELS: Record<string, string> = {
  layers_after: "layers between edit and output",
  pos_from_end: "distance to the last token",
  activation: "feature activation",
  rel_edit_size: "edit size vs. residual stream",
  abs_direct_effect: "size of the graph edge",
  downstream_err_frac: "unexplained (error) share downstream",
  attn_to_pos: "attention from the output to this token",
  n_active_downstream: "active features downstream",
  clean_top1_prob: "model confidence",
};

/** One plain-language sentence comparing the graph's prediction with what GPT-2 did. */
export function verdict(graph: number, real: number, target: string): string {
  const t = `“${tokenText(target)}”`;
  const fmt = (v: number) => `${v < 0 ? "lowers" : "raises"} log p(${t}) by ${Math.abs(v).toFixed(2)}`;
  if (Math.abs(real) < 0.01 && Math.abs(graph) < 0.01) return `Neither the graph nor GPT-2 sees any real effect on ${t}.`;
  const sameSign = Math.sign(graph) === Math.sign(real) || Math.abs(real) < 0.01;
  const ratio = Math.abs(graph) / Math.max(Math.abs(real), 1e-3);
  if (!sameSign) return `The graph says the edit ${fmt(graph)}, but GPT-2 moves the other way: it ${fmt(real)}.`;
  if (ratio > 2) return `The graph says the edit ${fmt(graph)}; GPT-2 only ${fmt(real)}. The graph overstates the effect ${ratio.toFixed(ratio > 10 ? 0 : 1)}×.`;
  if (ratio < 0.5) return `The graph says the edit ${fmt(graph)}; GPT-2 ${fmt(real)}. The graph understates the effect.`;
  return `The graph says the edit ${fmt(graph)}; GPT-2 ${fmt(real)}. The graph gets it about right.`;
}
