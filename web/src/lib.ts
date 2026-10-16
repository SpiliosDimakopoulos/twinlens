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

/** Signed value with a true minus sign: two decimals, three below 0.1. */
export function signed(v: number): string {
  const a = Math.abs(v);
  return `${v < 0 ? "−" : "+"}${a < 0.1 ? a.toFixed(3) : a.toFixed(2)}`;
}

/** One plain-language sentence comparing the graph's prediction with what GPT-2 did. */
export function verdict(graph: number, real: number): string {
  if (Math.abs(real) < 0.01 && Math.abs(graph) < 0.01) return "Neither the graph nor GPT-2 sees a real effect.";
  const head = `The graph predicts ${signed(graph)}. GPT-2 moves ${signed(real)}`;
  const sameSign = Math.sign(graph) === Math.sign(real) || Math.abs(real) < 0.01;
  const ratio = Math.abs(graph) / Math.max(Math.abs(real), 1e-3);
  if (!sameSign) return `${head}: the opposite direction.`;
  if (ratio > 2) return `${head}: overstated ${ratio.toFixed(ratio > 10 ? 0 : 1)}×.`;
  if (ratio < 0.5) return `${head}: understated.`;
  return `${head}: about right.`;
}
