import type { GenerateStep } from "./api";

/** Printable form of a token: escapes newlines/tabs so whitespace tokens stay visible. */
export function tokenText(t: string): string {
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
