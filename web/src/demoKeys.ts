/** Pure helpers the static demo uses to look up precomputed what-if payloads. */

/** The element of `xs` closest to `x` (the first one on a tie), or undefined when `xs` is empty. */
export function nearest(xs: readonly number[], x: number): number | undefined {
  let best: number | undefined;
  for (const v of xs) if (best === undefined || Math.abs(v - x) < Math.abs(best - x)) best = v;
  return best;
}

/** A multiplier as Python's str(float) prints it, the key format of the demo files: "0.0", "0.5", "2.0". */
export function presetKey(m: number): string {
  return Number.isInteger(m) ? m.toFixed(1) : String(m);
}
