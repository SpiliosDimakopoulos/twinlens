import { interpolateRgb } from "d3";
import { useEffect, useState } from "react";

const NAMES = [
  "--surface-1", "--text-primary", "--text-secondary", "--text-muted", "--gridline", "--baseline",
  "--series-1", "--series-2", "--series-3", "--div-neg", "--div-mid", "--div-pos", "--seq-0",
  "--seq-1", "--lv-1", "--lv-2", "--lv-3", "--lv-4", "--lv-5", "--fix", "--real", "--good", "--warning",
  "--critical",
] as const;
export type Tokens = Record<(typeof NAMES)[number], string>;

function read(): Tokens {
  const s = getComputedStyle(document.documentElement);
  return Object.fromEntries(NAMES.map((n) => [n, s.getPropertyValue(n).trim()])) as Tokens;
}

/** Live colour tokens; re-read when the OS theme or the data-theme attribute changes. */
export function useTokens(): Tokens {
  const [t, setT] = useState<Tokens>(read);
  useEffect(() => {
    const mq = matchMedia("(prefers-color-scheme: dark)");
    const update = () => setT(read());
    mq.addEventListener("change", update);
    const obs = new MutationObserver(update);
    obs.observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
    return () => {
      mq.removeEventListener("change", update);
      obs.disconnect();
    };
  }, []);
  return t;
}

/** Diverging scale: negative -> red, 0 -> neutral gray, positive -> blue. */
export function diverging(t: Tokens, maxAbs: number) {
  const neg = interpolateRgb(t["--div-mid"], t["--div-neg"]);
  const pos = interpolateRgb(t["--div-mid"], t["--div-pos"]);
  return (v: number) => {
    const x = Math.min(1, Math.abs(v) / (maxAbs || 1));
    return v < 0 ? neg(x) : pos(x);
  };
}

/** Sequential single-hue scale for magnitudes in [0, max]. */
export function sequential(t: Tokens, max = 1) {
  const f = interpolateRgb(t["--seq-0"], t["--seq-1"]);
  return (v: number) => f(Math.max(0, Math.min(1, v / (max || 1))) ** 0.8);
}

export const LADDER_VARS = ["--lv-1", "--lv-2", "--lv-3", "--lv-4", "--lv-5"] as const;
