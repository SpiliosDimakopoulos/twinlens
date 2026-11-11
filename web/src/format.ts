/** Number formatting shared by every view. Negative numbers use a true minus sign (U+2212). */

const MINUS = "−";

/** Replaces ASCII hyphen-minus signs with a true minus sign. */
export function trueMinus(s: string): string {
  return s.replace(/-/g, MINUS);
}

/** Explicit sign (+ or −) followed by |v| to a fixed number of decimals. */
export function signedFixed(v: number, digits: number): string {
  return `${v < 0 ? MINUS : "+"}${Math.abs(v).toFixed(digits)}`;
}

/** Signed value with a true minus sign: two decimals, three below 0.1. */
export function signed(v: number): string {
  return signedFixed(v, Math.abs(v) < 0.1 ? 3 : 2);
}

/** A share in [0, 1] as a percentage with a fixed number of decimals: 0.123 -> "12.3%". */
export function percent(v: number, digits: number): string {
  return `${(v * 100).toFixed(digits)}%`;
}

/** A percentage whose precision grows as the share shrinks: 2 decimals below 1%, 1 below 10%, else 0. */
export function adaptivePercent(v: number): string {
  return percent(v, v < 0.01 ? 2 : v < 0.1 ? 1 : 0);
}
