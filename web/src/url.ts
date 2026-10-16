/** Tiny URL-state helpers so any view of the dashboard can be shared as a link. */

export function readParam(name: string): string | null {
  try {
    return new URLSearchParams(location.search).get(name);
  } catch {
    return null;
  }
}

export function writeParams(values: Record<string, string | number | null | undefined>): void {
  try {
    const params = new URLSearchParams(location.search);
    for (const [k, v] of Object.entries(values)) {
      if (v === null || v === undefined || v === "") params.delete(k);
      else params.set(k, String(v));
    }
    const qs = params.toString();
    history.replaceState(null, "", `${location.pathname}${qs ? `?${qs}` : ""}${location.hash}`);
  } catch {
    /* sandboxed or file:// contexts: sharing links just won't update */
  }
}

/** True when the keyboard event comes from a text field (shortcuts should not fire). */
export function typing(e: KeyboardEvent): boolean {
  const t = e.target as HTMLElement | null;
  return !!t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.isContentEditable);
}
