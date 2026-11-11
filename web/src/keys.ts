/** Keyboard-shortcut helpers. */

const TEXT_TAGS = new Set(["INPUT", "TEXTAREA", "SELECT"]);
const KEY_ROLES = new Set(["radio", "slider", "tab"]);

/** True when the keyboard event comes from a control that uses these keys itself
 * (text fields, selects, sliders, radio groups), so page shortcuts must not fire. */
export function isTypingTarget(e: KeyboardEvent): boolean {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return false;
  return TEXT_TAGS.has(t.tagName) || t.isContentEditable || KEY_ROLES.has(t.getAttribute("role") ?? "");
}
