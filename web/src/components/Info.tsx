import { useCallback, useEffect, useId, useLayoutEffect, useRef, useState, type ReactNode } from "react";
import { createPortal } from "react-dom";

const GUTTER = 8;

/** A small "?" that explains a term on hover, focus or tap.
 * The popover is portalled to <body> with fixed positioning, so scrolling tables and cards
 * cannot clip it. It is kept inside the viewport, follows its button on scroll and resize,
 * stays open while the pointer is over it, and closes on Escape or a tap elsewhere. */
export function Info({ children, label = "What is this?", wide = false }: { children: ReactNode; label?: string; wide?: boolean }) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{ left: number; top: number; maxH: number } | null>(null);
  const btn = useRef<HTMLButtonElement>(null);
  const pop = useRef<HTMLSpanElement>(null);
  const leave = useRef<ReturnType<typeof setTimeout> | undefined>(undefined);
  const id = useId();

  const place = useCallback(() => {
    if (!btn.current || !pop.current) return;
    const b = btn.current.getBoundingClientRect();
    const w = pop.current.offsetWidth, h = pop.current.scrollHeight;
    const vw = document.documentElement.clientWidth, vh = window.innerHeight;
    const left = Math.min(Math.max(GUTTER, b.left + b.width / 2 - w / 2), vw - w - GUTTER);
    const below = vh - b.bottom - 6 - GUTTER, above = b.top - 6 - GUTTER;
    const up = h > below && above > below;
    const maxH = Math.max(80, up ? above : below);
    setPos({ left, top: up ? b.top - 6 - Math.min(h, maxH) : b.bottom + 6, maxH });
  }, []);

  useLayoutEffect(() => { if (open) place(); else setPos(null); }, [open, place]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => { if (e.key === "Escape") setOpen(false); };
    const onDown = (e: PointerEvent) => {
      const t = e.target as Node;
      if (!btn.current?.contains(t) && !pop.current?.contains(t)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("pointerdown", onDown);
    window.addEventListener("scroll", place, true);
    window.addEventListener("resize", place);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("pointerdown", onDown);
      window.removeEventListener("scroll", place, true);
      window.removeEventListener("resize", place);
    };
  }, [open, place]);

  useEffect(() => () => clearTimeout(leave.current), []);
  const show = () => { clearTimeout(leave.current); setOpen(true); };
  // A short delay lets the pointer travel from the button onto the popover (WCAG 1.4.13).
  const hideSoon = () => { clearTimeout(leave.current); leave.current = setTimeout(() => setOpen(false), 150); };

  return (
    <span className="relative inline-flex align-middle">
      <button
        ref={btn}
        type="button"
        aria-label={label}
        aria-expanded={open}
        aria-describedby={open ? id : undefined}
        onMouseEnter={show}
        onMouseLeave={hideSoon}
        onFocus={show}
        onBlur={hideSoon}
        onClick={show}
        // 16px circle with a 24px hit area (WCAG 2.2 target size).
        className="relative ml-1 inline-flex h-4 w-4 shrink-0 items-center justify-center rounded-full font-sans text-[10px] font-semibold normal-case before:absolute before:-inset-1 before:content-['']"
        style={{ border: "1px solid var(--text-muted)", color: "var(--text-secondary)" }}
      >
        ?
      </button>
      {open && createPortal(
        <span
          ref={pop}
          id={id}
          role="tooltip"
          onMouseEnter={show}
          onMouseLeave={hideSoon}
          className={`fixed z-50 block overflow-y-auto ${wide ? "w-[min(22rem,calc(100vw-16px))]" : "w-[min(16rem,calc(100vw-16px))]"} rounded-lg px-3 py-2 text-left font-sans text-xs font-normal normal-case leading-relaxed tracking-normal shadow-lg`}
          style={{
            left: pos?.left ?? 0, top: pos?.top ?? 0, maxHeight: pos?.maxH, visibility: pos ? "visible" : "hidden",
            background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)",
          }}
        >
          {children}
        </span>,
        document.body,
      )}
    </span>
  );
}
