import { useId, useState, type ReactNode } from "react";

/** A small "?" that explains a term on hover, focus or tap. */
export function Info({ children, label = "What is this?" }: { children: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  const id = useId();
  return (
    <span className="relative inline-flex align-middle">
      <button
        type="button"
        aria-label={label}
        aria-describedby={open ? id : undefined}
        onMouseEnter={() => setOpen(true)}
        onMouseLeave={() => setOpen(false)}
        onFocus={() => setOpen(true)}
        onBlur={() => setOpen(false)}
        onClick={() => setOpen((o) => !o)}
        className="ml-1 inline-flex h-4 w-4 items-center justify-center rounded-full text-[10px] font-semibold"
        style={{ border: "1px solid var(--baseline)", color: "var(--text-secondary)" }}
      >
        ?
      </button>
      {open && (
        <span
          id={id}
          role="tooltip"
          className="absolute left-1/2 top-6 z-40 w-64 -translate-x-1/2 rounded-lg px-3 py-2 text-xs font-normal leading-relaxed shadow-lg"
          style={{ background: "var(--surface-2)", border: "1px solid var(--border)", color: "var(--text-primary)" }}
        >
          {children}
        </span>
      )}
    </span>
  );
}
