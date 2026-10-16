import type { ReactNode } from "react";
import type { TopToken } from "../api";
import { Glyph } from "./Glyph";

/**
 * A figure plate: a numbered label, a title and an optional caption above the content, in the
 * manner of a printed paper. Used for every chart and panel.
 */
export function Card({ title, subtitle, children, right, label }: {
  title: ReactNode; subtitle?: ReactNode; children: ReactNode; right?: ReactNode; label?: string;
}) {
  return (
    <section className="plate min-w-0 px-4 pb-4 pt-3.5 sm:px-5 sm:pb-5">
      <header className="mb-3 flex items-start justify-between gap-4">
        <div className="min-w-0">
          {label && <div className="figlabel mb-1">{label}</div>}
          <h2 className="text-[15px] font-semibold leading-snug text-ink">{title}</h2>
          {subtitle && <p className="mt-1 max-w-prose text-[13px] leading-relaxed text-ink2">{subtitle}</p>}
        </div>
        {right}
      </header>
      {children}
    </section>
  );
}

/** Probability bars on an absolute 0-100% scale: hairline track, ink labels. */
export function TokenBars({ items, max = 1, color = "var(--series-1)" }: {
  items: TopToken[]; max?: number; color?: string;
}) {
  return (
    <ul className="space-y-[7px]">
      {items.map((t) => (
        <li key={t.id} className="grid grid-cols-[7.5rem_1fr_3.25rem] items-center gap-3 text-[13px]">
          <span className="tok truncate text-ink" title={JSON.stringify(t.token)}>{JSON.stringify(t.token).slice(1, -1)}</span>
          <span className="relative h-[6px] rounded-full" style={{ background: "var(--gridline)" }}>
            <span className="absolute inset-y-0 left-0 rounded-full" style={{ width: `${Math.max((t.prob / max) * 100, 0.6)}%`, background: color }} />
          </span>
          <span className="tnum text-right text-ink2">{(t.prob * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

export function Button({ children, onClick, disabled, kind = "primary", title }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; kind?: "primary" | "ghost"; title?: string;
}) {
  const base = "inline-flex items-center gap-1.5 whitespace-nowrap rounded-[5px] px-3.5 py-2 text-[13px] font-medium transition-colors disabled:opacity-45";
  const style = kind === "primary"
    ? { background: "var(--text-primary)", color: "var(--page)" }
    : { background: "transparent", color: "var(--text-primary)", boxShadow: "inset 0 0 0 1px var(--baseline)" };
  return (
    <button className={base} style={style} onClick={onClick} disabled={disabled} title={title}>{children}</button>
  );
}

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="border-t pt-3 rule">
      <div className="kicker">{label}</div>
      <div className="serif mt-1 text-[34px] leading-none tracking-tight text-ink tnum">{value}</div>
      {note && <div className="mt-1.5 text-[12px] leading-snug text-ink2">{note}</div>}
    </div>
  );
}

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p className="flex items-start gap-2 rounded-[5px] px-3 py-2 text-[13px]"
      style={{ boxShadow: "inset 3px 0 0 var(--critical)", background: "var(--surface-1)", color: "var(--text-primary)" }}>
      <Glyph name="alert" className="mt-0.5 shrink-0" />{error.replace(/^Error:\s*/, "")}
    </p>
  );
}
