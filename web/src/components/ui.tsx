import type { ReactNode } from "react";
import type { TopToken } from "../api";

export function Card({ title, subtitle, children, right }: {
  title: string; subtitle?: ReactNode; children: ReactNode; right?: ReactNode;
}) {
  return (
    <section className="card p-4 sm:p-5">
      <header className="mb-3 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-sm font-semibold text-ink">{title}</h2>
          {subtitle && <p className="mt-0.5 text-xs text-ink2">{subtitle}</p>}
        </div>
        {right}
      </header>
      {children}
    </section>
  );
}

/** Horizontal probability bars: thin marks, rounded data-end, labels in ink (not series colour). */
export function TokenBars({ items, max, color = "var(--series-1)" }: {
  items: TopToken[]; max?: number; color?: string;
}) {
  const m = max ?? Math.max(...items.map((i) => i.prob), 1e-9);
  return (
    <ul className="space-y-1.5">
      {items.map((t) => (
        <li key={t.id} className="grid grid-cols-[7rem_1fr_3.5rem] items-center gap-2 text-xs">
          <span className="tok truncate text-ink" title={JSON.stringify(t.token)}>{JSON.stringify(t.token).slice(1, -1)}</span>
          <span className="h-2 rounded-r" style={{ width: `${(t.prob / m) * 100}%`, background: color, minWidth: 2 }} />
          <span className="tnum text-right text-ink2">{(t.prob * 100).toFixed(1)}%</span>
        </li>
      ))}
    </ul>
  );
}

export function Button({ children, onClick, disabled, kind = "primary" }: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; kind?: "primary" | "ghost";
}) {
  const base = "rounded-lg px-3.5 py-2 text-sm font-medium transition disabled:opacity-50";
  const style = kind === "primary"
    ? { background: "var(--series-1)", color: "#fff" }
    : { background: "transparent", color: "var(--text-primary)", border: "1px solid var(--border)" };
  return (
    <button className={base} style={style} onClick={onClick} disabled={disabled}>{children}</button>
  );
}

export function Stat({ label, value, note }: { label: string; value: string; note?: string }) {
  return (
    <div className="card p-4">
      <div className="text-xs text-ink2">{label}</div>
      <div className="mt-1 text-3xl font-semibold text-ink">{value}</div>
      {note && <div className="mt-1 text-xs text-muted">{note}</div>}
    </div>
  );
}

export { tokenText } from "../lib";

export function ErrorNote({ error }: { error: string | null }) {
  if (!error) return null;
  return (
    <p className="rounded-lg border px-3 py-2 text-sm" style={{ borderColor: "var(--critical)", color: "var(--text-primary)" }}>
      <span aria-hidden>⚠ </span>{error}
    </p>
  );
}
