/** Placeholder blocks shown while GPT-2 and the twin are running. */
export function Skeleton({ className = "", style }: { className?: string; style?: React.CSSProperties }) {
  return (
    <div
      aria-hidden
      className={`animate-pulse rounded-lg ${className}`}
      style={{ background: "var(--gridline)", ...style }}
    />
  );
}

export function Stage({ steps, active }: { steps: string[]; active: number }) {
  return (
    <ol className="flex flex-wrap gap-x-4 gap-y-1 text-xs" aria-live="polite">
      {steps.map((s, i) => (
        <li key={s} className="flex items-center gap-1.5"
          style={{ color: i <= active ? "var(--text-primary)" : "var(--text-muted)" }}>
          <span aria-hidden className={i === active ? "animate-pulse" : ""}>
            {i < active ? "✓" : i === active ? "●" : "○"}
          </span>
          {s}
        </li>
      ))}
    </ol>
  );
}
