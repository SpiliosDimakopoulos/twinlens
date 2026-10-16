/** Small hand-drawn-weight SVG glyphs (1.6px strokes on a 16px grid) used instead of emoji. */
type Name = "check" | "cross" | "alert" | "up" | "down" | "sun" | "moon" | "auto" | "play" | "stop" | "github" | "doc" | "arrow";

const PATHS: Record<Name, string> = {
  check: "M3.5 8.5l3 3 6-7",
  cross: "M4.5 4.5l7 7M11.5 4.5l-7 7",
  alert: "M8 4v5M8 11.6v.4",
  up: "M8 12.5V3.5M4.5 7L8 3.5 11.5 7",
  down: "M8 3.5v9M4.5 9L8 12.5 11.5 9",
  sun: "M8 5.2a2.8 2.8 0 100 5.6 2.8 2.8 0 000-5.6zM8 1.8v1.4M8 12.8v1.4M1.8 8h1.4M12.8 8h1.4M3.6 3.6l1 1M11.4 11.4l1 1M3.6 12.4l1-1M11.4 4.6l1-1",
  moon: "M12.8 9.6A5.2 5.2 0 016.4 3.2a5.2 5.2 0 106.4 6.4z",
  auto: "M8 2.4a5.6 5.6 0 100 11.2V2.4z M8 2.4a5.6 5.6 0 110 11.2",
  play: "M5.5 4l6 4-6 4z",
  stop: "M5 5h6v6H5z",
  github: "M8 1.8a6.2 6.2 0 00-2 12.1c.3 0 .4-.1.4-.3v-1.1c-1.7.4-2.1-.8-2.1-.8-.3-.7-.7-.9-.7-.9-.6-.4 0-.4 0-.4.6 0 1 .6 1 .6.5.9 1.4.7 1.8.5 0-.4.2-.7.4-.8-1.4-.2-2.8-.7-2.8-3 0-.7.2-1.2.6-1.6 0-.2-.3-.8.1-1.6 0 0 .5-.2 1.7.6a5.8 5.8 0 013.1 0c1.2-.8 1.7-.6 1.7-.6.3.8.1 1.4.1 1.6.4.4.6 1 .6 1.6 0 2.4-1.4 2.9-2.8 3 .2.2.4.6.4 1.1v1.7c0 .2.1.4.4.3A6.2 6.2 0 008 1.8z",
  doc: "M4.5 2.5h5l2.5 2.5v8.5h-7.5zM9.5 2.5V5H12M6 8h4M6 10.5h4",
  arrow: "M3 8h10M9.5 4.5L13 8l-3.5 3.5",
};

export function Glyph({ name, size = 14, className = "", title }: { name: Name; size?: number; className?: string; title?: string }) {
  const filled = name === "github" || name === "play" || name === "stop";
  return (
    <svg width={size} height={size} viewBox="0 0 16 16" className={className} aria-hidden={title ? undefined : true}
      role={title ? "img" : undefined} aria-label={title}
      fill={filled ? "currentColor" : "none"} stroke={filled ? "none" : "currentColor"}
      strokeWidth={1.6} strokeLinecap="round" strokeLinejoin="round">
      <path d={PATHS[name]} />
    </svg>
  );
}
