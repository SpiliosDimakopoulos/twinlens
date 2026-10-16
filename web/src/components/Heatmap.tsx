import type { ReactNode } from "react";
import { useTooltip } from "./Tooltip";

type Props = {
  rows: number;
  cols: number;
  color: (r: number, c: number) => string;
  rowLabel?: (r: number) => string;
  colLabel?: (c: number) => string;
  cellText?: (r: number, c: number) => string;
  tooltip?: (r: number, c: number) => ReactNode;
  outlined?: (r: number, c: number) => boolean;
  onClick?: (r: number, c: number) => void;
  cell?: number;
  labelWidth?: number;
  ariaLabel: string;
};

/** SVG grid with a 2px surface gap between cells, per-cell hover tooltip and optional click. */
export function Heatmap({
  rows, cols, color, rowLabel, colLabel, cellText, tooltip, outlined, onClick,
  cell = 26, labelWidth = 44, ariaLabel,
}: Props) {
  const { show, hide } = useTooltip();
  const top = colLabel ? 64 : 4;
  const w = labelWidth + cols * cell;
  const h = top + rows * cell;
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" style={{ maxWidth: w * 1.4 }} role="img" aria-label={ariaLabel}>
      {colLabel &&
        Array.from({ length: cols }, (_, c) => (
          <text
            key={c}
            transform={`translate(${labelWidth + c * cell + cell / 2 + 3}, ${top - 6}) rotate(-55)`}
            fontSize={10}
            fill="var(--text-secondary)"
            className="tok"
          >
            {colLabel(c)}
          </text>
        ))}
      {rowLabel &&
        Array.from({ length: rows }, (_, r) => (
          <text key={r} x={labelWidth - 6} y={top + r * cell + cell / 2 + 3.5} fontSize={10}
            textAnchor="end" fill="var(--text-muted)" className="tnum">
            {rowLabel(r)}
          </text>
        ))}
      {Array.from({ length: rows }, (_, r) =>
        Array.from({ length: cols }, (_, c) => {
          const x = labelWidth + c * cell;
          const y = top + r * cell;
          const isOut = outlined?.(r, c);
          return (
            <g
              key={`${r}-${c}`}
              onMouseMove={(e) => tooltip && show(e, tooltip(r, c))}
              onMouseLeave={hide}
              onClick={() => onClick?.(r, c)}
              style={{ cursor: onClick ? "pointer" : "default" }}
            >
              <rect x={x + 1} y={y + 1} width={cell - 2} height={cell - 2} rx={3} fill={color(r, c)} />
              {isOut && (
                <rect x={x + 1.5} y={y + 1.5} width={cell - 3} height={cell - 3} rx={3} fill="none"
                  stroke="var(--text-primary)" strokeWidth={2} />
              )}
              {cellText && (
                <text x={x + cell / 2} y={y + cell / 2 + 3.5} fontSize={8.5} textAnchor="middle"
                  fill="var(--text-primary)" className="tok" style={{ pointerEvents: "none" }}>
                  {cellText(r, c)}
                </text>
              )}
            </g>
          );
        }),
      )}
    </svg>
  );
}
