import { useState } from "react";
import type { Feature } from "../api";
import { diverging, useTokens } from "../components/theme";
import { useTooltip } from "../components/useTooltip";
import { tokenText } from "../lib";

type Props = {
  tokens: string[];
  target: string;
  features: Feature[];
  selected: Feature | null;
  onSelect: (f: Feature) => void;
};

const KEY = (f: Feature) => `${f.layer}-${f.pos}-${f.feature}`;

// Geometry, in viewBox units.
const ROW_H = 30;        // height of one layer row
const LEFT = 34;         // left column for the layer labels
const TOP = 50;          // headroom above layer 11 for the output pill and the edges into it
const RIGHT = 20;        // right margin
const BOTTOM = 58;       // token labels (H - 32) and the legend (H - 10) below the rows
const OUT_Y = 20;        // centre of the output pill
const PILL_W = 128, PILL_H = 26; // output pill size
const NODE_STEP = 17;    // max spacing between nodes that share a (layer, word) cell
const CELL_PAD = 14;     // horizontal room kept free inside a cell when nodes are spread out
const HIT_PAD = 7;       // invisible hit area around each node
const RING_PAD = 3.5;    // selection/hover ring outside the node

/**
 * The reasoning path: each node is a transcoder feature placed at (token, layer), sized and
 * coloured by its direct effect on the predicted token (blue pushes it up, red pushes it down).
 * Edges run to the output node. Hovering a node isolates its edge; click (or Enter) to ask the
 * twin "what if?".
 */
export function ReasoningGraph({ tokens, target, features, selected, onSelect }: Props) {
  const t = useTokens();
  const { show, hide } = useTooltip();
  const [hover, setHover] = useState<string | null>(null);
  // Sized so the figure renders close to 1:1 in its column (text stays legible).
  // Columns between 40 and 84 wide, aiming for 600 in total.
  const colW = Math.max(40, Math.min(84, 600 / tokens.length));
  const rowH = ROW_H;
  const left = LEFT;
  const top = TOP;
  const W = left + colW * tokens.length + RIGHT;
  const H = top + rowH * 12 + BOTTOM;
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.effect)), 1e-6);
  const scale = diverging(t, maxAbs);
  // Floor the colour intensity so weak nodes stay visible against the surface.
  const color = (e: number) => scale(Math.sign(e) * Math.max(Math.abs(e), 0.3 * maxAbs));
  const out = { x: left + colW * (tokens.length - 0.5), y: OUT_Y };  // above the last word
  const focus = hover;  // only hovering isolates a node; selection is shown by its ring

  const cellCount = new Map<string, number>();
  for (const f of features) cellCount.set(`${f.layer}-${f.pos}`, (cellCount.get(`${f.layer}-${f.pos}`) ?? 0) + 1);
  const slots = new Map<string, number>();
  const placed = features.map((f) => {
    const k = `${f.layer}-${f.pos}`;
    const i = slots.get(k) ?? 0;
    slots.set(k, i + 1);
    const n = cellCount.get(k) ?? 1;
    const step = Math.min(NODE_STEP, (colW - CELL_PAD) / n);
    const x = left + colW * (f.pos + 0.5) + (i - (n - 1) / 2) * step;
    const y = top + rowH * (11 - f.layer) + rowH / 2;
    return { f, x, y, r: 4 + 5 * Math.sqrt(Math.abs(f.effect) / maxAbs) };  // radius 4 to 9, area ~ effect
  });

  const tip = (f: Feature) => (
    <div className="space-y-1">
      <div className="font-semibold">L{f.layer} · feature {f.feature}</div>
      <div>active on <span className="tok">“{tokenText(f.token)}”</span> (strength {f.activation.toFixed(2)})</div>
      <div>{f.effect >= 0 ? "pushes" : "pulls"} “{tokenText(target)}” {f.effect >= 0 ? "up" : "down"} by <b className="tnum">{Math.abs(f.effect).toFixed(3)}</b> logits</div>
      <div className="text-ink2">words it pushes up most: <span className="tok">{f.label.map(tokenText).join(" · ")}</span></div>
      <div className="text-muted">click to test switching it off</div>
    </div>
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 560 }}
      aria-label={`Reasoning path: features with the largest direct effect on predicting ${tokenText(target)}`}>
      <text x={left - 8} y={top - 8} fontSize={9.5} textAnchor="end" fill="var(--text-muted)">layer</text>
      {Array.from({ length: 12 }, (_, l) => (
        <g key={l}>
          <line x1={left} x2={W - 20} y1={top + rowH * (11 - l) + rowH / 2} y2={top + rowH * (11 - l) + rowH / 2}
            stroke="var(--gridline)" strokeWidth={1} />
          <text x={left - 8} y={top + rowH * (11 - l) + rowH / 2 + 3.5} fontSize={10} textAnchor="end"
            fill="var(--text-muted)" className="tnum">L{l}</text>
        </g>
      ))}
      {tokens.map((_, i) => i === tokens.length - 1 && (
        <rect key="last" x={left + colW * i + 2} y={top} width={colW - 4} height={rowH * 12} rx={8}
          fill="var(--series-1)" fillOpacity={0.05} />
      ))}
      {placed.map(({ f, x, y }) => {
        const k = KEY(f);
        const dim = focus !== null && focus !== k;
        const strong = focus === k || (focus === null && selected !== null && KEY(selected) === k);
        return (
          <path key={`e-${k}`} fill="none"
            // Cubic from the node up into the bottom edge of the output pill (out.y + PILL_H / 2).
            d={`M${x},${y} C${x},${y - (y - out.y) * 0.55} ${out.x},${out.y + 12 + (y - out.y) * 0.35} ${out.x},${out.y + PILL_H / 2}`}
            stroke={color(f.effect)} strokeOpacity={dim ? 0.07 : strong ? 0.9 : 0.32} strokeLinecap="round"
            strokeWidth={0.75 + 2.5 * (Math.abs(f.effect) / maxAbs) + (strong ? 1 : 0)}
            style={{ transition: "stroke-opacity 150ms" }} />
        );
      })}
      <g>
        <rect x={out.x - PILL_W / 2} y={out.y - PILL_H / 2} width={PILL_W} height={PILL_H} rx={PILL_H / 2} fill="var(--text-primary)" />
        <text x={out.x} y={out.y + 4} fontSize={11.5} textAnchor="middle" fill="var(--surface-1)" className="tok">
          → {tokenText(target)}
        </text>
      </g>
      {placed.map(({ f, x, y, r }) => {
        const k = KEY(f);
        const isSel = selected && KEY(selected) === k;
        const dim = focus !== null && focus !== k && !isSel;
        return (
          <g key={k} role="button" tabIndex={0} style={{ cursor: "pointer", outline: "none" }}
            aria-label={`Layer ${f.layer} feature ${f.feature} on ${tokenText(f.token)}, direct effect ${f.effect.toFixed(2)}`}
            aria-pressed={!!isSel}
            onClick={() => onSelect(f)}
            onKeyDown={(e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); onSelect(f); } }}
            onFocus={() => setHover(k)} onBlur={() => setHover(null)}
            onMouseEnter={() => setHover(k)}
            onMouseMove={(e) => show(e, tip(f))}
            onMouseLeave={() => { setHover(null); hide(); }}
          >
            <circle cx={x} cy={y} r={r + HIT_PAD} fill="transparent" />
            <circle cx={x} cy={y} r={r} fill={color(f.effect)} fillOpacity={dim ? 0.35 : 1}
              stroke="var(--surface-1)" strokeWidth={2} style={{ transition: "fill-opacity 150ms" }} />
            {(isSel || hover === k) && (
              <circle cx={x} cy={y} r={r + RING_PAD} fill="none" stroke="var(--text-primary)" strokeWidth={2} />
            )}
          </g>
        );
      })}
      {tokens.map((tok, i) => (
        <text key={i} x={left + colW * (i + 0.5)} y={H - 32} fontSize={11} textAnchor="middle"
          fill={i === tokens.length - 1 ? "var(--text-primary)" : "var(--text-secondary)"}
          fontWeight={i === tokens.length - 1 ? 600 : 400} className="tok">
          {tokenText(tok).slice(0, 10)}
        </text>
      ))}
      {/* Legend: three entries at fixed offsets along the bottom edge. */}
      <g transform={`translate(${left}, ${H - 10})`} fontSize={10.5} fill="var(--text-secondary)">
        <circle cx={4} cy={-3.5} r={4} fill={color(maxAbs)} />
        <text x={12} y={0}>supports the prediction</text>
        <circle cx={150} cy={-3.5} r={4} fill={color(-maxAbs)} />
        <text x={158} y={0}>works against it</text>
        <text x={270} y={0} fill="var(--text-muted)">bigger dot = bigger push</text>
      </g>
    </svg>
  );
}
