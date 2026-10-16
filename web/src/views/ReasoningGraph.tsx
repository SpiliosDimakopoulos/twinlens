import { useState } from "react";
import type { Feature } from "../api";
import { diverging, useTokens } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { tokenText } from "../lib";

type Props = {
  tokens: string[];
  target: string;
  features: Feature[];
  selected: Feature | null;
  onSelect: (f: Feature) => void;
};

const KEY = (f: Feature) => `${f.layer}-${f.pos}-${f.feature}`;

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
  const colW = Math.max(58, Math.min(88, 780 / tokens.length));
  const rowH = 30;
  const left = 34;
  const top = 50;
  const W = left + colW * tokens.length + 20;
  const H = top + rowH * 12 + 58;
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.effect)), 1e-6);
  const scale = diverging(t, maxAbs);
  // Floor the colour intensity so weak nodes stay visible against the surface.
  const color = (e: number) => scale(Math.sign(e) * Math.max(Math.abs(e), 0.3 * maxAbs));
  const out = { x: left + colW * (tokens.length - 0.5), y: 20 };
  const focus = hover;  // only hovering isolates a node; selection is shown by its ring

  const cellCount = new Map<string, number>();
  for (const f of features) cellCount.set(`${f.layer}-${f.pos}`, (cellCount.get(`${f.layer}-${f.pos}`) ?? 0) + 1);
  const slots = new Map<string, number>();
  const placed = features.map((f) => {
    const k = `${f.layer}-${f.pos}`;
    const i = slots.get(k) ?? 0;
    slots.set(k, i + 1);
    const n = cellCount.get(k) ?? 1;
    const step = Math.min(17, (colW - 14) / n);
    const x = left + colW * (f.pos + 0.5) + (i - (n - 1) / 2) * step;
    const y = top + rowH * (11 - f.layer) + rowH / 2;
    return { f, x, y, r: 4 + 5 * Math.sqrt(Math.abs(f.effect) / maxAbs) };
  });

  const tip = (f: Feature) => (
    <div className="space-y-1">
      <div className="font-semibold">L{f.layer} · feature {f.feature}</div>
      <div>fires on <span className="tok">“{tokenText(f.token)}”</span> (activation {f.activation.toFixed(2)})</div>
      <div>direct effect on “{tokenText(target)}”: <b className="tnum">{f.effect >= 0 ? "+" : ""}{f.effect.toFixed(3)}</b> logits</div>
      <div className="text-ink2">promotes <span className="tok">{f.label.map(tokenText).join(" · ")}</span></div>
      <div className="text-muted">click to simulate an edit</div>
    </div>
  );

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ minWidth: 560 }}
      aria-label={`Reasoning path: features with the largest direct effect on predicting ${tokenText(target)}`}>
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
          <line key={`e-${k}`} x1={x} y1={y} x2={out.x} y2={out.y + 12}
            stroke={color(f.effect)} strokeOpacity={dim ? 0.08 : strong ? 0.9 : 0.35}
            strokeWidth={0.75 + 2.5 * (Math.abs(f.effect) / maxAbs) + (strong ? 1 : 0)}
            style={{ transition: "stroke-opacity 150ms" }} />
        );
      })}
      <g>
        <rect x={out.x - 64} y={out.y - 13} width={128} height={26} rx={13} fill="var(--text-primary)" />
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
            <circle cx={x} cy={y} r={r + 7} fill="transparent" />
            <circle cx={x} cy={y} r={r} fill={color(f.effect)} fillOpacity={dim ? 0.35 : 1}
              stroke="var(--surface-1)" strokeWidth={2} style={{ transition: "fill-opacity 150ms" }} />
            {(isSel || hover === k) && (
              <circle cx={x} cy={y} r={r + 3.5} fill="none" stroke="var(--text-primary)" strokeWidth={2} />
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
      <g transform={`translate(${left}, ${H - 10})`} fontSize={10} fill="var(--text-secondary)">
        <circle cx={5} cy={-3} r={5} fill={color(maxAbs)} />
        <text x={14} y={0}>pushes the prediction up</text>
        <circle cx={170} cy={-3} r={5} fill={color(-maxAbs)} />
        <text x={179} y={0}>pushes it down</text>
        <text x={300} y={0} fill="var(--text-muted)">size = |direct effect| · shaded column = predicting position</text>
      </g>
    </svg>
  );
}
