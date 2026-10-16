import type { Feature } from "../api";
import { diverging, useTokens } from "../components/theme";
import { useTooltip } from "../components/Tooltip";
import { tokenText } from "../components/ui";

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
 * Edges run to the output node. Click a node to ask the twin "what if?".
 */
export function ReasoningGraph({ tokens, target, features, selected, onSelect }: Props) {
  const t = useTokens();
  const { show, hide } = useTooltip();
  const colW = Math.max(54, Math.min(84, 760 / tokens.length));
  const rowH = 30;
  const left = 34;
  const top = 46;
  const W = left + colW * tokens.length + 20;
  const H = top + rowH * 12 + 56;
  const maxAbs = Math.max(...features.map((f) => Math.abs(f.effect)), 1e-6);
  const color = diverging(t, maxAbs);
  const out = { x: left + colW * (tokens.length - 0.5), y: 18 };

  // Features sharing a (layer, token) cell are spread evenly across the column width.
  const cellCount = new Map<string, number>();
  for (const f of features) cellCount.set(`${f.layer}-${f.pos}`, (cellCount.get(`${f.layer}-${f.pos}`) ?? 0) + 1);
  const slots = new Map<string, number>();
  const placed = features.map((f) => {
    const k = `${f.layer}-${f.pos}`;
    const i = slots.get(k) ?? 0;
    slots.set(k, i + 1);
    const n = cellCount.get(k) ?? 1;
    const step = Math.min(16, (colW - 12) / n);
    const x = left + colW * (f.pos + 0.5) + (i - (n - 1) / 2) * step;
    const y = top + rowH * (11 - f.layer) + rowH / 2;
    return { f, x, y, r: 3.5 + 5 * Math.sqrt(Math.abs(f.effect) / maxAbs) };
  });

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" role="img"
      aria-label={`Features with the largest direct effect on predicting ${target}`}>
      {Array.from({ length: 12 }, (_, l) => (
        <g key={l}>
          <line x1={left} x2={W - 20} y1={top + rowH * (11 - l) + rowH / 2} y2={top + rowH * (11 - l) + rowH / 2}
            stroke="var(--gridline)" strokeWidth={1} />
          <text x={left - 8} y={top + rowH * (11 - l) + rowH / 2 + 3.5} fontSize={10} textAnchor="end"
            fill="var(--text-muted)" className="tnum">L{l}</text>
        </g>
      ))}
      {placed.map(({ f, x, y }) => (
        <line key={`e-${KEY(f)}`} x1={x} y1={y} x2={out.x} y2={out.y + 12}
          stroke={color(f.effect)} strokeOpacity={0.35}
          strokeWidth={0.75 + 2.5 * (Math.abs(f.effect) / maxAbs)} />
      ))}
      <g>
        <rect x={out.x - 60} y={out.y - 12} width={120} height={24} rx={12} fill="var(--text-primary)" />
        <text x={out.x} y={out.y + 4} fontSize={11} textAnchor="middle" fill="var(--surface-1)" className="tok">
          → {tokenText(target)}
        </text>
      </g>
      {placed.map(({ f, x, y, r }) => {
        const isSel = selected && KEY(selected) === KEY(f);
        return (
          <g key={KEY(f)} style={{ cursor: "pointer" }} onClick={() => onSelect(f)}
            onMouseMove={(e) => show(e, (
              <div className="space-y-1">
                <div className="font-semibold">L{f.layer} · feature {f.feature}</div>
                <div>on token <span className="tok">“{tokenText(f.token)}”</span>, activation {f.activation.toFixed(2)}</div>
                <div>direct effect on “{tokenText(target)}”: <b className="tnum">{f.effect >= 0 ? "+" : ""}{f.effect.toFixed(3)}</b> logits</div>
                <div className="text-ink2">promotes: <span className="tok">{f.label.map(tokenText).join(" · ")}</span></div>
                <div className="text-muted">click to simulate an edit</div>
              </div>
            ))}
            onMouseLeave={hide}
          >
            <circle cx={x} cy={y} r={r + 6} fill="transparent" />
            <circle cx={x} cy={y} r={r} fill={color(f.effect)} stroke="var(--surface-1)" strokeWidth={2} />
            {isSel && <circle cx={x} cy={y} r={r + 3.5} fill="none" stroke="var(--text-primary)" strokeWidth={2} />}
          </g>
        );
      })}
      {tokens.map((tok, i) => (
        <text key={i} x={left + colW * (i + 0.5)} y={H - 30} fontSize={10.5} textAnchor="middle"
          fill="var(--text-secondary)" className="tok">
          {tokenText(tok).slice(0, 10)}
        </text>
      ))}
      <g transform={`translate(${left}, ${H - 12})`} fontSize={10} fill="var(--text-secondary)">
        <circle cx={5} cy={-3} r={5} fill={color(maxAbs)} />
        <text x={14} y={0}>pushes the prediction up</text>
        <circle cx={170} cy={-3} r={5} fill={color(-maxAbs)} />
        <text x={179} y={0}>pushes it down</text>
        <text x={300} y={0} fill="var(--text-muted)">size = |direct effect|</text>
      </g>
    </svg>
  );
}
