import type { RunResponse } from "../api";
import { useTooltip } from "../components/Tooltip";
import { tokenText } from "../lib";

/** A written reading of the logit lens at the predicting position, plus how the answer's
 * probability builds up layer by layer. Generated from the data, so it is always specific. */
export function LensReading({ run }: { run: RunResponse }) {
  const { show, hide } = useTooltip();
  const last = run.tokens.length - 1;
  const rank = run.logit_lens.final_rank.map((row) => row[last]); // [emb, L0..L11]
  const prob = run.logit_lens.final_prob.map((row) => row[last]);
  const settled = rank.findIndex((_, i) => rank.slice(i).every((r) => r === 0));
  const firstTop = rank.findIndex((r) => r === 0);
  const name = (i: number) => (i === 0 ? "the embeddings" : `layer ${i - 1}`);
  const ans = `“${tokenText(run.target.token)}”`;
  const jump = prob.slice(1).map((p, i) => p - prob[i]);
  const big = jump.indexOf(Math.max(...jump)) + 1;

  const W = 300, H = 120, L = 28, R = 8, T = 10, B = 22;
  const x = (i: number) => L + (i / (prob.length - 1)) * (W - L - R);
  const y = (p: number) => T + (1 - p) * (H - T - B);
  const pts = prob.map((p, i) => [x(i), y(p)] as const);

  return (
    <div className="space-y-4">
      <div className="kicker">Reading it at “{tokenText(run.tokens[last])}”</div>
      <p className="serif text-[18px] leading-snug text-ink">
        {settled <= 0
          ? <>{ans} is the top guess from the very first layer.</>
          : <>{ans} becomes the top guess after {name(firstTop)}{firstTop !== settled && <> and holds from {name(settled)}</>}.</>}
        {" "}Its biggest single jump comes at {name(big)} (+{(jump[big - 1] * 100).toFixed(0)} points).
      </p>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ maxWidth: 360 }} role="img"
        aria-label={`Probability of ${tokenText(run.target.token)} after each layer`}>
        {[0, 0.5, 1].map((g) => (
          <g key={g}>
            <line x1={L} x2={W - R} y1={y(g)} y2={y(g)} stroke="var(--gridline)" />
            <text x={L - 6} y={y(g) + 3} fontSize={9} textAnchor="end" fill="var(--text-muted)" className="tnum">{g * 100}%</text>
          </g>
        ))}
        <polyline points={pts.map((p) => p.join(",")).join(" ")} fill="none" stroke="var(--series-1)" strokeWidth={2} />
        {pts.map(([px, py], i) => (
          <g key={i} onMouseMove={(e) => show(e, <div><b>{name(i)}</b>: p({ans}) = {(prob[i] * 100).toFixed(1)}%, rank {rank[i]}</div>)} onMouseLeave={hide}>
            <circle cx={px} cy={py} r={8} fill="transparent" />
            <circle cx={px} cy={py} r={i === big ? 4 : 2.5} fill={i === big ? "var(--text-primary)" : "var(--series-1)"} stroke="var(--surface-1)" strokeWidth={1.5} />
          </g>
        ))}
        <text x={x(0)} y={H - 6} fontSize={9} fill="var(--text-muted)">emb</text>
        <text x={x(prob.length - 1)} y={H - 6} fontSize={9} textAnchor="end" fill="var(--text-muted)">L11</text>
      </svg>
      <p className="text-[12.5px] leading-relaxed text-ink2">
        The heatmap reads bottom-up: each row decodes the residual stream after one more layer. Earlier positions
        mostly guess their own next word; the shaded column is where the answer forms.
      </p>
    </div>
  );
}
