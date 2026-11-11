import { EXAMPLES } from "../../examples";
import { tokenText } from "../../lib";

/** Example prompts presented as a specimen list: the task family over the prompt itself. */
export function Specimens({ active, onPick, disabled }: { active: number; onPick: (i: number) => void; disabled: boolean }) {
  return (
    <div className="-mx-1 flex gap-1 overflow-x-auto px-1 pb-1" role="group" aria-label="Example prompts">
      {EXAMPLES.map(([text, kind], i) => (
        <button key={text} aria-pressed={i === active} aria-disabled={disabled}
          onClick={() => { if (!disabled) onPick(i); }}
          className="group min-w-[150px] max-w-[220px] shrink-0 rounded-[5px] px-3 py-2 text-left transition-colors aria-disabled:opacity-60 lg:min-w-0 lg:max-w-none lg:flex-1"
          style={i === active ? { background: "var(--surface-1)", boxShadow: "inset 0 0 0 1px var(--baseline), inset 0 -2px 0 var(--text-primary)" } : {}}>
          <div className="kicker truncate" style={i === active ? { color: "var(--text-primary)" } : {}}>{String(i + 1).padStart(2, "0")} · {kind}</div>
          <div className="tok mt-1 truncate text-[12px] text-ink2 group-hover:text-ink">{tokenText(text)}</div>
        </button>
      ))}
    </div>
  );
}
