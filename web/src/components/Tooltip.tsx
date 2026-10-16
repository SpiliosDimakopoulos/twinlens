import { createContext, useCallback, useContext, useState, type ReactNode } from "react";

type Tip = { x: number; y: number; content: ReactNode } | null;
const Ctx = createContext<(tip: Tip) => void>(() => {});

/** One shared tooltip layer; marks call ``useTooltip()`` and pass their content on hover. */
export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<Tip>(null);
  return (
    <Ctx.Provider value={setTip}>
      {children}
      {tip && (
        <div
          role="tooltip"
          className="pointer-events-none fixed z-50 max-w-xs rounded-lg border px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(tip.x + 14, window.innerWidth - 260),
            top: tip.y + 14,
            background: "var(--surface-2)",
            borderColor: "var(--border)",
            color: "var(--text-primary)",
          }}
        >
          {tip.content}
        </div>
      )}
    </Ctx.Provider>
  );
}

export function useTooltip() {
  const set = useContext(Ctx);
  const show = useCallback(
    (e: { clientX: number; clientY: number }, content: ReactNode) =>
      set({ x: e.clientX, y: e.clientY, content }),
    [set],
  );
  const hide = useCallback(() => set(null), [set]);
  return { show, hide };
}
