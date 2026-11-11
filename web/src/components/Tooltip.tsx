import { useState, type ReactNode } from "react";
import { TooltipContext, type Tip } from "./useTooltip";

const OFFSET = 14;     // px between the pointer and the tooltip's corner
const MAX_RIGHT = 260; // keep the tooltip this far from the right edge of the viewport

/** One shared tooltip layer; marks call ``useTooltip()`` and pass their content on hover. */
export function TooltipProvider({ children }: { children: ReactNode }) {
  const [tip, setTip] = useState<Tip>(null);
  return (
    <TooltipContext.Provider value={setTip}>
      {children}
      {tip && (
        <div
          role="tooltip"
          className="pointer-events-none fixed z-50 max-w-xs rounded-lg border px-3 py-2 text-xs shadow-lg"
          style={{
            left: Math.min(tip.x + OFFSET, window.innerWidth - MAX_RIGHT),
            top: tip.y + OFFSET,
            background: "var(--surface-2)",
            borderColor: "var(--border)",
            color: "var(--text-primary)",
          }}
        >
          {tip.content}
        </div>
      )}
    </TooltipContext.Provider>
  );
}
