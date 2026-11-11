import { createContext, useCallback, useContext, type ReactNode } from "react";

export type Tip = { x: number; y: number; content: ReactNode } | null;
export const TooltipContext = createContext<(tip: Tip) => void>(() => {});

type Pointer = { clientX: number; clientY: number };

/** Show/hide the shared tooltip; `hover(content)` gives the two handlers a mark needs. */
export function useTooltip() {
  const set = useContext(TooltipContext);
  const show = useCallback((e: Pointer, content: ReactNode) => set({ x: e.clientX, y: e.clientY, content }), [set]);
  const hide = useCallback(() => set(null), [set]);
  const hover = useCallback(
    (content: ReactNode) => ({ onMouseMove: (e: Pointer) => show(e, content), onMouseLeave: hide }),
    [show, hide],
  );
  return { show, hide, hover };
}
