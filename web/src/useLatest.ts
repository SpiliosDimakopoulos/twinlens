import { useCallback, useRef } from "react";

/**
 * Guards against stale async results. Call `begin()` when a request starts; the returned
 * `isLatest()` stays true until the next `begin()`, so only the newest request updates state.
 */
export function useLatest(): () => () => boolean {
  const counter = useRef(0);
  return useCallback(() => {
    const id = ++counter.current;
    return () => id === counter.current;
  }, []);
}
