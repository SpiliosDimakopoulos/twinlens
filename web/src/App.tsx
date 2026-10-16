import { useState } from "react";
import { TooltipProvider } from "./components/Tooltip";
import { CaseStudiesView } from "./views/CaseStudiesView";
import { InternalsView } from "./views/InternalsView";
import { StudyView } from "./views/StudyView";
import { EXAMPLES, TwinView } from "./views/TwinView";

const TABS = [
  ["twin", "Twin"],
  ["internals", "Internals"],
  ["study", "Fidelity study"],
  ["cases", "Reproductions"],
] as const;
type Tab = (typeof TABS)[number][0];

function readTab(): Tab {
  try {
    const t = localStorage.getItem("twinlens.tab");
    return (TABS.some(([k]) => k === t) ? t : "twin") as Tab;
  } catch {
    return "twin";
  }
}

export default function App() {
  const [tab, setTabState] = useState<Tab>(readTab);
  const [prompt, setPrompt] = useState(EXAMPLES[0]);
  const setTab = (t: Tab) => {
    setTabState(t);
    try { localStorage.setItem("twinlens.tab", t); } catch { /* storage unavailable */ }
  };
  return (
    <TooltipProvider>
      <div className="mx-auto max-w-7xl px-4 pb-16 sm:px-6">
        <header className="flex flex-col gap-3 py-5 sm:flex-row sm:items-end sm:justify-between">
          <div>
            <h1 className="flex items-center gap-2 text-xl font-semibold text-ink">
              <svg width="26" height="18" viewBox="0 0 32 22" aria-hidden><circle cx="11" cy="11" r="8" fill="none" stroke="var(--lv-3)" strokeWidth="3" /><circle cx="21" cy="11" r="8" fill="none" stroke="var(--lv-5)" strokeWidth="3" /></svg>
              TwinLens
            </h1>
            <p className="text-sm text-ink2">A digital twin of GPT-2's internals that tells you how far to trust its own explanations.</p>
          </div>
          <nav className="flex gap-1 overflow-x-auto rounded-xl border p-1" style={{ borderColor: "var(--border)" }} aria-label="Views">
            {TABS.map(([k, label]) => (
              <button key={k} onClick={() => setTab(k)} aria-current={tab === k}
                className="whitespace-nowrap rounded-lg px-3 py-1.5 text-sm"
                style={tab === k ? { background: "var(--text-primary)", color: "var(--surface-1)" } : { color: "var(--text-secondary)" }}>
                {label}
              </button>
            ))}
          </nav>
        </header>
        <main>
          {tab === "twin" && <TwinView prompt={prompt} setPrompt={setPrompt} />}
          {tab === "internals" && <InternalsView prompt={prompt} />}
          {tab === "study" && <StudyView />}
          {tab === "cases" && <CaseStudiesView />}
        </main>
      </div>
    </TooltipProvider>
  );
}
