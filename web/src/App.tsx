import { useEffect, useState } from "react";
import { TooltipProvider } from "./components/Tooltip";
import { CaseStudiesView } from "./views/CaseStudiesView";
import { InternalsView } from "./views/InternalsView";
import { StudyView } from "./views/StudyView";
import { EXAMPLES } from "./examples";
import { TwinView } from "./views/TwinView";

const REPO = "https://github.com/SpiliosDimakopoulos/twinlens";

const TABS = [
  ["twin", "Twin", "Trace a prompt and test the explanation"],
  ["internals", "Internals", "Logit lens, attention, head attribution"],
  ["study", "Fidelity study", "8,064 interventions: how far the twin holds"],
  ["cases", "Reproductions", "Published circuits, recomputed live"],
] as const;
type Tab = (typeof TABS)[number][0];
type Theme = "system" | "light" | "dark";

function load<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return (allowed as readonly string[]).includes(v ?? "") ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}
function save(key: string, v: string) {
  try { localStorage.setItem(key, v); } catch { /* storage unavailable */ }
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => load("twinlens.theme", ["system", "light", "dark"], "system"));
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    save("twinlens.theme", theme);
  }, [theme]);
  const options: [Theme, string, string][] = [["system", "◐", "System theme"], ["light", "☀", "Light theme"], ["dark", "☾", "Dark theme"]];
  return (
    <div className="flex rounded-lg border p-0.5" style={{ borderColor: "var(--border)" }} role="radiogroup" aria-label="Theme">
      {options.map(([v, icon, label]) => (
        <button key={v} role="radio" aria-checked={theme === v} aria-label={label} title={label}
          onClick={() => setTheme(v)}
          className="h-7 w-7 rounded-md text-sm"
          style={theme === v ? { background: "var(--gridline)", color: "var(--text-primary)" } : { color: "var(--text-muted)" }}>
          {icon}
        </button>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTabState] = useState<Tab>(() => load("twinlens.tab", TABS.map(([k]) => k), "twin"));
  const [prompt, setPrompt] = useState(EXAMPLES[0][0]);
  const setTab = (t: Tab) => { setTabState(t); save("twinlens.tab", t); };
  const current = TABS.find(([k]) => k === tab)!;

  return (
    <TooltipProvider>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:px-3 focus:py-2"
        style={{ background: "var(--surface-2)" }}>Skip to content</a>
      <div className="mx-auto max-w-7xl px-4 pb-10 sm:px-6">
        <header className="flex flex-col gap-4 pt-5 pb-4 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center justify-between gap-4">
            <div>
              <h1 className="flex items-center gap-2 text-xl font-semibold tracking-tight text-ink">
                <svg width="28" height="19" viewBox="0 0 32 22" aria-hidden><circle cx="11" cy="11" r="8" fill="none" stroke="var(--lv-3)" strokeWidth="3" /><circle cx="21" cy="11" r="8" fill="none" stroke="var(--lv-5)" strokeWidth="3" /></svg>
                TwinLens
              </h1>
              <p className="mt-0.5 max-w-xl text-sm text-ink2">A digital twin of GPT-2's internals that tells you how far to trust its own explanations.</p>
            </div>
            <div className="md:hidden"><ThemeToggle /></div>
          </div>
          <div className="flex items-center gap-3">
            <nav className="flex gap-1 overflow-x-auto rounded-xl border p-1" style={{ borderColor: "var(--border)" }} aria-label="Views">
              {TABS.map(([k, label, hint]) => (
                <button key={k} onClick={() => setTab(k)} aria-current={tab === k ? "page" : undefined} title={hint}
                  className="whitespace-nowrap rounded-lg px-3 py-1.5 text-sm transition"
                  style={tab === k ? { background: "var(--text-primary)", color: "var(--surface-1)" } : { color: "var(--text-secondary)" }}>
                  {label}
                </button>
              ))}
            </nav>
            <div className="hidden md:block"><ThemeToggle /></div>
          </div>
        </header>
        <p className="mb-4 text-xs text-muted">{current[2]}</p>
        <main id="main">
          {tab === "twin" && <TwinView prompt={prompt} setPrompt={setPrompt} />}
          {tab === "internals" && <InternalsView prompt={prompt} />}
          {tab === "study" && <StudyView />}
          {tab === "cases" && <CaseStudiesView />}
        </main>
        <footer className="mt-10 flex flex-wrap items-center justify-between gap-2 border-t pt-4 text-xs text-muted" style={{ borderColor: "var(--gridline)" }}>
          <span>GPT-2 small · Dunefsky et al. transcoders · runs on a laptop CPU</span>
          <span className="flex gap-4">
            <a className="hover:text-ink" href={`${REPO}/blob/main/paper/REPORT.md`} target="_blank" rel="noreferrer">Technical report</a>
            <a className="hover:text-ink" href={REPO} target="_blank" rel="noreferrer">Source on GitHub</a>
          </span>
        </footer>
      </div>
    </TooltipProvider>
  );
}
