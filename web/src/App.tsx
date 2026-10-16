import { useEffect, useState } from "react";
import { STATIC } from "./api";
import { Glyph } from "./components/Glyph";
import { TooltipProvider } from "./components/Tooltip";
import { EXAMPLES, REPO } from "./examples";
import { readParam, writeParams } from "./url";
import { CaseStudiesView } from "./views/CaseStudiesView";
import { InternalsView } from "./views/InternalsView";
import { StudyView } from "./views/StudyView";
import { TwinView } from "./views/TwinView";

const TABS = [
  ["twin", "Twin"],
  ["study", "The study"],
  ["internals", "Internals"],
  ["cases", "Reproductions"],
] as const;
type Tab = (typeof TABS)[number][0];
type Theme = "system" | "light" | "dark";

function stored<T extends string>(key: string, allowed: readonly T[], fallback: T): T {
  try {
    const v = localStorage.getItem(key);
    return (allowed as readonly string[]).includes(v ?? "") ? (v as T) : fallback;
  } catch {
    return fallback;
  }
}
function store(key: string, v: string) {
  try { localStorage.setItem(key, v); } catch { /* storage unavailable */ }
}

function initialPrompt(): string {
  const p = Number(readParam("p"));
  if (Number.isInteger(p) && p >= 1 && p <= EXAMPLES.length) return EXAMPLES[p - 1][0];
  const q = readParam("q");
  return q && !STATIC ? q : EXAMPLES[0][0];
}

function ThemeToggle() {
  const [theme, setTheme] = useState<Theme>(() => stored("twinlens.theme", ["system", "light", "dark"], "system"));
  useEffect(() => {
    const root = document.documentElement;
    if (theme === "system") root.removeAttribute("data-theme");
    else root.setAttribute("data-theme", theme);
    store("twinlens.theme", theme);
  }, [theme]);
  const options: [Theme, "auto" | "sun" | "moon", string][] = [["system", "auto", "System theme"], ["light", "sun", "Light theme"], ["dark", "moon", "Dark theme"]];
  return (
    <div className="flex items-center gap-0.5" role="radiogroup" aria-label="Theme">
      {options.map(([v, icon, label]) => (
        <button key={v} role="radio" aria-checked={theme === v} aria-label={label} title={label}
          onClick={() => setTheme(v)}
          className="flex h-7 w-7 items-center justify-center rounded-[4px] transition-colors"
          style={theme === v ? { color: "var(--text-primary)", background: "var(--gridline)" } : { color: "var(--text-muted)" }}>
          <Glyph name={icon} size={14} />
        </button>
      ))}
    </div>
  );
}

export default function App() {
  const [tab, setTabState] = useState<Tab>(() => {
    const v = readParam("view");
    return (TABS.some(([k]) => k === v) ? v : stored("twinlens.tab", TABS.map(([k]) => k), "twin")) as Tab;
  });
  const [prompt, setPrompt] = useState(initialPrompt);
  const setTab = (t: Tab) => { setTabState(t); store("twinlens.tab", t); writeParams({ view: t === "twin" ? null : t }); window.scrollTo({ top: 0 }); };

  return (
    <TooltipProvider>
      <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded focus:px-3 focus:py-2"
        style={{ background: "var(--surface-2)" }}>Skip to content</a>
      <div className="mx-auto max-w-[1240px] px-4 pb-10 sm:px-8">
        <header className="flex items-center justify-between gap-4 border-b py-3.5 rule">
          <a href={location.pathname} className="flex items-baseline gap-2.5 no-underline" aria-label="TwinLens home">
            <svg width="26" height="17" viewBox="0 0 32 21" aria-hidden className="translate-y-[2px]">
              <circle cx="10.5" cy="10.5" r="8" fill="none" stroke="var(--text-primary)" strokeWidth="1.8" />
              <circle cx="21.5" cy="10.5" r="8" fill="none" stroke="var(--text-primary)" strokeWidth="1.8" strokeDasharray="2.6 2.2" />
            </svg>
            <span className="serif text-[22px] font-medium tracking-tight text-ink">TwinLens</span>
            <span className="kicker hidden sm:inline">GPT-2 small</span>
          </a>
          <div className="flex items-center gap-2">
            <ThemeToggle />
            <a href={REPO} target="_blank" rel="noreferrer" aria-label="Source on GitHub" title="Source on GitHub"
              className="flex h-7 w-7 items-center justify-center rounded-[4px] text-[var(--text-muted)] hover:text-ink">
              <Glyph name="github" size={15} />
            </a>
          </div>
        </header>
        <nav className="-mx-1 flex gap-5 overflow-x-auto border-b px-1 rule" aria-label="Views">
          {TABS.map(([k, label]) => (
            <button key={k} onClick={() => setTab(k)} aria-current={tab === k ? "page" : undefined}
              className="whitespace-nowrap py-2.5 text-[13.5px] transition-colors"
              style={tab === k
                ? { color: "var(--text-primary)", boxShadow: "inset 0 -2px 0 var(--text-primary)", fontWeight: 500 }
                : { color: "var(--text-muted)" }}>
              {label}
            </button>
          ))}
        </nav>
        <main id="main" className="pt-4">
          {tab === "twin" && <TwinView prompt={prompt} setPrompt={setPrompt} />}
          {tab === "study" && <StudyView />}
          {tab === "internals" && <InternalsView prompt={prompt} />}
          {tab === "cases" && <CaseStudiesView />}
        </main>
        <footer className="mt-14 grid gap-4 border-t pt-5 text-[12px] text-muted rule sm:grid-cols-[1fr_auto]">
          <p className="max-w-[70ch] leading-relaxed">
            TwinLens · GPT-2 small (124M) with Dunefsky, Chlenski &amp; Nanda's transcoders, run on our own hookable
            implementation. Every number on this page comes from real model runs; the study's raw data and code are in the repository.
          </p>
          <span className="flex gap-4 sm:justify-end">
            <a className="hover:text-ink" href={STATIC ? "./report/" : `${REPO}/blob/main/paper/REPORT.md`} target="_blank" rel="noreferrer">Technical report</a>
            <a className="hover:text-ink" href={REPO} target="_blank" rel="noreferrer">Source</a>
          </span>
        </footer>
      </div>
    </TooltipProvider>
  );
}
