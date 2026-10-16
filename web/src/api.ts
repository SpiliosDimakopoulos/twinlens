// Typed client for the TwinLens API (server/app.py).

export type TopToken = { token: string; id: number; prob: number; logit: number };

export type RunResponse = {
  tokens: string[];
  top: TopToken[];
  target: { id: number; token: string };
  logit_lens: {
    top_token: string[][]; // [layer+1][pos]
    top_prob: number[][];
    final_prob: number[][];
    final_rank: number[][];
  };
  attention: number[][][][]; // [layer][head][q][k]
  dla: { component: string; value: number }[];
};

export type Feature = {
  layer: number;
  pos: number;
  feature: number;
  activation: number;
  effect: number;
  token: string;
  label: string[];
};

export type ExplainResponse = {
  tokens: string[];
  target: string;
  features: Feature[];
  error_fraction: Record<string, number>;
};

export type LevelResult = { top: TopToken[]; dlogp: number; kl: number };

export type WhatIfResponse = {
  target: string;
  clean: TopToken[];
  levels: Record<string, LevelResult>;
  signals: Record<string, number>;
  trust: number | null;
  trust_factors: { signal: string; value: number; contribution: number }[];
  label: string[];
};

export type Intervention =
  | { kind: "head"; layer: number; head: number; mode?: "zero" | "mean"; scale?: number }
  | { kind: "neuron"; layer: number; neuron: number; value?: number; scale?: number }
  | { kind: "steer"; layer: number; plus: string; minus: string; coeff: number };

export type InterveneResponse = { before: TopToken[]; after: TopToken[]; kl: number };

export type CaseStudy = {
  title: string;
  grid: number[][];
  known: [number, number][];
  negative?: [number, number][];
  logit_diff?: number;
  example?: string;
};

export type Ci = [number, number, number]; // point, lo, hi
export type Summary = {
  n_edits: number;
  n_prompts: number;
  levels: string[];
  level_labels: Record<string, string>;
  ladder: Record<string, Record<string, Ci>>;
  decay: Record<string, { layers_after: string; median_cos: number; within_tol: number; catastrophic: number; n: number }[]>;
  decomposition: { frozen_layernorm: number; frozen_attention: number; transcoder_residual: number };
  trust: {
    cv_auroc: number; base_rate: number; cv_auroc_big: number; base_rate_big: number;
    ok_rate_trusted_half: number; ok_rate_flagged_half: number; single_signal_auroc: Record<string, number>;
  };
  real_effect: { median_abs_dlogp: number; frac_big: number; frac_flip: number };
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail ?? res.statusText);
  return res.json();
}

// ---------------------------------------------------------------- static demo
// Built with VITE_STATIC=1 the dashboard needs no server: it reads payloads precomputed by
// experiments/build_demo.py (same code path as the API) for a fixed set of prompts.
export const STATIC = import.meta.env.VITE_STATIC === "1";
const DEMO = `${import.meta.env.BASE_URL}demo/`;

type DemoPrompt = {
  prompt: string; kind: string; run: RunResponse; explain: ExplainResponse;
  whatif: Record<string, Record<string, WhatIfResponse>>;
  ablation: { before: TopToken[]; heads: Record<string, { after: TopToken[]; kl: number }> };
  generation: GenerateStep[];
};
type Manifest = { prompts: { prompt: string; kind: string; file: string }[]; presets: number[] };
type Shared = { summary: Summary; case_studies: Record<string, CaseStudy> };

const cache = new Map<string, Promise<unknown>>();
function json<T>(file: string): Promise<T> {
  if (!cache.has(file)) cache.set(file, fetch(DEMO + file).then((r) => {
    if (!r.ok) throw new Error(`demo data missing: ${file}`);
    return r.json();
  }));
  return cache.get(file) as Promise<T>;
}
export const demoManifest = () => json<Manifest>("manifest.json");
async function demoPrompt(prompt: string): Promise<DemoPrompt> {
  const m = await demoManifest();
  const entry = m.prompts.find((p) => p.prompt === prompt);
  if (!entry) throw new Error("This live demo is precomputed for the example prompts. Run TwinLens locally to trace any prompt.");
  return json<DemoPrompt>(entry.file);
}
const nearest = (xs: number[], x: number) => xs.reduce((a, b) => (Math.abs(b - x) < Math.abs(a - x) ? b : a));

const staticApi = {
  run: async (prompt: string) => (await demoPrompt(prompt)).run,
  explain: async (prompt: string) => (await demoPrompt(prompt)).explain,
  whatif: async (prompt: string, layer: number, pos: number, feature: number, value: number) => {
    const [d, m] = await Promise.all([demoPrompt(prompt), demoManifest()]);
    const f = d.explain.features.find((x) => x.layer === layer && x.pos === pos && x.feature === feature);
    const mult = nearest(m.presets, f && f.activation ? value / f.activation : 0);
    // keys follow Python str(float): "0.0", "0.5", "2.0"
    const key = Number.isInteger(mult) ? mult.toFixed(1) : String(mult);
    const hit = d.whatif[`${layer}-${pos}-${feature}`]?.[key];
    if (!hit) throw new Error("This edit was not precomputed in the live demo.");
    return hit;
  },
  intervene: async (prompt: string, interventions: Intervention[]) => {
    const d = await demoPrompt(prompt);
    const h = interventions[0];
    if (interventions.length !== 1 || h.kind !== "head") throw new Error("Only single-head ablations are precomputed in the live demo.");
    const r = d.ablation.heads[`${h.layer}-${h.head}`];
    return { before: d.ablation.before, after: r.after, kl: r.kl };
  },
  caseStudy: async (name: string) => (await json<Shared>("shared.json")).case_studies[name],
  summary: async () => (await json<Shared>("shared.json")).summary,
};

const liveApi = {
  run: (prompt: string) => post<RunResponse>("/api/run", { prompt }),
  explain: (prompt: string) => post<ExplainResponse>("/api/twin/explain", { prompt }),
  whatif: (prompt: string, layer: number, pos: number, feature: number, value: number) =>
    post<WhatIfResponse>("/api/twin/whatif", { prompt, layer, pos, feature, value }),
  intervene: (prompt: string, interventions: Intervention[]) =>
    post<InterveneResponse>("/api/intervene", { prompt, interventions }),
  caseStudy: (name: string) => get<CaseStudy>(`/api/case-studies/${name}`),
  summary: () => get<Summary>("/api/fidelity/summary"),
};

export const api: typeof liveApi = STATIC ? staticApi : liveApi;

export type GenerateStep = { token: string; top: TopToken[]; lens: string[]; lens_prob: number[] };

export function generate(
  prompt: string,
  onStep: (s: GenerateStep) => void,
  onDone: () => void,
  maxNewTokens = 20,
): () => void {
  if (STATIC) {
    let stopped = false;
    demoPrompt(prompt).then((d) => {
      d.generation.slice(0, maxNewTokens).forEach((s, i) => setTimeout(() => { if (!stopped) onStep(s); }, 140 * (i + 1)));
      setTimeout(() => { if (!stopped) onDone(); }, 140 * (Math.min(maxNewTokens, d.generation.length) + 1));
    }).catch(() => onDone());
    return () => { stopped = true; onDone(); };
  }
  const proto = location.protocol === "https:" ? "wss" : "ws";
  const ws = new WebSocket(`${proto}://${location.host}/ws/generate`);
  ws.onopen = () => ws.send(JSON.stringify({ prompt, max_new_tokens: maxNewTokens }));
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.done) {
      onDone();
      ws.close();
    } else onStep(msg);
  };
  ws.onerror = () => onDone();
  return () => ws.close();
}
