// Typed client for the TwinLens API (server/app.py).

import { nearest, presetKey } from "./demoKeys";

/** Twin levels, mirroring the Python modes (twinlens/twin/replacement.py) plus the real model. */
export const LEVELS = [
  "direct", "linear", "linear_lnjac", "linear_fulljac", "replacement", "replacement_ln", "replacement_full", "real",
] as const;
export type Level = (typeof LEVELS)[number];
/** A per-level map; a payload only carries the levels that were computed. */
export type LevelMap<T> = Partial<Record<Level, T>>;

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
  /** Every edit is run on today's graph (linear) and on the real model; other levels may be absent. */
  levels: LevelMap<LevelResult> & Record<"linear" | "real", LevelResult>;
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
export type Metric =
  | "spearman" | "median_cos" | "sign_agree" | "spearman_big" | "within_tol_big" | "within_tol" | "catastrophic" | "mean_err_clipped";
export type PairedMetric = "spearman" | "sign_agree" | "within_tol_big" | "catastrophic" | "mean_err_clipped";
export type DecayBin = { layers_after: string; median_cos: number; within_tol: number; catastrophic: number; n: number };
export type Summary = {
  n_edits: number;
  n_prompts: number;
  levels: Level[];
  level_labels: LevelMap<string>;
  ladder: LevelMap<Record<Metric, Ci>>;
  decay: LevelMap<DecayBin[]>;
  decomposition: { frozen_layernorm: number; frozen_attention: number; transcoder_residual: number };
  trust: {
    cv_auroc: number; base_rate: number; cv_auroc_big: number; base_rate_big: number;
    ok_rate_trusted_half: number; ok_rate_flagged_half: number; single_signal_auroc: Record<string, number>;
    transfer_lofo?: Record<string, number> & { mean: number }; // AUROC per held-out family, plus their mean
    payoff?: Record<string, { ok_all: number; ok_trusted: number; ok_flagged: number; kept_share: number; n: number }>;
  };
  real_effect: { median_abs_dlogp: number; frac_big: number; frac_flip: number };
  generalization?: Record<string, { label: string; n_edits: number; n_prompts: number; layers: number[]; paired_diff: Record<PairedMetric, Ci> }>;
  fix?: {
    levels: Level[];
    ladder: LevelMap<Record<Metric, Ci>>;
    by_category_spearman: Record<string, LevelMap<number>>;
    top_edges_catastrophic: Record<string, number>;
  };
};

export type GenerateStep = { token: string; top: TopToken[]; lens: string[]; lens_prob: number[] };

/** Everything the dashboard asks of the model; served live by the API or from the static demo files. */
export interface Api {
  run(prompt: string): Promise<RunResponse>;
  explain(prompt: string): Promise<ExplainResponse>;
  whatif(prompt: string, layer: number, pos: number, feature: number, value: number): Promise<WhatIfResponse>;
  intervene(prompt: string, interventions: Intervention[]): Promise<InterveneResponse>;
  caseStudy(name: string): Promise<CaseStudy>;
  summary(): Promise<Summary>;
  /** Streams generated tokens to `onStep`; `onDone` is called exactly once. Returns a stop function. */
  generate(prompt: string, onStep: (s: GenerateStep) => void, onDone: () => void, maxNewTokens?: number): () => void;
}

/** Wraps `f` so only its first call has an effect. */
function once(f: () => void): () => void {
  let called = false;
  return () => { if (!called) { called = true; f(); } };
}

/** FastAPI errors: `detail` is a string (4xx we raise) or a list of objects (422 validation). */
async function failure(res: Response): Promise<Error> {
  const body = await res.json().catch(() => ({}));
  const d = body?.detail;
  const msg = typeof d === "string" ? d
    : Array.isArray(d) ? d.map((x) => [x?.loc?.slice(1).join("."), x?.msg].filter(Boolean).join(": ")).join("; ")
    : res.statusText;
  return new Error(msg || `request failed (${res.status})`);
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw await failure(res);
  return res.json();
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(path);
  if (!res.ok) throw await failure(res);
  return res.json();
}

// ---------------------------------------------------------------- static demo
// Built with VITE_STATIC=1 the dashboard needs no server: it reads payloads precomputed by
// experiments/build_demo.py (same code path as the API) for a fixed set of prompts.
export const STATIC = import.meta.env.VITE_STATIC === "1";
const DEMO = `${import.meta.env.BASE_URL}demo/`;
const DEMO_STEP_MS = 140;  // replay pace of the precomputed generation

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
  let hit = cache.get(file);
  if (!hit) {
    hit = fetch(DEMO + file).then((r) => {
      if (!r.ok) throw new Error(`demo data missing: ${file}`);
      return r.json();
    });
    cache.set(file, hit);
  }
  return hit as Promise<T>;
}
export const demoManifest = () => json<Manifest>("manifest.json");
async function demoPrompt(prompt: string): Promise<DemoPrompt> {
  const m = await demoManifest();
  const entry = m.prompts.find((p) => p.prompt === prompt);
  if (!entry) throw new Error("This live demo is precomputed for the example prompts. Run TwinLens locally to trace any prompt.");
  return json<DemoPrompt>(entry.file);
}

const staticApi: Api = {
  run: async (prompt) => (await demoPrompt(prompt)).run,
  explain: async (prompt) => (await demoPrompt(prompt)).explain,
  whatif: async (prompt, layer, pos, feature, value) => {
    const [d, m] = await Promise.all([demoPrompt(prompt), demoManifest()]);
    const f = d.explain.features.find((x) => x.layer === layer && x.pos === pos && x.feature === feature);
    const mult = nearest(m.presets, f && f.activation ? value / f.activation : 0);
    const hit = mult === undefined ? undefined : d.whatif[`${layer}-${pos}-${feature}`]?.[presetKey(mult)];
    if (!hit) throw new Error("This edit was not precomputed in the live demo.");
    return hit;
  },
  intervene: async (prompt, interventions) => {
    const d = await demoPrompt(prompt);
    const h = interventions[0];
    if (interventions.length !== 1 || h?.kind !== "head") throw new Error("Only switching off a single head is precomputed in the live demo.");
    const r = d.ablation.heads[`${h.layer}-${h.head}`];
    if (!r) throw new Error("This head was not precomputed in the live demo.");
    return { before: d.ablation.before, after: r.after, kl: r.kl };
  },
  caseStudy: async (name) => {
    const c = (await json<Shared>("shared.json")).case_studies[name];
    if (!c) throw new Error(`case study ${name} is not in this build`);
    return c;
  },
  summary: async () => (await json<Shared>("shared.json")).summary,
  generate: (prompt, onStep, onDone, maxNewTokens = 20) => {
    // Replays the precomputed tokens; stopping clears the pending timers.
    const timers: ReturnType<typeof setTimeout>[] = [];
    let stopped = false;
    const finish = once(() => { stopped = true; timers.forEach(clearTimeout); onDone(); });
    demoPrompt(prompt).then((d) => {
      if (stopped) return;
      const steps = d.generation.slice(0, maxNewTokens);
      steps.forEach((s, i) => timers.push(setTimeout(() => onStep(s), DEMO_STEP_MS * (i + 1))));
      timers.push(setTimeout(finish, DEMO_STEP_MS * (steps.length + 1)));
    }).catch(finish);
    return finish;
  },
};

const liveApi: Api = {
  run: (prompt) => post<RunResponse>("/api/run", { prompt }),
  explain: (prompt) => post<ExplainResponse>("/api/twin/explain", { prompt }),
  whatif: (prompt, layer, pos, feature, value) =>
    post<WhatIfResponse>("/api/twin/whatif", { prompt, layer, pos, feature, value }),
  intervene: (prompt, interventions) => post<InterveneResponse>("/api/intervene", { prompt, interventions }),
  caseStudy: (name) => get<CaseStudy>(`/api/case-studies/${name}`),
  summary: () => get<Summary>("/api/fidelity/summary"),
  generate: (prompt, onStep, onDone, maxNewTokens = 20) => {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    const ws = new WebSocket(`${proto}://${location.host}/ws/generate`);
    let finished = false;
    const finish = once(() => { finished = true; onDone(); });
    ws.onopen = () => ws.send(JSON.stringify({ prompt, max_new_tokens: maxNewTokens }));
    ws.onmessage = (ev) => {
      let msg: Partial<GenerateStep> & { done?: boolean; error?: string };
      try { msg = JSON.parse(ev.data); } catch { return; }
      if (msg.done || msg.error) {
        finish();
        ws.close();
      } else if (!finished && msg.token !== undefined && msg.lens && msg.top) onStep(msg as GenerateStep);
    };
    ws.onerror = finish;
    ws.onclose = finish;  // a dropped socket must not leave the UI stuck on "Stop"
    return () => { finish(); ws.close(); };
  },
};

export const api: Api = STATIC ? staticApi : liveApi;
