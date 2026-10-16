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
  trust: { cv_auroc: number; base_rate: number; ok_rate_trusted_half: number; ok_rate_flagged_half: number; single_signal_auroc: Record<string, number> };
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

export const api = {
  run: (prompt: string) => post<RunResponse>("/api/run", { prompt }),
  explain: (prompt: string) => post<ExplainResponse>("/api/twin/explain", { prompt }),
  whatif: (prompt: string, layer: number, pos: number, feature: number, value: number) =>
    post<WhatIfResponse>("/api/twin/whatif", { prompt, layer, pos, feature, value }),
  intervene: (prompt: string, interventions: Intervention[]) =>
    post<InterveneResponse>("/api/intervene", { prompt, interventions }),
  caseStudy: (name: string) => get<CaseStudy>(`/api/case-studies/${name}`),
  summary: () => get<Summary>("/api/fidelity/summary"),
};

export type GenerateStep = { token: string; top: TopToken[]; lens: string[]; lens_prob: number[] };

export function generate(
  prompt: string,
  onStep: (s: GenerateStep) => void,
  onDone: () => void,
  maxNewTokens = 20,
): () => void {
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
