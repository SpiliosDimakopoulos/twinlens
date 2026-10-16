# TwinLens

[![ci](https://github.com/SpiliosDimakopoulos/twinlens/actions/workflows/ci.yml/badge.svg)](https://github.com/SpiliosDimakopoulos/twinlens/actions/workflows/ci.yml) [![live demo](https://img.shields.io/badge/live%20demo-open-2a78d6)](https://spiliosdimakopoulos.github.io/twinlens/) [![report](https://img.shields.io/badge/report-PDF-1baf7a)](docs/TwinLens_report.pdf) ![license](https://img.shields.io/badge/license-MIT-blue)

**A digital twin of GPT-2's internals that tells you how far to trust its own explanations, and
a fix that makes those explanations faithful.**

**[▶ Open the live demo](https://spiliosdimakopoulos.github.io/twinlens/)**: it runs in the browser, nothing to install.
**[Read the report](https://spiliosdimakopoulos.github.io/twinlens/report/)** ([PDF](docs/TwinLens_report.pdf)).

Attribution graphs explain an LLM through an interpretable *replacement model*, a "twin" of the
real network. TwinLens runs that twin side by side with the real GPT-2. Click any feature in the
reasoning graph and:

1. the twin **predicts** what turning the feature off or up will do, at five levels of
   approximation, plus the proposed fix;
2. the real model **answers**;
3. a trust score, computed *before* the real model runs, tells you whether to believe the graph.

![IOI: the attribution graph predicts the edit lowers log p(Mary) by 8.1; GPT-2 actually raises it by 0.23; the fixed graph predicts +0.32](docs/screenshots/hero.png)

*In the IOI prompt above, the standard attribution graph predicts that this feature edit
**lowers** the answer by 8.1 nats. GPT-2 actually **raises** it by 0.23. The same linear graph
with the two derivatives restored predicts +0.32.*

<details><summary>▶ 20-second tour (GIF)</summary>

![Tour of the live demo](docs/demo.gif)

</details>

## Headline results
From 8,064 interventions on GPT-2 small; full write-up in [`paper/REPORT.md`](paper/REPORT.md) ([web](https://spiliosdimakopoulos.github.io/twinlens/report/), [PDF](docs/TwinLens_report.pdf)).

| | |
|---|---|
| **Attribution graphs are weak counterfactual predictors** | the graph twin's rank correlation with the real effect is ρ = **0.20**; on edits that measurably move the output it is within tolerance only **54%** of the time (IOI: ρ = 0.01) |
| **The cause is two dropped derivatives** | frozen LayerNorm causes **88%** of the error and nearly all catastrophic misses; frozen attention causes the ranking errors |
| **The fix keeps the graph linear** | restoring the LayerNorm and attention-pattern Jacobians gives a still-linear graph that matches the nonlinear twin: ρ **0.20 → 0.53**, sign agreement **67% → 86%**, catastrophic misses **2.2% → 0.0%**, IOI **0.01 → 0.59** |
| **The twin knows when it is wrong** | a trust model predicts, from the clean run alone, when a graph prediction will fail: AUROC **0.93** on held-out prompts *and* on held-out task families; on the edges users inspect it lifts graph accuracy from 54% to 71% |

<p>
  <img src="experiments/figures/F5_fix.png" width="100%" alt="Keeping two derivatives makes the linear graph as faithful as the nonlinear twin">
</p>

<p>
  <img src="experiments/figures/F1_ladder.png" width="49%" alt="Twin ladder metrics">
  <img src="experiments/figures/F4_trust.png" width="49%" alt="Trust model AUROC and calibration">
</p>

## What's inside

**The measuring instrument is validated first.** A from-scratch, hookable GPT-2 in PyTorch
(logits match Hugging Face to 7e-5) reproduces published circuits, and each one is pinned in the
test suite:
- **Induction heads** (Olsson et al., 2022): 5.1, 5.5, 6.9, 7.2, 7.10, the exact top 5.
- **IOI name movers and negative name movers** (Wang et al., 2022): 9.9, 9.6, 10.0 and 10.7,
  11.10.
- **The early and late sites of factual recall** (Meng et al., 2022).

**The twin ladder.** The same hook machinery runs five twins plus the real model:
1. direct edges;
2. linear attribution graph;
3. local replacement model;
4. \+ live LayerNorm;
5. \+ live attention.

Each step unfreezes one component, so every error is attributed to its source.

**A live dashboard** (React + TypeScript + FastAPI):
- **Twin:** the reasoning-path graph, the what-if ladder with the trust meter, and live token
  streaming that shows the layer at which each token was "decided".
- **Internals:** logit lens, attention patterns, and per-head attribution with click-to-ablate.
- **Fidelity study:** the research results, interactive.
- **Reproductions:** the published circuits, computed live.

| Internals | Fidelity study |
|---|---|
| ![Internals](docs/screenshots/internals.png) | ![Study](docs/screenshots/study.png) |

<p>
  <img src="docs/screenshots/twin-dark.png" width="72%" alt="Dark theme">
  <img src="docs/screenshots/mobile.png" width="22%" alt="Phone layout">
</p>

**UX details.**
- A prompt is traced as soon as the page opens, and the strongest feature is already simulated,
  so there is no empty first screen.
- Each what-if comes with a plain-language verdict ("the graph overstates the effect 21×") and
  the three signals that drove its trust score.
- Hovering a node isolates it in the graph, and every node can be reached with the keyboard.
- Light, dark and system themes; the layout works at phone width; reduced motion is respected.

## Quickstart

```bash
python -m venv .venv && .venv/Scripts/activate      # Windows; use bin/activate elsewhere
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e ".[dev]"
cd web && npm ci && cd ..

make serve        # builds the dashboard; open http://localhost:8000
make test         # 33 Python tests (unit, reproduction, twin invariants, API)
make web-test     # type-check + vitest
make e2e          # 7 Playwright tests: flows, keyboard, dark mode, phone width (needs `make serve`)
make analyze      # re-derive every number in the report from the committed sweep data
```

On first use the app downloads GPT-2 small (0.5 GB) and the 12 transcoders (1.8 GB) from the
Hugging Face Hub. Docker: `docker compose up`.

## Engineering notes

- **Runs in < 1 GB of RAM on a laptop CPU.** Importing TransformerLens brought the process to
  1.4 GB before any model was loaded, which left no room on the target machine. So the model is ~250 lines of PyTorch, and weights
  and transcoders are memory-mapped: served from the OS page cache, never copied
  ([`DECISIONS.md`](DECISIONS.md)).
- **Sparse transcoder compute.** Only the active columns are computed (~66 of 24,576 per
  token), so the five twin simulations of one what-if take about 1.6 s together on CPU.
- **Correctness by invariants.** Tests check that:
  - every twin level reproduces the clean run exactly;
  - the linear twin is affine;
  - gradient × activation attribution equals the direct-path simulation;
  - DLA sums to the logit.

| CPU latency (Ryzen 7 5700U, 15 tokens) | |
|---|---|
| forward pass | 194 ms |
| twin trace (12 transcoders) | 1.0 s |
| attribution (all features → logit) | 0.9 s |
| what-if, graph twin / full twin | 0.17 s / 0.45 s |
| peak private memory | 0.82 GB |

## Layout

```
twinlens/core       hookable GPT-2, mmap safetensors
twinlens/causal     interventions, activation patching, causal tracing, DLA, logit lens
twinlens/circuits   datasets + published-circuit measurements
twinlens/twin       transcoders, the twin ladder, attribution
twinlens/fidelity   reliability signals, fidelity engine, trust model
server/             FastAPI (REST + WebSocket streaming)
web/                React + TypeScript dashboard, vitest, Playwright
experiments/        sweep, analysis, figures, committed results
paper/REPORT.md     the technical report
```

## Limitations
One small model (GPT-2 small) and one transcoder family; a 56-prompt corpus. See §5 of the
report.
