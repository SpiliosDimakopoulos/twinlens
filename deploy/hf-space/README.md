---
title: TwinLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: Test attribution-graph edges against real GPT-2
---

# TwinLens

Type any prompt. TwinLens builds GPT-2's attribution graph, switches features off, and compares
each edge's predicted effect with what GPT-2 actually does, with and without the fix that keeps
the LayerNorm and attention derivatives.

Runs on a free CPU: the first request after the Space wakes takes about a minute.

- Code and report: https://github.com/SpiliosDimakopoulos/twinlens
- Data: https://huggingface.co/datasets/SpiliosDmk/twinlens-fidelity

This Space is built from `deploy/hf-space/` in the GitHub repository.
