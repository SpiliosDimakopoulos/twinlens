---
title: TwinLens
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 6.32.0
app_file: app.py
pinned: false
license: mit
short_description: Test attribution-graph edges against real GPT-2
---

# TwinLens

Type any prompt. TwinLens builds GPT-2's attribution graph, switches features off, and compares
each edge's predicted effect with what GPT-2 actually does, with and without the fix that keeps
the LayerNorm and attention derivatives.

Needs CPU hardware (free CPU basic requires a Hugging Face PRO plan; ZeroGPU does not fit this
CPU-only app). After the Space wakes, the first request downloads GPT-2 and its transcoders
(about 2.3 GB) and takes a minute or two.

- Code and report: https://github.com/SpiliosDimakopoulos/twinlens
- Data: https://huggingface.co/datasets/SpiliosDmk/twinlens-fidelity

This Space is built from `deploy/hf-space/` in the GitHub repository: `app.py` fetches the source
at a pinned commit and the prebuilt dashboard from the v1.2.0 release, then serves the FastAPI app.
