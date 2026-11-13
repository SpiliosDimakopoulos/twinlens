# Changelog

## 1.2.0 (2026-11-13)

- Gemma-2-2B replication with Gemma Scope transcoders (report §4.9): 17,472 edits on all 26 layers, ρ 0.21 → 0.36, and the fix beats attribution patching on Gemma (Δρ +0.05 [0.03, 0.08]).
- README, dashboard text, PDF report and Hugging Face dataset card now cover all three models.
- The dataset publish script uploads the Gemma and attribution-patching files.

## 1.1.0 (2026-11-11)

- Hardened API input validation and WebSocket handling.
- Accessibility improvements and a plain-language dashboard UI.
- New `twinlens audit` CLI command.
- Reproducible `make analyze`, with robustness checks.
- Attribution-patching baseline (`experiments/atp_baseline.py`, report §4.8): the fixed graph beats a gradient through the real model, Δρ +0.15 [0.11, 0.19].
