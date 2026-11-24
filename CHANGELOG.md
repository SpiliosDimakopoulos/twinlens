# Changelog

## Unreleased

- Cross-layer transcoders (report §4.10): parity with circuit-tracer's graph (r > 0.9999999), 17,466 edits with the 426K Gemma-2-2B CLT, ρ 0.27 → 0.40 with the fix.
- Feature → feature edges (report §4.11, figure F9): 750k (edit, target) pairs on three models; the fix raises ρ from 0.45 to 0.68 (GPT-2), 0.49 to 0.57 (Gemma PLT) and 0.61 to 0.65 (Gemma CLT).
- External held-out prompt sets and a contrast target (DECISIONS.md D9; results pending).
- Re-analysis of the existing sweeps (report §4.12, DECISIONS.md D10-0): graph − Fix B disagreement as a warning light, normalisation-only fix against circuit-tracer's own intervention, top-10 recall, depth CLT vs PLT, and a Holm correction (18 of 28 paired tests survive; §4.7, §4.9, §4.10 reworded).
- `requirements.lock.txt`: exact versions behind the results, used by CI (#4).

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
