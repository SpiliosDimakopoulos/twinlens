<!-- Draft fragments; merged into REPORT.md once results are in. -->

## 2. Setup

**Model.** GPT-2 small (124M parameters, 12 layers, 12 heads, d_model = 768), run by our own
~250-line PyTorch implementation (`twinlens/core/gpt2.py`). It exposes TransformerLens-style hook
points and reads its weights through memory-mapped safetensors, which keeps total memory under
1 GB on a laptop CPU. Its logits match Hugging Face `GPT2LMHeadModel` to 7e-5 (max abs).

**Twin features.** Per-layer MLP transcoders for GPT-2 small (Dunefsky, Chlenski & Nanda, 2024):
24,576 ReLU features per layer. Each reads `LayerNormPre(resid_mid)` and predicts `mlp_out`. We
confirmed this convention empirically (L0 ≈ 66, FVU 0.41 at layer 8, against FVU > 1 for the
alternatives). Splicing a transcoder in place of its MLP recovers 63–94% of the loss gap between
the clean model and the same model with that MLP mean-ablated.

**Validation of the measuring instrument.** Before any new measurement, the implementation
reproduces four published GPT-2 small results. Each is pinned as a test (`tests/test_reproduction.py`):
- **Induction heads** (Olsson et al., 2022): the five highest induction scores are exactly
  5.1, 5.5, 6.9, 7.2 and 7.10.
- **IOI** (Wang et al., 2022): the top-3 heads by direct logit attribution are the name movers
  9.9, 9.6 and 10.0. The two most negative are the negative name movers 10.7 and 11.10. Head
  patching ranks 9.9 first. Mean-ablating the name movers cuts the logit difference from 2.85
  to 2.0; three random heads leave it unchanged. The drop is partial, which is consistent with
  the backup name movers the paper describes.
- **Causal tracing** (Meng et al., 2022): restoring the residual stream shows an early site at
  the last subject token and a late site at the final token. Attention effects concentrate at
  the final token in layers 8–10. In GPT-2 small the MLP effect peaks in layers 0–1, earlier
  than the mid-layer site reported for GPT-2 XL.
- **DLA completeness**: per-component attributions sum to the logit difference (to 1e-3).

## 3. The twin ladder

Fix a prompt and its clean run. A *feature edit* sets transcoder feature *f* at layer *l* and
position *p* from its clean activation *a* to a new value *a'* (0 to ablate, 2a to amplify).
We compare the change in the log-probability of the clean top-1 token, Δlog p, under six models:

| level | downstream features | LayerNorm scales | attention patterns |
|---|---|---|---|
| 1 Direct edges | frozen at clean | frozen | frozen |
| 2 Linear (graph) | ReLU gate frozen (linear) | frozen | frozen |
| 3 Replacement | recomputed (ReLU) | frozen | frozen |
| 4 + live LayerNorm | recomputed | live | frozen |
| 5 + live attention | recomputed | live | live |
| Real GPT-2 | (real MLPs) | live | live |

In all twin levels each layer's MLP is replaced by its transcoder plus the clean-run error term
(`mlp_out − transcoder(mlp_in)`), held fixed. At zero perturbation every level therefore
reproduces the clean logits exactly; this is tested. In the real model the edit is applied as
the edit of the feature's decoder direction, `mlp_out[p] += (a' − a)·W_dec[f]`, after which
everything is recomputed.

Level 2 is affine in any perturbation (tested). It is the object an attribution graph describes:
attribution-graph edges are its partial derivatives. Level 1 keeps only the edited feature's
direct path. Its predicted effect on a logit equals the gradient × activation attribution;
this equality is tested to 1%. Levels 3–5 unfreeze one component at a time, so the gap between
consecutive levels attributes the twin's error to frozen LayerNorm, frozen attention, and,
finally, the transcoders themselves.

**Edits sampled.** For each prompt and layer: the 3 active features with the largest |direct
effect| on the predicted token ("top", the edges a user would inspect first), plus 3 uniformly
random active features ("random"). Each is ablated and amplified (×2). The BOS position is
excluded. With 56 prompts in 8 families (factual recall, IOI, induction/copying, syntax, code,
sequences/arithmetic, narrative, sentiment), this gives 8,064 edits, each run at all 6 levels.

**Metrics.** Spearman ρ between predicted and real Δlog p; median cosine between predicted and
real change in centred logits; sign agreement on edits with |real Δlog p| > 0.05; the rate of
catastrophic misses (|error| > 1 nat); and a tolerance criterion, *within tolerance* ⇔
|predicted − real| ≤ 0.1 + 0.25·|real|. Uncertainty: 95% cluster bootstrap over prompts
(1,000 resamples), since edits within a prompt are not independent.

**Trust model.** From signals available *before* the real model runs (layers after the edit,
distance to the last token, activation, relative edit size, |direct effect|, downstream
transcoder error, attention from the last token to the edited token, number of active
downstream features, model confidence), a standardised L2-regularised logistic regression
predicts whether the level-2 (graph) prediction is within tolerance. It is evaluated with
5-fold cross-validation grouped by prompt, so no prompt is seen in both training and test.
